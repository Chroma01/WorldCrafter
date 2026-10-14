"""Context-parallel backend behind the existing WorldCrafter Demo interface."""
from datetime import timedelta
import json
import os

import torch
import torch.distributed as dist

from demo.config import DATA
from demo.engine import WorldCrafterEngine
from .context_parallel import install
from .control_channel import ControlChannel
from .coordinator import Coordinator


def initialize():
    torch.cuda.set_device(0)
    dist.init_process_group('nccl', timeout=timedelta(seconds=1800), device_id=torch.device('cuda:0'))
    if dist.get_world_size() not in (2, 4, 8):
        raise ValueError('Context parallel Demo requires 2, 4 or 8 ranks')


def configure(engine):
    compile_blocks = os.environ.get('WORLDCRAFTER_DEMO_COMPILE', '1') == '1'
    info = install(engine.pipe, os.environ.get('WORLDCRAFTER_KV_FUSION', 'rope'), compile_blocks)
    engine.info.update(gpu_count=dist.get_world_size(), gpu_mode=info['strategy'],
                       parallel_implementation=info, compile_enabled=compile_blocks,
                       control_transport='private_unix_socket')


def channel():
    return ControlChannel(os.environ['WORLDCRAFTER_CONTROL_SOCKET'], dist.get_rank(), dist.get_world_size())


def aggregate_memory(peers):
    devices = []
    for peer in peers:
        device = dict(peer['memory']['devices'][0]);device['device'] = peer['rank']
        devices.append(device)
    for extra in peers[0]['memory']['devices'][1:]:
        for key in ('allocated', 'reserved', 'peak_allocated', 'peak_reserved'):
            devices[extra['device']][key] += extra[key]
    return dict(devices=devices, **{key:sum(d[key] for d in devices)
                for key in ('allocated', 'reserved', 'peak_allocated', 'peak_reserved')})


class ContextParallelEngine(WorldCrafterEngine):
    def __init__(self):
        self.coordinator = None
        initialize()
        if dist.get_rank() != 0: raise RuntimeError('Only rank zero owns the server')
        super().__init__(enable_compile=False)
        configure(self)
        self.coordinator = Coordinator(self, channel())
        self.parallel_vae = None
        vae_mode = os.environ.get('WORLDCRAFTER_VAE_MODE', 'single')
        if vae_mode.startswith('pipeline'):
            from .parallel_vae import ParallelDecoder
            self.parallel_vae = ParallelDecoder(
                self.pipe.vae, dist.get_world_size(),
                optimized=os.environ.get('WORLDCRAFTER_DEMO_OPTIMIZED', '0') == '1',
                compile_stages=os.environ.get('WORLDCRAFTER_DEMO_COMPILE', '1') == '1')
            self.parallel_vae.warmup(self.pipe.vae.decode)
            self.pipe.vae.decode = self.parallel_vae.decode
            self.devices = list(range(dist.get_world_size()))
            self.info['vae_decode'] = self.parallel_vae.info
        else:
            self.info['vae_decode'] = dict(mode='single', gpu_count=1, warmed=True, compute_dtype='float32')
        (DATA/'reports/model_setup.json').write_text(json.dumps(self.info, indent=2)+'\n')

    def next_chunk(self, action, poses=None, capture=False):
        result = super().next_chunk(action, poses, capture)
        peers = self.coordinator.stats()
        result['memory'] = aggregate_memory(peers)
        result['parallel_strategy'] = self.info['gpu_mode']
        result['parallel_ranks'] = peers
        if self.parallel_vae is not None:
            result['vae_decode'] = dict(self.parallel_vae.info, seconds=self.parallel_vae.last_decode_s)
        return result

    def close_session(self):
        super().close_session()
        if self.coordinator is not None: self.coordinator.reset()

    def shutdown(self):
        try:
            super().close_session()
            if self.coordinator is not None:
                try:
                    self.coordinator.stop()
                except (BrokenPipeError, ConnectionResetError, EOFError):
                    # torchrun may have already stopped peers on a terminal signal.
                    pass
                finally:
                    self.coordinator.channel.close()
                    self.coordinator = None
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


def serve_peer():
    initialize()
    engine = WorldCrafterEngine(enable_compile=False)
    configure(engine)
    coordinator = Coordinator(engine, channel())
    try:
        coordinator.serve()
    except KeyboardInterrupt:
        pass
    finally:
        coordinator.channel.close()
        if dist.is_initialized(): dist.destroy_process_group()
