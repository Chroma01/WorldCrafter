"""Rank zero owns sampling, RNG, retrieval and sessions; peers execute DiT only."""
import torch
import torch.distributed as dist


class Coordinator:
    def __init__(self, engine, channel):
        self.engine, self.channel, self.rank = engine, channel, dist.get_rank()
        self.calls, self.handles = 0, []
        if self.rank == 0:
            for branch, model in enumerate((engine.pipe.stage_transformers[0], engine.pipe.stage_transformers[2])):
                def forward(module, args, kwargs, branch=branch):
                    self.send_forward(branch, args, kwargs)
                self.handles.append(model.register_forward_pre_hook(forward, with_kwargs=True, prepend=True))

    def pack(self, value, tensors):
        if torch.is_tensor(value):
            if not value.is_cuda:
                return ('cpu', str(value.dtype).split('.')[-1], value.tolist())
            index = len(tensors);tensors.append(value.contiguous())
            return ('cuda', index, tuple(value.shape), str(value.dtype).split('.')[-1])
        if isinstance(value, dict): return ('dict', [(k, self.pack(v, tensors)) for k, v in value.items()])
        if isinstance(value, (list, tuple)): return (type(value).__name__, [self.pack(v, tensors) for v in value])
        return ('scalar', value)

    def unpack(self, spec, tensors):
        kind = spec[0]
        if kind == 'cuda':
            _, index, shape, dtype = spec
            if index != len(tensors): raise ValueError('Invalid tensor order')
            value = torch.empty(shape, dtype=getattr(torch, dtype), device='cuda:0')
            tensors.append(value);return value
        if kind == 'cpu': return torch.tensor(spec[2], dtype=getattr(torch, spec[1]))
        if kind == 'dict': return {k:self.unpack(v, tensors) for k, v in spec[1]}
        if kind in ('list', 'tuple'):
            values = [self.unpack(v, tensors) for v in spec[1]]
            return tuple(values) if kind == 'tuple' else values
        if kind != 'scalar': raise ValueError('Invalid control payload')
        return spec[1]

    def send_forward(self, branch, args, kwargs):
        tensors = [];spec = self.pack((args, kwargs), tensors)
        self.channel.message(('forward', branch, self.engine.pipe.stage_forward_context, spec))
        for tensor in tensors: dist.broadcast(tensor, src=0)
        self.calls += 1

    def reset(self):
        self.calls = 0
        self.reset_geometry()
        self.channel.message(('reset',))
        self.channel.reply({'reset':True})

    def stats(self):
        self.channel.message(('stats',))
        result = self.channel.reply(self.local_stats())
        if any(p['calls'] != self.calls for p in result):
            raise RuntimeError('DiT rank forward counts diverged')
        return result

    def local_stats(self):
        return dict(rank=self.rank, calls=self.calls, memory=self.engine.memory(),
                    parallel=self.engine.info['parallel_implementation'],
                    geometry_cache=[cache.stats() for cache in self.engine.pipe.context_geometry_caches])

    def reset_geometry(self):
        for cache in self.engine.pipe.context_geometry_caches:
            cache.reset()

    def stop(self):
        try:
            self.channel.message(('stop',))
            self.channel.reply({'stopped':True})
        finally:
            for handle in self.handles:
                handle.remove()
            self.handles.clear()

    def serve(self):
        with torch.inference_mode():
            while True:
                message = self.channel.message()
                if message[0] == 'stop':
                    self.channel.reply({'stopped':True});break
                if message[0] == 'reset':
                    self.calls = 0
                    self.reset_geometry()
                    self.engine.pipe.stage_model_trace.clear()
                    torch.cuda.reset_peak_memory_stats()
                    self.channel.reply({'reset':True});continue
                if message[0] == 'stats':
                    self.channel.reply(self.local_stats());continue
                if message[0] != 'forward': raise ValueError('Unknown peer command')
                _, branch, context, spec = message
                tensors = [];args, kwargs = self.unpack(spec, tensors)
                for tensor in tensors: dist.broadcast(tensor, src=0)
                pipe = self.engine.pipe
                pipe.stage_forward_context = tuple(context)
                pipe.resident_branches.switch('equal' if branch == 0 else 'old')
                model = pipe.stage_transformers[0 if branch == 0 else 2]
                with model.cache_context('cond'): model(*args, **kwargs)
                self.calls += 1
