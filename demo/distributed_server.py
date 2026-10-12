"""One HTTP owner and sibling DiT workers supervised by torchrun."""
import os
import tempfile

# Set physical visibility before importing Torch, the engine or CUDA libraries.
rank = int(os.environ['LOCAL_RANK'])
visible = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
size = int(os.environ['WORLD_SIZE'])
if size not in (2, 4, 8) or len(visible) != size or not 0 <= rank < size:
    raise RuntimeError('Expected matching 2/4/8 GPU visibility and torchrun ranks')
vae = os.environ['WORLDCRAFTER_VAE_MODE']
if vae not in ('single', 'pipeline8') or (vae == 'pipeline8' and size != 8):
    raise RuntimeError('Invalid VAE device topology')
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(visible) if rank == 0 and vae == 'pipeline8' else visible[rank]

# PyTorch FX cache entries can otherwise be read while a sibling is writing
# them. Keep both compiler caches local and private to each rank.
for variable, default in (("TORCHINDUCTOR_CACHE_DIR", "torchinductor"),
                          ("TRITON_CACHE_DIR", "triton")):
    root = os.environ.get(variable, os.path.join(tempfile.gettempdir(), default))
    os.environ[variable] = os.path.join(root, f"rank_{rank}")


def main():
    if rank:
        from .multigpu.serving import serve_peer
        serve_peer()
    else:
        import argparse
        import uvicorn
        parser = argparse.ArgumentParser()
        parser.add_argument('--host', default='127.0.0.1')
        parser.add_argument('--port', type=int, default=8080)
        args = parser.parse_args()
        uvicorn.run('demo.server:app', host=args.host, port=args.port, ws='websockets')


if __name__ == '__main__': main()
