"""Load the released Base adapters, preserving the reference trainable boundary."""
from pathlib import Path
import torch
from peft import get_peft_model_state_dict
from safetensors.torch import save_file
from worldcrafter.diffusers.transformer import WorldCrafterTransformer3DModel
from worldcrafter.diffusers.pipeline import WorldCrafterPipeline
from worldcrafter.inference import validate_weights, load_model_adapter, configure_attention
from worldcrafter.kernels import replace_rmsnorm_with_fp32, replace_all_norms_with_flash_norms, replace_rope_with_flash_rope
from worldcrafter.ucpe.bridge import patch_worldcrafter_transformer_ucpe, load_ucpe_camera_adapter_weights, enable_ucpe_inference_sdpa_attention

EXPECTED_GROUPS = {'lora': (812, 623157248), 'camera': (400, 525388800)}


def load_base(model_path, device='cpu', *, gradient_checkpointing=True):
    paths = validate_weights(Path(model_path))
    enable_ucpe_inference_sdpa_attention()  # SDPA supports backward as well as inference.
    model = WorldCrafterTransformer3DModel.from_pretrained(paths['transformer'], torch_dtype=torch.bfloat16)
    model.requires_grad_(False)
    patch_worldcrafter_transformer_ucpe(model, method='relray_absmap', height=384, width=640,
                                      attn_compress=8, adaptation_method='parallel')
    info = load_ucpe_camera_adapter_weights(model, paths['adapter'])
    if info['loaded_tensor_keys'] != info['expected_tensor_keys']:
        raise RuntimeError(f'Incomplete Base camera adapter: {info}')
    model = replace_rmsnorm_with_fp32(model)
    model = replace_all_norms_with_flash_norms(model)
    replace_rope_with_flash_rope()
    configure_attention(model, 'native')
    pipe = WorldCrafterPipeline(tokenizer=None, text_encoder=None, vae=None, scheduler=None, transformer=model)
    load_model_adapter(pipe, paths['adapter'])
    for name, parameter in model.named_parameters():
        trainable = '.cam_self_attn.' in name or 'lora_' in name
        parameter.requires_grad_(trainable)
        if trainable:
            parameter.data = parameter.data.float()
    if gradient_checkpointing:
        model.enable_gradient_checkpointing()
    model.to(device)
    return model, paths, audit_groups(model)


def audit_groups(model):
    groups = {'lora': [], 'camera': []}
    for name, p in model.named_parameters():
        if p.requires_grad:
            group = 'camera' if '.cam_self_attn.' in name else 'lora'
            if group == 'lora' and 'lora_' not in name:
                raise RuntimeError(f'Unexpected trainable parameter: {name}')
            groups[group].append((name, p))
    for key, entries in groups.items():
        actual = (len(entries), sum(p.numel() for _, p in entries))
        if actual != EXPECTED_GROUPS[key]:
            raise RuntimeError(f'{key}: expected {EXPECTED_GROUPS[key]}, got {actual}')
    for config in model.peft_config.values():
        if config.r != 128 or config.lora_alpha != 128 or config.lora_dropout != 0:
            raise RuntimeError('Base requires the existing rank=alpha=128, dropout=0 LoRA')
    return groups


def save_adapters(model, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lora = get_peft_model_state_dict(model, adapter_name='worldcrafter')
    save_file({'transformer.' + k: v.detach().cpu().contiguous() for k, v in lora.items()},
              str(output / 'pytorch_lora_weights.safetensors'))
    camera = {name: p.detach().cpu() for name, p in model.state_dict().items() if '.cam_self_attn.' in name}
    torch.save(camera, output / 'camera_adapter.pth')
