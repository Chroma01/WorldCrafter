from __future__ import annotations

import torch


def compute_fx_from_fov_xi(
    x_fov: torch.Tensor | float,
    xi: torch.Tensor | float,
    width: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    def to_tensor_1d(x):
        if torch.is_tensor(x):
            return x.to(device=device, dtype=dtype)
        return torch.tensor([x], dtype=dtype, device=device)

    x_fov = to_tensor_1d(x_fov)
    xi = to_tensor_1d(xi)

    batch = max(x_fov.shape[0], xi.shape[0])
    x_fov = x_fov.view(-1).expand(batch)
    xi = xi.view(-1).expand(batch)

    theta = torch.deg2rad(0.5 * x_fov)
    eps = torch.finfo(dtype).eps
    denom = torch.sin(theta).clamp_min(eps)
    fx = (width * 0.5) * (torch.cos(theta) + xi) / denom
    return fx


def compute_fov_from_fx_xi(
    fx: torch.Tensor | float,
    xi: torch.Tensor | float,
    width: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    def to_tensor_1d(x):
        if torch.is_tensor(x):
            return x.to(device=device, dtype=dtype)
        return torch.tensor([x], dtype=dtype, device=device)

    fx = to_tensor_1d(fx).view(-1)
    xi = to_tensor_1d(xi).view(-1)
    batch = max(fx.shape[0], xi.shape[0])
    fx = fx.expand(batch)
    xi = xi.expand(batch)

    a = 2.0 * fx / width
    phi = torch.atan(1.0 / a)
    denom = torch.sqrt(a * a + 1.0)
    ratio = (xi / denom).clamp(-1.0, 1.0)
    theta = torch.asin(ratio) + phi
    return torch.rad2deg(2.0 * theta)
