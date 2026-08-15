from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


def sinc_resample_freq(H: torch.Tensor, target_nf: int = 128) -> torch.Tensor:
    """Resample a uniformly spaced complex frequency response at fixed bandwidth.

    The conversion goes through the delay domain so the complex phase slope is
    preserved. Frequency samples are stored in negative-to-positive order.
    """
    if H.shape[-1] == target_nf:
        return H
    if target_nf <= 0:
        raise ValueError("target_nf must be positive.")

    source_nf = H.shape[-1]
    delay_response = torch.fft.ifft(
        torch.fft.ifftshift(H.to(torch.complex64), dim=-1),
        dim=-1,
    )
    if target_nf > source_nf:
        delay_response = F.pad(delay_response, (0, target_nf - source_nf))
    else:
        delay_response = delay_response[..., :target_nf]
    return torch.fft.fftshift(
        torch.fft.fft(delay_response, n=target_nf, dim=-1),
        dim=-1,
    )


def select_single_rx(raw_csi: torch.Tensor, rx_index: int = 0) -> torch.Tensor:
    """
    Accept DeepMIMO-style CSI and return a single-Rx tensor of shape [1, N_tx, N_f].

    Supported inputs:
    - [N_rx, N_tx, N_f]
    - [N_tx, N_f]
    """
    if raw_csi.ndim == 2:
        return raw_csi.unsqueeze(0)
    if raw_csi.ndim != 3:
        raise ValueError(
            "raw_csi must have shape [N_rx, N_tx, N_f] or [N_tx, N_f], "
            f"got {tuple(raw_csi.shape)}"
        )
    if not 0 <= rx_index < raw_csi.shape[0]:
        raise IndexError(f"rx_index={rx_index} out of range for shape {tuple(raw_csi.shape)}")
    return raw_csi[rx_index : rx_index + 1]


def beamspace_tokenize(
    H_beam: torch.Tensor,
    beam_grid_shape: tuple[int, ...],
    n_rx: int,
    array_type: str,
    patch_1d: int = 4,
    patch_2d: tuple[int, int] = (2, 2),
) -> tuple[torch.Tensor, torch.Tensor]:
    N_rx, N_tx, N_f = H_beam.shape
    if N_rx != n_rx:
        raise ValueError(f"Expected n_rx={n_rx}, got {N_rx}")

    if len(beam_grid_shape) == 1:
        C = min(patch_1d, N_tx)
        K = math.ceil(N_tx / C)
        tokens = []
        positions = []
        for k in range(K):
            start = k * C
            end = min(start + C, N_tx)
            chunk = H_beam[:, start:end, :]
            if chunk.shape[1] < C:
                pad = torch.zeros(
                    N_rx,
                    C - chunk.shape[1],
                    N_f,
                    dtype=chunk.dtype,
                    device=chunk.device,
                )
                chunk = torch.cat([chunk, pad], dim=1)
            token = torch.cat([chunk.real, chunk.imag], dim=0).reshape(-1, N_f)
            center = (start + end - 1) / 2.0 / max(N_tx - 1, 1)
            tokens.append(token)
            positions.append(torch.tensor([center, 0.0], dtype=torch.float32))
        return torch.stack(tokens), torch.stack(positions)

    if len(beam_grid_shape) != 2 or array_type != "UPA":
        raise ValueError(f"Unsupported shape/type combo: {beam_grid_shape} / {array_type}")

    n_row_beam, n_col_beam = beam_grid_shape
    pr, pc = patch_2d
    Kr = math.ceil(n_row_beam / pr)
    Kc = math.ceil(n_col_beam / pc)
    H_2d = H_beam.reshape(N_rx, n_row_beam, n_col_beam, N_f)
    tokens = []
    positions = []
    for ir in range(Kr):
        for ic in range(Kc):
            r_start, r_end = ir * pr, min((ir + 1) * pr, n_row_beam)
            c_start, c_end = ic * pc, min((ic + 1) * pc, n_col_beam)
            chunk = H_2d[:, r_start:r_end, c_start:c_end, :]
            if chunk.shape[1] < pr or chunk.shape[2] < pc:
                padded = torch.zeros(N_rx, pr, pc, N_f, dtype=chunk.dtype, device=chunk.device)
                padded[:, : chunk.shape[1], : chunk.shape[2], :] = chunk
                chunk = padded
            token = torch.cat([chunk.real, chunk.imag], dim=0).reshape(-1, N_f)
            r_center = (r_start + r_end - 1) / 2.0 / max(n_row_beam - 1, 1)
            c_center = (c_start + c_end - 1) / 2.0 / max(n_col_beam - 1, 1)
            tokens.append(token)
            positions.append(torch.tensor([r_center, c_center], dtype=torch.float32))
    return torch.stack(tokens), torch.stack(positions)


def preprocess_sample(
    raw_csi: torch.Tensor,
    array_type: str,
    n_row: int,
    n_col: int,
    n_rx: int = 1,
    target_nf: int = 128,
    patch_1d: int = 4,
    patch_2d: tuple[int, int] = (2, 2),
    rx_index: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if raw_csi.ndim not in (2, 3):
        raise ValueError(
            "raw_csi must be [N_rx, N_tx, N_f] or [N_tx, N_f], "
            f"got shape {tuple(raw_csi.shape)}"
        )

    H = raw_csi
    if rx_index is not None:
        H = select_single_rx(H, rx_index=rx_index)
        n_rx = 1
    elif H.ndim == 2:
        H = H.unsqueeze(0)

    if H.shape[0] != n_rx:
        raise ValueError(f"Expected n_rx={n_rx}, got input shape {tuple(H.shape)}")
    phase_ref = H[0, 0, 0]
    H = H * torch.exp(-1j * torch.angle(phase_ref))
    H = sinc_resample_freq(H, target_nf=target_nf)

    if array_type == "ULA":
        H_beam = torch.fft.fft(H, dim=1)
        beam_grid_shape = (H.shape[1],)
    elif array_type == "UPA":
        N_tx = n_row * n_col
        H_2d = H.reshape(n_rx, n_row, n_col, target_nf)
        H_beam_2d = torch.fft.fft2(H_2d, dim=(1, 2))
        beam_grid_shape = (n_row, n_col)
        H_beam = H_beam_2d.reshape(n_rx, N_tx, target_nf)
    else:
        raise ValueError(f"Unsupported array_type: {array_type}")

    tokens, beam_positions = beamspace_tokenize(
        H_beam=H_beam,
        beam_grid_shape=beam_grid_shape,
        n_rx=n_rx,
        array_type=array_type,
        patch_1d=patch_1d,
        patch_2d=patch_2d,
    )
    metadata = {
        "beam_grid_shape": beam_grid_shape,
        "n_tokens": int(tokens.shape[0]),
        "d_token": int(tokens.shape[1]),
        "target_nf": int(tokens.shape[2]),
        "selected_n_rx": int(H.shape[0]),
        "selected_rx_index": int(rx_index) if rx_index is not None else None,
    }
    return tokens.float(), beam_positions.float(), metadata
