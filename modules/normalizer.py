"""
modules/normalizer.py
=====================
GPU-accelerated Normalization Engine for 3D Medical Volumes.
Converts raw CT/MRI volumes into 8-bit (0-255) arrays.

Pipeline:
1. Origin Shift: Entire 3D volume is shifted so the minimum value is exactly 0.
2. Normalization: Applies the requested function (linear, log, sigmoid) to the shifted array.
"""

from __future__ import annotations

import logging
import cupy as cp

log = logging.getLogger("normalizer")


def normalize_linear(shifted_vol: cp.ndarray, vmax: float | None = None) -> cp.ndarray:
    """
    Linear scaling to [0, 255].
    Assumes the volume has already been origin-shifted (min == 0).
    """
    if vmax is None: 
        vmax = float(cp.max(shifted_vol))
    
    if vmax < 1e-8:
        vmax = 1e-8
        
    normed = (shifted_vol / vmax) * 255.0
    normed = cp.clip(normed, 0.0, 255.0)
    
    return normed.astype(cp.uint8)


def normalize_log(shifted_vol: cp.ndarray, base: float = 10.0) -> cp.ndarray:
    """
    Logarithmic scaling to [0, 255] with a configurable base.
    Assumes the volume has already been origin-shifted (min == 0).
    """
    # Change of base formula: log_b(x) = ln(x) / ln(b)
    # We use log1p(x) which is ln(1 + x) to safely handle exact 0s without math errors
    
    if base == 'e':
        log_vol = cp.log1p(shifted_vol)
    else:
        log_vol = cp.log1p(shifted_vol) / cp.log(base)
    
    # Scale the resulting log values to 255
    log_max = float(cp.max(log_vol))
    if log_max < 1e-8:
        log_max = 1e-8
        
    normed = (log_vol / log_max) * 255.0
    return normed.astype(cp.uint8)


def normalize_sigmoid(shifted_vol: cp.ndarray, center: float, width: float) -> cp.ndarray:
    """
    Sigmoid scaling to [0, 255]. 
    Assumes the volume has already been origin-shifted (min == 0), 
    and that 'center' has been shifted accordingly.
    """
    if width < 1e-8: 
        width = 1e-8
        
    # Scale x for the sigmoid function: (v - c) / w
    # We divide width by 5 to make the S-curve steepness visually effective
    x = (shifted_vol - center) / (width / 5.0)
    
    # Clip to prevent overflow warnings in cp.exp()
    x = cp.clip(x, -20.0, 20.0)
    
    # Sigmoid formula: 1 / (1 + e^-x)
    sig = 1.0 / (1.0 + cp.exp(-x))
    
    normed = sig * 255.0
    return normed.astype(cp.uint8)


def process_volume(volume: cp.ndarray, method: str = 'linear', **kwargs) -> cp.ndarray:
    """
    Master router for volume normalization.
    Applies a global origin shift, then routes to the requested mathematical method.
    Returns a new CuPy array of dtype uint8.
    """
    
    # ── STEP 1: Global Origin Shift ──────────────────────────────────────────
    # Find the absolute minimum in the whole 3D array and shift everything up.
    v_min = float(cp.min(volume))
    shifted_volume = volume.astype(cp.float32) - v_min
    
    log.debug("Origin shift applied. Original min: %.2f. New min: 0.0", v_min)

    # ── STEP 2: Mathematical Normalization ───────────────────────────────────
    if method == 'linear':
        # If the user provided a custom max, we need to shift that max too
        vmax = kwargs.get('vmax')
        if vmax is not None:
            kwargs['vmax'] = vmax - v_min
            
        return normalize_linear(shifted_volume, **kwargs)

    elif method == 'log':
        # Safely extract 'base' if provided, otherwise defaults to 10.0 inside the func
        base = kwargs.get('base', 10.0)
        return normalize_log(shifted_volume, base=base)

    elif method == 'sigmoid':
        # DICOM windowing uses center and width.
        # If the user passes raw HU values (e.g. center=-500), we must shift the center
        # by the exact same amount we shifted the volume so the math aligns.
        center = kwargs.get('center')
        width  = kwargs.get('width')
        
        if center is None: 
            center = float(cp.mean(volume))
        if width is None:  
            width = float(cp.std(volume)) * 2.0
            
        shifted_center = center - v_min
        
        return normalize_sigmoid(shifted_volume, center=shifted_center, width=width)

    else:
        raise ValueError(f"Unknown normalization method: '{method}'. Choose from: linear, log, sigmoid.")
