"""
modules/renderer.py
===================
GPU-accelerated rendering engine for 2-D DICOM slices.
Handles resizing and RingBuffer management.

Note: Normalization (HU to 0-255 scaling) is handled upstream 
by normalizer.py. This engine expects uint8 arrays.
"""

from __future__ import annotations

from typing import Optional
import cupy as cp
import numpy as np
from PIL import Image, ImageTk
from cupyx.scipy.ndimage import zoom as cp_zoom

# ── constants ────────────────────────────────────────────────────────────────
RING_SIZE    = 12          # total ring slots
HALF_WINDOW  = 5           # slots on each side of current
FLUSH_OFFSET = 6           # diametrically opposite slot index


# ═══════════════════════════════════════════════════════════════════════════ #
#  GPU rendering pipeline                                                     #
# ═══════════════════════════════════════════════════════════════════════════ #

def gpu_resize(img_cp: cp.ndarray,
               target_h: int,
               target_w: int) -> cp.ndarray:
    """
    Bilinear resize on GPU using cupyx.scipy.ndimage.zoom.
    Returns uint8 ndarray of shape (target_h, target_w).
    """
    if img_cp.shape == (target_h, target_w):
        return img_cp
    
    zoom_y = target_h / img_cp.shape[0]
    zoom_x = target_w / img_cp.shape[1]
    
    resized = cp_zoom(img_cp.astype(cp.float32), (zoom_y, zoom_x), order=1)
    return cp.clip(resized, 0, 255).astype(cp.uint8)


def render_slice(volume: cp.ndarray,
                 z_index: int,
                 canvas_h: int,
                 canvas_w: int) -> ImageTk.PhotoImage:
    """
    Full GPU → CPU → PIL → PhotoImage pipeline for one slice.
    """
    raw      = volume[z_index]                               # Already uint8 from normalizer
    resized  = gpu_resize(raw, canvas_h, canvas_w)           # uint8 GPU resized
    arr_cpu  = cp.asnumpy(resized)                           # to CPU
    pil_img  = Image.fromarray(arr_cpu, mode="L")            # greyscale PIL
    return ImageTk.PhotoImage(pil_img)                       # Tk-compatible


# ═══════════════════════════════════════════════════════════════════════════ #
#  Ring buffer                                                                #
# ═══════════════════════════════════════════════════════════════════════════ #

class RingBuffer:
    """
    12-slot circular buffer of PhotoImage objects.
    """
    def __init__(self) -> None:
        self.slots:   list[Optional[ImageTk.PhotoImage]] = [None] * RING_SIZE
        self.z_index: list[Optional[int]]                = [None] * RING_SIZE
        self.head: int = 0

    def build(self,
              volume:    cp.ndarray,
              current_z: int,
              canvas_h:  int,
              canvas_w:  int) -> None:
        """(Re)build the entire ring starting at current_z."""
        n = volume.shape[0]
        self.head = 0

        for offset in range(RING_SIZE):
            ring_pos = offset % RING_SIZE

            if offset == FLUSH_OFFSET:
                # Flush slot starts empty
                self.slots[ring_pos]   = None
                self.z_index[ring_pos] = None
                continue

            if offset <= HALF_WINDOW:
                z = current_z + offset
            else:
                z = current_z - (RING_SIZE - offset)

            if 0 <= z < n:
                self.slots[ring_pos]   = render_slice(volume, z, canvas_h, canvas_w)
                self.z_index[ring_pos] = z
            else:
                self.slots[ring_pos]   = None
                self.z_index[ring_pos] = None


    def current_photo(self) -> Optional[ImageTk.PhotoImage]:
        return self.slots[self.head]


    def advance(self,
                direction: int,
                volume:    cp.ndarray,
                canvas_h:  int,
                canvas_w:  int) -> bool:
        """
        Move head by *direction* (+1 = higher Z, -1 = lower Z).
        Returns True if successful, False if hitting a boundary.
        """
        new_head = (self.head + direction) % RING_SIZE
        if self.slots[new_head] is None and self.z_index[new_head] is None:
            return False # Boundary

        # 1. The OLD flush slot is the one currently empty; it gets the new slice
        old_flush_pos = (self.head + FLUSH_OFFSET) % RING_SIZE
        
        # 2. Update head pointer
        self.head = new_head
        
        # 3. The NEW flush slot is the one that needs to be cleared
        new_flush_pos = (self.head + FLUSH_OFFSET) % RING_SIZE
        
        head_z = self.z_index[self.head]
        if head_z is None:
            return False

        # Calculate the Z-index for the newly exposed edge of the window
        new_z = head_z + direction * HALF_WINDOW
        n     = volume.shape[0]

        # Populate the old flush position with the new slice
        if 0 <= new_z < n:
            self.slots[old_flush_pos]   = render_slice(volume, new_z, canvas_h, canvas_w)
            self.z_index[old_flush_pos] = new_z
        else:
            self.slots[old_flush_pos]   = None
            self.z_index[old_flush_pos] = None

        # Clear out the new flush position to maintain the gap
        self.slots[new_flush_pos]   = None
        self.z_index[new_flush_pos] = None

        return True
