"""
main.py
=======
Prototype runner for the ILD DICOM 3-D Medical Imaging pipeline.

What it does
------------
1. Loads all DICOM series from the predefined folder via dicom_loader.
2. Prints a summary table of every loaded series.
3. Normalizes the volumes from raw HU to 8-bit arrays (0-255).
4. Reports volumes currently resident on GPU memory.
5. Launches the Tkinter viewer to render slices via GPU.
6. Flushes GPU memory completely upon exit.
"""

from __future__ import annotations

import argparse
import logging
import sys
import gc  # <-- Added for garbage collection

import cupy as cp

# ─── Module Imports ──────────────────────────────────────────────────────────
from modules.dicom_loader import load_dicom_folder, CorruptPixelDataError
from modules.normalizer import process_volume
from modules.interface import show_viewer


# ═══════════════════════════════════════════════════════════════════════════ #
#  Configuration                                                              #
# ═══════════════════════════════════════════════════════════════════════════ #

DICOM_FOLDER_PATH = "./Data/Sample1/Lung"  # <-- CHANGE THIS TO YOUR FOLDER PATH
LOG_LEVEL         = "INFO"                   


# ═══════════════════════════════════════════════════════════════════════════ #
#  Logging setup                                                              #
# ═══════════════════════════════════════════════════════════════════════════ #

def _setup_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    fmt   = "%(asctime)s  %(levelname)-8s  %(name)s: %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, stream=sys.stdout)


# ═══════════════════════════════════════════════════════════════════════════ #
#  Summary printer                                                            #
# ═══════════════════════════════════════════════════════════════════════════ #

def _print_summary(results: dict) -> None:
    """Print a human-readable table of every loaded series."""

    SEP  = "─" * 80
    SEP2 = "═" * 80

    print(f"\n{SEP2}")
    print(f"  LOADED SERIES SUMMARY  ({len(results)} series)")
    print(SEP2)

    for idx, (uid, entry) in enumerate(results.items(), start=1):
        vol  = entry["volume"]
        meta = entry["metadata"]

        dz = meta["SliceSpacingComputed"]
        ps = meta["PixelSpacing"]
        if dz is not None and ps is not None:
            spacing_str = f"Z={dz:.3f} mm  Y={ps[0]:.3f} mm  X={ps[1]:.3f} mm"
        elif ps is not None:
            thickness = meta["SliceThickness"]
            thick_str = f"{thickness:.3f}" if thickness is not None else "N/A"
            spacing_str = f"Z(tag)={thick_str} mm  Y={ps[0]:.3f} mm  X={ps[1]:.3f} mm"
        else:
            spacing_str = "spacing unknown"

        mem_mb = vol.nbytes / (1024 ** 2)

        print(f"\n  [{idx}] SeriesInstanceUID : {uid}")
        print(f"       Modality          : {meta['Modality'] or 'unknown'}")
        print(f"       Patient           : {meta['PatientName'] or 'N/A'}  "
              f"(ID={meta['PatientID'] or 'N/A'}  DOB={meta['PatientBirthDate'] or 'N/A'})")
        print(f"       Volume shape      : {vol.shape}  dtype={vol.dtype}")
        print(f"       Spacing           : {spacing_str}")
        print(f"       GPU memory        : {mem_mb:.2f} MB")
        print(f"  {SEP}")

    print()


# ═══════════════════════════════════════════════════════════════════════════ #
#  GPU info                                                                   #
# ═══════════════════════════════════════════════════════════════════════════ #

def _print_gpu_info() -> None:
    """Print basic CuPy / CUDA device information."""
    try:
        n_devices = cp.cuda.runtime.getDeviceCount()
        print(f"  GPU devices visible to CuPy: {n_devices}")
        for i in range(n_devices):
            with cp.cuda.Device(i):
                free, total = cp.cuda.runtime.memGetInfo()
                print(f"    Device {i}: {free / (1024**2):.0f} MB free / {total / (1024**2):.0f} MB total")
    except cp.cuda.runtime.CUDARuntimeError as exc:
        print(f"  (CuPy GPU query failed: {exc})")


# ═══════════════════════════════════════════════════════════════════════════ #
#  Entry point                                                                #
# ═══════════════════════════════════════════════════════════════════════════ #

def main() -> int:
    _setup_logging(LOG_LEVEL)
    log = logging.getLogger("main")

    # ── 1. GPU pre-flight check ───────────────────────────────────────
    print("\n" + "═" * 80)
    print("  ILD DETECTION — CT PROTOTYPE")
    print("═" * 80)
    _print_gpu_info()

    # ── 2. Load DICOM folder into GPU ─────────────────────────────────
    log.info("Loading DICOM folder: %s", DICOM_FOLDER_PATH)

    try:
        results = load_dicom_folder(DICOM_FOLDER_PATH)
    except Exception as exc:
        log.exception("Unexpected error during DICOM load: %s", exc)
        return 1

    if not results:
        log.warning("No valid DICOM series found in '%s'", DICOM_FOLDER_PATH)
        return 0

    # ── 3. Print loaded summary ───────────────────────────────────────
    _print_summary(results)
    
    # ── 4. Normalize Volumes to 8-bit ─────────────────────────────────
    log.info("Normalizing all volumes to 8-bit using 'sigmoid' (lung preset)...")
    for uid, entry in results.items():
        raw_vol = entry["volume"]
        
        # Apply normalization. center=-500, width=1500 is a classic CT lung window.
        normed_vol = process_volume(raw_vol, method='sigmoid', center=-500.0, width=1500.0)
        
        # Overwrite the raw volume with the normalized 8-bit volume
        entry["volume"] = normed_vol
        
        # Calculate new memory footprint
        new_mem = normed_vol.nbytes / (1024 ** 2)
        log.info("Series %s normalized. New GPU footprint: %.2f MB", uid[-12:], new_mem)

    # ── 5. Verify GPU memory consumption ──────────────────────────────
    print("\n  GPU memory after normalization:")
    _print_gpu_info()
    print()

    # ── 6. Launch viewer ──────────────────────────────────────────────
    log.info("Launching GUI Viewer...")
    show_viewer(results)

    # ── 7. GPU Memory Cleanup ─────────────────────────────────────────
    log.info("Viewer closed. Flushing GPU memory...")
    
    # Delete the Python references to the GPU arrays
    del results
    del normed_vol
    
    # Force Python to immediately garbage-collect the deleted objects
    gc.collect()
    
    # Instruct CuPy to release all cached memory blocks back to the OS
    cp.get_default_memory_pool().free_all_blocks()
    
    print("\n  GPU memory after flush (should return to baseline):")
    _print_gpu_info()
    
    log.info("Pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
