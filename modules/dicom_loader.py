"""
modules/dicom_loader.py
=======================
Load all DICOM files from a flat folder, group them by SeriesInstanceUID,
sort each group by Z position (ImagePositionPatient[2]), and stack the
pixel data into a CuPy 3-D array per series.

Axis order of the output volume:  (Z, Y, X)  →  (slices, rows, cols)

Return value
------------
dict[SeriesInstanceUID: str  →  SeriesResult: dict]

Each SeriesResult::

    {
        "volume":   cupy.ndarray   # shape (n_slices+1, Rows, Cols)
                                   # dtype float32 for CT (HU-rescaled)
                                   #       uint16  for MR / other
                                   # the extra (+1) Z slice is zero-padded
                                   # as a safety margin
        "metadata": {
            "PatientName":          str | None,
            "PatientID":            str | None,
            "PatientBirthDate":     str | None,
            "PatientSex":           str | None,
            "StudyInstanceUID":     str | None,
            "SeriesInstanceUID":    str,
            "Modality":             str | None,
            "Manufacturer":         str | None,
            "InstitutionName":      str | None,
            "Rows":                 int,
            "Columns":              int,
            "PixelSpacing":         [float, float] | None,
            "SliceThickness":       float | None,
            "SliceSpacingComputed": float | None,
            "NumberOfSlices":       int,
            "ZPositions":           list[float],
        }
    }

Logging
-------
Logger name: "dicom_loader"
  INFO    – series discovered, final volume shape per series
  WARNING – every skipped file (+ reason), every skipped series (+ reason)
  ERROR   – corrupt pixel data (logged immediately before the exception)

Exceptions
----------
* CorruptPixelDataError  – raised when pixel_array cannot be decoded for
                           any slice inside a series; the partial volume is
                           discarded and the exception propagates to the
                           caller so they can decide what to do.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from statistics import median
from typing import Any

import cupy as cp
import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError

# --------------------------------------------------------------------------- #
# Module-level logger                                                          #
# --------------------------------------------------------------------------- #
log = logging.getLogger("dicom_loader")

# DICOM SOP Class UID for the DICOMDIR media-storage directory object
_DICOMDIR_SOP_CLASS = "1.2.840.10008.1.3.10"

# Required tags every *image* file must carry (checked before grouping)
_REQUIRED_TAGS = ("SeriesInstanceUID", "ImagePositionPatient", "Rows", "Columns")


# --------------------------------------------------------------------------- #
# Public exception                                                             #
# --------------------------------------------------------------------------- #
class CorruptPixelDataError(RuntimeError):
    """Raised when pixel_array decoding fails for a slice in a series."""


# --------------------------------------------------------------------------- #
# Internal helpers                                                             #
# --------------------------------------------------------------------------- #

def _safe_str(value: Any) -> str | None:
    """Convert a pydicom tag value to a plain str, or None if absent/empty."""
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def _safe_float(value: Any) -> float | None:
    """Convert a pydicom tag value to float, or None."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_dicomdir(ds: pydicom.Dataset) -> bool:
    """Return True if this dataset is a DICOMDIR directory file."""
    sop = getattr(ds.file_meta, "MediaStorageSOPClassUID", None)
    if sop is not None and str(sop) == _DICOMDIR_SOP_CLASS:
        return True
    # Fallback: some writers omit file_meta but set the tag directly
    sop2 = getattr(ds, "MediaStorageSOPClassUID", None)
    return sop2 is not None and str(sop2) == _DICOMDIR_SOP_CLASS


def _has_required_tags(ds: pydicom.Dataset) -> tuple[bool, str]:
    """
    Check that all required image tags are present.
    Returns (ok: bool, missing_tag_name: str).
    missing_tag_name is empty string when ok=True.
    """
    for tag_name in _REQUIRED_TAGS:
        if not hasattr(ds, tag_name):
            return False, tag_name
        val = getattr(ds, tag_name)
        if val is None:
            return False, tag_name
    return True, ""


def _get_z(ds: pydicom.Dataset) -> float:
    """Extract the Z component of ImagePositionPatient."""
    ipp = ds.ImagePositionPatient
    return float(ipp[2])


def _apply_hu_rescale(arr: np.ndarray, ds: pydicom.Dataset) -> np.ndarray:
    """
    Apply CT Hounsfield Unit rescaling:
        HU = pixel_value * RescaleSlope + RescaleIntercept
    Falls back to slope=1, intercept=0 when tags are absent.
    """
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    return arr.astype(np.float32) * slope + intercept


def _extract_metadata(
    ds: pydicom.Dataset,
    series_uid: str,
    n_slices: int,
    z_positions: list[float],
    slice_spacing_computed: float | None,
) -> dict:
    """
    Build the metadata dict from the last DICOM dataset read in the series.
    All values are plain Python types — no pydicom objects.
    """
    # PixelSpacing → [row_mm, col_mm]
    pixel_spacing: list[float] | None = None
    ps = getattr(ds, "PixelSpacing", None)
    if ps is not None:
        try:
            pixel_spacing = [float(ps[0]), float(ps[1])]
        except (IndexError, TypeError, ValueError):
            pixel_spacing = None

    return {
        "PatientName":          _safe_str(getattr(ds, "PatientName", None)),
        "PatientID":            _safe_str(getattr(ds, "PatientID", None)),
        "PatientBirthDate":     _safe_str(getattr(ds, "PatientBirthDate", None)),
        "PatientSex":           _safe_str(getattr(ds, "PatientSex", None)),
        "StudyInstanceUID":     _safe_str(getattr(ds, "StudyInstanceUID", None)),
        "SeriesInstanceUID":    series_uid,
        "Modality":             _safe_str(getattr(ds, "Modality", None)),
        "Manufacturer":         _safe_str(getattr(ds, "Manufacturer", None)),
        "InstitutionName":      _safe_str(getattr(ds, "InstitutionName", None)),
        "Rows":                 int(ds.Rows),
        "Columns":              int(ds.Columns),
        "PixelSpacing":         pixel_spacing,
        "SliceThickness":       _safe_float(getattr(ds, "SliceThickness", None)),
        "SliceSpacingComputed": slice_spacing_computed,
        "NumberOfSlices":       n_slices,
        "ZPositions":           z_positions,
    }


# --------------------------------------------------------------------------- #
# Per-series builder                                                           #
# --------------------------------------------------------------------------- #

def _build_series(
    series_uid: str,
    file_paths: list[str],
) -> dict:
    """
    Process one series:
      1. Read headers → sort by Z
      2. Validate uniform geometry
      3. Allocate CuPy volume  (n_slices + 1, Rows, Cols)
      4. Fill slices (HU rescale for CT, raw uint16 for MR/other)
      5. Compute spacing, extract metadata from last file
    Returns a SeriesResult dict.
    Raises CorruptPixelDataError if any slice cannot be decoded.
    """

    # ------------------------------------------------------------------ #
    # Step 3a  —  read headers, collect Z values, sort                   #
    # ------------------------------------------------------------------ #
    slice_info: list[tuple[float, str]] = []   # (z_value, file_path)

    for path in file_paths:
        try:
            ds_hdr = pydicom.dcmread(path, stop_before_pixels=True)
        except Exception as exc:
            log.warning("Series %s: cannot re-read header of '%s': %s — skipping slice",
                        series_uid, os.path.basename(path), exc)
            continue
        try:
            z = _get_z(ds_hdr)
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            log.warning("Series %s: cannot read ImagePositionPatient[2] from '%s': %s — skipping slice",
                        series_uid, os.path.basename(path), exc)
            continue
        slice_info.append((z, path))

    if not slice_info:
        raise ValueError(f"Series {series_uid}: no valid slices after Z-sort pass")

    # Sort ascending by Z
    slice_info.sort(key=lambda t: t[0])
    z_positions  = [t[0] for t in slice_info]
    sorted_paths = [t[1] for t in slice_info]
    n_slices     = len(sorted_paths)

    # ------------------------------------------------------------------ #
    # Step 3b  —  validate uniform geometry                              #
    # ------------------------------------------------------------------ #
    # Read the first header to get reference Rows / Columns / Modality
    ds_ref = pydicom.dcmread(sorted_paths[0], stop_before_pixels=True)
    ref_rows    = int(ds_ref.Rows)
    ref_cols    = int(ds_ref.Columns)
    modality    = _safe_str(getattr(ds_ref, "Modality", None))

    for path in sorted_paths[1:]:
        try:
            ds_chk = pydicom.dcmread(path, stop_before_pixels=True)
        except Exception:
            continue  # already warned above; skip geometry check for this file
        if int(ds_chk.Rows) != ref_rows or int(ds_chk.Columns) != ref_cols:
            raise ValueError(
                f"Series {series_uid}: inconsistent geometry in '{os.path.basename(path)}' "
                f"({ds_chk.Rows}×{ds_chk.Columns} vs reference {ref_rows}×{ref_cols})"
            )

    # ------------------------------------------------------------------ #
    # Step 3c  —  allocate CuPy volume with Z safety margin              #
    # ------------------------------------------------------------------ #
    # +1 safety-margin slice (zero-padded) as agreed in the spec
    z_dim  = n_slices + 1
    is_ct  = (modality == "CT")
    dtype  = cp.float32 if is_ct else cp.uint16

    log.info("Series %s [%s]: %d slices → allocating CuPy volume (%d, %d, %d) dtype=%s",
             series_uid, modality, n_slices, z_dim, ref_rows, ref_cols, dtype)

    volume = cp.zeros((z_dim, ref_rows, ref_cols), dtype=dtype)

    # ------------------------------------------------------------------ #
    # Step 3d  —  fill slices                                            #
    # ------------------------------------------------------------------ #
    last_ds: pydicom.Dataset | None = None

    for i, path in enumerate(sorted_paths):
        try:
            ds = pydicom.dcmread(path)           # full read, pixels included
        except InvalidDicomError as exc:
            log.error("Series %s: InvalidDicomError reading '%s': %s",
                      series_uid, os.path.basename(path), exc)
            raise CorruptPixelDataError(
                f"Series {series_uid}: corrupt DICOM file '{os.path.basename(path)}': {exc}"
            ) from exc
        except Exception as exc:
            log.error("Series %s: unexpected error reading '%s': %s",
                      series_uid, os.path.basename(path), exc)
            raise CorruptPixelDataError(
                f"Series {series_uid}: cannot read '{os.path.basename(path)}': {exc}"
            ) from exc

        try:
            pixel_array: np.ndarray = ds.pixel_array   # shape (Rows, Cols)
        except Exception as exc:
            log.error("Series %s: cannot decode pixel_array from '%s': %s",
                      series_uid, os.path.basename(path), exc)
            raise CorruptPixelDataError(
                f"Series {series_uid}: corrupt pixel data in '{os.path.basename(path)}': {exc}"
            ) from exc

        # CT → Hounsfield Unit rescaling; MR/other → raw value
        if is_ct:
            pixel_array = _apply_hu_rescale(pixel_array, ds)
        else:
            pixel_array = pixel_array.astype(np.uint16)

        # Transfer NumPy slice → CuPy volume
        volume[i] = cp.asarray(pixel_array)

        last_ds = ds   # keep the last dataset for metadata extraction

    # Slice [n_slices] is already zero-filled (safety margin) — no action needed

    # ------------------------------------------------------------------ #
    # Step 3e / 3f  —  Z positions + spacing                            #
    # ------------------------------------------------------------------ #
    slice_spacing_computed: float | None = None
    if n_slices >= 2:
        diffs = [z_positions[k + 1] - z_positions[k] for k in range(n_slices - 1)]
        slice_spacing_computed = float(median(diffs))

    # ------------------------------------------------------------------ #
    # Step 3g  —  metadata (from last_ds)                                #
    # ------------------------------------------------------------------ #
    assert last_ds is not None, "No slices were successfully read — should have raised earlier"
    metadata = _extract_metadata(
        ds=last_ds,
        series_uid=series_uid,
        n_slices=n_slices,
        z_positions=z_positions,
        slice_spacing_computed=slice_spacing_computed,
    )

    log.info("Series %s: volume shape %s, spacing Z=%.3f Y=%.3f X=%.3f mm",
             series_uid,
             volume.shape,
             slice_spacing_computed or 0.0,
             (metadata["PixelSpacing"] or [0.0, 0.0])[0],
             (metadata["PixelSpacing"] or [0.0, 0.0])[1])

    return {
        "volume":   volume,
        "metadata": metadata,
    }


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def load_dicom_folder(folder_path: str) -> dict[str, dict]:
    """
    Load every DICOM image series found in *folder_path* (flat scan, no
    subdirectory recursion).

    Parameters
    ----------
    folder_path : str
        Absolute or relative path to the folder containing DICOM files.

    Returns
    -------
    dict[str, dict]
        Keys are SeriesInstanceUID strings.
        Values are SeriesResult dicts — see module docstring for structure.

    Notes
    -----
    * Files that are not valid DICOM, are DICOMDIR, or lack required tags
      are silently skipped with a WARNING log.
    * If a series has inconsistent geometry (mixed Rows/Columns), the whole
      series is skipped with a WARNING log.
    * If pixel data is corrupt in any slice, CorruptPixelDataError is raised
      and propagates to the caller; the partial volume is discarded.
    """

    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Not a directory: '{folder_path}'")

    # ------------------------------------------------------------------ #
    # Step 1  —  flat file discovery + initial tag validation            #
    # ------------------------------------------------------------------ #
    # Map SeriesInstanceUID → list of file paths
    series_files: dict[str, list[str]] = defaultdict(list)

    all_entries = sorted(os.listdir(folder_path))   # deterministic order
    log.info("Scanning folder '%s' — %d entries found", folder_path, len(all_entries))

    for filename in all_entries:
        path = os.path.join(folder_path, filename)

        # Only process regular files
        if not os.path.isfile(path):
            continue

        # Try reading the header
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=False)
        except InvalidDicomError:
            log.warning("Skipping '%s': not a valid DICOM file", filename)
            continue
        except Exception as exc:
            log.warning("Skipping '%s': unexpected read error: %s", filename, exc)
            continue

        # Skip DICOMDIR
        if _is_dicomdir(ds):
            log.warning("Skipping '%s': DICOMDIR index file", filename)
            continue

        # Check required tags
        ok, missing = _has_required_tags(ds)
        if not ok:
            log.warning("Skipping '%s': missing required tag '%s'", filename, missing)
            continue

        series_uid = str(ds.SeriesInstanceUID).strip()
        series_files[series_uid].append(path)

    log.info("Found %d series after file scan", len(series_files))

    # ------------------------------------------------------------------ #
    # Step 2 + 3  —  build one volume per series                        #
    # ------------------------------------------------------------------ #
    results: dict[str, dict] = {}

    for series_uid, file_list in series_files.items():
        log.info("Processing series %s (%d candidate files)", series_uid, len(file_list))
        try:
            series_result = _build_series(series_uid, file_list)
        except CorruptPixelDataError:
            # Propagate immediately — caller decides how to handle it
            raise
        except ValueError as exc:
            # Geometry mismatch or no valid slices — skip series, keep going
            log.warning("Skipping series %s: %s", series_uid, exc)
            continue

        results[series_uid] = series_result

    log.info("load_dicom_folder complete: %d series loaded successfully", len(results))
    return results
