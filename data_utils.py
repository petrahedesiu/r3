#!/usr/bin/env python3
"""
Data Utilities for Medical Image Segmentation
=============================================
Handles loading and aligning DICOM images with NRRD segmentation files,
accounting for different coordinate systems used by 3D Slicer.
"""

import numpy as np
import pydicom
import nrrd
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from tqdm import tqdm
import warnings


def load_dicom_series(dicom_dir: str, verbose: bool = True) -> Tuple[np.ndarray, Dict]:
    """
    Load a DICOM series from a directory.
    
    Returns:
        volume: 3D numpy array (H, W, D)
        metadata: Dictionary with spatial information
    """
    dicom_dir = Path(dicom_dir)
    dcm_files = list(dicom_dir.glob("*.dcm")) + list(dicom_dir.glob("*.DCM"))
    
    if not dcm_files:
        raise FileNotFoundError(f"No DICOM files found in {dicom_dir}")
    
    if verbose:
        print(f"Found {len(dcm_files)} DICOM files")
    
    # Load all slices
    slices = []
    for f in tqdm(dcm_files, desc="Loading DICOMs", disable=not verbose):
        try:
            ds = pydicom.dcmread(str(f))
            slices.append(ds)
        except Exception as e:
            if verbose:
                print(f"  Warning: Failed to load {f.name}: {e}")
    
    if not slices:
        raise ValueError("No valid DICOM files could be loaded")
    
    # Sort by slice position
    try:
        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
        sorted_by = "ImagePositionPatient"
    except AttributeError:
        try:
            slices.sort(key=lambda s: float(s.SliceLocation))
            sorted_by = "SliceLocation"
        except AttributeError:
            slices.sort(key=lambda s: float(s.InstanceNumber))
            sorted_by = "InstanceNumber"
    
    # Stack into 3D volume, converting stored pixel values to Hounsfield
    # units via the per-slice RescaleSlope / RescaleIntercept tags. HU is
    # required for fixed CT windowing; min-max-normalised pipelines are
    # unaffected since min-max is invariant to a positive affine rescale.
    volume = np.stack([
        s.pixel_array.astype(np.float32) * float(getattr(s, "RescaleSlope", 1.0))
        + float(getattr(s, "RescaleIntercept", 0.0))
        for s in slices
    ], axis=-1)

    # Extract metadata
    ds = slices[0]
    pixel_spacing = list(map(float, getattr(ds, "PixelSpacing", [1.0, 1.0])))
    slice_thickness = float(getattr(ds, "SliceThickness", 1.0))
    
    # Calculate slice spacing from positions if available
    if len(slices) > 1 and hasattr(slices[0], 'ImagePositionPatient'):
        pos0 = np.array(slices[0].ImagePositionPatient)
        pos1 = np.array(slices[1].ImagePositionPatient)
        slice_spacing = np.linalg.norm(pos1 - pos0)
    else:
        slice_spacing = slice_thickness
    
    metadata = {
        "pixel_spacing": pixel_spacing,
        "slice_thickness": slice_thickness,
        "slice_spacing": slice_spacing,
        "rows": ds.Rows,
        "columns": ds.Columns,
        "num_slices": len(slices),
        "modality": getattr(ds, "Modality", "Unknown"),
        "patient_id": getattr(ds, "PatientID", "Unknown"),
        "sorted_by": sorted_by,
        "image_position": list(map(float, getattr(ds, "ImagePositionPatient", [0, 0, 0]))),
        "image_orientation": list(map(float, getattr(ds, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0]))),
        "rescale_slope": float(getattr(ds, "RescaleSlope", 1.0)),
        "rescale_intercept": float(getattr(ds, "RescaleIntercept", 0.0)),
    }
    
    if verbose:
        print(f"DICOM volume shape: {volume.shape}")
        print(f"Pixel spacing: {pixel_spacing}, Slice spacing: {slice_spacing:.2f}")
    
    return volume, metadata


def load_nrrd_segmentation(nrrd_path: str, verbose: bool = True) -> Tuple[np.ndarray, Dict]:
    """
    Load NRRD segmentation file with metadata.
    
    Returns:
        segmentation: 3D numpy array with integer labels
        metadata: Dictionary with spatial info and segment names
    """
    if verbose:
        print(f"Loading NRRD from {nrrd_path}")
    
    data, header = nrrd.read(nrrd_path)
    
    # Parse space directions (3x3 matrix)
    space_directions = header.get("space directions", np.eye(3))
    if isinstance(space_directions, list):
        space_directions = np.array(space_directions)
    
    # Parse space origin
    space_origin = header.get("space origin", [0, 0, 0])
    if isinstance(space_origin, list):
        space_origin = np.array(space_origin)
    
    # Parse segment information
    segments = {}
    for key, value in header.items():
        if key.startswith("Segment") and "_Name" in key:
            seg_idx = key.split("_")[0].replace("Segment", "")
            label_key = f"Segment{seg_idx}_LabelValue"
            if label_key in header:
                try:
                    label = int(header[label_key])
                    segments[label] = value
                except:
                    pass
    
    metadata = {
        "space": header.get("space", "unknown"),
        "space_directions": space_directions,
        "space_origin": space_origin,
        "segments": segments,
        "header": header,
    }
    
    if verbose:
        print(f"NRRD shape: {data.shape}")
        print(f"Unique labels: {np.unique(data)}")
        print(f"Segments: {segments}")
    
    return data.astype(np.int64), metadata


def align_nrrd_to_dicom(
    dicom_volume: np.ndarray,
    nrrd_volume: np.ndarray,
    nrrd_metadata: Dict,
    verbose: bool = True,
    dicom_metadata: Optional[Dict] = None,
) -> Tuple[np.ndarray, bool]:
    """
    Align NRRD segmentation to DICOM volume using world coordinates.

    Uses DICOM spatial metadata (ImagePositionPatient, ImageOrientationPatient,
    PixelSpacing, slice_spacing) and NRRD spatial metadata (space directions,
    space origin) to compute a proper coordinate-based mapping.

    Returns:
        aligned_nrrd: NRRD aligned to DICOM shape (H, W, D)
        success: Whether alignment was successful
    """
    if verbose:
        print(f"\nAligning: DICOM {dicom_volume.shape} <-> NRRD {nrrd_volume.shape}")

    # Already aligned?
    if dicom_volume.shape == nrrd_volume.shape:
        if verbose:
            print("Shapes already match!")
        return nrrd_volume, True

    # --- Coordinate-based alignment (preferred) ---
    if dicom_metadata is not None:
        result = _align_by_coordinates(
            dicom_volume, nrrd_volume, dicom_metadata, nrrd_metadata, verbose
        )
        if result is not None:
            return result, True

    # --- Fallback: simple permutation (shape-only) ---
    result = _try_simple_alignment(dicom_volume, nrrd_volume, verbose)
    if result is not None:
        return result, True

    if verbose:
        print("Could not find alignment")
    return nrrd_volume, False


def _align_by_coordinates(
    dicom_volume: np.ndarray,
    nrrd_volume: np.ndarray,
    dicom_metadata: Dict,
    nrrd_metadata: Dict,
    verbose: bool = True,
) -> Optional[np.ndarray]:
    """Align using world-coordinate mapping between DICOM and NRRD.

    Builds the affine transforms for both volumes, computes the NRRD voxel
    index corresponding to each corner of the DICOM grid, extracts the
    sub-volume, and applies any necessary axis flips.
    """
    # --- DICOM affine components ---
    ipp = np.array(dicom_metadata.get("image_position", [0, 0, 0]), dtype=np.float64)
    iop = np.array(
        dicom_metadata.get("image_orientation", [1, 0, 0, 0, 1, 0]), dtype=np.float64
    )
    ps = dicom_metadata.get("pixel_spacing", [1.0, 1.0])
    row_spacing, col_spacing = float(ps[0]), float(ps[1])
    slice_spacing = float(dicom_metadata.get("slice_spacing", 1.0))

    # DICOM ImageOrientationPatient (PS3.3 C.7.6.2.1.1):
    #   iop[0:3] = "direction cosines of the first row"
    #            = direction ALONG the row = direction of COLUMN increase
    #   iop[3:6] = "direction cosines of the first column"
    #            = direction ALONG the column = direction of ROW increase
    col_dir = iop[:3]   # direction of increasing column index
    row_dir = iop[3:]   # direction of increasing row index
    slice_dir = np.cross(col_dir, row_dir)  # normal to image plane

    # DICOM PixelSpacing:
    #   ps[0] = row spacing (distance between adjacent rows)
    #   ps[1] = column spacing (distance between adjacent columns)
    #
    # DICOM pixel (row=r, col=c, slice=s) -> world:
    #   world = ipp + c * col_dir * col_spacing
    #                + r * row_dir * row_spacing
    #                + s * slice_dir * slice_spacing

    # --- NRRD affine components ---
    space_dirs = nrrd_metadata.get("space_directions", np.eye(3))
    if isinstance(space_dirs, list):
        space_dirs = np.array(space_dirs, dtype=np.float64)
    space_origin = np.array(
        nrrd_metadata.get("space_origin", [0, 0, 0]), dtype=np.float64
    )

    # NRRD voxel (i, j, k) -> world:  world = space_origin + space_dirs^T @ [i, j, k]
    # We invert: voxel = inv(space_dirs) @ (world - space_origin)
    try:
        inv_space_dirs = np.linalg.inv(space_dirs)
    except np.linalg.LinAlgError:
        if verbose:
            print("NRRD space_directions matrix is singular, cannot invert")
        return None

    H, W, D = dicom_volume.shape

    # Compute NRRD voxel index for DICOM voxel (0,0,0) and the three unit steps
    world_origin = ipp  # DICOM voxel (0,0,0)
    nrrd_origin = inv_space_dirs @ (world_origin - space_origin)

    # Step vectors in NRRD voxel space for one DICOM voxel step
    step_row = inv_space_dirs @ (row_dir * row_spacing)
    step_col = inv_space_dirs @ (col_dir * col_spacing)
    step_slice = inv_space_dirs @ (slice_dir * slice_spacing)

    if verbose:
        print(f"  NRRD origin (for DICOM 0,0,0): {nrrd_origin}")
        print(f"  NRRD step per DICOM row:   {step_row}")
        print(f"  NRRD step per DICOM col:   {step_col}")
        print(f"  NRRD step per DICOM slice: {step_slice}")

    # Build the full NRRD index array for every DICOM voxel would be huge.
    # Instead, since DICOM->NRRD is an affine mapping of axis-aligned grids,
    # each DICOM axis maps to exactly one NRRD axis (with possible sign flip).
    # Detect which NRRD axis each DICOM axis maps to.

    steps = np.array([step_row, step_col, step_slice])  # (3, 3)
    # For each DICOM axis, find the dominant NRRD axis
    axis_map = {}  # dicom_axis -> nrrd_axis
    axis_sign = {}  # dicom_axis -> +1 or -1
    for d_ax in range(3):
        abs_step = np.abs(steps[d_ax])
        n_ax = int(np.argmax(abs_step))
        # Verify this is truly axis-aligned (dominant component >> others)
        if abs_step[n_ax] < 1e-6:
            if verbose:
                print(f"  DICOM axis {d_ax} has zero step in NRRD space")
            return None
        off_axis = np.delete(abs_step, n_ax)
        if np.any(off_axis > 0.1 * abs_step[n_ax]):
            if verbose:
                print(f"  DICOM axis {d_ax} is not axis-aligned in NRRD space: {steps[d_ax]}")
            return None
        axis_map[d_ax] = n_ax
        axis_sign[d_ax] = 1 if steps[d_ax][n_ax] > 0 else -1

    # Check we have a valid 1-to-1 mapping
    if len(set(axis_map.values())) != 3:
        if verbose:
            print(f"  Axis mapping is not 1-to-1: {axis_map}")
        return None

    if verbose:
        labels = ["row", "col", "slice"]
        for d_ax in range(3):
            sign = "+" if axis_sign[d_ax] > 0 else "-"
            print(f"  DICOM {labels[d_ax]} -> NRRD axis {axis_map[d_ax]} ({sign})")

    # Compute the NRRD index range for each DICOM axis.
    # If the DICOM volume extends slightly beyond the NRRD canvas, clamp to
    # the valid range and zero-pad afterwards (those edge voxels are background).
    dicom_sizes = [H, W, D]
    slices_per_axis = [None, None, None]  # slice objects for NRRD extraction
    pad_before = [0, 0, 0]  # padding needed before the extracted region (per NRRD axis)
    pad_after = [0, 0, 0]   # padding needed after

    for d_ax in range(3):
        n_ax = axis_map[d_ax]
        start_nrrd = nrrd_origin[n_ax]
        step = steps[d_ax][n_ax]
        end_nrrd = start_nrrd + step * (dicom_sizes[d_ax] - 1)

        lo = min(start_nrrd, end_nrrd)
        hi = max(start_nrrd, end_nrrd)
        lo_int = int(round(lo))
        hi_int = int(round(hi))
        expected_size = hi_int - lo_int + 1

        # Clamp to valid NRRD range, track how much padding is needed
        clamped_lo = max(0, lo_int)
        clamped_hi = min(nrrd_volume.shape[n_ax] - 1, hi_int)

        pb = clamped_lo - lo_int   # voxels clipped at the low end
        pa = hi_int - clamped_hi   # voxels clipped at the high end

        # Reject if more than 30% of the axis is out of bounds
        if pb + pa > 0.3 * expected_size:
            if verbose:
                print(
                    f"  NRRD axis {n_ax}: range [{lo_int}, {hi_int}] has "
                    f"{pb + pa}/{expected_size} voxels out of bounds "
                    f"[0, {nrrd_volume.shape[n_ax] - 1}] (>30%)"
                )
            return None

        pad_before[n_ax] = pb
        pad_after[n_ax] = pa
        slices_per_axis[n_ax] = slice(clamped_lo, clamped_hi + 1)

    # Extract the sub-volume from NRRD
    extracted = nrrd_volume[slices_per_axis[0], slices_per_axis[1], slices_per_axis[2]]

    # Zero-pad if any edges were clipped
    if any(p > 0 for p in pad_before) or any(p > 0 for p in pad_after):
        pad_widths = [(pad_before[ax], pad_after[ax]) for ax in range(3)]
        extracted = np.pad(extracted, pad_widths, mode="constant", constant_values=0)
        if verbose:
            print(f"  Padded {pad_widths} to compensate for edge clipping")

    if verbose:
        print(f"  Extracted NRRD region: {[str(s) for s in slices_per_axis]}, shape {extracted.shape}")

    # Permute axes: we need NRRD axes in the order [axis_map[0], axis_map[1], axis_map[2]]
    # so that the result is (H, W, D) matching DICOM
    perm = [axis_map[0], axis_map[1], axis_map[2]]
    if perm != [0, 1, 2]:
        extracted = np.transpose(extracted, perm)

    # Apply flips for negative axis signs
    for d_ax in range(3):
        if axis_sign[d_ax] < 0:
            extracted = np.flip(extracted, axis=d_ax)

    # Verify shape
    if extracted.shape != dicom_volume.shape:
        if verbose:
            print(
                f"  Shape mismatch after extraction: {extracted.shape} vs {dicom_volume.shape}"
            )
        return None

    # Make contiguous copy (np.flip returns a view)
    extracted = np.ascontiguousarray(extracted)

    if verbose:
        n_labels = np.sum(extracted > 0)
        print(f"  Coordinate-based alignment successful, {n_labels} label voxels")

    return extracted


def _try_simple_alignment(
    dicom_volume: np.ndarray,
    nrrd_volume: np.ndarray,
    verbose: bool = True,
) -> Optional[np.ndarray]:
    """Try simple axis permutations to align volumes (shape-only fallback)."""

    dicom_shape = dicom_volume.shape
    nrrd_shape = nrrd_volume.shape

    permutations = [
        (0, 1, 2),
        (1, 0, 2),
        (0, 2, 1),
        (2, 1, 0),
        (1, 2, 0),
        (2, 0, 1),
    ]

    for perm in permutations:
        transformed = np.transpose(nrrd_volume, perm)
        if transformed.shape == dicom_shape:
            if verbose:
                print(f"  Fallback alignment: transpose{perm}")
            return transformed

    return None


def get_labeled_slice_indices(segmentation: np.ndarray) -> List[int]:
    """Get indices of slices containing labels."""
    labeled = []
    for i in range(segmentation.shape[-1]):
        if np.any(segmentation[:, :, i] > 0):
            labeled.append(i)
    return labeled


def load_patient_data(
    dicom_dir: str,
    nrrd_path: str,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load and align a patient's DICOM and NRRD data.
    
    Returns:
        dicom_volume: 3D DICOM volume
        aligned_segmentation: Aligned NRRD segmentation
        metadata: Combined metadata
    """
    # Load both
    dicom_vol, dicom_meta = load_dicom_series(dicom_dir, verbose)
    nrrd_vol, nrrd_meta = load_nrrd_segmentation(nrrd_path, verbose)
    
    # Align
    aligned_seg, success = align_nrrd_to_dicom(
        dicom_vol, nrrd_vol, nrrd_meta, verbose, dicom_metadata=dicom_meta
    )
    
    if not success:
        warnings.warn(f"Alignment failed for {dicom_dir}")
    
    # Combined metadata
    metadata = {
        **dicom_meta,
        "segments": nrrd_meta.get("segments", {}),
        "alignment_success": success,
    }
    
    return dicom_vol, aligned_seg, metadata


def discover_patients(base_dir: str) -> List[Dict]:
    """
    Discover all patient folders in a dataset directory.
    
    Returns list of dicts with 'dicom_dir' and 'nrrd_path' keys.
    """
    base = Path(base_dir)
    patients = []
    
    for patient_dir in sorted(base.iterdir()):
        if not patient_dir.is_dir() or patient_dir.name.startswith('.'):
            continue
        
        # Find NRRD file
        nrrd_files = list(patient_dir.glob("*.nrrd"))
        if not nrrd_files:
            continue
        
        # Find DICOM subdirectory — pick the one with the most .dcm files
        # (e.g. patient 001 has NL001/ with DICOMs and NL001_previews/ without)
        dicom_dirs = [d for d in patient_dir.iterdir() if d.is_dir()]
        if not dicom_dirs:
            continue

        best_dicom_dir = None
        best_dcm_count = 0
        for d in dicom_dirs:
            n = len(list(d.glob("*.dcm")) + list(d.glob("*.DCM")))
            if n > best_dcm_count:
                best_dcm_count = n
                best_dicom_dir = d

        if best_dicom_dir is None or best_dcm_count == 0:
            continue
        dicom_dir = best_dicom_dir
        dcm_files = list(dicom_dir.glob("*.dcm")) + list(dicom_dir.glob("*.DCM"))
        
        patients.append({
            "patient_id": patient_dir.name,
            "dicom_dir": str(dicom_dir),
            "nrrd_path": str(nrrd_files[0]),
        })
    
    return patients


if __name__ == "__main__":
    # Test with sample patient
    import sys
    
    if len(sys.argv) >= 3:
        dicom_dir = sys.argv[1]
        nrrd_path = sys.argv[2]
        
        dicom, seg, meta = load_patient_data(dicom_dir, nrrd_path)
        
        print(f"\n=== Results ===")
        print(f"DICOM shape: {dicom.shape}")
        print(f"Segmentation shape: {seg.shape}")
        print(f"Alignment success: {meta['alignment_success']}")
        print(f"Labeled slices: {len(get_labeled_slice_indices(seg))}")
    else:
        print("Usage: python data_utils.py <dicom_dir> <nrrd_path>")


