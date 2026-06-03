#!/usr/bin/env python3
# Tell the shell this file is a Python program.

"""
Seed-based FC (MNI whole-brain mask variant): same as 01_seed_fc_one_subject.py, but
maps and figures are clipped to fMRIPrep mask ∩ MNI152 whole-brain template.

Usage:
    python scripts/01_seed_fc_one_subject_mni.py --subject sub-55 --condition all
    python scripts/01_seed_fc_one_subject_mni.py --condition white-noise
"""

from __future__ import annotations  # Allow modern type hints (e.g. list[str]) on older Python.

import argparse  # Read command-line flags like --subject sub-55.
import json  # Save seed metadata (which Schaefer parcel was used).
import sys  # Exit codes and sys.exit.
from pathlib import Path  # Paths to files/folders in a cross-platform way.

import matplotlib.pyplot as plt  # Save PNG brain figures.
import nibabel as nib  # Load/save NIfTI brain images.
import numpy as np  # Arrays and math (correlations, masks).
import pandas as pd  # Read confounds TSV as a table.
from nilearn import datasets, image, plotting  # Atlases, image tools, brain plots.
from nilearn.maskers import NiftiMasker  # Extract time series from masks/parcels.

# BrainHack project folder (parent of scripts/).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# fMRIPrep outputs: flat func/ + anat/ layout from rsync.
SRC_DIR = PROJECT_ROOT / "fmriprep_reports"
# Task timing file (onsets of white-noise, words, etc.).
EVENTS_FILE = PROJECT_ROOT / "task-speech_events.tsv"
# Repetition time in seconds (one fMRI volume every 2 s).
TR = 2.0

# MNI coordinate near left Heschl's / primary auditory cortex (literature target).
# Used only to *choose* the closest left Schaefer parcel — not a sphere seed.
HESCHL_TARGET_MNI = (-42, -26, 10)
# Schaefer resolution: 100 parcels, 7 Yeo networks (matches pitch).
DEFAULT_SCHAEFER_ROIS = 100
# Default subject for quick testing.
DEFAULT_SUBJECT = "sub-55"
# Block types in task-speech_events.tsv (one FC map each with --all-block-maps).
BLOCK_CONDITIONS = ("white-noise", "words", "sentences", "reversed")
# Intelligible + unintelligible speech blocks combined (no white-noise).
SPEECH_CONDITIONS = ("words", "sentences", "reversed")
CONDITION_CHOICES = ("all", "speech", *BLOCK_CONDITIONS)
# Suffix on map/figure filenames (compare to 01_seed_fc_one_subject.py without this).
OUTPUT_VARIANT = "_mniBrain"
MNI_WHOLE_BRAIN_THRESHOLD = 0.1


def parse_args() -> argparse.Namespace:
    """Read optional command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Schaefer-seed FC with MNI whole-brain mask on saved maps/figures "
            "(duplicate of 01_seed_fc_one_subject.py)."
        )
    )
    parser.add_argument("--subject", default=DEFAULT_SUBJECT)  # e.g. sub-55
    parser.add_argument("--src-dir", type=Path, default=SRC_DIR)  # fmriprep_reports
    parser.add_argument(
        "--condition",
        choices=CONDITION_CHOICES,
        default="white-noise",
        help=(
            "white-noise (default) / words / sentences / reversed = that block only; "
            "all = full run; speech = words+sentences+reversed"
        ),
    )
    parser.add_argument(
        "--all-block-maps",
        action="store_true",
        help="Run FC once per block type plus combined speech (ignores --condition)",
    )
    parser.add_argument(
        "--schaefer-rois",
        type=int,
        default=DEFAULT_SCHAEFER_ROIS,
        choices=(100, 200, 300, 400, 500, 600, 800, 1000),
    )
    parser.add_argument(
        "--schaefer-max-dist",
        type=float,
        default=12.0,  # Include extra LH parcels within this many mm of target
    )
    parser.add_argument("--no-plot", action="store_true")  # Skip PNG if set
    parser.add_argument(
        "--smoothing-fwhm",
        type=float,
        default=4.0,
        help=(
            "Spatial smoothing FWHM (mm). Applied to 4D BOLD once, in-brain only, "
            "before maskers (maskers use smoothing_fwhm=None)."
        ),
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Re-save PNGs from existing maps (no FC recomputation)",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Optional suffix on output files (e.g. nosmooth -> *_nosmooth.png)",
    )
    parser.add_argument(
        "--legacy-masker-smoothing",
        action="store_true",
        help=(
            "Original pipeline: smooth inside NiftiMasker (no pre-smooth BOLD). "
            "For comparison with default pre-smooth."
        ),
    )
    return parser.parse_args()  # Parsed flags object


def find_bold_path(func_dir: Path, subject: str) -> Path:
    """Find this subject's preprocessed BOLD file in func/."""
    matches = sorted(func_dir.glob(f"{subject}_*_desc-preproc_bold.nii.gz"))  # Glob pattern
    if not matches:  # Nothing found
        raise FileNotFoundError(f"No preproc BOLD for {subject} in {func_dir}")
    return matches[0]  # First (only) match


def find_brain_mask(func_dir: Path, subject: str, bold_path: Path | None = None) -> Path:
    """
    Brain mask in the same grid as desc-preproc BOLD (MNI space).

    fMRIPrep often ships two masks; prefer space-MNI152NLin2009cAsym_* over
    the smaller native desc-brain_mask so we do not resample a mismatched mask.
    """
    matches = sorted(func_dir.glob(f"{subject}_*_desc-brain_mask.nii.gz"))
    if not matches:
        raise FileNotFoundError(f"No brain mask for {subject} in {func_dir}")

    mni = [p for p in matches if "MNI152NLin2009cAsym" in p.name]
    if mni:
        return mni[0]

    if bold_path is not None:
        bold_shape = nib.load(bold_path).shape[:3]
        for path in matches:
            if nib.load(path).shape[:3] == bold_shape:
                return path

    if len(matches) > 1:
        print(
            f"  Warning: multiple brain masks; using {matches[0].name}. "
            "Prefer MNI mask matching preproc BOLD."
        )
    return matches[0]


def find_confounds_tsv(func_dir: Path, subject: str) -> Path:
    """Find fMRIPrep confounds timeseries table."""
    matches = sorted(func_dir.glob(f"{subject}_*_desc-confounds_timeseries.tsv"))
    if not matches:
        raise FileNotFoundError(f"No confounds TSV for {subject} in {func_dir}")
    return matches[0]


def load_confound_matrix(tsv_path: Path) -> pd.DataFrame:
    """Build nuisance regressor matrix from fMRIPrep confounds (no global signal)."""
    df = pd.read_csv(tsv_path, sep="\t")  # Load TSV
    motion_base = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]  # 6 motion params
    motion_cols = [c for c in df.columns if c in motion_base]  # Keep those columns
    motion_deriv = [
        c for c in df.columns if any(c.startswith(m + "_") for m in motion_base)
    ]  # Motion derivatives (Friston-style expansion)
    cosine_cols = sorted(c for c in df.columns if c.startswith("cosine"))  # Slow drift
    tissue_cols = [c for c in ("white_matter", "csf") if c in df.columns]  # WM + CSF
    use_cols = motion_cols + motion_deriv + tissue_cols + cosine_cols  # Final list
    confounds = df[use_cols].apply(pd.to_numeric, errors="coerce")  # All numeric
    return confounds.fillna(confounds.mean(numeric_only=True))  # Fill NaNs with column mean


def _volume_indices_for_event_rows(
    rows: pd.DataFrame, n_volumes: int
) -> list[int]:
    """Collect BOLD volume indices covered by event rows."""
    volumes: list[int] = []
    for _, row in rows.iterrows():
        start = int(np.floor(row["onset"] / TR))
        end = int(np.ceil((row["onset"] + row["duration"]) / TR))
        volumes.extend(range(start, end))
    return volumes


def volume_indices_for_condition(events_path: Path, condition: str, n_volumes: int) -> np.ndarray:
    """Which timepoint indices belong to a task condition (or combined 'speech')."""
    events = pd.read_csv(events_path, sep="\t")
    if condition == "speech":
        names = SPEECH_CONDITIONS
    else:
        names = (condition,)
    volumes: list[int] = []
    for name in names:
        subset = events.loc[events["condition"] == name]
        if subset.empty:
            raise ValueError(f"No events rows for condition {name!r}")
        volumes.extend(_volume_indices_for_event_rows(subset, n_volumes))
    volumes = sorted({v for v in volumes if 0 <= v < n_volumes})
    if not volumes:
        raise ValueError(f"No valid volumes for condition {condition!r}")
    return np.asarray(volumes, dtype=int)


def subset_bold_for_condition(
    bold_img: nib.Nifti1Image,
    confounds: pd.DataFrame,
    condition: str,
    n_volumes: int,
) -> tuple[nib.Nifti1Image, pd.DataFrame, int | None]:
    """Return BOLD + confounds, optionally restricted to one condition's volumes."""
    if condition == "all":
        return bold_img, confounds, None
    vol_idx = volume_indices_for_condition(EVENTS_FILE, condition, n_volumes)
    bold_sub = image.index_img(bold_img, vol_idx)
    conf_sub = confounds.iloc[vol_idx].reset_index(drop=True)
    return bold_sub, conf_sub, len(vol_idx)


def _label_name(labels: list, label_id: int) -> str:
    """Schaefer label string for integer parcel ID."""
    if label_id < len(labels):
        name = labels[label_id]
        return name.decode() if isinstance(name, bytes) else str(name)  # bytes -> str
    return f"label_{label_id}"  # Fallback


def _parcel_centroid_mni(atlas_img: nib.Nifti1Image, label_id: int) -> tuple[float, float, float] | None:
    """Center of mass (MNI mm) for one Schaefer parcel."""
    data = np.asanyarray(atlas_img.dataobj, dtype=np.int32)  # Atlas voxel labels
    ijk = np.column_stack(np.where(data == label_id))  # Voxel indices for this parcel
    if len(ijk) == 0:  # Empty parcel
        return None
    xyz = nib.affines.apply_affine(atlas_img.affine, ijk)  # Voxel -> MNI coordinates
    return tuple(xyz.mean(axis=0))  # Average = centroid


def build_schaefer_lhg_seed(
    n_rois: int,
    target_mni: tuple[float, float, float],
    max_dist_mm: float,
    reference_bold_img=None,
) -> tuple[nib.Nifti1Image, dict]:
    """
    Build a binary mask for the left Schaefer parcel(s) nearest left Heschl's target.
    Returns seed mask image + metadata dict.
    """
    print(f"  Fetching Schaefer 2018 atlas ({n_rois} ROIs, 7 networks)...")
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=n_rois, yeo_networks=7)  # Download/load once
    atlas_img = image.load_img(atlas.maps)  # 3D label image
    labels = list(atlas.labels)  # Name per label ID
    data = np.asanyarray(atlas_img.dataobj, dtype=np.int32)
    unique_ids = [int(i) for i in np.unique(data) if i != 0]  # All parcel IDs (skip 0=background)

    candidates: list[dict] = []  # Left-hemisphere parcels with distance to target
    for label_id in unique_ids:
        name = _label_name(labels, label_id)
        if "_LH_" not in name and not name.startswith("LH_"):  # Left hemisphere only
            continue
        centroid = _parcel_centroid_mni(atlas_img, label_id)
        if centroid is None:
            continue
        dist = float(np.linalg.norm(np.array(centroid) - np.array(target_mni)))  # mm distance
        candidates.append(
            {"label_id": label_id, "name": name, "centroid": centroid, "dist_mm": dist}
        )

    if not candidates:
        raise RuntimeError("No left-hemisphere Schaefer parcels found.")

    candidates.sort(key=lambda x: x["dist_mm"])  # Nearest first
    closest = candidates[0]  # Best match to Heschl target
    selected = [closest]  # Start with closest parcel
    for c in candidates[1:]:  # Optionally add more LH parcels within max_dist_mm
        if c["dist_mm"] <= max_dist_mm:
            selected.append(c)

    seed_data = np.isin(data, [s["label_id"] for s in selected]).astype(np.uint8)  # Binary mask
    seed_img = nib.Nifti1Image(seed_data, atlas_img.affine, atlas_img.header)  # NIfTI mask

    if reference_bold_img is not None:  # Resample atlas (1 mm) -> BOLD grid (2 mm)
        seed_img = image.resample_to_img(
            seed_img, reference_bold_img, interpolation="nearest", copy=True
        )
        n_vox = int(np.sum(seed_img.get_fdata() > 0))  # Count seed voxels
        if n_vox < 10:
            raise RuntimeError(
                f"Schaefer seed has only {n_vox} voxels on BOLD grid — check atlas alignment."
            )
        meta_voxels = n_vox
    else:
        meta_voxels = int(seed_data.sum())

    meta = {  # Saved to JSON for methods / reproducibility
        "atlas": "Schaefer2018",
        "n_rois": n_rois,
        "target_mni_mm": list(target_mni),
        "max_dist_mm": max_dist_mm,
        "selected_parcels": selected,
        "n_voxels_on_bold_grid": meta_voxels,
    }
    return seed_img, meta


def smooth_bold_in_brain(
    bold_img: nib.Nifti1Image,
    brain_mask_img: nib.Nifti1Image,
    fwhm: float,
) -> nib.Nifti1Image:
    """
    Smooth 4D BOLD, then zero outside the brain mask.

    Avoids NiftiMasker smoothing the full FOV before extraction (edge bleed).
    """
    smoothed = image.smooth_img(bold_img, fwhm)
    mask_on_grid = image.resample_to_img(
        brain_mask_img, bold_img, interpolation="nearest", copy=True
    )
    in_brain = np.asarray(mask_on_grid.dataobj, dtype=np.float32) > 0
    data = np.asarray(smoothed.dataobj, dtype=np.float32)
    data = data * in_brain[..., np.newaxis]  # zero outside brain after smooth
    return nib.Nifti1Image(data, smoothed.affine, smoothed.header)


def extract_seed_timeseries(
    bold_img,
    confounds: np.ndarray,
    seed_img: nib.Nifti1Image,
    smoothing_fwhm: float | None = None,
) -> np.ndarray:
    """
    One seed time series: average BOLD inside Schaefer mask, with nuisance regression.
    seed_img must already be on the same 3D grid as bold_img.
    """
    masker = NiftiMasker(
        mask_img=seed_img,  # Schaefer seed mask
        smoothing_fwhm=smoothing_fwhm,
        detrend=True,  # Remove linear trend per voxel
        standardize=False,  # We z-score after averaging (not per voxel)
        memory="nilearn_cache",
        memory_level=1,
    )
    voxel_ts = masker.fit_transform(bold_img, confounds=confounds)  # shape: (n_time, n_voxels_in_seed)
    seed_ts = voxel_ts.mean(axis=1)  # One value per timepoint
    seed_ts = (seed_ts - seed_ts.mean()) / (seed_ts.std() + 1e-8)  # Z-score the averaged trace
    return seed_ts.ravel()  # 1D array length = n_timepoints


def seed_to_brain_correlation(
    bold_img,
    brain_mask_path: Path,
    confounds: pd.DataFrame,
    seed_ts: np.ndarray,
    smoothing_fwhm: float | None = None,
) -> nib.Nifti1Image:
    """Correlate seed time series with every in-brain voxel -> 3D correlation map."""
    confound_array = confounds.values  # NumPy for maskers
    brain_masker = NiftiMasker(
        mask_img=str(brain_mask_path),  # Only brain voxels
        smoothing_fwhm=smoothing_fwhm,
        detrend=True,
        standardize="zscore_sample",  # Z-score each voxel time series
        memory="nilearn_cache",
        memory_level=1,
    )
    brain_ts = brain_masker.fit_transform(bold_img, confounds=confound_array)  # (time, voxels)
    n_time = brain_ts.shape[0]  # Number of volumes
    # Pearson r (vectorized): each column of brain_ts vs seed_ts
    corr = np.clip((brain_ts.T @ seed_ts.ravel()) / (n_time - 1), -1.0, 1.0)
    return brain_masker.inverse_transform(corr.reshape(1, -1))  # Back to 3D NIfTI image


def condition_suffix(condition: str) -> str:
    """Filename tag, e.g. all, whitenoise, words, speech."""
    return "all" if condition == "all" else condition.replace("-", "")


def file_tag(tag: str) -> str:
    """Filename suffix; e.g. tag='nosmooth' -> '_nosmooth'."""
    tag = tag.strip().replace("-", "")
    return f"_{tag}" if tag else ""


def conditions_to_run(args: argparse.Namespace) -> list[str]:
    """Which condition(s) to process for this invocation."""
    if args.all_block_maps:
        return list(BLOCK_CONDITIONS) + ["speech"]
    return [args.condition]


def plot_threshold(condition: str, n_volumes: int | None) -> float:
    """Stricter display threshold for short block-only runs (less speckle)."""
    if condition == "all" or (n_volumes is not None and n_volumes > 150):
        return 0.2
    return 0.35


def mni152_background() -> nib.Nifti1Image:
    """MNI template for plot underlay (fetched once per process)."""
    if not hasattr(mni152_background, "_cache"):
        mni152_background._cache = datasets.load_mni152_template(resolution=2)  # type: ignore[attr-defined]
    return mni152_background._cache  # type: ignore[attr-defined]


def mni_whole_brain_on_grid(corr_img: nib.Nifti1Image) -> np.ndarray:
    """MNI152 whole-brain mask resampled to corr_img grid (boolean)."""
    template = mni152_background()
    mni_on_grid = image.resample_to_img(
        template, corr_img, interpolation="continuous", copy=True
    )
    return np.asanyarray(mni_on_grid.dataobj) > MNI_WHOLE_BRAIN_THRESHOLD


def mask_corr_for_display(
    corr_img: nib.Nifti1Image, brain_mask_img: nib.Nifti1Image
) -> nib.Nifti1Image:
    """Zero voxels outside fMRIPrep mask and MNI whole-brain template."""
    mask_on_grid = image.resample_to_img(
        brain_mask_img, corr_img, interpolation="nearest", copy=True
    )
    data = np.squeeze(corr_img.get_fdata()).astype(np.float32)
    in_fmri = np.asanyarray(mask_on_grid.dataobj) > 0
    in_mni = mni_whole_brain_on_grid(corr_img)
    data = np.where(in_fmri & in_mni, data, 0.0)
    return nib.Nifti1Image(data, corr_img.affine, corr_img.header)


def save_plot(
    corr_img: nib.Nifti1Image,
    fig_path: Path,
    title: str,
    cut_coords: tuple[float, float, float],
    brain_mask_img: nib.Nifti1Image,
    condition: str = "all",
    n_volumes: int | None = None,
) -> None:
    """Orthographic PNG on MNI anatomy, clipped to brain mask."""
    thresh = plot_threshold(condition, n_volumes)
    display_img = mask_corr_for_display(corr_img, brain_mask_img)
    display = plotting.plot_stat_map(
        display_img,
        bg_img=mni152_background(),
        threshold=thresh,
        vmax=0.6,
        title=title,
        cut_coords=cut_coords,
        draw_cross=True,
        display_mode="ortho",
        black_bg=False,
    )
    display.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> int:
    """Run Schaefer seed FC for one subject (one or many task conditions)."""
    args = parse_args()
    subject = args.subject if args.subject.startswith("sub-") else f"sub-{args.subject}"
    func_dir = args.src_dir / "func"
    run_conds = conditions_to_run(args)

    print("=" * 60)
    print("Seed-based FC + MNI whole-brain mask on maps/figures")
    print("=" * 60)
    print(f"Subject:       {subject}")
    print(f"Schaefer ROIs: {args.schaefer_rois}")
    print(f"Smoothing:     {args.smoothing_fwhm:.1f} mm FWHM")
    print(f"Conditions:    {', '.join(run_conds)}")
    print(f"Data folder:   {func_dir}")

    bold_path = find_bold_path(func_dir, subject)
    mask_path = find_brain_mask(func_dir, subject, bold_path)
    brain_mask_img = image.load_img(mask_path)
    print(f"  Brain mask:  {mask_path.name}")

    out_dir = PROJECT_ROOT / "derivatives" / "fc" / "maps"
    fig_dir = PROJECT_ROOT / "derivatives" / "fc" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    seed_info_path = (
        PROJECT_ROOT / "derivatives" / "fc" / "seeds"
        / f"{subject}_schaefer{args.schaefer_rois}_seed_info.json"
    )
    if seed_info_path.exists():
        with open(seed_info_path, encoding="utf-8") as f:
            meta = json.load(f)
        cut_coords = tuple(meta["selected_parcels"][0]["centroid"])
    else:
        cut_coords = HESCHL_TARGET_MNI

    suffix = file_tag(args.tag)

    if args.figures_only:
        print("\n--- Figures only (existing maps) ---")
        for condition in run_conds:
            cond_tag = condition_suffix(condition)
            map_path = out_dir / f"{subject}_desc-LHG_schaefer_corr_{cond_tag}{OUTPUT_VARIANT}{suffix}.nii.gz"
            if not map_path.exists():
                print(f"  Skip {condition}: missing {map_path.name}")
                continue
            n_vol = None if condition == "all" else len(
                volume_indices_for_condition(
                    EVENTS_FILE, condition, 10_000
                )
            )
            corr_img = image.load_img(map_path)
            fig_path = fig_dir / map_path.name.replace(".nii.gz", ".png")
            save_plot(
                corr_img,
                fig_path,
                f"{subject} | {condition}",
                cut_coords,
                brain_mask_img,
                condition=condition,
                n_volumes=n_vol,
            )
            print(f"  Saved figure: {fig_path}")
        print("\nDone.")
        return 0

    confounds_path = find_confounds_tsv(func_dir, subject)

    print("\nLoading BOLD (this can take a minute)...")
    bold_full = image.load_img(bold_path)
    n_volumes = bold_full.shape[-1]
    print(f"  {bold_path.name} | shape {bold_full.shape}")

    confounds_full = load_confound_matrix(confounds_path)
    print(f"  {confounds_full.shape[1]} nuisance regressors")

    smooth = None if args.smoothing_fwhm <= 0 else args.smoothing_fwhm
    if args.legacy_masker_smoothing:
        if smooth is None:
            print("\nLegacy masker smoothing: disabled (smoothing-fwhm 0).")
        else:
            print(
                f"\nLegacy pipeline: NiftiMasker smoothing FWHM {smooth:.1f} mm "
                "(no pre-smooth on BOLD)."
            )
    elif smooth is not None:
        print(
            f"\nPre-smoothing BOLD (FWHM {smooth:.1f} mm, in-brain only); "
            "maskers run without extra smoothing."
        )
        bold_full = smooth_bold_in_brain(bold_full, brain_mask_img, smooth)
    else:
        print("\nNo spatial smoothing (smoothing-fwhm 0).")
    masker_smooth = smooth if args.legacy_masker_smoothing else None

    print(f"\n--- Schaefer {args.schaefer_rois} ROIs: left parcel near Heschl's ---")
    seed_img, meta = build_schaefer_lhg_seed(
        args.schaefer_rois,
        HESCHL_TARGET_MNI,
        args.schaefer_max_dist,
        reference_bold_img=bold_full,
    )
    for p in meta["selected_parcels"]:
        c = p["centroid"]
        print(
            f"  Parcel {p['label_id']}: {p['name']} "
            f"| centroid ({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f}) "
            f"| {p['dist_mm']:.1f} mm from Heschl target"
        )
    print(f"  Seed voxels on BOLD grid: {meta['n_voxels_on_bold_grid']}")

    seed_info_dir = PROJECT_ROOT / "derivatives" / "fc" / "seeds"
    seed_info_dir.mkdir(parents=True, exist_ok=True)
    info_path = seed_info_dir / f"{subject}_schaefer{args.schaefer_rois}_seed_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved seed metadata: {info_path}")

    seed_mask_path = seed_info_dir / f"{subject}_schaefer{args.schaefer_rois}_seed_mask.nii.gz"
    seed_img.to_filename(seed_mask_path)
    print(f"  Saved seed mask: {seed_mask_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    cut_coords = tuple(meta["selected_parcels"][0]["centroid"])
    parcel_names = ", ".join(p["name"] for p in meta["selected_parcels"])
    last_out_path: Path | None = None

    for condition in run_conds:
        print(f"\n--- FC map: {condition} ---")
        bold_img, confounds, n_kept = subset_bold_for_condition(
            bold_full, confounds_full, condition, n_volumes
        )
        if n_kept is not None:
            print(f"  {n_kept} volumes ({condition} blocks only)")

        print("  Extracting seed time series...")
        seed_ts = extract_seed_timeseries(
            bold_img,
            confounds.values,
            seed_img,
            smoothing_fwhm=masker_smooth,
        )

        print("  Computing voxel correlations...")
        corr_img = seed_to_brain_correlation(
            bold_img,
            mask_path,
            confounds,
            seed_ts,
            smoothing_fwhm=masker_smooth,
        )
        corr_img = mask_corr_for_display(corr_img, brain_mask_img)

        cond_tag = condition_suffix(condition)
        out_name = f"{subject}_desc-LHG_schaefer_corr_{cond_tag}{OUTPUT_VARIANT}{suffix}.nii.gz"
        out_path = out_dir / out_name
        corr_img.to_filename(out_path)
        r_min, r_max = np.nanmin(corr_img.get_fdata()), np.nanmax(corr_img.get_fdata())
        print(f"  Saved map: {out_path}")
        print(f"  r range: {r_min:.3f} to {r_max:.3f}")
        last_out_path = out_path

        if not args.no_plot:
            pipe = "legacy masker smooth" if args.legacy_masker_smoothing else "pre-smooth BOLD"
            plot_title = f"{subject} | {condition} | MNI brain mask | {pipe}{suffix}"
            fig_path = fig_dir / out_name.replace(".nii.gz", ".png")
            save_plot(
                corr_img,
                fig_path,
                plot_title,
                cut_coords,
                brain_mask_img,
                condition=condition,
                n_volumes=n_kept,
            )
            print(f"  Saved figure: {fig_path}")

    print("\nDone.")
    if last_out_path is not None:
        print("Label clusters: python scripts/02_label_fc_map_clusters.py --map", last_out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())  # Run only when executed as script
