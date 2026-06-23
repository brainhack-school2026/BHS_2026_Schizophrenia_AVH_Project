#!/usr/bin/env python3
"""
Group statistics on seed-based FC maps: Fisher z, pairwise t-tests, one-way ANOVA.

Expects per-subject maps from 01_seed_fc_one_subject_mni.py, e.g.
  derivatives/fc/maps/sub-01_desc-LHG_schaefer_corr_sentences_mniBrain.nii.gz

Writes voxel-wise Benjamini-Hochberg FDR p-maps (*_p_fdr.nii.gz) and summary columns
min_p_fdr_bh / n_voxels_fdr_q (default q=0.05).

Usage:
    python scripts/03_group_fc_stats.py
    python scripts/03_group_fc_stats.py --map-tag sentences_mniBrain
    python scripts/03_group_fc_stats.py --plot
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn import datasets, plotting
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAPS_DIR = PROJECT_ROOT / "derivatives" / "fc" / "maps"
DEFAULT_SUBJECTS = PROJECT_ROOT / "scripts" / "fc_nine_per_group.tsv"
OUT_BASE = PROJECT_ROOT / "derivatives" / "fc" / "group_stats"

GROUP_SLUG = {"HC": "HC", "AVH-": "AVHm", "AVH+": "AVHp"}
PAIRWISE = (
    ("HC", "AVH-", "HC_vs_AVHm"),
    ("HC", "AVH+", "HC_vs_AVHp"),
    ("AVH-", "AVH+", "AVHm_vs_AVHp"),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fisher z + t-tests + ANOVA on FC r maps.")
    p.add_argument(
        "--map-tag",
        default="sentences_mniBrain",
        help="Suffix after corr_, e.g. sentences_mniBrain (full: *_corr_{tag}.nii.gz)",
    )
    p.add_argument("--subjects", type=Path, default=DEFAULT_SUBJECTS)
    p.add_argument("--maps-dir", type=Path, default=MAPS_DIR)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--min-r",
        type=float,
        default=1e-6,
        help="Mask voxels with mean |r| above this across loaded subjects",
    )
    p.add_argument("--plot", action="store_true", help="Save PNGs for t/F maps (p<0.05 mask)")
    p.add_argument("--plot-threshold", type=float, default=2.0, help="|t| or F display threshold")
    p.add_argument(
        "--fdr-q",
        type=float,
        default=0.05,
        help="Benjamini-Hochberg FDR q threshold (voxel-wise, in-brain only)",
    )
    return p.parse_args()


def load_subjects(tsv_path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with open(tsv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rows.append((row["participant_id"], row["group"]))
    return rows


def map_path(maps_dir: Path, subject: str, map_tag: str) -> Path:
    return maps_dir / f"{subject}_desc-LHG_schaefer_corr_{map_tag}.nii.gz"


def fisher_z(r: np.ndarray) -> np.ndarray:
    r = np.clip(r, -0.999999, 0.999999)
    return np.arctanh(r).astype(np.float32)


def voxelwise_anova_f(groups: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """One-way ANOVA F and p per voxel. Each group array shape (n_subj, n_voxels)."""
    k = len(groups)
    ns = [g.shape[0] for g in groups]
    n_total = sum(ns)
    grand = np.vstack(groups).mean(axis=0)
    ss_between = np.zeros(grand.shape, dtype=np.float64)
    for g, n_i in zip(groups, ns):
        ss_between += n_i * (g.mean(axis=0) - grand) ** 2
    ss_total = np.zeros(grand.shape, dtype=np.float64)
    for g in groups:
        ss_total += ((g - grand) ** 2).sum(axis=0)
    ss_within = ss_total - ss_between
    df_between = k - 1
    df_within = n_total - k
    ms_between = ss_between / df_between
    ms_within = ss_within / np.maximum(df_within, 1)
    f_stat = (ms_between / (ms_within + 1e-12)).astype(np.float32)
    p_vals = stats.f.sf(f_stat, df_between, df_within).astype(np.float32)
    return f_stat, p_vals


def save_nifti(data: np.ndarray, affine: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine), path)


def fdr_bh_in_mask(p_map: np.ndarray, brain_mask: np.ndarray, q: float) -> tuple[np.ndarray, int]:
    """
    Benjamini-Hochberg FDR across in-brain voxels.

    Returns full-volume adjusted p-map (1 outside mask) and count with p_fdr <= q.
    """
    p_fdr = np.ones_like(p_map, dtype=np.float32)
    p_in = p_map[brain_mask]
    if p_in.size:
        p_adj = stats.false_discovery_control(p_in, method="bh")
        p_fdr[brain_mask] = np.asarray(p_adj, dtype=np.float32)
    n_fdr = int(((p_fdr <= q) & brain_mask).sum())
    return p_fdr, n_fdr


def maybe_plot(
    stat_img: nib.Nifti1Image,
    out_png: Path,
    title: str,
    thresh: float,
    vmax: float | None = None,
) -> None:
    display = plotting.plot_stat_map(
        stat_img,
        bg_img=datasets.load_mni152_template(resolution=2),
        threshold=thresh,
        vmax=vmax,
        title=title,
        display_mode="ortho",
        draw_cross=False,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    display.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir or (OUT_BASE / args.map_tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = load_subjects(args.subjects)
    by_group: dict[str, list[np.ndarray]] = {"HC": [], "AVH-": [], "AVH+": []}
    affine = None
    shape = None
    missing: list[str] = []

    for sub, group in subjects:
        path = map_path(args.maps_dir, sub, args.map_tag)
        if not path.is_file():
            missing.append(str(path.name))
            continue
        img = nib.load(path)
        if affine is None:
            affine = img.affine
            shape = img.shape[:3]
        data = np.squeeze(img.get_fdata()).astype(np.float32)
        by_group[group].append(fisher_z(data))

    if missing:
        print("ERROR: Missing maps:")
        for m in missing:
            print(f"  {m}")
        print("Run batch FC first: bash scripts/run_fc_batch_mni_nine_per_group.sh")
        return 1

    for g, arrs in by_group.items():
        if len(arrs) != 9:
            print(f"WARNING: group {g} has {len(arrs)} subjects (expected 9)")

    stacked = {g: np.stack(arrs, axis=0) for g, arrs in by_group.items()}  # (9, nvox) after ravel
    n_voxels = int(np.prod(shape))
    for g in stacked:
        stacked[g] = stacked[g].reshape(stacked[g].shape[0], n_voxels)

    brain_mask = np.zeros(n_voxels, dtype=bool)
    for arrs in by_group.values():
        for z in arrs:
            brain_mask |= np.abs(z.ravel()) > args.min_r
    print(f"Voxels in mask: {brain_mask.sum()} / {n_voxels}")

    # Group mean z maps
    for g, z_stack in stacked.items():
        slug = GROUP_SLUG[g]
        mean_z = z_stack.mean(axis=0).reshape(shape)
        mean_z[~brain_mask.reshape(shape)] = 0.0
        save_nifti(mean_z, affine, out_dir / f"mean_fisherz_{slug}.nii.gz")
        print(f"Saved mean Fisher z: {slug}")

    # Pairwise t-tests (two-tailed)
    summary_path = out_dir / "group_comparison_summary.tsv"
    summary_rows = [
        [
            "comparison",
            "n_group1",
            "n_group2",
            "peak_stat",
            "min_p_uncorrected",
            "n_voxels_uncorrected_p_05",
            "min_p_fdr_bh",
            "n_voxels_fdr_q",
        ]
    ]

    for g1, g2, label in PAIRWISE:
        a = stacked[g1][:, brain_mask]
        b = stacked[g2][:, brain_mask]
        t_res = stats.ttest_ind(a, b, axis=0, equal_var=True, nan_policy="omit")
        t_map = np.zeros(n_voxels, dtype=np.float32)
        p_map = np.ones(n_voxels, dtype=np.float32)
        t_map[brain_mask] = np.asarray(t_res.statistic, dtype=np.float32)
        p_map[brain_mask] = np.asarray(t_res.pvalue, dtype=np.float32)
        t_vol = t_map.reshape(shape)
        p_vol = p_map.reshape(shape)

        save_nifti(t_vol, affine, out_dir / f"{label}_t.nii.gz")
        save_nifti(p_vol, affine, out_dir / f"{label}_p.nii.gz")

        p_fdr, n_fdr = fdr_bh_in_mask(p_map, brain_mask, args.fdr_q)
        save_nifti(p_fdr.reshape(shape), affine, out_dir / f"{label}_p_fdr.nii.gz")
        sig_fdr = ((p_fdr <= args.fdr_q) & brain_mask).astype(np.uint8).reshape(shape)
        save_nifti(sig_fdr, affine, out_dir / f"{label}_sig_fdr_q{args.fdr_q:.2f}.nii.gz")

        p_sig = p_map[brain_mask]
        n_unc = int((p_sig < 0.05).sum())
        if p_sig.size:
            idx_peak = int(np.argmin(p_sig))
            peak_p = float(p_sig[idx_peak])
            peak_t = float(t_map[brain_mask][idx_peak])
            min_p_fdr = float(p_fdr[brain_mask].min())
        else:
            peak_p, peak_t, min_p_fdr = np.nan, np.nan, np.nan

        summary_rows.append(
            [
                label,
                a.shape[0],
                b.shape[0],
                peak_t,
                peak_p,
                n_unc,
                min_p_fdr,
                n_fdr,
            ]
        )
        print(
            f"{label}: peak |t|={abs(peak_t):.2f}, min p(unc)={peak_p:.4g}, "
            f"voxels p<0.05 unc={n_unc}, min p(FDR)={min_p_fdr:.4g}, voxels FDR q<={args.fdr_q}: {n_fdr}"
        )

        if args.plot:
            t_img = nib.Nifti1Image(t_vol, affine)
            maybe_plot(
                t_img,
                out_dir / f"{label}_t.png",
                f"{label} (t map)",
                args.plot_threshold,
                vmax=4.0,
            )

    # One-way ANOVA (3 groups)
    hc, avhm, avhp = stacked["HC"][:, brain_mask], stacked["AVH-"][:, brain_mask], stacked["AVH+"][:, brain_mask]
    f_stat, p_vals = voxelwise_anova_f([hc, avhm, avhp])
    f_map = np.zeros(n_voxels, dtype=np.float32)
    p_anova = np.ones(n_voxels, dtype=np.float32)
    f_map[brain_mask] = f_stat
    p_anova[brain_mask] = p_vals
    f_vol = f_map.reshape(shape)
    p_vol = p_anova.reshape(shape)

    save_nifti(f_vol, affine, out_dir / "ANOVA_F.nii.gz")
    save_nifti(p_vol, affine, out_dir / "ANOVA_p.nii.gz")

    p_fdr_anova, n_fdr = fdr_bh_in_mask(p_anova, brain_mask, args.fdr_q)
    save_nifti(p_fdr_anova.reshape(shape), affine, out_dir / "ANOVA_p_fdr.nii.gz")
    sig_fdr = ((p_fdr_anova <= args.fdr_q) & brain_mask).astype(np.uint8).reshape(shape)
    save_nifti(sig_fdr, affine, out_dir / f"ANOVA_sig_fdr_q{args.fdr_q:.2f}.nii.gz")

    n_unc = int((p_vals < 0.05).sum())
    peak_p = float(p_vals.min()) if p_vals.size else np.nan
    peak_f = float(f_stat[np.argmin(p_vals)]) if p_vals.size else np.nan
    min_p_fdr = float(p_fdr_anova[brain_mask].min()) if brain_mask.any() else np.nan
    summary_rows.append(["ANOVA_3groups", 9, 9, peak_f, peak_p, n_unc, min_p_fdr, n_fdr])
    print(
        f"ANOVA: peak F={peak_f:.2f}, min p(unc)={peak_p:.4g}, voxels p<0.05 unc={n_unc}, "
        f"min p(FDR)={min_p_fdr:.4g}, voxels FDR q<={args.fdr_q}: {n_fdr}"
    )

    if args.plot:
        maybe_plot(
            nib.Nifti1Image(f_vol, affine),
            out_dir / "ANOVA_F.png",
            "One-way ANOVA (F)",
            args.plot_threshold,
            vmax=8.0,
        )

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(summary_rows)
    print(f"Summary table: {summary_path}")
    print(f"All group maps: {out_dir}")
    print(
        f"Note: n=9 per group. FDR = Benjamini-Hochberg across in-brain voxels (q={args.fdr_q}). "
        "Use n_voxels_fdr_q for inference; uncorrected counts are exploratory."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
