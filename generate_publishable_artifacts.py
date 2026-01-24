#!/usr/bin/env python3
"""
Generate publishable (paper/journal-ready) figures and tables from the existing `runs/` outputs.

Inputs (expected from the previous notebook):
- runs/_aggregate_metrics.csv
- runs/_ablation.csv (optional)
- runs/<run_id>/confusion_matrix.csv OR runs/<run_id>/confusion_matrix.json

Outputs (created):
- runs/publishable/figures/*.png and *.pdf
- runs/publishable/tables/*.csv and *.tex
- runs/publishable/index.csv

Minimal user steps:
1) Put this file at the repo root (same level as `runs/`).
2) Run:  python3 generate_publishable_artifacts.py
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RUNS_DIR = Path("runs")
OUT_DIR = RUNS_DIR / "publishable"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

def _safe_read_json(p: Path):
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception:
        return None

def _load_confusion(run_dir: Path):
    csv_p = run_dir / "confusion_matrix.csv"
    json_p = run_dir / "confusion_matrix.json"

    if csv_p.exists():
        try:
            df = pd.read_csv(csv_p, index_col=0)
            labels = df.index.astype(str).tolist()
            mat = df.values.astype(int)
            return labels, mat
        except Exception:
            pass

    if json_p.exists():
        obj = _safe_read_json(json_p)
        if obj and "labels" in obj and "matrix" in obj:
            labels = [str(x) for x in obj["labels"]]
            mat = np.asarray(obj["matrix"], dtype=int)
            return labels, mat

    return None, None

def plot_confusion_matrix(labels, cm, title, out_png: Path, out_pdf: Path):
    n = len(labels)
    fig_w = 8 if n <= 15 else 12
    fig_h = 7 if n <= 15 else 10
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)

    ax = plt.gca()
    im = ax.imshow(cm, interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    if n <= 25 and cm.size:
        thresh = cm.max() * 0.5
        for i in range(n):
            for j in range(n):
                ax.text(j, i, int(cm[i, j]),
                        ha="center", va="center",
                        fontsize=6,
                        color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

def make_table_tex(df: pd.DataFrame, path_tex: Path, caption="Results", label="tab:results"):
    tex = df.to_latex(index=False, escape=True, caption=caption, label=label)
    path_tex.write_text(tex)

def _round_df(df: pd.DataFrame, nd=4):
    out = df.copy()
    for c in out.columns:
        if out[c].dtype.kind in "fc":
            out[c] = out[c].round(nd)
    return out

def main():
    agg_csv = RUNS_DIR / "_aggregate_metrics.csv"
    ab_csv  = RUNS_DIR / "_ablation.csv"

    if not agg_csv.exists():
        raise FileNotFoundError(f"Missing {agg_csv}. Run the training notebook first.")

    agg = pd.read_csv(agg_csv)

    # Choose columns for Table IV (only those that exist)
    table_cols = [c for c in [
        "run_id","img_res","growth_rate","compression","depth",
        "accuracy","macro_precision","macro_recall","macro_f1",
        "avg_latency_ms","fps","p50_latency_ms","p90_latency_ms","p95_latency_ms","p99_latency_ms"
    ] if c in agg.columns]

    agg_clean = agg[table_cols].copy() if table_cols else agg.copy()
    agg_clean.to_csv(TAB_DIR / "Table_IV_aggregate_clean.csv", index=False)
    agg_round = _round_df(agg_clean, 4)
    agg_round.to_csv(TAB_DIR / "Table_IV_aggregate_rounded.csv", index=False)
    make_table_tex(agg_round, TAB_DIR / "Table_IV_aggregate.tex",
                   caption="Aggregate results (Table IV)", label="tab:tableIV")

    # Accuracy vs Resolution
    if "img_res" in agg.columns and "accuracy" in agg.columns:
        df = agg.dropna(subset=["img_res","accuracy"]).copy()
        if not df.empty:
            df["img_res"] = pd.to_numeric(df["img_res"], errors="coerce")
            df = df.dropna(subset=["img_res"]).sort_values("img_res")
            fig = plt.figure(figsize=(6.5, 4.2), dpi=200)
            plt.plot(df["img_res"], df["accuracy"], marker="o")
            plt.xlabel("Input resolution (pixels)")
            plt.ylabel("Accuracy")
            plt.title("Accuracy vs Resolution")
            plt.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(FIG_DIR / "accuracy_vs_resolution.png", bbox_inches="tight")
            fig.savefig(FIG_DIR / "accuracy_vs_resolution.pdf", bbox_inches="tight")
            plt.close(fig)

    # Latency vs Resolution
    if "img_res" in agg.columns and "avg_latency_ms" in agg.columns:
        df = agg.dropna(subset=["img_res","avg_latency_ms"]).copy()
        if not df.empty:
            df["img_res"] = pd.to_numeric(df["img_res"], errors="coerce")
            df = df.dropna(subset=["img_res"]).sort_values("img_res")
            fig = plt.figure(figsize=(6.5, 4.2), dpi=200)
            plt.plot(df["img_res"], df["avg_latency_ms"], marker="o")
            plt.xlabel("Input resolution (pixels)")
            plt.ylabel("Avg latency (ms), batch=1")
            plt.title("Inference Latency vs Resolution (batch=1)")
            plt.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(FIG_DIR / "latency_vs_resolution.png", bbox_inches="tight")
            fig.savefig(FIG_DIR / "latency_vs_resolution.pdf", bbox_inches="tight")
            plt.close(fig)

    # Ablation outputs
    if ab_csv.exists():
        ab = pd.read_csv(ab_csv)
        ab_cols = [c for c in [
            "run_id","img_res","growth_rate","compression","depth",
            "accuracy","macro_precision","macro_recall","macro_f1","avg_latency_ms","fps"
        ] if c in ab.columns]
        ab_clean = ab[ab_cols].copy() if ab_cols else ab.copy()
        ab_clean.to_csv(TAB_DIR / "Ablation_clean.csv", index=False)
        ab_round = _round_df(ab_clean, 4)
        ab_round.to_csv(TAB_DIR / "Ablation_rounded.csv", index=False)
        make_table_tex(ab_round, TAB_DIR / "Ablation.tex",
                       caption="Ablation study", label="tab:ablation")

        if "macro_f1" in ab_clean.columns:
            df = ab_clean.dropna(subset=["macro_f1"]).copy()
            if not df.empty:
                fig = plt.figure(figsize=(9.0, 4.2), dpi=200)
                x = np.arange(len(df))
                plt.plot(x, df["macro_f1"], marker="o")
                plt.xlabel("Ablation setting index")
                plt.ylabel("Macro F1")
                plt.title("Ablation: Macro F1 across settings")
                plt.grid(True, alpha=0.25)
                fig.tight_layout()
                fig.savefig(FIG_DIR / "ablation_macroF1.png", bbox_inches="tight")
                fig.savefig(FIG_DIR / "ablation_macroF1.pdf", bbox_inches="tight")
                plt.close(fig)

    # Per-run confusion matrix figures
    processed = []
    run_dirs = [p for p in RUNS_DIR.iterdir() if p.is_dir() and p.name not in ["publishable"]]
    for run_dir in sorted(run_dirs):
        labels, cm = _load_confusion(run_dir)
        if labels is None:
            continue
        out_png = FIG_DIR / f"{run_dir.name}_confusion_matrix.png"
        out_pdf = FIG_DIR / f"{run_dir.name}_confusion_matrix.pdf"
        plot_confusion_matrix(labels, cm, f"Confusion Matrix: {run_dir.name}", out_png, out_pdf)
        processed.append({
            "run_id": run_dir.name,
            "confusion_png": str(out_png),
            "confusion_pdf": str(out_pdf)
        })

    pd.DataFrame(processed).to_csv(OUT_DIR / "index.csv", index=False)
    (OUT_DIR / "README.txt").write_text(
        "Publishable artifacts generated.\n"
        f"- Figures: {FIG_DIR}\n"
        f"- Tables : {TAB_DIR}\n"
        "Run `python3 generate_publishable_artifacts.py` from the repo root.\n"
    )

    print("Done.")
    print("Figures:", FIG_DIR)
    print("Tables :", TAB_DIR)
    print("Index  :", OUT_DIR / "index.csv")

if __name__ == "__main__":
    main()
