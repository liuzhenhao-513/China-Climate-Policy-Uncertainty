"""Validation tables and figures: reconstructed indices vs the published CCPU."""
import csv
import json
from pathlib import Path
from .common import ROOT, utc_now, write_csv, write_json
from .index import compare_series, load_series

RECON = ROOT / "data" / "reconstructed"
FINAL = ROOT / "data" / "published"
VARIANTS = {  # name -> description
    "ccpu_macbert": "MacBERT labels, paper formula (CPU share)",
    "ccpu_macbert_count": "MacBERT labels, fixed denominator (CPU count)",
    "ccpu_tfidf": "TF-IDF + LR labels, CPU share",
    "ccpu_tfidf_count": "TF-IDF + LR labels, CPU count",
    "ccpu_rule": "Dictionary rule, share",
    "ccpu_rule_count": "Dictionary rule, count",
    "ccpu_climate": "Any climate term, share (no classifier)",
}
MAIN = "ccpu_macbert_count"
EVENTS = [("2009-12", "Copenhagen COP15"), ("2015-12", "Paris Agreement"),
          ("2020-09", "Dual-carbon pledge"), ("2021-10", "Peaking & neutrality guidance")]
EVENTS_ALL = [("2007-06", "National Climate Change Programme"), ("2009-11", "China announces 2020 carbon-intensity target"),
              ("2009-12", "Copenhagen COP15"), ("2011-12", "Durban COP17"), ("2015-06", "China submits NDC"),
              ("2015-12", "Paris Agreement"), ("2017-06", "US withdrawal from Paris announced"),
              ("2017-12", "National ETS launched"), ("2020-09", "Dual-carbon pledge"),
              ("2021-07", "National ETS trading starts"), ("2021-10", "Peaking & neutrality guidance"),
              ("2021-11", "Glasgow COP26")]

# Reference palette (dataviz skill): categorical slots 1-2, ink and chrome
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _dates(keys):
    from datetime import date
    return [date(int(k[:4]), int(k[5:7]) if len(k) > 4 else 7, 1 if len(k) > 4 else 1) for k in keys]


def plot_comparison(recon_file, published_file, out, frequency="monthly", title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a, b = load_series(recon_file, frequency), load_series(published_file, frequency)
    keys = sorted(k for k in a.keys() & b.keys() if k <= "2022-12")
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    x = _dates(keys)
    ax.plot(x, [b[k] for k in keys], color=ORANGE, linewidth=1.4, label="Published (Ma et al., 2023)")
    ax.plot(x, [a[k] for k in keys], color=BLUE, linewidth=1.4, label="Reconstructed (this project)")
    if frequency == "monthly":
        top = max(max(a[k] for k in keys), max(b[k] for k in keys))
        for i, (month, label) in enumerate(EVENTS):
            if month in a:
                d = _dates([month])[0]
                ax.axvline(d, color=AXIS, linewidth=0.8, linestyle=(0, (3, 3)), zorder=0)
                ax.text(d, top * (1.0 + 0.07 * (i % 2)), " " + label + " ", fontsize=7.5, color=INK2,
                        ha="right" if i % 2 == 0 else "left", va="bottom")
        ax.set_ylim(top=top * 1.16)
    ax.set_ylabel("CCPU (unit-SD scaled)", color=INK2, fontsize=9)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def plot_metrics(metrics, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(metrics)
    measures = ["accuracy", "precision", "recall", "f1"]
    fig, axes = plt.subplots(1, len(measures), figsize=(10, 2.8), dpi=150, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, m in zip(axes, measures):
        _style(ax)
        values = [metrics[n][m] for n in names]
        bars = ax.barh(names, values, color=BLUE, height=0.55)
        for bar, v in zip(bars, values):
            ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center", fontsize=8, color=INK2)
        ax.set_xlim(0, 1.15)
        ax.set_title(m.capitalize() if m != "f1" else "F1", fontsize=10, color=INK, loc="left")
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.grid(axis="y", visible=False)
        ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def event_check(path, start="2000-01", end="2022-12", top=10):
    """Where do landmark climate-policy months rank? (percentile within the sample)"""
    s = {k: v for k, v in load_series(path, "monthly").items() if start <= k <= end}
    ordered = sorted(s, key=s.get, reverse=True)
    rank = {k: i + 1 for i, k in enumerate(ordered)}
    return {"months": len(s),
            "events": [{"month": m, "event": e, "rank": rank.get(m), "percentile": round(1 - (rank[m] - 1) / len(s), 3)}
                       for m, e in EVENTS_ALL if m in s],
            "top_months": [{"month": k, "ccpu": round(s[k], 3)} for k in ordered[:top]]}


def build_report(cfg, out):
    out = Path(out)
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, description in VARIANTS.items():
        for frequency in ("monthly", "annual", "daily"):
            recon = RECON / f"{name}_{frequency}.csv"
            if not recon.exists():
                continue
            published = FINAL / f"national_{frequency}.csv"
            m = compare_series(recon, published, None, frequency, cfg["paper_start"], cfg["paper_end"])
            rows.append({"variant": name, "description": description, "frequency": frequency,
                         "overlap": m["overlap"], "pearson_r": round(m["pearson_r"], 4),
                         "spearman_rho": round(m["spearman_rho"], 4), "mae": round(m["mae"], 4),
                         "rmse": round(m["rmse"], 4)})
    write_csv(out / "validation" / "index_comparison.csv", list(rows[0]), rows)
    for name, label in ((MAIN, "count"), ("ccpu_macbert", "share")):
        for frequency in ("monthly", "annual"):
            recon = RECON / f"{name}_{frequency}.csv"
            if recon.exists():
                plot_comparison(recon, FINAL / f"national_{frequency}.csv",
                                figures / f"ccpu_{label}_{frequency}_vs_published.png", frequency,
                                f"China CCPU, national {frequency}: MacBERT, CPU {label}, vs published")
    classifier = {}
    run = ROOT / "models" / "macbert-ccpu" / "training_run.json"
    base = ROOT / "models" / "baselines" / "baseline_metrics.json"
    if run.exists():
        classifier["MacBERT"] = json.loads(run.read_text(encoding="utf-8"))["test"]
    if base.exists():
        b = json.loads(base.read_text(encoding="utf-8"))
        classifier["TF-IDF + LR"] = b["tfidf_lr"]
        classifier["Dictionary rule"] = b["rule"]
    if classifier:
        plot_metrics(classifier, figures / "classifier_test_metrics.png")
        write_json(out / "validation" / "classifier_test_metrics.json", classifier)
    events = {name: event_check(RECON / f"{name}_monthly.csv") for name in VARIANTS
              if (RECON / f"{name}_monthly.csv").exists()}
    events["published"] = event_check(FINAL / "national_monthly.csv")
    write_json(out / "validation" / "event_check.json", events)
    summary = {"generated_at_utc": utc_now(), "index_comparison": rows, "classifier_test_metrics": classifier,
               "event_check": {k: v["events"] for k, v in events.items()}}
    write_json(out / "validation" / "summary.json", summary)
    return summary
