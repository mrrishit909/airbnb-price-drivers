"""Step 6: charts.   ./venv/bin/python charts.py -> charts/00_cover.png, 01_prices.png, 02_ladder.png, 03_effects.png, 04_map.png, 05_errors.png"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch, Polygon

HERE = Path(__file__).parent
R, D = HERE / "results", HERE / "data"
INK, DIM, GRID, BG, BLUE, ORANGE, GRAY, GREEN = "#f2f2f0", "#8a8a87", "#1d1d1d", "#0b0b0b", "#3987e5", "#d95926", "#9a9a96", "#3aa876"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": INK,
                     "axes.edgecolor": "#3a3a3a", "axes.labelcolor": DIM, "xtick.color": DIM, "ytick.color": DIM,
                     "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 1,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 11, "axes.titlesize": 13,
                     "axes.titlelocation": "left", "axes.titlepad": 12})
SC, GR, EF, LO, ER, PR, AUD = (pd.read_csv(R / f"{f}.csv") for f in ("scores", "groups", "effects", "location", "errors", "prices", "audit")); AUD = AUD.set_index("check")["value"]
G = json.load(open(D / "neighbourhoods.geojson")); MIN_ZIP = 30
PREM = LinearSegmentedColormap.from_list("premium", [BLUE, "#222222", ORANGE]); NORM = Normalize(-40, 40)
STEPS = [("no features (the mean)", "the\nmean"), ("quote", "the\nquote"), ("+ size", "+ size"), ("+ type", "+ type"), ("+ place", "+ place"), ("+ host and reviews", "+ host,\nreviews"), ("+ amenities", "+ amenities")]


def save(fig, name):
    fig.tight_layout(); fig.savefig(HERE / "charts" / name, dpi=150); plt.close(fig)


def host_row(model):
    return SC[(SC["model"] == model) & (SC["folds"] == "by host")].iloc[0]


def zip_map(ax, labels=True):
    big = LO[LO["listings"] >= MIN_ZIP].set_index("zip")["premium_pct"]; pos = {}
    for f in G["features"]:
        z = str(f["properties"]["neighbourhood"]); g = f["geometry"]; parts = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for part in parts:
            ring = np.array(part[0]); col = PREM(NORM(big[int(z)])) if int(z) in big.index else BG
            ax.add_patch(Polygon(ring, closed=True, facecolor=col, edgecolor="#4a4a48", linewidth=0.6)); pos.setdefault(z, []).append(ring)
    allp = np.vstack([r for rs in pos.values() for r in rs]); ax.set_xlim(allp[:, 0].min() - 0.01, allp[:, 0].max() + 0.01); ax.set_ylim(allp[:, 1].min() - 0.01, allp[:, 1].max() + 0.01)
    ax.set_aspect(1 / np.cos(np.radians(30.27))); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    if labels:
        top = list(big.sort_values().index[-3:]) + list(big.sort_values().index[:3])
        for z in top:
            ring = max(pos[str(z)], key=len); c = (ring.min(axis=0) + ring.max(axis=0)) / 2; ax.text(c[0], c[1], str(z), ha="center", va="center", fontsize=8, color=INK, fontweight="bold")


def ladder(ax, size=10, horizontal=False):
    x = np.arange(len(STEPS) + 1); v = [host_row(m)["r2"] for m, _ in STEPS] + [host_row("gradient boosting")["r2"]]
    if horizontal:
        y = x[::-1]; ax.barh(y, [max(a, 0) for a in v], color=[BLUE] * len(STEPS) + [ORANGE], height=0.66)
        for yi, vi in zip(y, v):
            ax.text(max(vi, 0) + 0.01, yi, f"{max(vi, 0):.2f}", va="center", fontsize=size, color=INK)
        ax.set_yticks(y); ax.set_yticklabels([l.replace("\n", " ") for _, l in STEPS] + ["boosted trees"], fontsize=9.5); ax.set_xlim(0, 0.8); ax.set_xlabel("out-of-sample R² of the log price"); ax.grid(axis="y", visible=False); return
    ax.bar(x, [max(a, 0) for a in v], color=[BLUE] * len(STEPS) + [ORANGE], width=0.66)
    for xi, vi in zip(x, v):
        ax.text(xi, max(vi, 0) + 0.015, f"{max(vi, 0):.2f}", ha="center", fontsize=size, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in STEPS] + ["boosted\ntrees"], fontsize=9); ax.set_ylim(0, 0.8); ax.set_ylabel("out-of-sample R² of the log price"); ax.grid(axis="x", visible=False)


def groups_chart(ax):
    names = {"quote": "the quote", "size": "size", "type": "type", "place": "place (zip code)", "host and reviews": "host and reviews", "amenities": "amenities"}
    y = np.arange(len(GR))[::-1]; h = 0.26
    for dy, col, lab, c in ((h, BLUE, "alone", "alone_r2"), (0, GREEN, "added in this order", "added_r2"), (-h, ORANGE, "lost if removed", "lost_if_removed_r2")):
        ax.barh(y + dy, GR[c].clip(lower=0), height=h * 0.92, color=col, label=lab)
        for yi, v in zip(y + dy, GR[c]):
            ax.text(max(v, 0) + 0.008, yi, f"{max(v, 0):.2f}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([names[g] for g in GR["group"]]); ax.set_xlim(0, 0.62); ax.set_xlabel("out-of-sample R²"); ax.grid(axis="y", visible=False); ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=3)


def effects(ax):
    t = EF.sort_values("effect_pct").reset_index(drop=True); y = np.arange(len(t))
    ax.barh(y, t["effect_pct"], color=[ORANGE if v > 0 else BLUE for v in t["effect_pct"]], height=0.62)
    ax.hlines(y, t["ci_low_pct"], t["ci_high_pct"], color=INK, alpha=0.65, linewidth=1.3)
    for yi, v, hi, lo in zip(y, t["effect_pct"], t["ci_high_pct"], t["ci_low_pct"]):
        ax.text((hi if v > 0 else lo) + (1.2 if v > 0 else -1.2), yi, f"{v:+.0f}%".replace("-", "−"), va="center", ha="left" if v > 0 else "right", fontsize=9, color=INK)
    ax.axvline(0, color=DIM, linewidth=1); ax.set_yticks(y); ax.set_yticklabels(t["label"], fontsize=9.5); ax.set_xlim(-45, 62); ax.set_xlabel("change in the quoted price, other columns held fixed (line: 95% interval)"); ax.grid(axis="y", visible=False)


def prices(ax):
    L = pd.read_csv(D / "listings.csv.gz", usecols=["price", "room_type", "minimum_nights", "price_quote_checkin_date", "price_quote_checkout_date"])
    p = pd.to_numeric(L["price"].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce"); nn = (pd.to_datetime(L["price_quote_checkout_date"]) - pd.to_datetime(L["price_quote_checkin_date"])).dt.days
    m = p.notna() & (L["minimum_nights"] < 30) & nn.between(1, 7); p, rt = p[m], L["room_type"][m]; bins = np.logspace(np.log10(25), np.log10(60000), 46)
    ax.hist([p[rt == "Entire home/apt"], p[rt == "Private room"], p[rt.isin(["Hotel room", "Shared room"])]], bins=bins, stacked=True, color=[ORANGE, BLUE, GRAY], label=["entire home or apartment", "private room", "hotel or shared room"])
    ax.set_xscale("log"); ax.set_xticks([25, 50, 100, 250, 500, 1000, 2500, 10000, 50000]); ax.set_xticklabels([f"\\${t:,}" for t in (25, 50, 100, 250, 500, 1000, 2500, 10000, 50000)]); ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.axvline(p.median(), color=INK, linestyle="--", linewidth=1); ax.axvline(p.mean(), color=DIM, linestyle="--", linewidth=1)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15); top = ax.get_ylim()[1]; ax.text(p.median() * 0.93, top * 0.93, f"median \\${p.median():,.0f}", ha="right", fontsize=9, color=INK); ax.text(p.mean() * 1.07, top * 0.93, f"mean \\${p.mean():,.0f}", ha="left", fontsize=9, color=DIM)
    ax.set_xlabel("price quoted for a night (log scale)"); ax.set_ylabel("listings"); ax.grid(axis="x", visible=False); ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="upper right", bbox_to_anchor=(1.0, 0.82)); ax.set_title(f"What {int(AUD['priced, short minimum stay, quote of 1 to 7 nights (the sample)']):,} Austin listings were quoted for a night")


def errors(axes):
    x = np.arange(5); w = 0.38
    for dx, mdl, col in ((-w / 2, "least squares", BLUE), (w / 2, "gradient boosting", ORANGE)):
        t = ER[ER["model"] == mdl].sort_values("price_fifth"); axes[0].bar(x + dx, 100 * t["mdape"], width=w * 0.92, color=col, label=mdl)
        for xi, v in zip(x + dx, 100 * t["mdape"]):
            axes[0].text(xi, v + 0.7, f"{v:.0f}%", ha="center", fontsize=8.5, color=INK)
        axes[1].plot(x, t["median_fit_over_price"], marker="o", color=col, linewidth=1.8, label=mdl)
    lab = [f"{int(r.price_fifth)}\n\\${r.price_min:,.0f} to \\${r.price_max:,.0f}" for r in ER[ER["model"] == "least squares"].sort_values("price_fifth").itertuples()]
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels([l.replace("to \\$58,601", "up") for l in lab], fontsize=8); ax.grid(axis="x", visible=False); ax.set_xlabel("fifth of listings by actual price (1 = cheapest)")
    axes[0].set_ylim(0, 40); axes[0].set_ylabel("median error of the price (%)"); axes[0].set_title("How far off the typical prediction is", fontsize=12); axes[0].legend(frameon=False, labelcolor=INK, fontsize=8.5, loc="upper left")
    axes[1].axhline(1, color=DIM, linestyle="--", linewidth=1); axes[1].set_ylim(0.6, 1.4); axes[1].set_ylabel("median of predicted price ÷ actual price"); axes[1].set_title("Cheap listings are overpriced, dear ones underpriced", fontsize=12)


def main():
    (HERE / "charts").mkdir(exist_ok=True)
    gb, sz = host_row("gradient boosting"), host_row("+ size"); fig = plt.figure(figsize=(14, 8)); g = fig.add_gridspec(1, 2, width_ratios=[1, 1.15]); a = fig.add_subplot(g[0]); b = fig.add_subplot(g[1])
    zip_map(a); a.set_title("Listings quoted above or below what their size and type suggest, by zip code", fontsize=12)
    cax = fig.add_axes([0.07, 0.07, 0.28, 0.018]); cb = fig.colorbar(ScalarMappable(norm=NORM, cmap=PREM), cax=cax, orientation="horizontal"); cb.set_ticks([-40, -20, 0, 20, 40]); cb.set_ticklabels(["−40%", "−20%", "0", "+20%", "+40%"]); cb.outline.set_visible(False); cax.tick_params(colors=DIM, labelsize=9)
    ladder(b, 11); b.set_title("How much of the price each step explains (out of sample)", fontsize=14)
    fig.suptitle(f"What makes an Austin Airbnb expensive? Its size first, then where it is; even the best model misses by {100 * gb['mdape']:.0f}%", x=0.01, ha="left", fontsize=20, color=INK, y=0.98)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94)); fig.savefig(HERE / "charts" / "00_cover.png", dpi=100); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 3.8)); prices(ax); save(fig, "01_prices.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), gridspec_kw={"width_ratios": [1.05, 1]}); ladder(axes[0], 9, horizontal=True); axes[0].set_title("R² as groups of columns are added", fontsize=12); groups_chart(axes[1]); axes[1].set_title("What each group explains", fontsize=12); save(fig, "02_ladder.png")
    fig, ax = plt.subplots(figsize=(10, 4.6)); effects(ax); ax.set_title("What a feature goes with, in price, after the others are accounted for"); save(fig, "03_effects.png")
    fig = plt.figure(figsize=(10, 4.8)); g = fig.add_gridspec(1, 2, width_ratios=[1, 1.05]); ax = fig.add_subplot(g[0]); bx = fig.add_subplot(g[1])
    zip_map(ax); ax.set_title("Zip codes (outline only: fewer than 30 listings)", fontsize=12)
    cax = ax.inset_axes([0.1, -0.02, 0.8, 0.03]); cb = fig.colorbar(ScalarMappable(norm=NORM, cmap=PREM), cax=cax, orientation="horizontal"); cb.set_ticks([-40, -20, 0, 20, 40]); cb.set_ticklabels(["\u221240%", "\u221220%", "0", "+20%", "+40%"]); cb.outline.set_visible(False); cax.tick_params(colors=DIM, labelsize=8)
    big = LO[LO["listings"] >= MIN_ZIP].sort_values("premium_pct"); pick = pd.concat([big.head(8), big.tail(8)]); y = np.arange(len(pick))
    bx.barh(y, pick["premium_pct"], color=[PREM(NORM(v)) if abs(v) < 38 else (ORANGE if v > 0 else BLUE) for v in pick["premium_pct"]], height=0.7)
    for yi, v in zip(y, pick["premium_pct"]):
        bx.text(v + (1 if v > 0 else -1), yi, f"{v:+.0f}%".replace("-", "\u2212"), va="center", ha="left" if v > 0 else "right", fontsize=8.5, color=INK)
    bx.axvline(0, color=DIM, linewidth=1); bx.set_yticks(y); bx.set_yticklabels([f"{z}  ({k})" for z, k in zip(pick["zip"], pick["listings"])], fontsize=8.5); bx.set_xlim(-55, 62); bx.grid(axis="y", visible=False)
    bx.set_xlabel("median listing vs what its size and type suggest (zip, listings)"); bx.set_title("The eight lowest and eight highest zip codes", fontsize=12); save(fig, "04_map.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9)); errors(axes); save(fig, "05_errors.png")


if __name__ == "__main__":
    main()
