"""
Visualización de la evaluación del agente PPO de orientación de fibras musculares.

Uso:
    python plot_evaluation.py                          # usa results/eval_v2.csv por defecto
    python plot_evaluation.py --csv results/eval.csv
    python plot_evaluation.py --csv results/eval_v2.csv --compare results/eval.csv
    python plot_evaluation.py --save results/eval_plots.png
"""

import argparse
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Umbrales de aceptación definidos en HU5 ──────────────────────────────────
THR_PROTOTYPE = 10.0   # MAE objetivo prototipo
THR_PRODUCTION = 5.0   # MAE objetivo producción


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert {"theta_true", "theta_predicted", "error_deg"}.issubset(df.columns), (
        f"{path} debe tener columnas: theta_true, theta_predicted, error_deg"
    )
    return df


def compute_metrics(df: pd.DataFrame) -> dict:
    errors = df["error_deg"].values
    return {
        "mae": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "std": float(np.std(errors)),
        "pct_lt5": float(np.mean(errors < 5.0) * 100),
        "pct_lt10": float(np.mean(errors < 10.0) * 100),
        "max_error": float(np.max(errors)),
    }


def _label(name: str, metrics: dict) -> str:
    return (
        f"{name}\n"
        f"MAE={metrics['mae']:.2f}°  "
        f"<5°={metrics['pct_lt5']:.1f}%  "
        f"<10°={metrics['pct_lt10']:.1f}%"
    )


def plot_evaluation(
    df_main: pd.DataFrame,
    label_main: str,
    df_ref: pd.DataFrame | None = None,
    label_ref: str | None = None,
    save_path: str | None = None,
) -> None:
    m = compute_metrics(df_main)
    m_ref = compute_metrics(df_ref) if df_ref is not None else None

    fig = plt.figure(figsize=(18, 13))
    fig.suptitle(
        "Evaluación del agente PPO — Orientación de Fibras Musculares",
        fontsize=14, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(
        3, 3, figure=fig,
        hspace=0.45, wspace=0.35,
        left=0.07, right=0.97, top=0.93, bottom=0.07,
    )

    # ── 1. Scatter: theta_true vs theta_predicted ─────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    _scatter_plot(ax1, df_main, label_main, df_ref, label_ref)

    # ── 2. Tabla de métricas ──────────────────────────────────────────────────
    ax_table = fig.add_subplot(gs[0, 2])
    _metrics_table(ax_table, m, label_main, m_ref, label_ref)

    # ── 3. Error vs theta_true ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :2])
    _error_by_angle(ax2, df_main, label_main, df_ref, label_ref)

    # ── 4. Distribución acumulada del error ───────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 2])
    _cdf_plot(ax3, df_main, label_main, df_ref, label_ref)

    # ── 5. Histograma de errores ──────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    _histogram(ax4, df_main, label_main, df_ref, label_ref)

    # ── 6. Diagrama polar de errores ──────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1], projection="polar")
    _polar_error(ax5, df_main, label_main)

    # ── 7. Diagrama polar comparativo (si hay referencia) ────────────────────
    if df_ref is not None:
        ax6 = fig.add_subplot(gs[2, 2], projection="polar")
        _polar_error(ax6, df_ref, label_ref)
    else:
        ax6 = fig.add_subplot(gs[2, 2])
        _error_boxplot(ax6, df_main, label_main)

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Gráfico guardado en: {save_path}")
    else:
        plt.show()


# ── Subplots ──────────────────────────────────────────────────────────────────

def _scatter_plot(ax, df_main, label_main, df_ref, label_ref):
    ax.set_title("Ángulo predicho vs. ángulo real", fontsize=11)

    # Línea ideal y márgenes de error
    theta_range = np.linspace(0, 180, 300)
    ax.plot(theta_range, theta_range, "k--", lw=1.2, alpha=0.5, label="Ideal (error=0°)")
    ax.fill_between(theta_range, theta_range - 5, theta_range + 5,
                    alpha=0.08, color="green", label="±5° (producción)")
    ax.fill_between(theta_range, theta_range - 10, theta_range + 10,
                    alpha=0.08, color="orange", label="±10° (prototipo)")

    if df_ref is not None:
        ax.scatter(df_ref["theta_true"], df_ref["theta_predicted"],
                   s=18, alpha=0.4, color="gray", label=f"[ref] {label_ref}", zorder=2)

    # Colorear puntos según error
    sc = ax.scatter(
        df_main["theta_true"], df_main["theta_predicted"],
        c=df_main["error_deg"], cmap="RdYlGn_r", vmin=0, vmax=20,
        s=28, alpha=0.85, zorder=3, label=label_main,
    )
    plt.colorbar(sc, ax=ax, label="Error (°)", fraction=0.025)

    ax.set_xlabel("θ real (°)")
    ax.set_ylabel("θ predicho (°)")
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 185)
    ax.set_xticks(range(0, 181, 30))
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)


def _error_by_angle(ax, df_main, label_main, df_ref, label_ref):
    ax.set_title("Error angular por ángulo real", fontsize=11)

    if df_ref is not None:
        ax.plot(df_ref["theta_true"], df_ref["error_deg"],
                "o-", ms=3, lw=0.8, alpha=0.4, color="gray",
                label=f"[ref] {label_ref}")

    ax.plot(df_main["theta_true"], df_main["error_deg"],
            "o-", ms=4, lw=1.0, color="steelblue", label=label_main)

    ax.axhline(THR_PRODUCTION, color="green", lw=1.5, ls="--",
               label=f"Umbral producción ({THR_PRODUCTION}°)")
    ax.axhline(THR_PROTOTYPE, color="orange", lw=1.5, ls="--",
               label=f"Umbral prototipo ({THR_PROTOTYPE}°)")

    ax.set_xlabel("θ real (°)")
    ax.set_ylabel("Error angular (°)")
    ax.set_xlim(0, 180)
    ax.set_xticks(range(0, 181, 30))
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def _cdf_plot(ax, df_main, label_main, df_ref, label_ref):
    ax.set_title("CDF del error angular", fontsize=11)

    def _cdf(errors):
        s = np.sort(errors)
        return s, np.arange(1, len(s) + 1) / len(s) * 100

    if df_ref is not None:
        x, y = _cdf(df_ref["error_deg"].values)
        ax.plot(x, y, color="gray", lw=1.5, alpha=0.6, label=f"[ref] {label_ref}")

    x, y = _cdf(df_main["error_deg"].values)
    ax.plot(x, y, color="steelblue", lw=2, label=label_main)

    ax.axvline(THR_PRODUCTION, color="green", lw=1.5, ls="--",
               label=f"{THR_PRODUCTION}° (prod)")
    ax.axvline(THR_PROTOTYPE, color="orange", lw=1.5, ls="--",
               label=f"{THR_PROTOTYPE}° (proto)")
    ax.axhline(80, color="purple", lw=1.0, ls=":", alpha=0.6, label="80%")

    ax.set_xlabel("Error angular (°)")
    ax.set_ylabel("% acumulado")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def _histogram(ax, df_main, label_main, df_ref, label_ref):
    ax.set_title("Distribución del error angular", fontsize=11)
    bins = np.arange(0, df_main["error_deg"].max() + 2.5, 2.5)

    if df_ref is not None:
        ax.hist(df_ref["error_deg"], bins=bins, alpha=0.4, color="gray",
                label=f"[ref] {label_ref}", edgecolor="white", linewidth=0.3)

    ax.hist(df_main["error_deg"], bins=bins, alpha=0.8, color="steelblue",
            label=label_main, edgecolor="white", linewidth=0.3)

    m = compute_metrics(df_main)
    ax.axvline(m["mae"], color="red", lw=1.8, ls="-",
               label=f"MAE = {m['mae']:.2f}°")
    ax.axvline(m["median"], color="orange", lw=1.8, ls="--",
               label=f"Mediana = {m['median']:.2f}°")
    ax.axvline(THR_PRODUCTION, color="green", lw=1.5, ls=":",
               label=f"{THR_PRODUCTION}° (prod)")

    ax.set_xlabel("Error angular (°)")
    ax.set_ylabel("Frecuencia")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")


def _polar_error(ax, df, label):
    """Diagrama polar: cada rayo es un ángulo real, longitud = error."""
    short = label.split("\n")[0][:20]
    ax.set_title(f"Error polar\n{short}", fontsize=9, pad=10)

    thetas_rad = np.deg2rad(df["theta_true"].values)
    errors = df["error_deg"].values

    norm = plt.Normalize(vmin=0, vmax=20)
    cmap = plt.cm.RdYlGn_r

    for t, e in zip(thetas_rad, errors):
        ax.plot([t, t], [0, e], color=cmap(norm(e)), lw=1.5, alpha=0.75)
        ax.scatter([t], [e], color=cmap(norm(e)), s=18, zorder=3)

    # Umbral de producción
    theta_circle = np.linspace(0, 2 * np.pi, 300)
    ax.plot(theta_circle, [THR_PRODUCTION] * 300, "g--", lw=1.2, alpha=0.6)
    ax.plot(theta_circle, [THR_PROTOTYPE] * 300, color="orange", lw=1.2, ls=":", alpha=0.6)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(45)
    ax.tick_params(labelsize=7)


def _error_boxplot(ax, df, label):
    """Boxplot del error por cuartil de ángulo."""
    ax.set_title("Error por cuadrante angular", fontsize=11)

    bins = [0, 45, 90, 135, 180]
    labels = ["0–45°", "45–90°", "90–135°", "135–180°"]
    groups = [
        df[(df["theta_true"] >= bins[i]) & (df["theta_true"] < bins[i + 1])]["error_deg"].values
        for i in range(len(labels))
    ]

    bp = ax.boxplot(groups, labels=labels, patch_artist=True, notch=False)
    colors = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(THR_PRODUCTION, color="green", lw=1.5, ls="--", label=f"{THR_PRODUCTION}° (prod)")
    ax.axhline(THR_PROTOTYPE, color="orange", lw=1.5, ls=":", label=f"{THR_PROTOTYPE}° (proto)")
    ax.set_ylabel("Error angular (°)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")


def _metrics_table(ax, m, label_main, m_ref, label_ref):
    ax.axis("off")
    ax.set_title("Métricas de evaluación", fontsize=11, pad=10)

    def status(value, threshold, lower_is_better=True):
        ok = value < threshold if lower_is_better else value > threshold
        return "✓" if ok else "✗"

    rows = [
        ["Métrica", label_main[:18], "Umbral", "Estado"],
        ["MAE (°)", f"{m['mae']:.2f}", f"<{THR_PROTOTYPE}°",
         status(m["mae"], THR_PROTOTYPE)],
        ["Mediana (°)", f"{m['median']:.2f}", "—", "—"],
        ["Std (°)", f"{m['std']:.2f}", "—", "—"],
        ["% error <5°", f"{m['pct_lt5']:.1f}%", f">{100 - THR_PRODUCTION*10:.0f}%",
         status(m["pct_lt5"], 50.0, lower_is_better=False)],
        ["% error <10°", f"{m['pct_lt10']:.1f}%", ">80%",
         status(m["pct_lt10"], 80.0, lower_is_better=False)],
        ["Error máx (°)", f"{m['max_error']:.2f}", "—", "—"],
    ]

    if m_ref is not None:
        rows[0].insert(2, (label_ref or "ref")[:18])
        rows[1].insert(2, f"{m_ref['mae']:.2f}")
        rows[2].insert(2, f"{m_ref['median']:.2f}")
        rows[3].insert(2, f"{m_ref['std']:.2f}")
        rows[4].insert(2, f"{m_ref['pct_lt5']:.1f}%")
        rows[5].insert(2, f"{m_ref['pct_lt10']:.1f}%")
        rows[6].insert(2, f"{m_ref['max_error']:.2f}")

    col_widths = [0.35, 0.20, 0.20, 0.12] if m_ref is None else [0.30, 0.18, 0.18, 0.12, 0.12]
    table = ax.table(
        cellText=rows[1:],
        colLabels=rows[0],
        cellLoc="center",
        loc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.6)

    # Colorear encabezados
    for j in range(len(rows[0])):
        table[0, j].set_facecolor("#2c5f8a")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Colorear estado
    col_estado = len(rows[0]) - 1
    for i in range(1, len(rows)):
        cell = table[i, col_estado]
        text = cell.get_text().get_text()
        if text == "✓":
            cell.set_facecolor("#d4edda")
        elif text == "✗":
            cell.set_facecolor("#f8d7da")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualización de evaluación del agente PPO")
    parser.add_argument(
        "--csv", default="results/eval_v2.csv",
        help="CSV principal (default: results/eval_v2.csv)",
    )
    parser.add_argument(
        "--compare", default=None,
        help="CSV de referencia para comparar (opcional)",
    )
    parser.add_argument(
        "--save", default=None,
        help="Ruta para guardar la figura (PNG/PDF). Si no se especifica, muestra en pantalla.",
    )
    args = parser.parse_args()

    df_main = load_csv(args.csv)
    label_main = os.path.splitext(os.path.basename(args.csv))[0]

    df_ref, label_ref = None, None
    if args.compare:
        df_ref = load_csv(args.compare)
        label_ref = os.path.splitext(os.path.basename(args.compare))[0]

    m = compute_metrics(df_main)
    print("\n=== Resultados de evaluación ===")
    print(f"  Archivo       : {args.csv}")
    print(f"  N muestras    : {len(df_main)}")
    print(f"  MAE angular   : {m['mae']:.2f}°  (umbral prototipo: <{THR_PROTOTYPE}°)")
    print(f"  Mediana       : {m['median']:.2f}°")
    print(f"  Std           : {m['std']:.2f}°")
    print(f"  Error < 5°    : {m['pct_lt5']:.1f}%")
    print(f"  Error < 10°   : {m['pct_lt10']:.1f}%")
    print(f"  Error máximo  : {m['max_error']:.2f}°")
    print(f"  Estado HU5    : {'✓ MAE < 10° (prototipo OK)' if m['mae'] < THR_PROTOTYPE else '✗ MAE >= 10° (prototipo NO OK)'}")
    print()

    plot_evaluation(df_main, label_main, df_ref, label_ref, save_path=args.save)


if __name__ == "__main__":
    main()
