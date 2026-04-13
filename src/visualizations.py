"""
src/visualizations.py
=====================
Módulo de visualizações do pipeline LSTM-B3.

Gera todos os gráficos da dissertação:
    1. Série de preço + RSI e RSI-LPF (Seção 2.5 / 2.6)
    2. Curva de capital vs benchmark buy-and-hold (Seção 3.4)
    3. Curva A-R (Acurácia × Rejeição) — Seção 2.10.3
    4. Distribuição de retornos da estratégia vs benchmark
    5. Drawdown temporal
    6. Desempenho por janela walk-forward
    7. Distribuição de decisões (compra/venda/rejeição) + confiança

Referências:
    Seção 3.5.2 e Figura 4 da dissertação
    Rocha-Neto (2011) — Curvas A-R
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from typing import Optional
import warnings

# Estilo acadêmico sóbrio
plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.labelsize":     11,
    "legend.fontsize":    10,
    "figure.dpi":         150,
    "axes.grid":          True,
    "grid.alpha":         0.35,
    "grid.linestyle":     "--",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

COLORS = {
    "strategy":  "#1565C0",   # azul escuro
    "benchmark": "#B71C1C",   # vermelho
    "rsi":       "#F57F17",   # âmbar
    "rsi_lpf":   "#1B5E20",   # verde escuro
    "buy":       "#2E7D32",   # verde
    "sell":      "#C62828",   # vermelho
    "reject":    "#616161",   # cinza
    "drawdown":  "#880E4F",   # roxo escuro
}


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# 1. Preço + RSI + RSI-LPF
# ---------------------------------------------------------------------------

def plot_price_rsi(
    df: pd.DataFrame,
    features_df: pd.DataFrame,
    save_path: Optional[str] = None,
    asset: str = "BOVA11",
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Painel triplo:
        (a) Preço de fechamento
        (b) RSI14 com zonas 30/70
        (c) RSI-LPF com linha de regime (50)
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    ticker = asset.upper()
    if title is None:
        title = f"{ticker} — Preço, RSI e RSI-LPF"
    fig.suptitle(title, fontweight="bold")

    # (a) Preço
    axes[0].plot(df.index, df["close"], color=COLORS["strategy"], lw=1.2, label="Close")
    axes[0].set_ylabel("Preço (R$)")
    axes[0].legend(loc="upper left")

    # (b) RSI14
    if "rsi14" in features_df.columns:
        ax_rsi = axes[1]
        ax_rsi.plot(features_df.index, features_df["rsi14"], color=COLORS["rsi"], lw=1.0, label="RSI14")
        ax_rsi.axhline(70, ls="--", color="red",   alpha=0.6, lw=1.0, label="Sobrecompra (70)")
        ax_rsi.axhline(30, ls="--", color="green", alpha=0.6, lw=1.0, label="Sobrevenda (30)")
        ax_rsi.axhline(50, ls=":",  color="gray",  alpha=0.8, lw=0.8)
        ax_rsi.fill_between(features_df.index, 70, 100, alpha=0.08, color="red")
        ax_rsi.fill_between(features_df.index, 0,  30,  alpha=0.08, color="green")
        ax_rsi.set_ylim(0, 100)
        ax_rsi.set_ylabel("RSI14")
        ax_rsi.legend(loc="upper left", fontsize=9)

    # (c) RSI-LPF
    if "rsi_lpf" in features_df.columns:
        ax_lpf = axes[2]
        ax_lpf.plot(features_df.index, features_df["rsi_lpf"],
                    color=COLORS["rsi_lpf"], lw=1.4, label="RSI-LPF (p=5)")
        ax_lpf.axhline(50, ls="--", color="gray", lw=1.2, label="Regime 50")
        ax_lpf.fill_between(features_df.index, 50, features_df["rsi_lpf"],
                             where=features_df["rsi_lpf"] >= 50, alpha=0.15, color="green",
                             label="Alta")
        ax_lpf.fill_between(features_df.index, features_df["rsi_lpf"], 50,
                             where=features_df["rsi_lpf"] < 50, alpha=0.15, color="red",
                             label="Baixa")
        ax_lpf.set_ylim(0, 100)
        ax_lpf.set_ylabel("RSI-LPF")
        ax_lpf.legend(loc="upper left", fontsize=9)

    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 2. Curva de capital vs benchmark
# ---------------------------------------------------------------------------

def plot_equity_curve(
    equity_curve: pd.Series,
    benchmark_curve: pd.Series,
    save_path: Optional[str] = None,
    asset: str = "BOVA11",
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plota curva de capital e drawdown comparativos: estratégia vs Buy & Hold.

    Layout (3 painéis):
        [0] Curva de capital — R$ ao longo do tempo
        [1] Drawdown (%) — ambas as curvas sobrepostas
        [2] Barra de exposição — dias comprado vs caixa (quando disponível)
    """
    def _drawdown(series: pd.Series) -> np.ndarray:
        vals = series.values.astype(float)
        peak = np.maximum.accumulate(vals)
        return (vals - peak) / np.where(peak > 0, peak, 1.0) * 100.0

    ticker = asset.upper()
    if title is None:
        title = f"Curva de Capital — Estratégia LSTM vs Buy & Hold ({ticker})"

    fig, axes = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True,
        gridspec_kw={"height_ratios": [3, 2]},
    )
    fig.suptitle(title, fontweight="bold")

    # ── Painel 0: Curva de capital ──────────────────────────────────────────
    ax1 = axes[0]

    ret_s  = (equity_curve.iloc[-1]   / equity_curve.iloc[0]   - 1) * 100
    ret_bh = (benchmark_curve.iloc[-1] / benchmark_curve.iloc[0] - 1) * 100

    ax1.plot(
        equity_curve.index, equity_curve.values,
        color=COLORS["strategy"], lw=1.8,
        label=f"Estratégia ({ret_s:+.1f}%  |  R${equity_curve.iloc[-1]:,.0f})",
    )
    ax1.plot(
        benchmark_curve.index, benchmark_curve.values,
        color=COLORS["benchmark"], lw=1.8, ls="--",
        label=f"Buy & Hold {ticker} ({ret_bh:+.1f}%  |  R${benchmark_curve.iloc[-1]:,.0f})",
    )
    ax1.fill_between(
        equity_curve.index, equity_curve.values, benchmark_curve.values,
        where=(equity_curve.values >= benchmark_curve.values),
        color=COLORS["strategy"], alpha=0.08, label="Estratégia à frente",
    )
    ax1.fill_between(
        equity_curve.index, equity_curve.values, benchmark_curve.values,
        where=(equity_curve.values < benchmark_curve.values),
        color=COLORS["benchmark"], alpha=0.08, label="BH à frente",
    )
    ax1.set_ylabel("Capital (R$)")
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"R${x:,.0f}")
    )
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, lw=0.5)

    # ── Painel 1: Drawdown comparativo ─────────────────────────────────────
    ax2 = axes[1]

    dd_strat = _drawdown(equity_curve)
    dd_bh    = _drawdown(benchmark_curve)

    mdd_s  = dd_strat.min()
    mdd_bh = dd_bh.min()

    # BH em vermelho (perda maior), estratégia em azul/roxo
    ax2.fill_between(
        benchmark_curve.index, dd_bh, 0,
        color="#E24B4A", alpha=0.30,
        label=f"Drawdown BH (máx: {mdd_bh:.1f}%)",
    )
    ax2.fill_between(
        equity_curve.index, dd_strat, 0,
        color=COLORS["strategy"], alpha=0.55,
        label=f"Drawdown Estratégia (máx: {mdd_s:.1f}%)",
    )
    # Linhas sobre as áreas para melhor leitura
    ax2.plot(benchmark_curve.index, dd_bh,
             color="#E24B4A", lw=1.0, alpha=0.7)
    ax2.plot(equity_curve.index, dd_strat,
             color=COLORS["strategy"], lw=1.2)

    # Anota o pior ponto do BH
    idx_worst_bh = int(np.argmin(dd_bh))
    ax2.annotate(
        f"{mdd_bh:.1f}%",
        xy=(benchmark_curve.index[idx_worst_bh], mdd_bh),
        xytext=(10, -14), textcoords="offset points",
        fontsize=8, color="#A32D2D",
        arrowprops=dict(arrowstyle="->", color="#A32D2D", lw=0.8),
    )

    ax2.axhline(0, color="gray", lw=0.5, ls="--")
    ax2.set_ylabel("Drawdown (%)")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax2.legend(fontsize=9, loc="lower left")
    ax2.grid(True, alpha=0.3, lw=0.5)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    fig.autofmt_xdate(rotation=35, ha="right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 3. Curva A-R (Acurácia × Rejeição) — Seção 2.10.3
# ---------------------------------------------------------------------------

def plot_ar_curve(
    thresholds: np.ndarray,
    accuracies: np.ndarray,
    rejection_rates: np.ndarray,
    baseline_accuracy: float = 0.5,
    save_path: Optional[str] = None,
    asset: str = "BOVA11",
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plota a Curva A-R conforme Seção 2.10.3 da dissertação.

    Eixo X: taxa de rejeição (0% = nenhuma rejeição, 100% = tudo rejeitado)
    Eixo Y: acurácia nos exemplos aceitos

    Um modelo superior tem curva A-R consistentemente acima do baseline.

    Referência: Rocha-Neto (2011), Chow (1970)
    """
    ticker = asset.upper()
    if title is None:
        title = f"Curva Acurácia-Rejeição (A-R) — {ticker} — Modelo LSTM + Rejeição"
    valid = ~np.isnan(accuracies)

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(title, fontweight="bold")

    ax.plot(
        rejection_rates[valid] * 100,
        accuracies[valid] * 100,
        color=COLORS["strategy"],
        lw=2.0,
        marker="o",
        markersize=5,
        label="LSTM + Rejeição",
    )
    ax.axhline(
        baseline_accuracy * 100,
        ls="--",
        color="gray",
        lw=1.2,
        label=f"Baseline aleatório ({baseline_accuracy:.0%})",
    )

    # Anota τ ótimo (maior acurácia)
    if valid.sum() > 0:
        best_idx = np.nanargmax(accuracies[valid])
        best_rej = rejection_rates[valid][best_idx] * 100
        best_acc = accuracies[valid][best_idx] * 100
        best_tau = thresholds[valid][best_idx]
        ax.annotate(
            f"τ={best_tau:.2f}\n({best_acc:.1f}%, rej={best_rej:.1f}%)",
            xy=(best_rej, best_acc),
            xytext=(best_rej + 5, best_acc - 5),
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=9,
        )

    ax.set_xlabel("Taxa de Rejeição (%)")
    ax.set_ylabel("Acurácia nos Exemplos Aceitos (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(40, 100)
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 4. Distribuição de retornos
# ---------------------------------------------------------------------------

def plot_return_distribution(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    save_path: Optional[str] = None,
    asset: str = "BOVA11",
    title: Optional[str] = None,
) -> plt.Figure:
    """Histograma comparativo dos retornos diários (estratégia vs benchmark)."""
    ticker = asset.upper()
    if title is None:
        title = f"Distribuição de Retornos Diários — {ticker}"
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(title, fontweight="bold")

    ax.hist(strategy_returns * 100,  bins=60, alpha=0.65,
            color=COLORS["strategy"],  label="Estratégia LSTM", density=True)
    ax.hist(benchmark_returns * 100, bins=60, alpha=0.65,
            color=COLORS["benchmark"], label=f"Buy & Hold {ticker}", density=True)

    ax.axvline(strategy_returns.mean() * 100,  ls="--", color=COLORS["strategy"],  lw=1.5)
    ax.axvline(benchmark_returns.mean() * 100, ls="--", color=COLORS["benchmark"], lw=1.5)
    ax.set_xlabel("Retorno Diário (%)")
    ax.set_ylabel("Densidade")
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 5. Walk-forward — desempenho por janela
# ---------------------------------------------------------------------------

def plot_walk_forward_summary(
    results,    # list[WindowResult]
    save_path: Optional[str] = None,
    asset: str = "BOVA11",
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Barras de acurácia e taxa de rejeição para cada janela do walk-forward.
    """
    window_ids = [r.window_id    for r in results]
    accuracies = [r.accuracy_accepted if not np.isnan(r.accuracy_accepted) else 0.0
                  for r in results]
    rej_rates  = [r.rejection_rate for r in results]
    test_dates = [str(r.test_start.date()) for r in results]

    ticker = asset.upper()
    if title is None:
        title = f"Desempenho por Janela Walk-Forward — {ticker}"
    x = np.arange(len(window_ids))
    fig, ax = plt.subplots(figsize=(max(12, len(window_ids) * 0.6), 5))
    fig.suptitle(title, fontweight="bold")

    bars1 = ax.bar(x - 0.2, [a * 100 for a in accuracies], 0.35,
                   label="Acurácia (amostras aceitas, %)", color=COLORS["strategy"], alpha=0.8)
    bars2 = ax.bar(x + 0.2, [r * 100 for r in rej_rates],  0.35,
                   label="Taxa de Rejeição (%)", color=COLORS["reject"], alpha=0.8)

    ax.axhline(50, ls="--", color="red", lw=1.0, alpha=0.7, label="Baseline 50%")
    ax.set_xticks(x)
    ax.set_xticklabels([f"J{w}\n{d}" for w, d in zip(window_ids, test_dates)],
                       fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("(%)")
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 6. Sinais de compra/venda/rejeição no gráfico de preço
# ---------------------------------------------------------------------------

def plot_signals_on_price(
    price_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    save_path: Optional[str] = None,
    asset: str = "BOVA11",
    title: Optional[str] = None,
    n_days: int = 252,
) -> plt.Figure:
    """
    Plota os últimos n_days dias com sinais sobrepostos ao preço.
    Compra: triângulo verde ▲ | Venda: triângulo vermelho ▼ | Rejeição: ponto cinza
    """
    pred = predictions_df.copy()
    if len(pred) > n_days:
        pred = pred.iloc[-n_days:]

    prices = price_df["close"].reindex(pred.index).ffill()

    ticker = asset.upper()
    if title is None:
        title = f"Sinais de Negociação — {ticker} — LSTM + Rejeição (amostra)"
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(title, fontweight="bold")
    ax.plot(prices.index, prices.values, color="black", lw=1.0, alpha=0.7, label="Close")

    buys  = pred[pred["decision"] ==  1]
    sells = pred[pred["decision"] ==  0]
    holds = pred[pred["decision"] == -1]

    if not buys.empty:
        ax.scatter(buys.index,  prices.reindex(buys.index).values,
                   marker="^", color=COLORS["buy"],    s=60, zorder=5, label="Compra")
    if not sells.empty:
        ax.scatter(sells.index, prices.reindex(sells.index).values,
                   marker="v", color=COLORS["sell"],   s=60, zorder=5, label="Venda")
    if not holds.empty:
        ax.scatter(holds.index, prices.reindex(holds.index).values,
                   marker=".", color=COLORS["reject"],  s=20, zorder=4, alpha=0.5, label="Rejeição")

    ax.set_ylabel("Preço (R$)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig




# ---------------------------------------------------------------------------
# 8. Painel de Análise Técnica Comparativa
#    Médias Móveis (SMA/EMA) + Bandas de Bollinger + RSI + RSI-LPF
#    Apenas visualização — não alimenta o modelo
# ---------------------------------------------------------------------------

def plot_technical_analysis(
    df: pd.DataFrame,
    features_df: pd.DataFrame,
    asset: str = "BOVA11",
    n_days: int = 252,
    sma_periods: list = None,
    ema_periods: list = None,
    bb_period: int = 20,
    bb_std: float = 2.0,
    show_volume: bool = True,
) -> plt.Figure:
    """
    Painel de análise técnica para comparação visual — não afeta o modelo.

    Painéis:
        (a) Preço + MMs (SMA e EMA) + Bandas de Bollinger
        (b) Volume relativo (opcional)
        (c) RSI14 com zonas 30/70
        (d) RSI-LPF com regime 50

    Parâmetros
    ----------
    df          : DataFrame com colunas close, high, low, volume
    features_df : DataFrame com colunas rsi14, rsi_lpf (saída de build_feature_matrix)
    asset       : ticker do ativo
    n_days      : janela de visualização (padrão últimos 252 pregões)
    sma_periods : lista de períodos para SMA (padrão [20, 50, 200])
    ema_periods : lista de períodos para EMA (padrão [9, 21])
    bb_period   : período das Bandas de Bollinger (padrão 20)
    bb_std      : desvios-padrão das bandas (padrão 2.0)
    show_volume : exibe painel de volume (padrão True)
    """
    if sma_periods is None:
        sma_periods = [20, 50, 200]
    if ema_periods is None:
        ema_periods = [9, 21]

    # Recorta para os últimos n_days
    df_plot  = df.iloc[-n_days:].copy() if len(df) > n_days else df.copy()
    feat_plot = features_df.reindex(df_plot.index)
    close    = df_plot["close"]
    ticker   = asset.upper()

    # Paletas de cores para as MMs
    _sma_colors = ["#1565C0", "#6A1B9A", "#00695C", "#E65100", "#4E342E"]
    _ema_colors = ["#F57F17", "#AD1457"]

    # ── Calcula indicadores ──────────────────────────────────────────────────
    # SMAs
    smas = {p: close.rolling(p, min_periods=p).mean() for p in sma_periods}

    # EMAs
    emas = {p: close.ewm(span=p, adjust=False).mean() for p in ema_periods}

    # Bandas de Bollinger
    bb_mid   = close.rolling(bb_period, min_periods=bb_period).mean()
    bb_sigma = close.rolling(bb_period, min_periods=bb_period).std()
    bb_upper = bb_mid + bb_std * bb_sigma
    bb_lower = bb_mid - bb_std * bb_sigma

    # ── Layout ───────────────────────────────────────────────────────────────
    n_rows   = 4 if show_volume else 3
    ratios   = [4, 1, 2, 2] if show_volume else [4, 2, 2]
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(15, 11), sharex=True,
        gridspec_kw={"height_ratios": ratios},
    )
    fig.suptitle(
        f"Análise Técnica Comparativa — {ticker} (últimos {n_days} pregões)",
        fontweight="bold", fontsize=13,
    )

    # ── (a) Preço + MMs + Bollinger ──────────────────────────────────────────
    ax_p = axes[0]
    ax_p.plot(close.index, close.values, color="black", lw=1.0,
              alpha=0.85, label="Close", zorder=3)

    # Bandas de Bollinger primeiro (fundo)
    ax_p.fill_between(close.index, bb_upper.values, bb_lower.values,
                      alpha=0.08, color="#1565C0", label=f"Bollinger ({bb_period},{bb_std}σ)")
    ax_p.plot(close.index, bb_upper.values, lw=0.8, ls="--",
              color="#1565C0", alpha=0.6)
    ax_p.plot(close.index, bb_lower.values, lw=0.8, ls="--",
              color="#1565C0", alpha=0.6)
    ax_p.plot(close.index, bb_mid.values,   lw=0.8, ls="-",
              color="#1565C0", alpha=0.4, label=f"BB Média ({bb_period})")

    # SMAs
    for i, (p, sma) in enumerate(smas.items()):
        color = _sma_colors[i % len(_sma_colors)]
        ax_p.plot(sma.index, sma.values, lw=1.2, color=color,
                  alpha=0.85, label=f"SMA{p}", zorder=2)

    # EMAs
    for i, (p, ema) in enumerate(emas.items()):
        color = _ema_colors[i % len(_ema_colors)]
        ax_p.plot(ema.index, ema.values, lw=1.4, ls="dashdot",
                  color=color, alpha=0.90, label=f"EMA{p}", zorder=2)

    ax_p.set_ylabel("Preço (R$)")
    ax_p.legend(loc="upper left", fontsize=8, ncol=3)
    ax_p.grid(True, alpha=0.3, lw=0.5)

    # ── (b) Volume relativo (opcional) ───────────────────────────────────────
    row = 1
    if show_volume and "volume" in df_plot.columns:
        ax_v = axes[row]
        vol  = df_plot["volume"].values
        vol_ma = df_plot["volume"].rolling(20, min_periods=1).mean().values
        # Barras coloridas: verde se close > close anterior, vermelho caso contrário
        ret_sign = np.sign(close.diff().values)
        colors_v = ["#2E7D32" if s >= 0 else "#C62828" for s in ret_sign]
        ax_v.bar(close.index, vol, color=colors_v, alpha=0.5, width=1.0)
        ax_v.plot(close.index, vol_ma, color="#F57F17", lw=1.0, label="Vol MA20")
        ax_v.set_ylabel("Volume")
        ax_v.legend(loc="upper left", fontsize=8)
        ax_v.grid(True, alpha=0.2)
        row += 1

    # ── (c) RSI14 com zonas clássicas ────────────────────────────────────────
    ax_rsi = axes[row]
    if "rsi14" in feat_plot.columns:
        rsi_vals = feat_plot["rsi14"].values
        ax_rsi.plot(feat_plot.index, rsi_vals,
                    color=COLORS["rsi"], lw=1.1, label="RSI14")
        ax_rsi.axhline(70, ls="--", color="red",   lw=1.0, alpha=0.7,
                       label="Sobrecompra (70)")
        ax_rsi.axhline(30, ls="--", color="green", lw=1.0, alpha=0.7,
                       label="Sobrevenda (30)")
        ax_rsi.axhline(50, ls=":",  color="gray",  lw=0.8, alpha=0.6)
        ax_rsi.fill_between(feat_plot.index, 70, 100,
                             alpha=0.07, color="red")
        ax_rsi.fill_between(feat_plot.index, 0, 30,
                             alpha=0.07, color="green")
        # Destaca cruzamentos: saída de sobrecompra
        rsi_s  = feat_plot["rsi14"]
        cruz_sc = (rsi_s.shift(1) >= 70) & (rsi_s < 70)
        cruz_sv = (rsi_s.shift(1) <= 30) & (rsi_s > 30)
        if cruz_sc.any():
            ax_rsi.scatter(rsi_s.index[cruz_sc], rsi_s[cruz_sc],
                           marker="v", color="red", s=80, zorder=5,
                           label="Cruzamento 70↓ (venda)")
        if cruz_sv.any():
            ax_rsi.scatter(rsi_s.index[cruz_sv], rsi_s[cruz_sv],
                           marker="^", color="green", s=80, zorder=5,
                           label="Cruzamento 30↑ (compra)")
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_ylabel("RSI14")
    ax_rsi.legend(loc="upper left", fontsize=8)
    ax_rsi.grid(True, alpha=0.3)
    row += 1

    # ── (d) RSI-LPF com regime 50 (Guerra, 2025) ─────────────────────────────
    ax_lpf = axes[row]
    if "rsi_lpf" in feat_plot.columns:
        lpf_vals = feat_plot["rsi_lpf"]
        ax_lpf.plot(feat_plot.index, lpf_vals.values,
                    color=COLORS["rsi_lpf"], lw=1.4, label="RSI-LPF (p=5)")
        ax_lpf.axhline(50, ls="--", color="gray", lw=1.2, label="Regime 50")
        ax_lpf.fill_between(feat_plot.index, 50, lpf_vals.values,
                             where=(lpf_vals >= 50), alpha=0.15,
                             color="green", label="Alta (LPF>50)")
        ax_lpf.fill_between(feat_plot.index, lpf_vals.values, 50,
                             where=(lpf_vals < 50), alpha=0.15,
                             color="red", label="Baixa (LPF<50)")
    ax_lpf.set_ylim(0, 100)
    ax_lpf.set_ylabel("RSI-LPF")
    ax_lpf.legend(loc="upper left", fontsize=8)
    ax_lpf.grid(True, alpha=0.3)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig
# ---------------------------------------------------------------------------
# 7. Painel completo (todas as figuras em uma chamada)
# ---------------------------------------------------------------------------

def generate_all_figures(
    df_price: pd.DataFrame,
    features_df: pd.DataFrame,
    backtest_result,
    predictions_df: pd.DataFrame,
    walk_results,
    metrics,
    output_dir: str = "resultados/figuras",
    asset: str = "BOVA11",
) -> dict[str, Path]:
    """
    Gera e salva todas as figuras do relatório.

    Parâmetros
    ----------
    df_price : DataFrame com preços (close, high, low)
    features_df : DataFrame de features (saída de build_feature_matrix)
    backtest_result : BacktestResult
    predictions_df  : DataFrame consolidado do walk-forward
    walk_results    : list[WindowResult]
    metrics         : FinancialMetrics
    output_dir      : diretório de saída
    asset           : ticker do ativo (ex: "bova11", "ivvb11", "qbtc11")

    Retorna
    -------
    dict com paths das figuras salvas.
    """
    ticker  = asset.upper()
    out_dir = _ensure_dir(output_dir)
    saved: dict[str, Path] = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # 1. Preço + RSI + RSI-LPF
        p = out_dir / f"01_preco_rsi_lpf_{ticker}.png"
        plot_price_rsi(df_price, features_df, save_path=str(p), asset=asset)
        saved["price_rsi"] = p
        plt.close("all")

        # 2. Curva de capital
        p = out_dir / f"02_equity_curve_{ticker}.png"
        plot_equity_curve(
            backtest_result.equity_curve,
            backtest_result.benchmark_curve,
            save_path=str(p),
            asset=asset,
        )
        saved["equity"] = p
        plt.close("all")

        # 3. Curva A-R
        p = out_dir / f"03_ar_curve_{ticker}.png"
        plot_ar_curve(
            metrics.ar_thresholds,
            metrics.ar_accuracies,
            metrics.ar_rejection_rates,
            save_path=str(p),
            asset=asset,
        )
        saved["ar_curve"] = p
        plt.close("all")

        # 4. Distribuição de retornos
        p = out_dir / f"04_return_distribution_{ticker}.png"
        plot_return_distribution(
            backtest_result.daily_returns,
            backtest_result.benchmark_returns,
            save_path=str(p),
            asset=asset,
        )
        saved["ret_dist"] = p
        plt.close("all")

        # 5. Walk-forward
        if walk_results:
            p = out_dir / f"05_walk_forward_{ticker}.png"
            plot_walk_forward_summary(walk_results, save_path=str(p), asset=asset)
            saved["walk_forward"] = p
            plt.close("all")

        # 6. Sinais no preço
        if not predictions_df.empty:
            p = out_dir / f"06_signals_{ticker}.png"
            plot_signals_on_price(df_price, predictions_df, save_path=str(p), asset=asset)
            saved["signals"] = p
            plt.close("all")

    print(f"[Visualizações] {len(saved)} figuras salvas em '{out_dir}'")
    return saved



