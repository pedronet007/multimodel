"""
src/metrics.py
==============
Métricas de desempenho financeiro e classificação (Seção 3.5.2).

Implementa:
    - Índice de Sharpe anualizado
    - Índice de Sortino anualizado
    - Máximo Drawdown (MDD)
    - Alpha vs benchmark
    - Taxa de rejeição
    - Curva A-R (Acurácia × Rejeição) — Seção 2.10.3
    - Métricas de classificação: acurácia, precisão, recall, F1

Referências:
    Seção 3.5.2 da dissertação
    Rocha-Neto (2011) — Curvas A-R
    Chow (1970) — Classificação com opção de rejeição
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Dataclass de métricas financeiras
# ---------------------------------------------------------------------------

@dataclass
class FinancialMetrics:
    """Conjunto completo de métricas financeiras e de classificação."""

    # Desempenho financeiro
    total_return_strategy: float    # retorno total da estratégia
    total_return_benchmark: float   # retorno total buy-and-hold
    cagr_strategy: float            # CAGR da estratégia
    sharpe_ratio: float             # Índice de Sharpe anualizado
    sortino_ratio: float            # Índice de Sortino anualizado
    max_drawdown: float             # máximo drawdown (negativo)
    alpha: float                    # alpha vs benchmark
    beta: float                     # beta vs benchmark
    calmar_ratio: float             # CAGR / |MDD|

    # Operacional
    hit_rate: float                 # taxa de acerto de trades lucrativos
    n_trades: int
    n_buy: int
    n_sell: int
    n_rejected: int

    # Classificação
    accuracy_all: float             # acurácia em todas as amostras
    accuracy_accepted: float        # acurácia nas amostras aceitas
    rejection_rate: float           # fração de amostras rejeitadas

    # Curva A-R (arrays)
    ar_thresholds: np.ndarray = field(default_factory=lambda: np.array([]))
    ar_accuracies: np.ndarray = field(default_factory=lambda: np.array([]))
    ar_rejection_rates: np.ndarray = field(default_factory=lambda: np.array([]))


# ---------------------------------------------------------------------------
# Funções individuais de métricas
# ---------------------------------------------------------------------------

def sharpe_ratio(
    returns: pd.Series | np.ndarray,
    risk_free_annual: float = 0.1175,
    periods_per_year: int = 252,
) -> float:
    """
    Calcula o Índice de Sharpe anualizado.

    Sharpe = (Rp − Rf) / σp × √T

    Parâmetros
    ----------
    returns : série de retornos diários
    risk_free_annual : taxa livre de risco anual (default: SELIC média ~11.75%)
    periods_per_year : número de pregões por ano

    Retorna
    -------
    float : Sharpe anualizado
    """
    returns = np.asarray(returns)
    rf_daily = (1 + risk_free_annual) ** (1 / periods_per_year) - 1
    excess   = returns - rf_daily
    std      = np.std(excess, ddof=1)
    if std < 1e-10:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series | np.ndarray,
    risk_free_annual: float = 0.1175,
    periods_per_year: int = 252,
) -> float:
    """
    Calcula o Índice de Sortino (penaliza apenas volatilidade negativa).

    Sortino = (Rp − Rf) / σ_down × √T

    Parâmetros
    ----------
    returns : série de retornos diários
    risk_free_annual : taxa livre de risco anual
    periods_per_year : número de pregões por ano

    Retorna
    -------
    float : Sortino anualizado
    """
    returns  = np.asarray(returns)
    rf_daily = (1 + risk_free_annual) ** (1 / periods_per_year) - 1
    excess   = returns - rf_daily
    downside = excess[excess < 0]
    if len(downside) < 2:
        return 0.0
    sigma_down = np.std(downside, ddof=1)
    if sigma_down < 1e-10:
        return 0.0
    return float(np.mean(excess) / sigma_down * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: pd.Series | np.ndarray) -> float:
    """
    Calcula o Máximo Drawdown (maior queda percentual acumulada).

    MDD = max_t [ (peak_t − equity_t) / peak_t ]

    Parâmetros
    ----------
    equity_curve : curva de capital

    Retorna
    -------
    float : MDD como número negativo (ex.: -0.25 = −25%)
    """
    equity = np.asarray(equity_curve, dtype=float)
    if len(equity) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd   = (equity - peak) / np.where(peak > 0, peak, 1)
    return float(dd.min())


def cagr(
    equity_curve: pd.Series | np.ndarray,
    periods_per_year: int = 252,
) -> float:
    """
    Calcula o CAGR (Compound Annual Growth Rate).

    CAGR = (V_final / V_inicial)^(T/anos) − 1

    Parâmetros
    ----------
    equity_curve : curva de capital
    periods_per_year : períodos por ano

    Retorna
    -------
    float : CAGR como decimal
    """
    equity = np.asarray(equity_curve, dtype=float)
    if len(equity) < 2 or equity[0] <= 0:
        return 0.0
    years = len(equity) / periods_per_year
    if years <= 0:
        return 0.0
    return float((equity[-1] / equity[0]) ** (1 / years) - 1)


def alpha_beta(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    risk_free_annual: float = 0.1175,
    periods_per_year: int = 252,
) -> tuple[float, float]:
    """
    Calcula Alpha e Beta vs benchmark.

    CAPM: Rp = α + β × Rm + ε
    α = excesso de retorno após ajuste pelo risco

    Parâmetros
    ----------
    strategy_returns  : retornos diários da estratégia
    benchmark_returns : retornos diários do benchmark (BOVA11)
    risk_free_annual  : taxa livre de risco anual
    periods_per_year  : períodos por ano

    Retorna
    -------
    (alpha_annual, beta)
    """
    s = np.asarray(strategy_returns)
    b = np.asarray(benchmark_returns)
    n = min(len(s), len(b))
    s, b = s[:n], b[:n]

    rf_daily = (1 + risk_free_annual) ** (1 / periods_per_year) - 1
    xs, xb   = s - rf_daily, b - rf_daily

    var_b = np.var(xb, ddof=1)
    if var_b < 1e-12:
        return 0.0, 1.0

    beta_val  = float(np.cov(xs, xb, ddof=1)[0, 1] / var_b)
    alpha_val = float(np.mean(xs) - beta_val * np.mean(xb))
    alpha_ann = float((1 + alpha_val) ** periods_per_year - 1)

    return alpha_ann, beta_val


# ---------------------------------------------------------------------------
# Curva A-R (Acurácia × Rejeição) — Seção 2.10.3
# ---------------------------------------------------------------------------

def accuracy_rejection_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    n_points: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula a Curva Acurácia-Rejeição (A-R Curve).

    Para cada limiar τ (do mais alto ao mais baixo):
        - Aceita apenas amostras com confiança ≥ τ
        - Calcula acurácia nessas amostras
        - Registra taxa de rejeição

    Um modelo superior tem curva A-R consistentemente acima da concorrente
    (Rocha-Neto, 2011 — Seção 2.10.3).

    Parâmetros
    ----------
    y_true      : rótulos verdadeiros (0/1)
    y_pred      : predições do modelo (-1=rejeição, 0=venda, 1=compra)
    confidence  : p̂_max para cada amostra
    n_points    : número de pontos na curva

    Retorna
    -------
    thresholds      : array de limiares τ
    accuracies      : acurácia nos exemplos aceitos para cada τ
    rejection_rates : fração de exemplos rejeitados para cada τ
    """
    thresholds = np.linspace(0.50, 0.99, n_points)
    accuracies      = []
    rejection_rates = []

    y_true = np.asarray(y_true)
    conf   = np.asarray(confidence)

    for tau in thresholds:
        accepted_mask = conf >= tau
        rej_rate = 1.0 - accepted_mask.mean()

        if accepted_mask.sum() < 5:
            accuracies.append(np.nan)
            rejection_rates.append(rej_rate)
            continue

        # Predição para exemplos aceitos: argmax da confiança
        # (1 se p(compra) > p(venda), 0 caso contrário)
        y_pred_accepted = (y_pred[accepted_mask] == 1).astype(int)
        y_true_accepted = y_true[accepted_mask].astype(int)

        acc = (y_pred_accepted == y_true_accepted).mean()
        accuracies.append(acc)
        rejection_rates.append(rej_rate)

    return (
        np.array(thresholds),
        np.array(accuracies),
        np.array(rejection_rates),
    )


# ---------------------------------------------------------------------------
# Função principal: compila todas as métricas
# ---------------------------------------------------------------------------

def compute_all_metrics(
    backtest_result,           # BacktestResult
    predictions_df: pd.DataFrame,
    config: dict,
    n_years: Optional[float] = None,
) -> FinancialMetrics:
    """
    Compila todas as métricas financeiras e de classificação.

    Parâmetros
    ----------
    backtest_result : BacktestResult
    predictions_df  : DataFrame consolidado do walk_forward
    config : dict
    n_years : float | None  — anos do período analisado (calculado automaticamente)

    Retorna
    -------
    FinancialMetrics
    """
    eval_cfg = config.get("evaluation", {})
    rf_ann   = eval_cfg.get("risk_free_rate_annual", 0.1175)
    ar_pts   = eval_cfg.get("ar_curve_thresholds", 20)

    eq  = backtest_result.equity_curve
    bm  = backtest_result.benchmark_curve
    dr  = backtest_result.daily_returns
    bmr = backtest_result.benchmark_returns

    # Retornos totais
    total_ret_s  = (eq.iloc[-1] / eq.iloc[0]) - 1.0
    total_ret_bm = (bm.iloc[-1] / bm.iloc[0]) - 1.0

    # Métricas financeiras
    sharpe = sharpe_ratio(dr.values, risk_free_annual=rf_ann)
    sortino = sortino_ratio(dr.values, risk_free_annual=rf_ann)
    mdd    = max_drawdown(eq.values)
    cagr_s = cagr(eq.values)
    calmar = float(cagr_s / abs(mdd)) if mdd < -1e-6 else 0.0
    alph, bet = alpha_beta(dr.values, bmr.values, risk_free_annual=rf_ann)

    # Métricas de classificação (todas amostras)
    pred = predictions_df.dropna(subset=["y_cls_true", "decision"])
    y_true = pred["y_cls_true"].values
    y_dec  = pred["decision"].values
    conf   = pred["confidence"].values if "confidence" in pred.columns else np.ones(len(pred)) * 0.6

    # Acurácia geral (excluindo rejeições)
    accepted_mask = y_dec != -1
    if accepted_mask.sum() > 0:
        acc_accepted = (y_dec[accepted_mask] == y_true[accepted_mask]).mean()
    else:
        acc_accepted = np.nan

    acc_all = (
        (y_dec[y_dec != -1] == y_true[y_dec != -1]).mean()
        if (y_dec != -1).sum() > 0 else np.nan
    )

    rej_rate = (~accepted_mask).mean()

    # Curva A-R
    taus, accs, rej_rates = accuracy_rejection_curve(
        y_true, y_dec, conf, n_points=ar_pts
    )

    return FinancialMetrics(
        total_return_strategy  = float(total_ret_s),
        total_return_benchmark = float(total_ret_bm),
        cagr_strategy          = float(cagr_s),
        sharpe_ratio           = sharpe,
        sortino_ratio          = sortino,
        max_drawdown           = mdd,
        alpha                  = alph,
        beta                   = bet,
        calmar_ratio           = calmar,
        hit_rate               = backtest_result.hit_rate,
        n_trades               = backtest_result.n_trades,
        n_buy                  = backtest_result.n_buy,
        n_sell                 = backtest_result.n_sell,
        n_rejected             = backtest_result.n_rejected,
        accuracy_all           = float(acc_all) if not np.isnan(acc_all) else 0.0,
        accuracy_accepted      = float(acc_accepted) if not np.isnan(acc_accepted) else 0.0,
        rejection_rate         = float(rej_rate),
        ar_thresholds          = taus,
        ar_accuracies          = accs,
        ar_rejection_rates     = rej_rates,
    )


def print_metrics_report(m: FinancialMetrics) -> None:
    """Exibe relatório formatado das métricas."""
    sep = "─" * 55
    print(f"\n{'═'*55}")
    print(f"{'RELATÓRIO DE DESEMPENHO':^55}")
    print(f"{'Sistemas Inteligentes para Alocação de Capital — B3':^55}")
    print(f"{'═'*55}")

    print(f"\n{'DESEMPENHO FINANCEIRO':}")
    print(sep)
    print(f"  Retorno Total Estratégia : {m.total_return_strategy:+.2%}")
    print(f"  Retorno Total Benchmark  : {m.total_return_benchmark:+.2%}")
    print(f"  CAGR Estratégia          : {m.cagr_strategy:+.2%}")
    print(f"  Índice de Sharpe         : {m.sharpe_ratio:.4f}")
    print(f"  Índice de Sortino        : {m.sortino_ratio:.4f}")
    print(f"  Máximo Drawdown          : {m.max_drawdown:.2%}")
    print(f"  Alpha (vs BOVA11)        : {m.alpha:+.4f}")
    print(f"  Beta                     : {m.beta:.4f}")
    print(f"  Índice de Calmar         : {m.calmar_ratio:.4f}")

    print(f"\n{'OPERACIONAL':}")
    print(sep)
    print(f"  Total de Operações  : {m.n_trades}")
    print(f"  Compras             : {m.n_buy}")
    print(f"  Vendas              : {m.n_sell}")
    print(f"  Sinais Rejeitados   : {m.n_rejected}")
    print(f"  Taxa de Acerto      : {m.hit_rate:.2%}")

    print(f"\n{'CLASSIFICAÇÃO':}")
    print(sep)
    print(f"  Acurácia (aceitas)  : {m.accuracy_accepted:.4f}")
    print(f"  Acurácia (geral)    : {m.accuracy_all:.4f}")
    print(f"  Taxa de Rejeição    : {m.rejection_rate:.2%}")
    print(f"{'═'*55}\n")


# ---------------------------------------------------------------------------
# Calibração de Wr — Chow (1970)
# ---------------------------------------------------------------------------

def calibrate_wr(
    y_true:     np.ndarray,
    confidence: np.ndarray,
    Wc:         float = 0.00,
    We:         float = 0.05,
    n_wr:       int   = 50,
    min_accept: float = 0.30,
    max_accept: float = 0.90,
    verbose:    bool  = True,
) -> dict:
    """
    Calibra Wr varrendo uma grade e escolhendo o valor que maximiza
    acurácia sobre os aceitos, respeitando limites de taxa de rejeição.

    Abordagem:
        Para cada Wr na grade [Wc+ε, We-ε]:
            1. Calcula τ = (We − Wr) / (We − Wc)    [fórmula de Chow]
            2. Aplica τ sobre confidence → aceitos vs rejeitados
            3. Calcula acurácia nos aceitos
            4. Registra (Wr, τ, acurácia, taxa_rejeição)

        O Wr ótimo é o que maximiza acurácia dentro das restrições:
            min_accept ≤ fração_aceita ≤ max_accept

    Parâmetros
    ----------
    y_true      : rótulos verdadeiros (0/1), shape (N,)
    confidence  : max(p̂) para cada amostra, shape (N,)
    Wc          : custo de acertar (fixo, tipicamente 0)
    We          : custo de errar   (fixo, calibrado externamente)
    n_wr        : número de valores de Wr a testar na grade
    min_accept  : fração mínima de amostras que devem ser aceitas
    max_accept  : fração máxima de amostras aceitas (evita Wr alto demais)
    verbose     : imprime tabela de resultados

    Retorna
    -------
    dict com campos:
        Wr_opt      : melhor Wr encontrado
        tau_opt     : τ correspondente
        acc_opt     : acurácia nos aceitos com Wr_opt
        rej_opt     : taxa de rejeição com Wr_opt
        grid        : DataFrame com todos os resultados da grade
    """
    import pandas as pd

    eps  = (We - Wc) * 0.02          # margem para não violar Wc < Wr < We
    wr_grid = np.linspace(Wc + eps, We - eps, n_wr)

    y_true = np.asarray(y_true).astype(int)
    conf   = np.asarray(confidence)

    rows = []
    for Wr in wr_grid:
        tau      = (We - Wr) / (We - Wc)
        accepted = conf >= tau
        rej_rate = 1.0 - accepted.mean()
        n_acc    = accepted.sum()

        if n_acc < 5:
            rows.append(dict(Wr=Wr, tau=tau, acc=np.nan, rej_rate=rej_rate, n_acc=n_acc))
            continue

        acc = (y_true[accepted] == (conf[accepted] >= 0.5).astype(int)).mean()
        rows.append(dict(Wr=Wr, tau=tau, acc=acc, rej_rate=rej_rate, n_acc=n_acc))

    grid = pd.DataFrame(rows)

    # Filtra pelas restrições de taxa de aceitação
    frac_accept = 1.0 - grid["rej_rate"]
    mask = (frac_accept >= min_accept) & (frac_accept <= max_accept) & grid["acc"].notna()

    if mask.sum() == 0:
        # Relaxa: usa linha com maior acurácia sem restrição
        best_idx = grid["acc"].idxmax()
    else:
        best_idx = grid.loc[mask, "acc"].idxmax()

    best = grid.loc[best_idx]

    if verbose:
        sep = "─" * 58
        print(f"\n{'═'*58}")
        print(f"{'CALIBRAÇÃO DE Wr — Chow (1970)':^58}")
        print(f"{'═'*58}")
        print(f"  Wc = {Wc:.4f}  |  We = {We:.4f}  |  n_Wr = {n_wr}")
        print(f"  Restrição: {min_accept:.0%} ≤ aceitos ≤ {max_accept:.0%}")
        print(sep)
        print(f"  {'Wr':>8}  {'τ':>6}  {'Acurácia':>10}  {'% Rej':>7}  {'N aceitos':>10}")
        print(sep)
        # Imprime ~10 linhas representativas
        step = max(1, len(grid) // 10)
        for _, row in grid.iloc[::step].iterrows():
            marker = " ←" if row.name == best_idx else ""
            acc_str = f"{row['acc']:.4f}" if not np.isnan(row['acc']) else "   —  "
            print(f"  {row['Wr']:>8.4f}  {row['tau']:>6.4f}  {acc_str:>10}  "
                  f"{row['rej_rate']:>6.1%}  {int(row['n_acc']):>10}{marker}")
        print(sep)
        print(f"\n  Wr ótimo  : {best['Wr']:.4f}")
        print(f"  τ ótimo   : {best['tau']:.4f}")
        print(f"  Acurácia  : {best['acc']:.4f}")
        print(f"  % Rejeição: {best['rej_rate']:.1%}")
        print(f"{'═'*58}\n")

    return {
        "Wr_opt":  float(best["Wr"]),
        "tau_opt": float(best["tau"]),
        "acc_opt": float(best["acc"]),
        "rej_opt": float(best["rej_rate"]),
        "grid":    grid,
    }
