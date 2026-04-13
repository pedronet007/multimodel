"""
src/backtest.py
===============
Motor de Backtesting (Seção 3.4).

Lógica de posição — três decisões com semânticas distintas:
─────────────────────────────────────────────────────────────
  decision =  1  → COMPRA / MANTER LONG
                   O modelo está confiante de alta.
                   Se em caixa: entra na posição (cobra custo de entrada).
                   Se já em LONG: mantém (sem custo).

  decision =  0  → VENDA (sinal explícito de saída)
                   O modelo está confiante de baixa/neutro.
                   Se em LONG: fecha a posição (aplica ret do dia, cobra custo).
                   Se em caixa: permanece em caixa.

  decision = -1  → REJEIÇÃO (Chow, 1970) — "não sei"
                   O modelo não atingiu o limiar de confiança τ.
                   NÃO É UMA ORDEM. É ausência de sinal.
                   Comportamento: MANTÉM O ESTADO ATUAL.
                     - Se em LONG: continua comprado (aplica ret, sem custo).
                     - Se em caixa: permanece em caixa.
                   Justificativa: tratar rejeição como venda é conceitualmente
                   errado — Chow (1970) define rejeição como abstenção, não como
                   predição de queda. Forçar saída da posição em toda rejeição
                   resulta em whipsaw e custos desnecessários.

Opção B — Fallback por Tendência (regime_guerra):
─────────────────────────────────────────────────
  Camada opcional sobre a rejeição. Ativa com use_trend_fallback=True.
  Fonte: RSI-LPF (Guerra, 2025) — independente do mecanismo de Chow.

  Quando o modelo rejeita (decision=-1) E está em CAIXA:
    regime_guerra = +1  →  entra LONG (fallback de tendência)
    regime_guerra =  0  →  permanece em caixa
    regime_guerra = -1  →  permanece em caixa

  Nota: se já estiver em LONG e o modelo rejeitar, mantém LONG
  independentemente do regime_guerra (a rejeição já preserva o estado).

Custos de transação:
─────────────────────
  Cobrado apenas nas transições de estado:
    CAIXA → LONG  : custo de entrada
    LONG  → CAIXA : custo de saída (só em decision=0)
  Rejeição não gera custo — não há transação.

Stop-loss e Take-profit:
─────────────────────────
  Calculados sobre retorno acumulado desde abertura da posição.
  Fecham a posição independentemente do sinal do modelo.

Benchmark Buy & Hold:
──────────────────────
  Compra R$ capital₀ no primeiro pregão e segura até o final.
  Sem custos, sem saídas intermediárias.

Referências:
  Seção 3.4 da dissertação
  Chow (1970) — On Optimum Recognition Error and Reject Trade-off
  Guerra (2025) — Revisiting the RSI: From Oscillator to Trend-Following
  Bailey et al. (2014) — The Probability of Backtest Overfitting
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Dataclass de resultado
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Resultado completo de um backtest."""
    equity_curve:       pd.Series     # patrimônio da estratégia ao longo do tempo
    benchmark_curve:    pd.Series     # buy-and-hold
    daily_returns:      pd.Series     # retornos diários da estratégia
    benchmark_returns:  pd.Series     # retornos diários do benchmark
    trades:             pd.DataFrame  # log de operações (entradas e saídas)
    n_trades:           int           # número de operações fechadas
    n_buy:              int           # dias com decision=1
    n_sell:             int           # dias com decision=0 (sinal explícito de venda)
    n_rejected:         int           # dias com decision=-1 (rejeição Chow)
    n_rejected_long:    int           # rejeições enquanto em LONG (manteve posição)
    n_rejected_cash:    int           # rejeições enquanto em caixa (ficou parado)
    n_fallback:         int           # rejeições em caixa convertidas em LONG (Opção B)
    hit_rate:           float         # fração de operações lucrativas
    final_capital:      float
    initial_capital:    float
    use_trend_fallback: bool = False


# ---------------------------------------------------------------------------
# Opção B — Fallback por tendência (Guerra, 2025)
# ---------------------------------------------------------------------------

def _apply_fallback(regime_guerra: int) -> bool:
    """
    Decide se uma rejeição em caixa deve virar LONG pela Opção B.

    Só é chamada quando decision=-1 E in_position=False.
    O regime_guerra vem do RSI-LPF — fonte completamente independente
    do mecanismo de rejeição de Chow.

    Retorna True se deve entrar LONG via fallback.
    """
    return regime_guerra == 1


# ---------------------------------------------------------------------------
# Motor principal
# ---------------------------------------------------------------------------

def run_backtest(
    predictions_df:     pd.DataFrame,
    price_df:           pd.DataFrame,
    config:             dict,
    use_selic:          bool = False,
    use_trend_fallback: bool = False,
) -> BacktestResult:
    """
    Motor principal de backtesting.

    Parâmetros
    ----------
    predictions_df     : DataFrame consolidado do walk_forward. Colunas
                         obrigatórias: 'decision', 'y_ret_true', 'rejected'.
                         Opcionais: 'confidence', 'regime_guerra'.
    price_df           : DataFrame OHLCV com coluna 'close'.
    config             : dicionário de configuração (config.yaml).
    use_selic          : capital em caixa rende SELIC diária.
    use_trend_fallback : ativa Opção B (requer coluna 'regime_guerra').
    """
    if use_trend_fallback and "regime_guerra" not in predictions_df.columns:
        raise ValueError(
            "use_trend_fallback=True requer coluna 'regime_guerra' em "
            "predictions_df. Verifique walk_forward.consolidate_results()."
        )

    bt_cfg          = config["backtest"]
    initial_capital = bt_cfg["initial_capital"]
    cost_rate       = bt_cfg["transaction_cost"]

    pred  = predictions_df.copy()
    pred  = pred[~pred.index.duplicated(keep="first")]
    close = price_df["close"].reindex(pred.index).ffill()

    if use_selic and "selic_pct" in price_df.columns:
        selic_daily = price_df["selic_pct"].reindex(pred.index).ffill() / 252 / 100
    else:
        selic_daily = pd.Series(0.0, index=pred.index)

    capital       = initial_capital
    in_position   = False
    entry_price   = None
    entry_capital = None

    trades           = []
    equity_series    = {}
    daily_ret_series = {}
    prev_capital     = capital

    # contadores para diagnóstico
    n_rejected_long = 0
    n_rejected_cash = 0
    n_fallback      = 0

    for date, row in pred.iterrows():
        decision  = int(row.get("decision", -1))
        ret_true  = float(row.get("y_ret_true", 0.0))
        price_now = close.get(date, np.nan)
        selic_d   = float(selic_daily.get(date, 0.0))

        if pd.isna(price_now):
            equity_series[date]    = capital
            daily_ret_series[date] = 0.0
            continue

        # ── Lógica de posição por decisão ─────────────────────────────────────
        # Saída SOMENTE por sinal do modelo (decision=0).
        # Stop-loss e take-profit foram removidos intencionalmente para que
        # o comparativo entre modelos reflita exclusivamente a qualidade do
        # sinal gerado — sem interferência de regras externas de gestão.
        
        # Identifica o regime atual (0 se a Opção B estiver desligada)
        regime = int(row.get("regime_guerra", 0)) if use_trend_fallback else 0

        # --- APLICAÇÃO DO VETO DIRECIONAL ---
        if use_trend_fallback:
            # VETO DE VENDA: IA quer vender (0) em pleno regime de Alta (+1)
            if decision == 0 and regime == 1:
                decision = -1  # Força status quo
            
            # VETO DE COMPRA: IA quer comprar (1) em pleno regime de Baixa (-1)
            elif decision == 1 and regime == -1:
                decision = -1  # Força status quo

        if decision == 1:
            # ── COMPRA / MANTER LONG (Sincronizado com a Tendência) ───────────
            if not in_position:
                capital      *= (1.0 - cost_rate)
                in_position   = True
                entry_price   = price_now
                entry_capital = capital
            capital *= np.exp(ret_true)

        elif decision == 0:
            # ── VENDA — sinal explícito de saída (Sincronizado com a Tendência)
            if in_position:
                # Captura retorno do dia antes de fechar
                capital *= np.exp(ret_true)
                capital *= (1.0 - cost_rate)
                exit_ret = (capital / entry_capital) - 1.0 if entry_capital else 0.0
                trades.append({
                    "date":        date,
                    "action":      "EXIT",
                    "triggered":   "signal_0",
                    "return_pct":  round(exit_ret * 100, 3),
                    "capital":     round(capital, 2),
                    "profitable":  capital > entry_capital if entry_capital else False,
                    "via_fallback": False,
                })
                in_position   = False
                entry_price   = None
                entry_capital = None
            capital *= (1.0 + selic_d)

        else:
            # ── REJEIÇÃO (decision=-1) — Chow / Veto Direcional ──────────────
            if in_position:
                capital *= np.exp(ret_true)
                n_rejected_long += 1
            else:
                n_rejected_cash += 1
                
                # O Fallback de compra ocorre apenas se a IA originalmente rejeitou
                # ou foi vetada, E o regime é de alta.
                if use_trend_fallback and _apply_fallback(regime):
                    capital      *= (1.0 - cost_rate)
                    in_position   = True
                    entry_price   = price_now
                    entry_capital = capital
                    capital      *= np.exp(ret_true)
                    n_fallback   += 1
                    trades.append({
                        "date":        date,
                        "action":      "ENTRY_FALLBACK",
                        "return_pct":  0.0,
                        "capital":     round(capital, 2),
                        "profitable":  None,
                        "via_fallback": True,
                    })
                else:
                    capital *= (1.0 + selic_d)

        # ── 3. Registra capital e retorno diário ──────────────────────────────
        capital = max(capital, 0.01)
        equity_series[date]    = capital
        daily_ret_series[date] = (capital - prev_capital) / prev_capital
        prev_capital           = capital

    # ── Benchmark Buy & Hold ──────────────────────────────────────────────────
    dates_idx      = list(equity_series.keys())
    close_aligned  = close.reindex(dates_idx).ffill()
    close_start    = close_aligned.iloc[0]
    benchmark_vals = initial_capital * (close_aligned / close_start)

    equity_s    = pd.Series(equity_series,                               name="equity")
    benchmark_s = pd.Series(benchmark_vals.values, index=equity_s.index, name="benchmark")
    daily_ret_s = pd.Series(daily_ret_series,                            name="daily_return")
    bench_ret_s = benchmark_s.pct_change().fillna(0)

    trades_df = (
        pd.DataFrame(trades) if trades else
        pd.DataFrame(columns=["date", "action", "return_pct", "capital",
                               "profitable", "via_fallback"])
    )

    exit_trades = [t for t in trades if t.get("action") == "EXIT"]
    n_entries   = len(exit_trades)
    n_buy       = int((pred["decision"] == 1).sum())
    n_sell      = int((pred["decision"] == 0).sum())
    n_rejected  = int(pred["rejected"].sum()) if "rejected" in pred.columns else 0
    profitable  = [t for t in exit_trades if t.get("profitable", False)]
    hit_rate    = len(profitable) / n_entries if n_entries > 0 else 0.0

    return BacktestResult(
        equity_curve        = equity_s,
        benchmark_curve     = benchmark_s,
        daily_returns       = daily_ret_s,
        benchmark_returns   = bench_ret_s,
        trades              = trades_df,
        n_trades            = n_entries,
        n_buy               = n_buy,
        n_sell              = n_sell,
        n_rejected          = n_rejected,
        n_rejected_long     = n_rejected_long,
        n_rejected_cash     = n_rejected_cash,
        n_fallback          = n_fallback,
        hit_rate            = hit_rate,
        final_capital       = float(equity_s.iloc[-1]),
        initial_capital     = initial_capital,
        use_trend_fallback  = use_trend_fallback,
    )


# ---------------------------------------------------------------------------
# Backtest simplificado — modo baseline (RSI-LPF puro, sem LSTM)
# ---------------------------------------------------------------------------

def simple_signal_backtest(
    dates:           pd.DatetimeIndex,
    decisions:       np.ndarray,
    log_returns:     np.ndarray,
    initial_capital: float = 10_000.0,
    cost_rate:       float = 0.001,
) -> pd.Series:
    """
    Backtest simplificado LONG-only para análise rápida de sinais.

    Aplica a mesma semântica correta do motor principal:
        decision =  1 → LONG: capital × exp(ret)
        decision =  0 → VENDA: fecha posição
        decision = -1 → REJEIÇÃO: mantém estado atual

    Custo cobrado apenas nas transições CAIXA→LONG e LONG→CAIXA.
    """
    capital     = initial_capital
    in_position = False
    equity      = []

    for dec, lret in zip(decisions, log_returns):
        if dec == 1:
            if not in_position:
                capital    *= (1.0 - cost_rate)
                in_position = True
            capital *= np.exp(lret)

        elif dec == 0:
            if in_position:
                capital    *= (1.0 - cost_rate)
                in_position = False
            # Em caixa: capital estável

        else:
            # Rejeição: mantém estado atual
            if in_position:
                capital *= np.exp(lret)
            # Em caixa: capital estável

        capital = max(capital, 0.01)
        equity.append(capital)

    return pd.Series(equity, index=dates, name="equity")
