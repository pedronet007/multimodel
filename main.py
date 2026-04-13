"""
main.py
=======
Script principal — Pipeline B3 com modelos plugáveis.
"Sistemas Inteligentes para Alocação de Capital no Mercado Acionário"

Autor: Pedro Wilson Félix Magalhães Neto
Instituição: IFCE — Programa de Pós-Graduação em Engenharia de Telecomunicações

Uso básico:
    python main.py --asset bova11                          # LSTM (padrão)
    python main.py --asset bova11 --model gru              # modelo único
    python main.py --asset bova11 --models lstm gru mlp    # múltiplos sequencial
    python main.py --asset bova11 --models all             # todos os modelos
    python main.py --asset bova11 --models all --demo      # todos em modo rápido

Outros modos:
    python main.py --asset bova11 --mode baseline          # RSI-LPF puro (sem rede)
    python main.py --all --models lstm gru                 # todos os CSVs × 2 modelos
    python main.py --asset petr4 --file /caminho/dados.csv # CSV personalizado

Modelos disponíveis: lstm, gru, mlp, knn
"""

import sys
import argparse
import warnings
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_asset, load_and_prepare
from src.features import build_feature_matrix, create_targets
from src.backtest import run_backtest, simple_signal_backtest
from src.metrics import compute_all_metrics, print_metrics_report
from src.models import MODEL_REGISTRY


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config.yaml", base_dir: str = ".") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["_base_dir"] = base_dir
    return config


# ---------------------------------------------------------------------------
# Modo Baseline — RSI-LPF puro (sem rede neural)
# ---------------------------------------------------------------------------

def run_analysis_mode(
    config: dict,
    asset: str,
    demo: bool = False,
    custom_file: Optional[str] = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")

    from src.features import build_feature_matrix, create_targets
    from src.backtest import simple_signal_backtest
    from src.metrics import (sharpe_ratio, sortino_ratio, max_drawdown,
                              cagr, alpha_beta, accuracy_rejection_curve,
                              print_metrics_report, FinancialMetrics)
    from src.visualizations import (plot_price_rsi, plot_equity_curve,
                                     plot_ar_curve, _ensure_dir)

    ticker = asset.upper()
    print(f"\n{'='*60}")
    print(f"  Modo Análise (RSI-LPF Baseline) — Ativo: {ticker}")
    print(f"{'='*60}")

    print("\n[1/5] Carregando dados...")
    df = load_asset(asset, config, custom_file)
    print(f"      {len(df)} pregões | {df.index[0].date()} → {df.index[-1].date()}")

    print("\n[2/5] Calculando features RSI e RSI-LPF...")
    features_df = build_feature_matrix(df, config)
    targets_df  = create_targets(df, config)
    common_idx  = features_df.index.intersection(targets_df.index)
    features_df = features_df.loc[common_idx]
    targets_df  = targets_df.loc[common_idx]
    print(f"      {len(features_df)} amostras | {features_df.shape[1]} features")

    print("\n[3/5] Aplicando estratégia RSI-LPF baseline...")
    rsi_lpf  = features_df["rsi_lpf"].values
    log_rets = targets_df["y_ret"].values
    dates    = features_df.index

    rejection_band = 3.0
    decisions = np.where(
        np.abs(rsi_lpf - 50) < rejection_band, -1,
        np.where(rsi_lpf > 50, 1, 0)
    )

    equity_strategy = simple_signal_backtest(
        dates, decisions, log_rets,
        initial_capital=config["backtest"]["initial_capital"],
        cost_rate=config["backtest"]["transaction_cost"],
    )

    close      = df["close"].reindex(dates)
    bh_returns = np.log(close / close.shift(1)).fillna(0).values
    equity_bh  = simple_signal_backtest(
        dates, np.ones(len(dates), dtype=int), bh_returns,
        initial_capital=config["backtest"]["initial_capital"],
        cost_rate=0.0,
    )

    print("\n[4/5] Calculando métricas...")
    rf_ann    = config["evaluation"]["risk_free_rate_annual"]
    dr_strat  = equity_strategy.pct_change().fillna(0).values
    dr_bm     = equity_bh.pct_change().fillna(0).values

    sharpe_s  = sharpe_ratio(dr_strat,  risk_free_annual=rf_ann)
    sortino_s = sortino_ratio(dr_strat, risk_free_annual=rf_ann)
    mdd_s     = max_drawdown(equity_strategy.values)
    cagr_s    = cagr(equity_strategy.values)
    alph, bet = alpha_beta(dr_strat, dr_bm, risk_free_annual=rf_ann)
    total_ret_s  = (equity_strategy.iloc[-1] / equity_strategy.iloc[0]) - 1
    total_ret_bm = (equity_bh.iloc[-1] / equity_bh.iloc[0]) - 1

    confidence_arr = np.clip(np.abs(rsi_lpf - 50) / 50.0 + 0.5, 0.5, 1.0)
    y_true_arr     = targets_df["y_decision"].values
    n_accepted     = (decisions != -1).sum()
    n_rejected     = (decisions == -1).sum()
    acc_accepted   = (
        (decisions[decisions != -1] == y_true_arr[decisions != -1]).mean()
        if n_accepted > 0 else np.nan
    )
    taus, accs, rejs = accuracy_rejection_curve(
        y_true_arr, decisions, confidence_arr,
        n_points=config["evaluation"]["ar_curve_thresholds"],
    )

    m = FinancialMetrics(
        total_return_strategy  = float(total_ret_s),
        total_return_benchmark = float(total_ret_bm),
        cagr_strategy          = cagr_s,
        sharpe_ratio           = sharpe_s,
        sortino_ratio          = sortino_s,
        max_drawdown           = mdd_s,
        alpha                  = alph,
        beta                   = bet,
        calmar_ratio           = float(cagr_s / abs(mdd_s)) if mdd_s < -1e-6 else 0.0,
        hit_rate               = float(acc_accepted) if not np.isnan(acc_accepted) else 0.0,
        n_trades               = int(np.sum(np.diff(decisions) != 0)),
        n_buy                  = int((decisions == 1).sum()),
        n_sell                 = int((decisions == 0).sum()),
        n_rejected             = int(n_rejected),
        accuracy_all           = float(acc_accepted) if not np.isnan(acc_accepted) else 0.0,
        accuracy_accepted      = float(acc_accepted) if not np.isnan(acc_accepted) else 0.0,
        rejection_rate         = float(n_rejected / len(decisions)),
        ar_thresholds          = taus,
        ar_accuracies          = accs,
        ar_rejection_rates     = rejs,
    )
    print_metrics_report(m)

    print("\n[5/5] Gerando figuras...")
    import matplotlib.pyplot as plt
    out_dir    = _ensure_dir(Path(config["output"]["figures_dir"]))
    report_dir = _ensure_dir(config["output"]["reports_dir"])

    plot_price_rsi(df, features_df,
                   save_path=str(out_dir / f"01_preco_rsi_lpf_{asset}.png"),
                   asset=asset)
    plt.close("all")

    plot_equity_curve(equity_strategy, equity_bh,
                      save_path=str(out_dir / f"02_equity_curve_{asset}.png"),
                      asset=asset)
    plt.close("all")

    plot_ar_curve(taus, accs, rejs,
                  save_path=str(out_dir / f"03_ar_curve_{asset}.png"),
                  asset=asset)
    plt.close("all")

    metrics_dict = {
        "asset": ticker,
        "mode": "rsi_lpf_baseline",
        "total_return_strategy_pct":  round(m.total_return_strategy * 100, 2),
        "total_return_benchmark_pct": round(m.total_return_benchmark * 100, 2),
        "cagr_strategy_pct":          round(m.cagr_strategy * 100, 2),
        "sharpe_ratio":               round(m.sharpe_ratio, 4),
        "sortino_ratio":              round(m.sortino_ratio, 4),
        "max_drawdown_pct":           round(m.max_drawdown * 100, 2),
        "alpha":                      round(m.alpha, 4),
        "beta":                       round(m.beta, 4),
        "accuracy_accepted_pct":      round(m.accuracy_accepted * 100, 2),
        "rejection_rate_pct":         round(m.rejection_rate * 100, 2),
    }
    with open(report_dir / f"metrics_{asset}_baseline.json", "w") as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Figuras em '{out_dir}'")
    print(f"[OK] Métricas em '{report_dir}/metrics_{asset}_baseline.json'")


# ---------------------------------------------------------------------------
# Modo modelo plugável — LSTM / GRU / MLP / KNN
# ---------------------------------------------------------------------------

def run_model_mode(
    config: dict,
    asset: str,
    model_name: str = "LSTM",
    demo: bool = False,
    custom_file: Optional[str] = None,
) -> dict:
    """
    Executa o pipeline completo para um modelo específico.

    Parâmetros
    ----------
    config     : dict de configuração
    asset      : ticker do ativo (ex: 'bova11')
    model_name : nome do modelo — deve estar em MODEL_REGISTRY
    demo       : modo rápido (2 janelas, 5 épocas)
    custom_file: caminho alternativo para o CSV

    Retorna
    -------
    dict com métricas resumidas (útil para comparação entre modelos)
    """
    model_name_upper = model_name.upper()

    if model_name_upper not in MODEL_REGISTRY:
        modelos_disp = list(MODEL_REGISTRY.keys())
        print(f"[ERRO] Modelo '{model_name_upper}' não encontrado.")
        print(f"       Disponíveis: {modelos_disp}")
        return {}

    # Modelos neurais exigem TensorFlow
    if model_name_upper in ("LSTM", "GRU", "MLP", "TRANSFORMER"):
        try:
            import tensorflow as tf
            tf.get_logger().setLevel("ERROR")
        except ImportError:
            print(
                f"\n[AVISO] TensorFlow não encontrado — necessário para {model_name_upper}.\n"
                "Instale com: pip install tensorflow\n"
                "Executando modo baseline RSI-LPF como alternativa...\n"
            )
            run_analysis_mode(config, asset, demo, custom_file)
            return {}

    import matplotlib
    matplotlib.use("Agg")

    from src.walk_forward import run_walk_forward, consolidate_results
    from src.backtest import run_backtest
    from src.metrics import compute_all_metrics, print_metrics_report
    from src.visualizations import generate_all_figures, _ensure_dir

    model_class = MODEL_REGISTRY[model_name_upper]
    ticker      = asset.upper()

    print(f"\n{'='*60}")
    print(f"  Pipeline {model_name_upper} Walk-Forward — Ativo: {ticker}")
    print(f"{'='*60}")

    if demo:
        print(f"\n[DEMO] Limitando a 2 janelas walk-forward para teste rápido.")
        config = config.copy()
        config["walk_forward"] = config["walk_forward"].copy()
        config["walk_forward"]["train_size"] = 300
        config["walk_forward"]["test_size"]  = 30
        config["walk_forward"]["step_size"]  = 30
        config["training"] = config["training"].copy()
        config["training"]["epochs"] = 5

    t_inicio = time.time()

    # 1. Carrega ativo
    print(f"\n[1/5] Carregando dados...")
    df = load_asset(asset, config, custom_file)
    print(f"      {len(df)} pregões | {df.index[0].date()} → {df.index[-1].date()}")

    # 2. Walk-forward com o modelo escolhido
    print(f"\n[2/5] Executando walk-forward [{model_name_upper}]...")
    walk_results = run_walk_forward(df, config, model_class=model_class, verbose=True)
    if not walk_results:
        print("[ERRO] Nenhuma janela walk-forward completada.")
        return {}

    pred_df = consolidate_results(walk_results)

    # 3. Backtest
    print(f"\n[3/5] Executando backtest...")
    bt_result = run_backtest(pred_df, df, config)
    print(f"      Capital final: R${bt_result.final_capital:,.2f} "
          f"(inicial: R${bt_result.initial_capital:,.2f})")
    print(f"      Operações: {bt_result.n_trades} | Acerto: {bt_result.hit_rate:.1%}")

    # 4. Métricas
    print(f"\n[4/5] Calculando métricas...")
    features_df = build_feature_matrix(df, config)
    metrics     = compute_all_metrics(bt_result, pred_df, config)
    print_metrics_report(metrics)

    # 5. Salva resultados — sufixo identifica o modelo
    print(f"\n[5/5] Salvando resultados...")
    rep_dir = _ensure_dir(config["output"]["reports_dir"])

    # CSV de predições: predictions_bova11_gru.csv (sem sufixo para LSTM)
    suffix   = f"_{model_name_upper.lower()}" if model_name_upper != "LSTM" else ""
    csv_path = rep_dir / f"predictions_{asset}{suffix}.csv"
    pred_df.to_csv(csv_path)
    print(f"      Predições → '{csv_path}'")

    # JSON de métricas
    t_elapsed = time.time() - t_inicio
    resumo = {
        "asset":         ticker,
        "model":         model_name_upper,
        "n_pregoes":     len(df),
        "n_janelas":     len(walk_results),
        "tempo_seg":     round(t_elapsed, 1),
        "total_return_strategy_pct":  round(metrics.total_return_strategy  * 100, 2),
        "total_return_benchmark_pct": round(metrics.total_return_benchmark * 100, 2),
        "cagr_strategy_pct":          round(metrics.cagr_strategy          * 100, 2),
        "sharpe_ratio":               round(metrics.sharpe_ratio,                 4),
        "sortino_ratio":              round(metrics.sortino_ratio,                4),
        "max_drawdown_pct":           round(metrics.max_drawdown            * 100, 2),
        "alpha_pct":                  round(metrics.alpha                   * 100, 2),
        "beta":                       round(metrics.beta,                         4),
        "calmar_ratio":               round(metrics.calmar_ratio,                 4),
        "accuracy_accepted_pct":      round(metrics.accuracy_accepted       * 100, 2),
        "rejection_rate_pct":         round(metrics.rejection_rate          * 100, 2),
        "hit_rate_pct":               round(metrics.hit_rate                * 100, 2),
        "n_trades":                   metrics.n_trades,
        "n_buy":                      metrics.n_buy,
        "n_sell":                     metrics.n_sell,
        "n_rejected":                 metrics.n_rejected,
    }

    json_path = rep_dir / f"metrics_{asset}_{model_name_upper.lower()}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)
    print(f"      Métricas   → '{json_path}'")

    # Figuras
    try:
        out_dir = _ensure_dir(config["output"]["figures_dir"])
        generate_all_figures(
            df_price        = df,
            features_df     = features_df,
            backtest_result = bt_result,
            predictions_df  = pred_df,
            walk_results    = walk_results,
            metrics         = metrics,
            output_dir      = str(out_dir / model_name_upper.lower()),
            asset           = asset,
        )
        print(f"      Figuras    → '{out_dir / model_name_upper.lower()}'")
    except Exception as e:
        print(f"      [AVISO] Erro ao gerar figuras: {e}")

    print(f"\n  Tempo total [{model_name_upper}]: {t_elapsed:.0f}s")
    print(f"{'='*60}\n")

    return resumo


# ---------------------------------------------------------------------------
# Modo --models: roda múltiplos modelos sequencialmente e compara
# ---------------------------------------------------------------------------

def run_multiple_models(
    config: dict,
    asset: str,
    models: list[str],
    demo: bool = False,
    custom_file: Optional[str] = None,
) -> None:
    """
    Executa vários modelos sequencialmente sobre o mesmo ativo e
    imprime uma tabela comparativa ao final.

    Parâmetros
    ----------
    config  : configuração
    asset   : ticker do ativo
    models  : lista de nomes de modelos (ex: ['LSTM','GRU','MLP','KNN'])
    demo    : modo rápido
    """
    # Expande "all" para todos os modelos disponíveis
    modelos_disponiveis = list(MODEL_REGISTRY.keys())
    if len(models) == 1 and models[0].upper() == "ALL":
        models_final = modelos_disponiveis
    else:
        models_final = [m.upper() for m in models]
        invalidos    = [m for m in models_final if m not in modelos_disponiveis]
        if invalidos:
            print(f"[ERRO] Modelos não reconhecidos: {invalidos}")
            print(f"       Disponíveis: {modelos_disponiveis}")
            return

    ticker = asset.upper()
    print(f"\n{'='*60}")
    print(f"  Comparação de Modelos — Ativo: {ticker}")
    print(f"  Modelos: {models_final}")
    print(f"{'='*60}")

    resultados = {}
    t_total    = time.time()

    for modelo in models_final:
        print(f"\n{'─'*60}")
        print(f"  Iniciando: {modelo} ({models_final.index(modelo)+1}/{len(models_final)})")
        print(f"{'─'*60}")

        try:
            r = run_model_mode(
                config, asset,
                model_name  = modelo,
                demo        = demo,
                custom_file = custom_file,
            )
            if r:
                resultados[modelo] = r
        except KeyboardInterrupt:
            print(f"\n[INTERROMPIDO] Pulando {modelo}...")
            continue
        except Exception as e:
            print(f"\n[ERRO] {modelo} falhou: {e}")
            import traceback
            traceback.print_exc()
            continue

    # ── Tabela comparativa ───────────────────────────────────────────────────
    if len(resultados) < 2:
        print("\n[AVISO] Menos de 2 modelos concluídos — sem tabela comparativa.")
        return

    t_total_elapsed = time.time() - t_total

    print(f"\n\n{'='*60}")
    print(f"  TABELA COMPARATIVA — {ticker}")
    print(f"{'='*60}")

    # Cabeçalho
    colunas = ["Ret%", "BH%", "CAGR%", "Sharpe", "Sortino", "MDD%", "Acc%", "Rej%", "Trades"]
    print(f"  {'Modelo':8s}", end="")
    for c in colunas:
        print(f"  {c:>8s}", end="")
    print()
    print("  " + "─" * (8 + len(colunas) * 10))

    for modelo, r in resultados.items():
        print(f"  {modelo:8s}", end="")
        print(f"  {r.get('total_return_strategy_pct', 0):8.1f}", end="")
        print(f"  {r.get('total_return_benchmark_pct', 0):8.1f}", end="")
        print(f"  {r.get('cagr_strategy_pct', 0):8.1f}", end="")
        print(f"  {r.get('sharpe_ratio', 0):8.2f}", end="")
        print(f"  {r.get('sortino_ratio', 0):8.2f}", end="")
        print(f"  {r.get('max_drawdown_pct', 0):8.1f}", end="")
        print(f"  {r.get('accuracy_accepted_pct', 0):8.1f}", end="")
        print(f"  {r.get('rejection_rate_pct', 0):8.1f}", end="")
        print(f"  {r.get('n_trades', 0):8d}")

    print(f"\n  Tempo total: {t_total_elapsed:.0f}s")
    print(f"{'='*60}\n")

    # Salva tabela comparativa em JSON
    rep_dir = Path(config["output"]["reports_dir"])
    rep_dir.mkdir(parents=True, exist_ok=True)
    nomes_str = "_".join(m.lower() for m in resultados.keys())
    comp_path = rep_dir / f"comparacao_{asset}_{nomes_str}.json"
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(
            {"asset": ticker, "modelos": resultados, "tempo_total_seg": round(t_total_elapsed, 1)},
            f, indent=2, ensure_ascii=False
        )
    print(f"[OK] Tabela comparativa salva em '{comp_path}'")


# ---------------------------------------------------------------------------
# Modo --all: processa todos os CSVs da pasta data/
# ---------------------------------------------------------------------------

def run_all_assets(
    config: dict,
    mode: str = "lstm",
    models: list[str] = None,
    demo: bool = False,
) -> None:
    base_dir = Path(config.get("_base_dir", "."))
    data_dir = base_dir / config["data"].get("data_dir", "data")
    suffix   = config["data"].get("file_suffix", "_daily_adjusted.csv")

    csvs    = sorted(data_dir.glob(f"*{suffix}"))
    if not csvs:
        print(f"[ERRO] Nenhum CSV encontrado em '{data_dir}' com sufixo '{suffix}'.")
        sys.exit(1)

    tickers = [p.name.replace(suffix, "") for p in csvs]
    print(f"\n[--all] {len(tickers)} ativo(s): {[t.upper() for t in tickers]}")

    for ticker in tickers:
        try:
            if mode == "baseline":
                run_analysis_mode(config, ticker, demo)
            elif models and len(models) > 1:
                run_multiple_models(config, ticker, models, demo)
            else:
                modelo = (models[0] if models else "LSTM").upper()
                run_model_mode(config, ticker, model_name=modelo, demo=demo)
        except Exception as e:
            print(f"\n[AVISO] Falha ao processar {ticker.upper()}: {e}")


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

def main():
    modelos_disp = list(MODEL_REGISTRY.keys())

    parser = argparse.ArgumentParser(
        description="Pipeline B3 — Dissertação PPGET/IFCE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Modelos disponíveis: {", ".join(modelos_disp)}

Exemplos:
  python main.py --asset bova11                            # LSTM (padrão)
  python main.py --asset bova11 --model gru                # modelo único
  python main.py --asset bova11 --models lstm gru mlp      # múltiplos sequencial
  python main.py --asset bova11 --models all               # todos os modelos
  python main.py --asset bova11 --models all --demo        # todos em modo rápido
  python main.py --asset bova11 --mode baseline            # RSI-LPF puro
  python main.py --all --model lstm                        # todos os CSVs, LSTM
  python main.py --all --models lstm gru                   # todos CSVs × 2 modelos
  python main.py --asset petr4 --file /caminho/dados.csv   # CSV personalizado

Saídas geradas (em resultados/):
  predictions_{{asset}}_{{modelo}}.csv    — predições do walk-forward
  metrics_{{asset}}_{{modelo}}.json       — métricas financeiras
  comparacao_{{asset}}_*.json            — tabela comparativa (--models)
  figuras/{{modelo}}/                    — gráficos por modelo
        """
    )

    parser.add_argument("--config", default="config.yaml",
                        help="Caminho do config.yaml (padrão: config.yaml)")
    parser.add_argument("--asset", default=None,
                        help="Ticker do ativo (ex: petr4, bova11). "
                             "Qualquer ticker com CSV em data/.")
    parser.add_argument("--file", default=None, metavar="CAMINHO",
                        help="Caminho explícito do CSV (sobrescreve convenção de nomes).")
    parser.add_argument("--mode", default="lstm",
                        choices=["lstm", "baseline"],
                        help="Modo de operação: lstm (qualquer modelo neural) | "
                             "baseline (RSI-LPF puro, sem rede)")
    parser.add_argument("--model", default=None,
                        metavar="MODELO",
                        help=f"Modelo único a executar. Opções: {', '.join(modelos_disp)}")
    parser.add_argument("--models", default=None,
                        nargs="+", metavar="MODELO",
                        help=f"Um ou mais modelos sequencialmente, com tabela comparativa. "
                             f"Use 'all' para todos. Opções: {', '.join(modelos_disp)} all")
    parser.add_argument("--demo", action="store_true",
                        help="Execução rápida: 2 janelas walk-forward, 5 épocas")
    parser.add_argument("--all", action="store_true",
                        help="Processa todos os CSVs em data/")

    args = parser.parse_args()

    # Carrega configuração
    base_dir = str(Path(args.config).parent.resolve())
    config   = load_config(args.config, base_dir=base_dir)

    # Cria diretórios de saída
    Path(config["output"]["figures_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["output"]["reports_dir"]).mkdir(parents=True, exist_ok=True)

    # Banner
    print("\n" + "="*60)
    print("  Sistemas Inteligentes para Alocação de Capital — B3")
    print("  Dissertação PPGET/IFCE — Pedro Wilson Félix")
    print("="*60)

    # ── Determina lista de modelos a rodar ───────────────────────────────────
    # Prioridade: --models > --model > --mode > padrão LSTM
    if args.models:
        lista_modelos = args.models       # pode conter "all"
    elif args.model:
        lista_modelos = [args.model]
    elif args.mode == "baseline":
        lista_modelos = None              # sinaliza modo baseline
    else:
        lista_modelos = ["LSTM"]          # padrão

    # ── Modo --all ───────────────────────────────────────────────────────────
    if args.all:
        modo_str = " ".join(lista_modelos) if lista_modelos else "baseline"
        print(f"  Modo  : ALL × [{modo_str.upper()}]{'  DEMO' if args.demo else ''}")
        print("="*60)
        run_all_assets(
            config,
            mode    = args.mode,
            models  = lista_modelos,
            demo    = args.demo,
        )
        return

    # ── Ativo único ──────────────────────────────────────────────────────────
    asset = (args.asset or config["data"].get("primary_asset", "bova11")).lower()

    if lista_modelos is None:
        # Modo baseline
        print(f"  Ativo : {asset.upper()}")
        print(f"  Modo  : {'DEMO ' if args.demo else ''}BASELINE (RSI-LPF)")
        if args.file:
            print(f"  CSV   : {args.file}")
        print("="*60)
        run_analysis_mode(config, asset, demo=args.demo, custom_file=args.file)

    elif len(lista_modelos) == 1 and lista_modelos[0].upper() != "ALL":
        # Modelo único
        modelo = lista_modelos[0].upper()
        print(f"  Ativo : {asset.upper()}")
        print(f"  Modelo: {'DEMO ' if args.demo else ''}{modelo}")
        if args.file:
            print(f"  CSV   : {args.file}")
        print("="*60)
        run_model_mode(
            config, asset,
            model_name  = modelo,
            demo        = args.demo,
            custom_file = args.file,
        )

    else:
        # Múltiplos modelos — tabela comparativa
        print(f"  Ativo  : {asset.upper()}")
        modelos_expandidos = list(MODEL_REGISTRY.keys()) \
            if lista_modelos[0].upper() == "ALL" \
            else [m.upper() for m in lista_modelos]
        print(f"  Modelos: {modelos_expandidos}{'  [DEMO]' if args.demo else ''}")
        if args.file:
            print(f"  CSV    : {args.file}")
        print("="*60)
        run_multiple_models(
            config, asset,
            models      = lista_modelos,
            demo        = args.demo,
            custom_file = args.file,
        )


if __name__ == "__main__":
    main()
