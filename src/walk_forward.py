"""
src/walk_forward.py
===================
Protocolo de Validação Walk-Forward (Seção 3.5.1).

Implementa a validação temporal sequencial que:
    - Respeita a causalidade (modelo treinado apenas com dados passados)
    - Avalia estabilidade em diferentes regimes de mercado
    - Simula operação real ("produção hipotética")

Esquema (Seção 3.5.1):
    Treino₁[t₀→tT] | Teste₁[tT+1→tT+h] →
    Treino₂[t₁→tT+1] | Teste₂[tT+2→tT+h+1] → ...

REFATORAÇÃO — Modelos Plugáveis:
    O loop walk-forward agora recebe um model_class (qualquer subclasse de
    BaseModel) em vez de usar LSTMTrainer diretamente. O comportamento para
    o LSTM é 100% idêntico ao original — apenas a instanciação mudou.

    Para usar um modelo diferente:
        from src.models import MODEL_REGISTRY
        run_walk_forward(df, config, model_class=MODEL_REGISTRY["GRU"])

Referências:
    Seção 3.5.1 da dissertação
    Bailey et al. (2014) — The Probability of Backtest Overfitting
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Iterator, Type
import warnings

from .features import (
    build_feature_matrix,
    create_targets,
    create_sequences,
    fit_scaler,
    apply_scaler,
)
from .models.base import BaseModel, PredictionResult
from .models import MODEL_REGISTRY


# ---------------------------------------------------------------------------
# Dataclass para resultados de cada janela
# ---------------------------------------------------------------------------

@dataclass
class WindowResult:
    """Resultado de uma janela do walk-forward."""
    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int
    model_name: str = "LSTM"

    # Predições e alvos
    dates_test: np.ndarray = field(default_factory=lambda: np.array([]))
    y_ret_true: np.ndarray = field(default_factory=lambda: np.array([]))
    y_trend_true: np.ndarray = field(default_factory=lambda: np.array([]))
    y_cls_true: np.ndarray = field(default_factory=lambda: np.array([]))

    y_ret_pred: np.ndarray = field(default_factory=lambda: np.array([]))
    y_trend_pred: np.ndarray = field(default_factory=lambda: np.array([]))
    y_cls_proba: np.ndarray = field(default_factory=lambda: np.array([]))
    decisions: np.ndarray = field(default_factory=lambda: np.array([]))
    confidence: np.ndarray = field(default_factory=lambda: np.array([]))
    rejection_mask: np.ndarray = field(default_factory=lambda: np.array([]))
    regime_guerra: np.ndarray = field(default_factory=lambda: np.array([]))

    # Métricas rápidas da janela
    accuracy_accepted: float = np.nan
    rejection_rate: float = np.nan
    tau_used: float = np.nan


# ---------------------------------------------------------------------------
# Gerador de janelas walk-forward
# ---------------------------------------------------------------------------

def walk_forward_windows(
    n_samples: int,
    train_size: int,
    test_size: int,
    step_size: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    start = 0
    while True:
        train_end = start + train_size
        test_end  = train_end + test_size
        if test_end > n_samples:
            break
        yield np.arange(start, train_end), np.arange(train_end, test_end)
        start += step_size


# ---------------------------------------------------------------------------
# Loop principal walk-forward — genérico para qualquer BaseModel
# ---------------------------------------------------------------------------

def run_walk_forward(
    df:          pd.DataFrame,
    config:      dict,
    model_class: Type[BaseModel] = None,
    verbose:     bool = True,
) -> list[WindowResult]:
    """
    Executa o protocolo completo de validação walk-forward com qualquer modelo.

    Parâmetros
    ----------
    df          : pd.DataFrame OHLCV + selic_pct, indexado por data
    config      : dict carregado do config.yaml
    model_class : subclasse de BaseModel (default: LSTMModel)
                  Exemplos: MODEL_REGISTRY["GRU"], MODEL_REGISTRY["MLP"],
                             MODEL_REGISTRY["KNN"]
    verbose     : bool — exibe progresso no terminal
    """
    if model_class is None:
        model_class = MODEL_REGISTRY["LSTM"]

    model_name = model_class.__name__.replace("Model", "")

    wf_cfg = config["walk_forward"]
    f_cfg  = config["features"]
    bt_cfg = config["backtest"]

    train_size = wf_cfg["train_size"]
    test_size  = wf_cfg["test_size"]
    step_size  = wf_cfg["step_size"]
    L          = f_cfg["window_size"]
    tau_fixed  = bt_cfg["rejection_threshold"]

    features_df = build_feature_matrix(df, config)
    targets_df  = create_targets(df, config)

    common_idx  = features_df.index.intersection(targets_df.index)
    features_df = features_df.loc[common_idx]
    targets_df  = targets_df.loc[common_idx]

    feat_arr = features_df.values.astype(np.float32)
    tgt_arr  = targets_df.values.astype(np.float32)
    dates    = features_df.index

    if "regime_guerra" in features_df.columns:
        regime_arr = features_df["regime_guerra"].values.astype(np.int8)
    else:
        regime_arr = np.zeros(len(features_df), dtype=np.int8)

    n_samples     = len(feat_arr)
    feature_names = list(features_df.columns)
    config["_n_features"] = feat_arr.shape[1]

    windows   = list(walk_forward_windows(n_samples, train_size, test_size, step_size))
    n_windows = len(windows)

    if verbose:
        print(f"\n[WalkForward] Modelo={model_name} | {n_windows} janelas | "
              f"features={len(feature_names)} | L={L}")
        print(f"[WalkForward] Treino={train_size}d | Teste={test_size}d | Passo={step_size}d")

    results: list[WindowResult] = []

    for win_id, (train_idx, test_idx) in enumerate(windows):
        feat_train = feat_arr[train_idx]
        tgt_train  = tgt_arr[train_idx]

        test_start_idx    = test_idx[0]
        test_end_idx      = test_idx[-1] + 1
        extended_test_idx = np.arange(test_start_idx - L, test_end_idx)

        feat_test      = feat_arr[extended_test_idx]
        tgt_test       = tgt_arr[extended_test_idx]
        dates_test_ext = dates[extended_test_idx]

        X_train, y_train = create_sequences(feat_train, tgt_train, L)
        X_test,  y_test  = create_sequences(feat_test,  tgt_test,  L)

        if len(X_train) < 50 or len(X_test) < 5:
            if verbose:
                print(f"  Janela {win_id+1:02d}: amostras insuficientes — pulando.")
            continue

        scaler  = fit_scaler(X_train)
        X_train = apply_scaler(X_train, scaler)
        X_test  = apply_scaler(X_test,  scaler)

        y_ret_train   = y_train[:, 0]
        y_trend_train = y_train[:, 1]
        y_cls_train   = y_train[:, 2]

        y_ret_test    = y_test[:, 0]
        y_trend_test  = y_test[:, 1]
        y_cls_test    = y_test[:, 2]

        n_val = max(20, int(0.15 * len(X_train)))
        X_tr,  X_pval  = X_train[:-n_val], X_train[-n_val:]
        y_ret_tr       = y_ret_train[:-n_val]
        y_trend_tr     = y_trend_train[:-n_val]
        y_cls_tr       = y_cls_train[:-n_val]
        y_cls_pval     = y_cls_train[-n_val:]

        n_ensemble = wf_cfg.get("n_ensemble", 3)

        best_model    = None
        best_acc_pval = -1.0

        for run_idx in range(n_ensemble):
            seed = win_id * 100 + run_idx
            _model = model_class(config)
            try:
                _model.fit(
                    X_tr, y_ret_tr, y_trend_tr, y_cls_tr,
                    X_val=X_pval,
                    y_ret_val=y_ret_train[-n_val:],
                    y_trend_val=y_trend_train[-n_val:],
                    y_cls_val=y_cls_pval,
                    seed=seed,
                )
            except Exception as e:
                import traceback
                print(f"\n[ERRO] Janela {win_id+1} run {run_idx+1} ({model_name}):")
                print(f"  {type(e).__name__}: {e}")
                traceback.print_exc()
                continue

            _pred_pval = _model.predict(X_pval, tau=tau_fixed)
            _mask_ok   = ~_pred_pval.rejection_mask
            if _mask_ok.sum() >= 5:
                _acc = (_pred_pval.y_decision[_mask_ok] == y_cls_pval[_mask_ok].astype(int)).mean()
            else:
                _acc = 0.0

            if verbose and n_ensemble > 1:
                print(f"    run {run_idx+1}/{n_ensemble} seed={seed} acc_pval={_acc:.3f}")

            if _acc > best_acc_pval:
                best_acc_pval = _acc
                best_model    = _model

            # KNN é determinístico — um run basta
            if model_name == "KNN":
                break

        if best_model is None:
            print(f"\n[AVISO] Janela {win_id+1}: todos os runs falharam — pulando.")
            continue

        # ── Calibração de Wr (Chow, 1970) ────────────────────────────────────
        # Etapa 1: calibrate_wr() varre uma grade de Wr ∈ (Wc, We) e escolhe
        # o valor que maximiza acurácia nos aceitos dentro das restrições de
        # taxa de aceitação. Atualiza config["backtest"]["Wr"] internamente.
        #
        # Etapa 2: find_optimal_tau() usa o τ já influenciado pelo Wr calibrado
        # (se use_cost_tau=True) ou faz busca empírica independente (fallback).
        #
        # Esse fluxo é idêntico ao LSTMTrainer original — agora disponível
        # para GRU, MLP e KNN via BaseModel.calibrate_wr().
        use_cost_tau = config["backtest"].get("use_cost_tau", True)
        if use_cost_tau:
            try:
                wr_result = best_model.calibrate_wr(
                    X_pval,
                    y_cls_pval.astype(int),
                    min_accept = bt_cfg.get("wr_min_accept", 0.30),
                    max_accept = bt_cfg.get("wr_max_accept", 0.90),
                    verbose    = False,   # silencia por janela; verbose geral já existe
                )
                if verbose:
                    print(f"    [Chow] Wr_opt={wr_result['Wr_opt']:.4f} "
                          f"τ={wr_result['tau_opt']:.4f} "
                          f"acc_val={wr_result['acc_opt']:.3f} "
                          f"rej={wr_result['rej_opt']:.1%}")
            except Exception as e:
                if verbose:
                    print(f"    [Chow] calibrate_wr falhou ({e}) — usando Wr do config.")

        try:
            tau_opt = best_model.find_optimal_tau(X_pval, y_cls_pval.astype(int))
        except Exception:
            tau_opt = tau_fixed

        pred_result = best_model.predict(X_test, tau=tau_opt)

        dates_aligned       = dates_test_ext[L:]
        regime_test_aligned = regime_arr[extended_test_idx][L:]

        n_accepted = (~pred_result.rejection_mask).sum()
        rej_rate   = pred_result.rejection_mask.mean()
        if n_accepted > 0:
            mask_ok = ~pred_result.rejection_mask
            acc = (pred_result.y_decision[mask_ok] == y_cls_test[mask_ok]).mean()
        else:
            acc = np.nan

        result = WindowResult(
            window_id      = win_id + 1,
            train_start    = dates[train_idx[0]],
            train_end      = dates[train_idx[-1]],
            test_start     = dates[test_idx[0]],
            test_end       = dates[test_idx[-1]],
            n_train        = len(X_train),
            n_test         = len(X_test),
            model_name     = model_name,
            dates_test     = dates_aligned,
            y_ret_true     = y_ret_test,
            y_trend_true   = y_trend_test,
            y_cls_true     = y_cls_test,
            y_ret_pred     = pred_result.y_ret_pred,
            y_trend_pred   = pred_result.y_trend_pred,
            y_cls_proba    = pred_result.y_cls_proba,
            decisions      = pred_result.y_decision,
            confidence     = pred_result.confidence,
            rejection_mask = pred_result.rejection_mask,
            regime_guerra  = regime_test_aligned,
            accuracy_accepted = acc,
            rejection_rate    = rej_rate,
            tau_used          = tau_opt,
        )
        results.append(result)

        if verbose:
            print(
                f"  Janela {win_id+1:02d}/{n_windows} [{model_name}] | "
                f"Teste: {result.test_start.date()} → {result.test_end.date()} | "
                f"Acc={acc:.3f} | Rej={rej_rate:.1%} | τ={tau_opt:.2f}"
            )

    if verbose:
        print(f"\n[WalkForward] Concluído: {len(results)} janelas processadas.")

    return results


# ---------------------------------------------------------------------------
# Consolidação dos resultados
# ---------------------------------------------------------------------------

def consolidate_results(results: list[WindowResult]) -> pd.DataFrame:
    """
    Consolida predições de todas as janelas em um único DataFrame.

    Colunas retornadas:
        date, y_ret_true, y_trend_true, y_cls_true,
        y_ret_pred, y_trend_pred, decision, confidence,
        rejected, regime_guerra, window_id, model_name
    """
    rows = []
    for r in results:
        n = len(r.dates_test)
        for i in range(n):
            rows.append({
                "date":          r.dates_test[i],
                "y_ret_true":    r.y_ret_true[i]      if i < len(r.y_ret_true)     else np.nan,
                "y_trend_true":  r.y_trend_true[i]    if i < len(r.y_trend_true)   else np.nan,
                "y_cls_true":    r.y_cls_true[i]       if i < len(r.y_cls_true)     else np.nan,
                "y_ret_pred":    r.y_ret_pred[i]       if i < len(r.y_ret_pred)     else np.nan,
                "y_trend_pred":  r.y_trend_pred[i]     if i < len(r.y_trend_pred)   else np.nan,
                "decision":      r.decisions[i]         if i < len(r.decisions)      else np.nan,
                "confidence":    r.confidence[i]        if i < len(r.confidence)     else np.nan,
                "rejected":      r.rejection_mask[i]    if i < len(r.rejection_mask) else True,
                "regime_guerra": int(r.regime_guerra[i]) if i < len(r.regime_guerra) else 0,
                "window_id":     r.window_id,
                "model_name":    r.model_name,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.set_index("date").sort_index()
    return df
