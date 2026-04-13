"""
src/models/gru.py
=================
Modelo GRU (Gated Recurrent Unit) Multitarefa — TensorFlow/Keras.

A GRU é uma variante simplificada da LSTM proposta por Cho et al. (2014).
Usa apenas dois gates (reset e update) em vez de três (input, forget, output),
resultando em ~25% menos parâmetros e treinamento mais rápido com desempenho
comparável em séries temporais financeiras de horizonte curto.

Arquitetura (espelhada à LSTM para comparação justa):
    Entrada:   (batch, L, d)
    GRU-1:     64 unidades  → Dropout(0.20)
    GRU-2:     32 unidades  → último hidden state
    Dense:     16 neurônios, ReLU  (camada compartilhada)
    Head-1:    Dense(1)  linear    → retorno t+1         [MSE,  α=0.30]
    Head-2:    Dense(1)  sigmoid   → tendência 30 dias   [BCE,  β=0.35]
    Head-3:    Dense(2)  softmax   → compra/venda        [CCE,  γ=0.35]

Referências:
    Cho et al. (2014) — Learning Phrase Representations using RNN Encoder-Decoder
    Chung et al. (2014) — Empirical Evaluation of Gated Recurrent Neural Networks
    Chow (1970) — On Optimum Recognition Error and Reject Trade-off
"""

from __future__ import annotations

import numpy as np
from typing import Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

from .base import BaseModel, PredictionResult, apply_rejection

tf.get_logger().setLevel("ERROR")


class GRUModel(BaseModel):
    """
    GRU empilhada com 3 cabeças de saída — arquitetura paralela ao LSTM.

    Diferença estrutural chave em relação ao LSTM:
        - Não possui cell state (c_t) separado do hidden state (h_t)
        - Gate de reset: decide quanto do estado anterior esquecer
        - Gate de update: decide quanto do novo estado aceitar
        → Menos parâmetros, convergência mais rápida, similar em qualidade
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.model: Optional[keras.Model] = None

    def _build_model(self, n_features: int) -> keras.Model:
        m_cfg = self.config.get("model_gru", self.config.get("model", {}))
        L     = self.config["features"]["window_size"]

        gru_units_1   = m_cfg.get("gru_units_1",   64)
        gru_units_2   = m_cfg.get("gru_units_2",   32)
        dense_units   = m_cfg.get("dense_units",   16)
        dropout_rate  = m_cfg.get("dropout_rate",  0.20)
        learning_rate = m_cfg.get("learning_rate", 0.001)
        lw            = m_cfg.get("loss_weights", {"alpha": 0.30, "beta": 0.35, "gamma": 0.35})

        inputs = keras.Input(shape=(L, n_features), name="input_seq")

        # GRU empilhada — mesma topologia da LSTM para comparação controlada
        x = layers.GRU(gru_units_1, return_sequences=True, name="gru_1")(inputs)
        x = layers.Dropout(dropout_rate, name="drop_1")(x)
        x = layers.GRU(gru_units_2, return_sequences=False, name="gru_2")(x)
        x = layers.Dropout(dropout_rate, name="drop_2")(x)

        shared    = layers.Dense(dense_units, activation="relu", name="dense_shared")(x)
        out_ret   = layers.Dense(1, activation="linear",  name="head_return")(shared)
        out_trend = layers.Dense(1, activation="sigmoid", name="head_trend")(shared)
        out_cls   = layers.Dense(2, activation="softmax", name="head_decision")(shared)

        model = keras.Model(
            inputs=inputs,
            outputs=[out_ret, out_trend, out_cls],
            name="GRU_Multitask",
        )
        a, b, g = lw["alpha"], lw["beta"], lw["gamma"]
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss={
                "head_return":   "mse",
                "head_trend":    "binary_crossentropy",
                "head_decision": "sparse_categorical_crossentropy",
            },
            loss_weights={"head_return": a, "head_trend": b, "head_decision": g},
            metrics={"head_trend": ["accuracy"], "head_decision": ["accuracy"]},
        )
        return model

    def fit(
        self,
        X_train:     np.ndarray,
        y_ret:       np.ndarray,
        y_trend:     np.ndarray,
        y_cls:       np.ndarray,
        X_val:       Optional[np.ndarray] = None,
        y_ret_val:   Optional[np.ndarray] = None,
        y_trend_val: Optional[np.ndarray] = None,
        y_cls_val:   Optional[np.ndarray] = None,
        seed:        Optional[int]        = None,
    ) -> "GRUModel":
        if seed is not None:
            tf.random.set_seed(seed)
            np.random.seed(seed)

        t_cfg = self.config["training"]
        _, _, d = X_train.shape
        self.model = self._build_model(d)

        y_train_dict = {
            "head_return":   y_ret.reshape(-1, 1).astype(np.float32),
            "head_trend":    y_trend.reshape(-1, 1).astype(np.float32),
            "head_decision": y_cls.astype(np.int32),
        }

        monitor = "val_loss" if X_val is not None else "loss"
        cb_list = [
            callbacks.EarlyStopping(
                monitor=monitor,
                patience=t_cfg["early_stopping_patience"],
                restore_best_weights=True,
                min_delta=1e-6,
            ),
            callbacks.ReduceLROnPlateau(
                monitor=monitor, factor=0.5, patience=5, min_lr=1e-6, verbose=0,
            ),
        ]


        fit_kwargs = dict(
            x=X_train.astype(np.float32),
            y=y_train_dict,
            epochs=t_cfg["epochs"],
            batch_size=t_cfg["batch_size"],
            shuffle=True,
            verbose=0,
            callbacks=cb_list,
        )

        if X_val is not None:
            fit_kwargs["validation_data"] = (
                X_val.astype(np.float32),
                {
                    "head_return":   y_ret_val.reshape(-1, 1).astype(np.float32),
                    "head_trend":    y_trend_val.reshape(-1, 1).astype(np.float32),
                    "head_decision": y_cls_val.astype(np.int32),
                },
            )

        self.model.fit(**fit_kwargs)
        return self

    def predict(
        self,
        X:   np.ndarray,
        tau: Optional[float] = None,
    ) -> PredictionResult:
        if self.model is None:
            raise RuntimeError("Modelo não treinado. Chame .fit() primeiro.")
        if tau is None:
            tau = self._resolve_tau()

        outputs = self.model.predict(X.astype(np.float32), verbose=0)
        y_ret_pred   = np.array(outputs[0]).squeeze()
        y_trend_pred = np.array(outputs[1]).squeeze()
        y_cls_proba  = np.array(outputs[2])

        decisions, confidence, rejection_mask = apply_rejection(y_cls_proba, tau=tau)
        return PredictionResult(
            y_ret_pred=y_ret_pred,
            y_trend_pred=y_trend_pred,
            y_cls_proba=y_cls_proba,
            y_decision=decisions,
            confidence=confidence,
            rejection_mask=rejection_mask,
        )

    def get_params(self) -> dict:
        m_cfg = self.config.get("model_gru", self.config.get("model", {}))
        return {
            "model_type":   "GRU",
            "gru_units_1":  m_cfg.get("gru_units_1",  64),
            "gru_units_2":  m_cfg.get("gru_units_2",  32),
            "dense_units":  m_cfg.get("dense_units",  16),
            "dropout_rate": m_cfg.get("dropout_rate", 0.20),
        }

    def find_optimal_tau(
        self,
        X_val:      np.ndarray,
        y_cls_val:  np.ndarray,
        tau_values: Optional[np.ndarray] = None,
    ) -> float:
        """Versão otimizada: inferência única + varredura de τ."""
        if tau_values is None:
            tau_values = np.linspace(0.50, 0.95, 19)
        if self.model is None:
            raise RuntimeError("Modelo não treinado.")

        outputs   = self.model.predict(X_val.astype(np.float32), verbose=0)
        cls_proba = np.array(outputs[2])

        tau_fixed = self.config["backtest"].get("rejection_threshold", 0.60)
        best_tau, best_acc = tau_fixed, 0.0

        for tau in tau_values:
            dec, _, mask = apply_rejection(cls_proba, tau=tau)
            accepted = ~mask
            if accepted.sum() < 10:
                continue
            acc = (dec[accepted] == y_cls_val[accepted].astype(int)).mean()
            if acc > best_acc:
                best_acc, best_tau = acc, tau

        return float(best_tau)
