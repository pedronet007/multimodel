"""
src/models/mlp.py
=================
Modelo MLP (Multilayer Perceptron) Multitarefa — TensorFlow/Keras.

O MLP não é um modelo sequencial — ele não possui memória entre passos
temporais. Para manter a interface (N, L, d) compatível com o walk_forward,
a janela temporal L é achatada: cada amostra vira um vetor de L×d features.

Isso tem consequências teóricas importantes:
    - O modelo aprende relações entre features de instantes diferentes,
      mas não aprende dependências de ordem (t-2 antes de t-1 antes de t).
    - Servem como baseline "forte" para avaliar se a sequencialidade
      do LSTM/GRU agrega valor além de simplesmente ter mais features.

Arquitetura:
    Entrada:   (batch, L*d)  — achatado
    Dense-1:   128 neurônios, ReLU → BatchNorm → Dropout(0.30)
    Dense-2:    64 neurônios, ReLU → BatchNorm → Dropout(0.20)
    Dense-3:    32 neurônios, ReLU (camada compartilhada)
    Head-1:    Dense(1)  linear    → retorno t+1         [MSE,  α=0.30]
    Head-2:    Dense(1)  sigmoid   → tendência 30 dias   [BCE,  β=0.35]
    Head-3:    Dense(2)  softmax   → compra/venda        [CCE,  γ=0.35]

Referências:
    Rumelhart et al. (1986) — Learning Representations by Backpropagating Errors
    Ioffe & Szegedy (2015)  — Batch Normalization
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


class MLPModel(BaseModel):
    """
    MLP multitarefa — baseline não-sequencial para comparação com LSTM/GRU.

    Recebe X com shape (N, L, d) e achata para (N, L*d) antes de processar.
    Todas as 3 cabeças de saída são idênticas às dos modelos recorrentes,
    permitindo comparação direta das métricas de backtest.
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.model: Optional[keras.Model] = None

    def _build_model(self, n_features_flat: int) -> keras.Model:
        """
        Constrói a arquitetura MLP.

        Parâmetros
        ----------
        n_features_flat : int — L * d (dimensão após achatamento)
        """
        m_cfg = self.config.get("model_mlp", self.config.get("model", {}))

        units_1       = m_cfg.get("mlp_units_1",   128)
        units_2       = m_cfg.get("mlp_units_2",    64)
        units_shared  = m_cfg.get("dense_units",    32)
        dropout_1     = m_cfg.get("mlp_dropout_1", 0.30)
        dropout_2     = m_cfg.get("mlp_dropout_2", 0.20)
        learning_rate = m_cfg.get("learning_rate",  0.001)
        lw            = m_cfg.get("loss_weights", {"alpha": 0.30, "beta": 0.35, "gamma": 0.35})

        inputs = keras.Input(shape=(n_features_flat,), name="input_flat")

        # Camadas densas com BatchNorm — importante para estabilizar o MLP
        # com features heterogêneas (preços, RSI, volume) na mesma escala
        x = layers.Dense(units_1, activation="relu", name="dense_1")(inputs)
        x = layers.BatchNormalization(name="bn_1")(x)
        x = layers.Dropout(dropout_1, name="drop_1")(x)

        x = layers.Dense(units_2, activation="relu", name="dense_2")(x)
        x = layers.BatchNormalization(name="bn_2")(x)
        x = layers.Dropout(dropout_2, name="drop_2")(x)

        shared    = layers.Dense(units_shared, activation="relu", name="dense_shared")(x)
        out_ret   = layers.Dense(1, activation="linear",  name="head_return")(shared)
        out_trend = layers.Dense(1, activation="sigmoid", name="head_trend")(shared)
        out_cls   = layers.Dense(2, activation="softmax", name="head_decision")(shared)

        model = keras.Model(
            inputs=inputs,
            outputs=[out_ret, out_trend, out_cls],
            name="MLP_Multitask",
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

    @staticmethod
    def _flatten(X: np.ndarray) -> np.ndarray:
        """Achata (N, L, d) → (N, L*d)."""
        N = X.shape[0]
        return X.reshape(N, -1)

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
    ) -> "MLPModel":
        if seed is not None:
            tf.random.set_seed(seed)
            np.random.seed(seed)

        t_cfg = self.config["training"]
        X_flat = self._flatten(X_train)
        self.model = self._build_model(X_flat.shape[1])

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
            x=X_flat.astype(np.float32),
            y=y_train_dict,
            epochs=t_cfg["epochs"],
            batch_size=t_cfg["batch_size"],
            shuffle=True,
            verbose=0,
            callbacks=cb_list,
        )

        if X_val is not None:
            fit_kwargs["validation_data"] = (
                self._flatten(X_val).astype(np.float32),
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

        X_flat   = self._flatten(X).astype(np.float32)
        outputs  = self.model.predict(X_flat, verbose=0)

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
        m_cfg = self.config.get("model_mlp", self.config.get("model", {}))
        return {
            "model_type":   "MLP",
            "mlp_units_1":  m_cfg.get("mlp_units_1",  128),
            "mlp_units_2":  m_cfg.get("mlp_units_2",   64),
            "dense_units":  m_cfg.get("dense_units",   32),
            "dropout_1":    m_cfg.get("mlp_dropout_1", 0.30),
            "dropout_2":    m_cfg.get("mlp_dropout_2", 0.20),
        }

    def find_optimal_tau(
        self,
        X_val:      np.ndarray,
        y_cls_val:  np.ndarray,
        tau_values: Optional[np.ndarray] = None,
    ) -> float:
        if tau_values is None:
            tau_values = np.linspace(0.50, 0.95, 19)
        if self.model is None:
            raise RuntimeError("Modelo não treinado.")

        X_flat    = self._flatten(X_val).astype(np.float32)
        outputs   = self.model.predict(X_flat, verbose=0)
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
