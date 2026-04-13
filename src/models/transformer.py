"""
src/models/transformer.py
==========================
Transformer Encoder Multitarefa para Séries Financeiras.

Baseado em: Vaswani et al. (2017) — Attention Is All You Need.
Adaptado para previsão direcional em séries temporais financeiras.

Arquitetura (encoder-only):
    Entrada:   (batch, L, d)  — L passos temporais, d features
    ↓
    Projeção linear: d → d_model
    ↓
    Positional Encoding (seno/cosseno fixo, Seção 3.5 do artigo)
    ↓
    N × TransformerEncoderLayer:
        ├── Multi-Head Self-Attention (h cabeças)
        │       Attention(Q,K,V) = softmax(QK^T / √d_k) V
        ├── Add & LayerNorm  (conexão residual)
        ├── Feed-Forward: FFN(x) = max(0, xW₁+b₁)W₂+b₂
        └── Add & LayerNorm
    ↓
    Global Average Pooling (L passos → 1 vetor)
    ↓
    Dense(d_model, ReLU)  — camada compartilhada
    ↓
    Head-1: Dense(1, linear)   → retorno t+1         [MSE,  α=0.30]
    Head-2: Dense(1, sigmoid)  → tendência 30 dias   [BCE,  β=0.35]
    Head-3: Dense(2, softmax)  → compra/venda        [CCE,  γ=0.35]

Por que encoder-only (sem decoder)?
    O decoder do artigo original é para geração autoregressiva de
    sequências (tradução). Na previsão financeira queremos um único
    vetor de saída por janela — o encoder captura as dependências
    temporais entre os L dias e produz representações contextualizadas.
    O pooling final agrega esses L vetores em uma decisão única.

Por que não usar LSTM?
    O LSTM processa a sequência passo a passo (O(L) operações
    sequenciais). O Transformer processa todos os L passos em
    paralelo via atenção, capturando dependências de longo alcance
    (ex: padrões entre o dia t e t-20) que o LSTM pode perder por
    gradientes que decaem.

Referências:
    Vaswani et al. (2017) — Attention Is All You Need. NeurIPS.
    Chow (1970) — On Optimum Recognition Error and Reject Trade-off.
    Li et al. (2019) — Enhancing the Locality and Breaking the Memory
                       Bottleneck of Transformer on Time Series Forecasting.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

from .base import BaseModel, PredictionResult, apply_rejection

tf.get_logger().setLevel("ERROR")


# ---------------------------------------------------------------------------
# Positional Encoding — Seção 3.5 de Vaswani et al. (2017)
# ---------------------------------------------------------------------------

def positional_encoding(max_len: int, d_model: int) -> tf.Tensor:
    """
    Codificação posicional por seno/cosseno (fixa, não aprendida).

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    O artigo usa essa formulação porque para qualquer offset fixo k,
    PE(pos+k) pode ser expresso como transformação linear de PE(pos),
    permitindo que o modelo aprenda atenção por posição relativa.

    Parâmetros
    ----------
    max_len : comprimento máximo da sequência (L)
    d_model : dimensão do modelo

    Retorna
    -------
    tf.Tensor shape (1, max_len, d_model)
    """
    positions = np.arange(max_len)[:, np.newaxis]          # (L, 1)
    dims      = np.arange(d_model)[np.newaxis, :]           # (1, d_model)
    angles    = positions / np.power(10000, (2 * (dims // 2)) / d_model)

    # Seno nas dimensões pares, cosseno nas ímpares
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])

    return tf.cast(angles[np.newaxis, :, :], dtype=tf.float32)  # (1, L, d_model)


# ---------------------------------------------------------------------------
# Bloco Transformer Encoder — Figura 1 esquerda do artigo
# ---------------------------------------------------------------------------

class TransformerEncoderBlock(layers.Layer):
    """
    Um bloco do encoder do Transformer (Vaswani et al., 2017).

    Estrutura:
        x → MultiHeadAttention(x,x,x) → Add+LayerNorm → FFN → Add+LayerNorm

    O mecanismo de self-attention (Q=K=V=x) permite que cada posição
    temporal "olhe" para todas as outras posições da janela — capturando
    dependências como "o RSI de 5 dias atrás influencia a decisão hoje".

    Parâmetros
    ----------
    d_model     : dimensão interna do modelo
    num_heads   : número de cabeças de atenção (h no artigo)
    dff         : dimensão da camada feed-forward interna
    dropout_rate: taxa de dropout aplicada após atenção e FFN
    """

    def __init__(
        self,
        d_model:      int,
        num_heads:    int,
        dff:          int,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Multi-Head Self-Attention
        # d_k = d_v = d_model / num_heads (conforme artigo, h=8, d_k=64)
        self.mha = layers.MultiHeadAttention(
            num_heads   = num_heads,
            key_dim     = d_model // num_heads,
            dropout     = dropout_rate,
            name        = "multi_head_attention",
        )

        # Feed-Forward: FFN(x) = max(0, xW₁+b₁)W₂+b₂
        self.ffn = keras.Sequential([
            layers.Dense(dff,     activation="relu", name="ffn_1"),
            layers.Dense(d_model, activation=None,   name="ffn_2"),
        ], name="feed_forward")

        # Layer Normalization (Add & Norm após cada sublayer)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6, name="ln_1")
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6, name="ln_2")

        # Dropout sobre as saídas de cada sublayer
        self.dropout1 = layers.Dropout(dropout_rate, name="drop_attn")
        self.dropout2 = layers.Dropout(dropout_rate, name="drop_ffn")

    def call(self, x, training=False):
        # ── Self-Attention + Add & Norm ──────────────────────────────────────
        attn_out = self.mha(x, x, x, training=training)   # Q=K=V=x
        attn_out = self.dropout1(attn_out, training=training)
        x = self.layernorm1(x + attn_out)                  # conexão residual

        # ── Feed-Forward + Add & Norm ────────────────────────────────────────
        ffn_out = self.ffn(x, training=training)
        ffn_out = self.dropout2(ffn_out, training=training)
        x = self.layernorm2(x + ffn_out)                   # conexão residual

        return x


# ---------------------------------------------------------------------------
# Modelo completo — Transformer Encoder Multitarefa
# ---------------------------------------------------------------------------

class TransformerModel(BaseModel):
    """
    Transformer Encoder Multitarefa para decisão financeira.

    Configuração padrão (conservadora para séries financeiras):
        d_model   = 64   (artigo usa 512 para NLP — muito grande para 15 features)
        num_heads = 4    (artigo usa 8  — d_k = 64/4 = 16)
        num_layers= 2    (artigo usa 6  — 2 suficiente para L=30)
        dff       = 128  (artigo usa 2048 — proporcional ao d_model menor)
        dropout   = 0.1

    Hiperparâmetros no config.yaml (seção model_transformer):
        d_model, num_heads, num_layers, dff, dropout_rate, learning_rate
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.model: Optional[keras.Model] = None
        self._pe_cache: dict = {}   # cache do positional encoding por tamanho

    def _get_pe(self, L: int, d_model: int) -> tf.Tensor:
        key = (L, d_model)
        if key not in self._pe_cache:
            self._pe_cache[key] = positional_encoding(L, d_model)
        return self._pe_cache[key]

    def _build_model(self, n_features: int) -> keras.Model:
        cfg = self.config.get("model_transformer", {})
        L            = self.config["features"]["window_size"]
        d_model      = cfg.get("d_model",      64)
        num_heads    = cfg.get("num_heads",      4)
        num_layers   = cfg.get("num_layers",     2)
        dff          = cfg.get("dff",          128)
        dropout_rate = cfg.get("dropout_rate", 0.1)
        lr           = cfg.get("learning_rate", 0.001)
        lw           = cfg.get("loss_weights",
                                {"alpha": 0.30, "beta": 0.35, "gamma": 0.35})

        # Garante que d_model é divisível por num_heads
        if d_model % num_heads != 0:
            d_model = (d_model // num_heads) * num_heads

        # ── Entrada ──────────────────────────────────────────────────────────
        inputs = keras.Input(shape=(L, n_features), name="input_seq")

        # ── Projeção linear: n_features → d_model ────────────────────────────
        # Necessário porque o positional encoding espera d_model dimensões
        x = layers.Dense(d_model, name="input_projection")(inputs)

        # ── Positional Encoding ───────────────────────────────────────────────
        # Adicionado como constante (não aprendido), conforme artigo
        pe = self._get_pe(L, d_model)
        x  = x + pe   # broadcast: (batch, L, d_model) + (1, L, d_model)

        x = layers.Dropout(dropout_rate, name="input_dropout")(x)

        # ── N blocos Transformer Encoder ─────────────────────────────────────
        for i in range(num_layers):
            x = TransformerEncoderBlock(
                d_model      = d_model,
                num_heads    = num_heads,
                dff          = dff,
                dropout_rate = dropout_rate,
                name         = f"encoder_block_{i+1}",
            )(x)

        # ── Global Average Pooling ────────────────────────────────────────────
        # Agrega os L vetores de contexto em um único vetor de decisão.
        # Alternativa: usar só o último token (como LSTM), mas GAP é mais
        # estável e aproveita toda a sequência.
        x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)

        # ── Camada densa compartilhada ────────────────────────────────────────
        shared = layers.Dense(d_model, activation="relu", name="dense_shared")(x)
        shared = layers.Dropout(dropout_rate, name="shared_dropout")(shared)

        # ── Cabeças de saída (idênticas ao LSTM/GRU) ─────────────────────────
        out_ret   = layers.Dense(1, activation="linear",  name="head_return")(shared)
        out_trend = layers.Dense(1, activation="sigmoid", name="head_trend")(shared)
        out_cls   = layers.Dense(2, activation="softmax", name="head_decision")(shared)

        model = keras.Model(
            inputs=inputs,
            outputs=[out_ret, out_trend, out_cls],
            name="Transformer_Multitask",
        )

        a, b, g = lw["alpha"], lw["beta"], lw["gamma"]
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss={
                "head_return":   "mse",
                "head_trend":    "binary_crossentropy",
                "head_decision": "sparse_categorical_crossentropy",
            },
            loss_weights={"head_return": a, "head_trend": b, "head_decision": g},
            metrics={
                "head_trend":    ["accuracy"],
                "head_decision": ["accuracy"],
            },
        )
        return model

    # ── Interface BaseModel ───────────────────────────────────────────────────

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
    ) -> "TransformerModel":
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
        cfg = self.config.get("model_transformer", {})
        return {
            "model_type":   "Transformer",
            "d_model":      cfg.get("d_model",      64),
            "num_heads":    cfg.get("num_heads",      4),
            "num_layers":   cfg.get("num_layers",     2),
            "dff":          cfg.get("dff",          128),
            "dropout_rate": cfg.get("dropout_rate", 0.1),
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

    def summary(self):
        if self.model:
            self.model.summary()
        else:
            print("Modelo não construído ainda.")
