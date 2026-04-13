"""
src/model.py
============
Modelo LSTM Multitarefa com Mecanismo de Rejeição — TensorFlow/Keras.

Implementa a arquitetura descrita na Seção 3.3.5 da dissertação:

    Entrada:   (batch, L, d)
    LSTM-1:    64 unidades  → Dropout(0.20)
    LSTM-2:    32 unidades  → último hidden state
    Dense:     16 neurônios, ReLU  (camada compartilhada)
    Head-1:    Dense(1)  linear    → retorno t+1         [MSE, α=0.30]
    Head-2:    Dense(1)  sigmoid   → tendência 30 dias   [BCE, β=0.35]
    Head-3:    Dense(2)  softmax   → compra/venda        [CCE, γ=0.35]

Função de perda multitarefa (Equação 3.4):
    L = α·MSE(ret) + β·BCE(trend) + γ·CCE(cls)

Mecanismo de rejeição (Seção 3.3.8):
    se max(p̂) >= τ → classificar   (compra=1 ou venda=0)
    se max(p̂) <  τ → rejeitar      (decisão = -1, não opera)

Referências:
    Hochreiter & Schmidhuber (1997)
    Chow (1970) — On Optimum Recognition Error and Reject Trade-off
    Rocha-Neto (2011) — SINPATCO II
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks


# Suprime logs verbosos do TensorFlow (mantém só erros)
tf.get_logger().setLevel("ERROR")


# ---------------------------------------------------------------------------
# Dataclass para resultados de predição
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    y_ret_pred:     np.ndarray
    y_trend_pred:   np.ndarray
    y_cls_proba:    np.ndarray
    y_decision:     np.ndarray
    confidence:     np.ndarray
    rejection_mask: np.ndarray


# ---------------------------------------------------------------------------
# Mecanismo de Rejeição — Chow (1970)
# ---------------------------------------------------------------------------

def tau_from_costs(Wc: float, Wr: float, We: float) -> float:
    """
    Calcula o limiar de rejeição τ a partir dos custos de classificação
    de Chow (Seção 2.10.1, pág. 33 da dissertação).

    Derivação:
        Decide C_k  se  p(C_k | x) ≥ (We − Wr) / (We − Wc)
        Rejeita     se  p(C_k | x) <  (We − Wr) / (We − Wc)

        Logo:  τ = (We − Wr) / (We − Wc)

    Parâmetros
    ----------
    Wc : custo de classificar corretamente  (tipicamente 0)
    Wr : custo de rejeitar                  (custo de dizer "não sei")
    We : custo de classificar errado        (deve ser > Wr)

    Condição obrigatória: Wc < Wr < We
        - rejeitar é pior que acertar  (Wr > Wc)
        - errar é pior que rejeitar    (We > Wr)

    Retorna
    -------
    float  τ ∈ (0, 1)

    Exemplos
    --------
    >>> tau_from_costs(0.0, 0.01, 0.05)   # padrão
    0.80
    >>> tau_from_costs(0.0, 0.02, 0.05)
    0.60
    >>> tau_from_costs(0.0, 0.04, 0.05)   # Wr próximo de We → τ baixo
    0.20
    """
    if not (Wc < Wr < We):
        raise ValueError(
            f"Custos de Chow violam Wc < Wr < We: Wc={Wc}, Wr={Wr}, We={We}"
        )
    return (We - Wr) / (We - Wc)


def apply_rejection(
    cls_proba: np.ndarray,
    tau:       float = 0.80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aplica o mecanismo de rejeição de Chow sobre as probabilidades softmax.

    Regra:
        confidence = max_i p(C_i | x)
        se confidence >= τ  → classifica  (compra=1 ou venda=0)
        se confidence <  τ  → rejeita     (decisão = -1)

    τ deve ser derivado de tau_from_costs(Wc, Wr, We) para respeitar a
    formulação teórica da dissertação (Seção 2.10.1, Equação 2.11).

    Parâmetros
    ----------
    cls_proba : np.ndarray shape (N, 2)
        Probabilidades softmax [p_venda, p_compra].
    tau : float
        Limiar de confiança — preferencialmente calculado via tau_from_costs().

    Retorna
    -------
    decisions      : np.ndarray int   — 1=compra, 0=venda, -1=rejeição
    confidence     : np.ndarray float — max(p) para cada amostra
    rejection_mask : np.ndarray bool  — True onde rejeitou
    """
    confidence     = cls_proba.max(axis=1)
    decisions      = cls_proba.argmax(axis=1).astype(int)
    rejection_mask = confidence < tau
    decisions[rejection_mask] = -1
    return decisions, confidence, rejection_mask


# ---------------------------------------------------------------------------
# Construção da arquitetura Keras (API funcional)
# ---------------------------------------------------------------------------

def build_lstm_model(
    n_features:   int,
    window_size:  int,
    lstm_units_1: int   = 64,
    lstm_units_2: int   = 32,
    dense_units:  int   = 16,
    dropout_rate: float = 0.20,
    learning_rate: float = 0.001,
    loss_weights: dict  = None,
) -> keras.Model:
    """
    Constrói o modelo LSTM multitarefa com API funcional do Keras.

    A API funcional é preferida à Subclassing API porque:
      - permite visualizar o grafo com model.summary()
      - facilita salvar/carregar com model.save()
      - é mais estável entre versões do TensorFlow

    Parâmetros
    ----------
    n_features    : número de features de entrada (d)
    window_size   : comprimento da janela temporal (L)
    lstm_units_1  : unidades da primeira camada LSTM
    lstm_units_2  : unidades da segunda camada LSTM
    dense_units   : neurônios da camada densa compartilhada
    dropout_rate  : taxa de dropout aplicada após cada LSTM
    learning_rate : taxa de aprendizado do otimizador Adam
    loss_weights  : dict com alpha, beta, gamma para perda multitarefa

    Retorna
    -------
    keras.Model compilado
    """
    if loss_weights is None:
        loss_weights = {"alpha": 0.30, "beta": 0.35, "gamma": 0.35}

    # ── Entrada ──────────────────────────────────────────────────────────────
    inputs = keras.Input(shape=(window_size, n_features), name="input_seq")

    # ── Backbone LSTM empilhado ───────────────────────────────────────────────
    # return_sequences=True  → passa sequência completa para a LSTM-2
    x = layers.LSTM(lstm_units_1, return_sequences=True, name="lstm_1")(inputs)
    x = layers.Dropout(dropout_rate, name="drop_1")(x)

    # return_sequences=False → retorna apenas o último hidden state (vetor)
    x = layers.LSTM(lstm_units_2, return_sequences=False, name="lstm_2")(x)
    x = layers.Dropout(dropout_rate, name="drop_2")(x)

    # ── Camada densa compartilhada ────────────────────────────────────────────
    shared = layers.Dense(dense_units, activation="relu", name="dense_shared")(x)

    # ── Cabeças de saída ──────────────────────────────────────────────────────
    # Head-1: previsão de retorno (regressão, sem ativação)
    out_ret   = layers.Dense(1, activation="linear",  name="head_return")(shared)

    # Head-2: classificação de tendência 30d (binário, sigmoid)
    out_trend = layers.Dense(1, activation="sigmoid", name="head_trend")(shared)

    # Head-3: compra/venda (categórico, softmax sobre 2 classes)
    out_cls   = layers.Dense(2, activation="softmax", name="head_decision")(shared)

    # ── Modelo e compilação ───────────────────────────────────────────────────
    model = keras.Model(
        inputs  = inputs,
        outputs = [out_ret, out_trend, out_cls],
        name    = "LSTM_Multitask_Rejection",
    )

    a = loss_weights["alpha"]
    b = loss_weights["beta"]
    g = loss_weights["gamma"]

    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate),
        loss = {
            "head_return":   "mse",
            "head_trend":    "binary_crossentropy",
            "head_decision": "sparse_categorical_crossentropy",
        },
        loss_weights = {
            "head_return":   a,
            "head_trend":    b,
            "head_decision": g,
        },
        metrics = {
            "head_trend":    ["accuracy"],
            "head_decision": ["accuracy"],
        },
    )
    return model


# ---------------------------------------------------------------------------
# Classe treinadora — interface unificada com o restante do pipeline
# ---------------------------------------------------------------------------

class LSTMTrainer:
    """
    Encapsula construção, treino e inferência do modelo LSTM multitarefa.

    Uso típico:
        trainer = LSTMTrainer(config)
        trainer.fit(X_train, y_ret, y_trend, y_cls, X_val, ...)
        result  = trainer.predict(X_test)
        tau_opt = trainer.find_optimal_tau(X_val, y_cls_val)
    """

    def __init__(self, config: dict):
        self.config = config
        self.model: Optional[keras.Model] = None

    def _build_model(self, n_features: int) -> keras.Model:
        m   = self.config["model"]
        t   = self.config["training"]
        cfg = self.config["features"]
        return build_lstm_model(
            n_features    = n_features,
            window_size   = cfg["window_size"],
            lstm_units_1  = m["lstm_units_1"],
            lstm_units_2  = m["lstm_units_2"],
            dense_units   = m["dense_units"],
            dropout_rate  = m["dropout_rate"],
            learning_rate = t["learning_rate"],
            loss_weights  = t["loss_weights"],
        )

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
    ) -> "LSTMTrainer":
        """
        Treina o modelo com early stopping e reduce-on-plateau.

        Parâmetros
        ----------
        X_train     : (N, L, d)
        y_ret       : (N,)   retorno log futuro
        y_trend     : (N,)   tendência binária 30d  {0, 1}
        y_cls       : (N,)   classe compra/venda     {0, 1}
        X_val, ...  : dados de validação (opcionais)
        """
        t_cfg = self.config["training"]
        _, _, d = X_train.shape

        # Reconstrói o modelo a cada janela walk-forward (pesos zerados)
        self.model = self._build_model(d)

        # Dicionários de saída compatíveis com os nomes das camadas Keras
        y_train = {
            "head_return":   y_ret.reshape(-1, 1).astype(np.float32),
            "head_trend":    y_trend.reshape(-1, 1).astype(np.float32),
            "head_decision": y_cls.astype(np.int32),
        }

        fit_kwargs = dict(
            x               = X_train.astype(np.float32),
            y               = y_train,
            epochs          = t_cfg["epochs"],
            batch_size      = t_cfg["batch_size"],
            shuffle         = True,
            verbose         = 0,
        )

        cb_list = [
            callbacks.EarlyStopping(
                monitor              = "val_loss" if X_val is not None else "loss",
                patience             = t_cfg["early_stopping_patience"],
                restore_best_weights = True,
                min_delta            = 1e-6,
            ),
            callbacks.ReduceLROnPlateau(
                monitor  = "val_loss" if X_val is not None else "loss",
                factor   = 0.5,
                patience = 5,
                min_lr   = 1e-6,
                verbose  = 0,
            ),
        ]

        if X_val is not None:
            fit_kwargs["validation_data"] = (
                X_val.astype(np.float32),
                {
                    "head_return":   y_ret_val.reshape(-1, 1).astype(np.float32),
                    "head_trend":    y_trend_val.reshape(-1, 1).astype(np.float32),
                    "head_decision": y_cls_val.astype(np.int32),
                },
            )

        fit_kwargs["callbacks"] = cb_list
        self.model.fit(**fit_kwargs)
        return self

    def predict(self, X: np.ndarray, tau: Optional[float] = None) -> PredictionResult:
        """
        Realiza predição e aplica o mecanismo de rejeição de Chow.

        O limiar τ é calculado a partir dos custos Wc, Wr, We definidos em
        config.yaml (use_cost_tau: true) ou usa rejection_threshold fixo
        (use_cost_tau: false).

        Parâmetros
        ----------
        X   : (N, L, d)
        tau : se fornecido explicitamente, sobrescreve o config

        Retorna
        -------
        PredictionResult com todos os campos preenchidos
        """
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
            y_ret_pred     = y_ret_pred,
            y_trend_pred   = y_trend_pred,
            y_cls_proba    = y_cls_proba,
            y_decision     = decisions,
            confidence     = confidence,
            rejection_mask = rejection_mask,
        )

    def _resolve_tau(self) -> float:
        """
        Resolve o limiar τ conforme a configuração:
          - use_cost_tau: true  → τ = (We − Wr) / (We − Wc)  [Chow, 1970]
          - use_cost_tau: false → τ fixo = rejection_threshold
        """
        bt = self.config["backtest"]
        if bt.get("use_cost_tau", True):
            Wc  = bt.get("Wc", 0.00)
            Wr  = bt.get("Wr", 0.01)
            We  = bt.get("We", 0.05)
            tau = tau_from_costs(Wc, Wr, We)
            print(f"[Chow] τ = (We−Wr)/(We−Wc) = ({We}−{Wr})/({We}−{Wc}) = {tau:.4f}")
        else:
            tau = bt.get("rejection_threshold", 0.60)
            print(f"[Chow] τ fixo = {tau:.4f}  (use_cost_tau: false)")
        return tau

    def find_optimal_tau(
        self,
        X_val:      np.ndarray,
        y_cls_val:  np.ndarray,
        tau_values: Optional[np.ndarray] = None,
    ) -> float:
        """
        Busca empírica do τ ótimo que maximiza acurácia sobre os aceitos.

        Complementa a abordagem teórica de Chow: enquanto tau_from_costs()
        deriva τ dos custos econômicos, este método encontra empiricamente
        o τ que maximizou acurácia no conjunto de validação — útil para
        comparar o τ teórico com o empírico e validar a escolha de Wr.

        Parâmetros
        ----------
        X_val      : (N, L, d)
        y_cls_val  : (N,) rótulos reais {0, 1}
        tau_values : grade de τ a testar (padrão: 0.50 → 0.95 em 19 pontos)

        Retorna
        -------
        float — melhor τ empírico encontrado
        """
        if tau_values is None:
            tau_values = np.linspace(0.50, 0.95, 19)
        if self.model is None:
            raise RuntimeError("Modelo não treinado.")

        outputs      = self.model.predict(X_val.astype(np.float32), verbose=0)
        cls_proba    = np.array(outputs[2])

        best_tau, best_acc = 0.60, 0.0
        for tau in tau_values:
            dec, _, mask = apply_rejection(cls_proba, tau=tau)
            accepted = ~mask
            if accepted.sum() < 10:
                continue
            acc = (dec[accepted] == y_cls_val[accepted].astype(int)).mean()
            if acc > best_acc:
                best_acc, best_tau = acc, tau

        # Exibe também o τ teórico de Chow para comparação
        tau_chow = self._resolve_tau()
        print(f"[Chow] τ teórico  = {tau_chow:.4f}  (derivado de Wc/Wr/We)")
        print(f"[Chow] τ empírico = {best_tau:.4f}  (máx. acurácia em val, acc={best_acc:.3f})")

        return float(best_tau)

    def calibrate_wr(
        self,
        X_val:      np.ndarray,
        y_cls_val:  np.ndarray,
        min_accept: float = 0.30,
        max_accept: float = 0.90,
        verbose:    bool  = True,
    ) -> dict:
        """
        Calibra Wr usando os dados de validação da janela walk-forward.

        Varre uma grade de Wr ∈ (Wc, We), calcula τ = (We−Wr)/(We−Wc) para
        cada valor e escolhe o Wr que maximiza acurácia sobre os aceitos
        dentro das restrições de taxa de aceitação.

        Após a chamada, o config é atualizado com Wr_opt para que
        _resolve_tau() use automaticamente o novo valor.

        Parâmetros
        ----------
        X_val       : (N, L, d)  dados de validação
        y_cls_val   : (N,)       rótulos reais {0, 1}
        min_accept  : fração mínima de amostras a aceitar
        max_accept  : fração máxima de amostras a aceitar
        verbose     : imprime tabela de calibração

        Retorna
        -------
        dict com Wr_opt, tau_opt, acc_opt, rej_opt, grid
        """
        from src.metrics import calibrate_wr as _calibrate_wr

        if self.model is None:
            raise RuntimeError("Modelo não treinado. Chame .fit() primeiro.")

        bt = self.config["backtest"]
        Wc = bt.get("Wc", 0.00)
        We = bt.get("We", 0.05)

        # Obtém as probabilidades softmax do modelo
        outputs   = self.model.predict(X_val.astype(np.float32), verbose=0)
        cls_proba = np.array(outputs[2])           # shape (N, 2)
        confidence = cls_proba.max(axis=1)

        result = _calibrate_wr(
            y_true     = y_cls_val,
            confidence = confidence,
            Wc         = Wc,
            We         = We,
            min_accept = min_accept,
            max_accept = max_accept,
            verbose    = verbose,
        )

        # Atualiza Wr no config para uso imediato em predict()
        self.config["backtest"]["Wr"] = result["Wr_opt"]

        return result

    def summary(self):
        """Imprime o resumo da arquitetura Keras."""
        if self.model:
            self.model.summary()
        else:
            print("Modelo não construído ainda. Chame .fit() primeiro.")


# ---------------------------------------------------------------------------
# Retrocompatibilidade: main.py e outros scripts que importam LSTMTrainer
# diretamente de src.model continuam funcionando sem alteração.
# ---------------------------------------------------------------------------
# A nova interface plugável está em src/models/ (pacote).
# Para novos usos, prefira:
#   from src.models import MODEL_REGISTRY
#   model = MODEL_REGISTRY["LSTM"](config)
