"""
src/models/knn.py
=================
Modelo KNN (K-Nearest Neighbors) — Scikit-learn.

O KNN é um estimador não-paramétrico e lazy (sem fase de treinamento real):
a predição consiste em encontrar os K vizinhos mais próximos no espaço de
features e agregar seus rótulos.

No contexto desta dissertação, serve como **baseline clássico** para
quantificar o quanto os modelos neurais (LSTM, GRU, MLP) superam uma
abordagem sem aprendizado de representação.

Adaptações para o sistema multitarefa:
    - Head-1 (retorno): KNeighborsRegressor(K, pesos='distance')
    - Head-2 (tendência): KNeighborsClassifier(K, pesos='distance') → proba
    - Head-3 (compra/venda): KNeighborsClassifier(K, pesos='distance') → proba

Para compatibilidade com o mecanismo de rejeição de Chow, as probabilidades
soft são geradas via predict_proba() dos classificadores sklearn. Quando K=1,
as probabilidades são {0, 1} — o que faz τ > 0.5 aceitar tudo. Para K≥5
com pesos por distância, as probabilidades são mais suaves e o mecanismo
de rejeição funciona de forma mais informativa.

Convenção de achatamento (idêntica ao MLP):
    X shape (N, L, d) → achatado para (N, L*d) antes de passar ao sklearn.
    O KNN não tem noção de sequência temporal — o achatamento concatena
    todos os L*d valores como features independentes.

Referências:
    Cover & Hart (1967) — Nearest Neighbor Pattern Classification
    Chow (1970) — On Optimum Recognition Error and Reject Trade-off
    Pedregosa et al. (2011) — Scikit-learn: Machine Learning in Python
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder

from .base import BaseModel, PredictionResult, apply_rejection


class KNNModel(BaseModel):
    """
    KNN multitarefa usando scikit-learn — baseline não-paramétrico.

    Três estimadores independentes, um por cabeça de saída:
        _reg_ret    : KNeighborsRegressor   → y_ret_pred
        _clf_trend  : KNeighborsClassifier  → y_trend_pred (proba)
        _clf_cls    : KNeighborsClassifier  → y_cls_proba  (compra/venda)

    A seleção de K segue a heurística √N (raiz quadrada do número de
    amostras de treino), com mínimo de 3 e máximo de 51 — ajustável
    via config.yaml na chave model_knn.n_neighbors.

    Nota sobre complexidade:
        O KNN tem O(N·d) por predição. Com N~500 amostras de treino e
        d = L*d_features ~ 30*15 = 450, o tempo de predição pode ser
        lento para datasets grandes. Considere KD-tree (algoritmo='kd_tree')
        ou Ball-tree para d < 20. Para d alto, 'brute' é geralmente mais
        rápido (sem overhead de construção da árvore).
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._reg_ret:   Optional[KNeighborsRegressor]  = None
        self._clf_trend: Optional[KNeighborsClassifier] = None
        self._clf_cls:   Optional[KNeighborsClassifier] = None
        self._n_neighbors: int = 5

    @staticmethod
    def _flatten(X: np.ndarray) -> np.ndarray:
        """Achata (N, L, d) → (N, L*d)."""
        return X.reshape(X.shape[0], -1)

    def _get_k(self, n_train: int) -> int:
        """
        Determina K:
            1. Se config define model_knn.n_neighbors → usa esse valor
            2. Senão: heurística √N com clamp [3, 51]

        A heurística √N é amplamente usada na literatura como ponto de
        partida (Duda et al., 2001). Valores ímpares evitam empates binários.
        """
        m_cfg = self.config.get("model_knn", {})
        if "n_neighbors" in m_cfg:
            k = int(m_cfg["n_neighbors"])
        else:
            k = int(np.sqrt(n_train))
            k = max(3, min(51, k))
            if k % 2 == 0:
                k += 1   # força ímpar para desempate
        return k

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
        seed:        Optional[int]        = None,   # ignorado (KNN é determinístico)
    ) -> "KNNModel":
        """
        'Treina' o KNN — na prática, apenas armazena os dados de treino.

        O parâmetro seed é aceito para compatibilidade com a interface,
        mas não tem efeito — KNN é completamente determinístico.
        """
        m_cfg = self.config.get("model_knn", {})
        X_flat  = self._flatten(X_train)
        n_train = X_flat.shape[0]

        k         = self._get_k(n_train)
        self._n_neighbors = k
        weights   = m_cfg.get("weights",   "distance")   # 'uniform' ou 'distance'
        algorithm = m_cfg.get("algorithm", "auto")        # 'auto','kd_tree','ball_tree','brute'
        metric    = m_cfg.get("metric",    "euclidean")

        self._reg_ret = KNeighborsRegressor(
            n_neighbors=k, weights=weights,
            algorithm=algorithm, metric=metric,
        )
        self._clf_trend = KNeighborsClassifier(
            n_neighbors=k, weights=weights,
            algorithm=algorithm, metric=metric,
        )
        self._clf_cls = KNeighborsClassifier(
            n_neighbors=k, weights=weights,
            algorithm=algorithm, metric=metric,
        )

        # Garante que os alvos de classificação são inteiros {0, 1}
        y_trend_int = y_trend.astype(int)
        y_cls_int   = y_cls.astype(int)

        # Evita erro do sklearn quando só há uma classe no treino
        # (pode ocorrer em janelas pequenas ou desbalanceadas)
        unique_trend = np.unique(y_trend_int)
        unique_cls   = np.unique(y_cls_int)

        self._reg_ret.fit(X_flat, y_ret.astype(np.float32))

        if len(unique_trend) < 2:
            # Classe única → prediz constante; armazena classe dominante
            self._single_trend = int(unique_trend[0])
        else:
            self._single_trend = None
            self._clf_trend.fit(X_flat, y_trend_int)

        if len(unique_cls) < 2:
            self._single_cls = int(unique_cls[0])
        else:
            self._single_cls = None
            self._clf_cls.fit(X_flat, y_cls_int)

        return self

    def predict(
        self,
        X:   np.ndarray,
        tau: Optional[float] = None,
    ) -> PredictionResult:
        if self._reg_ret is None:
            raise RuntimeError("Modelo não treinado. Chame .fit() primeiro.")
        if tau is None:
            tau = self._resolve_tau()

        X_flat = self._flatten(X).astype(np.float32)
        N      = X_flat.shape[0]

        # ── Regressão de retorno ──────────────────────────────────────────────
        y_ret_pred = self._reg_ret.predict(X_flat).astype(np.float32)

        # ── Classificação de tendência ────────────────────────────────────────
        if self._single_trend is not None:
            # Degenerado: classe única
            y_trend_pred = np.full(N, float(self._single_trend))
        else:
            y_trend_pred = self._clf_trend.predict_proba(X_flat)[:, 1]

        # ── Classificação compra/venda ────────────────────────────────────────
        if self._single_cls is not None:
            # Degenerado: todos compra ou todos venda com confiança total
            dominant = self._single_cls   # 0=venda, 1=compra
            if dominant == 0:
                y_cls_proba = np.column_stack([
                    np.ones(N), np.zeros(N)
                ]).astype(np.float32)
            else:
                y_cls_proba = np.column_stack([
                    np.zeros(N), np.ones(N)
                ]).astype(np.float32)
        else:
            # predict_proba retorna [[p_venda, p_compra]] — índices = classes
            # O sklearn ordena as classes em ordem crescente: 0, 1
            y_cls_proba = self._clf_cls.predict_proba(X_flat).astype(np.float32)
            # Garante 2 colunas mesmo que só uma classe apareça no teste
            if y_cls_proba.shape[1] == 1:
                classes = self._clf_cls.classes_
                if classes[0] == 0:
                    y_cls_proba = np.column_stack([y_cls_proba, np.zeros(N)])
                else:
                    y_cls_proba = np.column_stack([np.zeros(N), y_cls_proba])
            y_cls_proba = y_cls_proba.astype(np.float32)

        decisions, confidence, rejection_mask = apply_rejection(y_cls_proba, tau=tau)
        return PredictionResult(
            y_ret_pred=y_ret_pred,
            y_trend_pred=y_trend_pred.astype(np.float32),
            y_cls_proba=y_cls_proba,
            y_decision=decisions,
            confidence=confidence,
            rejection_mask=rejection_mask,
        )

    def get_params(self) -> dict:
        m_cfg = self.config.get("model_knn", {})
        return {
            "model_type":  "KNN",
            "n_neighbors": self._n_neighbors,
            "weights":     m_cfg.get("weights",   "distance"),
            "algorithm":   m_cfg.get("algorithm", "auto"),
            "metric":      m_cfg.get("metric",    "euclidean"),
        }
