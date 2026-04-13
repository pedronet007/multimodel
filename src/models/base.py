"""
src/models/base.py
==================
Contrato (interface) que todos os modelos devem implementar.

Todos os modelos do sistema herdam desta classe e implementam os métodos
abstratos fit() e predict(). Isso garante que walk_forward.py e backtest.py
funcionem sem saber qual modelo está sendo usado — princípio OCP do SOLID.

Referências:
    Gamma et al. (1994) — Design Patterns: Strategy Pattern
    Martin (2003) — Agile Software Development: OCP (Open/Closed Principle)
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Dataclass unificada de predição — usada por TODOS os modelos
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """
    Resultado padronizado de predição, independente do modelo.

    Campos
    ------
    y_ret_pred     : (N,) predição de retorno logarítmico t+1
    y_trend_pred   : (N,) probabilidade de tendência de alta 30d ∈ [0,1]
    y_cls_proba    : (N, 2) probabilidades [p_venda, p_compra]
    y_decision     : (N,) inteiros — 1=compra, 0=venda, -1=rejeição
    confidence     : (N,) max(p_cls) para cada amostra
    rejection_mask : (N,) bool — True onde o modelo rejeitou (conf < τ)

    Nota sobre modelos sem saída probabilística (ex: KNN em modo hard):
        y_cls_proba será uma versão sintética: [[1-p, p]] onde p ∈ {0,1}.
        y_trend_pred pode ser igual ao sinal de y_ret_pred se o modelo não
        tiver cabeça de tendência separada.
    """
    y_ret_pred:     np.ndarray
    y_trend_pred:   np.ndarray
    y_cls_proba:    np.ndarray
    y_decision:     np.ndarray
    confidence:     np.ndarray
    rejection_mask: np.ndarray


# ---------------------------------------------------------------------------
# Função de rejeição de Chow — compartilhada por todos os modelos
# ---------------------------------------------------------------------------

def apply_rejection(
    cls_proba: np.ndarray,
    tau:       float = 0.80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aplica o mecanismo de rejeição de Chow (1970) sobre probabilidades.

    Regra:
        confidence = max_i p(C_i | x)
        se confidence >= τ  → classifica  (compra=1 ou venda=0)
        se confidence <  τ  → rejeita     (decisão = -1)

    Parâmetros
    ----------
    cls_proba : (N, 2) — [p_venda, p_compra]
    tau       : limiar de confiança ∈ (0, 1)

    Retorna
    -------
    decisions      : (N,) int    — 1=compra, 0=venda, -1=rejeição
    confidence     : (N,) float  — max(p) por amostra
    rejection_mask : (N,) bool   — True onde rejeitou

    Referência: Chow (1970) — On Optimum Recognition Error and Reject Trade-off
    """
    confidence     = cls_proba.max(axis=1)
    decisions      = cls_proba.argmax(axis=1).astype(int)
    rejection_mask = confidence < tau
    decisions[rejection_mask] = -1
    return decisions, confidence, rejection_mask


# ---------------------------------------------------------------------------
# Classe base abstrata
# ---------------------------------------------------------------------------

class BaseModel(ABC):
    """
    Interface comum a todos os modelos do sistema.

    Subclasses devem implementar obrigatoriamente:
        fit()     — treina o modelo com dados de entrada
        predict() — gera PredictionResult para novos dados

    Subclasses podem sobrescrever opcionalmente:
        get_params()         — retorna hiperparâmetros como dict
        find_optimal_tau()   — busca empírica do τ ótimo (padrão: grade simples)

    Convenção de formato de entrada (X):
        Shape (N, L, d) — N amostras, L passos temporais, d features.
        Modelos que não usam janelas temporais (ex: MLP, KNN) recebem X
        com L=1 ou achatam para (N, d) internamente — o contrato externo
        mantém o shape (N, L, d) para que walk_forward.py não precise
        se preocupar com isso.

    Convenção de formato de saída (y):
        y_ret   : (N,) float32 — retorno logarítmico
        y_trend : (N,) float32 — tendência binária {0, 1}
        y_cls   : (N,) int32   — classe compra/venda {0, 1}
    """

    def __init__(self, config: dict) -> None:
        """
        Parâmetros
        ----------
        config : dict
            Configuração global do sistema (lida do config.yaml).
            Cada subclasse lê as chaves relevantes para si.
        """
        self.config = config

    # ── Métodos obrigatórios ──────────────────────────────────────────────────

    @abstractmethod
    def fit(
        self,
        X_train:     np.ndarray,          # (N, L, d)
        y_ret:       np.ndarray,          # (N,)
        y_trend:     np.ndarray,          # (N,)
        y_cls:       np.ndarray,          # (N,)
        X_val:       Optional[np.ndarray] = None,
        y_ret_val:   Optional[np.ndarray] = None,
        y_trend_val: Optional[np.ndarray] = None,
        y_cls_val:   Optional[np.ndarray] = None,
        seed:        Optional[int]        = None,
    ) -> "BaseModel":
        """
        Treina o modelo.

        Deve retornar self para permitir encadeamento:
            model.fit(...).predict(X_test)
        """
        ...

    @abstractmethod
    def predict(
        self,
        X:   np.ndarray,            # (N, L, d)
        tau: Optional[float] = None,
    ) -> PredictionResult:
        """
        Gera predições e aplica mecanismo de rejeição de Chow.

        O τ deve ser resolvido internamente se não fornecido explicitamente,
        lendo config["backtest"]["rejection_threshold"] como fallback.
        """
        ...

    # ── Métodos opcionais com implementação padrão ────────────────────────────

    def get_params(self) -> dict:
        """
        Retorna hiperparâmetros relevantes do modelo como dict.

        Usado para logging e para comparação no painel Streamlit.
        Subclasses devem sobrescrever para expor seus próprios parâmetros.
        """
        return {"model_type": self.__class__.__name__}

    def find_optimal_tau(
        self,
        X_val:      np.ndarray,
        y_cls_val:  np.ndarray,
        tau_values: Optional[np.ndarray] = None,
    ) -> float:
        """
        Busca empírica do τ ótimo que maximiza acurácia nos aceitos.

        Implementação padrão baseada em grade — funciona para qualquer modelo
        que implemente predict(). Subclasses com saída probabilística nativa
        (LSTM, GRU) podem sobrescrever com versões mais eficientes.

        Parâmetros
        ----------
        X_val      : (N, L, d)
        y_cls_val  : (N,) rótulos {0, 1}
        tau_values : grade de τ a testar (padrão: 0.50 → 0.95, 19 pontos)

        Retorna
        -------
        float — melhor τ encontrado (mínimo de 0.55 para evitar τ degenerado)
        """
        if tau_values is None:
            tau_values = np.linspace(0.50, 0.95, 19)

        tau_fixed = self.config["backtest"].get("rejection_threshold", 0.60)
        best_tau, best_acc = tau_fixed, 0.0

        for tau in tau_values:
            result = self.predict(X_val, tau=tau)
            accepted = ~result.rejection_mask
            if accepted.sum() < 10:
                continue
            acc = (result.y_decision[accepted] == y_cls_val[accepted].astype(int)).mean()
            if acc > best_acc:
                best_acc, best_tau = acc, tau

        return float(best_tau)

    def calibrate_wr(
        self,
        X_val:      np.ndarray,
        y_cls_val:  np.ndarray,
        min_accept: float = 0.30,
        max_accept: float = 0.90,
        verbose:    bool  = False,
    ) -> dict:
        """
        Calibra Wr empiricamente sobre os dados de validação da janela.

        Implementação genérica baseada em confidence scores — funciona para
        qualquer modelo que implemente predict() com saída probabilística.

        Algoritmo (Seção 2.10.1 da dissertação):
            Para cada Wr candidato em grade [Wc+ε, We−ε]:
                τ = (We − Wr) / (We − Wc)      ← fórmula de Chow
                aceitos = amostras com confidence ≥ τ
                acurácia = % acertos nos aceitos
            Escolhe Wr que maximiza acurácia, respeitando:
                min_accept ≤ fração_aceita ≤ max_accept

        Após a chamada, atualiza config["backtest"]["Wr"] com o Wr ótimo,
        de modo que o próximo _resolve_tau() use o valor calibrado.

        Parâmetros
        ----------
        X_val      : (N, L, d) — dados de validação
        y_cls_val  : (N,) — rótulos reais {0, 1}
        min_accept : fração mínima de amostras aceitas (evita τ alto demais)
        max_accept : fração máxima de amostras aceitas (evita τ baixo demais)
        verbose    : exibe tabela de calibração no terminal

        Retorna
        -------
        dict com Wr_opt, tau_opt, acc_opt, rej_opt, grid

        Referência: Chow (1970) — On Optimum Recognition Error and Reject Trade-off
        """
        from src.metrics import calibrate_wr as _calibrate_wr

        bt = self.config["backtest"]
        Wc = bt.get("Wc", 0.00)
        We = bt.get("We", 0.05)

        # Obtém confidence scores via predict() com τ=0 (aceita tudo)
        # para ter o espectro completo de probabilidades sem filtro
        result_full = self.predict(X_val, tau=0.0)
        confidence  = result_full.confidence          # max(p̂) por amostra

        result = _calibrate_wr(
            y_true     = y_cls_val.astype(int),
            confidence = confidence,
            Wc         = Wc,
            We         = We,
            min_accept = min_accept,
            max_accept = max_accept,
            verbose    = verbose,
        )

        # Atualiza Wr no config → _resolve_tau() passa a usar o valor calibrado
        self.config["backtest"]["Wr"] = result["Wr_opt"]

        return result

    def _resolve_tau(self) -> float:
        """
        Resolve o limiar τ padrão a partir da configuração.

        Ordem de prioridade:
            1. use_cost_tau: true  → τ = (We−Wr)/(We−Wc)  [Chow, 1970]
               Wr pode ter sido atualizado por calibrate_wr() nesta janela.
            2. use_cost_tau: false → rejection_threshold fixo
        """
        bt = self.config["backtest"]
        if bt.get("use_cost_tau", True):
            Wc = bt.get("Wc", 0.00)
            Wr = bt.get("Wr", 0.01)
            We = bt.get("We", 0.05)
            if not (Wc < Wr < We):
                return bt.get("rejection_threshold", 0.60)
            return (We - Wr) / (We - Wc)
        return bt.get("rejection_threshold", 0.60)
