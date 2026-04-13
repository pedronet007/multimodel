"""
src/models/__init__.py
======================
Registro central de modelos plugáveis.

Para adicionar um novo modelo:
    1. Crie src/models/meu_modelo.py com uma classe que herde de BaseModel
    2. Importe e registre em MODEL_REGISTRY abaixo

O walk_forward e o app.py usam apenas MODEL_REGISTRY e BaseModel —
nunca importam modelos específicos diretamente.
"""

from .base import BaseModel, PredictionResult
from .lstm import LSTMModel
from .gru import GRUModel
from .mlp import MLPModel
from .knn import KNNModel
from .transformer import TransformerModel

# ─────────────────────────────────────────────────────────────────────────────
# Registro central: chave = nome exibido no Streamlit, valor = classe do modelo
# ─────────────────────────────────────────────────────────────────────────────
MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "LSTM":        LSTMModel,
    "GRU":         GRUModel,
    "MLP":         MLPModel,
    "KNN":         KNNModel,
    "TRANSFORMER": TransformerModel,
}

__all__ = ["BaseModel", "PredictionResult", "MODEL_REGISTRY"]
