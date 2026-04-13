# Pipeline LSTM-B3 — Sistemas Inteligentes para Alocação de Capital

**Dissertação:** "Sistemas Inteligentes para Alocação de Capital no Mercado Acionário"  
**Autor:** Pedro Wilson Félix Magalhães Neto  
**Instituição:** IFCE — PPGET (Programa de Pós-Graduação em Engenharia de Telecomunicações)  

---

## Visão Geral

Este repositório implementa o pipeline completo descrito na dissertação, incluindo:

| Componente | Seção | Descrição |
|---|---|---|
| **RSI-LPF** | §2.6.1 | RSI com filtro passa-baixas (Guerra, 2025) |
| **Features** | §3.2.1 | 9 variáveis derivadas do RSI + macro |
| **LSTM Multitarefa** | §3.3.5 | 3 cabeças: retorno, tendência, compra/venda |
| **Rejeição** | §3.3.8 | Chow (1970) / Rocha-Neto (2011) |
| **Walk-Forward** | §3.5.1 | Validação temporal sem look-ahead |
| **Backtesting** | §3.4 | Stop-loss, take-profit, custo de transação |
| **Curva A-R** | §2.10.3 | Acurácia × Taxa de Rejeição |

---

## Arquitetura do Modelo (§3.3.5)

```
Entrada: (L=30, d=13) ── janela temporal × features
    │
    ▼
LSTM-1: 64 unidades, return_sequences=True
    │
Dropout: 0.20
    │
    ▼
LSTM-2: 32 unidades, return_sequences=False
    │
Dense: 16 neurônios (ReLU) ── compartilhada
    │
    ├──── Head 1: Dense(1) linear  → retorno t+1     [MSE]
    ├──── Head 2: Dense(1) sigmoid → tendência 30d   [BCE]
    └──── Head 3: Dense(2) softmax → compra/venda    [CCE]
         │
         ▼
    Rejeição: se max(p̂) < τ → "manter" (não opera)
```

**Função de perda:** `L = α·MSE + β·BCE + γ·CCE`  
com pesos `(α=0.30, β=0.35, γ=0.35)` configuráveis em `config.yaml`

---

## Features (§3.2.1)

| # | Feature | Descrição | Referência |
|---|---|---|---|
| 1 | `rsi14` | RSI padrão 14 períodos | Wilder (1978) |
| 2 | `rsi7` | RSI curto 7 períodos | Wilder (1978) |
| 3 | `rsi_binary` | I(RSI ≥ 50) | Dissertação §3.2.1 |
| 4 | `rsi_dist50` | \|RSI − 50\| | Dissertação §3.2.1 |
| 5 | `rsi_ratio` | RSI14 / RSI7 | Dissertação §3.2.1 |
| 6 | `rsi_deriv` | dRSI/dt | Dissertação §3.2.1 |
| 7 | `rsl` | (close − MA66) / MA66 × 100 | Dissertação §3.2.1 |
| 8 | `rsi_lpf` | RSI suavizado (SMA p=5) | Guerra (2025), Ehlers (2001) |
| 9 | `rsi_slope` | RSI_LPF_t − RSI_LPF_{t−1} | Guerra (2025) |
| + | `log_return` | log(close_t / close_{t−1}) | — |
| + | `realized_vol` | σ rolling 20d dos retornos | — |
| + | `selic_pct` | Taxa SELIC anual (COPOM) | BCB (§3.3.4) |

---

## Instalação

```bash
# 1. Clone e entre no diretório
cd projeto_lstm_b3

# 2. Instale dependências
pip install -r requirements.txt

# 3. Coloque os CSVs em data/
# data/bova11_daily_adjusted.csv
# data/ivvb11_daily_adjusted.csv
```

---

## Uso

### Modo Baseline (sem TensorFlow)
Executa a estratégia RSI-LPF pura como baseline (não requer TF):
```bash
python main.py --mode baseline --asset bova11
python main.py --mode baseline --asset ivvb11
```

### Modo LSTM Completo (requer TensorFlow)
Executa o pipeline walk-forward com LSTM multitarefa:
```bash
python main.py --mode lstm --asset bova11
python main.py --mode lstm --asset ivvb11
```

### Modo Demo (teste rápido — 2 janelas)
```bash
python main.py --demo --mode lstm --asset bova11
```

### Configuração personalizada
```bash
python main.py --config meu_config.yaml --asset bova11
```

---

## Estrutura do Projeto

```
projeto_lstm_b3/
├── main.py                    # Ponto de entrada
├── config.yaml                # Todos os hiperparâmetros
├── requirements.txt           # Dependências Python
├── data/
│   ├── bova11_daily_adjusted.csv
│   └── ivvb11_daily_adjusted.csv
├── src/
│   ├── data_loader.py         # Carregamento + SELIC histórica
│   ├── features.py            # RSI, RSI-LPF, engenharia de variáveis
│   ├── model.py               # LSTM multitarefa + mecanismo de rejeição
│   ├── walk_forward.py        # Protocolo walk-forward (§3.5.1)
│   ├── backtest.py            # Motor de backtesting (§3.4)
│   ├── metrics.py             # Sharpe, Sortino, MDD, Alpha, Curva A-R
│   └── visualizations.py     # Todas as figuras
└── resultados/
    ├── figuras/               # PNGs gerados automaticamente
    └── metrics_*.json         # Métricas em JSON
```

---

## Saídas Geradas

| Arquivo | Descrição |
|---|---|
| `01_preco_rsi_lpf.png` | Preço + RSI14 + RSI-LPF com linha de regime |
| `02_equity_curve.png` | Curva de capital vs buy-and-hold + drawdown |
| `03_ar_curve.png` | Curva Acurácia-Rejeição (Rocha-Neto, 2011) |
| `04_return_distribution.png` | Distribuição de retornos estratégia vs benchmark |
| `05_walk_forward.png` | Acurácia por janela walk-forward |
| `06_signals.png` | Sinais compra/venda/rejeição sobrepostos ao preço |
| `metrics_*.json` | Relatório completo de métricas |
| `predictions_*.csv` | Predições consolidadas por janela |

---

## Métricas (§3.5.2)

| Métrica | Descrição |
|---|---|
| **Sharpe** | Retorno excedente por unidade de risco total |
| **Sortino** | Sharpe penalizando apenas volatilidade negativa |
| **Max Drawdown** | Maior queda percentual acumulada |
| **Alpha** | Retorno excedente após ajuste pelo beta vs BOVA11 |
| **Curva A-R** | Acurácia × taxa de rejeição (Chow, 1970) |

---

## Mecanismo de Rejeição (§3.3.8 / §2.10.1)

Baseado na regra ótima de **Chow (1970)**:

```
p̂_max = max(P(compra|X), P(venda|X))

se p̂_max ≥ τ → classificar (compra ou venda)
se p̂_max < τ → rejeitar (não opera — "manter")
```

O limiar `τ` é selecionado em pseudo-validação (últimos 15% de cada janela de treino)
maximizando a acurácia nas amostras aceitas. Isso implementa o conceito das
curvas A-R (Rocha-Neto, 2011): quanto maior τ, maior a acurácia e maior a rejeição.

---

## Referências Principais

- Wilder (1978) — New Concepts in Technical Trading Systems
- Hochreiter & Schmidhuber (1997) — Long Short-Term Memory
- Chow (1970) — On Optimum Recognition Error and Reject Trade-off
- Rocha-Neto (2011) — SINPATCO II
- Ehlers (2001) — Rocket Science for Traders
- Guerra (2025) — Revisiting the RSI: From Oscillator to Trend-Following
- Banco Central do Brasil (2025) — COPOM / Taxa SELIC
