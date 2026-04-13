"""
src/features.py
===============
Engenharia de variáveis (feature engineering).

Implementa as features descritas na Seção 3.2.1 da dissertação:
    1.  RSI14      — Índice de Força Relativa padrão (14 períodos)
    2.  RSI7       — RSI curto (7 períodos) para razão
    3.  RSI_binary — I(RSI >= 50)
    4.  RSI_dist50 — |RSI − 50|  (distância do nível neutro)
    5.  RSI_ratio  — RSI14 / RSI7
    6.  RSI_deriv  — dRSI/dt (derivada de primeira ordem)
    7.  RSL        — (close − MA66) / MA66 × 100
    8.  RSI_LPF    — RSI suavizado por SMA (filtro passa-baixas)
    9.  RSI_slope  — RSI_LPF_t − RSI_LPF_{t−1}
    10. RSI_dist50_signed — RSI_LPF − 50  (com sinal)
    11. regime_guerra     — +1/−1/0 conforme Guerra (2025)

Além disso:
    - log_return   — retorno logarítmico diário
    - realized_vol — desvio-padrão rolling dos retornos
    - selic_norm   — SELIC normalizada (variável macroeconômica exógena)

Referências:
    Wilder (1978) — New Concepts in Technical Trading Systems
    Ehlers (2001) — Rocket Science for Traders (filtros digitais)
    Guerra (2025) — Revisiting the RSI: From Oscillator to Trend-Following
    Seção 3.2.1, 3.3.3, 3.3.4 da dissertação
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# RSI — método de Wilder (1978)
# ---------------------------------------------------------------------------

def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calcula o RSI usando o método de suavização de Wilder.

    Equação (Seção 2.5):
        RSI_t = 100 − 100 / (1 + RS_t)
        RS_t  = EWM(U, span=period) / EWM(D, span=period)
        U_t   = max(ΔP_t, 0)
        D_t   = max(−ΔP_t, 0)

    Parâmetros
    ----------
    prices : pd.Series
        Série de preços de fechamento.
    period : int
        Janela (padrão 14 conforme Wilder).

    Retorna
    -------
    pd.Series
        RSI no intervalo [0, 100].
    """
    delta = prices.diff()
    up   = delta.clip(lower=0)
    down = (-delta).clip(lower=0)

    # Wilder usa EWM com alpha = 1/period (equivalente a SMMA)
    alpha = 1.0 / period
    avg_up   = up.ewm(alpha=alpha, adjust=False).mean()
    avg_down = down.ewm(alpha=alpha, adjust=False).mean()

    rs = avg_up / avg_down.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.clip(0, 100)
    rsi.name = f"rsi{period}"
    return rsi


# ---------------------------------------------------------------------------
# RSI-LPF — filtro passa-baixas sobre o RSI (Seção 2.6.1)
# ---------------------------------------------------------------------------

def compute_rsi_lpf(rsi: pd.Series, p: int = 5) -> pd.Series:
    """
    Aplica filtro passa-baixas (SMA) ao RSI para obter RSI-LPF.

    Equação (2.9):
        RSI_LPF_t = (1/p) * Σ_{i=0}^{p-1} RSI_{t−i}

    Reduz ruído de alta frequência preservando a tendência de fundo.
    A linha 50 funciona como divisor de regime:
        RSI_LPF > 50  → tendência de alta
        RSI_LPF < 50  → tendência de baixa
        RSI_LPF ≈ 50  → regime lateral / zona de rejeição

    Parâmetros
    ----------
    rsi : pd.Series
        Série RSI original.
    p : int
        Janela do filtro (padrão 5).

    Retorna
    -------
    pd.Series
        RSI suavizado (RSI-LPF).

    Referências
    -----------
    Ehlers (2001) — Rocket Science for Traders
    Guerra (2025) — Revisiting the RSI
    """
    rsi_lpf = rsi.rolling(window=p, min_periods=p).mean()
    rsi_lpf.name = f"rsi_lpf{p}"
    return rsi_lpf


def compute_rsi_slope(rsi_lpf: pd.Series) -> pd.Series:
    """
    Inclinação do RSI-LPF: RSI_slope_t = RSI_LPF_t − RSI_LPF_{t−1}.

    Equação (Seção 2.6.1.3):
        Positivo → aceleração da força compradora
        Negativo → fortalecimento da pressão vendedora

    Referência
    ----------
    Guerra (2025) — Revisiting the RSI
    """
    slope = rsi_lpf.diff()
    slope.name = "rsi_slope"
    return slope


# ---------------------------------------------------------------------------
# Regime de mercado — Guerra (2025)
# ---------------------------------------------------------------------------

def compute_regime_guerra(
    rsi_lpf:     pd.Series,
    slope:       pd.Series,
    zona_neutra: float = 2.0,
) -> pd.Series:
    """
    Classifica o regime de mercado conforme a lógica de Guerra (2025).

    Regras (exatamente como na aula):
        +1  → Tendência de ALTA   : RSI_LPF > 50 + zona_neutra  E  slope > 0
        −1  → Tendência de BAIXA  : RSI_LPF < 50 − zona_neutra  E  slope < 0
         0  → REJEIÇÃO / neutro   : próximo de 50 ou slope ambíguo

    A zona neutra evita operar quando o RSI-LPF está "em cima do 50",
    reduzindo whipsaws (chicotadas). O slope confirma a direção — sem ele,
    um RSI-LPF > 52 mas caindo ainda geraria sinal de compra erroneamente.

    Parâmetros
    ----------
    rsi_lpf     : pd.Series  — RSI suavizado pelo filtro passa-baixas
    slope       : pd.Series  — inclinação (diff) do RSI-LPF
    zona_neutra : float      — largura da banda de rejeição em torno de 50
                               (padrão 2.0, ou seja, faixa 48–52)

    Retorna
    -------
    pd.Series com valores +1, −1, 0

    Referência
    ----------
    Guerra (2025) — Revisiting the RSI: From Oscillator to Trend-Following
    Dissertação §2.6.1.3 e §2.6.1.4
    """
    regime = pd.Series(0, index=rsi_lpf.index, name="regime_guerra")
    regime[(rsi_lpf > 50 + zona_neutra) & (slope > 0)] =  1
    regime[(rsi_lpf < 50 - zona_neutra) & (slope < 0)] = -1
    return regime


# ---------------------------------------------------------------------------
# RSL — Relative Strength to Moving Average (Seção 3.2.1, item 6)
# ---------------------------------------------------------------------------

def compute_rsl(prices: pd.Series, ma_period: int = 66) -> pd.Series:
    """
    Calcula RSL: distância relativa entre preço e média móvel de longo prazo.

        RSL_t = (close_t − MA66_t) / MA66_t × 100

    Parâmetros
    ----------
    prices : pd.Series
        Série de preços de fechamento.
    ma_period : int
        Período da média móvel (padrão 66 pregões ≈ 3 meses).

    Retorna
    -------
    pd.Series
        RSL em percentual.
    """
    ma = prices.rolling(window=ma_period, min_periods=ma_period).mean()
    rsl = (prices - ma) / ma * 100.0
    rsl.name = "rsl"
    return rsl


# ---------------------------------------------------------------------------
# Construção da matriz de features completa
# ---------------------------------------------------------------------------

def build_feature_matrix(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Constrói a matriz de features completa a partir dos dados OHLCV + SELIC.

    Features produzidas (Seção 3.2.1 + Guerra 2025):
        rsi14             — RSI padrão 14 períodos
        rsi7              — RSI curto 7 períodos
        rsi_binary        — I(RSI14 >= 50)
        rsi_dist50        — |RSI14 − 50|  (distância absoluta)
        rsi_ratio         — RSI14 / RSI7
        rsi_deriv         — primeira diferença do RSI14
        rsl               — Relative Strength to MA66
        rsi_lpf           — RSI14 suavizado (SMA, p=5)
        rsi_slope         — inclinação do RSI-LPF
        rsi_dist50_signed — RSI_LPF − 50 (com sinal)
        regime_guerra     — +1/−1/0 conforme Guerra (2025)

    Features de mercado:
        log_return    — log(close_t / close_{t−1})
        realized_vol  — std rolling dos log_returns (20d)
        high_low_pct  — (high − low) / close
        selic_pct     — taxa SELIC anual normalizada

    Features de antecipação de risco (adicionadas para reduzir drawdown):
        macd_hist     — histograma MACD (12-26-9): deterioração antes do RSI reagir
        macd_line     — linha MACD normalizada pelo preço
        ret_acum_5d   — retorno acumulado 5 pregões (momentum direto)
        ret_acum_10d  — retorno acumulado 10 pregões
        vol_accel     — vol5d / vol20d: expansão de volatilidade precede quedas
        dist_topo_52s — distância percentual do máximo de 52 semanas (sempre ≤ 0)
        volume_rel    — volume / média 20d: confirma pressão direcional

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame OHLCV + selic_pct, indexado por data.
    config : dict
        Configuração carregada do config.yaml.

    Retorna
    -------
    pd.DataFrame
        Matriz de features, sem NaN nas linhas iniciais.
    """
    feat_cfg = config["features"]
    rsi_period       = feat_cfg["rsi_period"]
    rsi_short        = feat_cfg["rsi_short_period"]
    lpf_period       = feat_cfg["rsi_lpf_period"]
    ma_period        = feat_cfg["ma_period"]
    vol_window       = feat_cfg.get("log_return_window", 20)

    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    features = pd.DataFrame(index=df.index)

    # --- Log-retorno e volatilidade ---
    features["log_return"]   = np.log(close / close.shift(1))
    features["realized_vol"] = features["log_return"].rolling(vol_window, min_periods=vol_window).std()
    features["high_low_pct"] = (high - low) / close

    # --- RSI14 e derivadas (Seção 3.2.1) ---
    rsi14 = compute_rsi(close, period=rsi_period)
    rsi7  = compute_rsi(close, period=rsi_short)

    features["rsi14"]      = rsi14
    features["rsi7"]       = rsi7
    features["rsi_binary"] = (rsi14 >= 50).astype(float)
    features["rsi_dist50"] = (rsi14 - 50.0).abs()
    features["rsi_ratio"]  = rsi14 / rsi7.replace(0, np.nan)
    features["rsi_deriv"]  = rsi14.diff()

    # --- RSL — Relative Strength to MA (Seção 3.2.1, item 6) ---
    features["rsl"] = compute_rsl(close, ma_period=ma_period)

    # --- RSI-LPF e slope (Seção 2.6.1) ---
    rsi_lpf = compute_rsi_lpf(rsi14, p=lpf_period)
    features["rsi_lpf"]   = rsi_lpf
    features["rsi_slope"] = compute_rsi_slope(rsi_lpf)

    # --- Regime de mercado — Guerra (2025) ---
    zona_neutra = feat_cfg.get("zona_neutra", 2.0)
    rsi_slope   = compute_rsi_slope(rsi_lpf)
    features["rsi_dist50_signed"] = rsi_lpf - 50.0          # com sinal (positivo=alta)
    features["regime_guerra"]     = compute_regime_guerra(
        rsi_lpf, rsi_slope, zona_neutra=zona_neutra
    )

    # --- Variável macroeconômica exógena: SELIC (Seção 3.3.4) ---
    if "selic_pct" in df.columns:
        features["selic_pct"] = df["selic_pct"]
    else:
        features["selic_pct"] = np.nan

    # Remove linhas com NaN (resultantes das janelas de cálculo)
    features = features.dropna()

    return features


# ---------------------------------------------------------------------------
# Rotulagem supervisionada dos alvos (Seção 3.3.1 / 3.4.1)
# ---------------------------------------------------------------------------

def create_targets(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Cria os três alvos do modelo multitarefa.

    Alvos (Seção 3.3.1):
        y_ret     : log-retorno em t + horizon_ret  (regressão)
        y_trend30 : 1 se log(P_{t+h}/P_t) > threshold  (classificação binária)
        y_decision: rótulo de compra/venda com critério triplo (ver abaixo)

    Critério de rotulagem triplo para y_decision (Seção 3.3.1 revisada):

        VENDA (0) se qualquer das condições abaixo for verdadeira:
            A) retorno acumulado nos próximos h_trend dias < threshold
               (critério fundamental — o mercado de fato caiu)

            B) RSI14 cruza de volta abaixo de 70 (saída de sobrecompra clássica)
               O cruzamento ocorre no instante t em que RSI14_{t-1} >= 70
               e RSI14_t < 70. Captura o momento em que o ímpeto comprador
               se esgota após sobrecompra — regra clássica de Wilder (1978).
               Válido especialmente em ações individuais.

            C) RSI14, RSI7 e RSI_LPF todos abaixo de 50 simultaneamente
               (pressão vendedora confirmada pelo filtro passa-baixas)
               Teoria de Guerra (2025): o RSI_LPF cruzando 50 para baixo
               com slope negativo indica regime de baixa estabelecido.
               Os três juntos eliminam sinais espúrios.

        COMPRA (1) caso contrário.

    Os critérios B e C são configuráveis individualmente via config.yaml.
    Isso permite desligar cada um para experimentos comparativos na tese.

    O rótulo "rejeição" não é atribuído aqui — é inferido pós-previsão
    pelo mecanismo de Chow (1970) com limiar τ (Seção 3.3.8).

    Parâmetros
    ----------
    df     : DataFrame com coluna 'close', indexado por data.
    config : dict carregado do config.yaml.

    Retorna
    -------
    pd.DataFrame com colunas: y_ret, y_trend30, y_decision.
    """
    tgt_cfg  = config["targets"]
    feat_cfg = config["features"]
    h_ret    = tgt_cfg["horizon_ret"]
    h_trend  = tgt_cfg["horizon_trend"]
    thresh   = tgt_cfg["return_threshold"]

    # Lê modo de rotulagem técnica do config
    # "nenhum"  → só retorno futuro (critério original)
    # "guerra"  → + RSI_LPF cruzando faixa de 50 (RSI14, RSI7 e RSI_LPF < 50)
    # "classico"→ + RSI14 saindo de sobrecompra (cruzamento 70↓)
    # "ambos"   → guerra + classico simultaneamente
    modo = tgt_cfg.get("rotulagem_tecnica", "nenhum").lower()

    rsi_sobrecompra = tgt_cfg.get("rsi_sobrecompra", 70)
    rsi_sobrevenda  = tgt_cfg.get("rsi_sobrevenda",  30)
    lpf_limiar      = tgt_cfg.get("rsi_lpf_limiar",  50)

    rsi_period = feat_cfg.get("rsi_period",       14)
    rsi_short  = feat_cfg.get("rsi_short_period",  7)
    lpf_period = feat_cfg.get("rsi_lpf_period",    5)

    close = df["close"]
    targets = pd.DataFrame(index=df.index)

    # ── Alvo 1: retorno log no próximo pregão (regressão) ────────────────────
    targets["y_ret"] = np.log(close.shift(-h_ret) / close)

    # ── Alvo 2: tendência em h_trend pregões (classificação binária) ─────────
    cumret = np.log(close.shift(-h_trend) / close)
    targets["y_trend30"] = (cumret > thresh).astype(float)

    # ── Alvo 3: decisão com critério configurável ─────────────────────────────
    # Critério A (sempre ativo): retorno futuro abaixo do threshold
    crit_a = cumret <= thresh

    # Calcula RSIs só se necessário
    crit_b = pd.Series(False, index=close.index)
    crit_c = pd.Series(False, index=close.index)

    if modo in ("classico", "ambos"):
        # Critério B — Sobrecompra clássica (Wilder, 1978):
        # RSI14 cruza de volta ABAIXO de rsi_sobrecompra.
        # Espera o cruzamento (não a entrada na zona) — evita vender
        # no meio de uma tendência de alta forte.
        rsi14_b = compute_rsi(close, period=rsi_period)
        crit_b  = (rsi14_b.shift(1) >= rsi_sobrecompra) & (rsi14_b < rsi_sobrecompra)

    if modo in ("guerra", "ambos"):
        # Critério C — Regime de baixa pelo RSI_LPF (Guerra, 2025):
        # RSI14, RSI7 e RSI_LPF todos abaixo de lpf_limiar (padrão 50).
        # Os três juntos confirmam a tendência e eliminam sinais espúrios.
        rsi14_c   = compute_rsi(close, period=rsi_period)
        rsi7_c    = compute_rsi(close, period=rsi_short)
        rsi_lpf_c = compute_rsi_lpf(rsi14_c, p=lpf_period)
        crit_c    = (rsi14_c < lpf_limiar) & (rsi7_c < lpf_limiar) & (rsi_lpf_c < lpf_limiar)

    # Venda se qualquer critério ativo for verdadeiro
    sinal_venda = crit_a | crit_b | crit_c
    y_dec = pd.Series(1, index=close.index, dtype=float)
    y_dec[sinal_venda] = 0
    targets["y_decision"] = y_dec

    # Remove linhas onde os alvos futuros não existem
    targets = targets.dropna()

    return targets


# ---------------------------------------------------------------------------
# Criação de sequências para a LSTM (janela deslizante)
# ---------------------------------------------------------------------------

def create_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transforma a matriz de features em tensores de sequência para a LSTM.

    Para cada instante t, cria a sequência X_t = [x_{t-L+1}, ..., x_t]
    onde L = window_size (Equação 3.3 da dissertação).

    Parâmetros
    ----------
    features : np.ndarray shape (n, d)
        Matriz de features.
    targets : np.ndarray shape (n, k)
        Matriz de alvos.
    window_size : int
        Comprimento da janela L.

    Retorna
    -------
    X : np.ndarray shape (n - L, L, d)
    y : np.ndarray shape (n - L, k)
    """
    X, y = [], []
    n = len(features)
    for i in range(window_size, n):
        X.append(features[i - window_size : i, :])
        y.append(targets[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ---------------------------------------------------------------------------
# Normalização condicional ao conjunto de treino (sem look-ahead)
# ---------------------------------------------------------------------------

def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """
    Ajusta StandardScaler apenas nos dados de treino.

    IMPORTANTE: nunca use dados de validação/teste para ajustar o scaler.
    Isso evita vazamento de informação (look-ahead bias).

    Parâmetros
    ----------
    X_train : np.ndarray shape (n_samples, L, d)

    Retorna
    -------
    StandardScaler ajustado nos dados de treino.
    """
    n, L, d = X_train.shape
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, d))
    return scaler


def apply_scaler(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """
    Aplica o scaler a um conjunto de dados (treino ou teste).

    Parâmetros
    ----------
    X : np.ndarray shape (n_samples, L, d)
    scaler : StandardScaler

    Retorna
    -------
    np.ndarray normalizado.
    """
    n, L, d = X.shape
    X_flat  = X.reshape(-1, d)
    X_norm  = scaler.transform(X_flat)
    return X_norm.reshape(n, L, d)
