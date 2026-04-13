"""
src/data_loader.py
==================
Carregamento e pré-processamento dos dados.

Implementa:
    - Carregamento dos CSVs (BOVA11, IVVB11) — Seção 3.1.2
    - Série histórica da SELIC (BCB/COPOM) — Seção 3.3.4
    - Pré-processamento padronizado (Seção 3.1.2):
        * Conversão numérica
        * Tratamento de NaN
        * Winsorização (0.1% – 99.9%)
        * Remoção de inconsistências

Referências:
    B3 (2025) — Dados de mercado (BOVA11, IVVB11, QBTC11)
    Banco Central do Brasil (2025) — Taxa SELIC / COPOM
    Wilder (1978) — New Concepts in Technical Trading Systems
    QR Asset Management (2025) — QBTC11 (Bitcoin via B3)
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings
from typing import Optional


# ---------------------------------------------------------------------------
# Série histórica da SELIC — decisões COPOM (taxa anual em %)
# Fonte: Banco Central do Brasil — https://www.bcb.gov.br/controleinflacao/taxaselic
# ---------------------------------------------------------------------------
_SELIC_COPOM: dict[str, float] = {
    "2015-01-21": 12.25, "2015-03-04": 12.75, "2015-04-29": 13.25,
    "2015-07-29": 13.75, "2015-09-02": 14.25, "2015-10-21": 14.25,
    "2015-11-25": 14.25, "2016-01-20": 14.25, "2016-03-02": 14.25,
    "2016-04-27": 14.25, "2016-06-08": 14.25, "2016-07-20": 14.25,
    "2016-08-31": 14.25, "2016-10-19": 14.00, "2016-11-30": 13.75,
    "2017-01-11": 13.00, "2017-02-22": 12.25, "2017-04-12": 11.25,
    "2017-05-31": 10.25, "2017-07-26": 9.25,  "2017-09-06": 8.25,
    "2017-10-25": 7.50,  "2017-12-06": 7.00,  "2018-02-07": 6.75,
    "2018-03-21": 6.50,  "2018-05-16": 6.50,  "2018-06-20": 6.50,
    "2018-08-01": 6.50,  "2018-09-19": 6.50,  "2018-10-31": 6.50,
    "2018-12-12": 6.50,  "2019-02-06": 6.50,  "2019-03-20": 6.50,
    "2019-05-08": 6.50,  "2019-06-19": 6.50,  "2019-07-31": 6.00,
    "2019-09-18": 5.50,  "2019-10-30": 5.00,  "2019-12-11": 4.50,
    "2020-02-05": 4.25,  "2020-03-18": 3.75,  "2020-05-06": 3.00,
    "2020-06-17": 2.25,  "2020-08-05": 2.00,  "2020-09-16": 2.00,
    "2020-10-28": 2.00,  "2020-12-09": 2.00,  "2021-01-20": 2.00,
    "2021-03-17": 2.75,  "2021-05-05": 3.50,  "2021-06-16": 4.25,
    "2021-08-04": 5.25,  "2021-09-22": 6.25,  "2021-10-27": 7.75,
    "2021-12-08": 9.25,  "2022-02-02": 10.75, "2022-03-16": 11.75,
    "2022-05-04": 12.75, "2022-06-15": 13.25, "2022-08-03": 13.75,
    "2022-09-21": 13.75, "2022-10-26": 13.75, "2022-12-07": 13.75,
    "2023-02-01": 13.75, "2023-03-22": 13.75, "2023-05-03": 13.75,
    "2023-06-21": 13.75, "2023-08-02": 13.25, "2023-09-20": 12.75,
    "2023-11-01": 12.25, "2023-12-13": 11.75, "2024-01-31": 11.25,
    "2024-03-20": 10.75, "2024-05-08": 10.50, "2024-06-19": 10.50,
    "2024-07-31": 10.50, "2024-09-18": 10.75, "2024-11-06": 11.25,
    "2024-12-11": 12.25, "2025-01-29": 13.25, "2025-03-19": 14.25,
    "2025-05-07": 14.75,
}


# ---------------------------------------------------------------------------
# Funções públicas
# ---------------------------------------------------------------------------

def load_price_data(filepath: str | Path) -> pd.DataFrame:
    """
    Carrega série histórica de preços de um arquivo CSV.

    Parâmetros
    ----------
    filepath : str | Path
        Caminho para o arquivo CSV com colunas:
        timestamp, open, high, low, close, volume

    Retorna
    -------
    pd.DataFrame
        DataFrame indexado por data, ordenado cronologicamente.
    """
    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    df = df.rename(columns={"timestamp": "date"})
    df = df.sort_values("date").reset_index(drop=True)
    df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    return df


def create_selic_series(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> pd.Series:
    """
    Constrói série diária da SELIC por forward-fill das decisões COPOM.

    A SELIC é uma variável exógena (macroeconômica) no modelo multimodal.
    Entre reuniões do COPOM (a cada ~45 dias), a taxa permanece constante,
    portanto aplica-se forward-fill para alinhar ao calendário de pregões.

    Parâmetros
    ----------
    start_date, end_date : str | pd.Timestamp
        Intervalo desejado.

    Retorna
    -------
    pd.Series
        Série diária com a taxa SELIC anual (em %), indexada por data.

    Referência
    ----------
    Banco Central do Brasil (2025) — Comitê de Política Monetária (COPOM).
    """
    copom_series = pd.Series(
        _SELIC_COPOM,
        name="selic_anual_pct",
    )
    copom_series.index = pd.to_datetime(copom_series.index)
    copom_series = copom_series.sort_index()

    # Cria índice diário completo e aplica forward-fill
    full_idx = pd.date_range(start=start_date, end=end_date, freq="D")
    selic_daily = copom_series.reindex(
        copom_series.index.union(full_idx)
    ).ffill().bfill()
    selic_daily = selic_daily.reindex(full_idx)
    return selic_daily


def preprocess_data(df: pd.DataFrame, winsor_limits: tuple = (0.001, 0.999)) -> pd.DataFrame:
    """
    Pré-processamento padronizado conforme Seção 3.1.2 da dissertação.

    Etapas:
        1. Conversão de todos os campos para numérico
        2. Substituição de infinitos por NaN
        3. Winsorização (0.1% – 99.9%) para limitar valores extremos
        4. Remoção de registros com inconsistências graves

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com colunas OHLCV.
    winsor_limits : tuple
        Limites inferior e superior da winsorização.

    Retorna
    -------
    pd.DataFrame
        DataFrame pré-processado.
    """
    df = df.copy()

    # 1. Conversão numérica
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2. Infinitos → NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # 3. Winsorização em close e volume
    for col in ["close", "volume"]:
        if col in df.columns:
            lo = df[col].quantile(winsor_limits[0])
            hi = df[col].quantile(winsor_limits[1])
            df[col] = df[col].clip(lower=lo, upper=hi)

    # 4. Remove linhas onde close é NaN ou zero
    df = df[df["close"].notna() & (df["close"] > 0)]

    return df


def resolve_asset_path(
    ticker: str,
    config: dict,
    custom_file: Optional[str] = None,
) -> Path:
    """
    Resolve o caminho do CSV de um ativo pelo ticker.

    Convenção padrão:
        data/{ticker}_daily_adjusted.csv
    Exemplo:
        PETR4  → data/petr4_daily_adjusted.csv
        BOVA11 → data/bova11_daily_adjusted.csv

    Parâmetros
    ----------
    ticker      : str  — código do ativo (qualquer case)
    config      : dict — configuração do config.yaml
    custom_file : str | None — caminho explícito (sobrescreve convenção)

    Retorna
    -------
    Path resolvido
    """
    if custom_file:
        return Path(custom_file)

    base_dir = Path(config.get("_base_dir", "."))
    data_cfg = config["data"]
    data_dir = data_cfg.get("data_dir", "data")
    suffix   = data_cfg.get("file_suffix", "_daily_adjusted.csv")

    return base_dir / data_dir / f"{ticker.lower()}{suffix}"


def load_asset(
    ticker: str,
    config: dict,
    custom_file: Optional[str] = None,
) -> pd.DataFrame:
    """
    Carrega, pré-processa e adiciona SELIC a um único ativo.

    Parâmetros
    ----------
    ticker      : str  — código do ativo (ex: "PETR4", "bova11")
    config      : dict — configuração do config.yaml
    custom_file : str | None — caminho CSV explícito

    Retorna
    -------
    pd.DataFrame com colunas OHLCV + selic_pct

    Raises
    ------
    FileNotFoundError se o CSV não for encontrado.
    """
    path = resolve_asset_path(ticker, config, custom_file)

    if not path.exists():
        msg = (
            f"\n[DataLoader] CSV não encontrado: '{path}'\n"
            f"  Adicione o arquivo de dados do ativo '{ticker.upper()}' em:\n"
            f"  {path.resolve()}\n"
            f"  O arquivo deve ter as colunas: timestamp,open,high,low,close,volume\n"
        )
        raise FileNotFoundError(msg)

    df    = preprocess_data(load_price_data(path))
    start = df.index.min()
    end   = df.index.max()
    selic = create_selic_series(start, end)
    df["selic_pct"] = selic.reindex(df.index).ffill().bfill()

    print(
        f"[DataLoader] {ticker.upper()}: {len(df)} pregões | "
        f"{start.date()} → {end.date()} | "
        f"SELIC: {df['selic_pct'].min():.2f}%–{df['selic_pct'].max():.2f}%"
    )
    return df


def load_and_prepare(
    config: dict,
    tickers: Optional[list[str]] = None,
    custom_files: Optional[dict[str, str]] = None,
) -> dict[str, pd.DataFrame]:
    """
    Carrega e prepara um ou mais ativos.

    Parâmetros
    ----------
    config       : dict — configuração do config.yaml
    tickers      : list[str] | None
        Lista de tickers a carregar.
        Se None, carrega todos os CSVs encontrados em data_dir.
    custom_files : dict[str, str] | None
        Mapeamento ticker → caminho CSV explícito.
        Ex: {"meu_ativo": "/caminho/para/dados.csv"}

    Retorna
    -------
    dict[str, pd.DataFrame]
        Dicionário {ticker_lower: DataFrame} para cada ativo carregado.

    Exemplos
    --------
    # Carrega só BOVA11
    datasets = load_and_prepare(config, tickers=["bova11"])

    # Carrega BOVA11 + PETR4
    datasets = load_and_prepare(config, tickers=["bova11", "petr4"])

    # Carrega de caminho personalizado
    datasets = load_and_prepare(config,
                                tickers=["minha_acao"],
                                custom_files={"minha_acao": "data/minha_acao.csv"})
    """
    custom_files = custom_files or {}
    base_dir = Path(config.get("_base_dir", "."))
    data_cfg = config["data"]
    data_dir = base_dir / data_cfg.get("data_dir", "data")
    suffix   = data_cfg.get("file_suffix", "_daily_adjusted.csv")

    # Se não informou tickers, descobre automaticamente pelos CSVs na pasta
    if tickers is None:
        csvs = sorted(data_dir.glob(f"*{suffix}"))
        tickers = [p.name.replace(suffix, "") for p in csvs]
        if not tickers:
            raise FileNotFoundError(
                f"Nenhum CSV encontrado em '{data_dir}' com sufixo '{suffix}'.\n"
                f"Adicione arquivos no formato: {{ticker}}{suffix}"
            )
        print(f"[DataLoader] CSVs encontrados: {[t.upper() for t in tickers]}")

    datasets: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        key = ticker.lower()
        try:
            df = load_asset(ticker, config, custom_files.get(key))
            datasets[key] = df
        except FileNotFoundError as e:
            print(f"[DataLoader] AVISO: {e}")

    if not datasets:
        raise RuntimeError("Nenhum ativo pôde ser carregado. Verifique os arquivos CSV.")

    return datasets
