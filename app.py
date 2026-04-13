import streamlit as st
import pandas as pd
import numpy as np
import yaml
import io
import json
import zipfile
import re  # <-- NOVA IMPORTAÇÃO
from pathlib import Path
import matplotlib.pyplot as plt

try:
    import yfinance as yf
    YFINANCE_OK = True
except ImportError:
    YFINANCE_OK = False

# ---------------------------------------------------------------------------
# Download automático via yfinance
# ---------------------------------------------------------------------------

def _format_ticker_yf(ticker: str) -> str:
    """Verifica o padrão do ticker e adiciona '.SA' apenas para ativos da B3."""
    ticker = ticker.strip().upper()
    
    # Se já tem um sufixo válido, retorna como está
    if any(ticker.endswith(s) for s in (".SA", "=X", "-USD")):
        return ticker
        
    # Padrão B3: 4 letras, 1 ou 2 números (ex: 3, 4, 11), 'F' opcional (fracionário)
    padrao_b3 = r'^[A-Z]{4}\d{1,2}F?$'
    
    if re.match(padrao_b3, ticker):
        return f"{ticker}.SA"
        
    # Para ETFs/Ações gringas (ex: QQQ, AAPL), retorna o original
    return ticker

def _ticker_to_filename(ticker: str) -> str:
    """Converte ticker para nome do arquivo CSV (mesmo padrão do script manual)."""
    # Alterado de replace("-", "/") para replace("-", "_") para não criar pastas falsas
    label = ticker.upper().replace(".SA", "").replace("=X", "")
    return label.lower() + "_daily_adjusted.csv"

def download_asset_csv(ticker: str, data_dir: str = "data", period: str = "10y") -> Path:
    """
    Baixa dados históricos do ticker via yfinance e salva em data/<ticker>.csv
    no mesmo formato esperado pelo data_loader do pipeline.

    Parâmetros
    ----------
    ticker   : ex. 'WEGE3', 'PETR4', 'QQQ'
    data_dir : pasta onde salvar (padrão: 'data')
    period   : período histórico (padrão: '10y')

    Retorna
    -------
    Path para o CSV salvo.

    Levanta
    -------
    ImportError  : se yfinance não estiver instalado
    ValueError   : se o ticker não retornar dados
    """
    if not YFINANCE_OK:
        raise ImportError(
            "yfinance não está instalado. Execute: pip install yfinance"
        )

    # Formata o ticker aplicando a regra de Regex
    ticker_yf = _format_ticker_yf(ticker)
																	 
									 

    t = yf.Ticker(ticker_yf)
    df = t.history(period=period, auto_adjust=True, actions=False)

    if df.empty:
        raise ValueError(
            f"Ticker '{ticker_yf}' não retornou dados. "
            "Verifique se o código está correto (ex: PETR4, WEGE3, BOVA11, QQQ)."
        )

    # Formata para o padrão do pipeline
    df = df.reset_index()
    df["timestamp"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    df = df.rename(columns={
        "Open": "open", "High": "high",
        "Low":  "low",  "Close": "close", "Volume": "volume",
    })[["timestamp", "open", "high", "low", "close", "volume"]]
    df["open"]   = df["open"].round(4)
    df["high"]   = df["high"].round(4)
    df["low"]    = df["low"].round(4)
    df["close"]  = df["close"].round(4)
    df["volume"] = df["volume"].fillna(0).astype(int)
    df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)

    # Salva na pasta data/
    out_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / _ticker_to_filename(ticker_yf)
    df.to_csv(out_path, index=False)
    return out_path


def check_asset_exists(ticker: str, data_dir: str = "data") -> tuple[bool, Path]:
    """
    Verifica se o CSV do ativo já existe na pasta data/.

    Retorna
    -------
    (existe: bool, caminho: Path)
    """
    # Formata o ticker aplicando a regra de Regex
																	 
    ticker_yf = _format_ticker_yf(ticker)
    
    path = Path(data_dir) / _ticker_to_filename(ticker_yf)
    return path.exists(), path


# Importando os módulos do seu pipeline
from src.data_loader import load_asset
from src.backtest import run_backtest
from src.metrics import compute_all_metrics
from src.visualizations import (
    plot_equity_curve,
    plot_signals_on_price,
    plot_ar_curve,
    plot_return_distribution,
    plot_technical_analysis,
)
from src.features import build_feature_matrix
from src.walk_forward import run_walk_forward, consolidate_results
from src.models import MODEL_REGISTRY
from gerar_graficos_comparativos import (
    grafico_barras, grafico_drawdown, grafico_tempo,
)

# ==========================================
# Configuração da Página
# ==========================================
st.set_page_config(
    page_title="Dashboard LSTM B3",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Sistemas Inteligentes para Alocação de Capital")
st.markdown("**Pesquisador:** Pedro Wilson Félix | **Pipeline:** LSTM B3 / RSI-LPF")
st.divider()

# ==========================================
# Funções Auxiliares
# ==========================================
@st.cache_data
def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["_base_dir"] = "."
    return config

@st.cache_data
def load_price_data(asset: str, _config: dict):
    """Carrega apenas os dados de preço para o backtest."""
    return load_asset(asset, _config)

@st.cache_data
def load_features(asset: str, _config: dict):
    """Carrega e calcula features técnicas (cache para evitar re-cálculo no slider)."""
    df = load_asset(asset, _config)
    return build_feature_matrix(df, _config)

def run_full_pipeline(asset: str, config: dict, model_name: str = "LSTM"):
    """Executa o walk-forward completo com o modelo selecionado."""
    df = load_asset(asset, config)
    model_class = MODEL_REGISTRY[model_name]
    walk_results = run_walk_forward(df, config, model_class=model_class, verbose=True)
    pred_df = consolidate_results(walk_results)

    # Salva com sufixo do modelo para não sobrescrever resultados anteriores
    Path("resultados").mkdir(exist_ok=True)
    suffix = f"_{model_name.lower()}" if model_name != "LSTM" else ""
    pred_df.to_csv(f"resultados/predictions_{asset}{suffix}.csv")
    return pred_df

# ==========================================
# Sidebar: Controles e Parâmetros
# ==========================================
st.sidebar.header("⚙️ Configurações Gerais")

# ── Seleção de modelo ────────────────────────────────────────────────────────
_MODEL_DESCRIPTIONS = {
    "LSTM":        "LSTM — Redes recorrentes com memória longa/curta (Hochreiter & Schmidhuber, 1997). Modelo principal da dissertação.",
    "GRU":         "GRU  — Variante simplificada da LSTM com ~25% menos parâmetros (Cho et al., 2014). Mais rápida, desempenho similar.",
    "MLP":         "MLP  — Rede densa sem memória temporal. Baseline neural: avalia se sequencialidade agrega valor.",
    "KNN":         "KNN  — K-Vizinhos Mais Próximos (Cover & Hart, 1967). Baseline clássico não-paramétrico, sem treino.",
    "TRANSFORMER": "Transformer — Encoder com mecanismo de auto-atenção multi-cabeça (Vaswani et al., 2017). Captura dependências de longo alcance em O(1).",
}
selected_model_name = st.sidebar.selectbox(
    "🤖 Modelo",
    options=list(MODEL_REGISTRY.keys()),
    index=0,
    help="\n\n".join(_MODEL_DESCRIPTIONS.values()),
)
st.sidebar.caption(_MODEL_DESCRIPTIONS.get(selected_model_name, selected_model_name))
st.sidebar.divider()

# ── Seleção de ativo com detecção automática de CSV ─────────────────────────
_ticker_input = st.sidebar.text_input(
    "Ativo (Ticker)",
    value="BOVA11",
    help="Digite o código sem .SA (ex: BOVA11, PETR4, WEGE3). "
         "Se o CSV não existir na pasta data/, será baixado automaticamente.",
).strip().upper()

selected_asset = _ticker_input.lower().replace(".sa", "")

# Verifica existência do CSV e exibe status no sidebar
_data_dir = "data"
_csv_exists, _csv_path = check_asset_exists(_ticker_input, _data_dir)

if _csv_exists:
    st.sidebar.success(f"✅ CSV encontrado: `{_csv_path.name}`")
else:
    st.sidebar.warning(f"⚠️ CSV não encontrado para **{_ticker_input}**")
    if not YFINANCE_OK:
        st.sidebar.error("yfinance não instalado. Execute: `pip install yfinance`")
    else:
        if st.sidebar.button("⬇️ Baixar dados do yfinance", use_container_width=True):
            with st.spinner(f"Baixando {_ticker_input} via yfinance..."):
                try:
                    _saved = download_asset_csv(_ticker_input, _data_dir)
                    st.sidebar.success(f"✅ Salvo em `{_saved}`")
                    st.rerun()   # recarrega para detectar o CSV recém-criado
                except ValueError as e:
                    st.sidebar.error(str(e))
                except Exception as e:
                    st.sidebar.error(f"Erro inesperado: {e}")

# Período histórico para download (opcional)
with st.sidebar.expander("⚙️ Opções de Download"):
    _download_period = st.selectbox(
        "Período histórico",
        ["10y", "5y", "3y", "2y", "1y", "max"],
        index=0,
        help="Usado apenas ao baixar novos ativos.",
    )
    if st.button("🔄 Re-baixar CSV (forçar atualização)", use_container_width=True):
        if not YFINANCE_OK:
            st.error("yfinance não instalado.")
        else:
            with st.spinner(f"Re-baixando {_ticker_input}..."):
                try:
                    _saved = download_asset_csv(_ticker_input, _data_dir, period=_download_period)
                    st.success(f"✅ CSV atualizado: `{_saved}`")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

modo_operacao = st.sidebar.radio(
    "Modo de Operação",
    ["Carregar Resultados Prontos", "Executar Pipeline Completo"]
)

st.sidebar.header("🛡️ Parâmetros de Risco (Tempo real)")
rejection_tau = st.sidebar.slider("Limiar de Rejeição (τ)", 0.50, 0.99, 0.60, 0.01)

st.sidebar.header("🔄 Opção B — Fallback de Tendência")
use_trend_fallback = st.sidebar.toggle(
    "Ativar Fallback por Veto Direcional",
    value=False,
    help=(
        "Quando ativado o Veto Direcional (Opção B), dias de rejeição do modelo com "
        "regime_guerra = +1 (RSI-LPF em alta) viram posição LONG "
        "em vez de ficarem em caixa.\n\n"
        "Requer coluna 'regime_guerra' no predictions_df — gerada "
        "automaticamente pelo walk_forward.py atualizado.\n\n"
        "Referência: Guerra (2025) — Revisiting the RSI."
    ),
)

# Inicializa o session_state para armazenar predições em memória
if "pred_df" not in st.session_state:
    st.session_state.pred_df = pd.DataFrame()
if "current_asset" not in st.session_state:
    st.session_state.current_asset = ""

# Limpa o cache de predições se o ativo mudar
if st.session_state.current_asset != selected_asset:
    st.session_state.pred_df = pd.DataFrame()
    st.session_state.current_asset = selected_asset

# ==========================================
# Lógica Principal
# ==========================================
config = load_config()

# Bloqueia execução se o CSV ainda não existir
_csv_exists_now, _ = check_asset_exists(_ticker_input, _data_dir)
if not _csv_exists_now:
    st.info(
        f"📥 **Dados não encontrados para {_ticker_input}.** \n"
        "Use o botão **⬇️ Baixar dados do yfinance** no menu lateral para "
        "baixar automaticamente, ou adicione o CSV manualmente na pasta `data/`."
    )
    st.stop()

df_price = load_price_data(selected_asset, config)

# ==========================================
# Abas principais do dashboard
# ==========================================
if "loaded_model_name" not in st.session_state:
    st.session_state.loaded_model_name = "LSTM"

tab_modelo, tab_tecnica = st.tabs([
    "📊 Resultados do Modelo",
    "📉 Análise Técnica Comparativa",
])

# ── ABA 2: Análise Técnica (independente do modelo) ──────────────────────────
with tab_tecnica:
    st.subheader("📉 Análise Técnica Comparativa")
    st.caption(
        "Visualização de indicadores clássicos para comparação com o preço bruto. "
        "Estes indicadores não alimentam o modelo — servem apenas para referência visual."
    )

    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        at_n_days   = st.slider("Período (pregões)", 60, 1260, 252, 21)
        at_show_vol = st.checkbox("Painel de volume", value=True)
    with col_t2:
        st.markdown("**SMA — Média Móvel Simples**")
        at_sma20  = st.checkbox("SMA 20",  value=True)
        at_sma50  = st.checkbox("SMA 50",  value=True)
        at_sma200 = st.checkbox("SMA 200", value=True)
    with col_t3:
        st.markdown("**EMA — Média Móvel Exponencial**")
        at_ema9  = st.checkbox("EMA 9",  value=True)
        at_ema21 = st.checkbox("EMA 21", value=True)
        st.markdown("**Bandas de Bollinger**")
        at_bb_period = st.select_slider("Período BB", [10, 15, 20, 25, 30], value=20)
        at_bb_std    = st.select_slider("Desvios BB", [1.5, 2.0, 2.5, 3.0], value=2.0)

    at_sma_periods = [p for p, sel in [(20, at_sma20),(50, at_sma50),(200, at_sma200)] if sel]
    at_ema_periods = [p for p, sel in [(9, at_ema9),(21, at_ema21)] if sel]

    df_feats = load_features(selected_asset, config)
    fig_ta = plot_technical_analysis(
        df_price, df_feats,
        asset       = selected_asset,
        n_days      = at_n_days,
        sma_periods = at_sma_periods or [20],
        ema_periods = at_ema_periods,
        bb_period   = at_bb_period,
        bb_std      = at_bb_std,
        show_volume = at_show_vol,
    )
    st.pyplot(fig_ta)
    plt.close(fig_ta)

    st.divider()
    with st.expander("Legenda dos indicadores"):
        st.markdown(
            "- **SMA(n)**: média aritmética dos últimos n preços\n"
            "- **EMA(n)**: média exponencial — peso maior nos preços recentes\n"
            "- **Bollinger(n,σ)**: bandas a ±σ desvios-padrão da SMA(n); mede volatilidade\n"
            "- **RSI14**: Wilder (1978) — zonas 30/70; triângulos = cruzamentos de retorno\n"
            "- **RSI-LPF**: RSI suavizado (Guerra, 2025) — regime pela linha 50"
        )

    # ── Comparativo de Modelos ────────────────────────────────────────────────
    st.divider()
    st.subheader("📊 Comparativo de Modelos — Walk-Forward")
    st.caption(
        "Gráficos gerados automaticamente a partir do JSON de comparação em "
        "`resultados/`. Execute `python main.py --asset bova11 --models all` "
        "para gerar ou atualizar os resultados."
    )

    # Localiza o JSON de comparação mais recente para o ativo selecionado
    _comp_json_paths = sorted(
        Path("resultados").glob(f"comparacao_{selected_asset.lower()}_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not _comp_json_paths:
        st.info(
            "ℹ️ Nenhum arquivo de comparação encontrado em `resultados/` para "
            f"**{selected_asset}**. Execute o pipeline com múltiplos modelos para gerar."
        )
    else:
        _comp_json_path = _comp_json_paths[0]

        @st.cache_data
        def _load_comp_json(path: str):
            with open(path, encoding="utf-8") as _f:
                return json.load(_f)

        _comp = _load_comp_json(str(_comp_json_path))
        _modelos_comp = list(_comp["modelos"].keys())
        _bh_ret = _comp["modelos"][_modelos_comp[0]]["total_return_benchmark_pct"]

        st.caption(
            f"📁 `{_comp_json_path.name}` — "
            f"{len(_modelos_comp)} modelos: {', '.join(_modelos_comp)}"
        )

        def _vals(campo):
            return [_comp["modelos"][m][campo] for m in _modelos_comp]

        # Seleção de qual gráfico exibir
        _graf_opcoes = {
            "Retorno Total (%)":           "retorno",
            "Máx. Drawdown (%)":           "drawdown",
            "Taxa de Rejeição (%)":        "rejeicao",
            "Taxa de Acerto / Hit Rate (%)": "acerto",
            "Eficiência Computacional (s)": "tempo",
        }
        _graf_sel = st.radio(
            "Selecionar gráfico:",
            options=list(_graf_opcoes.keys()),
            horizontal=True,
            key="radio_graf_comp",
        )
        _graf_key = _graf_opcoes[_graf_sel]

        import matplotlib
        matplotlib.use("Agg")

        if _graf_key == "retorno":
            _fig_comp = grafico_barras(
                modelos   = _modelos_comp,
                valores   = _vals("total_return_strategy_pct"),
                titulo    = f"Retorno Total Acumulado — {selected_asset}",
                xlabel    = "Retorno Total (%)",
                linha_ref = _bh_ret,
                label_ref = f"B&H: {_bh_ret:.1f}%",
                fmt       = "{:.2f}",
                sufixo    = "%",
            )

        elif _graf_key == "drawdown":
            _fig_comp = grafico_drawdown(
                modelos   = _modelos_comp,
                valores   = _vals("max_drawdown_pct"),
                titulo    = f"Máximo Drawdown — {selected_asset}",
                xlabel    = "Drawdown (%, módulo — menor = melhor)",
                linha_ref = -46.9,
                label_ref = "B&H: -46,9%",
            )

        elif _graf_key == "rejeicao":
            _fig_comp = grafico_barras(
                modelos      = _modelos_comp,
                valores      = _vals("rejection_rate_pct"),
                titulo       = f"Taxa de Rejeição — {selected_asset}",
                xlabel       = "Taxa de Rejeição (%) — menor = mais operações",
                destaque_min = True,
                fmt          = "{:.2f}",
                sufixo       = "%",
            )

        elif _graf_key == "acerto":
            _fig_comp = grafico_barras(
                modelos   = _modelos_comp,
                valores   = _vals("hit_rate_pct"),
                titulo    = f"Taxa de Acerto (Hit Rate) — {selected_asset}",
                xlabel    = "Taxa de Acerto (%) — operações fechadas com lucro",
                linha_ref = 50.0,
                label_ref = "Baseline 50%",
                fmt       = "{:.2f}",
                sufixo    = "%",
            )

        elif _graf_key == "tempo":
            _fig_comp = grafico_tempo(
                modelos = _modelos_comp,
                valores = _vals("tempo_seg"),
                titulo  = f"Eficiência Computacional — {selected_asset}\n"
                          "(30 janelas walk-forward, treinamento + inferência)",
            )

        st.pyplot(_fig_comp)
        plt.close(_fig_comp)

        # Botão para gerar e baixar todos os 5 de uma vez
        st.divider()
        with st.expander("⬇️ Exportar todos os gráficos comparativos (PNG)"):
            if st.button("Gerar e baixar ZIP com os 5 gráficos", key="btn_exp_comp"):
                import io, zipfile, tempfile
                from gerar_graficos_comparativos import gerar_todos

                with tempfile.TemporaryDirectory() as _tmpdir:
                    gerar_todos(str(_comp_json_path), _tmpdir)
                    _zip_buf = io.BytesIO()
                    with zipfile.ZipFile(_zip_buf, "w", zipfile.ZIP_DEFLATED) as _zf:
                        for _png in sorted(Path(_tmpdir).glob("*.png")):
                            _zf.write(_png, _png.name)
                    st.download_button(
                        label="📦 Baixar graficos_comparativos.zip",
                        data=_zip_buf.getvalue(),
                        file_name=f"graficos_comparativos_{selected_asset.lower()}.zip",
                        mime="application/zip",
                        use_container_width=True,
                    )

# ── ABA 1: Resultados do Modelo ───────────────────────────────────────────────
with tab_modelo:
    col_main1, col_main2 = st.columns([1, 3])

    with col_main1:
        if modo_operacao == "Carregar Resultados Prontos":
            suffix = f"_{selected_model_name.lower()}" if selected_model_name != "LSTM" else ""
            file_path_model = Path(f"resultados/predictions_{selected_asset}{suffix}.csv")
            file_path_lstm  = Path(f"resultados/predictions_{selected_asset}.csv")

            if st.button(f"Carregar CSV — {selected_model_name}", use_container_width=True):
                fp = file_path_model if file_path_model.exists() else file_path_lstm
                if fp.exists():
                    st.session_state.pred_df = pd.read_csv(fp, index_col="date", parse_dates=True)
                    st.session_state.loaded_model_name = selected_model_name
                    st.success(f"✅ Carregado: `{fp.name}`")
                else:
                    st.error(
                        f"Arquivo não encontrado para {selected_model_name}. "
                        "Execute o pipeline com este modelo primeiro."
                    )

        elif modo_operacao == "Executar Pipeline Completo":
            _tempo_estimado = {"LSTM": "~10–30 min", "GRU": "~8–25 min",
                               "MLP": "~5–15 min",  "KNN": "~1–3 min"}
            st.warning(
                f"**{selected_model_name}** — Tempo estimado: "
                f"{_tempo_estimado.get(selected_model_name, '?')}"
            )
            if st.button(
                f"🚀 Treinar {selected_model_name} + Walk-Forward",
                type="primary", use_container_width=True
            ):
                with st.spinner(f"Executando pipeline com {selected_model_name}..."):
                    st.session_state.pred_df = run_full_pipeline(
                        selected_asset, config, model_name=selected_model_name
                    )
                    st.session_state.loaded_model_name = selected_model_name
                st.success(f"✅ Pipeline {selected_model_name} concluído e resultados salvos!")

    if not st.session_state.pred_df.empty:
        pred_df = st.session_state.pred_df.copy()

        # Badge visual do modelo ativo
        _active_model = st.session_state.get("loaded_model_name", "LSTM")
        _model_colors = {"LSTM": "🔵", "GRU": "🟣", "MLP": "🟠", "KNN": "🟢"}
        _icon = _model_colors.get(_active_model, "⚪")
        st.subheader(f"{_icon} Resultados — Modelo: **{_active_model}**")

        # Se pred_df tem coluna model_name, exibe distribuição por janela
        if "model_name" in pred_df.columns:
            _models_in_df = pred_df["model_name"].unique()
            if len(_models_in_df) > 1:
                st.info(f"⚠️ Este CSV contém resultados de múltiplos modelos: {list(_models_in_df)}")

        # Atualiza configurações de risco dinamicamente
        current_config = config.copy()
        current_config["backtest"]["rejection_threshold"] = rejection_tau

        # Recalcula a rejeição com o novo τ em tempo real
        # O slider apenas atualiza a coluna 'rejected' (confidence < τ).
        # A decisão original do modelo (compra=1 / venda=0) é preservada —
        # ela foi calculada pelo argmax das probabilidades durante o walk-forward
        # e não depende de τ. Só o threshold de rejeição muda com o slider.
        if "confidence" in pred_df.columns:
            # Salva decisão original antes do slider modificar qualquer coisa
            if "_decision_original" not in pred_df.columns:
                pred_df["_decision_original"] = pred_df["decision"].copy()

            # Atualiza rejected com o τ atual do slider
            pred_df["rejected"] = pred_df["confidence"] < rejection_tau

            # Reconstrói decision: rejeição onde abaixo de τ, original onde aceito
            pred_df["decision"] = np.where(
                pred_df["rejected"],
                -1,
                pred_df["_decision_original"]
            )

        # --- APLICAÇÃO DO VETO DIRECIONAL AQUI ---
        # Aplica o Veto Direcional visualmente e logicamente após o slider do Tau
        if use_trend_fallback and "regime_guerra" in pred_df.columns:
            mask_venda_na_alta = (pred_df['decision'] == 0) & (pred_df['regime_guerra'] == 1)
            pred_df.loc[mask_venda_na_alta, 'decision'] = -1
        
            mask_compra_na_baixa = (pred_df['decision'] == 1) & (pred_df['regime_guerra'] == -1)
            pred_df.loc[mask_compra_na_baixa, 'decision'] = -1

        # Motor de backtesting (rápido) e extração de métricas
        bt_result = run_backtest(pred_df, df_price, current_config, use_trend_fallback=use_trend_fallback)
        metrics = compute_all_metrics(bt_result, pred_df, current_config)

        # --- Análise do Mecanismo de Rejeição (Curva A-R) ---
        st.subheader("📉 Análise do Mecanismo de Rejeição (Chow, 1970)")
        st.markdown(f"**Taxa atual de rejeição:** {metrics.rejection_rate:.2%} | **Acurácia (Aceitos):** {metrics.accuracy_accepted:.2%}")
    
        fig_ar = plot_ar_curve(
            thresholds=metrics.ar_thresholds,
            accuracies=metrics.ar_accuracies,
            rejection_rates=metrics.ar_rejection_rates,
            asset=selected_asset
        )
        st.pyplot(fig_ar)
        st.divider()

        # --- Painel de Métricas Financeiras ---
        st.subheader("💰 Desempenho Financeiro")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Retorno Total", f"{metrics.total_return_strategy:.2%}")
        col2.metric("CAGR", f"{metrics.cagr_strategy:.2%}")
        col3.metric("Índice de Sharpe", f"{metrics.sharpe_ratio:.2f}")
        col4.metric("Max Drawdown", f"{metrics.max_drawdown:.2%}")
        col5.metric("Taxa de Acerto", f"{metrics.hit_rate:.2%}")

        # Opção B: painel de métricas adicional (aparece só quando ativa)
        if use_trend_fallback:
            total_rejected = int((pred_df["decision"] == -1).sum()) if "decision" in pred_df.columns else bt_result.n_rejected
            fallback_pct   = bt_result.n_fallback / total_rejected if total_rejected > 0 else 0.0
            st.subheader("🔄 Opção B — Fallback por Tendência")
            col_fb1, col_fb2, col_fb3 = st.columns(3)
            col_fb1.metric(
                "Fallbacks Ativados",
                f"{bt_result.n_fallback} dias",
                help="Rejeições convertidas em LONG pelo Veto Direcional=+1 (Opção B).",
            )
            col_fb2.metric(
                "% das Rejeições",
                f"{fallback_pct:.1%}",
                help="Proporção de dias rejeitados que viraram LONG via fallback.",
            )
            col_fb3.metric(
                "Rejeições Puras",
                f"{bt_result.n_rejected - bt_result.n_fallback} dias",
                help="Dias que permaneceram em caixa mesmo com Opção B ativa.",
            )
            st.divider()

        # --- Gráficos de Capital, Retornos e Sinais ---
        st.subheader("📊 Curva de Capital Comparativa")
        fig_equity = plot_equity_curve(bt_result.equity_curve, bt_result.benchmark_curve, asset=selected_asset)
        st.pyplot(fig_equity)

        # NOVO: Histograma de Distribuição de Retornos
        st.subheader("⚖️ Distribuição de Retornos Diários")
        fig_returns = plot_return_distribution(
            bt_result.daily_returns, 
            bt_result.benchmark_returns, 
            asset=selected_asset
        )
        st.pyplot(fig_returns)

        st.subheader("🎯 Sinais e Preço (Últimos 252 pregões)")
        fig_signals = plot_signals_on_price(df_price, pred_df, asset=selected_asset, n_days=252)
        st.pyplot(fig_signals)

        with st.expander("Ver Tabela de Decisões e Probabilidades"):
            display_df = pred_df.copy()

            # Coluna: rótulo textual da decisão do modelo
            _decision_map = {1: "✅ Compra", 0: "🔴 Venda", -1: "⚠️ Rejeição"}
            display_df["decisão_modelo"] = display_df["decision"].map(_decision_map).fillna("—")

            # Coluna: regime_guerra com ícone visual
            if "regime_guerra" in display_df.columns:
                _regime_map = {1: "📈 Alta", 0: "➡️ Neutro", -1: "📉 Baixa"}
                display_df["regime"] = display_df["regime_guerra"].map(_regime_map).fillna("—")

            # Colunas de fallback (só quando Opção B ativa)
            if use_trend_fallback and "regime_guerra" in display_df.columns:
                display_df["fallback_ativado"] = (
                    (display_df["decision"] == -1) & (display_df["regime_guerra"] == 1)
                ).map({True: "🔄 Sim", False: ""})
                display_df["fallback_acum"] = (
                    (display_df["decision"] == -1) & (display_df["regime_guerra"] == 1)
                ).cumsum()

            # Seleciona e reordena colunas
            cols_base = ["decisão_modelo", "confidence", "rejected"]
            if "regime_guerra" in display_df.columns:
                cols_base += ["regime"]
            if use_trend_fallback and "fallback_ativado" in display_df.columns:
                cols_base += ["fallback_ativado", "fallback_acum"]
            cols_base += ["y_ret_true", "y_ret_pred", "window_id"]
            cols_show = [c for c in cols_base if c in display_df.columns]

            st.caption(
                f"Últimas 100 linhas · "
                f"{'🔄 Opção B ativa' if use_trend_fallback else '⛔ Opção B desativada'}"
            )
            st.dataframe(
                display_df[cols_show].tail(100),
                use_container_width=True,
                column_config={
                    "confidence":       st.column_config.ProgressColumn("Confiança", min_value=0.0, max_value=1.0, format="%.2f"),
                    "rejected":         st.column_config.CheckboxColumn("Rejeitado?"),
                    "fallback_ativado": st.column_config.TextColumn("Fallback?"),
                    "fallback_acum":    st.column_config.NumberColumn("Fallbacks (acum.)", format="%d"),
                    "y_ret_true":       st.column_config.NumberColumn("Retorno Real", format="%.4f"),
                    "y_ret_pred":       st.column_config.NumberColumn("Retorno Pred.", format="%.4f"),
                    "window_id":        st.column_config.NumberColumn("Janela WF", format="%d"),
                },
            )
        
        # ==========================================
        # NOVO: Exportação de Imagens
        # ==========================================
        st.divider()
        st.subheader("📸 Exportar Visualizações")
        col_btn1, col_btn2, col_btn3 = st.columns([2, 2, 1]) # Ajustei as proporções
    
        with col_btn1:
            if st.button("Salvar Gráficos na Pasta Local", use_container_width=True):
                with st.spinner("Salvando imagens localmente..."):
                    # Coleta o caminho absoluto de onde o app.py está salvo
                    base_dir = Path(__file__).parent.resolve()
                    fig_dir = base_dir / "resultados" / "figuras"
                
                    # Garante que o diretório existe
                    fig_dir.mkdir(parents=True, exist_ok=True)
                
                    # Salva cada figura gerada com o nome do ativo atual
                    fig_ar.savefig(fig_dir / f"03_ar_curve_{selected_asset}.png", bbox_inches="tight", dpi=150)
                    fig_equity.savefig(fig_dir / f"02_equity_curve_{selected_asset}.png", bbox_inches="tight", dpi=150)
                    fig_returns.savefig(fig_dir / f"04_return_distribution_{selected_asset}.png", bbox_inches="tight", dpi=150)
                    fig_signals.savefig(fig_dir / f"06_signals_{selected_asset}.png", bbox_inches="tight", dpi=150)
                
                st.success(f"✅ 4 gráficos salvos com sucesso em: `{fig_dir}`")
        with col_btn2:
            # Cria um buffer de memória para o arquivo ZIP
            zip_buffer = io.BytesIO()
        
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                # Função auxiliar para converter figura em bytes e adicionar ao zip
                def add_fig_to_zip(fig, filename):
                    img_buffer = io.BytesIO()
                    fig.savefig(img_buffer, format="png", bbox_inches="tight", dpi=150)
                    zip_file.writestr(filename, img_buffer.getvalue())
            
                # Adiciona os 4 gráficos ao ZIP em memória
                add_fig_to_zip(fig_ar, f"03_ar_curve_{selected_asset}.png")
                add_fig_to_zip(fig_equity, f"02_equity_curve_{selected_asset}.png")
                add_fig_to_zip(fig_returns, f"04_return_distribution_{selected_asset}.png")
                add_fig_to_zip(fig_signals, f"06_signals_{selected_asset}.png")
            
            # O Streamlit lida nativamente com o download de buffers de bytes
            st.download_button(
                label="📦 Baixar Gráficos (.zip)",
                data=zip_buffer.getvalue(),
                file_name=f"graficos_finais_{selected_asset}.zip",
                mime="application/zip",
                use_container_width=True
            )
    else:
        st.info("👆 Utilize o menu lateral e os botões acima para carregar ou processar os dados do modelo.")
