"""
gerar_graficos_comparativos.py
==============================
Gera 5 gráficos de barras comparando os modelos LSTM, GRU, MLP, KNN e Transformer
a partir do arquivo comparacao_bova11_lstm_gru_mlp_knn_transformer.json.

Gráficos gerados (1 por arquivo PNG):
    01_retorno_total.png          — Retorno Total (%)
    02_max_drawdown.png           — Máx. Drawdown (%)
    03_taxa_rejeicao.png          — Taxa de Rejeição (%)
    04_taxa_acerto.png            — Taxa de Acerto / Hit Rate (%)
    05_eficiencia_computacional.png — Tempo de execução (s)

Uso:
    python gerar_graficos_comparativos.py
    python gerar_graficos_comparativos.py --json resultados/comparacao_bova11.json
    python gerar_graficos_comparativos.py --json comparacao.json --out graficos/

Referências:
    Vaswani et al. (2017) — Attention Is All You Need
    Chow (1970) — On Optimum Recognition Error and Reject Tradeoff
"""

import json
import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Estilo acadêmico — compatível com dissertações e artigos CBA
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.30,
    "grid.linestyle":    "--",
    "legend.fontsize":   10,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})

# Paleta de cores: uma cor distinta por modelo
CORES = {
    "LSTM":        "#1565C0",   # azul escuro
    "GRU":         "#6A1B9A",   # roxo
    "MLP":         "#E65100",   # laranja
    "KNN":         "#2E7D32",   # verde escuro
    "TRANSFORMER": "#AD1457",   # rosa/vinho
}
COR_BENCHMARK = "#B71C1C"       # vermelho — linha de referência B&H


def _cor(modelo: str) -> str:
    return CORES.get(modelo.upper(), "#546E7A")


def _rotulo_modelo(modelo: str) -> str:
    """Rótulo legível para eixo: TRANSFORMER → Transformer."""
    return modelo.capitalize() if modelo.upper() == "TRANSFORMER" else modelo.upper()


def _salvar(fig, path: Path, nome: str):
    fig.savefig(path / nome)
    print(f"  ✓  {nome}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Função genérica de barra horizontal
# ─────────────────────────────────────────────────────────────────────────────
def grafico_barras(
    modelos:       list,
    valores:       list,
    titulo:        str,
    xlabel:        str,
    destaque_min:  bool = False,   # True → menor valor é o melhor (ex: drawdown, rejeição)
    linha_ref:     float = None,   # valor de referência vertical (ex: B&H)
    label_ref:     str   = None,
    fmt:           str   = "{:.1f}",
    sufixo:        str   = "",
    cor_positivo:  str   = None,   # se None, usa cor por modelo
    invertido:     bool  = False,  # True → valores negativos (MDD): barra para esquerda
) -> plt.Figure:
    """
    Gráfico de barras horizontais com anotação de valor em cada barra.
    O modelo com melhor resultado recebe destaque de borda dourada.
    """
    n = len(modelos)
    rotulos = [_rotulo_modelo(m) for m in modelos]

    # Identifica o melhor
    if destaque_min:
        idx_best = int(np.argmin(valores))
    else:
        idx_best = int(np.argmax(valores))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.suptitle(titulo, fontweight="bold", fontsize=13, y=1.01)

    y_pos = np.arange(n)

    for i, (m, val) in enumerate(zip(modelos, valores)):
        cor = _cor(m)
        borda_lw   = 2.5 if i == idx_best else 0.8
        borda_cor  = "#FFD600" if i == idx_best else "white"  # dourado = melhor

        # Para MDD (negativos), usa valor absoluto para comprimento da barra
        # mas mantém o valor real na anotação
        val_barra = abs(val) if invertido else val
        ax.barh(
            y_pos[i], val_barra,
            color=cor, alpha=0.82,
            edgecolor=borda_cor, linewidth=borda_lw,
            height=0.55,
        )

        # Anotação do valor dentro ou fora da barra
        texto = fmt.format(val) + sufixo
        offset = val_barra * 0.02
        ax.text(
            val_barra + offset, y_pos[i],
            texto,
            va="center", ha="left",
            fontsize=10,
            fontweight="bold" if i == idx_best else "normal",
            color="#333333",
        )

    # Linha de referência vertical (ex: B&H)
    if linha_ref is not None:
        ref_val = abs(linha_ref) if invertido else linha_ref
        ax.axvline(ref_val, color=COR_BENCHMARK, lw=1.8, ls="--", zorder=5)
        ax.text(
            ref_val, n - 0.05,
            label_ref or f"Ref: {linha_ref:.1f}",
            color=COR_BENCHMARK, fontsize=9,
            ha="left", va="top",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(rotulos, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=11)

    # Margem direita para caber anotações
    xmax = max(abs(v) for v in valores)
    ax.set_xlim(0, xmax * 1.22)

    # Legenda de destaque
    patch_best = mpatches.Patch(
        facecolor=_cor(modelos[idx_best]),
        edgecolor="#FFD600", linewidth=2,
        label=f"Melhor: {_rotulo_modelo(modelos[idx_best])}"
    )
    ax.legend(handles=[patch_best], loc="lower right", fontsize=9)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Gráfico especial para Drawdown (valores negativos — "menos negativo = melhor")
# ─────────────────────────────────────────────────────────────────────────────
def grafico_drawdown(modelos, valores, titulo, xlabel, linha_ref=None, label_ref=None):
    """
    Barras horizontais para MDD: quanto menor em módulo, melhor.
    Eixo mostra valores absolutos; anotação mostra o valor real (negativo).
    """
    n = len(modelos)
    rotulos = [_rotulo_modelo(m) for m in modelos]
    abs_vals = [abs(v) for v in valores]
    idx_best = int(np.argmin(abs_vals))   # menor queda = melhor

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.suptitle(titulo, fontweight="bold", fontsize=13, y=1.01)

    y_pos = np.arange(n)

    for i, (m, val, av) in enumerate(zip(modelos, valores, abs_vals)):
        cor = _cor(m)
        borda_lw  = 2.5 if i == idx_best else 0.8
        borda_cor = "#FFD600" if i == idx_best else "white"

        ax.barh(y_pos[i], av, color=cor, alpha=0.82,
                edgecolor=borda_cor, linewidth=borda_lw, height=0.55)

        texto = f"{val:.1f}%"
        ax.text(av * 1.02, y_pos[i], texto,
                va="center", ha="left", fontsize=10,
                fontweight="bold" if i == idx_best else "normal",
                color="#333333")

    if linha_ref is not None:
        ax.axvline(abs(linha_ref), color=COR_BENCHMARK, lw=1.8, ls="--", zorder=5)
        ax.text(abs(linha_ref), n - 0.05,
                label_ref or f"B&H: {linha_ref:.1f}%",
                color=COR_BENCHMARK, fontsize=9, ha="left", va="top")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(rotulos, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=11)

    xmax = max(abs_vals)
    ax.set_xlim(0, xmax * 1.22)

    patch_best = mpatches.Patch(
        facecolor=_cor(modelos[idx_best]),
        edgecolor="#FFD600", linewidth=2,
        label=f"Menor queda: {_rotulo_modelo(modelos[idx_best])}"
    )
    ax.legend(handles=[patch_best], loc="lower right", fontsize=9)

    # Zoom no range de variação — amplia diferença entre modelos
    margem = (xmax - min(abs_vals)) * 0.8
    xmin_zoom = max(0, min(abs_vals) - margem)
    ax.set_xlim(xmin_zoom, xmax * 1.22)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Gráfico especial para Eficiência Computacional (escala log)
# ─────────────────────────────────────────────────────────────────────────────
def grafico_tempo(modelos, valores, titulo):
    n = len(modelos)
    rotulos = [_rotulo_modelo(m) for m in modelos]
    idx_best = int(np.argmin(valores))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.suptitle(titulo, fontweight="bold", fontsize=13, y=1.01)

    y_pos = np.arange(n)

    for i, (m, val) in enumerate(zip(modelos, valores)):
        cor = _cor(m)
        borda_lw  = 2.5 if i == idx_best else 0.8
        borda_cor = "#FFD600" if i == idx_best else "white"

        ax.barh(y_pos[i], val, color=cor, alpha=0.82,
                edgecolor=borda_cor, linewidth=borda_lw, height=0.55)

        ax.text(val * 1.03, y_pos[i], f"{val:.1f} s",
                va="center", ha="left", fontsize=10,
                fontweight="bold" if i == idx_best else "normal",
                color="#333333")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(rotulos, fontsize=11)
    ax.set_xlabel("Tempo total — 30 janelas walk-forward (s)", fontsize=11)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.0f} s")
    )
    xmax = max(valores)
    ax.set_xlim(1, xmax * 3)

    # Anotação de speedup em relação ao mais lento
    mais_lento = max(valores)
    for i, val in enumerate(valores):
        speedup = mais_lento / val
        if speedup > 1.5:
            ax.text(xmax * 2.2, y_pos[i],
                    f"{speedup:.0f}× mais rápido",
                    va="center", ha="right", fontsize=8.5,
                    color="#555555", style="italic")

    patch_best = mpatches.Patch(
        facecolor=_cor(modelos[idx_best]),
        edgecolor="#FFD600", linewidth=2,
        label=f"Mais rápido: {_rotulo_modelo(modelos[idx_best])}"
    )
    ax.legend(handles=[patch_best], loc="lower right", fontsize=9)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────
def gerar_todos(json_path: str, output_dir: str):
    # 1. Carrega dados
    with open(json_path, encoding="utf-8") as f:
        dados = json.load(f)

    asset   = dados.get("asset", "BOVA11")
    modelos_dict = dados["modelos"]
    modelos = list(modelos_dict.keys())

    bh_ret = modelos_dict[modelos[0]]["total_return_benchmark_pct"]
    bh_mdd = -46.9   # MDD do B&H aproximado

    def vals(campo):
        return [modelos_dict[m][campo] for m in modelos]

    # 2. Cria diretório de saída
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\nAtivo : {asset}")
    print(f"Modelos: {modelos}")
    print(f"Saída  : {out.resolve()}\n")

    # ── Gráfico 1: Retorno Total ──────────────────────────────────────────────
    fig = grafico_barras(
        modelos   = modelos,
        valores   = vals("total_return_strategy_pct"),
        titulo    = f"Retorno Total Acumulado — {asset}",
        xlabel    = "Retorno Total (%)",
        linha_ref = bh_ret,
        label_ref = f"B&H: {bh_ret:.1f}%",
        fmt       = "{:.2f}",
        sufixo    = "%",
    )
    _salvar(fig, out, "01_retorno_total.png")

    # ── Gráfico 2: Máximo Drawdown ────────────────────────────────────────────
    fig = grafico_drawdown(
        modelos   = modelos,
        valores   = vals("max_drawdown_pct"),
        titulo    = f"Máximo Drawdown — {asset}",
        xlabel    = "Drawdown (%, módulo — menor = melhor)",
        linha_ref = bh_mdd,
        label_ref = f"B&H: {bh_mdd:.1f}%",
    )
    _salvar(fig, out, "02_max_drawdown.png")

    # ── Gráfico 3: Taxa de Rejeição ───────────────────────────────────────────
    fig = grafico_barras(
        modelos      = modelos,
        valores      = vals("rejection_rate_pct"),
        titulo       = f"Taxa de Rejeição — {asset}",
        xlabel       = "Taxa de Rejeição (%) — menor = mais operações",
        destaque_min = True,
        fmt          = "{:.2f}",
        sufixo       = "%",
    )
    _salvar(fig, out, "03_taxa_rejeicao.png")

    # ── Gráfico 4: Taxa de Acerto ─────────────────────────────────────────────
    fig = grafico_barras(
        modelos   = modelos,
        valores   = vals("hit_rate_pct"),
        titulo    = f"Taxa de Acerto (Hit Rate) — {asset}",
        xlabel    = "Taxa de Acerto (%) — operações fechadas com lucro",
        linha_ref = 50.0,
        label_ref = "Baseline 50%",
        fmt       = "{:.2f}",
        sufixo    = "%",
    )
    _salvar(fig, out, "04_taxa_acerto.png")

    # ── Gráfico 5: Eficiência Computacional ───────────────────────────────────
    fig = grafico_tempo(
        modelos = modelos,
        valores = vals("tempo_seg"),
        titulo  = f"Eficiência Computacional — {asset}\n"
                  "(30 janelas walk-forward, treinamento + inferência)",
    )
    _salvar(fig, out, "05_eficiencia_computacional.png")

    print(f"\n5 gráficos salvos em: {out.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
# Ponto de entrada
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gera gráficos comparativos a partir do JSON de comparação de modelos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python gerar_graficos_comparativos.py
  python gerar_graficos_comparativos.py --json resultados/comparacao_bova11_lstm_gru_mlp_knn_transformer.json
  python gerar_graficos_comparativos.py --json comparacao.json --out graficos/bova11/
        """
    )
    parser.add_argument(
        "--json",
        default="resultados/comparacao_bova11_lstm_gru_mlp_knn_transformer.json",
        help="Caminho do JSON de comparação (padrão: resultados/comparacao_bova11_lstm_gru_mlp_knn_transformer.json)"
    )
    parser.add_argument(
        "--out",
        default="resultados/figuras/comparativo",
        help="Diretório de saída dos PNGs (padrão: resultados/figuras/comparativo)"
    )
    args = parser.parse_args()
    gerar_todos(args.json, args.out)
