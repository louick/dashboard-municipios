from __future__ import annotations

from pathlib import Path
import os
import numpy as np
import pandas as pd

from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
from dash import dash_table

import plotly.express as px


# =========================
# Config
# =========================
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_XLSX = BASE_DIR / "data" / "Base de Dados - 144 Municípios.xlsx"
XLSX_PATH = Path(os.getenv("XLSX_PATH", str(DEFAULT_XLSX))).expanduser().resolve()

MUNICIPAL_IBGE_MIN = 1500000
MUNICIPAL_IBGE_MAX = 1600000  # exclusivo

YEAR_OFFSET = 2  # <-- ANO EXIBIDO = ANO REAL + 2

# Financeiro: exibir em milhões (para KPIs e gráficos financeiros)
MONEY_SCALE = 1_000_000  # 1 milhão

# Branding (inspirado no topo Gov PA / Fapespa)
BRAND_BLUE = "#0B4D8C"
BRAND_BLUE_DARK = "#083A69"
BRAND_RED = "#E31B23"
BRAND_BG = "#F4F7FB"
BRAND_CARD = "#FFFFFF"
BRAND_TEXT = "#0B1320"
BRAND_MUTED = "#5B6B7C"
BRAND_BORDER = "#E6ECF5"


def display_year(ano_real: int) -> int:
    return int(ano_real) + YEAR_OFFSET


def apply_year_ticks(fig, anos_reais: list[int], axis_title: str = ""):
    """
    Elimina frações de ano e mostra ANO EXIBIDO (ano real + YEAR_OFFSET).
    """
    anos_reais = sorted([int(a) for a in anos_reais if a is not None])
    fig.update_xaxes(
        tickmode="array",
        tickvals=anos_reais,
        ticktext=[str(display_year(a)) for a in anos_reais],
        dtick=1,
        type="linear",
        title=axis_title,
        showgrid=True,
        gridcolor=BRAND_BORDER,
        zeroline=False,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=BRAND_BORDER,
        zeroline=False,
    )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=BRAND_CARD,
        plot_bgcolor=BRAND_CARD,
        separators=".,",  # pt-BR: milhar "." e decimal ","
        font=dict(
            family="system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Arial",
            color=BRAND_TEXT,
        ),
    )
    return fig


# =========================
# Helpers
# =========================
def format_ptbr_number(x: float | int, decimals: int = 2) -> str:
    # 1234567.89 -> "1.234.567,89"
    s = f"{float(x):,.{decimals}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def format_number_mi(v: float | int | None, decimals: int = 1) -> str:
    """
    Formata número já em milhões (sem prefixo "R$" — fica mais limpo pros cards).
    Ex: 541.7 -> "541,7"
    """
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return format_ptbr_number(float(v), decimals=decimals)


def format_value(indicador: str, v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"

    if isinstance(v, (str, bool)):
        return str(v)

    if isinstance(v, (np.integer, np.floating)):
        v = v.item()

    if isinstance(v, (int, float)):
        ind_upper = (indicador or "").upper()
        if "R$" in ind_upper:
            return f"R$ {format_ptbr_number(v)}"
        if "PERCENT" in ind_upper or "%" in indicador:
            return f"{format_ptbr_number(v)}%"
        if float(v).is_integer():
            return f"{int(v):,}".replace(",", ".")
        return format_ptbr_number(v)

    return str(v)


def only_municipios_from_indicador_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Abas com:
    - linha "Pará"
    - linhas "RI Xxxxx"
    - depois municípios.
    Regra: manter apenas IBGE municipal (15xxxxxx), remover RI/Pará.
    """
    if "Código IBGE" not in df.columns or "Indicador" not in df.columns:
        return df.copy()

    ibge = pd.to_numeric(df["Código IBGE"], errors="coerce")
    ibge_int = ibge.fillna(0).astype(int)

    nome = df["Indicador"].astype(str).str.strip()
    is_para = nome.str.upper().isin(["PARÁ", "PARA"])
    is_ri = nome.str.upper().str.startswith("RI ")

    is_muni = (ibge_int >= MUNICIPAL_IBGE_MIN) & (ibge_int < MUNICIPAL_IBGE_MAX)

    out = df[is_muni & (~is_para) & (~is_ri)].copy()
    out["Código IBGE"] = ibge_int[is_muni & (~is_para) & (~is_ri)]
    return out


def only_municipios_from_nome_municipio_sheet(df: pd.DataFrame) -> pd.DataFrame:
    if "Código IBGE" not in df.columns:
        return df.copy()
    ibge = pd.to_numeric(df["Código IBGE"], errors="coerce").fillna(0).astype(int)
    is_muni = (ibge >= MUNICIPAL_IBGE_MIN) & (ibge < MUNICIPAL_IBGE_MAX)
    out = df[is_muni].copy()
    out["Código IBGE"] = ibge[is_muni]
    return out


def wide_to_kv(df: pd.DataFrame, municipio: str, nome_col: str) -> pd.DataFrame:
    """
    Pega 1 linha (município) e transforma em tabela 2-colunas: Indicador / Valor.
    """
    row = df.loc[df[nome_col] == municipio]
    if row.empty:
        return pd.DataFrame([{"Indicador": "—", "Valor": "Município não encontrado nessa aba"}])

    row = row.iloc[0].to_dict()

    # Remover chaves técnicas / metadados (inclui IBGE para não aparecer como “Indicador”)
    for k in list(row.keys()):
        if k in ["Código IBGE", "IBGE", "R. Integ.", "Indicador", "Nome_Município"]:
            row.pop(k, None)

    items = [{"Indicador": str(k), "Valor": format_value(str(k), v)} for k, v in row.items()]
    return pd.DataFrame(items)


def melt_years(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """
    Converte abas tipo Receita/Despesa/FPM (colunas 2019..2023) para formato longo.
    """
    df = df.copy()
    df = df.loc[:, [c for c in df.columns if not (isinstance(c, str) and c.startswith("Unnamed"))]]
    df = only_municipios_from_nome_municipio_sheet(df)

    year_cols = [c for c in df.columns if isinstance(c, (int, np.integer))]
    year_cols = sorted(year_cols)

    base_cols = []
    for c in ["Código IBGE", "Nome_Município"]:
        if c in df.columns:
            base_cols.append(c)

    out = df.melt(
        id_vars=base_cols,
        value_vars=year_cols,
        var_name="Ano",
        value_name=value_name,
    )

    out.rename(columns={"Código IBGE": "IBGE", "Nome_Município": "Municipio"}, inplace=True)
    out["IBGE"] = pd.to_numeric(out["IBGE"], errors="coerce").fillna(0).astype(int)
    out["Ano"] = pd.to_numeric(out["Ano"], errors="coerce").astype(int)
    out[value_name] = pd.to_numeric(out[value_name], errors="coerce")
    return out


# =========================
# Load Data
# =========================
if not XLSX_PATH.exists():
    raise FileNotFoundError(
        f"Não achei a planilha em: {XLSX_PATH}\n"
        f"Coloque o arquivo em: {DEFAULT_XLSX}\n"
        f"Ou rode com: XLSX_PATH=/caminho/arquivo.xlsx python app.py"
    )

SHEETS_INDICADOR = {
    "Geral": "Geral",
    "Economia 01": "Economia 01",
    "Economia 02": "Economia 02",
    "Infraest. 01": "Infraest. 01",
    "Turismo - Empreendimentos": "Turismo - Empreendimentos",
    "Turismo - Empregos": "Turismo - Empregos",
}

dfs_kv: dict[str, pd.DataFrame] = {}
for label, sheet in SHEETS_INDICADOR.items():
    df = pd.read_excel(XLSX_PATH, sheet_name=sheet)
    df = only_municipios_from_indicador_sheet(df)
    df.rename(columns={"Código IBGE": "IBGE", "Indicador": "Municipio"}, inplace=True)
    dfs_kv[label] = df

df_receita_wide = pd.read_excel(XLSX_PATH, sheet_name="Receita")
df_despesa_wide = pd.read_excel(XLSX_PATH, sheet_name="Despesa")
df_fpm_wide = pd.read_excel(XLSX_PATH, sheet_name="FPM")

receita_long = melt_years(df_receita_wide, "Receita")
despesa_long = melt_years(df_despesa_wide, "Despesa")
fpm_long = melt_years(df_fpm_wide, "FPM")

MUNICIPIOS = sorted(receita_long["Municipio"].dropna().unique().tolist())
ANOS = sorted(receita_long["Ano"].dropna().unique().tolist())
DEFAULT_MUNI = MUNICIPIOS[0] if MUNICIPIOS else None
DEFAULT_ANO = max(ANOS) if ANOS else None


# =========================
# UI Components
# =========================
def make_kv_table_component(table_id: str, title: str) -> html.Div:
    return html.Div(
        className="block-card",
        children=[
            html.Div(
                className="block-head",
                children=[
                    html.Div(title, className="block-title"),
                ],
            ),
            dash_table.DataTable(
                id=table_id,
                columns=[
                    {"name": "Indicador", "id": "Indicador"},
                    {"name": "Valor", "id": "Valor"},
                ],
                data=[],
                page_size=18,
                style_table={"overflowX": "auto"},
                style_cell={
                    "textAlign": "left",
                    "padding": "10px",
                    "whiteSpace": "normal",
                    "height": "auto",
                    "fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Arial",
                    "fontSize": "14px",
                    "border": f"1px solid {BRAND_BORDER}",
                },
                style_header={
                    "fontWeight": "800",
                    "backgroundColor": BRAND_BLUE,
                    "color": "white",
                    "border": f"1px solid {BRAND_BLUE}",
                },
                style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#FBFCFF"}],
            ),
        ],
    )


def ranking_fig(df_long: pd.DataFrame, value_col: str, ano: int, municipio: str, top_n: int = 20):
    if ano is None or municipio is None:
        fig = px.bar(title="—")
        fig.update_layout(height=420)
        return fig

    dfy = df_long[df_long["Ano"] == ano].copy()
    dfy = dfy.dropna(subset=[value_col])
    dfy = dfy.sort_values(value_col, ascending=False)

    top = dfy.head(top_n).copy()

    sel = dfy[dfy["Municipio"] == municipio]
    if not sel.empty and municipio not in top["Municipio"].values:
        top = pd.concat([top, sel.iloc[[0]]], ignore_index=True)

    top["_sel"] = np.where(top["Municipio"] == municipio, "Selecionado", "Outros")
    top = top.sort_values(value_col, ascending=True)

    # valores em milhões
    top["_value_mi"] = pd.to_numeric(top[value_col], errors="coerce") / MONEY_SCALE

    fig = px.bar(
        top,
        x="_value_mi",
        y="Municipio",
        orientation="h",
        color="_sel",
        color_discrete_map={"Selecionado": BRAND_RED, "Outros": BRAND_BLUE},
        title="",
    )

    fig.update_traces(hovertemplate="%{y}<br>R$ %{x:,.1f}<extra></extra>")

    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        height=520,
        showlegend=False,
        template="plotly_white",
        paper_bgcolor=BRAND_CARD,
        plot_bgcolor=BRAND_CARD,
        separators=".,",
        font=dict(
            family="system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Arial",
            color=BRAND_TEXT,
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=BRAND_BORDER,
        zeroline=False,
        title="R$ milhões",
        tickformat=",.0f",
    )
    fig.update_yaxes(showgrid=False, title="")
    return fig


def series_fig_money_mi(df_long: pd.DataFrame, value_col: str, municipio: str, title: str):
    """
    Série (financeiro) em R$ milhões.
    """
    if municipio is None:
        fig = px.line(title="—")
        fig.update_layout(height=360)
        return fig

    d = df_long[df_long["Municipio"] == municipio].copy()
    d = d.sort_values("Ano")
    d["_value_mi"] = pd.to_numeric(d[value_col], errors="coerce") / MONEY_SCALE
    d["_ano_exib"] = d["Ano"].apply(display_year).astype(int)

    fig = px.line(d, x="Ano", y="_value_mi", markers=True, title=title)
    fig.update_layout(margin=dict(l=10, r=10, t=70, b=10), height=360)

    fig.update_traces(
        name=value_col,
        line=dict(width=3, color=BRAND_BLUE),
        marker=dict(size=8, color=BRAND_BLUE),
        customdata=d["_ano_exib"],
        hovertemplate=f"{value_col}<br>Ano: %{{customdata}}<br>R$ %{{y:,.1f}}<extra></extra>",
    )

    apply_year_ticks(fig, ANOS, axis_title="")
    fig.update_yaxes(title="R$ milhões", tickformat=",.0f")
    fig.update_layout(legend_title_text="")
    return fig


def get_value(df_long: pd.DataFrame, value_col: str, municipio: str, ano: int) -> float | None:
    d = df_long[(df_long["Municipio"] == municipio) & (df_long["Ano"] == ano)]
    if d.empty:
        return None
    v = d.iloc[0][value_col]
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return float(v)


# =========================
# App
# =========================
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
    title="PEV — Municípios (PA)",
)
server = app.server

INLINE_CSS = f"""
:root {{
  --brand-blue: {BRAND_BLUE};
  --brand-blue-dark: {BRAND_BLUE_DARK};
  --brand-red: {BRAND_RED};
  --bg: {BRAND_BG};
  --card: {BRAND_CARD};
  --text: {BRAND_TEXT};
  --muted: {BRAND_MUTED};
  --border: {BRAND_BORDER};
}}

html, body {{ background: var(--bg); }}

.topbar {{
  background: linear-gradient(90deg, var(--brand-blue-dark), var(--brand-blue));
  color: #fff;
  border-bottom: 4px solid var(--brand-red);
  padding: 14px 18px;
  border-radius: 18px;
  box-shadow: 0 10px 30px rgba(8, 58, 105, 0.18);
}}

.topbar .row1 {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}}

.brand-left {{
  display: flex;
  align-items: center;
  gap: 12px;
}}

.badge-dot {{
  width: 42px;
  height: 42px;
  border-radius: 14px;
  background: rgba(255,255,255,0.14);
  border: 1px solid rgba(255,255,255,0.22);
  display: grid;
  place-items: center;
}}

.badge-dot i {{ font-size: 18px; }}

.topbar .title {{
  font-size: 20px;
  font-weight: 900;
  letter-spacing: .2px;
  margin: 0;
}}

.topbar .subtitle {{
  margin-top: 2px;
  font-size: 12px;
  opacity: .92;
}}

.pill {{
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border-radius: 999px;
  background: rgba(255,255,255,0.12);
  border: 1px solid rgba(255,255,255,0.22);
  font-size: 12px;
  white-space: nowrap;
}}

.shell {{ margin-top: 14px; }}

.sidebar-card {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 18px;
  box-shadow: 0 12px 28px rgba(17, 33, 56, 0.08);
  position: sticky;
  top: 14px;
}}

.side-title {{
  font-weight: 900;
  font-size: 16px;
  margin-bottom: 4px;
}}

.side-muted {{
  color: var(--muted);
  font-size: 12px;
}}

.side-section {{ margin-top: 14px; }}

.side-label {{
  font-weight: 800;
  font-size: 12px;
  color: var(--text);
  margin-bottom: 6px;
}}

.kpi-grid {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
}}

.kpi-card {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 18px;
  box-shadow: 0 12px 28px rgba(17, 33, 56, 0.06);
}}

.kpi-card .kpi-label {{
  color: var(--muted);
  font-weight: 800;
  font-size: 12px;
}}

.kpi-card .kpi-value {{
  font-size: 20px;
  font-weight: 900;
  margin-top: 4px;
  white-space: nowrap;
}}

.block-card {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 18px;
  box-shadow: 0 12px 28px rgba(17, 33, 56, 0.06);
  padding: 14px;
}}

.block-head {{
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 10px;
}}

.block-title {{ font-weight: 900; font-size: 16px; }}
.block-subtitle {{ color: var(--muted); font-size: 12px; }}

.tabs-wrap {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 18px;
  box-shadow: 0 12px 28px rgba(17, 33, 56, 0.06);
  padding: 8px;
}}

.custom-tabs {{ padding: 4px; }}

.custom-tab {{
  border: none !important;
  border-radius: 14px !important;
  padding: 10px 12px !important;
  font-weight: 800 !important;
  color: var(--muted) !important;
  background: transparent !important;
}}

.custom-tab--selected {{
  background: rgba(11, 77, 140, 0.10) !important;
  color: var(--brand-blue) !important;
  border: 1px solid rgba(11, 77, 140, 0.18) !important;
}}

hr {{ border-color: var(--border); }}
"""

INDEX_TEMPLATE = """
<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>/*__INLINE_CSS__*/</style>
  </head>
  <body>
    {%app_entry%}
    <footer>
      {%config%}
      {%scripts%}
      {%renderer%}
    </footer>
  </body>
</html>
"""
app.index_string = INDEX_TEMPLATE.replace("/*__INLINE_CSS__*/", INLINE_CSS)

topbar = html.Div(
    className="topbar",
    children=[
        html.Div(
            className="row1",
            children=[
                html.Div(
                    className="brand-left",
                    children=[
                        html.Div(className="badge-dot", children=[html.I(className="bi bi-bar-chart-line-fill")]),
                        html.Div(
                            children=[
                                html.Div("PEV 2025 • Perfis Econômicos Vocacionais", className="title"),
                                html.Div("Municípios Paraenses • somente municípios (RI removido)", className="subtitle"),
                            ]
                        ),
                    ],
                ),
                html.Div(),  # (vazio) — removido texto bobo do ano
            ],
        ),
    ],
)

sidebar = dbc.Card(
    className="sidebar-card",
    children=[
        dbc.CardBody(
            [
                html.Div(className="side-title", children=["Controle"]),
                html.Div(className="side-muted", children=["Selecione o município e o ano das contas públicas."]),
                html.Hr(),
                html.Div(
                    className="side-section",
                    children=[
                        html.Div(className="side-label", children=[html.I(className="bi bi-geo-alt-fill me-2"), "Município"]),
                        dcc.Dropdown(
                            id="municipio",
                            options=[{"label": m, "value": m} for m in MUNICIPIOS],
                            value=DEFAULT_MUNI,
                            clearable=False,
                            searchable=True,
                            placeholder="Selecione um município…",
                        ),
                        html.Div(id="meta_municipio", style={"marginTop": "10px"}),
                    ],
                ),
            ]
        )
    ],
)

def year_dropdown(id_, value):
    return dcc.Dropdown(
        id=id_,
        options=[{"label": str(display_year(a)), "value": a} for a in ANOS],
        value=value,
        clearable=False,
    )

def wrap_graph(graph_id: str, height: int | None = None):
    g = dcc.Graph(id=graph_id, style={"height": f"{height}px"} if height else {})
    return dbc.Spinner(html.Div(g), color="primary", delay_show=150)

# =========================
# TABS (AGRUPADO)
# =========================
tabs = dcc.Tabs(
    id="tabs",
    value="contas_publicas",
    className="custom-tabs",
    parent_className="custom-tabs",
    children=[
        dcc.Tab(
            label="Contas Públicas",
            value="contas_publicas",
            className="custom-tab",
            selected_className="custom-tab custom-tab--selected",
            children=[
                html.Br(),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("Ano", className="side-label"),
                                year_dropdown("ano_resumo", DEFAULT_ANO),
                            ],
                            md=4,
                        ),
                    ]
                ),
                html.Br(),

                html.Div(
                    className="kpi-grid",
                    children=[
                        dbc.Card(
                            dbc.CardBody([
                                html.Div("Receita (R$ mi)", className="kpi-label"),
                                html.Div(id="card_receita", className="kpi-value"),
                            ]),
                            className="kpi-card",
                        ),
                        dbc.Card(
                            dbc.CardBody([
                                html.Div("Despesa (R$ mi)", className="kpi-label"),
                                html.Div(id="card_despesa", className="kpi-value"),
                            ]),
                            className="kpi-card",
                        ),
                        dbc.Card(
                            dbc.CardBody([
                                html.Div("FPM (R$ mi)", className="kpi-label"),
                                html.Div(id="card_fpm", className="kpi-value"),
                            ]),
                            className="kpi-card",
                        ),
                        dbc.Card(
                            dbc.CardBody([
                                html.Div("Saldo (R$ mi)", className="kpi-label"),
                                html.Div(id="card_saldo", className="kpi-value"),
                            ]),
                            className="kpi-card",
                        ),
                    ],
                ),
                html.Br(),

                html.Div(className="block-card", children=[wrap_graph("fig_receita_vs_despesa")]),
                html.Br(),

                html.Div(
                    className="block-card",
                    children=[
                        html.Div(className="block-head", children=[
                            html.Div("Ranking do ano selecionado — Top 20", className="block-title"),
                            html.Div("Receita • Despesa • FPM", className="block-subtitle"),
                        ]),
                        dbc.Row(
                            [
                                dbc.Col(wrap_graph("fig_rank_receita_resumo"), md=4),
                                dbc.Col(wrap_graph("fig_rank_despesa_resumo"), md=4),
                                dbc.Col(wrap_graph("fig_rank_fpm_resumo"), md=4),
                            ],
                            className="g-2",
                        ),
                    ],
                ),
                html.Br(),

                html.Div(
                    className="block-card",
                    children=[
                        html.Div(className="block-head", children=[
                            html.Div("Séries históricas", className="block-title"),
                            html.Div("Valores em R$ milhões", className="block-subtitle"),
                        ]),
                        dbc.Row(
                            [
                                dbc.Col(wrap_graph("fig_series_receita_resumo"), md=4),
                                dbc.Col(wrap_graph("fig_series_despesa_resumo"), md=4),
                                dbc.Col(wrap_graph("fig_series_fpm_resumo"), md=4),
                            ],
                            className="g-2",
                        ),
                    ],
                ),
                html.Br(),

                html.Div(className="block-card", children=[wrap_graph("fig_fpm_series")]),
            ],
        ),

        dcc.Tab(
            label="Síntese Municipal",
            value="sintese_municipal",
            className="custom-tab",
            selected_className="custom-tab custom-tab--selected",
            children=[
                html.Br(),
                make_kv_table_component("table_geral", "Geral (por município)"),
            ],
        ),

        dcc.Tab(
            label="Síntese Econômica",
            value="sintese_economica",
            className="custom-tab",
            selected_className="custom-tab custom-tab--selected",
            children=[
                html.Br(),
                dbc.Row(
                    [
                        dbc.Col(make_kv_table_component("table_eco1", "Economia 01 (por município)"), md=6),
                        dbc.Col(make_kv_table_component("table_eco2", "Economia 02 (por município)"), md=6),
                    ],
                    className="g-2",
                ),
            ],
        ),

        dcc.Tab(
            label="Infraestrutura",
            value="infraestrutura",
            className="custom-tab",
            selected_className="custom-tab custom-tab--selected",
            children=[
                html.Br(),
                make_kv_table_component("table_inf1", "Infraest. 01 (por município)"),
            ],
        ),

        dcc.Tab(
            label="Turismo",
            value="turismo",
            className="custom-tab",
            selected_className="custom-tab custom-tab--selected",
            children=[
                html.Br(),
                dbc.Row(
                    [
                        dbc.Col(make_kv_table_component("table_tur1", "Turismo — Empreendimentos (por município)"), md=6),
                        dbc.Col(make_kv_table_component("table_tur2", "Turismo — Empregos (por município)"), md=6),
                    ],
                    className="g-2",
                ),
            ],
        ),
    ],
)

app.layout = dbc.Container(
    fluid=True,
    children=[
        html.Br(),
        topbar,
        html.Div(
            className="shell",
            children=[
                dbc.Row(
                    [
                        dbc.Col(sidebar, md=3),
                        dbc.Col(html.Div(className="tabs-wrap", children=[tabs]), md=9),
                    ],
                    style={"minHeight": "92vh"},
                ),
            ],
        ),
        html.Br(),
    ],
)


# =========================
# Callbacks
# =========================
@app.callback(Output("meta_municipio", "children"), Input("municipio", "value"))
def update_meta(municipio: str):
    if not municipio:
        return "—"
    row = receita_long[receita_long["Municipio"] == municipio]
    if row.empty:
        return "—"
    ibge = int(row.iloc[0]["IBGE"])
    return html.Div(className="pill", children=[html.I(className="bi bi-hash"), html.Span(f"IBGE: {ibge}")])


@app.callback(
    Output("card_receita", "children"),
    Output("card_despesa", "children"),
    Output("card_fpm", "children"),
    Output("card_saldo", "children"),
    Output("fig_receita_vs_despesa", "figure"),
    Output("fig_fpm_series", "figure"),
    Output("fig_rank_receita_resumo", "figure"),
    Output("fig_rank_despesa_resumo", "figure"),
    Output("fig_rank_fpm_resumo", "figure"),
    Output("fig_series_receita_resumo", "figure"),
    Output("fig_series_despesa_resumo", "figure"),
    Output("fig_series_fpm_resumo", "figure"),
    Input("municipio", "value"),
    Input("ano_resumo", "value"),
)
def update_contas_publicas(municipio: str, ano: int):
    r = get_value(receita_long, "Receita", municipio, ano)
    d = get_value(despesa_long, "Despesa", municipio, ano)
    f = get_value(fpm_long, "FPM", municipio, ano)

    r_mi = None if r is None else (r / MONEY_SCALE)
    d_mi = None if d is None else (d / MONEY_SCALE)
    f_mi = None if f is None else (f / MONEY_SCALE)

    r_txt = format_number_mi(r_mi, decimals=1)
    d_txt = format_number_mi(d_mi, decimals=1)
    f_txt = format_number_mi(f_mi, decimals=1)

    saldo = None if (r is None or d is None) else (r - d)
    saldo_mi = None if saldo is None else (saldo / MONEY_SCALE)
    s_txt = format_number_mi(saldo_mi, decimals=1)

    # Série Receita vs Despesa (em milhões)
    dd = pd.merge(
        receita_long[receita_long["Municipio"] == municipio][["Ano", "Receita"]],
        despesa_long[despesa_long["Municipio"] == municipio][["Ano", "Despesa"]],
        on="Ano",
        how="outer",
    ).sort_values("Ano")

    dd["Receita_mi"] = pd.to_numeric(dd["Receita"], errors="coerce") / MONEY_SCALE
    dd["Despesa_mi"] = pd.to_numeric(dd["Despesa"], errors="coerce") / MONEY_SCALE
    dd["_ano_exib"] = dd["Ano"].apply(display_year).astype(int)

    fig_rd = px.line(
        dd,
        x="Ano",
        y=["Receita_mi", "Despesa_mi"],
        markers=True,
        title="Série histórica — Receita vs Despesa<br><sup>Valores em R$ milhões</sup>",
    )
    fig_rd.update_layout(margin=dict(l=10, r=10, t=70, b=10), height=420, separators=".,")
    fig_rd.update_layout(legend_title_text="")
    fig_rd.update_yaxes(title="", tickformat=",.0f")
    fig_rd.update_xaxes(title="")

    # Cores e hover (Ano = ano exibido, sem texto explicando)
    for tr in fig_rd.data:
        if tr.name == "Receita_mi":
            tr.name = "Receita"
            tr.line.color = BRAND_BLUE
            tr.marker.color = BRAND_BLUE
        elif tr.name == "Despesa_mi":
            tr.name = "Despesa"
            tr.line.color = BRAND_RED
            tr.marker.color = BRAND_RED
        tr.line.width = 3
        tr.marker.size = 8
        tr.customdata = dd["_ano_exib"]
        tr.hovertemplate = "%{fullData.name}<br>Ano: %{customdata}<br>R$ %{y:,.1f}<extra></extra>"

    apply_year_ticks(fig_rd, ANOS, axis_title="")

    # FPM grande (série em milhões)
    fig_f_big = series_fig_money_mi(
        fpm_long,
        "FPM",
        municipio,
        "Série histórica — FPM<br><sup>Valores em R$ milhões</sup>",
    )
    fig_f_big.update_layout(height=420)

    # Rankings
    fig_rank_r = ranking_fig(receita_long, "Receita", ano, municipio, top_n=20)
    fig_rank_r.update_layout(height=420, title="Receita")

    fig_rank_d = ranking_fig(despesa_long, "Despesa", ano, municipio, top_n=20)
    fig_rank_d.update_layout(height=420, title="Despesa")

    fig_rank_f = ranking_fig(fpm_long, "FPM", ano, municipio, top_n=20)
    fig_rank_f.update_layout(height=420, title="FPM")

    # Séries
    fig_sr = series_fig_money_mi(receita_long, "Receita", municipio, "Receita — Série histórica<br><sup>Valores em R$ milhões</sup>")
    fig_sd = series_fig_money_mi(despesa_long, "Despesa", municipio, "Despesa — Série histórica<br><sup>Valores em R$ milhões</sup>")
    fig_sf = series_fig_money_mi(fpm_long, "FPM", municipio, "FPM — Série histórica<br><sup>Valores em R$ milhões</sup>")

    return (
        r_txt, d_txt, f_txt, s_txt,
        fig_rd, fig_f_big,
        fig_rank_r, fig_rank_d, fig_rank_f,
        fig_sr, fig_sd, fig_sf,
    )


def _update_kv_table(sheet_label: str, municipio: str):
    df = dfs_kv[sheet_label]
    kv = wide_to_kv(df, municipio, "Municipio")
    return kv.to_dict("records")


@app.callback(Output("table_geral", "data"), Input("municipio", "value"))
def update_table_geral(municipio: str):
    return _update_kv_table("Geral", municipio)

@app.callback(Output("table_eco1", "data"), Input("municipio", "value"))
def update_table_eco1(municipio: str):
    return _update_kv_table("Economia 01", municipio)

@app.callback(Output("table_eco2", "data"), Input("municipio", "value"))
def update_table_eco2(municipio: str):
    return _update_kv_table("Economia 02", municipio)

@app.callback(Output("table_inf1", "data"), Input("municipio", "value"))
def update_table_inf1(municipio: str):
    return _update_kv_table("Infraest. 01", municipio)

@app.callback(Output("table_tur1", "data"), Input("municipio", "value"))
def update_table_tur1(municipio: str):
    return _update_kv_table("Turismo - Empreendimentos", municipio)

@app.callback(Output("table_tur2", "data"), Input("municipio", "value"))
def update_table_tur2(municipio: str):
    return _update_kv_table("Turismo - Empregos", municipio)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
