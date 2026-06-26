import glob
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Dashboard BI Accidentes USA",
    page_icon="🚗",
    layout="wide",
)

DATA_DIR = Path("data")

@st.cache_data(show_spinner=False)
def load_parquet(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        st.error(f"No se encontró el archivo requerido: {path.as_posix()}")
        st.stop()
    return pd.read_parquet(path)

@st.cache_data(show_spinner=True)
def load_data():
    return {
        "kpis": load_parquet("kpis_anio.parquet"),
        "mes": load_parquet("accidentes_mes.parquet"),
        "severidad": load_parquet("severidad_anio.parquet"),
        "hora_sev": load_parquet("hora_severidad.parquet"),
        "dia_luz": load_parquet("dia_luz.parquet"),
        "estados": load_parquet("top_estados.parquet"),
        "ciudades": load_parquet("top_ciudades.parquet"),
        "clima": load_parquet("clima.parquet"),
        "infra": load_parquet("infraestructura.parquet"),
        "dur_sev": load_parquet("duracion_severidad.parquet"),
        "dur_estado": load_parquet("duracion_estado.parquet"),
        "dur_clima": load_parquet("duracion_clima.parquet"),
    }

def filter_year(df: pd.DataFrame, year):
    if "anio" not in df.columns or year == "Todos":
        return df.copy()
    return df[df["anio"] == year].copy()

def weighted_avg(df, value_col, weight_col="total_accidentes"):
    df = df.dropna(subset=[value_col])
    w = df[weight_col].astype(float)
    if len(df) == 0 or w.sum() == 0:
        return np.nan
    return np.average(df[value_col].astype(float), weights=w)

def aggregate_by(df, group_cols, sum_cols=None, weighted_cols=None):
    sum_cols = sum_cols or []
    weighted_cols = weighted_cols or []
    if len(df) == 0:
        return pd.DataFrame(columns=group_cols + sum_cols + weighted_cols)
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for col in sum_cols:
            row[col] = g[col].sum()
        for col in weighted_cols:
            row[col] = weighted_avg(g, col)
        rows.append(row)
    return pd.DataFrame(rows)

def fmt_int(x):
    try:
        return f"{int(round(float(x))):,}".replace(",", ".")
    except Exception:
        return "0"

def fmt_float(x, nd=1):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "N/D"

# ─────────────────────────────────────────────────────────────
# Carga
# ─────────────────────────────────────────────────────────────
data = load_data()

st.title("🚗 Dashboard BI — Accidentes de Tránsito en EE.UU.")
st.caption("Dashboard optimizado para Streamlit Cloud con datos agregados del Data Mart.")

kpis = data["kpis"].copy()
years = sorted([int(y) for y in kpis["anio"].dropna().unique()])

with st.sidebar:
    st.header("Filtros")
    year = st.selectbox("Año", options=["Todos"] + years, index=0)
    st.caption("La app usa tablas agregadas para cargar rápido en la nube.")

# ─────────────────────────────────────────────────────────────
# KPIs
# ─────────────────────────────────────────────────────────────
k = filter_year(kpis, year)
if len(k) == 0:
    st.warning("No hay datos para el año seleccionado.")
    st.stop()

total = k["total_accidentes"].sum()
graves = k["accidentes_graves"].sum()
tasa_graves = graves / total * 100 if total else 0
sev_prom = weighted_avg(k, "severidad_promedio")
dur_prom = weighted_avg(k, "duracion_promedio_min")
dist_prom = weighted_avg(k, "distancia_promedio")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total accidentes", fmt_int(total))
c2.metric("Accidentes graves", fmt_int(graves))
c3.metric("Tasa de gravedad", f"{tasa_graves:.1f}%")
c4.metric("Severidad promedio", fmt_float(sev_prom, 2))
c5.metric("Duración promedio", f"{fmt_float(dur_prom, 1)} min")

# ─────────────────────────────────────────────────────────────
# Tendencia mensual y distribución severidad
# ─────────────────────────────────────────────────────────────
left, right = st.columns([1.35, 1])

with left:
    mes = filter_year(data["mes"], year)
    if year == "Todos":
        mes_plot = aggregate_by(
            mes, ["anio", "mes"],
            sum_cols=["total_accidentes", "accidentes_graves"],
            weighted_cols=["severidad_promedio", "duracion_promedio_min"],
        )
        mes_plot["periodo"] = mes_plot["anio"].astype(str) + "-" + mes_plot["mes"].astype(int).astype(str).str.zfill(2)
    else:
        mes_plot = mes.sort_values("mes").copy()
        mes_plot["periodo"] = mes_plot["mes"].astype(int).map({1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"})
    fig = px.line(
        mes_plot.sort_values(["anio", "mes"] if "anio" in mes_plot.columns else "mes"),
        x="periodo", y="total_accidentes", markers=True,
        title="Tendencia de accidentes por mes",
        labels={"periodo":"Periodo", "total_accidentes":"Accidentes"},
    )
    fig.update_layout(height=420, margin=dict(t=60, b=30))
    st.plotly_chart(fig, use_container_width=True)

with right:
    sev = filter_year(data["severidad"], year)
    sev_plot = aggregate_by(sev, ["severity"], sum_cols=["total_accidentes"], weighted_cols=["duracion_promedio_min"])
    sev_plot["severity"] = sev_plot["severity"].astype(str)
    fig = px.pie(
        sev_plot,
        names="severity", values="total_accidentes", hole=0.42,
        title="Distribución por severidad",
    )
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(height=420, margin=dict(t=60, b=20))
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# Estados y ciudades
# ─────────────────────────────────────────────────────────────
left, right = st.columns(2)

with left:
    estados = filter_year(data["estados"], year)
    estados_plot = aggregate_by(
        estados, ["state"],
        sum_cols=["total_accidentes", "accidentes_graves"],
        weighted_cols=["severidad_promedio", "duracion_promedio_min", "distancia_promedio"],
    ).nlargest(15, "total_accidentes")
    fig = px.bar(
        estados_plot.sort_values("total_accidentes"),
        x="total_accidentes", y="state", orientation="h",
        color="severidad_promedio", color_continuous_scale="Reds",
        title="Top 15 estados por accidentes",
        labels={"total_accidentes":"Accidentes", "state":"Estado", "severidad_promedio":"Sev. prom."},
    )
    fig.update_layout(height=520, margin=dict(t=60, b=30))
    st.plotly_chart(fig, use_container_width=True)

with right:
    ciudades = filter_year(data["ciudades"], year)
    ciudades_plot = aggregate_by(
        ciudades, ["city", "state"],
        sum_cols=["total_accidentes", "accidentes_graves"],
        weighted_cols=["severidad_promedio", "duracion_promedio_min"],
    )
    ciudades_plot["ciudad_estado"] = ciudades_plot["city"].astype(str) + ", " + ciudades_plot["state"].astype(str)
    ciudades_plot = ciudades_plot.nlargest(15, "accidentes_graves")
    fig = px.bar(
        ciudades_plot.sort_values("accidentes_graves"),
        x="accidentes_graves", y="ciudad_estado", orientation="h",
        color="severidad_promedio", color_continuous_scale="Oranges",
        title="Top 15 ciudades por accidentes graves",
        labels={"accidentes_graves":"Accidentes graves", "ciudad_estado":"Ciudad", "severidad_promedio":"Sev. prom."},
    )
    fig.update_layout(height=520, margin=dict(t=60, b=30))
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# Clima e infraestructura
# ─────────────────────────────────────────────────────────────
left, right = st.columns(2)

with left:
    clima = filter_year(data["clima"], year)
    clima_plot = aggregate_by(
        clima, ["weather_condition"],
        sum_cols=["total_accidentes", "accidentes_graves"],
        weighted_cols=["severidad_promedio", "duracion_promedio_min", "temperatura_promedio", "humedad_promedio", "visibilidad_promedio", "viento_promedio", "precipitacion_promedio"],
    ).nlargest(15, "total_accidentes")
    fig = px.bar(
        clima_plot.sort_values("total_accidentes"),
        x="total_accidentes", y="weather_condition", orientation="h",
        color="accidentes_graves", color_continuous_scale="Blues",
        title="Top 15 condiciones climáticas por accidentes",
        labels={"total_accidentes":"Accidentes", "weather_condition":"Clima", "accidentes_graves":"Graves"},
    )
    fig.update_layout(height=520, margin=dict(t=60, b=30))
    st.plotly_chart(fig, use_container_width=True)

with right:
    infra = filter_year(data["infra"], year)
    infra_plot = aggregate_by(infra, ["infraestructura"], sum_cols=["total_accidentes", "accidentes_graves"])
    infra_plot["tasa_graves_pct"] = np.where(
        infra_plot["total_accidentes"] > 0,
        infra_plot["accidentes_graves"] / infra_plot["total_accidentes"] * 100,
        0,
    )
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Total", x=infra_plot["infraestructura"], y=infra_plot["total_accidentes"]))
    fig.add_trace(go.Bar(name="Graves", x=infra_plot["infraestructura"], y=infra_plot["accidentes_graves"]))
    fig.update_layout(
        title="Accidentes por infraestructura vial",
        barmode="group", height=520,
        xaxis_title="Infraestructura", yaxis_title="Accidentes",
        margin=dict(t=60, b=90), legend=dict(orientation="h", y=1.08),
    )
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# Hora x severidad y día/luz
# ─────────────────────────────────────────────────────────────
left, right = st.columns(2)

with left:
    hs = filter_year(data["hora_sev"], year)
    hs_plot = aggregate_by(hs, ["hora", "severity"], sum_cols=["total_accidentes"])
    pivot = hs_plot.pivot_table(index="severity", columns="hora", values="total_accidentes", aggfunc="sum", fill_value=0)
    fig = px.imshow(
        pivot,
        aspect="auto",
        title="Mapa de calor: hora del día vs severidad",
        labels=dict(x="Hora", y="Severidad", color="Accidentes"),
    )
    fig.update_layout(height=460, margin=dict(t=60, b=30))
    st.plotly_chart(fig, use_container_width=True)

with right:
    dia = filter_year(data["dia_luz"], year)
    dia_plot = aggregate_by(dia, ["dia_semana", "civil_twilight"], sum_cols=["total_accidentes", "accidentes_graves"])
    dias_map = {"Monday":"Lun", "Tuesday":"Mar", "Wednesday":"Mié", "Thursday":"Jue", "Friday":"Vie", "Saturday":"Sáb", "Sunday":"Dom"}
    orden = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
    dia_plot["dia"] = dia_plot["dia_semana"].map(dias_map).fillna(dia_plot["dia_semana"])
    dia_plot["dia"] = pd.Categorical(dia_plot["dia"], categories=orden, ordered=True)
    dia_plot = dia_plot.sort_values("dia")
    fig = px.bar(
        dia_plot,
        x="dia", y="total_accidentes", color="civil_twilight",
        title="Accidentes por día de semana y condición de luz",
        labels={"dia":"Día", "total_accidentes":"Accidentes", "civil_twilight":"Luz"},
    )
    fig.update_layout(height=460, margin=dict(t=60, b=30), legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# Duraciones y ambiente
# ─────────────────────────────────────────────────────────────
left, mid, right = st.columns(3)

with left:
    ds = filter_year(data["dur_sev"], year)
    ds_plot = aggregate_by(ds, ["severity"], sum_cols=["total_accidentes"], weighted_cols=["duracion_promedio_min"])
    fig = px.bar(
        ds_plot,
        x="severity", y="duracion_promedio_min",
        title="Duración promedio por severidad",
        labels={"severity":"Severidad", "duracion_promedio_min":"Minutos"},
        text=ds_plot["duracion_promedio_min"].round(0),
    )
    fig.update_layout(height=430, margin=dict(t=60, b=30))
    st.plotly_chart(fig, use_container_width=True)

with mid:
    de = filter_year(data["dur_estado"], year)
    de_plot = aggregate_by(de, ["state"], sum_cols=["total_accidentes"], weighted_cols=["duracion_promedio_min"]).nlargest(12, "duracion_promedio_min")
    fig = px.bar(
        de_plot.sort_values("duracion_promedio_min"),
        x="duracion_promedio_min", y="state", orientation="h",
        title="Estados con mayor duración promedio",
        labels={"duracion_promedio_min":"Minutos", "state":"Estado"},
    )
    fig.update_layout(height=430, margin=dict(t=60, b=30))
    st.plotly_chart(fig, use_container_width=True)

with right:
    dc = filter_year(data["dur_clima"], year)
    dc_plot = aggregate_by(dc, ["weather_condition"], sum_cols=["total_accidentes"], weighted_cols=["duracion_promedio_min"]).nlargest(12, "duracion_promedio_min")
    fig = px.bar(
        dc_plot.sort_values("duracion_promedio_min"),
        x="duracion_promedio_min", y="weather_condition", orientation="h",
        title="Climas con mayor duración promedio",
        labels={"duracion_promedio_min":"Minutos", "weather_condition":"Clima"},
    )
    fig.update_layout(height=430, margin=dict(t=60, b=30))
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Resumen ejecutivo")
col1, col2, col3 = st.columns(3)
with col1:
    top_state = estados_plot.sort_values("total_accidentes", ascending=False).head(1)
    if len(top_state):
        st.info(f"El estado con mayor volumen es **{top_state.iloc[0]['state']}** con **{fmt_int(top_state.iloc[0]['total_accidentes'])}** accidentes.")
with col2:
    top_clima = clima_plot.sort_values("total_accidentes", ascending=False).head(1)
    if len(top_clima):
        st.info(f"La condición climática más frecuente es **{top_clima.iloc[0]['weather_condition']}**.")
with col3:
    st.info(f"La tasa de accidentes graves en el filtro actual es **{tasa_graves:.1f}%**.")

with st.expander("Ver tablas agregadas cargadas"):
    for name, df in data.items():
        st.write(f"**{name}** — {len(df):,} filas")
