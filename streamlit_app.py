import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="BI Accidentes USA",
    page_icon="🚗",
    layout="wide",
)

DATA_PATH = Path("data/datamart_accidentes.parquet")

COLOR_SEV = {1: "#2ecc71", 2: "#f39c12", 3: "#e67e22", 4: "#e74c3c"}
DIAS_MAP_EN = {
    "Monday": "Lun",
    "Tuesday": "Mar",
    "Wednesday": "Mié",
    "Thursday": "Jue",
    "Friday": "Vie",
    "Saturday": "Sáb",
    "Sunday": "Dom",
}
DIAS_ORDER = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]


@st.cache_data(show_spinner="Cargando datos...")
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error(
            "No se encontró `data/datamart_accidentes.parquet`. "
            "Primero ejecuta el cuadernillo `exportar_datamart.ipynb` y sube la carpeta `data/` al repositorio."
        )
        st.stop()

    df = pd.read_parquet(DATA_PATH)
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df["severity"] = pd.to_numeric(df["severity"], errors="coerce").astype("Int64")

    for col in ["crossing", "junction", "station", "stop", "traffic_signal"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    return df


def empty_fig(title: str):
    fig = go.Figure()
    fig.update_layout(title=title, xaxis={"visible": False}, yaxis={"visible": False})
    fig.add_annotation(text="Sin datos para los filtros seleccionados", x=0.5, y=0.5, showarrow=False)
    return fig


def filter_df(df: pd.DataFrame, state, severity, clima, twilight, anio) -> pd.DataFrame:
    out = df.copy()
    if state != "Todos":
        out = out[out["state"] == state]
    if severity != "Todas":
        out = out[out["severity"] == int(severity)]
    if clima != "Todas":
        out = out[out["weather_condition"] == clima]
    if twilight != "Día y Noche":
        value = "Day" if twilight == "Día" else "Night"
        out = out[out["civil_twilight"] == value]
    if anio != "Todos":
        out = out[out["anio"] == int(anio)]
    return out


def build_figures(df: pd.DataFrame):
    if df.empty:
        return [empty_fig("Sin datos")] * 11

    # Heatmap hora x severidad
    hora_sev = df.groupby(["hora", "severity"]).size().reset_index(name="count")
    pivot = hora_sev.pivot(index="severity", columns="hora", values="count").fillna(0)
    if not pivot.empty:
        pivot = pivot.reindex(sorted(pivot.index))
        fig_heatmap = go.Figure(go.Heatmap(
            z=pivot.values,
            x=[f"{int(h):02d}:00" for h in pivot.columns],
            y=[f"Sev {int(s)}" for s in pivot.index],
            colorscale="YlOrRd",
            colorbar=dict(title="N° accidentes"),
        ))
        fig_heatmap.update_layout(
            title="Accidentes por Hora del Día y Severidad<br><sup>¿En qué horarios ocurren más accidentes graves?</sup>",
            xaxis_title="Hora del día",
            yaxis_title="Severidad",
            margin=dict(t=70, b=50),
        )
    else:
        fig_heatmap = empty_fig("Accidentes por Hora del Día y Severidad")

    # Día semana x luz
    dia_luz = df.groupby(["dia_semana", "civil_twilight"]).size().reset_index(name="count")
    dia_luz["dia_nombre"] = dia_luz["dia_semana"].map(DIAS_MAP_EN).fillna(dia_luz["dia_semana"])
    fig_dia_luz = px.bar(
        dia_luz,
        x="dia_nombre",
        y="count",
        color="civil_twilight",
        color_discrete_map={"Day": "#f39c12", "Night": "#2c3e50"},
        category_orders={"dia_nombre": DIAS_ORDER},
        labels={"count": "N° Accidentes", "dia_nombre": "Día", "civil_twilight": "Luz"},
        title="Accidentes por Día y Condición de Luz<br><sup>Día vs Noche por día de la semana</sup>",
        barmode="stack",
    )
    fig_dia_luz.update_layout(margin=dict(t=70, b=40), legend=dict(orientation="h", y=1.08))

    # Top estados
    states_df = (
        df.groupby("state")
        .agg(total=("id", "count"), graves=("severity", lambda x: (x >= 3).sum()), sev_prom=("severity", "mean"))
        .reset_index()
        .nlargest(15, "total")
    )
    fig_states = px.bar(
        states_df,
        x="total",
        y="state",
        orientation="h",
        color="sev_prom",
        color_continuous_scale="Reds",
        text="graves",
        labels={"total": "Total accidentes", "state": "Estado", "sev_prom": "Sev. prom.", "graves": "Graves (3-4)"},
        title="Top 15 Estados — Total y Severidad<br><sup>¿Qué estados concentran más accidentes graves?</sup>",
    )
    fig_states.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig_states.update_layout(yaxis={"categoryorder": "total ascending"}, margin=dict(t=70, b=30))

    # Top ciudades
    cities_df = (
        df.groupby("city")
        .agg(total=("id", "count"), graves=("severity", lambda x: (x >= 3).sum()), sev_prom=("severity", "mean"))
        .reset_index()
        .nlargest(15, "graves")
    )
    fig_cities = px.bar(
        cities_df,
        x="graves",
        y="city",
        orientation="h",
        color="sev_prom",
        color_continuous_scale="Oranges",
        labels={"graves": "Accidentes Sev. 3-4", "city": "Ciudad", "sev_prom": "Sev. prom."},
        title="Top 15 Ciudades — Accidentes de Alta Severidad<br><sup>¿Qué ciudades concentran más accidentes graves?</sup>",
    )
    fig_cities.update_layout(yaxis={"categoryorder": "total ascending"}, margin=dict(t=70, b=30))

    # Clima count
    clima_count = df.groupby("weather_condition").agg(total=("id", "count")).reset_index().nlargest(15, "total")
    fig_clima_count = px.bar(
        clima_count,
        x="total",
        y="weather_condition",
        orientation="h",
        color="total",
        color_continuous_scale="Blues",
        labels={"total": "N° Accidentes", "weather_condition": "Condición climática"},
        title="Distribución de Accidentes por Clima (Top 15)<br><sup>¿Qué clima concentra más accidentes?</sup>",
    )
    fig_clima_count.update_layout(yaxis={"categoryorder": "total ascending"}, margin=dict(t=70, b=30), showlegend=False)

    # Clima graves
    clima_sev = (
        df[df["severity"] >= 3]
        .groupby("weather_condition")
        .agg(graves=("id", "count"), sev_prom=("severity", "mean"))
        .reset_index()
    )
    clima_sev = clima_sev[clima_sev["graves"] >= 5].nlargest(15, "graves")
    fig_clima_sev = px.bar(
        clima_sev,
        x="graves",
        y="weather_condition",
        orientation="h",
        color="sev_prom",
        color_continuous_scale="Reds",
        labels={"graves": "Accidentes Sev. 3-4", "weather_condition": "Condición climática", "sev_prom": "Sev. prom."},
        title="Clima vs Accidentes Graves (Sev. 3-4) — Top 15<br><sup>¿Qué clima se asocia más a accidentes graves?</sup>",
    )
    fig_clima_sev.update_layout(yaxis={"categoryorder": "total ascending"}, margin=dict(t=70, b=30))

    # Infraestructura
    infra_cols = {
        "Cruce peatonal": "crossing",
        "Intersección": "junction",
        "Estación": "station",
        "Señal Stop": "stop",
        "Semáforo": "traffic_signal",
    }
    rows = []
    for nombre, col in infra_cols.items():
        sub = df[df[col] == 1]
        total_col = len(sub)
        graves_col = int((sub["severity"] >= 3).sum())
        tasa_col = round(graves_col / total_col * 100, 1) if total_col > 0 else 0.0
        rows.append({"infraestructura": nombre, "total": total_col, "graves": graves_col, "tasa_graves_%": tasa_col})
    infra_df = pd.DataFrame(rows)

    fig_infra = go.Figure()
    fig_infra.add_trace(go.Bar(
        name="Total accidentes",
        x=infra_df["infraestructura"],
        y=infra_df["total"],
        marker_color="#3498db",
        opacity=0.7,
        text=infra_df["total"].apply(lambda v: f"{v:,}"),
        textposition="outside",
    ))
    fig_infra.add_trace(go.Bar(
        name="Sev. alta (3-4)",
        x=infra_df["infraestructura"],
        y=infra_df["graves"],
        marker_color="#e74c3c",
        opacity=0.9,
        text=infra_df["tasa_graves_%"].apply(lambda v: f"{v}%"),
        textposition="outside",
    ))
    fig_infra.update_layout(
        title="Accidentes por Infraestructura Vial — Total vs Severidad Alta<br><sup>Etiquetas rojas = % de accidentes graves sobre el total en esa infraestructura</sup>",
        barmode="group",
        xaxis_title="Tipo de infraestructura",
        yaxis_title="N° Accidentes",
        legend=dict(orientation="h", y=1.08),
        margin=dict(t=80, b=40),
    )

    # Duración severidad
    dur_sev = df.groupby("severity").agg(dur_prom=("duracion_minutos", "mean")).reset_index().sort_values("severity")
    dur_sev["sev_label"] = dur_sev["severity"].map({1: "Leve (1)", 2: "Moderado (2)", 3: "Grave (3)", 4: "Muy Grave (4)"})
    dur_sev["severity_int"] = dur_sev["severity"].astype(int)
    fig_dur_sev = px.bar(
        dur_sev,
        x="sev_label",
        y="dur_prom",
        color="severity_int",
        color_discrete_map=COLOR_SEV,
        text=dur_sev["dur_prom"].round(0).fillna(0).astype(int),
        labels={"dur_prom": "Duración prom. (min)", "sev_label": "Severidad", "severity_int": "Severidad"},
        title="Duración Prom. por Severidad<br><sup>¿Cuánto dura según la gravedad?</sup>",
    )
    fig_dur_sev.update_traces(textposition="outside")
    fig_dur_sev.update_layout(showlegend=False, margin=dict(t=70, b=40))

    # Duración clima
    dur_clima = df.groupby("weather_condition").agg(dur_prom=("duracion_minutos", "mean"), total=("id", "count")).reset_index()
    dur_clima = dur_clima[dur_clima["total"] >= 10].nlargest(12, "dur_prom")
    fig_dur_clima = px.bar(
        dur_clima,
        x="dur_prom",
        y="weather_condition",
        orientation="h",
        color="dur_prom",
        color_continuous_scale="Blues",
        text=dur_clima["dur_prom"].round(0).fillna(0).astype(int),
        labels={"dur_prom": "Duración prom. (min)", "weather_condition": "Condición climática"},
        title="Duración Prom. por Clima (Top 12)<br><sup>¿Qué clima prolonga más los accidentes?</sup>",
    )
    fig_dur_clima.update_traces(textposition="outside")
    fig_dur_clima.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False, margin=dict(t=70, b=30))

    # Duración estado
    dur_estado = df.groupby("state").agg(dur_prom=("duracion_minutos", "mean"), total=("id", "count")).reset_index()
    dur_estado = dur_estado[dur_estado["total"] >= 10].nlargest(12, "dur_prom")
    fig_dur_estado = px.bar(
        dur_estado,
        x="dur_prom",
        y="state",
        orientation="h",
        color="dur_prom",
        color_continuous_scale="Purples",
        text=dur_estado["dur_prom"].round(0).fillna(0).astype(int),
        labels={"dur_prom": "Duración prom. (min)", "state": "Estado"},
        title="Duración Prom. por Estado (Top 12)<br><sup>¿En qué zonas duran más los accidentes?</sup>",
    )
    fig_dur_estado.update_traces(textposition="outside")
    fig_dur_estado.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False, margin=dict(t=70, b=30))

    # Pie severidad
    sev_pie = df.groupby("severity").size().reset_index(name="count")
    sev_pie["label"] = sev_pie["severity"].map({1: "Leve (1)", 2: "Moderado (2)", 3: "Grave (3)", 4: "Muy Grave (4)"})
    fig_pie = px.pie(
        sev_pie,
        values="count",
        names="label",
        color="severity",
        color_discrete_map=COLOR_SEV,
        title="Distribución General por Nivel de Severidad",
        hole=0.35,
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    fig_pie.update_layout(showlegend=True, margin=dict(t=60, b=20))

    return [
        fig_heatmap,
        fig_dia_luz,
        fig_states,
        fig_cities,
        fig_clima_count,
        fig_clima_sev,
        fig_infra,
        fig_dur_sev,
        fig_dur_clima,
        fig_dur_estado,
        fig_pie,
    ]


st.title("Dashboard — Accidentes de Tránsito en EE.UU.")
st.caption("Análisis multidimensional · Datamart BI · US-Accidents (2016–2023)")
st.caption("Integrantes: Diego Saldaña · Diego Rodríguez · Sebastián Vinces · Mario Auqui")

df_main = load_data()

with st.sidebar:
    st.header("Filtros")
    estados = ["Todos"] + sorted(df_main["state"].dropna().astype(str).unique().tolist())
    severidades = ["Todas"] + sorted(df_main["severity"].dropna().astype(int).unique().tolist())
    climas = ["Todas"] + sorted(df_main["weather_condition"].dropna().astype(str).unique().tolist())
    anios = ["Todos"] + sorted(df_main["anio"].dropna().astype(int).unique().tolist())

    state = st.selectbox("Estado", estados)
    severity = st.selectbox("Severidad", severidades, format_func=lambda x: x if x == "Todas" else f"Severidad {x}")
    clima = st.selectbox("Condición climática", climas)
    twilight = st.selectbox("Condición de luz", ["Día y Noche", "Día", "Noche"])
    anio = st.selectbox("Año", anios)

filtered = filter_df(df_main, state, severity, clima, twilight, anio)

total = len(filtered)
sev_alta = int((filtered["severity"] >= 3).sum()) if total > 0 else 0
tasa_alta = round(sev_alta / total * 100, 1) if total > 0 else 0.0
dur_prom = round(filtered["duracion_minutos"].mean(), 1) if total > 0 else 0.0
dist_prom = round(filtered["distance"].mean(), 2) if total > 0 else 0.0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Tasa Sev. Alta (3-4)", f"{tasa_alta} %")
k2.metric("Duración Promedio", f"{dur_prom} min")
k3.metric("Total Accidentes", f"{total:,}")
k4.metric("Accidentes Sev. 3-4", f"{sev_alta:,}")
k5.metric("Distancia Prom.", f"{dist_prom} mi")

figs = build_figures(filtered)

c1, c2 = st.columns([1.45, 1])
with c1:
    st.plotly_chart(figs[0], use_container_width=True)
with c2:
    st.plotly_chart(figs[1], use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    st.plotly_chart(figs[2], use_container_width=True)
with c4:
    st.plotly_chart(figs[3], use_container_width=True)

c5, c6 = st.columns(2)
with c5:
    st.plotly_chart(figs[4], use_container_width=True)
with c6:
    st.plotly_chart(figs[5], use_container_width=True)

st.plotly_chart(figs[6], use_container_width=True)

c7, c8, c9 = st.columns(3)
with c7:
    st.plotly_chart(figs[7], use_container_width=True)
with c8:
    st.plotly_chart(figs[8], use_container_width=True)
with c9:
    st.plotly_chart(figs[9], use_container_width=True)

st.plotly_chart(figs[10], use_container_width=True)

st.caption("Datos exportados desde proyecto_destino (SQL Server) a Parquet para despliegue en Streamlit Cloud.")
