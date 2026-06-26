from pathlib import Path
import glob

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Dashboard Accidentes USA",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path("data")
DATA_PATTERN = "datamart_accidentes_*.parquet"

# -----------------------------------------------------------------------------
# ESTILOS
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
        .kpi-card {
            background: #ffffff;
            border: 1px solid #e6e9ef;
            border-radius: 14px;
            padding: 18px 14px;
            box-shadow: 0 2px 9px rgba(0,0,0,0.06);
            text-align: center;
            min-height: 105px;
        }
        .kpi-value {font-size: 1.85rem; font-weight: 800; margin: 0; color: #1f2937;}
        .kpi-label {font-size: 0.86rem; color: #6b7280; margin: 0.2rem 0 0 0;}
        .small-note {font-size: 0.82rem; color: #6b7280;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# FUNCIONES DE CARGA Y LIMPIEZA
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Cargando archivos parquet...")
def load_data() -> pd.DataFrame:
    files = sorted(DATA_DIR.glob(DATA_PATTERN))

    if not files:
        st.error(
            "No encontré archivos parquet en `data/datamart_accidentes_*.parquet`. "
            "Verifica que en GitHub exista la carpeta `data` y que dentro estén los archivos generados por el cuadernillo, "
            "por ejemplo `datamart_accidentes_001.parquet`."
        )
        st.stop()

    frames = []
    for file in files:
        try:
            frames.append(pd.read_parquet(file))
        except Exception as e:
            st.error(f"No pude leer `{file}`: {e}")
            st.stop()

    df = pd.concat(frames, ignore_index=True)
    df.columns = [str(c).strip() for c in df.columns]

    # Normalización de fechas y columnas numéricas esperadas
    if "start_time" in df.columns:
        df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")

    if "anio" not in df.columns and "start_time" in df.columns:
        df["anio"] = df["start_time"].dt.year

    if "mes" not in df.columns and "start_time" in df.columns:
        df["mes"] = df["start_time"].dt.month

    if "hora" not in df.columns and "start_time" in df.columns:
        df["hora"] = df["start_time"].dt.hour

    numeric_cols = [
        "severity", "duracion_minutos", "distance", "temperature", "humidity",
        "visibility", "wind_speed", "precipitation", "anio", "mes", "hora"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    bool_cols = ["crossing", "junction", "station", "stop", "traffic_signal", "es_fin_semana"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .map({True: 1, False: 0, "True": 1, "False": 0, "true": 1, "false": 0, "1": 1, "0": 0})
                .fillna(pd.to_numeric(df[col], errors="coerce"))
                .fillna(0)
                .astype(int)
            )

    return df


def metric_card(label: str, value: str):
    st.markdown(
        f"""
        <div class="kpi-card">
            <p class="kpi-value">{value}</p>
            <p class="kpi-label">{label}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def safe_top(df: pd.DataFrame, col: str, n: int = 10):
    if col not in df.columns or df.empty:
        return pd.DataFrame(columns=[col, "total"])
    return df[col].fillna("Sin dato").value_counts().head(n).reset_index().rename(columns={"index": col, col: "total"})


def empty_fig(title: str):
    fig = go.Figure()
    fig.update_layout(title=title, xaxis={"visible": False}, yaxis={"visible": False})
    fig.add_annotation(text="Sin datos para los filtros seleccionados", x=0.5, y=0.5, showarrow=False)
    return fig

# -----------------------------------------------------------------------------
# APP
# -----------------------------------------------------------------------------
df = load_data()

st.title("🚗 Dashboard de Accidentes de Tránsito en USA")
st.caption("Dashboard generado para despliegue en Streamlit Cloud usando archivos Parquet en la carpeta `data/`.")

with st.sidebar:
    st.header("Filtros")

    # Filtro por año solicitado
    years = []
    if "anio" in df.columns:
        years = sorted([int(y) for y in df["anio"].dropna().unique()])
    selected_years = st.multiselect(
        "Año",
        options=years,
        default=years,
    )

    states = sorted(df["state"].dropna().astype(str).unique()) if "state" in df.columns else []
    selected_states = st.multiselect("Estado", options=states, default=[])

    severities = sorted([int(x) for x in df["severity"].dropna().unique()]) if "severity" in df.columns else []
    selected_sev = st.multiselect("Severidad", options=severities, default=[])

    weather = sorted(df["weather_condition"].dropna().astype(str).unique()) if "weather_condition" in df.columns else []
    selected_weather = st.multiselect("Condición climática", options=weather, default=[])

    twilight = sorted(df["civil_twilight"].dropna().astype(str).unique()) if "civil_twilight" in df.columns else []
    selected_twilight = st.multiselect("Luz / Twilight", options=twilight, default=[])

    st.divider()
    st.caption(f"Archivos leídos: {len(list(DATA_DIR.glob(DATA_PATTERN)))}")

filtered = df.copy()
if selected_years and "anio" in filtered.columns:
    filtered = filtered[filtered["anio"].isin(selected_years)]
if selected_states and "state" in filtered.columns:
    filtered = filtered[filtered["state"].astype(str).isin(selected_states)]
if selected_sev and "severity" in filtered.columns:
    filtered = filtered[filtered["severity"].isin(selected_sev)]
if selected_weather and "weather_condition" in filtered.columns:
    filtered = filtered[filtered["weather_condition"].astype(str).isin(selected_weather)]
if selected_twilight and "civil_twilight" in filtered.columns:
    filtered = filtered[filtered["civil_twilight"].astype(str).isin(selected_twilight)]

if filtered.empty:
    st.warning("No hay datos para los filtros seleccionados. Ajusta los filtros del panel lateral.")
    st.stop()

# -----------------------------------------------------------------------------
# KPIs RELEVANTES
# -----------------------------------------------------------------------------
total_acc = len(filtered)
graves = int((filtered["severity"] >= 3).sum()) if "severity" in filtered.columns else 0
pct_graves = (graves / total_acc * 100) if total_acc else 0
sev_prom = filtered["severity"].mean() if "severity" in filtered.columns else np.nan
dur_prom = filtered["duracion_minutos"].mean() if "duracion_minutos" in filtered.columns else np.nan
dist_prom = filtered["distance"].mean() if "distance" in filtered.columns else np.nan

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    metric_card("Total accidentes", f"{total_acc:,.0f}")
with c2:
    metric_card("Accidentes graves (sev. 3-4)", f"{graves:,.0f}")
with c3:
    metric_card("% graves", f"{pct_graves:.1f}%")
with c4:
    metric_card("Severidad promedio", "N/D" if pd.isna(sev_prom) else f"{sev_prom:.2f}")
with c5:
    metric_card("Duración promedio", "N/D" if pd.isna(dur_prom) else f"{dur_prom:.1f} min")

st.markdown("---")

# -----------------------------------------------------------------------------
# GRÁFICOS
# -----------------------------------------------------------------------------
left, right = st.columns(2)

with left:
    if "anio" in filtered.columns:
        by_year = filtered.groupby("anio", dropna=True).size().reset_index(name="accidentes")
        fig = px.line(
            by_year,
            x="anio",
            y="accidentes",
            markers=True,
            title="Evolución anual de accidentes",
            labels={"anio": "Año", "accidentes": "N° accidentes"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.plotly_chart(empty_fig("Evolución anual de accidentes"), use_container_width=True)

with right:
    if "severity" in filtered.columns:
        sev = filtered["severity"].value_counts().sort_index().reset_index()
        sev.columns = ["severity", "accidentes"]
        fig = px.bar(
            sev,
            x="severity",
            y="accidentes",
            title="Accidentes por severidad",
            labels={"severity": "Severidad", "accidentes": "N° accidentes"},
            text="accidentes",
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.plotly_chart(empty_fig("Accidentes por severidad"), use_container_width=True)

left, right = st.columns(2)

with left:
    if "state" in filtered.columns:
        top_states = filtered.groupby("state").size().reset_index(name="accidentes").nlargest(15, "accidentes")
        fig = px.bar(
            top_states.sort_values("accidentes"),
            x="accidentes",
            y="state",
            orientation="h",
            title="Top 15 estados con más accidentes",
            labels={"state": "Estado", "accidentes": "N° accidentes"},
            text="accidentes",
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.plotly_chart(empty_fig("Top estados"), use_container_width=True)

with right:
    if "city" in filtered.columns:
        top_cities = filtered.groupby("city").size().reset_index(name="accidentes").nlargest(15, "accidentes")
        fig = px.bar(
            top_cities.sort_values("accidentes"),
            x="accidentes",
            y="city",
            orientation="h",
            title="Top 15 ciudades con más accidentes",
            labels={"city": "Ciudad", "accidentes": "N° accidentes"},
            text="accidentes",
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.plotly_chart(empty_fig("Top ciudades"), use_container_width=True)

left, right = st.columns(2)

with left:
    if "hora" in filtered.columns:
        by_hour = filtered.groupby("hora").size().reset_index(name="accidentes").sort_values("hora")
        fig = px.area(
            by_hour,
            x="hora",
            y="accidentes",
            title="Accidentes por hora del día",
            labels={"hora": "Hora", "accidentes": "N° accidentes"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.plotly_chart(empty_fig("Accidentes por hora"), use_container_width=True)

with right:
    if "weather_condition" in filtered.columns:
        weather_df = (
            filtered["weather_condition"]
            .fillna("Sin dato")
            .value_counts()
            .head(12)
            .reset_index()
        )
        weather_df.columns = ["weather_condition", "accidentes"]
        fig = px.bar(
            weather_df.sort_values("accidentes"),
            x="accidentes",
            y="weather_condition",
            orientation="h",
            title="Condiciones climáticas más frecuentes",
            labels={"weather_condition": "Clima", "accidentes": "N° accidentes"},
            text="accidentes",
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.plotly_chart(empty_fig("Condiciones climáticas"), use_container_width=True)

left, right = st.columns(2)

with left:
    if "mes" in filtered.columns:
        by_month = filtered.groupby("mes").size().reset_index(name="accidentes").sort_values("mes")
        fig = px.bar(
            by_month,
            x="mes",
            y="accidentes",
            title="Accidentes por mes",
            labels={"mes": "Mes", "accidentes": "N° accidentes"},
            text="accidentes",
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.plotly_chart(empty_fig("Accidentes por mes"), use_container_width=True)

with right:
    bool_available = [c for c in ["crossing", "junction", "station", "stop", "traffic_signal"] if c in filtered.columns]
    if bool_available:
        infra = pd.DataFrame({
            "factor": bool_available,
            "accidentes": [int(filtered[c].sum()) for c in bool_available],
        }).sort_values("accidentes", ascending=False)
        name_map = {
            "crossing": "Cruce",
            "junction": "Intersección",
            "station": "Estación",
            "stop": "Stop",
            "traffic_signal": "Semáforo",
        }
        infra["factor"] = infra["factor"].map(name_map).fillna(infra["factor"])
        fig = px.bar(
            infra,
            x="factor",
            y="accidentes",
            title="Accidentes asociados a infraestructura vial",
            labels={"factor": "Factor", "accidentes": "N° accidentes"},
            text="accidentes",
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.plotly_chart(empty_fig("Infraestructura vial"), use_container_width=True)

# Tabla resumen
st.markdown("---")
st.subheader("Resumen por estado")
if "state" in filtered.columns:
    agg_dict = {"accidentes": ("state", "count")}
    if "severity" in filtered.columns:
        agg_dict.update({
            "severidad_promedio": ("severity", "mean"),
            "accidentes_graves": ("severity", lambda s: int((s >= 3).sum())),
        })
    if "duracion_minutos" in filtered.columns:
        agg_dict["duracion_promedio_min"] = ("duracion_minutos", "mean")
    if "distance" in filtered.columns:
        agg_dict["distancia_promedio"] = ("distance", "mean")

    resumen = filtered.groupby("state").agg(**agg_dict).reset_index().sort_values("accidentes", ascending=False)
    st.dataframe(resumen, use_container_width=True, hide_index=True)
else:
    st.info("No existe la columna `state` en los parquet cargados.")

with st.expander("Ver columnas disponibles"):
    st.write(list(df.columns))
