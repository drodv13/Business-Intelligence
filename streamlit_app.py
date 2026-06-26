from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="BI Accidentes USA", page_icon="🚗", layout="wide")

DATA_DIR = Path("data")

# ------------------------------
# Utilidades
# ------------------------------
@st.cache_data(show_spinner=False)
def load_parquet(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)

@st.cache_data(show_spinner=True)
def load_all_data():
    files = {
        "kpis": "kpis_anio.parquet",
        "mes": "accidentes_mes.parquet",
        "severidad": "severidad_anio.parquet",
        "hora": "hora.parquet",
        "dia_semana": "dia_semana.parquet",
        "hora_dia": "hora_dia_semana.parquet",
        "estados": "top_estados.parquet",
        "ciudades": "top_ciudades.parquet",
        "clima": "clima.parquet",
        "infra": "infraestructura.parquet",
        "luz": "luz.parquet",
        "fin_semana": "fin_semana.parquet",
        "temperatura": "temperatura_bins.parquet",
        "humedad": "humedad_bins.parquet",
        "visibilidad": "visibilidad_bins.parquet",
        "precipitacion": "precipitacion_bins.parquet",
        "viento": "viento_bins.parquet",
        "trimestre": "trimestre.parquet",
        "timezone": "timezone.parquet",
        "mapa": "mapa_ciudades.parquet",
        "metricas_clima": "metricas_clima_anio.parquet",
        "estado_severidad": "estado_severidad.parquet",
    }
    return {key: load_parquet(file) for key, file in files.items()}

def filter_year(df: pd.DataFrame, years):
    if df.empty or "anio" not in df.columns:
        return df
    return df[df["anio"].isin(years)].copy()

def fmt_int(x):
    try:
        return f"{int(round(float(x))):,}".replace(",", ".")
    except Exception:
        return "0"

def fmt_pct(x):
    try:
        return f"{float(x):.1f}%"
    except Exception:
        return "0,0%"

def top_n(df, col_value="total_accidentes", n=15):
    if df.empty:
        return df
    return df.sort_values(col_value, ascending=False).head(n)

def add_rate(df):
    df = df.copy()
    if {"accidentes_graves", "total_accidentes"}.issubset(df.columns):
        df["tasa_graves_pct"] = np.where(df["total_accidentes"] > 0, df["accidentes_graves"] / df["total_accidentes"] * 100, 0)
    return df

# ------------------------------
# Carga
# ------------------------------
data = load_all_data()

if not DATA_DIR.exists() or all(df.empty for df in data.values()):
    st.error("No encontré archivos .parquet en la carpeta data/. Ejecuta primero exportar_datamart_agregado_v2.ipynb y sube la carpeta data al repositorio.")
    st.stop()

kpis_all = data["kpis"]
if kpis_all.empty or "anio" not in kpis_all.columns:
    st.error("No encontré data/kpis_anio.parquet o no contiene la columna anio.")
    st.stop()

years_available = sorted(pd.to_numeric(kpis_all["anio"], errors="coerce").dropna().astype(int).unique().tolist())

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.title("🎛️ Filtros")
selected_years = st.sidebar.multiselect(
    "Año",
    options=years_available,
    default=years_available,
)
if not selected_years:
    st.warning("Selecciona al menos un año.")
    st.stop()

metric_option = st.sidebar.radio(
    "Métrica principal",
    ["Total accidentes", "Accidentes graves", "Tasa de gravedad"],
    index=0,
)
metric_col = {
    "Total accidentes": "total_accidentes",
    "Accidentes graves": "accidentes_graves",
    "Tasa de gravedad": "tasa_graves_pct",
}[metric_option]

st.sidebar.caption("Los datos están agregados para que el dashboard cargue rápido en Streamlit Cloud.")

# ------------------------------
# Título
# ------------------------------
st.title("🚗 Dashboard BI — Accidentes de Tránsito en EE. UU.")
st.caption("Análisis agregado del Data Mart: temporalidad, severidad, geografía, clima, iluminación e infraestructura vial.")

# ------------------------------
# KPIs
# ------------------------------
kpis = filter_year(kpis_all, selected_years)
if kpis.empty:
    st.warning("No hay datos para el año seleccionado.")
    st.stop()

total_acc = kpis["total_accidentes"].sum()
graves = kpis["accidentes_graves"].sum() if "accidentes_graves" in kpis else 0
tasa_graves = graves / total_acc * 100 if total_acc else 0
sev_prom = np.average(kpis["severidad_promedio"], weights=kpis["total_accidentes"]) if "severidad_promedio" in kpis else 0
dur_prom = np.average(kpis["duracion_promedio_min"], weights=kpis["total_accidentes"]) if "duracion_promedio_min" in kpis else 0
dist_prom = np.average(kpis["distancia_promedio"], weights=kpis["total_accidentes"]) if "distancia_promedio" in kpis else 0
anio_peak = kpis.loc[kpis["total_accidentes"].idxmax(), "anio"] if not kpis.empty else "-"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total accidentes", fmt_int(total_acc))
c2.metric("Accidentes graves", fmt_int(graves))
c3.metric("Tasa de gravedad", fmt_pct(tasa_graves))
c4.metric("Año con más accidentes", str(int(anio_peak)) if pd.notna(anio_peak) else "-")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Severidad promedio", f"{sev_prom:.2f}")
c6.metric("Duración promedio", f"{dur_prom:.1f} min")
c7.metric("Distancia promedio", f"{dist_prom:.2f} mi")
c8.metric("Años analizados", f"{len(selected_years)}")

# ------------------------------
# Tabs
# ------------------------------
tab_resumen, tab_temporal, tab_geo, tab_clima, tab_infra, tab_detalle = st.tabs([
    "📌 Resumen", "📈 Temporal", "🌎 Geografía", "🌦️ Clima", "🚦 Infraestructura", "📋 Detalle"
])

with tab_resumen:
    st.subheader("Resumen ejecutivo")
    col1, col2 = st.columns(2)

    mes = filter_year(data["mes"], selected_years)
    if not mes.empty:
        mes["periodo"] = mes["anio"].astype(str) + "-" + mes["mes"].astype(str).str.zfill(2)
        fig = px.line(mes.sort_values(["anio", "mes"]), x="periodo", y="total_accidentes", markers=True,
                      title="Tendencia mensual de accidentes")
        fig.update_layout(xaxis_title="Periodo", yaxis_title="Accidentes")
        col1.plotly_chart(fig, use_container_width=True)

    sev = filter_year(data["severidad"], selected_years)
    if not sev.empty:
        sev_sum = sev.groupby("severity", as_index=False)["total_accidentes"].sum()
        fig = px.pie(sev_sum, names="severity", values="total_accidentes", title="Distribución por severidad", hole=0.35)
        col2.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    estados = add_rate(filter_year(data["estados"], selected_years))
    if not estados.empty:
        est = estados.groupby("state", as_index=False).agg(total_accidentes=("total_accidentes", "sum"), accidentes_graves=("accidentes_graves", "sum"))
        est = add_rate(est)
        fig = px.bar(top_n(est, metric_col if metric_col in est else "total_accidentes", 15),
                     y="state", x=metric_col if metric_col in est else "total_accidentes", orientation="h",
                     title=f"Top estados por {metric_option.lower()}")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_title=metric_option, yaxis_title="Estado")
        col3.plotly_chart(fig, use_container_width=True)

    clima = add_rate(filter_year(data["clima"], selected_years))
    if not clima.empty:
        cli = clima.groupby("weather_condition", as_index=False).agg(total_accidentes=("total_accidentes", "sum"), accidentes_graves=("accidentes_graves", "sum"))
        cli = add_rate(cli)
        fig = px.bar(top_n(cli, "total_accidentes", 12), y="weather_condition", x="total_accidentes", orientation="h",
                     title="Condiciones climáticas con más accidentes")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_title="Accidentes", yaxis_title="Clima")
        col4.plotly_chart(fig, use_container_width=True)

with tab_temporal:
    st.subheader("Patrones temporales")
    col1, col2 = st.columns(2)

    anio = add_rate(filter_year(data["kpis"], selected_years))
    if not anio.empty:
        fig = px.bar(anio.sort_values("anio"), x="anio", y="total_accidentes", title="Accidentes por año")
        fig.update_layout(xaxis_title="Año", yaxis_title="Accidentes")
        col1.plotly_chart(fig, use_container_width=True)

        fig = px.line(anio.sort_values("anio"), x="anio", y="tasa_graves_pct", markers=True, title="Tasa de gravedad por año")
        fig.update_layout(xaxis_title="Año", yaxis_title="Accidentes graves (%)")
        col2.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    hora = filter_year(data["hora"], selected_years)
    if not hora.empty:
        h = hora.groupby("hora", as_index=False)["total_accidentes"].sum()
        fig = px.line(h.sort_values("hora"), x="hora", y="total_accidentes", markers=True, title="Accidentes por hora del día")
        fig.update_layout(xaxis_title="Hora", yaxis_title="Accidentes")
        col3.plotly_chart(fig, use_container_width=True)

    dia = filter_year(data["dia_semana"], selected_years)
    if not dia.empty:
        dias_en = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dias_es = {"Monday":"Lun", "Tuesday":"Mar", "Wednesday":"Mié", "Thursday":"Jue", "Friday":"Vie", "Saturday":"Sáb", "Sunday":"Dom"}
        d = dia.groupby("dia_semana", as_index=False)["total_accidentes"].sum()
        d["orden"] = d["dia_semana"].map({v:i for i,v in enumerate(dias_en)})
        d["día"] = d["dia_semana"].map(dias_es).fillna(d["dia_semana"])
        fig = px.bar(d.sort_values("orden"), x="día", y="total_accidentes", title="Accidentes por día de la semana")
        fig.update_layout(xaxis_title="Día", yaxis_title="Accidentes")
        col4.plotly_chart(fig, use_container_width=True)

    hd = filter_year(data["hora_dia"], selected_years)
    if not hd.empty:
        st.markdown("#### Heatmap hora × día")
        dias_en = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dias_es = {"Monday":"Lun", "Tuesday":"Mar", "Wednesday":"Mié", "Thursday":"Jue", "Friday":"Vie", "Saturday":"Sáb", "Sunday":"Dom"}
        pivot = hd.groupby(["dia_semana", "hora"], as_index=False)["total_accidentes"].sum()
        pivot["día"] = pivot["dia_semana"].map(dias_es).fillna(pivot["dia_semana"])
        p = pivot.pivot_table(index="día", columns="hora", values="total_accidentes", aggfunc="sum", fill_value=0)
        ordered_days = [dias_es[d] for d in dias_en if dias_es[d] in p.index]
        p = p.reindex(ordered_days)
        fig = px.imshow(p, aspect="auto", title="Concentración de accidentes por hora y día")
        fig.update_layout(xaxis_title="Hora", yaxis_title="Día")
        st.plotly_chart(fig, use_container_width=True)

    col5, col6 = st.columns(2)
    trimestre = filter_year(data["trimestre"], selected_years)
    if not trimestre.empty:
        tri = trimestre.groupby("trimestre", as_index=False)["total_accidentes"].sum()
        fig = px.bar(tri.sort_values("trimestre"), x="trimestre", y="total_accidentes", title="Accidentes por trimestre")
        fig.update_layout(xaxis_title="Trimestre", yaxis_title="Accidentes")
        col5.plotly_chart(fig, use_container_width=True)

    fs = filter_year(data["fin_semana"], selected_years)
    if not fs.empty:
        fs_sum = fs.groupby("tipo_dia", as_index=False)["total_accidentes"].sum()
        fig = px.pie(fs_sum, names="tipo_dia", values="total_accidentes", title="Accidentes: semana vs fin de semana", hole=0.35)
        col6.plotly_chart(fig, use_container_width=True)

with tab_geo:
    st.subheader("Distribución geográfica")
    col1, col2 = st.columns(2)
    estados = add_rate(filter_year(data["estados"], selected_years))
    ciudades = add_rate(filter_year(data["ciudades"], selected_years))

    if not estados.empty:
        est = estados.groupby("state", as_index=False).agg(total_accidentes=("total_accidentes", "sum"), accidentes_graves=("accidentes_graves", "sum"), severidad_promedio=("severidad_promedio", "mean"), duracion_promedio_min=("duracion_promedio_min", "mean"))
        est = add_rate(est)
        fig = px.bar(top_n(est, "total_accidentes", 20), y="state", x="total_accidentes", orientation="h", title="Top 20 estados por accidentes")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_title="Accidentes", yaxis_title="Estado")
        col1.plotly_chart(fig, use_container_width=True)

        fig = px.bar(top_n(est, "tasa_graves_pct", 20), y="state", x="tasa_graves_pct", orientation="h", title="Estados con mayor tasa de gravedad")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_title="Accidentes graves (%)", yaxis_title="Estado")
        col2.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    if not ciudades.empty:
        city = ciudades.groupby(["state", "city"], as_index=False).agg(total_accidentes=("total_accidentes", "sum"), accidentes_graves=("accidentes_graves", "sum"), duracion_promedio_min=("duracion_promedio_min", "mean"))
        city["ciudad_estado"] = city["city"].astype(str) + ", " + city["state"].astype(str)
        city = add_rate(city)
        fig = px.bar(top_n(city, "total_accidentes", 20), y="ciudad_estado", x="total_accidentes", orientation="h", title="Top 20 ciudades por accidentes")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_title="Accidentes", yaxis_title="Ciudad")
        col3.plotly_chart(fig, use_container_width=True)

        fig = px.bar(top_n(city, "accidentes_graves", 20), y="ciudad_estado", x="accidentes_graves", orientation="h", title="Top 20 ciudades por accidentes graves")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_title="Accidentes graves", yaxis_title="Ciudad")
        col4.plotly_chart(fig, use_container_width=True)

    mapa = filter_year(data["mapa"], selected_years)
    if not mapa.empty and {"lat", "lon"}.issubset(mapa.columns):
        st.markdown("#### Mapa agregado por ciudad")
        mapa_sum = mapa.groupby(["state", "city"], as_index=False).agg(
            lat=("lat", "mean"), lon=("lon", "mean"), total_accidentes=("total_accidentes", "sum"), accidentes_graves=("accidentes_graves", "sum")
        )
        mapa_sum = mapa_sum.sort_values("total_accidentes", ascending=False).head(1500)
        fig = px.scatter_mapbox(
            mapa_sum, lat="lat", lon="lon", size="total_accidentes", color="accidentes_graves",
            hover_name="city", hover_data=["state", "total_accidentes", "accidentes_graves"],
            zoom=3, height=600, title="Concentración geográfica de accidentes"
        )
        fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":50,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)

with tab_clima:
    st.subheader("Condiciones climáticas y ambientales")
    col1, col2 = st.columns(2)
    clima = add_rate(filter_year(data["clima"], selected_years))
    if not clima.empty:
        cli = clima.groupby("weather_condition", as_index=False).agg(total_accidentes=("total_accidentes", "sum"), accidentes_graves=("accidentes_graves", "sum"), severidad_promedio=("severidad_promedio", "mean"), duracion_promedio_min=("duracion_promedio_min", "mean"))
        cli = add_rate(cli)
        fig = px.bar(top_n(cli, "total_accidentes", 18), y="weather_condition", x="total_accidentes", orientation="h", title="Top condiciones climáticas")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_title="Accidentes", yaxis_title="Clima")
        col1.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(top_n(cli, "total_accidentes", 40), x="total_accidentes", y="tasa_graves_pct", size="duracion_promedio_min", hover_name="weather_condition", title="Volumen vs tasa de gravedad por clima")
        fig.update_layout(xaxis_title="Accidentes", yaxis_title="Accidentes graves (%)")
        col2.plotly_chart(fig, use_container_width=True)

    bins = [
        ("temperatura", "temperature_bin", "Accidentes por rango de temperatura", "Temperatura"),
        ("humedad", "humidity_bin", "Accidentes por rango de humedad", "Humedad"),
        ("visibilidad", "visibility_bin", "Accidentes por rango de visibilidad", "Visibilidad"),
        ("precipitacion", "precipitation_bin", "Accidentes por rango de precipitación", "Precipitación"),
        ("viento", "wind_speed_bin", "Accidentes por rango de velocidad del viento", "Viento"),
    ]
    for i in range(0, len(bins), 2):
        cols = st.columns(2)
        for j, (key, label_col, title, xlabel) in enumerate(bins[i:i+2]):
            dfb = filter_year(data[key], selected_years)
            if not dfb.empty and label_col in dfb.columns:
                b = dfb.groupby(label_col, as_index=False)["total_accidentes"].sum()
                fig = px.bar(b, x=label_col, y="total_accidentes", title=title)
                fig.update_layout(xaxis_title=xlabel, yaxis_title="Accidentes")
                cols[j].plotly_chart(fig, use_container_width=True)

    mc = filter_year(data["metricas_clima"], selected_years)
    if not mc.empty:
        st.markdown("#### Indicadores ambientales promedio por año")
        mc_long = mc.melt(id_vars="anio", value_vars=[c for c in ["temperatura_promedio", "humedad_promedio", "visibilidad_promedio", "viento_promedio", "precipitacion_promedio"] if c in mc.columns], var_name="variable", value_name="valor")
        fig = px.line(mc_long, x="anio", y="valor", color="variable", markers=True, title="Evolución de variables climáticas promedio")
        st.plotly_chart(fig, use_container_width=True)

with tab_infra:
    st.subheader("Infraestructura vial e iluminación")
    col1, col2 = st.columns(2)
    infra = add_rate(filter_year(data["infra"], selected_years))
    if not infra.empty:
        inf = infra.groupby("infraestructura", as_index=False).agg(total_accidentes=("total_accidentes", "sum"), accidentes_graves=("accidentes_graves", "sum"))
        inf = add_rate(inf)
        fig = px.bar(inf.sort_values("total_accidentes"), y="infraestructura", x="total_accidentes", orientation="h", title="Accidentes asociados a infraestructura vial")
        fig.update_layout(xaxis_title="Accidentes", yaxis_title="Infraestructura")
        col1.plotly_chart(fig, use_container_width=True)

        fig = px.bar(inf.sort_values("tasa_graves_pct"), y="infraestructura", x="tasa_graves_pct", orientation="h", title="Tasa de gravedad por infraestructura")
        fig.update_layout(xaxis_title="Accidentes graves (%)", yaxis_title="Infraestructura")
        col2.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    luz = add_rate(filter_year(data["luz"], selected_years))
    if not luz.empty:
        l = luz.groupby("civil_twilight", as_index=False).agg(total_accidentes=("total_accidentes", "sum"), accidentes_graves=("accidentes_graves", "sum"))
        l = add_rate(l)
        fig = px.pie(l, names="civil_twilight", values="total_accidentes", title="Accidentes según iluminación", hole=0.35)
        col3.plotly_chart(fig, use_container_width=True)

        fig = px.bar(l, x="civil_twilight", y="tasa_graves_pct", title="Tasa de gravedad según iluminación")
        fig.update_layout(xaxis_title="Iluminación", yaxis_title="Accidentes graves (%)")
        col4.plotly_chart(fig, use_container_width=True)

    estado_sev = filter_year(data["estado_severidad"], selected_years)
    if not estado_sev.empty:
        st.markdown("#### Severidad por estado")
        top_states = estados.groupby("state", as_index=False)["total_accidentes"].sum().sort_values("total_accidentes", ascending=False).head(12)["state"].tolist() if not estados.empty else []
        es = estado_sev[estado_sev["state"].isin(top_states)].copy()
        if not es.empty:
            fig = px.bar(es, x="state", y="total_accidentes", color="severity", title="Distribución de severidad en los principales estados")
            fig.update_layout(xaxis_title="Estado", yaxis_title="Accidentes")
            st.plotly_chart(fig, use_container_width=True)

with tab_detalle:
    st.subheader("Tablas agregadas")
    st.write("Estas tablas son resúmenes pequeños generados desde el Data Mart. Puedes usarlas para validar cifras o explicar el modelo BI.")
    option = st.selectbox("Tabla", list(data.keys()))
    df_show = filter_year(data[option], selected_years)
    st.dataframe(df_show, use_container_width=True, height=500)
    st.download_button(
        "Descargar tabla filtrada CSV",
        df_show.to_csv(index=False).encode("utf-8"),
        file_name=f"{option}_filtrado.csv",
        mime="text/csv",
    )
