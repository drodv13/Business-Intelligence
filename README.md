# Proyecto BI - Streamlit Cloud

## Flujo simple

1. Abre `exportar_datamart.ipynb` en Jupyter.
2. Ejecuta todas las celdas.
3. Se creará `data/datamart_accidentes.parquet`.
4. Sube a GitHub:
   - `streamlit_app.py`
   - `requirements.txt`
   - carpeta `data/`
5. En Streamlit Cloud selecciona como archivo principal:
   - `streamlit_app.py`

El dashboard en la nube no usa SQL Server; solo lee el archivo Parquet generado.
