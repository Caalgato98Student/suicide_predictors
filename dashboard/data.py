"""Carga y caché de datos y modelos para el dashboard de Streamlit."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st
from joblib import load

# Raíz del proyecto
ROOT: Path = Path(__file__).parent.parent


@st.cache_data
def load_df() -> pd.DataFrame:
    """Carga el dataset de adolescentes limpio con tipos correctos.

    Fuerza la columna ``grupo_edad`` como ``Categorical`` ordenado para que
    Plotly no interprete "10-12" como una fecha.

    Returns:
        DataFrame con los datos descriptivos limpios.
    """
    df = pd.read_csv(ROOT / "data" / "adolescentes_limpio.csv")
    df["grupo_edad"] = pd.Categorical(
        df["grupo_edad"],
        categories=["10-12", "13-15", "16-19"],
        ordered=True,
    )
    return df


@st.cache_data
def load_ml_df() -> pd.DataFrame:
    """Carga el dataset codificado para Machine Learning.

    Returns:
        DataFrame con la matriz de características codificadas para ML.
    """
    return pd.read_csv(ROOT / "data" / "adolescentes_ml.csv")


@st.cache_data
def load_comparacion_modelos(mode: str = "bin") -> pd.DataFrame:
    """Carga la tabla comparativa de rendimiento de los modelos.

    Busca primero la ruta canónica y, si no existe, la ruta con sufijo de modo.

    Args:
        mode: Sufijo del modo de modelado (por defecto ``"bin"``).

    Returns:
        DataFrame con métricas CV de los 4 modelos, o DataFrame vacío si no existe.
    """
    p = ROOT / "output" / "clasificacion" / "comparacion_modelos.csv"
    if not p.exists():
        p = ROOT / "output" / "clasificacion" / f"comparacion_modelos_{mode}.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data
def load_shap_importancia(mode: str = "bin") -> pd.DataFrame:
    """Carga la tabla de importancias SHAP globales.

    Args:
        mode: Sufijo del modo de modelado (por defecto ``"bin"``).

    Returns:
        DataFrame con importancias SHAP, o DataFrame vacío si no existe.
    """
    p = ROOT / "output" / "shap" / "shap_importancia.csv"
    if not p.exists():
        p = ROOT / "output" / "shap" / f"shap_importancia_{mode}.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data
def load_mejor_modelo_info(mode: str = "bin") -> dict:
    """Carga los metadatos del mejor modelo entrenado.

    Args:
        mode: Sufijo del modo de modelado (por defecto ``"bin"``).

    Returns:
        Diccionario con los metadatos del modelo, o diccionario vacío si no existe.
    """
    p = ROOT / "output" / "clasificacion" / "mejor_modelo_info.json"
    if not p.exists():
        p = ROOT / "output" / "clasificacion" / f"mejor_modelo_info_{mode}.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_resource
def load_mejor_modelo(mode: str = "bin"):
    """Carga el mejor modelo serializado con joblib.

    Args:
        mode: Sufijo del modo de modelado (por defecto ``"bin"``).

    Returns:
        Modelo scikit-learn cargado, o ``None`` si no existe el archivo.
    """
    p = ROOT / "output" / "clasificacion" / "mejor_modelo.joblib"
    if not p.exists():
        p = ROOT / "output" / "clasificacion" / f"mejor_modelo_{mode}.joblib"
    return load(p) if p.exists() else None


@st.cache_data
def load_geojson() -> dict:
    """Carga el GeoJSON de estados de México.

    Returns:
        Diccionario con la geometría de los estados.
    """
    with open(ROOT / "data" / "states_simple.geojson", encoding="utf-8") as f:
        return json.load(f)


def load_all() -> tuple:
    """Carga todos los datasets esenciales de una vez.

    Returns:
        Tupla ``(df, ml_df, geojson)`` con los tres datasets principales.
    """
    with st.spinner("Cargando base de datos y modelos..."):
        return (
            load_df(),
            load_ml_df(),
            load_geojson(),
        )
