"""Funciones utilitarias para el dashboard de Streamlit."""

import pandas as pd


def aplicar_filtros(data: pd.DataFrame, sexo: str, edad: str) -> pd.DataFrame:
    """Filtra el DataFrame por sexo y grupo de edad según los valores del sidebar.

    Args:
        data: DataFrame a filtrar (dataset de adolescentes limpio).
        sexo: Valor de sexo seleccionado en el sidebar ("Todos", "Hombre" o "Mujer").
        edad: Grupo de edad seleccionado en el sidebar ("Todos", "10-12", "13-15" o "16-19").

    Returns:
        DataFrame filtrado según los criterios indicados.
    """
    if sexo != "Todos":
        data = data[data["sexo"] == sexo]
    if edad != "Todos":
        data = data[data["grupo_edad"] == edad]
    return data


def etiqueta_filtros(sexo: str, edad: str) -> str:
    """Genera una etiqueta legible a partir de los filtros activos.

    Args:
        sexo: Valor de sexo activo en el sidebar.
        edad: Grupo de edad activo en el sidebar.

    Returns:
        Cadena con los filtros activos separados por " · ", o
        "Todos los adolescentes" si no hay filtros aplicados.
    """
    partes = []
    if sexo != "Todos":
        partes.append(sexo)
    if edad != "Todos":
        partes.append(f"{edad} años")
    return " · ".join(partes) if partes else "Todos los adolescentes"
