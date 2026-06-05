"""Utilidades compartidas para el pipeline de Datos y ML."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

# Rutas estándar

DATA_RAW: Path = Path("data/adolescentes_ensanut2024_w.csv")
DATA_LIMPIO: Path = Path("data/adolescentes_limpio.csv")
DATA_ML: Path = Path("data/adolescentes_ml.csv")

# Códigos de no respuesta en ENSANUT que deben mapearse a NaN

ENSANUT_NULLS: list[int] = [8, 9, 77, 88, 99]

# Paleta de colores (Catppuccin Mocha) — compartida por todos los scripts src/

CAT_BLUE: str = "#89b4fa"
CAT_PINK: str = "#f5c2e7"
CAT_GREEN: str = "#a6e3a1"
CAT_PEACH: str = "#fab387"
CAT_MAUVE: str = "#cba6f7"
CAT_RED: str = "#f38ba8"
CAT_FLAMINGO: str = "#f2cdcd"
CAT_YELLOW: str = "#f9e2af"
CAT_LAVENDER: str = "#b4befe"
CAT_SAPPHIRE: str = "#74c7ec"

COLORS: dict[str, str] = {"Hombre": CAT_BLUE, "Mujer": CAT_PINK, "Total": CAT_GREEN}

# Diccionarios de etiquetas compartidos (mapeos de códigos a texto legible)

METODO_LBL: dict[str, str] = {
    "metodo_medicamentos": "Envenenamiento\nmedicamentos",
    "metodo_narcoticos": "Envenenamiento\nnarcóticos",
    "metodo_alcohol": "Envenenamiento\nalcohol",
    "metodo_hidrocarburos": "Inhalación\nhidrocarburos",
    "metodo_fumigantes": "Fumigantes/\ninsecticidas",
    "metodo_quimicos": "Químicos/\nácidos",
    "metodo_ahorcamiento": "Ahorcamiento",
    "metodo_arma_fuego": "Arma de fuego",
    "metodo_quemadura": "Quemadura",
    "metodo_cortantes": "Objetos\ncortantes",
    "metodo_arrojarse": "Arrojarse\nal vacío",
}

AGRESOR_LBL: dict[int, str] = {
    1: "Pareja",
    2: "Familiar",
    3: "Amigo/a",
    4: "Novio/a",
    5: "Vecino/conocido",
    6: "Desconocido/a",
    7: "Policía",
}

# Funciones de preprocesamiento


def to_num(s: pd.Series) -> pd.Series:
    """Convierte a numérico; maneja coma decimal y espacios vacíos (→ NaN).

    Args:
        s: Serie de pandas a convertir.

    Returns:
        Serie con valores numéricos; los no convertibles quedan como NaN.
    """
    return pd.to_numeric(
        s.astype(str).str.strip().str.replace(",", ".", regex=False),
        errors="coerce",
    )


def clean_ensanut_nans(
    data: pd.DataFrame | pd.Series,
    cols: list[str] | str | None = None,
    null_codes: list[int] | None = None,
) -> pd.DataFrame | pd.Series:
    """Reemplaza los códigos de no respuesta ENSANUT por NaN.

    Puede recibir un DataFrame (y una lista de columnas) o una Serie directamente.

    Args:
        data: DataFrame o Serie a limpiar.
        cols: Columnas a procesar (solo aplica cuando ``data`` es un DataFrame).
            Si es ``None``, se procesan todas las columnas.
        null_codes: Códigos que representan no-respuesta. Por defecto usa
            :data:`ENSANUT_NULLS`.

    Returns:
        DataFrame o Serie con los códigos de no-respuesta sustituidos por NaN.
    """
    if null_codes is None:
        null_codes = ENSANUT_NULLS

    if isinstance(data, pd.Series):
        return data.where(~data.isin(null_codes), np.nan)

    df_clean = data.copy()
    cols = cols or df_clean.columns.tolist()
    if isinstance(cols, str):
        cols = [cols]
    for col in cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].where(~df_clean[col].isin(null_codes), np.nan)
    return df_clean


# Clases para ML


class BalancedGradientBoosting(GradientBoostingClassifier):
    """GradientBoostingClassifier con balanceo automático via sample_weight.

    Calcula pesos balanceados por clase y los combina con cualquier
    ``sample_weight`` externo proporcionado al llamar a :meth:`fit`.

    Example:
        >>> clf = BalancedGradientBoosting(random_state=42)
        >>> clf.fit(X_train, y_train)
    """

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> "BalancedGradientBoosting":
        """Entrena el modelo aplicando balanceo de clases.

        Args:
            X: Matriz de características.
            y: Vector de etiquetas.
            sample_weight: Pesos externos opcionales; se multiplican por los
                pesos balanceados calculados internamente.

        Returns:
            La instancia entrenada.
        """
        bal_weights = compute_sample_weight("balanced", y)
        sample_weight = bal_weights if sample_weight is None else np.array(sample_weight) * bal_weights
        return super().fit(X, y, sample_weight=sample_weight)
