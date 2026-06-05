"""
Preparacion descriptiva desde dataset crudo.

Lee data/adolescentes_ensanut2024_w.csv y genera:
  data/adolescentes_limpio.csv
"""

# Standard library

# Third-party
import numpy as np
import pandas as pd

# Local
from utils import DATA_LIMPIO, DATA_RAW, to_num

# Cargar base cruda
df = pd.read_csv(DATA_RAW, sep=";", encoding="utf-8-sig", low_memory=False)
print(f"Registros originales en la base cruda: {len(df)}")

print("\n[A] Generando dataset limpio para analisis descriptivo...")

df_desc = df.copy()

# Columnas clave a numerico
cols_num = [
    "sexo",
    "edad",
    "ponde_f",
    "d0810",
    "d0811",
    "d0812",
    "d0817",
    "d0819",
    "D0821A",
    "D0821B",
    "D0821C",
    "D0821D",
    "D0821E1",
    "D0821F",
    "D0821G",
    "D0821H",
    "D0821I",
    "D0821J",
    "D0821K",
    "D0821L",
    "d0801",
    "d0813",
    "d0814",
    "d0815",
]
df_desc[cols_num] = df_desc[cols_num].apply(to_num)

# Codigos de no respuesta por variable
NAN_BY_COL = {
    "d0810": [8, 9],
    "d0811": [9],
    "d0812": [9],
    "d0817": [8],
    "d0801": [9],
    "d0813": [77, 99],
    "d0814": [88],
    "d0815": [77, 88, 99],
}
for col, codes in NAN_BY_COL.items():
    df_desc[col] = df_desc[col].where(~df_desc[col].isin(codes), other=np.nan)

# Normaliza metodos de intento
METHOD_COLS = ["D0821A", "D0821B", "D0821C"]
for c in METHOD_COLS:
    df_desc[c] = df_desc[c].where(~df_desc[c].isin([77, 99]), other=np.nan)

# Variables derivadas basicas
df_desc["asi"] = np.where(
    df_desc["d0810"].isin([1, 2]),
    1,
    np.where(df_desc["d0810"] == 3, 0, np.nan),
)

df_desc["ideacion"] = np.where(
    df_desc["d0817"] == 1,
    1,
    np.where(df_desc["d0817"] == 2, 0, np.nan),
)

df_desc["intento"] = np.where(
    df_desc["d0819"].isin([1, 2]),
    1,
    np.where(df_desc["d0819"] == 3, 0, np.nan),
)

df_desc["grupo_edad"] = pd.cut(
    df_desc["edad"],
    bins=[9, 12, 15, 19],
    labels=["10-12", "13-15", "16-19"],
)

# Mapa de metodos de intento
METHOD_MAP = {
    1: "metodo_medicamentos",
    2: "metodo_narcoticos",
    3: "metodo_alcohol",
    4: "metodo_hidrocarburos",
    5: "metodo_fumigantes",
    6: "metodo_quimicos",
    7: "metodo_ahorcamiento",
    8: "metodo_arma_fuego",
    9: "metodo_quemadura",
    10: "metodo_cortantes",
    11: "metodo_arrojarse",
}
intento_mask = df_desc["intento"] != 1
method_data = {}
for code, name in METHOD_MAP.items():
    col = df_desc[METHOD_COLS].isin([code]).any(axis=1).astype(float)
    method_data[name] = col.where(~intento_mask, other=np.nan)

df_desc = pd.concat([df_desc, pd.DataFrame(method_data, index=df_desc.index)], axis=1)

# Renombre a etiquetas legibles
rename_cols = {
    "sexo": "sexo",
    "edad": "edad",
    "ponde_f": "ponderador",
    "d0810": "abuso_sexual",
    "d0811": "sexo_agresor",
    "d0812": "parentesco_agresor",
    "d0819": "intento_suicidio",
    "d0801": "violencia_reciente",
    "d0813": "atencion_tras_asi",
    "d0814": "denuncia_asi",
    "d0815": "razon_no_denuncia",
}

keep_cols = (
    list(rename_cols.keys())
    + [
        "asi",
        "ideacion",
        "intento",
        "grupo_edad",
    ]
    + list(METHOD_MAP.values())
    + ["estrato", "upm", "entidad"]
)

df_limpio_final = df_desc[keep_cols].rename(columns=rename_cols)

# Etiquetas legibles para categorias principales
df_limpio_final["sexo"] = df_limpio_final["sexo"].map({1: "Hombre", 2: "Mujer"})
df_limpio_final["asi_label"] = df_limpio_final["asi"].map({1: "Con ASI", 0: "Sin ASI"})
df_limpio_final["ideacion_label"] = df_limpio_final["ideacion"].map({1: "Con ideacion", 0: "Sin ideacion"})
df_limpio_final["intento_label"] = df_limpio_final["intento"].map({1: "Con intento", 0: "Sin intento"})

n_antes = len(df_limpio_final)
df_limpio_final = df_limpio_final[df_limpio_final["asi"].notna() & df_limpio_final["intento"].notna()].copy()
print(f"  Listwise deletion (ASI + intento): {n_antes} -> {len(df_limpio_final)} filas")

DATA_LIMPIO.parent.mkdir(parents=True, exist_ok=True)
df_limpio_final.to_csv(DATA_LIMPIO, index=False, encoding="utf-8-sig")
print(f"  -> Dataset descriptivo limpio guardado en: {DATA_LIMPIO} ({len(df_limpio_final)} filas)")
