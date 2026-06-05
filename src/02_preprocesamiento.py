"""
KDD: Preprocesamiento y Transformacion para Machine Learning.

Lee data/adolescentes_ensanut2024_w.csv y genera:
  data/adolescentes_ml.csv
"""

# Standard library
from pathlib import Path

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Local
from utils import DATA_ML, DATA_RAW, clean_ensanut_nans, to_num

FIG_DIR = Path("figuras/preprocesamiento")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Cargar base cruda
df = pd.read_csv(DATA_RAW, sep=";", encoding="utf-8-sig", low_memory=False)
print(f"Registros originales en la base cruda: {len(df)}")

# Dataset ML (modelado)
print("\n[ML] Generando dataset codificado para Machine Learning (KDD)...")

feats = pd.DataFrame(index=df.index)

# Sociodemograficas
feats["edad"] = to_num(df["edad"])
feats["sexo"] = to_num(df["sexo"]).map({1: "Hombre", 2: "Mujer"})

# Variables de diseño complejo de ENSANUT
feats["upm"] = df["upm"]
feats["estrato"] = df["estrato"]
feats["ponde_f"] = to_num(df["ponde_f"])

# Exposicion principal: ASI (d0810)
d0810 = to_num(df["d0810"])
feats["asi"] = np.where(d0810.isin([1, 2]), 1, np.where(d0810 == 3, 0, np.nan))

# CES-D 7 items (0-3 por reactivo, reverso item F)
CESD = ["d0601a", "d0601b", "d0601c", "d0601d", "d0601e", "d0601f", "d0601g"]
cesd_raw = clean_ensanut_nans(df[CESD].apply(to_num))

cesd_recoded = pd.DataFrame(index=df.index)
for col in CESD:
    if col == "d0601f":
        cesd_recoded[col] = 4 - cesd_raw[col]
    else:
        cesd_recoded[col] = cesd_raw[col] - 1

feats["depresion_cesd"] = cesd_recoded.sum(axis=1)
feats.loc[cesd_recoded.isna().any(axis=1), "depresion_cesd"] = np.nan

# Conductas alimentarias de riesgo
ALIM = [f"d06a{i}" for i in range(1, 11)]
alim_raw = clean_ensanut_nans(df[ALIM].apply(to_num))
feats["conducta_alim"] = alim_raw.sum(axis=1)
feats.loc[alim_raw.isna().any(axis=1), "conducta_alim"] = np.nan

# Consumo de sustancias — no-respuestas (8, 9, 88, 99) → NaN
_d0101 = clean_ensanut_nans(to_num(df["d0101"]))
feats["fuma_actual"] = np.where(_d0101.isin([1, 2]), 1, np.where(_d0101.isna(), np.nan, 0))

_d0107 = clean_ensanut_nans(to_num(df["d0107"]))
feats["vapea_actual"] = np.where(_d0107.isin([1, 2]), 1, np.where(_d0107.isna(), np.nan, 0))

_d0108 = clean_ensanut_nans(to_num(df["d0108"]))
feats["alcohol_12m"] = np.where(_d0108.isin([1, 2, 3, 4]), 1, np.where(_d0108.isna(), np.nan, 0))

# Binge drinking — no-respuestas → NaN
feats["binge_alcohol"] = np.nan
feats.loc[_d0108.isin([5, 6]), "binge_alcohol"] = 0
# Hombres (sexo==1)
_d0111 = clean_ensanut_nans(to_num(df["d0111"]))
feats.loc[(to_num(df["sexo"]) == 1) & (_d0108.between(1, 4)) & (_d0111 == 1), "binge_alcohol"] = 1
feats.loc[(to_num(df["sexo"]) == 1) & (_d0108.between(1, 4)) & (_d0111 == 2), "binge_alcohol"] = 0
# Mujeres (sexo==2)
_d0112 = clean_ensanut_nans(to_num(df["d0112"]))
feats.loc[(to_num(df["sexo"]) == 2) & (_d0108.between(1, 4)) & (_d0112 == 1), "binge_alcohol"] = 1
feats.loc[(to_num(df["sexo"]) == 2) & (_d0108.between(1, 4)) & (_d0112 == 2), "binge_alcohol"] = 0
# binge_alcohol: NaN se preserva y se imputará con mediana

# Drogas ilegales — no-respuestas → NaN
ILEGALES = [f"d01a08{c}" for c in "abcdefghi"]
_ileg = clean_ensanut_nans(df[ILEGALES].apply(to_num))
# NaN si todos los reactivos son NaN; 1 si al menos uno es positivo; 0 en otro caso
feats["drogas_ilegales"] = np.where(_ileg.isna().all(axis=1), np.nan, _ileg.eq(1).any(axis=1).astype(float))

# Sustancias médicas — no-respuestas → NaN
MEDICAS = [f"d01a03{c}" for c in "abcd"]
_med = clean_ensanut_nans(df[MEDICAS].apply(to_num))
feats["sustancias_medicas"] = np.where(_med.isna().all(axis=1), np.nan, _med.eq(1).any(axis=1).astype(float))

# Factores biopsicosociales

# Biologicos y fisiologicos

# Enfermedades crónicas: diabetes, presión alta, colesterol, triglicéridos (d02o1a-d)
CRONICAS = ["d02o1a", "d02o1b", "d02o1c", "d02o1d"]
_cron = clean_ensanut_nans(df[CRONICAS].apply(to_num))
feats["enfermedad_cronica"] = np.where(_cron.isna().all(axis=1), np.nan, _cron.eq(1).any(axis=1).astype(float))

# Discapacidad física: dificultad severa para caminar o cuidarse (d0407 / d0412, valores 3-4)
_d0407 = clean_ensanut_nans(to_num(df["d0407"]))
_d0412 = clean_ensanut_nans(to_num(df["d0412"]))
feats["discapacidad_fisica"] = np.where(
    (_d0407.isin([3, 4])) | (_d0412.isin([3, 4])), 1, np.where(_d0407.isna() & _d0412.isna(), np.nan, 0)
)

# Psicologicos y cognitivos

# Ansiedad frecuente (d0421): escala invertida — 1=Nunca → 5=Diariamente
_d0421 = clean_ensanut_nans(to_num(df["d0421"]))
feats["ansiedad_frec"] = np.where(_d0421.isna(), np.nan, (6 - _d0421))  # rango 1-5

# Déficit cognitivo: déficit para aprender, recordar o concentrarse (d0415-d0417, valores 3-4)
COG = ["d0415", "d0416", "d0417"]
_cog = clean_ensanut_nans(df[COG].apply(to_num))
feats["deficit_cognitivo"] = np.where(_cog.isna().all(axis=1), np.nan, _cog.isin([3, 4]).any(axis=1).astype(float))

# Desregulación conductual: dificultad para aceptar rutinas o controlar conducta (d0418-d0419)
COND = ["d0418", "d0419"]
_cond = clean_ensanut_nans(df[COND].apply(to_num))
feats["desregulacion_conductual"] = np.where(
    _cond.isna().all(axis=1), np.nan, _cond.isin([3, 4]).any(axis=1).astype(float)
)

# Sociales y contextuales

# Aislamiento social: dificultad para hacer amigos (d0420, valores 3-4 = sin habilidad)
_d0420 = clean_ensanut_nans(to_num(df["d0420"]))
feats["aislamiento_social"] = np.where(_d0420.isin([3, 4]), 1, np.where(_d0420.isna(), np.nan, 0))

# Violencia familiar: disciplina violenta en el último mes (d0901a-k)
FAMVIO = ["d0901a", "d0901b", "d0901c", "d0901d", "d0901e", "d0901f", "d0901g", "d0901h", "d0901i", "d0901j", "d0901k"]
FAMVIO_OK = [c for c in FAMVIO if c in df.columns]  # filtrar columnas que existen
if len(FAMVIO_OK) > 0:
    _famvio = clean_ensanut_nans(df[FAMVIO_OK].apply(to_num))
if len(FAMVIO_OK) > 0:
    feats["violencia_familiar"] = np.where(_famvio.isna().all(axis=1), np.nan, _famvio.eq(1).any(axis=1).astype(float))
else:
    feats["violencia_familiar"] = np.nan
    print("  [WARN] No se encontraron columnas d0901* para violencia_familiar")

# Contexto sexual y reproductivo

_d0309 = clean_ensanut_nans(to_num(df["d0309"]))
feats["relacion_forzada"] = np.where(_d0309 == 1, 1, np.where(_d0309.isna(), np.nan, 0))

_d0313 = clean_ensanut_nans(to_num(df["d0313"]))
feats["embarazo"] = np.where(_d0313 == 1, 1, np.where(_d0313.isna(), np.nan, 0))

_d0801 = clean_ensanut_nans(to_num(df["d0801"]))
feats["violencia_reciente"] = np.where(_d0801 == 1, 1, np.where(_d0801.isna(), np.nan, 0))

# Target
feats["intento_suicidio"] = to_num(df["d0819"])

# Limpieza de nulos
feats = feats[feats["intento_suicidio"].notna()].copy()
feats["intento_suicidio"] = feats["intento_suicidio"].astype(int)

feats = feats[feats["asi"].notna()].copy()
feats["asi"] = feats["asi"].astype(int)

# NaN se preservan; la imputación se delega a los scripts downstream dentro de CV
print("  NaN preservados en adolescentes_ml.csv (imputación delegada a scripts downstream)")

# Target binario derivado
feats["intento_bin"] = feats["intento_suicidio"].isin([1, 2]).astype(int)

# Variables categoricas que SI entran al dataset de ML
CAT_FEATURES_ML = ["sexo"]
NUM_BIN = [
    "edad",
    "asi",
    "depresion_cesd",
    "conducta_alim",
    "fuma_actual",
    "vapea_actual",
    "alcohol_12m",
    "binge_alcohol",
    "drogas_ilegales",
    "sustancias_medicas",
    "enfermedad_cronica",
    "relacion_forzada",
    "embarazo",
    "violencia_reciente",
    # Factores biopsicosociales
    "discapacidad_fisica",
    "ansiedad_frec",
    "deficit_cognitivo",
    "desregulacion_conductual",
    "aislamiento_social",
    "violencia_familiar",
]

# Mapeos para categoricas
MAPS = {
    "sexo": {"Hombre": "Hombre", "Mujer": "Mujer"},
}

df_enc = feats[NUM_BIN].copy()

for c in CAT_FEATURES_ML:
    serie = feats[c].map(MAPS[c])
    serie = serie.fillna("NoResponde")
    dummies = pd.get_dummies(serie, prefix=c, dtype=int, drop_first=True)
    df_enc = pd.concat([df_enc, dummies], axis=1)

df_enc["intento_bin"] = feats["intento_bin"].values
df_enc["upm"] = feats["upm"].values
df_enc["estrato"] = feats["estrato"].values
df_enc["ponde_f"] = feats["ponde_f"].values

DATA_ML.parent.mkdir(parents=True, exist_ok=True)
df_enc.to_csv(DATA_ML, index=False, encoding="utf-8-sig")
print(f"  -> Dataset codificado para ML guardado en: {DATA_ML} ({len(df_enc)} filas)")

# Colinealidad relacion_forzada vs violencia_reciente (umbral: r>=0.70 problemático)
_r_vio = df_enc["relacion_forzada"].corr(df_enc["violencia_reciente"])
_tag = "[WARN] REVISION REQUERIDA" if _r_vio >= 0.70 else "[INFO] Correlacion moderada" if _r_vio >= 0.40 else "[OK]  "
print(f"  {_tag} corr(relacion_forzada, violencia_reciente) = {_r_vio:.4f}")

# Histograma del target binario en ML
clase_labels_bin = {0: "No intentó", 1: "Intentó"}
conteo = feats["intento_bin"].value_counts().sort_index()
prop = (conteo / conteo.sum() * 100).round(1)

fig, ax = plt.subplots(figsize=(6, 5))
colores = ["#89b4fa", "#f38ba8"]
barras = ax.bar([clase_labels_bin[k] for k in conteo.index], conteo.values, color=colores, width=0.5)
for b, n, p in zip(barras, conteo.values, prop.values, strict=False):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{n}\n({p}%)", ha="center", va="bottom", fontsize=11)
ax.set_ylabel("Frecuencia (n adolescentes)")
ax.set_title("Distribución del intento de suicidio (target binario)\nENSANUT 2024 — adolescentes (sin ponderar)")
ax.set_ylim(0, conteo.max() * 1.15)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR / "14_target_intento_frecuencias.png", dpi=150)
plt.close(fig)
print(f"  -> Histograma de target guardado en: {FIG_DIR / '14_target_intento_frecuencias.png'}")
