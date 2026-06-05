"""Calcula valores SHAP para el mejor modelo (bin) y guarda figuras/tablas."""

# Standard library
import json
import sys
from pathlib import Path

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline

# Local
# Se importa utils para que esté cargado en sys.modules y joblib pueda deserializar BalancedGradientBoosting
from utils import DATA_ML

try:
    import shap
except ImportError:
    sys.exit("ERROR: falta la librería 'shap'.\nInstálala con:  .venv\\Scripts\\python.exe -m pip install shap")

DATA_IN: Path = DATA_ML
OUT_DIR: Path = Path("output/shap")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR: Path = Path("figuras/shap")
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEED = 0

print("Interpretabilidad SHAP — clasificacion binaria")

# Cargar modelo y metadatos
modelo = load(Path("output/clasificacion") / "mejor_modelo.joblib")
with open(Path("output/clasificacion") / "mejor_modelo_info.json", encoding="utf-8") as fh:
    info = json.load(fh)
features = info["features"]
print(f"Mejor modelo : {info['mejor_modelo']}")
print(f"Features     : {len(features)}")

df = pd.read_csv(DATA_IN, encoding="utf-8-sig")
X = df[features].copy()

# Imputación global con mediana (aceptable para explicar el modelo final)
n_nan = X.isna().sum().sum()
X = X.fillna(X.median())
print(f"  Imputados {n_nan} valores NaN con mediana global")

# Estimador final (desempaqueta el pipeline si aplica)
final_est = modelo.steps[-1][1] if isinstance(modelo, Pipeline) else modelo
# TreeExplainer se usa con Random Forest y Gradient Boosting; el resto usa predict_proba.
es_arbol = isinstance(final_est, (RandomForestClassifier, GradientBoostingClassifier))

# Muestras para SHAP (submuestreo por velocidad)
X_explica = shap.utils.sample(X, min(200, len(X)), random_state=SEED)

print("\nCalculando valores SHAP...")
if es_arbol:
    explainer = shap.TreeExplainer(final_est)
    expl = explainer(X_explica, check_additivity=False)
else:
    # Modelo no basado en árboles (LogReg / SVM): explainer agnóstico sobre
    # predict_proba, con un fondo (background) reducido para que sea viable.
    X_fondo = shap.utils.sample(X, min(50, len(X)), random_state=SEED)
    explainer = shap.Explainer(modelo.predict_proba, X_fondo)
    expl = explainer(X_explica)

vals = np.asarray(expl.values)
print(f"  Forma de los valores SHAP: {vals.shape}")

# Importancia global: media de |SHAP|
abs_mean = np.abs(vals).mean(axis=0)  # (f,) o (f, n_clases)
importancia = abs_mean.mean(axis=1) if abs_mean.ndim == 2 else abs_mean

imp_df = (
    pd.DataFrame({"feature": features, "importancia_shap": importancia})
    .sort_values("importancia_shap", ascending=False)
    .reset_index(drop=True)
)
imp_df.to_csv(OUT_DIR / "shap_importancia.csv", index=False, encoding="utf-8-sig")
print(f"\nImportancia global -> {OUT_DIR / 'shap_importancia.csv'}")
print(imp_df.head(15).to_string(index=False))

# Figura de importancia global (matplotlib propio, robusto entre versiones)
top = imp_df.head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(top["feature"], top["importancia_shap"], color="#89b4fa")
ax.set_xlabel("Importancia media |SHAP|")
ax.set_title(f"Importancia global de variables (SHAP)\nModelo: {info['mejor_modelo']}")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR / "18_shap_importancia_global.png", dpi=150)
plt.close(fig)
print(f"Figura global      -> {FIG_DIR / '18_shap_importancia_global.png'}")

# Beeswarm (clasificacion binaria — clase 1 = "Intentó")
# PermutationExplainer devuelve shape (n, f, 2); beeswarm necesita 2D.
# Seleccionamos clase 1 (positiva) que es el outcome de interes clinico.
try:
    expl_plot = expl[:, :, 1] if vals.ndim == 3 else expl
    plt.figure()
    shap.plots.beeswarm(expl_plot, show=False, max_display=12)
    plt.title('Impacto de variables en "Intentó" (SHAP — clase positiva)')
    plt.tight_layout()
    plt.savefig(FIG_DIR / "19_shap_beeswarm.png", dpi=150)
    plt.close()
    print(f"Beeswarm -> {FIG_DIR / '19_shap_beeswarm.png'}")
except Exception as e:
    print(f"  (No se pudo generar beeswarm: {e})")
