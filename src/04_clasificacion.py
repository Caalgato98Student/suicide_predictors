"""Compara 4 modelos para clasificacion binaria (intentó vs. no intentó).
Genera metricas CV, matrices de confusion y guarda el mejor modelo.
Incluye hold-out test set (80/20) basado en grupos (UPM) e imputación/selección per-fold para evitar data leakage.
"""

# Standard library
import json
from pathlib import Path

# Third-party
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from joblib import dump
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Local
from utils import CAT_MAUVE, DATA_ML, BalancedGradientBoosting

# Habilitar ruteo de metadatos para soportar sample_weight y groups correctamente en scikit-learn >= 1.4
sklearn.set_config(enable_metadata_routing=True)

DATA_IN: Path = DATA_ML
OUT_DIR: Path = Path("output/clasificacion")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR: Path = Path("figuras/clasificacion")
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "intento_bin"
SEED = 42
N_FOLDS = 5
MIN_FEATURES = 3
CLASE_LABELS = {0: "Nunca", 1: "Intentó"}

print("Clasificacion binaria — 4 modelos (con hold-out test set y diseño muestral)")

# Carga de datos
df = pd.read_csv(DATA_IN, encoding="utf-8-sig")

y_all = df[TARGET]
upm_all = df["upm"]
ponde_all = df["ponde_f"]

# Predictoras: excluir target y variables de diseño muestral
X_all = df.drop(columns=[TARGET, "upm", "estrato", "ponde_f"])

print(f"Features iniciales           : {X_all.shape[1]}")
print(f"Observaciones totales        : {len(X_all)}")
print(f"Clases                       : {sorted(y_all.unique())}")

# Hold-out split 80/20 respetando grupos de UPM
sgkf_outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
train_idx, test_idx = next(sgkf_outer.split(X_all, y_all, groups=upm_all))

X_dev = X_all.iloc[train_idx].copy()
y_dev = y_all.iloc[train_idx]
upm_dev = upm_all.iloc[train_idx]
ponde_dev = ponde_all.iloc[train_idx]

X_test = X_all.iloc[test_idx].copy()
y_test = y_all.iloc[test_idx]
upm_test = upm_all.iloc[test_idx]

overlap = set(upm_dev).intersection(set(upm_test))

print("\nHold-out split (Stratified Group 80/20):")
print(f"  Dev  : {len(X_dev)} obs  (clase 1: {y_dev.sum()})  |  UPMs: {upm_dev.nunique()}")
print(f"  Test : {len(X_test)} obs  (clase 1: {y_test.sum()})  |  UPMs: {upm_test.nunique()}")
print(f"  Traslape de UPMs entre Dev y Test: {len(overlap)}")

# Modelos con ruteo de pesos muestrales
modelos = {
    "Regresión Logística": Pipeline(
        [
            ("scaler", StandardScaler().set_fit_request(sample_weight=False)),
            (
                "clf",
                LogisticRegression(class_weight="balanced", max_iter=5000, random_state=SEED).set_fit_request(
                    sample_weight=True
                ),
            ),
        ]
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=400, class_weight="balanced", random_state=SEED, n_jobs=-1
    ).set_fit_request(sample_weight=True),
    "Gradient Boosting": BalancedGradientBoosting(random_state=SEED).set_fit_request(sample_weight=True),
    "SVM (RBF)": Pipeline(
        [
            ("scaler", StandardScaler().set_fit_request(sample_weight=False)),
            (
                "clf",
                SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=SEED).set_fit_request(
                    sample_weight=True
                ),
            ),
        ]
    ),
}

cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# Scorer con ruteo de pesos
mcc_scorer = make_scorer(matthews_corrcoef).set_score_request(sample_weight=True)

# Validación cruzada sobre el dev set (RFECV + imputación per-fold)
print("\nEvaluando modelos por CV con RFECV e imputación per-fold (evita data leakage)...")
preds_oof = {nombre: np.zeros(len(X_dev), dtype=int) for nombre in modelos}
proba_oof = {nombre: np.zeros(len(X_dev)) for nombre in modelos}
METRICAS_CV = ["MCC", "balanced_accuracy", "f1_macro", "sensibilidad", "especificidad", "ppv", "auc_roc"]
scores_folds = {nombre: {m: [] for m in METRICAS_CV} for nombre in modelos}

for fold, (idx_train, idx_val) in enumerate(cv.split(X_dev, y_dev, groups=upm_dev), 1):
    print(f"  -> Procesando Fold {fold}/{N_FOLDS}...")

    X_train_fold = X_dev.iloc[idx_train].copy()
    X_val_fold = X_dev.iloc[idx_val].copy()
    y_train_fold, y_val_fold = y_dev.iloc[idx_train], y_dev.iloc[idx_val]
    upm_train_fold = upm_dev.iloc[idx_train]
    ponde_train_fold = ponde_dev.iloc[idx_train]

    # Imputación per-fold: fit en train, transform en ambos
    imputer = SimpleImputer(strategy="median")
    X_train_fold = pd.DataFrame(
        imputer.fit_transform(X_train_fold),
        columns=X_train_fold.columns,
        index=X_train_fold.index,
    )
    X_val_fold = pd.DataFrame(
        imputer.transform(X_val_fold),
        columns=X_val_fold.columns,
        index=X_val_fold.index,
    )

    # RFECV local en train (evita data leakage geográfico)
    cv_interno = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)

    estimador_sel = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    estimador_sel.set_fit_request(sample_weight=True)

    rfecv_local = RFECV(
        estimator=estimador_sel,
        step=1,
        cv=cv_interno,
        scoring=mcc_scorer,
        min_features_to_select=MIN_FEATURES,
        n_jobs=-1,
    )

    rfecv_local.fit(X_train_fold, y_train_fold, groups=upm_train_fold, sample_weight=ponde_train_fold)
    features_fold = X_train_fold.columns[rfecv_local.support_].tolist()
    print(f"     [Fold {fold}] Variables seleccionadas: {len(features_fold)}")

    X_train_sel = X_train_fold[features_fold]
    X_val_sel = X_val_fold[features_fold]

    for nombre, modelo in modelos.items():
        modelo_fold = clone(modelo)

        modelo_fold.fit(X_train_sel, y_train_fold, sample_weight=ponde_train_fold)

        preds = modelo_fold.predict(X_val_sel)
        proba = modelo_fold.predict_proba(X_val_sel)[:, 1]
        preds_oof[nombre][idx_val] = preds
        proba_oof[nombre][idx_val] = proba

        scores_folds[nombre]["MCC"].append(matthews_corrcoef(y_val_fold, preds))
        scores_folds[nombre]["balanced_accuracy"].append(balanced_accuracy_score(y_val_fold, preds))
        scores_folds[nombre]["f1_macro"].append(f1_score(y_val_fold, preds, average="macro"))
        scores_folds[nombre]["sensibilidad"].append(recall_score(y_val_fold, preds, pos_label=1, zero_division=0))
        scores_folds[nombre]["especificidad"].append(recall_score(y_val_fold, preds, pos_label=0, zero_division=0))
        scores_folds[nombre]["ppv"].append(precision_score(y_val_fold, preds, pos_label=1, zero_division=0))
        scores_folds[nombre]["auc_roc"].append(roc_auc_score(y_val_fold, proba))

# Métricas CV consolidadas
resultados = []
for nombre in modelos:
    fila = {"modelo": nombre}
    for m in METRICAS_CV:
        fila[f"{m}_media"] = np.mean(scores_folds[nombre][m])
        fila[f"{m}_sd"] = np.std(scores_folds[nombre][m])
    resultados.append(fila)
    print(
        f"  {nombre:<22} MCC = {fila['MCC_media']:.3f} ± {fila['MCC_sd']:.3f}"
        f"  |  Sens = {fila['sensibilidad_media']:.3f}"
        f"  |  Spec = {fila['especificidad_media']:.3f}"
        f"  |  PPV  = {fila['ppv_media']:.3f}"
        f"  |  AUC  = {fila['auc_roc_media']:.3f}"
    )

tabla = pd.DataFrame(resultados).sort_values("MCC_media", ascending=False)
tabla.to_csv(OUT_DIR / "comparacion_modelos.csv", index=False, encoding="utf-8-sig")
print(f"\nComparación -> {OUT_DIR / 'comparacion_modelos.csv'}")
print("\n" + tabla.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

mejor_nombre = tabla.iloc[0]["modelo"]
print(f"\n>> Mejor modelo (MCC): {mejor_nombre}")

# Reporte de clasificación del mejor (out-of-fold sobre dev set)
print(f"\nReporte de clasificación (out-of-fold, dev set) — {mejor_nombre}:")
print(
    classification_report(
        y_dev, preds_oof[mejor_nombre], target_names=[CLASE_LABELS[c] for c in sorted(y_dev.unique())], zero_division=0
    )
)

# Matrices de confusión (out-of-fold, dev set)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
etiquetas = [CLASE_LABELS[c] for c in sorted(y_dev.unique())]
for ax, (nombre, yhat) in zip(axes.ravel(), preds_oof.items(), strict=False):
    cm = confusion_matrix(y_dev, yhat, labels=sorted(y_dev.unique()))
    disp = ConfusionMatrixDisplay(cm, display_labels=etiquetas)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    mcc_val = tabla.loc[tabla["modelo"] == nombre, "MCC_media"].values[0]
    ax.set_title(f"{nombre}\nMCC = {mcc_val:.3f}")
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
fig.suptitle("Matrices de confusión (out-of-fold, 5-fold CV sobre dev set)", fontsize=14)
fig.tight_layout()
fig.savefig(FIG_DIR / "16_matrices_confusion.png", dpi=150)
plt.close(fig)
print(f"\nMatrices de confusión -> {FIG_DIR / '16_matrices_confusion.png'}")

fig, ax = plt.subplots(figsize=(8, 5))
orden = tabla.sort_values("MCC_media")
ax.barh(orden["modelo"], orden["MCC_media"], xerr=orden["MCC_sd"], color=CAT_MAUVE, capsize=4)
ax.set_xlabel("MCC (validación cruzada)")
ax.set_title("Comparación de algoritmos — MCC")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR / "17_comparacion_mcc.png", dpi=150)
plt.close(fig)
print(f"Comparación MCC       -> {FIG_DIR / '17_comparacion_mcc.png'}")

# Curvas ROC (OOF, dev set, los 4 modelos superpuestos)
# Colores por modelo
COLORES: dict[str, str] = {
    "Regresión Logística": "#7c3aed",
    "Gradient Boosting": "#0ea5e9",
    "SVM (RBF)": "#f59e0b",
    "Random Forest": "#10b981",
}

fig, ax = plt.subplots(figsize=(8, 7))

for nombre in modelos:
    auc_val = tabla.loc[tabla["modelo"] == nombre, "auc_roc_media"].values[0]
    fpr, tpr, _ = roc_curve(y_dev, proba_oof[nombre])
    ax.plot(fpr, tpr, color=COLORES.get(nombre), linewidth=2, label=f"{nombre}  (AUC = {auc_val:.3f})")

ax.plot([0, 1], [0, 1], linestyle="--", color="#9ca3af", linewidth=1.2, label="Clasificador aleatorio")

ax.set_xlabel("Tasa de Falsos Positivos (1 − Especificidad)", fontsize=11)
ax.set_ylabel("Tasa de Verdaderos Positivos (Sensibilidad)", fontsize=11)
ax.set_title("Curvas ROC — comparación de algoritmos\n(predicciones out-of-fold, 5-fold CV, dev set)", fontsize=12)
ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(-0.01, 1.01)
fig.tight_layout()
fig.savefig(FIG_DIR / "18_curvas_roc.png", dpi=150)
plt.close(fig)
print(f"Curvas ROC            -> {FIG_DIR / '18_curvas_roc.png'}")

# Selección de features finales sobre el Dev Set completo
print("\nSeleccionando características finales en todo el Dev set (RFECV sin fuga)...")
# Imputación sobre todo el dev set
imputer_dev = SimpleImputer(strategy="median")
X_dev_imp = pd.DataFrame(
    imputer_dev.fit_transform(X_dev),
    columns=X_dev.columns,
    index=X_dev.index,
)

cv_final_sel = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
estimador_final_sel = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=SEED,
    n_jobs=-1,
)
estimador_final_sel.set_fit_request(sample_weight=True)

rfecv_final = RFECV(
    estimator=estimador_final_sel,
    step=1,
    cv=cv_final_sel,
    scoring=mcc_scorer,
    min_features_to_select=MIN_FEATURES,
    n_jobs=-1,
)
# Fit en todo el Dev set respetando grupos y pesos
rfecv_final.fit(X_dev_imp, y_dev, groups=upm_dev, sample_weight=ponde_dev)

n_opt = rfecv_final.n_features_
features_sel = X_dev_imp.columns[rfecv_final.support_].tolist()

print(f"\nNº óptimo de features seleccionadas en Dev   : {n_opt}")
print("Features finalistas:")
for f in features_sel:
    print(f"  - {f}")

# Guardar ranking y features seleccionadas para mantener compatibilidad con 05_shap.py
OUT_SELECCION_DIR = Path("output/seleccion")
OUT_SELECCION_DIR.mkdir(parents=True, exist_ok=True)

ranking = (
    pd.DataFrame({"feature": X_dev.columns, "ranking": rfecv_final.ranking_, "seleccionada": rfecv_final.support_})
    .sort_values(["ranking", "feature"])
    .reset_index(drop=True)
)
ranking.to_csv(OUT_SELECCION_DIR / "ranking_features.csv", index=False, encoding="utf-8-sig")
print(f"Ranking final guardado -> {OUT_SELECCION_DIR / 'ranking_features.csv'}")

with open(OUT_SELECCION_DIR / "features_seleccionadas.json", "w", encoding="utf-8") as fh:
    json.dump({"target": TARGET, "n_features": int(n_opt), "features": features_sel}, fh, ensure_ascii=False, indent=2)
print(f"Lista de features finalistas guardada -> {OUT_SELECCION_DIR / 'features_seleccionadas.json'}")

# Graficar curva de selección de características
scores = rfecv_final.cv_results_["mean_test_score"]
stds = rfecv_final.cv_results_["std_test_score"]
n_features_axis = np.arange(MIN_FEATURES, MIN_FEATURES + len(scores))

FIG_SELECCION_DIR = Path("figuras/seleccion")
FIG_SELECCION_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(n_features_axis, scores, marker="o", color="#2171B5")
ax.fill_between(n_features_axis, scores - stds, scores + stds, alpha=0.2, color="#6BAED6")
ax.axvline(n_opt, ls="--", color="#C23B22", label=f"Óptimo = {n_opt} features")
ax.set_xlabel("Número de características")
ax.set_ylabel("MCC (validación cruzada)")
ax.set_title(
    "RFECV — selección de características (clasificacion binaria)\n"
    "Random Forest balanceado, 5-fold CV por UPM sobre Dev Set"
)
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_SELECCION_DIR / "15_rfecv_seleccion.png", dpi=150)
plt.close(fig)
print(f"Curva de selección guardada -> {FIG_SELECCION_DIR / '15_rfecv_seleccion.png'}")

# Evaluación en hold-out test set
print("\n--- Evaluación en hold-out test set ---")

# Imputar con el subconjunto finalista (fit en dev, transform en test)
imputer_final = SimpleImputer(strategy="median")
X_dev_sel_imp = pd.DataFrame(
    imputer_final.fit_transform(X_dev[features_sel]),
    columns=features_sel,
    index=X_dev.index,
)
X_test_sel_imp = pd.DataFrame(
    imputer_final.transform(X_test[features_sel]),
    columns=features_sel,
    index=X_test.index,
)

mejor_modelo = clone(modelos[mejor_nombre])

mejor_modelo.fit(X_dev_sel_imp, y_dev, sample_weight=ponde_dev)

preds_test = mejor_modelo.predict(X_test_sel_imp)
proba_test = mejor_modelo.predict_proba(X_test_sel_imp)[:, 1]

mcc_test = matthews_corrcoef(y_test, preds_test)
ba_test = balanced_accuracy_score(y_test, preds_test)
f1_test = f1_score(y_test, preds_test, average="macro")
sens_test = recall_score(y_test, preds_test, pos_label=1, zero_division=0)
spec_test = recall_score(y_test, preds_test, pos_label=0, zero_division=0)
ppv_test = precision_score(y_test, preds_test, pos_label=1, zero_division=0)
auc_test = roc_auc_score(y_test, proba_test)

print(f"  MCC (test)              : {mcc_test:.3f}")
print(f"  Balanced Accuracy (test): {ba_test:.3f}")
print(f"  F1 macro (test)         : {f1_test:.3f}")
print(f"  Sensibilidad (test)     : {sens_test:.3f}")
print(f"  Especificidad (test)    : {spec_test:.3f}")
print(f"  PPV (test)              : {ppv_test:.3f}")
print(f"  AUC-ROC (test)          : {auc_test:.3f}")
print(f"\nReporte de clasificación (hold-out test) — {mejor_nombre}:")
print(
    classification_report(
        y_test, preds_test, target_names=[CLASE_LABELS[c] for c in sorted(y_test.unique())], zero_division=0
    )
)

# Modelo final: reentrenar en todos los datos (dev + test) con las features finalistas
X_final = df[features_sel]
imputer_all = SimpleImputer(strategy="median")
X_final_imp = pd.DataFrame(
    imputer_all.fit_transform(X_final),
    columns=features_sel,
    index=df.index,
)

mejor_modelo_final = clone(modelos[mejor_nombre])

mejor_modelo_final.fit(X_final_imp, y_all, sample_weight=ponde_all)
dump(mejor_modelo_final, OUT_DIR / "mejor_modelo.joblib")

info = {
    "mejor_modelo": mejor_nombre,
    "target": TARGET,
    "features": features_sel,
    "clases": sorted(int(c) for c in y_all.unique()),
    "n_dev": int(len(X_dev)),
    "n_test": int(len(X_test)),
    "metricas_cv": tabla[tabla["modelo"] == mejor_nombre].iloc[0].to_dict(),
    "metricas_holdout": {
        "MCC": mcc_test,
        "balanced_accuracy": ba_test,
        "f1_macro": f1_test,
        "sensibilidad": sens_test,
        "especificidad": spec_test,
        "ppv": ppv_test,
        "auc_roc": auc_test,
    },
}
with open(OUT_DIR / "mejor_modelo_info.json", "w", encoding="utf-8") as fh:
    json.dump(info, fh, ensure_ascii=False, indent=2, default=float)

print(f"\nMejor modelo reentrenado (todos los datos) -> {OUT_DIR / 'mejor_modelo.joblib'}")
print(f"Metadatos                                  -> {OUT_DIR / 'mejor_modelo_info.json'}")
