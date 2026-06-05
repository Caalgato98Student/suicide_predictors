from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.data import load_comparacion_modelos, load_mejor_modelo_info

ROOT = Path(__file__).parent.parent.parent


def render() -> None:
    st.subheader("Modelado predictivo con machine learning (KDD)")
    st.info(
        "En esta pestaña se comparan 4 algoritmos de clasificación supervisada entrenados en Python "
        "para predecir el riesgo de intento de suicidio (Intentó vs. Nunca) a partir del perfil bio-psico-social del adolescente."
    )

    # El modelo binario se consolidó como el mejor y más robusto
    mode = "bin"

    # Carga de datos
    df_comp = load_comparacion_modelos(mode)
    info = load_mejor_modelo_info(mode)

    if df_comp.empty or not info:
        st.warning(
            "No se encontraron resultados para este modo de modelado. Asegúrate de ejecutar el pipeline primero."
        )
        return

    # Destacar mejor modelo
    st.markdown("### Mejor modelo seleccionado")
    col1, col2 = st.columns([2, 3])
    with col1:
        st.metric(
            label="Algoritmo ganador (por MCC)",
            value=info["mejor_modelo"],
        )
    with col2:
        mcc_val = info["metricas_cv"]["MCC_media"]
        f1_val = info["metricas_cv"]["f1_macro_media"]
        bal_acc = info["metricas_cv"]["balanced_accuracy_media"]

        c1, c2, c3 = st.columns(3)
        c1.metric("MCC promedio", f"{mcc_val:.3f}")
        c2.metric("F1-score macro", f"{f1_val:.3f}")
        c3.metric("Balanced accuracy", f"{bal_acc:.3f}")

    st.caption("Métricas promedio evaluadas mediante validación cruzada estratificada de 5 Folds.")
    st.divider()

    # Tabla comparativa de todos los modelos
    st.markdown("### Tabla comparativa de rendimiento")
    st.caption("Matthews Correlation Coefficient (MCC) es la métrica principal debido al fuerte desbalance del target.")

    # Formatear la tabla para mostrar media ± sd
    df_disp = pd.DataFrame()
    df_disp["Modelo"] = df_comp["modelo"]
    df_disp["MCC"] = df_comp.apply(lambda r: f"{r['MCC_media']:.3f} ± {r['MCC_sd']:.3f}", axis=1)
    df_disp["Balanced accuracy"] = df_comp.apply(
        lambda r: f"{r['balanced_accuracy_media']:.3f} ± {r['balanced_accuracy_sd']:.3f}", axis=1
    )
    df_disp["F1-score macro"] = df_comp.apply(lambda r: f"{r['f1_macro_media']:.3f} ± {r['f1_macro_sd']:.3f}", axis=1)

    st.dataframe(df_disp, hide_index=True, width="stretch")
    st.divider()

    # Imágenes estáticas generadas por el pipeline de ML
    st.markdown("### Visualización del rendimiento y errores")

    fig_comp = ROOT / "figuras" / "clasificacion" / "17_comparacion_mcc.png"
    if not fig_comp.exists():
        fig_comp = ROOT / "figuras" / "clasificacion" / f"17_comparacion_mcc_{mode}.png"

    if fig_comp.exists():
        st.image(
            str(fig_comp),
            caption="Comparación del Matthews Correlation Coefficient (MCC) por algoritmo.",
            width="stretch",
        )
    else:
        st.caption("Ejecuta el pipeline de ML para generar la gráfica de barras de MCC.")

    st.write("")  # Espaciado vertical

    fig_cm = ROOT / "figuras" / "clasificacion" / "16_matrices_confusion.png"
    if not fig_cm.exists():
        fig_cm = ROOT / "figuras" / "clasificacion" / f"16_matrices_confusion_{mode}.png"

    if fig_cm.exists():
        st.image(
            str(fig_cm),
            caption="Matrices de confusión Out-of-Fold (CV) de los 4 clasificadores.",
            width="stretch",
        )
    else:
        st.caption("Ejecuta el pipeline de ML para generar las matrices de confusión.")

    st.divider()

    # Documentación técnica
    with st.expander("Decisiones de ingeniería de datos y modelado (KDD)"):
        st.markdown(f"""
        * **Target:** El modelo predice la columna `{info["target"]}`.
        * **Selección de características (Feature Selection):** Solo se introducen las `{len(info["features"])}` variables más significativas filtradas de forma automática mediante **Recursive Feature Elimination (RFECV)**.
        * **Desbalance de Clases:** Se implementa balanceo de pesos de clase (`class_weight='balanced'`) en todos los algoritmos para evitar el sesgo hacia la clase mayoritaria (no intento).
        * **Validación Cruzada Estratificada (5-Fold):** Garantiza que las evaluaciones de rendimiento sean generalizables y libres de fugas de información (*data leakage*).
        """)
