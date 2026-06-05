from pathlib import Path

import streamlit as st

from dashboard.data import load_mejor_modelo_info, load_shap_importancia

ROOT = Path(__file__).parent.parent.parent


def render() -> None:
    st.subheader("Interpretabilidad y Explicabilidad Científica (SHAP)")
    st.info(
        "La Inteligencia Artificial Explicable (XAI) con valores SHAP (SHapley Additive exPlanations) "
        "permite romper la 'caja negra' del mejor modelo y entender qué variables aumentan o reducen el riesgo suicida."
    )

    # El modelo binario se consolidó como el mejor y mas robusto
    mode = "bin"

    # Carga de datos
    df_shap = load_shap_importancia(mode)
    info = load_mejor_modelo_info(mode)

    if df_shap.empty or not info:
        st.warning("No se encontraron resultados SHAP para este modo. Asegúrate de ejecutar el pipeline primero.")
        return

    st.markdown(f"**Modelo actualmente explicado:** `{info['mejor_modelo']}`")
    st.divider()

    # Importancia global de variables
    st.markdown("### Importancia global de variables")
    st.caption("Representa el impacto promedio de cada variable en la salida de predicción del modelo.")

    col1, col2 = st.columns([3, 2])
    with col1:
        fig_global = ROOT / "figuras" / "shap" / "18_shap_importancia_global.png"
        if not fig_global.exists():
            fig_global = ROOT / "figuras" / "shap" / f"18_shap_importancia_global_{mode}.png"

        if fig_global.exists():
            st.image(
                str(fig_global),
                caption="Las 15 variables predictoras con mayor impacto SHAP.",
                width="stretch",
            )
        else:
            st.caption("Ejecuta el pipeline de SHAP para generar la figura de importancia global.")

    with col2:
        st.markdown("**Valores de Impacto SHAP en Tabla:**")
        # Mostrar tabla formateada
        df_disp = df_shap.copy()
        df_disp.columns = ["Variable Predictora", "Impacto SHAP Medio"]
        st.dataframe(df_disp, hide_index=True, width="stretch", height=400)

    st.divider()

    # Beeswarm por clase
    st.markdown("### Gráficos beeswarm de distribución de riesgo")
    st.caption(
        "Muestran cómo los valores altos (rojo) o bajos (azul) de cada variable predictora "
        "empujan el riesgo hacia la derecha (aumenta el riesgo de esa clase) o hacia la izquierda (reduce el riesgo)."
    )

    fig_bee = ROOT / "figuras" / "shap" / "19_shap_beeswarm.png"
    if not fig_bee.exists():
        fig_bee = ROOT / "figuras" / "shap" / "19_shap_beeswarm_bin.png"

    if fig_bee.exists():
        st.image(
            str(fig_bee),
            caption="Distribución de fuerzas SHAP para Clasificación Binaria (Intentó vs Nunca).",
            width="stretch",
        )
    else:
        # Intentar cargar por clases específicas
        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            fig_c0 = ROOT / "figuras" / "shap" / "19_shap_beeswarm_bin_clase_0.png"
            if fig_c0.exists():
                st.image(str(fig_c0), caption="Beeswarm Clase: Nunca", width="stretch")
        with sub_col2:
            fig_c1 = ROOT / "figuras" / "shap" / "19_shap_beeswarm_bin_clase_1.png"
            if fig_c1.exists():
                st.image(str(fig_c1), caption="Beeswarm Clase: Intentó", width="stretch")

    st.divider()

    # Guia de interpretacion
    with st.expander("Cómo interpretar estos gráficos"):
        st.markdown("""
        * **Posición en el eje X (SHAP Value):** Un valor a la derecha de la línea cero central (positivo) significa que ese factor **aumenta la probabilidad** de predecir esa clase. Un valor a la izquierda (negativo) significa que la reduce.
        * **Color del punto (Feature Value):** Representa el valor real que tomó esa variable para ese adolescente en la encuesta. **Rojo** significa valor alto (por ejemplo: Edad alta, Presencia de Depresión = 1, Con ASI = 1). **Azul** significa valor bajo (Depresión = 0, Sin ASI = 0).
        * **Densidad (Grosor vertical):** Representa la acumulación de adolescentes que compartieron ese mismo nivel de impacto.
        * *Ejemplo clínico:* Si ves que la variable `depresion` tiene puntos **rojos** acumulados a la **derecha** de la línea cero, significa que tener sintomatología depresiva aumenta fuertemente la probabilidad de predecir un intento de suicidio.
        """)
