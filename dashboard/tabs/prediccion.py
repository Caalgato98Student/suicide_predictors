import pandas as pd
import streamlit as st

from dashboard.data import load_mejor_modelo, load_mejor_modelo_info


def render() -> None:
    st.subheader("Simulador clínico de predicción de riesgo en tiempo real")
    st.info(
        "Esta herramienta interactiva utiliza el mejor modelo predictivo entrenado "
        "(Regresión Logística balanceada con RFECV) para estimar la probabilidad personalizada de "
        "riesgo de intento de suicidio en un adolescente ficticio a partir de sus variables bio-psico-sociales."
    )

    # Carga de modelo
    modelo = load_mejor_modelo("bin")
    info = load_mejor_modelo_info("bin")

    if modelo is None or not info:
        st.warning("El modelo predictivo no está disponible. Asegúrate de ejecutar el pipeline primero.")
        return

    features = info["features"]

    st.markdown("### Configuración del perfil del adolescente")
    st.caption(
        "Los predictores que se configuran a continuación corresponden exactamente a las variables "
        "seleccionadas automáticamente por el algoritmo de eliminación recursiva de características (RFECV) "
        "para optimizar el Matthews Correlation Coefficient (MCC)."
    )

    col1, col2, col3 = st.columns(3, gap="medium")

    # COLUMNA 1: PERFIL Y VIOLENCIA
    with col1:
        st.markdown("##### Perfil y violencia")
        edad = st.slider("Edad (años):", 10, 19, 14, help="Rango de edad de la muestra ENSANUT (10 a 19 años).")
        sexo = st.selectbox(
            "Sexo del adolescente:", ["Mujer", "Hombre"], help="Hombre se toma como categoría de referencia implícita."
        )
        violencia_reciente = st.selectbox(
            "¿Sufrió agresión/violencia física/emocional en los últimos 12 meses?", ["No", "Sí"]
        )
        violencia_familiar = st.selectbox(
            "¿Sufrió disciplina violenta familiar (golpes, gritos) en el último mes?", ["No", "Sí"]
        )

        st.write("")
        asi = st.selectbox("¿Sufrió abuso sexual en la infancia/adolescencia?", ["No", "Sí"])

        # Mapeos de etiquetas para variables descriptivas (excluidas del ML)
        PARENTESCO_MAP = {
            "Familiar": "Familiar",
            "Vecino o conocido": "VecinoConocido",
            "Amigo/a": "Amigo",
            "Novio/a": "Novio",
            "Desconocido/a": "Desconocido",
            "Pareja": "Pareja",
            "Policía": "Policia",
        }

        ATENCION_MAP = {
            "Nadie lo atendió": "Nadie",
            "Psicólogo/a o terapeuta": "Psicologo",
            "Médico/a en consultorio": "Medico",
            "Clínica u hospital": "ClinicaHospital",
            "Remedios caseros o automedicación": "RemediosCaseros",
            "Encargado/a de la comunidad": "EncargadoComunidad",
            "Curandero/a": "Curandero",
            "Huesero/a": "Huesero",
        }

        RAZON_MAP = {
            "Miedo": "Miedo",
            "Vergüenza": "Verguenza",
            "Amenazas": "Amenazas",
            "No sabía que podía hacerlo": "NoSabia",
            "Otra razón": "Otro",
        }

        if asi == "Sí":
            # Contenedor descriptivo con fondo oscuro
            st.markdown(
                "<div style='background-color:#181825; border-radius:8px; padding:12px; border:1px solid #313244; margin-bottom:10px;'>"
                "<span style='color:#fab387; font-size:0.85rem; font-weight:bold;'>Nota sobre variables de ASI:</span><br>"
                "<span style='color:#a6adc8; font-size:0.8rem;'>"
                "Los detalles estructurales de la agresión (agresor, denuncia, etc.) se recolectan de forma "
                "<b>descriptiva únicamente</b> y fueron excluidos del modelo de ML para evitar el sesgo proxy "
                "y garantizar la validez ética."
                "</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            _ = st.selectbox("Sexo del agresor (Descriptivo):", ["Hombre", "Mujer"])
            _ = st.selectbox("Relación con el agresor (Descriptivo):", list(PARENTESCO_MAP.keys()))
            _ = st.selectbox("Atención tras ataque (Descriptivo):", list(ATENCION_MAP.keys()))
            denuncia_asi = st.selectbox("¿Se denunció el ataque? (Descriptivo):", ["No", "Sí"])
            if denuncia_asi == "No":
                _ = st.selectbox("Razón de no denuncia (Descriptivo):", list(RAZON_MAP.keys()))

    # COLUMNA 2: SALUD MENTAL Y COGNICIÓN
    with col2:
        st.markdown("##### Salud mental y cognición")
        depresion_cesd = st.slider(
            "Sintomatología depresiva (CES-D 7):",
            0,
            21,
            5,
            help="Suma de los 7 reactivos (0 a 3 puntos cada uno). Puntuación >= 9 indica sintomatología depresiva clínicamente significativa.",
        )

        # Alerta visual de depresión activa en el simulador
        depresion_bin = 1 if depresion_cesd >= 9 else 0
        if depresion_bin == 1:
            st.warning("Puntuación CES-D califica como depresión activa.")

        conducta_alim = st.slider(
            "Riesgo de conductas alimentarias (CES-D-7A):",
            10,
            40,
            15,
            help="Suma de conductas alimentarias de riesgo (rango de 10 a 40 puntos).",
        )

        ansiedad_frec_sel = st.selectbox(
            "Frecuencia de síntomas de ansiedad (d0421):",
            [
                "1. Nunca",
                "2. Raramente / Poco frecuente",
                "3. Algunas veces",
                "4. Muy frecuente",
                "5. Diariamente (Frecuencia máxima)",
            ],
            index=1,
            help="Percepción subjetiva de frecuencia de estados ansiosos en las últimas semanas.",
        )
        # Extraer el entero inicial del string
        ansiedad_frec = int(ansiedad_frec_sel[0])

        st.write("")
        deficit_cognitivo = st.selectbox(
            "¿Tiene dificultad severa para concentrarse, aprender o recordar?", ["No", "Sí"]
        )
        desregulacion_conductual = st.selectbox(
            "¿Tiene gran dificultad para controlar su conducta o aceptar rutinas?", ["No", "Sí"]
        )
        aislamiento_social = st.selectbox(
            "¿Tiene mucha dificultad para hacer amistades o convivir con pares?", ["No", "Sí"]
        )

    # COLUMNA 3: CONSUMO DE SUSTANCIAS
    with col3:
        st.markdown("##### Consumo de sustancias")
        vapea_actual = st.selectbox("¿Vapea (cigarros electrónicos) actualmente?", ["No", "Sí"])
        alcohol_12m = st.selectbox("¿Ha consumido alcohol en los últimos 30 días?", ["No", "Sí"])

        if alcohol_12m == "Sí":
            binge_alcohol = st.selectbox(
                "¿Ha bebido en exceso (4/5+ copas en una sola ocasión) en los últimos 30 días?", ["No", "Sí"]
            )
        else:
            st.caption("Consumo excesivo (Binge drinking) bloqueado (no consume alcohol).")
            binge_alcohol = "No"

        drogas_ilegales = st.selectbox(
            "¿Ha consumido alguna droga ilegal (marihuana, solventes, etc.) alguna vez?", ["No", "Sí"]
        )
        sustancias_medicas = st.selectbox(
            "¿Ha usado medicamentos controlados (sedantes, estimulantes) sin receta?", ["No", "Sí"]
        )

    # Construcción de la fila de predicción
    # Se inicializan a 0 todas las características del mejor modelo seleccionado
    row_df = pd.DataFrame(0, index=[0], columns=features)

    # Llenar la fila con las variables reales
    row_df["edad"] = edad
    row_df["asi"] = 1 if asi == "Sí" else 0
    row_df["depresion_cesd"] = depresion_cesd
    row_df["conducta_alim"] = conducta_alim
    row_df["vapea_actual"] = 1 if vapea_actual == "Sí" else 0
    row_df["alcohol_12m"] = 1 if alcohol_12m == "Sí" else 0
    row_df["binge_alcohol"] = 1 if (alcohol_12m == "Sí" and binge_alcohol == "Sí") else 0
    row_df["drogas_ilegales"] = 1 if drogas_ilegales == "Sí" else 0
    row_df["sustancias_medicas"] = 1 if sustancias_medicas == "Sí" else 0
    row_df["violencia_reciente"] = 1 if violencia_reciente == "Sí" else 0
    row_df["ansiedad_frec"] = ansiedad_frec
    row_df["deficit_cognitivo"] = 1 if deficit_cognitivo == "Sí" else 0
    row_df["desregulacion_conductual"] = 1 if desregulacion_conductual == "Sí" else 0
    row_df["aislamiento_social"] = 1 if aislamiento_social == "Sí" else 0
    row_df["violencia_familiar"] = 1 if violencia_familiar == "Sí" else 0
    row_df["sexo_Mujer"] = 1 if sexo == "Mujer" else 0

    # Asegurar el orden de las columnas coincidente con el modelo
    row_df = row_df[features]

    # Predicción del score de riesgo
    st.divider()
    st.markdown("### Resultado del análisis clínico de riesgo")

    probs = modelo.predict_proba(row_df)[0]  # [prob_0, prob_1]
    prob_riesgo = probs[1] * 100

    col_res1, col_res2 = st.columns([2, 3])

    with col_res1:
        if prob_riesgo < 30:
            st.success("Bajo riesgo estimado")
        elif prob_riesgo < 60:
            st.warning("Riesgo moderado estimado")
        else:
            st.error("Alto riesgo clínico detectado")

        st.metric(
            label="Probabilidad de conducta suicida estimada",
            value=f"{prob_riesgo:.1f}%",
        )
        st.caption(f"Clasificación generada por el algoritmo: **{info['mejor_modelo']}**")

    with col_res2:
        st.markdown("**Factores de riesgo críticos identificados en el perfil simulado:**")

        factores_activos = []
        if asi == "Sí":
            factores_activos.append(
                "- **Antecedente de abuso sexual infantil (ASI):** Factor de alta vulnerabilidad estructural a largo plazo."
            )
        if depresion_bin == 1:
            factores_activos.append(
                f"- **Depresión clínica activa (CES-D = {depresion_cesd}):** Requiere valoración por un terapeuta o psicólogo de inmediato."
            )
        if ansiedad_frec >= 4:
            factores_activos.append(
                f"- **Ansiedad muy frecuente o diaria (Nivel = {ansiedad_frec}):** Alta comorbilidad asociada a ideación autolítica."
            )
        if conducta_alim >= 25:
            factores_activos.append(
                f"- **Riesgo alimentario elevado (CES-D-7A = {conducta_alim}):** Posible presencia de trastorno de conducta alimentaria."
            )
        if violencia_reciente == "Sí":
            factores_activos.append(
                "- **Agresión o daño por violencia reciente (12 meses):** Factor gatillo ambiental agudo."
            )
        if violencia_familiar == "Sí":
            factores_activos.append(
                "- **Maltrato o disciplina violenta en el hogar (1 mes):** Entorno familiar hostil activo."
            )
        if binge_alcohol == "Sí":
            factores_activos.append(
                "- **Consumo excesivo de alcohol (Binge drinking):** Marcador de impulsividad extrema y desinhibición."
            )
        if drogas_ilegales == "Sí" or sustancias_medicas == "Sí":
            factores_activos.append(
                "- **Consumo de drogas o automedicación de controlados:** Abuso de sustancias ligado a conductas autolesivas."
            )
        if deficit_cognitivo == "Sí" or desregulacion_conductual == "Sí":
            factores_activos.append(
                "- **Dificultad de autorregulación o déficit cognitivo:** Menores recursos cognitivos de afrontamiento y resiliencia."
            )
        if aislamiento_social == "Sí":
            factores_activos.append(
                "- **Aislamiento social y desintegración de red de pares:** Falta de soporte de red social."
            )

        if factores_activos:
            for f in factores_activos:
                st.markdown(f)
        else:
            st.markdown(
                "*No se detectaron factores de riesgo críticos activos en el perfil simulado. (Perfil clínico de bajo riesgo).*"
            )

    st.divider()
    st.caption(
        "**Aviso legal de responsabilidad científica:** Esta herramienta fue diseñada con propósitos puramente académicos "
        "y de investigación en salud pública. No representa bajo ninguna circunstancia un diagnóstico psiquiátrico "
        "profesional. En caso de detectar riesgo real en un adolescente, se debe canalizar urgentemente a los servicios "
        "médicos de urgencia de salud mental."
    )
