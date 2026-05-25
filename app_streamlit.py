import streamlit as st
from app import analyze_patient


st.set_page_config(
    page_title="Sistema Predictivo de Diabetes",
    page_icon="🩺",
    layout="centered"
)


st.title("Sistema Predictivo de Diabetes")
st.write("Modelo predictivo con sistema experto para análisis académico e investigativo.")


st.warning(
    "Este sistema no entrega diagnóstico médico definitivo. "
    "Su uso es únicamente académico, investigativo y de apoyo explicativo."
)


st.header("Datos del paciente")


gender = st.selectbox(
    "Género",
    options=["Male", "Female"]
)

age = st.number_input(
    "Edad",
    min_value=0,
    max_value=120,
    value=45,
    step=1
)

hypertension_text = st.selectbox(
    "¿Tiene hipertensión?",
    options=["No", "Sí"]
)

heart_disease_text = st.selectbox(
    "¿Tiene enfermedad cardíaca?",
    options=["No", "Sí"]
)

smoking_history = st.selectbox(
    "Historial de tabaquismo",
    options=["never", "former", "current", "not current", "ever", "No Info"]
)

bmi = st.number_input(
    "IMC / BMI",
    min_value=10.0,
    max_value=80.0,
    value=27.0,
    step=0.1
)

hba1c = st.number_input(
    "Nivel de HbA1c",
    min_value=3.0,
    max_value=15.0,
    value=5.8,
    step=0.1
)

blood_glucose = st.number_input(
    "Nivel de glucosa en sangre",
    min_value=50,
    max_value=400,
    value=120,
    step=1
)


hypertension = 1 if hypertension_text == "Sí" else 0
heart_disease = 1 if heart_disease_text == "Sí" else 0


patient_data = {
    "gender": gender,
    "age": age,
    "hypertension": hypertension,
    "heart_disease": heart_disease,
    "smoking_history": smoking_history,
    "bmi": bmi,
    "HbA1c_level": hba1c,
    "blood_glucose_level": blood_glucose
}


st.divider()


if st.button("Analizar paciente"):
    try:
        with st.spinner("Analizando paciente y generando explicación..."):
            result = analyze_patient(patient_data)

        prediction_result = result["prediction_result"]
        expert_result = result["expert_result"]

        st.subheader("Resultado del modelo predictivo")

        if prediction_result["prediction"] == 1:
            st.error(prediction_result["label"])
        else:
            st.success(prediction_result["label"])

        if result.get("explanation_source") == "LM Studio":
            st.info(
                "El porcentaje estimado se presenta dentro de la respuesta generada por LM Studio."
            )
        else:
            probability_percentage = prediction_result["probability"] * 100
            st.metric(
                label="Probabilidad estimada de diabetes",
                value=f"{probability_percentage:.2f}%"
            )

        st.subheader("Resultado del sistema experto")

        if expert_result["risk_level"] == "Alto":
            st.error(f"Nivel de riesgo: {expert_result['risk_level']}")
        elif expert_result["risk_level"] == "Medio":
            st.warning(f"Nivel de riesgo: {expert_result['risk_level']}")
        else:
            st.success(f"Nivel de riesgo: {expert_result['risk_level']}")

        st.write("Hechos identificados:")

        for fact in expert_result["facts"]:
            st.write(f"- {fact}")

        st.subheader("Explicación final")

        if result.get("explanation_source") == "LM Studio":
            st.success("Explicación generada con LM Studio")
        else:
            st.warning("Explicación generada con respaldo local")

        st.markdown(result["explanation"])

    except FileNotFoundError:
        st.error(
            "No se encontró el archivo modelo_diabetes.pkl. "
            "Primero debes ejecutar train_model.py para entrenar y guardar el modelo."
        )

    except Exception as e:
        st.error("Ocurrió un error durante el análisis.")
        st.exception(e)
