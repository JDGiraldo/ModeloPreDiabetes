import joblib
import pandas as pd

from lmstudio_cliente import generar_explicacion_lmstudio


MODEL_PATH = "modelo_diabetes.pkl"


def load_model():
    # Cargar modelo entrenado
    model = joblib.load(MODEL_PATH)
    return model


def predict_diabetes(patient_data):
    """
    Recibe un diccionario con los datos del paciente.
    Retorna predicción, probabilidad y etiqueta.
    """

    model = load_model()

    patient_df = pd.DataFrame([patient_data])

    prediction = model.predict(patient_df)[0]
    probability = model.predict_proba(patient_df)[0][1]

    if prediction == 1:
        label = "Riesgo de diabetes"
    else:
        label = "Sin diabetes según el modelo"

    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "label": label
    }


def expert_system_rules(patient_data):
    """
    Sistema experto basado en reglas clínicas generales.
    No reemplaza valoración médica.
    """

    facts = []
    risk_level = "Bajo"

    age = patient_data["age"]
    hypertension = patient_data["hypertension"]
    heart_disease = patient_data["heart_disease"]
    bmi = patient_data["bmi"]
    hba1c = patient_data["HbA1c_level"]
    glucose = patient_data["blood_glucose_level"]
    smoking_history = patient_data["smoking_history"]

    # Regla por HbA1c
    if hba1c >= 6.5:
        facts.append("El nivel de HbA1c es igual o superior a 6.5%, valor compatible con criterio de diabetes.")
        risk_level = "Alto"
    elif 5.7 <= hba1c < 6.5:
        facts.append("El nivel de HbA1c se encuentra entre 5.7% y 6.4%, rango asociado a prediabetes.")
        if risk_level != "Alto":
            risk_level = "Medio"
    else:
        facts.append("El nivel de HbA1c está por debajo de 5.7%, lo cual no sugiere alteración glucémica por este criterio.")

    # Regla por glucosa
    if glucose >= 200:
        facts.append("El nivel de glucosa en sangre es igual o superior a 200 mg/dL, lo cual puede ser compatible con diabetes según el contexto clínico.")
        risk_level = "Alto"
    elif 140 <= glucose < 200:
        facts.append("El nivel de glucosa está entre 140 y 199 mg/dL, rango que puede sugerir alteración en la tolerancia a la glucosa.")
        if risk_level != "Alto":
            risk_level = "Medio"
    elif 100 <= glucose < 140:
        facts.append("El nivel de glucosa está entre 100 y 139 mg/dL, lo cual puede requerir seguimiento clínico.")
        if risk_level == "Bajo":
            risk_level = "Medio"
    else:
        facts.append("El nivel de glucosa está por debajo de 100 mg/dL.")

    # Regla por IMC
    if bmi >= 30:
        facts.append("El IMC es igual o superior a 30, valor asociado a obesidad y mayor riesgo metabólico.")
        if risk_level != "Alto":
            risk_level = "Medio"
    elif 25 <= bmi < 30:
        facts.append("El IMC está entre 25 y 29.9, rango asociado a sobrepeso.")
        if risk_level == "Bajo":
            risk_level = "Medio"
    else:
        facts.append("El IMC se encuentra por debajo de 25.")

    # Regla por edad
    if age >= 45:
        facts.append("La edad es igual o superior a 45 años, factor asociado a mayor riesgo de diabetes tipo 2.")
        if risk_level == "Bajo":
            risk_level = "Medio"

    # Regla por hipertensión
    if hypertension == 1:
        facts.append("El paciente registra hipertensión, condición relacionada con mayor riesgo cardiometabólico.")
        if risk_level == "Bajo":
            risk_level = "Medio"

    # Regla por enfermedad cardíaca
    if heart_disease == 1:
        facts.append("El paciente registra antecedente de enfermedad cardíaca, lo cual aumenta el riesgo clínico general.")
        if risk_level == "Bajo":
            risk_level = "Medio"

    # Regla por tabaquismo
    if smoking_history in ["current", "former", "ever"]:
        facts.append("El historial de tabaquismo puede aumentar el riesgo cardiometabólico.")
        if risk_level == "Bajo":
            risk_level = "Medio"

    return {
        "risk_level": risk_level,
        "facts": facts
    }


def generate_explanation(patient_data, prediction_result, expert_result):
    """
    Genera una explicación textual sin usar todavía LM Studio.
    Luego esta función se puede conectar con un LLM local.
    """

    probability_percentage = prediction_result["probability"] * 100

    explanation = f"""
Resultado del modelo predictivo:
{prediction_result["label"]}

Probabilidad estimada de diabetes:
{probability_percentage:.2f}%

Nivel de riesgo según sistema experto:
{expert_result["risk_level"]}

Análisis del sistema experto:
"""

    for fact in expert_result["facts"]:
        explanation += f"- {fact}\n"

    explanation += """

Recomendación general:
Este resultado debe interpretarse como una herramienta académica e investigativa de apoyo.
No representa un diagnóstico médico definitivo. Se recomienda valoración por personal de salud,
especialmente si existen valores elevados de HbA1c, glucosa en sangre, IMC alto u otros factores de riesgo.
"""

    return explanation


def analyze_patient(patient_data):
    """
    Función principal que une:
    - Modelo predictivo
    - Sistema experto
    - Explicación final
    """

    prediction_result = predict_diabetes(patient_data)
    expert_result = expert_system_rules(patient_data)

    try:
        explanation = generar_explicacion_lmstudio(
            patient_data=patient_data,
            prediction_result=prediction_result,
            expert_result=expert_result,
        )
        explanation_source = "LM Studio"
    except Exception as error:
        explanation = generate_explanation(patient_data, prediction_result, expert_result)
        explanation += (
            "\n\nNota tecnica:\n"
            "No fue posible generar la explicacion con LM Studio. "
            f"Se uso la explicacion local del sistema. Detalle: {error}"
        )
        explanation_source = "Local"

    return {
        "prediction_result": prediction_result,
        "expert_result": expert_result,
        "explanation": explanation,
        "explanation_source": explanation_source
    }


if __name__ == "__main__":
    sample_patient = {
        "gender": "Male",
        "age": 55,
        "hypertension": 1,
        "heart_disease": 0,
        "smoking_history": "former",
        "bmi": 31.5,
        "HbA1c_level": 6.8,
        "blood_glucose_level": 180
    }

    result = analyze_patient(sample_patient)

    print(result["explanation"])
