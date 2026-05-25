import os
import re

import requests


LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "phi-3-mini-4k-instruct")


VARIABLE_LABELS = {
    "gender": "Genero",
    "age": "Edad",
    "hypertension": "Hipertension",
    "heart_disease": "Enfermedad cardiaca",
    "smoking_history": "Historial de tabaquismo",
    "bmi": "IMC/BMI",
    "HbA1c_level": "HbA1c",
    "blood_glucose_level": "Glucosa en sangre",
}


def _valor_si_no(value):
    return "Si" if value == 1 else "No"


def crear_bloque_variables(patient_data, prediction_result, expert_result):
    probability_percentage = prediction_result["probability"] * 100

    variables = {
        "gender": patient_data["gender"],
        "age": patient_data["age"],
        "hypertension": _valor_si_no(patient_data["hypertension"]),
        "heart_disease": _valor_si_no(patient_data["heart_disease"]),
        "smoking_history": patient_data["smoking_history"],
        "bmi": patient_data["bmi"],
        "HbA1c_level": patient_data["HbA1c_level"],
        "blood_glucose_level": patient_data["blood_glucose_level"],
    }

    lines = ["## Variables analizadas", "", "Variables de entrada del paciente:"]
    for key, value in variables.items():
        lines.append(f"- {VARIABLE_LABELS[key]}: {value}")

    lines.extend(
        [
            "",
            "Variables calculadas por el sistema:",
            f"- Prediccion numerica del modelo: {prediction_result['prediction']}",
            f"- Etiqueta del modelo: {prediction_result['label']}",
            f"- Probabilidad estimada de diabetes: {probability_percentage:.2f}%",
            f"- Nivel de riesgo del sistema experto: {expert_result['risk_level']}",
        ]
    )

    return "\n".join(lines)


def crear_prompt_explicacion(patient_data, prediction_result, expert_result):
    probability_percentage = prediction_result["probability"] * 100
    facts_text = "\n".join(f"- {fact}" for fact in expert_result["facts"])
    variables_text = crear_bloque_variables(
        patient_data=patient_data,
        prediction_result=prediction_result,
        expert_result=expert_result,
    )

    return f"""
Eres un asistente academico especializado en sistemas expertos y modelos predictivos
aplicados al analisis de riesgo de diabetes. Tu tarea es redactar una explicacion clara,
ordenada y responsable para un proyecto academico.

Contexto del sistema:
- Existe un modelo de machine learning entrenado para estimar riesgo de diabetes.
- Existe un sistema experto basado en reglas clinicas generales.
- La respuesta debe integrar ambos enfoques sin afirmar diagnosticos medicos definitivos.

Bloque obligatorio de variables exactas:
<VARIABLES_EXACTAS>
{variables_text}
</VARIABLES_EXACTAS>

Resultado del modelo predictivo:
- Etiqueta: {prediction_result["label"]}
- Probabilidad estimada de diabetes: {probability_percentage:.2f}%

Resultado del sistema experto:
- Nivel de riesgo: {expert_result["risk_level"]}
- Hechos identificados:
{facts_text}

Instrucciones de respuesta:
1. Escribe en espanol.
2. Usa un tono academico, claro y comprensible.
3. Inicia copiando literalmente el contenido dentro de <VARIABLES_EXACTAS>, sin traducir, resumir ni cambiar valores.
4. Explica como se relacionan los datos del paciente con el resultado del modelo.
5. Explica el aporte del sistema experto y sus reglas.
6. Si hay discrepancia entre el modelo y las reglas, describela con prudencia.
7. No digas que el paciente "tiene diabetes", "padece diabetes" ni uses frases equivalentes como diagnostico definitivo.
8. Incluye una recomendacion general de seguimiento con personal de salud.
9. No inventes datos que no esten en la entrada.
10. Conserva los valores exactos de las variables. No traduzcas valores codificados como gender o smoking_history.
11. Usa "riesgo estimado", "probabilidad estimada" o "posible riesgo"; evita afirmaciones diagnosticas.
12. No agregues texto fuera de las secciones solicitadas.
13. La respuesta debe tener estas secciones:
   - Variables analizadas
   - Interpretacion del modelo predictivo
   - Interpretacion del sistema experto
   - Integracion de resultados
   - Recomendacion general
""".strip()


def limpiar_respuesta_lmstudio(content):
    removable_lines = {
        "<VARIABLES_EXACTAS>",
        "</VARIABLES_EXACTAS>",
    }
    lines = []
    for line in content.splitlines():
        stripped_line = line.strip()
        if stripped_line in removable_lines:
            continue

        section_match = re.fullmatch(r"<([^<>]+)>", stripped_line)
        if section_match:
            lines.append(f"## {section_match.group(1)}")
            continue

        lines.append(line)

    return "\n".join(lines).strip()


def generar_explicacion_lmstudio(patient_data, prediction_result, expert_result):
    prompt = crear_prompt_explicacion(
        patient_data=patient_data,
        prediction_result=prediction_result,
        expert_result=expert_result,
    )

    url = f"{LM_STUDIO_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {LM_STUDIO_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LM_STUDIO_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Eres un asistente academico para explicar resultados de un "
                    "modelo predictivo y un sistema experto de riesgo de diabetes. "
                    "No reemplazas valoracion medica."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.1,
        "max_tokens": 900,
        "stop": ["ranking_algorithm", "ranking_modelo"],
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()

    data = response.json()
    content = data["choices"][0]["message"]["content"].strip()
    return limpiar_respuesta_lmstudio(content)


if __name__ == "__main__":
    sample_patient = {
        "gender": "Male",
        "age": 55,
        "hypertension": 1,
        "heart_disease": 0,
        "smoking_history": "former",
        "bmi": 31.5,
        "HbA1c_level": 6.8,
        "blood_glucose_level": 180,
    }

    from app import analyze_patient

    result = analyze_patient(sample_patient)
    explanation = generar_explicacion_lmstudio(
        patient_data=sample_patient,
        prediction_result=result["prediction_result"],
        expert_result=result["expert_result"],
    )

    print(explanation)
