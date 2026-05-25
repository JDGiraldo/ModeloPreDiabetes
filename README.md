# Sistema Predictivo de Diabetes con Sistema Experto y LM Studio

Proyecto academico que combina un modelo de machine learning, un sistema experto basado en reglas y una explicacion generada con LM Studio para analizar el riesgo estimado de diabetes a partir de variables clinicas generales.

> Este proyecto no entrega diagnosticos medicos definitivos. Su uso es academico, investigativo y de apoyo explicativo.

## Caracteristicas

- Entrenamiento de un modelo predictivo con `scikit-learn`.
- Uso de un pipeline con preprocesamiento de variables numericas y categoricas.
- Sistema experto con reglas sobre HbA1c, glucosa, IMC, edad, hipertension, enfermedad cardiaca y tabaquismo.
- Interfaz web con Streamlit.
- Integracion con LM Studio para generar una explicacion clara de los resultados.
- Respaldo local si LM Studio no esta disponible.

## Estructura del proyecto

```text
.
|-- app.py                 # Logica principal: modelo, reglas y explicacion final
|-- app_streamlit.py       # Interfaz web en Streamlit
|-- lmstudio_cliente.py    # Cliente para consumir la API local de LM Studio
|-- train_model.py         # Entrenamiento y guardado del modelo
|-- requirements.txt       # Dependencias del proyecto
|-- diabetes.csv           # Dataset usado para entrenar
`-- modelo_diabetes.pkl    # Modelo entrenado generado por train_model.py
```

## Requisitos

- Python 3.11 o superior.
- LM Studio instalado, si se desea usar explicacion con LLM local.
- Un modelo cargado en LM Studio compatible con chat completions.

## Instalacion

Crear y activar un entorno virtual:

```powershell
python -m venv venv
.\venv\Scripts\activate
```

Instalar dependencias:

```powershell
pip install -r requirements.txt
```

## Entrenar el modelo

Ejecutar:

```powershell
python train_model.py
```

Esto genera el archivo:

```text
modelo_diabetes.pkl
```

## Ejecutar la aplicacion

Con Streamlit:

```powershell
streamlit run app_streamlit.py
```

Luego abrir la URL local que muestra Streamlit en la terminal.

## Configuracion de LM Studio

1. Abrir LM Studio.
2. Cargar un modelo de lenguaje.
3. Activar el servidor local compatible con OpenAI.
4. Verificar que este disponible en:

```text
http://127.0.0.1:1234/v1
```

Por defecto, el proyecto usa:

```text
LM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1
LM_STUDIO_API_KEY=lm-studio
LM_STUDIO_MODEL=phi-3-mini-4k-instruct
```

Si se quiere usar otro modelo, se puede configurar la variable de entorno:

```powershell
$env:LM_STUDIO_MODEL="nombre-del-modelo"
streamlit run app_streamlit.py
```

## Variables usadas por el sistema

El sistema analiza las siguientes variables:

- `gender`
- `age`
- `hypertension`
- `heart_disease`
- `smoking_history`
- `bmi`
- `HbA1c_level`
- `blood_glucose_level`

El modelo predictivo calcula:

- Prediccion numerica.
- Etiqueta del resultado.
- Probabilidad estimada de diabetes.

El sistema experto calcula:

- Nivel de riesgo: `Bajo`, `Medio` o `Alto`.
- Hechos identificados por reglas.

LM Studio recibe estos datos y genera una explicacion en lenguaje natural.

## Flujo de funcionamiento

1. El usuario ingresa los datos del paciente en Streamlit.
2. El modelo entrenado calcula la prediccion y la probabilidad.
3. El sistema experto evalua reglas clinicas generales.
4. LM Studio genera la explicacion final con las variables y resultados.
5. Si LM Studio falla, se muestra una explicacion local de respaldo.

## Uso responsable

Este sistema es una herramienta academica. No debe usarse como reemplazo de una evaluacion medica profesional. Los resultados deben interpretarse como apoyo explicativo y no como diagnostico definitivo.


