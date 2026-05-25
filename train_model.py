import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


DATASET_PATH = "diabetes.csv"
MODEL_PATH = "modelo_diabetes.pkl"


def load_dataset(path):
    # Cargar dataset
    df = pd.read_csv(path)

    # Eliminar duplicados
    df = df.drop_duplicates()

    return df


def train_model():
    df = load_dataset(DATASET_PATH)

    print("Primeras filas del dataset:")
    print(df.head())

    print("\nTamaño del dataset:")
    print(df.shape)

    print("\nValores nulos:")
    print(df.isnull().sum())

    print("\nDistribución de la variable objetivo:")
    print(df["diabetes"].value_counts())

    # Variables predictoras y variable objetivo
    X = df.drop("diabetes", axis=1)
    y = df["diabetes"]

    # Columnas categóricas
    categorical_features = ["gender", "smoking_history"]

    # Columnas numéricas
    numeric_features = [
        "age",
        "hypertension",
        "heart_disease",
        "bmi",
        "HbA1c_level",
        "blood_glucose_level"
    ]

    # Preprocesador
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    # Modelo final
    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced",
        max_depth=None
    )

    # Pipeline completo: preprocesamiento + modelo
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # Separar entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Entrenar modelo
    pipeline.fit(X_train, y_train)

    # Predicciones
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Métricas
    print("\nMétricas del modelo:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))

    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=["No diabetes", "Diabetes"]))

    # Guardar modelo
    joblib.dump(pipeline, MODEL_PATH)

    print(f"\nModelo guardado correctamente en: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()