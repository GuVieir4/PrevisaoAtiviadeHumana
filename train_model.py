from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "modelo_final.joblib"


def load_dataset() -> pd.DataFrame:
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Coloque os arquivos train.csv e test.csv dentro da pasta data/."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return pd.concat([train_df, test_df], ignore_index=True)


def main() -> None:
    df = load_dataset()
    df_model = df.dropna(subset=["Activity"]).copy()

    feature_names = [col for col in df_model.columns if col not in ["subject", "Activity"]]
    X = df_model[feature_names]
    y = df_model["Activity"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        stratify=y_encoded,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold(threshold=0.01)),
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", C=10, gamma="scale", random_state=42)),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
    }

    final_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold(threshold=0.01)),
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", C=10, gamma="scale", random_state=42)),
        ]
    )
    final_pipeline.fit(X, y_encoded)

    artifact = {
        "model": final_pipeline,
        "label_encoder": label_encoder,
        "feature_names": feature_names,
        "classes": label_encoder.classes_.tolist(),
        "metrics": metrics,
        "sample_input": X.head(1),
        "model_name": "SVC-RBF Tunado",
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)

    print(f"Modelo salvo em: {MODEL_PATH}")
    print(f"Acuracia holdout: {metrics['accuracy']:.4f}")
    print(f"F1 macro holdout: {metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
