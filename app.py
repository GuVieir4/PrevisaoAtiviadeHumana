from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "modelo_final.joblib"


ACTIVITY_TEXT = {
    "LAYING": "A pessoa provavelmente está deitada.",
    "SITTING": "A pessoa provavelmente está sentada.",
    "STANDING": "A pessoa provavelmente está em pé.",
    "WALKING": "A pessoa provavelmente está caminhando.",
    "WALKING_DOWNSTAIRS": "A pessoa provavelmente está descendo escadas.",
    "WALKING_UPSTAIRS": "A pessoa provavelmente está subindo escadas.",
}


SIGNAL_MEANINGS = {
    "tBodyAcc": "aceleracao do corpo no dominio do tempo",
    "tGravityAcc": "aceleracao da gravidade no dominio do tempo",
    "tBodyAccJerk": "variacao da aceleracao do corpo no tempo",
    "tBodyGyro": "velocidade angular do giroscopio no dominio do tempo",
    "tBodyGyroJerk": "variacao da velocidade angular do giroscopio",
    "tBodyAccMag": "magnitude da aceleracao do corpo",
    "tGravityAccMag": "magnitude da aceleracao da gravidade",
    "tBodyAccJerkMag": "magnitude da variacao da aceleracao do corpo",
    "tBodyGyroMag": "magnitude da velocidade angular do giroscopio",
    "tBodyGyroJerkMag": "magnitude da variacao da velocidade angular",
    "fBodyAcc": "aceleracao do corpo no dominio da frequencia",
    "fBodyAccJerk": "variacao da aceleracao do corpo no dominio da frequencia",
    "fBodyGyro": "giroscopio no dominio da frequencia",
    "fBodyAccMag": "magnitude da aceleracao no dominio da frequencia",
    "fBodyBodyAccJerkMag": "magnitude da variacao da aceleracao no dominio da frequencia",
    "fBodyBodyGyroMag": "magnitude do giroscopio no dominio da frequencia",
    "fBodyBodyGyroJerkMag": "magnitude da variacao do giroscopio no dominio da frequencia",
    "angle": "angulo entre sinais do sensor e a gravidade",
}

STAT_MEANINGS = {
    "mean()": "media do sinal",
    "std()": "desvio padrao do sinal",
    "mad()": "desvio absoluto mediano",
    "max()": "valor maximo",
    "min()": "valor minimo",
    "sma()": "area/magnitude media do sinal",
    "energy()": "energia do sinal",
    "iqr()": "intervalo interquartil",
    "entropy()": "entropia do sinal",
    "arCoeff()": "coeficiente autorregressivo",
    "correlation()": "correlacao entre eixos",
    "maxInds": "indice da maior frequencia",
    "meanFreq()": "media ponderada das frequencias",
    "skewness()": "assimetria da distribuicao",
    "kurtosis()": "curtose da distribuicao",
    "bandsEnergy()": "energia em uma faixa de frequencia",
}

AXIS_MEANINGS = {
    "X": "eixo X do sensor",
    "Y": "eixo Y do sensor",
    "Z": "eixo Z do sensor",
}


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("Modelo nao encontrado. Rode primeiro: python train_model.py")
        st.stop()
    return joblib.load(MODEL_PATH)


def prepare_input(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    df = df.copy()

    for col in ["subject", "Activity"]:
        if col in df.columns:
            df = df.drop(columns=col)

    missing = [col for col in feature_names if col not in df.columns]
    if missing:
        raise ValueError(
            "O arquivo/entrada nao possui todas as colunas esperadas pelo modelo."
        )

    return df[feature_names].apply(pd.to_numeric, errors="coerce")


def explain_feature(feature: str) -> str:
    signal = next(
        (meaning for key, meaning in SIGNAL_MEANINGS.items() if feature.startswith(key)),
        "variavel numerica extraida dos sensores",
    )
    statistic = next(
        (meaning for key, meaning in STAT_MEANINGS.items() if key in feature),
        "medida calculada sobre o sinal",
    )

    axis = ""
    for suffix, meaning in AXIS_MEANINGS.items():
        if feature.endswith(f"-{suffix}") or feature.endswith(f",{suffix}"):
            axis = f" no {meaning}"
            break

    return f"{statistic} da {signal}{axis}."


st.set_page_config(page_title="Preditor de Atividade Humana", layout="wide")

artifact = load_model()
model = artifact["model"]
label_encoder = artifact["label_encoder"]
feature_names = artifact["feature_names"]
metrics = artifact["metrics"]

st.title("Preditor de Atividade Humana")
st.write(
    "Aplicacao simples em Streamlit para carregar o modelo salvo e prever a atividade "
    "com base nas variaveis do dataset."
)

left, right = st.columns(2)
left.metric("Modelo carregado", artifact["model_name"])
right.metric("F1 macro no teste", f"{metrics['f1_macro']:.4f}")

st.subheader("Entrada de dados")
input_mode = st.radio(
    "Escolha como preencher os dados:",
    ["Usar exemplo editavel", "Enviar CSV"],
    horizontal=True,
)

if input_mode == "Enviar CSV":
    uploaded_file = st.file_uploader("Envie um CSV com as mesmas colunas do dataset", type="csv")
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        st.dataframe(input_df.head(), use_container_width=True)
    else:
        input_df = None
else:
    st.caption("Edite os valores se quiser. A linha abaixo ja esta no formato esperado pelo modelo.")
    input_df = st.data_editor(
        artifact["sample_input"],
        use_container_width=True,
        num_rows="fixed",
        hide_index=True,
    )

with st.expander("Dicionario das variaveis"):
    st.write(
        "Os nomes das colunas seguem o padrao do dataset de sensores. "
        "Exemplo: `tBodyAcc-mean()-X` significa media da aceleracao do corpo "
        "no dominio do tempo, no eixo X."
    )

    glossary = pd.DataFrame(
        [
            ["t", "sinal no dominio do tempo"],
            ["f", "sinal no dominio da frequencia"],
            ["BodyAcc", "aceleracao do corpo"],
            ["GravityAcc", "aceleracao da gravidade"],
            ["Gyro", "giroscopio"],
            ["Jerk", "variacao do sinal ao longo do tempo"],
            ["Mag", "magnitude do sinal"],
            ["mean()", "media"],
            ["std()", "desvio padrao"],
            ["X, Y, Z", "eixos do sensor"],
        ],
        columns=["Termo", "Significado"],
    )
    st.dataframe(glossary, use_container_width=True, hide_index=True)

    feature_dictionary = pd.DataFrame(
        {
            "variavel": feature_names,
            "significado": [explain_feature(feature) for feature in feature_names],
        }
    )
    st.dataframe(feature_dictionary, use_container_width=True, hide_index=True)

if st.button("Executar predicao", type="primary"):
    if input_df is None or input_df.empty:
        st.warning("Preencha os dados ou envie um arquivo CSV antes de prever.")
    else:
        try:
            X_input = prepare_input(input_df, feature_names)
            encoded_predictions = model.predict(X_input)
            predictions = label_encoder.inverse_transform(encoded_predictions)

            result_df = pd.DataFrame(
                {
                    "linha": range(1, len(predictions) + 1),
                    "atividade_prevista": predictions,
                    "interpretacao": [
                        ACTIVITY_TEXT.get(pred, f"Atividade prevista: {pred}")
                        for pred in predictions
                    ],
                }
            )

            st.subheader("Resultado")
            st.success(f"Predicao concluida: {predictions[0]}")
            st.dataframe(result_df, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Erro ao executar a predicao: {exc}")

with st.expander("Informacoes do modelo"):
    st.write("Classes possiveis:")
    st.write(", ".join(artifact["classes"]))
    st.write(f"Quantidade de variaveis usadas: {len(feature_names)}")
    st.write("Formato salvo: `model/modelo_final.joblib`")
