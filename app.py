import os
import re

import joblib
import numpy as np
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Model", "melhor_modelo_spam.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "Model", "scaler_spam.pkl")

FEATURES_PALAVRAS = [
    "make", "address", "all", "3d", "our", "over", "remove", "internet", "order", "mail", "receive",
    "will", "people", "report", "addresses", "free", "business", "email", "you", "credit", "your",
    "font", "000", "money", "hp", "hpl", "george", "650", "lab", "labs", "telnet", "857", "data",
    "415", "85", "technology", "1999", "parts", "pm", "direct", "cs", "meeting", "original", "project",
    "re", "edu", "table", "conference", ";", "(", "[", "!", "$", "#",
]


def extrair_caracteristicas(texto):
    texto_lower = texto.lower()
    vetor = []
    palavras = re.findall(r"\w+", texto_lower)
    palavras_totais = len(palavras) if palavras else 1

    for feat in FEATURES_PALAVRAS:
        if feat in [";", "(", "[", "!", "$", "#"]:
            qtd = texto_lower.count(feat)
            porcentagem = (qtd / palavras_totais) * 100
        else:
            qtd = len(re.findall(r"\b" + feat + r"\b", texto_lower))
            porcentagem = (qtd / palavras_totais) * 100
        vetor.append(porcentagem)

    while len(vetor) < 57:
        vetor.append(0.0)

    return np.array(vetor).reshape(1, -1)


@st.cache_resource
def carregar_modelos():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None
    modelo = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return modelo, scaler


st.set_page_config(page_title="Detector Anti-Spam", page_icon="📧", layout="centered")

st.title("Detector Anti-Spam")
st.caption("Classificador de e-mails baseado em machine learning")

modelo, scaler = carregar_modelos()
if modelo is None or scaler is None:
    st.error(
        "Arquivos do modelo não encontrados. "
        "Adicione `melhor_modelo_spam.pkl` e `scaler_spam.pkl` na pasta `Model/`."
    )
    st.stop()

texto_email = st.text_area(
    "Cole o texto do e-mail aqui:",
    height=150,
    placeholder="Cole o texto do e-mail aqui...",
)

arquivo = st.file_uploader("Ou carregue um arquivo .txt", type=["txt"])
if arquivo is not None:
    texto_email = arquivo.read().decode("utf-8")
    st.text_area("Conteúdo do arquivo:", value=texto_email, height=150, disabled=True)

if st.button("Verificar e-mail", type="primary", use_container_width=True):
    if not texto_email.strip():
        st.warning("Digite ou carregue um texto antes de verificar.")
    else:
        with st.spinner("Analisando com o modelo..."):
            features = extrair_caracteristicas(texto_email)
            features_scaled = scaler.transform(features)
            predicao = modelo.predict(features_scaled)[0]
            probabilidade = modelo.predict_proba(features_scaled)[0][1] * 100

        if predicao == 1:
            st.error(f"🚨 É SPAM! (Certeza: {probabilidade:.1f}%)")
        else:
            st.success(f"✅ E-mail confiável (Ham)! (Chance de spam: {probabilidade:.1f}%)")
