import streamlit as st
import requests
from bs4 import BeautifulSoup
import random

st.set_page_config(page_title="DetectaOdonto - Combate à Desinformação", layout="centered")
st.title("🦷 DetectaOdonto – IA contra desinformação na odontologia")

# Função para extrair texto de uma URL
def extrair_texto(url):
    try:
        resposta = requests.get(url, timeout=10)
        resposta.raise_for_status()
        sopa = BeautifulSoup(resposta.text, "html.parser")
        paragrafos = sopa.find_all("p")
        texto = " ".join(p.text for p in paragrafos)
        return texto, None
    except Exception as e:
        return None, f"Erro ao extrair texto: {e}"

# Entrada de URL
url = st.text_input("📎 Cole aqui a URL de uma postagem ou matéria:")

if st.button("🔍 Analisar"):
    if not url:
        st.warning("Por favor, insira uma URL válida.")
    else:
        texto, erro = extrair_texto(url)

        if texto is None:
            st.error(erro)
        else:
            st.subheader("📝 Texto extraído (trecho):")
            st.write(texto[:800] + "...")

            # Detecção simples de claims com palavras-chave
            palavras_chave = ["cura", "substitui", "elimina", "resolve", "sem dor", "milagre", "definitivo"]
            claims = [frase.strip() for frase in texto.split(".") if any(p in frase.lower() for p in palavras_chave)]

            if not claims:
                st.info("🔎 Nenhum claim relevante detectado neste texto.")
            else:
                st.subheader("🔍 Claims detectados:")

                for i, claim in enumerate(claims):
                    st.markdown(f"**Claim {i+1}:** {claim}")

                    # Simula avaliação de risco pela IA
                    score = random.random()
                    if score > 0.6:
                        st.error("⚠️ Potencial desinformação")
                    else:
                        st.success("✅ Baixo risco de desinformação")

                    # Geração de explicação (simulada por enquanto)
                    st.caption("_Explicação automática será gerada aqui..._")

                    # Validação do especialista
                    val = st.radio(f"Você concorda com a avaliação do Claim {i+1}?", ("Sim", "Não"), key=f"validacao_{i+1}")
