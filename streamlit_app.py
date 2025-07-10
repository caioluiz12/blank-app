import streamlit as st
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import os

# Configurar chave da API Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("models/gemini-pro-vision")

# Função para extrair texto de uma página web
def extrair_texto(url):
    try:
        resposta = requests.get(url, timeout=10)
        sopa = BeautifulSoup(resposta.text, "html.parser")
        paragrafos = sopa.find_all("p")
        texto = " ".join([p.get_text() for p in paragrafos])
        return texto
    except Exception as e:
        return f"Erro ao acessar o link: {str(e)}"

# Função para gerar o resumo usando Gemini
def gerar_resumo_gemini(texto):
    prompt = (
        "Você é uma IA especializada em checagem científica."
        " Analise o seguinte texto de uma matéria e gere um resumo técnico."
        " Em seguida, indique se há potencial de desinformação com base no conhecimento científico atual: \n\n"
        f"{texto}"
    )
    try:
        resposta = model.generate_content(prompt)
        return resposta.text
    except Exception as e:
        return f"Erro ao gerar resposta do Gemini: {str(e)}"

# Interface com Streamlit
st.title("Detector de Desinformação em Odontologia (via Gemini ✨)")
st.markdown("Cole abaixo o link da matéria que você deseja analisar:")

url = st.text_input("URL da matéria")

if url:
    with st.spinner("Extraindo texto e analisando com IA..."):
        texto_extraido = extrair_texto(url)
        if "Erro" in texto_extraido:
            st.error(texto_extraido)
        else:
            st.success("Texto extraído com sucesso! Agora analisando com Gemini...")
            resultado = gerar_resumo_gemini(texto_extraido)
            st.markdown("### Resultado da Análise IA:")
            st.write(resultado)

st.markdown("---")
st.markdown("Desenvolvido por Caio — Projeto FAPEMIG 🧠🔬")
