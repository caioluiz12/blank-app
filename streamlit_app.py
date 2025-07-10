import streamlit as st
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import os
import re

# Configurar chave da API Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("models/gemini-1.5-flash")

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

# Função para gerar a análise usando Gemini
def gerar_analise_desinformacao(texto):
    prompt = (
        "Você é uma inteligência artificial especializada em checagem científica."
        " Receberá o texto de uma matéria sobre odontologia e deve avaliá-lo cientificamente."
        " Busque referências científicas confiáveis (como PubMed, Cochrane, etc.) para sustentar sua avaliação."
        " Retorne os seguintes itens:\n"
        "1. Um resumo técnico do conteúdo.\n"
        "2. Avaliação do risco de desinformação: 'Baixo risco', 'Potencial risco' ou 'Alto risco'.\n"
        "3. Justificativa com base científica (cite ao menos 1 a 2 fontes científicas reais com nome, ano, base e link acessível).\n"
        "\nTexto da matéria:\n"
        f"{texto}\n"
        "\nRetorne apenas os três itens solicitados, de forma objetiva."
    )
    try:
        resposta = model.generate_content(prompt)
        return resposta.text
    except Exception as e:
        return f"Erro ao gerar resposta do Gemini: {str(e)}"

# Função para extrair e separar os links ao final
def extrair_links(texto):
    url_regex = r"(https?://\S+)"
    links = re.findall(url_regex, texto)
    return links

# Função para transformar links em formato clicável no corpo do texto
def transformar_links_em_html(texto):
    url_regex = r"(https?://\S+)"
    return re.sub(url_regex, r'<a href="\1" target="_blank">\1</a>', texto)

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
            resultado = gerar_analise_desinformacao(texto_extraido)
            links_extraidos = extrair_links(resultado)
            resultado_com_links = transformar_links_em_html(resultado)

            st.markdown("### Resultado da Análise IA:")
            st.markdown(resultado_com_links, unsafe_allow_html=True)

            st.markdown("#### Referências Científicas Citadas:")
            if links_extraidos:
                for link in links_extraidos:
                    st.markdown(f"- [Acessar referência]({link})")
            else:
                st.markdown("_Nenhuma referência científica com link foi identificada pela IA._")

            st.markdown("---")
            opiniao = st.radio("Você concorda com essa avaliação da IA?", ["Sim", "Não", "Parcialmente"])
            st.markdown(f"**Sua resposta:** {opiniao}")

st.markdown("---")
st.markdown("Desenvolvido por Caio — Projeto FAPEMIG 🧠🔬")
