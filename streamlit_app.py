# ============================================================
# Detector de Desinformação Odonto - Streamlit App
# Autor: Caio Luiz Bitencourt Reis
# Descrição: Aplicativo para análise de risco de desinformação
# em conteúdos odontológicos, com integração à API Gemini.
# ============================================================

# --- IMPORTAÇÕES ---
import streamlit as st
import requests
import re
import os
from bs4 import BeautifulSoup
import google.generativeai as genai

# ⚙️ Configuração da página (PRECISA ser o primeiro comando Streamlit)
st.set_page_config(
    page_title="Detector de Desinformação Odonto",
    layout="centered",
    page_icon="🦷",
    initial_sidebar_state="auto"
)

# --- CONFIGURAÇÃO DO GEMINI ---
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

# --- TÍTULO E INTRODUÇÃO ---
st.title("🦷 Detector de Desinformação em Odontologia (via Gemini ✨)")
st.markdown("""
Este aplicativo tem como objetivo **identificar e classificar conteúdos com risco de desinformação**
em textos sobre Odontologia, com base em evidências científicas reais.
Cole um link ou texto abaixo para análise.
""")

# --- ESCOLHA DO TIPO DE ENTRADA ---
input_type = st.radio(
    "Selecione o tipo de entrada:",
    ["🔗 Link de notícia", "📝 Texto manual"]
)

user_input = ""
if input_type == "🔗 Link de notícia":
    user_input = st.text_input("Cole o link da publicação:")
else:
    user_input = st.text_area("Cole o texto a ser analisado:", height=200)

# --- FUNÇÕES ---

def extrair_texto(url):
    """Extrai o texto principal de uma página web."""
    try:
        resposta = requests.get(url, timeout=10)
        sopa = BeautifulSoup(resposta.text, "html.parser")
        paragrafos = sopa.find_all("p")
        texto = " ".join([p.get_text() for p in paragrafos])
        return texto
    except Exception as e:
        return f"Erro ao acessar o link: {str(e)}"

def gerar_analise_desinformacao(texto):
    """Gera a análise com o modelo Gemini."""
    prompt = (
        "Você é uma inteligência artificial especializada em checagem científica."
        " Receberá o texto de uma matéria sobre odontologia e deve avaliá-lo cientificamente."
        " Busque referências científicas confiáveis (como PubMed, Cochrane, etc.) para sustentar sua avaliação."
        " Para encontrar artigos científicos relevantes, extraia os principais termos do texto (em português), traduza para o inglês, "
        "e pesquise usando palavras-chave no estilo: 'substance name AND dental health', 'ingredient AND tooth whitening', ou 'abrasion AND enamel'."
        " Utilize o site https://pubmed.ncbi.nlm.nih.gov/ e, se possível, inclua links diretos para os estudos."
        " Mesmo que os artigos estejam atrás de paywall, forneça os títulos, autores, ano, base (ex: PubMed) e links."
        " Retorne os seguintes itens:\n"
        "1. Um resumo técnico do conteúdo.\n"
        "2. Avaliação do risco de desinformação: 'Baixo risco', 'Potencial risco' ou 'Alto risco'.\n"
        "3. Justificativa com base científica (cite pelo menos 1 a 2 fontes reais com links clicáveis).\n"
        "\nTexto da matéria:\n"
        f"{texto}\n"
        "\nRetorne apenas os três itens solicitados, de forma objetiva."
    )
    try:
        resposta = model.generate_content(prompt)
        return resposta.text
    except Exception as e:
        return f"Erro ao gerar resposta do Gemini: {str(e)}"

def extrair_links(texto):
    """Extrai links da resposta do modelo."""
    url_regex = r"(https?://\S+)"
    return re.findall(url_regex, texto)

def transformar_links_em_html(texto):
    """Transforma links em formato clicável."""
    url_regex = r"(https?://\S+)"
    return re.sub(url_regex, r'<a href="\1" target="_blank">\1</a>', texto)

def destacar_risco(resultado):
    """Adiciona destaque colorido conforme o nível de risco."""
    if "alto risco" in resultado.lower():
        cor = "#FF4B4B"
        risco = "🟥 Alto risco de desinformação"
    elif "potencial risco" in resultado.lower():
        cor = "#FFD700"
        risco = "🟨 Potencial risco de desinformação"
    else:
        cor = "#32CD32"
        risco = "🟩 Baixo risco de desinformação"
    return f'<div style="background-color:{cor};padding:10px;border-radius:8px;font-weight:bold">{risco}</div>'

# --- EXECUÇÃO DA ANÁLISE ---
if st.button("🔍 Analisar conteúdo"):
    if not user_input.strip():
        st.warning("Por favor, insira um link ou texto para análise.")
    else:
        with st.spinner("Analisando o conteúdo com IA..."):
            texto_extraido = user_input
            if input_type == "🔗 Link de notícia":
                texto_extraido = extrair_texto(user_input)

            if "Erro" in texto_extraido:
                st.error(texto_extraido)
            else:
                resultado = gerar_analise_desinformacao(texto_extraido)
                destaque_html = destacar_risco(resultado)
                links_extraidos = extrair_links(resultado)
                resultado_com_links = transformar_links_em_html(resultado)

                st.markdown("### Resultado da Análise IA:")
                st.markdown(destaque_html, unsafe_allow_html=True)
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
