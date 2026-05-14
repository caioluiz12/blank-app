# ============================================================
# Detector de Desinformação Odonto - Streamlit App
# Autor: Caio Luiz Bitencourt Reis
# Descrição: Aplicativo para análise de risco de desinformação
# em conteúdos odontológicos, com integração à API Gemini e 
# Google Custom Search (RAG).
# ============================================================

# --- IMPORTAÇÕES ---
import time # <- Garanta que isso está na primeira linha do seu arquivo junto com os outros imports
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

# --- CONFIGURAÇÃO DAS APIs ---
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    
    # Esta parte abaixo DEVE estar alinhada (4 espaços para a direita)
    config_geracao = {
        "temperature": 0.0,
        "top_p": 0.95,
        "top_k": 0,
        "max_output_tokens": 8192,
    }

    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        generation_config=config_geracao
    )
    
except Exception as e:
    st.error(f"Erro na conexão com a IA: {e}")

GOOGLE_SEARCH_API_KEY = st.secrets["GOOGLE_SEARCH_API_KEY"]
SEARCH_ENGINE_ID = "c49cbaece0d6a4c06"

# --- TÍTULO E INTRODUÇÃO ---
st.title("🦷 Detector de Desinformação em Odontologia (via Gemini ✨)")
st.markdown("""
Este aplicativo tem como objetivo **identificar e classificar conteúdos com risco de desinformação**
em textos sobre Odontologia, com base EXCLUSIVA em diretrizes oficiais (AAE, AAP, AAOMS, AAPD, etc.).
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

# --- FUNÇÕES CORE ---

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

def obter_palavras_chave(texto):
    """Usa o Gemini para extrair palavras-chave em inglês focadas em odontologia."""
    prompt = f"""
    Extraia de 3 a 4 palavras-chave em inglês que representem o tema clínico principal deste texto odontológico.
    Retorne APENAS as palavras-chave separadas por espaço (ex: fluoride tooth decay prevention).
    Texto: {texto[:2000]}
    """
    try:
        return model.generate_content(prompt).text.strip()
    except:
        return "dental health dentistry" # Fallback em caso de erro

def buscar_referencias_confiaveis(palavras_chave):
    """Busca evidências usando o Google Custom Search configurado."""
    if not GOOGLE_SEARCH_API_KEY:
        return "ERRO: Chave da API do Google (GOOGLE_SEARCH_API_KEY) não configurada."
        
    url = f"https://www.googleapis.com/customsearch/v1?q={palavras_chave}&key={GOOGLE_SEARCH_API_KEY}&cx={SEARCH_ENGINE_ID}"
    
    try:
        resposta = requests.get(url)
        dados = resposta.json()
        
        if "items" not in dados:
            return "Nenhuma evidência encontrada no arsenal de associações para este tema."
            
        referencias = ""
        # Pega os 4 primeiros resultados retornados pelo buscador restrito
        for item in dados["items"][:4]:
            titulo = item.get("title")
            snippet = item.get("snippet")
            link = item.get("link")
            referencias += f"- TÍTULO: {titulo}\n- RESUMO: {snippet}\n- LINK OFICIAL: {link}\n\n"
            
        return referencias
    except Exception as e:
        return f"Erro na busca: {str(e)}"

def gerar_analise_desinformacao(texto_materia, contexto_cientifico):
    prompt = f"""
    VOCÊ É UM AUDITOR CIENTÍFICO DE ODONTOLOGIA EXTREMAMENTE RÍGIDO.
    
    ARSENAL DE EVIDÊNCIAS COLETADAS (Pesquisa em tempo real):
    {contexto_cientifico}

    BIBLIOTECA DE DIRETRIZES GERAIS (Escolha APENAS a que corresponde ao tema da matéria):
    - [Higiene Geral e Cárie] American Dental Association (ADA): https://www.ada.org/resources/research/science-and-research-institute/oral-health-topics/
    - [Periodontia e Gengiva] American Academy of Periodontology (AAP): https://www.perio.org/resource/periodontal-disease-clinical-practice-guidelines/
    - [Endodontia e Canal] American Association of Endodontists (AAE): https://www.aae.org/specialty/clinical-resources/
    - [Odontopediatria] American Academy of Pediatric Dentistry (AAPD): https://www.aapd.org/research/oral-health-policies--recommendations/
    - [Cirurgia e Implantes] American Association of Oral and Maxillofacial Surgeons (AAOMS): https://www.aaoms.org/practice-resources/

    TEXTO DA MATÉRIA:
    {texto_materia}

    FORMATO DE RESPOSTA OBRIGATÓRIO:

    ### 1. Resumo Técnico
    (Resumo objetivo)

    ### 2. Avaliação de Risco
    (🟨 Potencial Risco | 🟥 Alto Risco | 🟩 Baixo Risco)
    
    ### 3. Justificativa de Auditoria
    (Explique o confronto entre a notícia e as diretrizes.)

    ### 4. Padrão-Ouro (Conduta Clínica Oficial)
    Aqui você deve descrever o que é eficaz para o problema citado.
    - Selecione mentalmente a especialidade correta na 'Biblioteca de Diretrizes Gerais' acima.
    - Descreva o tratamento/conduta padrão. Ex: Se a matéria for sobre canal, descreva o padrão-ouro endodôntico e cite a [Fonte: AAE]. Se for cárie, cite a [Fonte: ADA].

    ### 5. Referências e Links Oficiais
    - Liste APENAS o link da 'Biblioteca de Diretrizes Gerais' que corresponde à especialidade da matéria.
    - Além disso, liste os links encontrados no 'Arsenal de Evidências Coletadas' (se a pesquisa não vier vazia).
    """
    try:
        resposta = model.generate_content(prompt)
        return resposta.text
    except Exception as e:
        return f"Erro ao gerar resposta: {str(e)}"
        
# --- FUNÇÕES DE INTERFACE ---

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
    return f'<div style="background-color:{cor};padding:10px;border-radius:8px;font-weight:bold;margin-bottom:15px">{risco}</div>'



# --- EXECUÇÃO DA ANÁLISE ---
if st.button("🔍 Analisar conteúdo"):
    if not user_input.strip():
        st.warning("Por favor, insira um link ou texto para análise.")
    else:
        with st.spinner("1/3 - Extraindo texto da fonte..."):
            texto_extraido = user_input
            if input_type == "🔗 Link de notícia":
                texto_extraido = extrair_texto(user_input)

        if "Erro" in texto_extraido:
            st.error(texto_extraido)
        else:
            # Reduzimos o texto para o Gemini não achar que é abuso de volume
            texto_curto = texto_extraido[:2000] 

            # Passo 1: Busca no Google (Isso não gasta cota do Gemini)
            with st.spinner("2/3 - Consultando diretrizes odontológicas oficiais..."):
                # Usamos os primeiros 100 caracteres como busca simples para evitar 1 chamada de IA
                busca_simples = texto_curto[:100]
                evidencias = buscar_referencias_confiaveis(busca_simples)
                
            # Passo 2: O Grande Segredo - Pausa longa para resetar o limite por minuto
            with st.spinner("Aguardando estabilização da API (15 segundos)..."):
                time.sleep(15) 
                
            with st.spinner("3/3 - IA avaliando evidências..."):
                try:
                    resultado = gerar_analise_desinformacao(texto_curto, evidencias)
                    
                    if "429" in resultado or "quota" in resultado.lower():
                        st.error("O Google ainda está limitando as requisições. Aguarde 1 minuto e tente novamente.")
                    else:
                        destaque_html = destacar_risco(resultado)
                        resultado_com_links = transformar_links_em_html(resultado)

                        st.markdown("### Resultado da Análise IA:")
                        st.markdown(destaque_html, unsafe_allow_html=True)
                        st.markdown(resultado_com_links, unsafe_allow_html=True)

                        with st.expander("Ver Referências Encontradas"):
                            st.text(evidencias)
                except Exception as e:
                    st.error(f"Erro inesperado: {e}")

st.markdown("---")
st.markdown("Desenvolvido por Caio — Projeto FAPEMIG 🧠🔬")
