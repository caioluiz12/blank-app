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
    
    # Definimos o modelo fixo agora que sabemos que ele funciona
    # O 2.5-flash é excelente para o seu projeto FAPEMIG
    model = genai.GenerativeModel("gemini-2.5-flash")
    
except Exception as e:
    st.error(f"Erro na conexão com a IA. Por favor, verifique as chaves de API.")

GOOGLE_SEARCH_API_KEY = st.secrets["GOOGLE_SEARCH_API_KEY"]
SEARCH_ENGINE_ID = "c49cbaece0d6a4c06"

# CONFIGURAÇÃO DE PRECISÃO: 
    # temperature=0.0 torna a resposta determinística (sempre igual)
    # top_p=0.95 garante que ela escolha as palavras mais prováveis tecnicamente
    config_geracao = {
        "temperature": 0.0,
        "top_p": 0.95,
        "top_k": 0,
        "max_output_tokens": 2048,
    }

    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        generation_config=config_geracao
    )
    
except Exception as e:
    st.error(f"Erro na conexão com a IA.")

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
    
    CRITÉRIO DE JULGAMENTO OBRIGATÓRIO:
    1. Se as 'Evidências Confiáveis' não contiverem uma diretriz específica sobre o tema, você DEVE classificar como 🟨 POTENCIAL RISCO. 
    2. Nunca classifique como 'Baixo Risco' apenas porque a fonte é uma universidade ou jornal. Sem diretriz clínica = Risco de interpretação equivocada.
    3. O foco do risco é: "O paciente pode substituir o tratamento padrão por essa nova informação?"

    ARSENAL DE EVIDÊNCIAS COLETADAS:
    {contexto_cientifico}

    TEXTO DA MATÉRIA:
    {texto_materia}

    FORMATO DE RESPOSTA:

    ### 1. Resumo Técnico
    (Resumo objetivo)

    ### 2. Avaliação de Risco
    (Aplique a regra rígida: Sem diretriz = 🟨 Potencial Risco)
    
    ### 3. Justificativa de Auditoria
    (Explique que, embora a fonte seja reputável, a conduta ainda não é consenso clínico nas associações de classe e não deve substituir o tratamento padrão.)

    ### 4. Padrão-Ouro (Conduta Clínica Oficial)
    (Descreva o que a AAP/AAE/ADA recomenda como eficaz para este problema gengival/dentário. Foque no controle mecânico do biofilme e visitas regulares.)

    ### 5. Referências e Links Oficiais
    (Mesmo que o tema específico não tenha sido encontrado, você DEVE listar os links das associações para as diretrizes de saúde periodontal/geral. 
    Exemplos: 
    - American Academy of Periodontology (perio.org)
    - American Dental Association (ada.org)
    - American Association of Endodontists (aae.org))
    """
    try:
        # Mantendo temperature 0 para consistência total
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
