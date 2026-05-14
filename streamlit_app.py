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
    VOCÊ ESTÁ EM MODO DE AUDITORIA EXTREMA (ZERO TOLERÂNCIA).
    Sua única fonte de verdade é a 'BIBLIOTECA DE DIRETRIZES OFICIAIS' e o 'ARSENAL DE EVIDÊNCIAS'.

    Siga este protocolo de raciocínio antes de responder:
    PASSO 1: Identifique a especialidade central da matéria.
    PASSO 2: Verifique se há uma diretriz específica no 'Arsenal' ou na 'Biblioteca' para a alegação da matéria.
    PASSO 3: Se a alegação NÃO estiver explicitamente validada nos links fornecidos, ela é automaticamente considerada NÃO COMPROVADA, independentemente da fonte da notícia.

    BIBLIOTECA DE DIRETRIZES OFICIAIS (FONTE ÚNICA):
    - [Endodontia]: https://www.aae.org/specialty/clinical-resources/guidelines-position-statements/
    - [Periodontia, Implantodontia]: https://aap.onlinelibrary.wiley.com/doi/toc/10.1002/19433670.aap-clin-sci-papers?page=1
    - [Cirurgia]: https://aaoms.org/publications/position-papers/clinical-papers/
    - [Odontopediatria]: https://www.aapd.org/research/oral-health-policies--recommendations/
    - [Ortodontia]: https://www2.aaoinfo.org/advocacy/advocacy-efforts/orthofacts/
    - [Reabilitação]: https://www.theaapd.org/research_awards/research/research_committee_publications/

    ARSENAL DE EVIDÊNCIAS COLETADAS:
    {contexto_cientifico}

    TEXTO DA MATÉRIA:
    {texto_materia}

    REGRAS INVIOLÁVEIS:
    1. PROIBIDO usar conhecimento prévio sobre estudos universitários se eles não forem diretrizes de associações.
    2. Se a biblioteca não confirmar a eficácia, o risco é 🟨 POTENCIAL RISCO ou 🟥 ALTO RISCO. Nunca Verde.
    3. O 'Padrão-Ouro' deve ser extraído apenas do link da especialidade correspondente.

    FORMATO DE RESPOSTA:

    ### 🛡️ PARECER DE AUDITORIA INTERNA
    - Especialidade identificada: 
    - Link da Biblioteca utilizado:
    - A alegação da matéria consta no link? (Sim/Não)

    ### 1. Resumo Técnico
    (Resumo objetivo)

    ### 2. Avaliação de Risco
    (🟨 Potencial Risco | 🟥 Alto Risco | 🟩 Baixo Risco)
    
    ### 3. Justificativa de Auditoria
    (Confronte: "A matéria diz X, mas a biblioteca oficial na especialidade Y não lista X como procedimento padrão".)

    ### 4. Padrão-Ouro (Conduta Clínica Oficial)
    (Descreva apenas o que está no catálogo da associação escolhida.)

    ### 5. Referências e Diretrizes Oficiais
    - 📂 **Catálogo Oficial da Especialidade:** [Insira o link da biblioteca correspondente]
    - 📄 **Documento Padrão-Ouro:** (Escreva o NOME EXATO da diretriz, manual ou 'Position Paper' oficial da associação que o usuário deve buscar dentro do catálogo acima para confirmar a conduta. Ex: "Parameters of Care", "Clinical Practice Guideline on...").
    - 🔗 **Links Específicos Coletados:** (Se o 'Arsenal de Evidências' retornou algum link real que confirme a conduta, liste-o aqui. Caso o arsenal esteja vazio sobre o padrão-ouro, escreva: "A busca atual não retornou links diretos; consulte o documento nomeado acima no catálogo oficial.")
    """
    try:
        # Temperature 0.0 é o que garante que ela não "fuja" da biblioteca
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
