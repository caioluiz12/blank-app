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
    PASSO 1: Identifique a(s) especialidade(s) central(is) da matéria (Pode ser mais de uma).
    PASSO 2: Identifique a CONDIÇÃO CLÍNICA ou ANATOMIA foco (Ex: Doença Periodontal, Cárie, etc.), separando-a da "intervenção" (Ex: Dieta, Óleo essencial).
    PASSO 3: Se a "intervenção" da matéria NÃO constar nos catálogos oficiais para aquela "condição clínica", a alegação é NÃO COMPROVADA.

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
    1. PROIBIDO usar conhecimento prévio para validar tratamentos. Só valide se estiver nos links.
    2. No Item 5, MESMO QUE a intervenção da matéria não exista nas diretrizes, você OBRIGATORIAMENTE deve nomear o documento mestre que rege a CONDIÇÃO CLÍNICA identificada.

    FORMATO DE RESPOSTA:

    ### 🛡️ PARECER DE AUDITORIA INTERNA
    - Especialidade(s) identificada(s): (Liste uma ou mais)
    - Condição Clínica Foco: (Qual é a doença ou estrutura anatômica em jogo?)
    - Link(s) da Biblioteca utilizado(s):
    - A intervenção da matéria consta nas diretrizes dessa condição? (Sim/Não)

    ### 1. Resumo Técnico
    (Resumo objetivo da matéria)

    ### 2. Avaliação de Risco
    (🟨 Potencial Risco | 🟥 Alto Risco | 🟩 Baixo Risco)
    
    ### 3. Justificativa de Auditoria
    (Confronte a matéria com a biblioteca. Ex: "A matéria propõe X para a condição Y. No entanto, as diretrizes da especialidade Z não reconhecem X como padrão...")

    ### 4. Padrão-Ouro (Conduta Clínica Oficial)
    (Descreva o tratamento padrão para a Condição Clínica Foco, segundo a especialidade.)

  ### 5. Referências e Diretrizes Oficiais
    - 📂 **Catálogo(s) Oficial(is):** [Liste o(s) link(s) da(s) especialidade(s) selecionada(s) na Biblioteca]
    
    - 📄 **Documento Padrão-Ouro Identificado:** (Você OBRIGATORIAMENTE deve identificar e escrever o NOME REAL, EXATO e OFICIAL do documento pilar, diretriz ou 'Position Paper' da associação que dita o tratamento da Condição Clínica Foco. Ex: Se o foco for Cárie na AAPD, o documento real é 'Clinical Practice Guideline on Fluoride Therapy'. Se for Doença Periodontal na AAP, é 'Clinical Practice Guidelines for the Treatment of Stage I–III Periodontitis').
    
    - 🔗 **Link Direto de Verificação:** [Acessar documento: NOME_DO_DOCUMENTO_IDENTIFICADO](https://www.google.com/search?q=NOME_DO_DOCUMENTO_IDENTIFICADO+SIGLA_DA_ASSOCIAÇÃO)
    
    *(Diretriz de Formatação: No link acima, substitua os espaços do nome do documento por '+'. Exemplo real: se o documento for 'Clinical Practice Guideline on Fluoride Therapy' da AAPD, o link deve ser exatamente: [Acessar documento: Clinical Practice Guideline on Fluoride Therapy](https://www.google.com/search?q=Clinical+Practice+Guideline+on+Fluoride+Therapy+AAPD) )*

    - 🔗 **Links Específicos Coletados:** (Se o 'Arsenal de Evidências' trouxe algum link de artigo específico da matéria, liste aqui. Caso contrário, informe: "Nenhum link externo validou a intervenção alternativa da matéria.")
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
