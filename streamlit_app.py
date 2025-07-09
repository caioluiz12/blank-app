import streamlit as st
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from googletrans import Translator
import numpy as np

st.set_page_config(page_title="DetectaOdonto – Avaliação científica com tradução", layout="centered")
st.title("🧠 DetectaOdonto – Avaliação automática com tradução PT→EN")

@st.cache_resource(show_spinner=False)
def carregar_modelo_embedding():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource(show_spinner=False)
def carregar_translator():
    return Translator()

modelo_embed = carregar_modelo_embedding()
translator = carregar_translator()

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

def dividir_texto(texto, tamanho=500):
    palavras = texto.split()
    return [" ".join(palavras[i:i+tamanho]) for i in range(0, len(palavras), tamanho)]

def traduzir_texto(texto_pt):
    blocos = dividir_texto(texto_pt, tamanho=200)
    texto_ingles = ""
    for bloco in blocos:
        resultado = translator.translate(bloco, src='pt', dest='en')
        texto_ingles += resultado.text + " "
    return texto_ingles.strip()

def buscar_artigos_pubmed(query, max_artigos=3):
    try:
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        search = requests.get(base_url + "esearch.fcgi", params={
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": max_artigos
        }).json()

        ids = search["esearchresult"]["idlist"]
        if not ids:
            return []

        fetch = requests.get(base_url + "efetch.fcgi", params={
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "xml"
        })
        soup = BeautifulSoup(fetch.content, "xml")
        abstracts = [artigo.Abstract.Text.text for artigo in soup.find_all("PubmedArticle") if artigo.Abstract]
        return abstracts
    except Exception as e:
        return []

url = st.text_input("📎 Cole aqui a URL da matéria odontológica:")

if st.button("🔍 Avaliar conteúdo"):
    if not url:
        st.warning("Por favor, insira uma URL.")
    else:
        texto_pt, erro = extrair_texto(url)
        if texto_pt is None:
            st.error(erro)
        else:
            st.subheader("📝 Texto extraído (português):")
            st.write(texto_pt[:2000] + "..." if len(texto_pt) > 2000 else texto_pt)

            st.info("🌐 Traduzindo texto para inglês para busca científica...")
            texto_en = traduzir_texto(texto_pt)
            st.write(texto_en[:2000] + "..." if len(texto_en) > 2000 else texto_en)

            st.info("🔬 Buscando artigos científicos relacionados no PubMed...")
            palavras_chave = (
                "dentistry OR dental treatment OR root canal OR orthodontics OR "
                "laser therapy OR dental aesthetics OR oral health OR fluoride OR "
                "caries OR gingivitis OR periodontitis OR whitening OR implants"
            )
            artigos = buscar_artigos_pubmed(palavras_chave)

            if not artigos:
                st.warning("Nenhum artigo científico encontrado para comparação.")
            else:
                blocos = dividir_texto(texto_en)
                emb_blocos = modelo_embed.encode(blocos, convert_to_tensor=True)

                melhor_score = 0
                melhor_abstract = ""
                for abstract in artigos:
                    emb_abstract = modelo_embed.encode(abstract, convert_to_tensor=True)
                    scores = util.cos_sim(emb_blocos, emb_abstract)
                    media = np.mean(scores).item()
                    if media > melhor_score:
                        melhor_score = media
                        melhor_abstract = abstract

                st.subheader("📊 Resultado da comparação:")
                if melhor_score > 0.6:
                    st.success(f"✅ Conteúdo com alta similaridade com evidência científica (score: {melhor_score:.2f})")
                elif melhor_score > 0.3:
                    st.warning(f"⚠️ Similaridade moderada com evidência científica (score: {melhor_score:.2f})")
                else:
                    st.error(f"❌ Baixa similaridade – potencial desinformação (score: {melhor_score:.2f})")

                st.markdown("**🧪 Artigo mais semelhante encontrado:**")
                st.caption(melhor_abstract[:700] + "...")
