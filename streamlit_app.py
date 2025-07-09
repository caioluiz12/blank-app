import streamlit as st
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
import numpy as np

st.set_page_config(page_title="DetectaOdonto – Evidência Científica", layout="centered")
st.title("🧠 DetectaOdonto – Avaliação científica automatizada de conteúdos odontológicos")

# Carrega modelo de embeddings
@st.cache_resource
def carregar_modelo():
    return SentenceTransformer('all-MiniLM-L6-v2')

modelo = carregar_modelo()

# Extrai texto de um link
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

# Busca artigos no PubMed com E-utilities
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

# Interface do usuário
url = st.text_input("📎 Cole aqui a URL de uma matéria odontológica:")

if st.button("🔍 Avaliar conteúdo"):
    if not url:
        st.warning("Por favor, insira uma URL.")
    else:
        texto, erro = extrair_texto(url)
        if texto is None:
            st.error(erro)
        else:
            st.subheader("📝 Trecho do conteúdo analisado:")
            st.write(texto[:800] + "...")

            # Busca artigos relacionados com base em palavras frequentes
            st.info("🔬 Buscando artigos científicos relacionados...")
            palavras_chave = "odontologia OR tratamento dentário OR canal OR ortodontia OR laser OR estética dental"
            artigos = buscar_artigos_pubmed(palavras_chave)

            if not artigos:
                st.warning("Nenhum artigo científico encontrado para comparação.")
            else:
                # Embeddings e comparação por similaridade
                emb_texto = modelo.encode(texto, convert_to_tensor=True)
                similaridades = []
                for i, abstract in enumerate(artigos):
                    emb_abstract = modelo.encode(abstract, convert_to_tensor=True)
                    sim = util.cos_sim(emb_texto, emb_abstract).item()
                    similaridades.append((i, sim, abstract))

                # Mostra o mais similar
                mais_proximo = max(similaridades, key=lambda x: x[1])
                indice, score, abstract = mais_proximo

                st.subheader("📊 Resultado da comparação:")
                if score > 0.6:
                    st.success(f"✅ Conteúdo com alta similaridade com evidência científica (score: {score:.2f})")
                elif score > 0.3:
                    st.warning(f"⚠️ Similaridade moderada com evidência científica (score: {score:.2f})")
                else:
                    st.error(f"❌ Baixa similaridade – potencial desinformação (score: {score:.2f})")

                st.markdown("**🧪 Artigo utilizado na comparação:**")
                st.caption(abstract[:700] + "...")
