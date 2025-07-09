import streamlit as st
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from googletrans import Translator
from keybert import KeyBERT
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize

# Baixar pacote NLTK se necessário (descomente para primeira execução)
# nltk.download('punkt')

st.set_page_config(page_title="DetectaOdonto – Checagem granular", layout="centered")
st.title("🦷 DetectaOdonto – Checagem granular de claims com busca científica")

@st.cache_resource(show_spinner=False)
def carregar_modelos():
    modelo_embed = SentenceTransformer('all-MiniLM-L6-v2')
    translator = Translator()
    kw_model = KeyBERT('all-MiniLM-L6-v2')
    return modelo_embed, translator, kw_model

modelo_embed, translator, kw_model = carregar_modelos()

def extrair_texto(url):
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        paragrafos = soup.find_all("p")
        texto = " ".join(p.text for p in paragrafos)
        return texto, None
    except Exception as e:
        return None, f"Erro ao extrair texto: {e}"

def extrair_termos_chave(texto, top_n=10):
    keywords = kw_model.extract_keywords(texto, keyphrase_ngram_range=(1,2), stop_words='english', top_n=top_n)
    termos = [k[0] for k in keywords]
    return termos

def traduzir_lista(textos_pt):
    traduzidos = []
    for texto in textos_pt:
        trad = translator.translate(texto, src='pt', dest='en').text
        traduzidos.append(trad)
    return traduzidos

def buscar_artigos_pubmed(termos, max_artigos=5):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    query = " OR ".join(termos)
    try:
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
        abstracts = [art.Abstract.Text.text for art in soup.find_all("PubmedArticle") if art.Abstract]
        return abstracts
    except Exception as e:
        return []

def dividir_em_claims(texto):
    # Usar nltk para dividir em sentenças (claims)
    sentencas = sent_tokenize(texto)
    # Opcional: filtrar sentenças muito curtas
    claims = [s for s in sentencas if len(s) > 20]
    return claims

st.write("## Insira a URL da matéria odontológica para avaliação:")
url = st.text_input("URL:")

if st.button("Analisar"):
    if not url:
        st.warning("Insira uma URL válida.")
    else:
        texto_pt, erro = extrair_texto(url)
        if texto_pt is None:
            st.error(erro)
        else:
            st.subheader("📝 Texto extraído:")
            st.write(texto_pt[:2000] + "..." if len(texto_pt) > 2000 else texto_pt)

            # Extrair termos chave para busca
            termos_pt = extrair_termos_chave(texto_pt, top_n=8)
            st.write("🔑 Termos-chave extraídos (PT):", termos_pt)

            termos_en = traduzir_lista(termos_pt)
            st.write("🌐 Termos-chave traduzidos (EN):", termos_en)

            st.info("🔬 Buscando artigos científicos relacionados no PubMed...")
            artigos = buscar_artigos_pubmed(termos_en, max_artigos=5)

            if not artigos:
                st.warning("Nenhum artigo científico encontrado para comparação.")
            else:
                st.success(f"{len(artigos)} artigos encontrados.")

                claims = dividir_em_claims(texto_pt)
                st.write(f"🧾 Número de claims extraídos: {len(claims)}")

                emb_claims = modelo_embed.encode(claims, convert_to_tensor=True)
                emb_artigos = modelo_embed.encode(artigos, convert_to_tensor=True)

                st.subheader("🔍 Avaliação dos claims:")
                for i, claim in enumerate(claims):
                    scores = util.cos_sim(emb_claims[i], emb_artigos)
                    max_sim = scores.max().item()

                    if max_sim > 0.6:
                        status = "✅ Suporte científico"
                        color = "green"
                    elif max_sim > 0.3:
                        status = "⚠️ Similaridade moderada"
                        color = "orange"
                    else:
                        status = "❌ Sem suporte aparente (possível desinformação)"
                        color = "red"

                    st.markdown(f"**Claim {i+1}:** {claim}")
                    st.markdown(f"<span style='color:{color}'>{status} (similaridade: {max_sim:.2f})</span>", unsafe_allow_html=True)
                    st.write("---")
