import streamlit as st
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from googletrans import Translator
import openai
import numpy as np

st.set_page_config(page_title="DetectaOdonto com OpenAI", layout="centered")
st.title("🦷 DetectaOdonto – Resumo inteligente com OpenAI")

@st.cache_resource
def carregar_modelo_embedding():
    return SentenceTransformer('all-MiniLM-L6-v2')

modelo_embed = carregar_modelo_embedding()
translator = Translator()

def extrair_texto(url):
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        paragrafos = soup.find_all("p")
        texto = " ".join(p.text for p in paragrafos)
        return texto, None
    except Exception as e:
        return None, str(e)

def gerar_resumo_openai(texto, chave_api):
    openai.api_key = chave_api
    prompt = f"Resuma em português os principais claims científicos da seguinte notícia odontológica:\n\n{texto}\n\nResumo:"
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.3,
    )
    resumo = response.choices[0].message.content.strip()
    return resumo

def traduzir_texto(texto_pt):
    trad = translator.translate(texto_pt, src='pt', dest='en')
    return trad.text

def buscar_artigos_pubmed(query, max_artigos=5):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
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

url = st.text_input("URL da notícia odontológica")
api_key = st.text_input("Chave API OpenAI (sk-...)", type="password")

if st.button("Analisar"):

    if not url:
        st.warning("Insira a URL da notícia.")
    elif not api_key:
        st.warning("Insira a chave API OpenAI.")
    else:
        texto_pt, erro = extrair_texto(url)
        if texto_pt is None:
            st.error(f"Erro ao extrair texto: {erro}")
        else:
            st.subheader("📝 Texto extraído (PT):")
            st.write(texto_pt[:1500] + ("..." if len(texto_pt) > 1500 else ""))

            st.info("🧠 Gerando resumo dos claims com OpenAI...")
            resumo_pt = gerar_resumo_openai(texto_pt, api_key)
            st.write(resumo_pt)

            st.info("🌐 Traduzindo resumo para inglês...")
            resumo_en = traduzir_texto(resumo_pt)
            st.write(resumo_en)

            st.info("🔬 Buscando artigos no PubMed...")
            artigos = buscar_artigos_pubmed(resumo_en, max_artigos=5)

            if not artigos:
                st.warning("Nenhum artigo científico encontrado para comparação.")
            else:
                emb_resumo = modelo_embed.encode(resumo_en, convert_to_tensor=True)
                emb_artigos = modelo_embed.encode(artigos, convert_to_tensor=True)

                scores = util.cos_sim(emb_resumo, emb_artigos)
                melhor_score = scores.max().item()
                indice_melhor = scores.argmax().item()

                st.subheader("🔍 Resultado da comparação:")
                if melhor_score > 0.6:
                    st.success(f"✅ Conteúdo com alta similaridade científica (score: {melhor_score:.2f})")
                elif melhor_score > 0.3:
                    st.warning(f"⚠️ Similaridade moderada (score: {melhor_score:.2f})")
                else:
                    st.error(f"❌ Baixa similaridade – possível desinformação (score: {melhor_score:.2f})")

                st.markdown("### Artigo mais semelhante encontrado:")
                st.write(artigos[indice_melhor][:700] + "...")

