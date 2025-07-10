import streamlit as st
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from googletrans import Translator
import openai
import os
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

# ✅ Lê a chave do ambiente (definida no secrets.toml ou painel)
openai.api_key = os.getenv("OPENAI_API_KEY")

# ✅ Verifica se a chave existe
if openai.api_key is None:
    st.error("🚫 A chave OPENAI_API_KEY não está configurada. Verifique o arquivo `.streamlit/secrets.toml` ou os secrets do Streamlit Cloud.")
    st.stop()

st.set_page_config(page_title="DetectaOdonto", layout="centered")
st.title("🦷 DetectaOdonto – Analisando conteúdo com base científica")

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

def gerar_resumo_openai(texto):
    prompt = f"Resuma os principais claims científicos da seguinte notícia odontológica em português:\n\n{texto}\n\nResumo:"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()

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
    except Exception:
        return []

url = st.text_input("🔗 Insira o link da notícia odontológica")

if st.button("Analisar conteúdo"):
    if not url:
        st.warning("Por favor, insira a URL de uma matéria.")
    else:
        texto_pt, erro = extrair_texto(url)
        if texto_pt is None:
            st.error(f"Erro ao extrair o texto: {erro}")
        else:
            st.subheader("📝 Texto extraído:")
            st.write(texto_pt[:2000] + ("..." if len(texto_pt) > 2000 else ""))

            st.info("🧠 Gerando resumo dos claims...")
            resumo_pt = gerar_resumo_openai(texto_pt)
            st.write(resumo_pt)

            st.info("🌐 Traduzindo para inglês...")
            resumo_en = traduzir_texto(resumo_pt)
            st.write(resumo_en)

            st.info("🔬 Pesquisando no PubMed...")
            artigos = buscar_artigos_pubmed(resumo_en)

            if not artigos:
                st.warning("Nenhum artigo relevante encontrado.")
            else:
                emb_resumo = modelo_embed.encode(resumo_en, convert_to_tensor=True)
                emb_artigos = modelo_embed.encode(artigos, convert_to_tensor=True)
                scores = util.cos_sim(emb_resumo, emb_artigos)

                melhor_score = scores.max().item()
                indice_melhor = scores.argmax().item()

                st.subheader("🔍 Resultado:")
                if melhor_score > 0.6:
                    st.success(f"✅ Alta similaridade com evidência científica (score: {melhor_score:.2f})")
                elif melhor_score > 0.3:
                    st.warning(f"⚠️ Similaridade moderada (score: {melhor_score:.2f})")
                else:
                    st.error(f"❌ Baixa similaridade – possível desinformação (score: {melhor_score:.2f})")

                st.markdown("### 📚 Artigo mais próximo encontrado:")
                st.write(artigos[indice_melhor][:800] + "...")
