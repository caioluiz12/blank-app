import streamlit as st


import requests
from beautifulsoup4 import Article
from sklearn.externals import joblib  # se usar modelo treinado
# ou use HuggingFace, transformers, etc.

# Configuração do app
st.set_page_config(page_title="Detecção de Desinformação Odontológica", layout="centered")
st.title("🦷 DetectaOdonto – IA contra desinformação na odontologia")

# Entrada do usuário
url = st.text_input("Cole aqui a URL de uma postagem ou matéria:")

if st.button("Analisar"):
    if url:
        try:
            # Coleta e limpeza do conteúdo
            artigo = Article(url)
            artigo.download()
            artigo.parse()
            texto = artigo.text

            st.subheader("📝 Texto extraído:")
            st.write(texto[:800] + "...")  # mostra só um pedaço

            # Detectar possíveis claims (exemplo bem simples)
            claims = [frase for frase in texto.split(".") if any(x in frase.lower() for x in ["cura", "substitui", "elimina", "resolve", "sem dor"])]

            st.subheader("🔍 Claims detectados:")
            for i, claim in enumerate(claims):
                st.markdown(f"**Claim {i+1}:** {claim.strip()}")

                # Aqui entraria seu classificador (exemplo com score simulado)
                import random
                score = random.random()

                if score > 0.6:
                    st.error("⚠️ Potencial desinformação")
                else:
                    st.success("✅ Baixo risco de desinformação")

                # Resposta da IA (simulação por enquanto)
                st.caption("_Resposta explicativa gerada pela IA aqui..._")

                # Validação pelo especialista
                st.radio(f"Você concorda com a avaliação do Claim {i+1}?", ["Sim", "Não"], key=f"validacao_{i+1}")

        except Exception as e:
            st.error(f"Erro ao processar URL: {e}")
