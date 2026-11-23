"""Interface web : streamlit run app.py"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from rag import Assistant  # noqa: E402

st.set_page_config(page_title="Assistant conseiller bancaire", page_icon="💬")
st.title("💬 Assistant documentaire du conseiller")
st.caption("Recherche dans la base de procédures (documents fictifs) et réponse sourcée.")


@st.cache_resource
def charger():
    return Assistant()


q = st.text_input("Question du client", "J'ai perdu ma carte, que dois-je faire ?")
if q:
    r = charger().repondre(q)
    st.markdown(f"**Réponse** ({r.get('mode', '–')}) : {r['reponse']}")
    with st.expander("Passages utilisés"):
        for s in r["sources"]:
            st.write(f"`{s['source']}` · {s['section']} · score {s['score']}")
