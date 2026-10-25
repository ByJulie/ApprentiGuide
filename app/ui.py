import streamlit as st
from pipeline import answer

st.title("<Product name>")
q = st.text_input("Your question")
if q:
    out = answer(q)
    st.write(out["answer"])
    st.caption(f"Sources: {', '.join(out['sources']) or 'none'} · {out['seconds']} s")
