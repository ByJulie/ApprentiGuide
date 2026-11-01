import os

import streamlit as st
import yaml

from pipeline import answer

CFG = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "config.yaml")))
SNAPSHOT = CFG.get("corpus_snapshot", "")

st.title("ApprentiGuide")
st.caption("Questions sur l'apprentissage, le travail et le séjour des étudiants. Chaque réponse cite ses textes.")
q = st.text_input("Ta question")
audio = st.audio_input("Ou pose-la à la voix")
if audio is not None and not q:
    from voice import transcribe
    with st.spinner("Transcription..."):
        heard = transcribe(audio.getvalue())
    q = heard["text"]
    # Shown so the user can check what was understood before trusting the answer.
    st.info(f"Question entendue : {q}" if q else "Je n'ai rien entendu, essaie encore ou écris ta question.")
if q:
    out = answer(q)
    st.write(out["answer"])
    refs = out.get("citations", {})
    if refs.get("cited"):
        st.subheader("Sources")
        for s in refs["cited"]:
            date = f", version du {s['version_date']}" if s.get("version_date") else ""
            st.markdown(f"**[{s['id']}]({s['url']})** · {s['source_label']}{date}")
            with st.expander("Voir le passage"):
                for p in s["passages"]:
                    st.markdown(p["text"])
    if refs.get("invented"):
        st.warning(f"Citation non vérifiée (absente des textes trouvés) : {', '.join(refs['invented'])}")
    if refs.get("consulted"):
        with st.expander("Autres textes consultés"):
            for s in refs["consulted"]:
                st.markdown(f"[{s['id']}]({s['url']}) · {s['source_label']}")
    st.caption(f"Textes : Légifrance et Service-Public.fr/DILA (Licence Ouverte), fichiers du {SNAPSHOT}. "
               f"Vérifie toujours avec ton école ou tes RH · {out['seconds']} s")
