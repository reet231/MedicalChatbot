"""
app.py — Streamlit UI for the MedQuAD-based medical Q&A chatbot.

Run with:
    streamlit run app.py
"""
import streamlit as st
from src.chatbot import MedicalChatbot

st.set_page_config(page_title="Medical Q&A Chatbot", page_icon="🩺", layout="wide")

ENTITY_COLORS = {"diseases": "#e0562b", "symptoms": "#2b7de0", "treatments": "#2ba85a"}
ENTITY_LABELS = {"diseases": "Conditions", "symptoms": "Symptoms", "treatments": "Treatments"}


@st.cache_resource
def get_chatbot():
    return MedicalChatbot()


def render_entities(entities):
    chips = []
    for category, terms in entities.items():
        color = ENTITY_COLORS.get(category, "#888")
        for term in terms:
            chips.append(
                f'<span style="background:{color}22;border:1px solid {color};color:{color};'
                f'padding:2px 10px;border-radius:12px;margin:2px;display:inline-block;font-size:0.85em;">'
                f'{term} <span style="opacity:0.7">· {ENTITY_LABELS.get(category, category)}</span></span>'
            )
    if chips:
        st.markdown(" ".join(chips), unsafe_allow_html=True)
    else:
        st.caption("No recognized medical entities in this question.")


def main():
    st.title("🩺 Medical Q&A Chatbot")
    st.caption("Retrieval-based answers over the MedQuAD dataset, with basic medical entity recognition.")

    try:
        bot = get_chatbot()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    with st.sidebar:
        st.header("About")
        st.write(
            "This chatbot retrieves the most relevant answer(s) to your question from the "
            "**MedQuAD** dataset using TF-IDF similarity search, and highlights any medical "
            "entities (conditions, symptoms, treatments) it recognizes in your question."
        )
        st.info(f"**Dataset in use:**\n\n{bot.dataset_label}")
        top_k = st.slider("Number of answers to show", min_value=1, max_value=5, value=3)
        min_score = st.slider("Minimum relevance score", min_value=0.0, max_value=0.5, value=0.05, step=0.01)
        st.divider()
        st.subheader("Try asking...")
        for example in [
            "What are the symptoms of diabetes?",
            "How is asthma treated?",
            "What causes high blood pressure?",
            "What is the difference between a cold and the flu?",
            "What are the warning signs of a stroke?",
        ]:
            if st.button(example, use_container_width=True):
                st.session_state["query_input"] = example

    if "history" not in st.session_state:
        st.session_state.history = []
    if "query_input" not in st.session_state:
        st.session_state.query_input = ""

    query = st.text_input(
        "Ask a medical question:",
        key="query_input",
        placeholder="e.g. What are the symptoms of asthma?",
    )
    ask_clicked = st.button("Ask", type="primary")

    if ask_clicked and query.strip():
        with st.spinner("Searching MedQuAD..."):
            response = bot.answer(query, top_k=top_k, min_score=min_score)
        st.session_state.history.insert(0, response)

    if st.session_state.history:
        st.divider()
        for i, response in enumerate(st.session_state.history):
            st.markdown(f"### 🧑 {response['query']}")
            render_entities(response["entities"])

            if not response["results"]:
                st.warning(
                    "I couldn't find a confident match in the dataset for that question. "
                    "Try rephrasing, mentioning a specific condition by name, or lowering the "
                    "minimum relevance score in the sidebar."
                )
            else:
                best, *others = response["results"]
                st.success(best["answer"])
                meta_bits = []
                if best.get("focus"):
                    meta_bits.append(f"**Topic:** {best['focus']}")
                if best.get("source"):
                    meta_bits.append(f"**Source:** {best['source']}")
                meta_bits.append(f"**Match confidence:** {best['score']:.0%}")
                st.caption(" · ".join(meta_bits))

                if others:
                    with st.expander(f"See {len(others)} more related answer(s)"):
                        for res in others:
                            st.markdown(f"**{res['question']}**")
                            st.write(res["answer"])
                            st.caption(f"Match confidence: {res['score']:.0%}")
                            st.markdown("---")

            st.caption(f"⚠️ {response['disclaimer']}")
            if i < len(st.session_state.history) - 1:
                st.divider()


if __name__ == "__main__":
    main()
