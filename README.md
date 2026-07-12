# Medical Q&A Chatbot (MedQuAD)

A retrieval-based medical question-answering chatbot built on the
[MedQuAD dataset](https://github.com/abachaa/MedQuAD), with basic medical
entity recognition and a Streamlit chat interface.

> ⚠️ **Not medical advice.** This is an educational/demo project. Answers are
> retrieved from a public dataset and are not a substitute for professional
> medical care.

## What's inside

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI — the chat interface |
| `src/chatbot.py` | Glues retrieval + entity recognition into `MedicalChatbot.answer()` |
| `src/retriever.py` | TF-IDF + cosine-similarity retrieval over the QA dataset |
| `src/entity_recognizer.py` | Rule-based recognizer for diseases, symptoms, treatments |
| `src/data_loader.py` | Loads whichever dataset is available (full or sample) |
| `build_dataset.py` | Parses a real local clone of MedQuAD's XML files into one CSV |
| `data/build_sample_data.py` | Generates a small bundled sample dataset (fallback) |
| `data/medical_vocab.json` | Curated vocabulary used for entity recognition |
| `data/sample_medquad.csv` | ~50 sample QA pairs so the app runs out of the box |

## How it works

1. **Retrieval** — Questions and answers are indexed with a TF-IDF vectorizer
   (unigrams + bigrams, over `focus disease + question` text). A user's query
   is vectorized the same way and matched by cosine similarity. This needs no
   internet access or model download, so it runs anywhere.
2. **Entity recognition** — `MedicalEntityRecognizer` matches the user's
   question against curated phrase lists for **diseases**, **symptoms**, and
   **treatments** (`data/medical_vocab.json`), using longest-match-first,
   word-boundary-aware regex. If the raw query scores too low against the
   dataset, the chatbot retries the search using just the recognized disease
   name(s) as a fallback — this handles phrasing like "tell me about the flu"
   that doesn't closely match a stored question like "What is the flu?".
3. **UI** — Streamlit renders the conversation, highlights recognized
   entities as colored chips, shows the best-matching answer plus a few
   related ones, and lets you tune the number of results / relevance
   threshold from the sidebar.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Option A — run immediately with the bundled sample data

The repo already includes `data/sample_medquad.csv` (~50 hand-written QA
pairs covering common conditions), so you can just run:

```bash
streamlit run app.py
```

### Option B — use the full, real MedQuAD dataset (recommended)

```bash
git clone https://github.com/abachaa/MedQuAD.git
python build_dataset.py --medquad_dir ./MedQuAD --out data/medquad_full.csv
streamlit run app.py
```

`data_loader.py` automatically prefers `data/medquad_full.csv` over the
sample file if it exists, so no other code changes are needed. Note that
MedQuAD's license excludes some sub-collections (e.g. some CDC/GARD content
has redistribution notes) — see the dataset's own README for details.

## Extending this project

- **Better retrieval:** swap `src/retriever.py`'s TF-IDF vectorizer for a
  sentence-embedding model (e.g. `sentence-transformers`) plus a vector index
  (e.g. FAISS/Chroma) for semantic (not just lexical) matching.
- **Better entity recognition:** swap `MedicalEntityRecognizer` for a proper
  clinical NER model, e.g. scispaCy's `en_ner_bc5cdr_md`, or a UMLS-linked
  model — the `extract(text) -> dict` interface is intentionally
  model-agnostic so this is a drop-in change.
- **Answer re-ranking:** use the recognized entities to boost/filter retrieved
  candidates whose `focus` field matches a recognized disease.
- **Conversation memory:** track prior turns to resolve follow-ups like
  "What about children?" after a question about a specific disease.

## Testing without Streamlit

```bash
python -c "
from src.chatbot import MedicalChatbot
bot = MedicalChatbot()
r = bot.answer('What are the symptoms of diabetes?')
print(r['entities'])
print(r['results'][0]['answer'])
"
```
