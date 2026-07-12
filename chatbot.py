"""
chatbot.py

Ties the retriever and the entity recognizer together into a single
MedicalChatbot.answer(query) call used by the Streamlit UI (and reusable
from any other front end, e.g. a CLI or an API).
"""
from src.data_loader import load_dataset
from src.retriever import Retriever
from src.entity_recognizer import MedicalEntityRecognizer

DISCLAIMER = (
    "This information is for general educational purposes only and is not a substitute "
    "for professional medical advice, diagnosis, or treatment. Always consult a qualified "
    "health provider with questions about a medical condition."
)


class MedicalChatbot:
    def __init__(self):
        self.df, self.dataset_label = load_dataset()
        self.retriever = Retriever(self.df)
        self.ner = MedicalEntityRecognizer()

    def answer(self, query, top_k=3, min_score=0.05):
        """
        Returns:
            {
              "query": str,
              "entities": {"diseases": [...], "symptoms": [...], "treatments": [...]},
              "results": [ {question, answer, focus, score, ...}, ... ],
              "disclaimer": str,
            }
        """
        entities = self.ner.extract(query)
        results = self.retriever.search(query, top_k=top_k, min_score=min_score)

        # If the raw query scores poorly (e.g. "tell me about the flu" vs the
        # dataset phrasing "What is the flu?"), retry using recognized disease
        # entities as a fallback query -- this materially improves recall.
        if not results and entities.get("diseases"):
            fallback_query = " ".join(entities["diseases"])
            results = self.retriever.search(fallback_query, top_k=top_k, min_score=min_score)

        return {
            "query": query,
            "entities": entities,
            "results": results,
            "disclaimer": DISCLAIMER,
        }
