"""
entity_recognizer.py

A lightweight, dependency-free medical entity recognizer.

It does NOT use a heavyweight NLP model (spaCy/scispaCy). Instead it matches
against curated vocabularies of diseases, symptoms, and treatments using
word-boundary-aware phrase matching. This keeps the app installable and
runnable anywhere (no model downloads), while still giving useful, structured
output. Longer phrases are matched before their sub-strings so that e.g.
"type 2 diabetes" is captured as one entity rather than being split into
"type 2" + "diabetes".

Swap-in point: if you want higher recall, replace `MedicalEntityRecognizer`
with a call to scispaCy's `en_ner_bc5cdr_md` model — the public interface
(`extract(text) -> dict`) is intentionally model-agnostic.
"""
import json
import os
import re
from collections import defaultdict


class MedicalEntityRecognizer:
    def __init__(self, vocab_path=None):
        if vocab_path is None:
            vocab_path = os.path.join(os.path.dirname(__file__), "..", "data", "medical_vocab.json")
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        self.categories = list(vocab.keys())  # e.g. diseases, symptoms, treatments
        # Compile one regex per category, longest phrases first so multi-word
        # terms win over shorter overlapping ones.
        self._patterns = {}
        for category, terms in vocab.items():
            terms_sorted = sorted(set(terms), key=len, reverse=True)
            escaped = [re.escape(t) for t in terms_sorted]
            pattern = r"\b(" + "|".join(escaped) + r")\b"
            self._patterns[category] = re.compile(pattern, flags=re.IGNORECASE)

    def extract(self, text):
        """
        Returns {category: [unique matched entities in order of first appearance]}
        for every category in the vocabulary, e.g.:
            {"diseases": ["diabetes"], "symptoms": ["fatigue", "blurred vision"], "treatments": []}
        """
        text = text or ""
        results = defaultdict(list)
        already_spanned = []  # (start, end) spans already claimed by a longer match

        for category in self.categories:
            seen = set()
            for m in self._patterns[category].finditer(text):
                span = (m.start(), m.end())
                # avoid re-claiming a substring already covered by a longer match
                if any(span[0] >= s and span[1] <= e for s, e in already_spanned):
                    continue
                term = m.group(1).lower()
                if term not in seen:
                    seen.add(term)
                    results[category].append(term)
                already_spanned.append(span)

        # ensure every category key is present even if empty
        for category in self.categories:
            results.setdefault(category, [])
        return dict(results)

    def has_entities(self, text):
        ents = self.extract(text)
        return any(len(v) > 0 for v in ents.values())


if __name__ == "__main__":
    ner = MedicalEntityRecognizer()
    sample = "I have a fever, sore throat, and cough. Could this be the flu? I've also read insulin helps with type 2 diabetes."
    print(ner.extract(sample))
