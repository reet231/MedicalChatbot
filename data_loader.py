"""
data_loader.py

Loads the QA dataset the chatbot retrieves from.

Resolution order:
  1. data/medquad_full.csv   -- produced by running build_dataset.py against a
                                 real local clone of https://github.com/abachaa/MedQuAD
  2. data/sample_medquad.csv -- small bundled sample so the app runs out of the box
"""
import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FULL_PATH = os.path.join(DATA_DIR, "medquad_full.csv")
SAMPLE_PATH = os.path.join(DATA_DIR, "sample_medquad.csv")


def load_dataset():
    """Returns (dataframe, source_label)."""
    if os.path.exists(FULL_PATH):
        df = pd.read_csv(FULL_PATH)
        label = f"Full MedQuAD dataset ({len(df)} QA pairs)"
    elif os.path.exists(SAMPLE_PATH):
        df = pd.read_csv(SAMPLE_PATH)
        label = f"Bundled sample dataset ({len(df)} QA pairs) — run build_dataset.py for the full MedQuAD dataset"
    else:
        raise FileNotFoundError(
            "No dataset found. Run `python data/build_sample_data.py` to create the sample dataset, "
            "or `python build_dataset.py --medquad_dir <path>` for the full MedQuAD dataset."
        )

    df = df.dropna(subset=["question", "answer"]).reset_index(drop=True)
    for col in ["focus", "question_type", "source", "url"]:
        if col not in df.columns:
            df[col] = ""
    return df, label
