"""
build_dataset.py

Parses the official MedQuAD dataset (https://github.com/abachaa/MedQuAD) into a
single flat CSV that the chatbot can load.

MedQuAD ships as a folder of sub-collections (e.g. "1_CancerGov_QA",
"2_GARD_QA", ...), each containing many XML files shaped roughly like:

    <Document id="..." source="..." url="...">
      <Focus>Disease Name</Focus>
      <QAPairs>
        <QAPair>
          <Question qid="..." qtype="information">...</Question>
          <Answer>...</Answer>
        </QAPair>
        ...
      </QAPairs>
    </Document>

Usage:
    1. git clone https://github.com/abachaa/MedQuAD.git
    2. python build_dataset.py --medquad_dir /path/to/MedQuAD --out data/medquad_full.csv
"""
import argparse
import csv
import os
import xml.etree.ElementTree as ET


def parse_file(path):
    """Yield dict rows for every QA pair in one MedQuAD XML file."""
    try:
        tree = ET.parse(path)
    except ET.ParseError:
        return
    root = tree.getroot()

    focus_el = root.find("Focus")
    focus = focus_el.text.strip() if focus_el is not None and focus_el.text else ""
    source = root.attrib.get("source", "")
    url = root.attrib.get("url", "")

    for qa_pairs in root.findall("QAPairs"):
        for qa in qa_pairs.findall("QAPair"):
            q_el = qa.find("Question")
            a_el = qa.find("Answer")
            if q_el is None or a_el is None:
                continue
            question = (q_el.text or "").strip()
            answer = (a_el.text or "").strip()
            if not question or not answer:
                continue
            yield {
                "qid": q_el.attrib.get("qid", ""),
                "focus": focus,
                "question": question,
                "answer": answer,
                "question_type": q_el.attrib.get("qtype", ""),
                "source": source,
                "url": url,
            }


def main():
    parser = argparse.ArgumentParser(description="Build a flat CSV from raw MedQuAD XML files.")
    parser.add_argument("--medquad_dir", required=True, help="Path to a local clone of the MedQuAD repo")
    parser.add_argument("--out", default="data/medquad_full.csv", help="Output CSV path")
    args = parser.parse_args()

    rows = []
    for dirpath, _, filenames in os.walk(args.medquad_dir):
        for fname in filenames:
            if fname.lower().endswith(".xml"):
                rows.extend(parse_file(os.path.join(dirpath, fname)))

    if not rows:
        print("No QA pairs found. Check --medquad_dir points at the cloned MedQuAD repo.")
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fieldnames = ["qid", "focus", "question", "answer", "question_type", "source", "url"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} QA pairs to {args.out}")


if __name__ == "__main__":
    main()
