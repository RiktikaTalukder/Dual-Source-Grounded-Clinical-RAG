"""
dynamic_chunker.py
Compresses a list of retrieved text chunks into a ~150-token summary
using google/flan-t5-base.
"""

from transformers import pipeline
import textwrap

_summarizer = None

def get_summarizer():
    global _summarizer
    if _summarizer is None:
        print("Loading flan-t5-base summarizer (first time ~1 min)...")
        _summarizer = pipeline(
            "summarization",
            model="google/flan-t5-base",
            tokenizer="google/flan-t5-base",
            max_new_tokens=150,
            min_new_tokens=40,
        )
        print("Summarizer ready.")
    return _summarizer

def summarize_chunks(chunks: list[str], max_input_chars: int = 2000) -> str:
    """
    Takes a list of text chunks, joins them, and returns a
    ~150-token clinical summary.
    """
    combined = " ".join(chunks)[:max_input_chars]
    prompt   = f"Summarize the following clinical notes concisely:\n\n{combined}"
    result   = get_summarizer()(prompt, truncation=True)
    return result[0]["summary_text"]


if __name__ == "__main__":
    test_chunks = [
        "Patient presented with high fever, elevated WBC, and positive blood cultures indicating sepsis.",
        "Started on broad-spectrum antibiotics. Blood pressure stabilised after fluid resuscitation.",
        "ICU admission required. Ventilator support initiated on day 2 due to respiratory failure.",
    ]
    summary = summarize_chunks(test_chunks)
    print("Summary:", summary)
