"""
dynamic_chunker.py
Compresses retrieved text chunks into a ~150-token summary
using google/flan-t5-base directly (no pipeline — compatible with transformers v5).
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from config import MODEL_REVISIONS

_model = None
_tokenizer = None

def get_model():
    global _model, _tokenizer
    if _model is None:
        print("Loading flan-t5-base summarizer...")
        _tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base", revision=MODEL_REVISIONS["google/flan-t5-base"])
        _model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", revision=MODEL_REVISIONS["google/flan-t5-base"])
        _model.eval()
        print("Summarizer ready.")
    return _model, _tokenizer

def summarize_chunks(chunks: list, max_input_chars: int = 2000) -> str:
    combined = " ".join(chunks)[:max_input_chars]
    prompt = "Summarize the following clinical notes concisely:\n\n" + combined
    model, tokenizer = get_model()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150, min_new_tokens=40)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    test_chunks = [
        "Patient presented with high fever, elevated WBC, and positive blood cultures indicating sepsis.",
        "Started on broad-spectrum antibiotics. Blood pressure stabilised after fluid resuscitation.",
        "ICU admission required. Ventilator support initiated on day 2 due to respiratory failure.",
    ]
    summary = summarize_chunks(test_chunks)
    print("Summary:", summary)
