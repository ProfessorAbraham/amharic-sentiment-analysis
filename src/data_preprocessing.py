import jsonlines
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

def load_data(file_path):
    """Load data from jsonl file"""
    data = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data.append(obj['content'])
    return data

def preprocess_text(text):
    """Basic text preprocessing for Amharic"""
    text = text.strip()
    text = ' '.join(text.split())  # Normalize whitespace
    return text

def prepare_data(data, tokenizer_name="Davlan/afro-xlmr-mini", test_size=0.2):
    """Preprocess and tokenize Amharic text"""
    texts = [preprocess_text(text) for text in data]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    return encoded_inputs

if __name__ == "__main__":
    # Example usage for testing
    data = load_data("data/raw/amharic_data.jsonl")
    encoded = prepare_data(data[:100])
    print(f"Processed {len(encoded['input_ids'])} samples")