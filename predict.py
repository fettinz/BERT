import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import Dataset
import os


tokenizer = None
model = None
label2id = None
id2label = None
test_file_path = 'test.csv'

# ---------------- CONFIGURAZIONE ----------------
def load_model(): 
    saved_model_path = os.path.abspath(os.path.join("saved_model", "sentiment-model"))
    saved_tokenizer_path = os.path.abspath(os.path.join("saved_tokenizer", "sentiment-model"))
    assert os.path.exists(saved_tokenizer_path), f"Tokenizer non trovato in: {saved_tokenizer_path}"
    assert os.path.exists(saved_model_path), f"Modello non trovato in: {saved_model_path}"
    train_file_path = 'train.csv'

    # ---------------- CARICAMENTO MODELLO ----------------
    tokenizer = BertTokenizer.from_pretrained(saved_tokenizer_path)
    model = BertForSequenceClassification.from_pretrained(saved_model_path)
    model.to("cpu")
    model.eval()

    # ---------------- RICOSTRUISCI MAPPING LABELS ----------------
    df = pd.read_csv(train_file_path, encoding='utf-8')
    train_dataset = Dataset.from_pandas(df)
    unique_labels = train_dataset.unique("sentiment")
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for i, label in enumerate(unique_labels)}

# ---------------- FUNZIONE PER TEST SET ----------------
def test_model_on_test_set():
    test_data = pd.read_csv(test_file_path, encoding='utf-8')
    assert 'Text' in test_data.columns, "Il csv deve contenere una colonna 'Text'"

    for i, text in enumerate(test_data["tweet_text"]):
        inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
        print(f"Esempio {i+1}: Testo: {text[:100]}... | Predizione: {id2label[pred]}")
        if i >= 19:
            break

# ---------------- FUNZIONE PER FRASI ----------------
def test_sentence(sentence):
    inputs = tokenizer(sentence, padding="max_length", return_tensors = "pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim = 1).item()
    print(f"Predizione: {id2label[pred]}")

