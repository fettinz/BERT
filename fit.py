import pandas as pd
from sklearn.model_selection import train_test_split    #divide the data into training and testing data
from transformers import BertTokenizer, BertForSequenceClassification, BertModel, Trainer, TrainingArguments
from datasets import Dataset
import torch
from datasets import ClassLabel
from math import ceil
import os

'''-------------------READ DATA-------------------'''
train_file_path = 'betsentiment-IT-tweets-sentiment-players.csv' 
df = pd.read_csv(train_file_path, encoding='utf-8')
df = df.drop(columns=["tweet_date_created", "language", "sentiment_score"])

'''-------------------PREPROCESSING-------------------'''

def clear_sentences(sentence):
    try:
        sentence = sentence.encode("latin").decode("windows-1252")
    except:
        pass

    try: 
        sentence = sentence.encode("cp1252").decode("windows-1252")
    except:
        pass
    return sentence

# Primo split: 80% train, 20% temp (val + test)
train_data, temp_data = train_test_split(df, test_size=0.2, random_state=42)

# Secondo split: divide temp in 50% val, 50% test → 10% ciascuno rispetto al totale
eval_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
# Salva anche il test set per il testing successivo
test_data.to_csv("test.csv", index=False)

train_dataset = Dataset.from_pandas(train_data)
eval_dataset = Dataset.from_pandas(eval_data)

unique_labels = train_dataset.unique("sentiment")
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for i, label in enumerate(unique_labels)}

def map_labels_and_clear_sentences(example):
    example["sentiment"] = label2id[example["sentiment"]]
    example["tweet_text"] = clear_sentences(example["tweet_text"])
    return example

train_dataset = train_dataset.map(map_labels_and_clear_sentences)
eval_dataset = eval_dataset.map(map_labels_and_clear_sentences)

train_dataset = train_dataset.cast_column("sentiment", ClassLabel(names=unique_labels))
eval_dataset = eval_dataset.cast_column("sentiment", ClassLabel(names=unique_labels))

'''-------------------FINE PREPROCESSING-------------------'''

model_name = 'dbmdz/bert-base-italian-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
model.to("cpu")

def tokenize_function(examples):
    examples["tweet_text"] = [str(text) for text in examples["tweet_text"]]
    return tokenizer(examples["tweet_text"], padding="max_length", truncation=True)

tokanized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokanized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

def format_dataset(dataset):
    dataset.set_format("torch")
    return dataset

tokanized_train_dataset = format_dataset(tokanized_train_dataset)
tokanized_eval_dataset = format_dataset(tokanized_eval_dataset)

train_batch_size = 16
num_train_epochs = 3

steps_per_epoch = ceil(len(train_dataset) / train_batch_size) 
total_training_steps = steps_per_epoch * num_train_epochs

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    lr_scheduler_type="linear",   
    warmup_steps=int(0.1 * total_training_steps),
)

def compute_metrics(pred):
    predictions = torch.argmax(torch.tensor(pred.predictions), dim=-1)
    labels = torch.tensor(pred.label_ids)
    accuracy = (predictions == labels).float().mean().item()
    return {"accuracy": accuracy}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokanized_train_dataset,
    eval_dataset=tokanized_eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

def train_model(): 
    saved_model_path = os.path.abspath(os.path.join("saved_model", "sentiment-model"))
    saved_tokenizer_path = os.path.abspath(os.path.join("saved_tokenizer", "sentiment-model"))
    trainer.train() #train the model
    trainer.save_model(saved_model_path) #save the model
    tokenizer.save_pretrained(saved_tokenizer_path) #save the tokenizer

'''-------------------FINE TRAINING-------------------'''   
