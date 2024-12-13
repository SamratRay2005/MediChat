# Install required libraries
!pip install transformers datasets torch
!pip install seqeval

import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Load and preprocess the data
bert_data = pd.read_csv('bert.csv')

# Prepare symptoms and labels
symptoms = bert_data.columns[1:]  # Assuming the first column is the disease name
bert_data['text'] = bert_data.apply(lambda x: ' '.join(symptoms[x[symptoms] == 1]), axis=1)

# No data fractioning - Use entire dataset
X_train, X_val, y_train, y_val = train_test_split(bert_data['text'], bert_data[symptoms].astype(float), test_size=0.1)

# Tokenization using ClinicalBERT
tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(list(X_val), truncation=True, padding=True, max_length=128)

# Dataset class
class SymptomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels.iloc[idx].values, dtype=torch.float)  # Ensure labels are float
        return item

    def __len__(self):
        return len(self.labels)

# Create train and validation datasets
train_dataset = SymptomDataset(train_encodings, y_train.reset_index(drop=True))
val_dataset = SymptomDataset(val_encodings, y_val.reset_index(drop=True))

# Load Pre-trained BERT model for multi-label classification
model = BertForSequenceClassification.from_pretrained(
    'emilyalsentzer/Bio_ClinicalBERT',
    num_labels=len(symptoms),
    problem_type="multi_label_classification"
)

# Metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = (pred.predictions > 0.5).astype(int)
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {'accuracy': accuracy, 'f1': f1}

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    report_to="none",
    save_total_limit=2,  # Limits the total number of checkpoints
    save_steps=500
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Saving the model and tokenizer
trainer.save_model('./my_bert_model')  # Saves the model, tokenizer, and config
tokenizer.save_pretrained('./my_bert_model')

print("Model and tokenizer saved successfully.")

# Loading the model and tokenizer
def load_model_and_tokenizer(save_directory):
    model = BertForSequenceClassification.from_pretrained(save_directory)
    tokenizer = BertTokenizer.from_pretrained(save_directory)
    return model, tokenizer

loaded_model, loaded_tokenizer = load_model_and_tokenizer('./my_bert_model')
print("Model and tokenizer loaded successfully.")
