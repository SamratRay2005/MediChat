# Install required libraries if not already installed
!pip install transformers datasets torch
!pip install seqeval

import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# Load and preprocess the data
bert_data = pd.read_csv('/kaggle/input/newone/bert.csv')

# Prepare symptoms and labels
symptoms = bert_data.columns[1:]  # Assuming the first column is the disease name
bert_data['text'] = bert_data.apply(lambda x: ' '.join(symptoms[x[symptoms] == 1]), axis=1)

# Extract the disease label as a single column and map each unique disease to a numeric label
bert_data['label'] = bert_data[bert_data.columns[0]].factorize()[0]  # Factorize the disease column

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(bert_data['text'], bert_data['label'], test_size=0.1)


# Tokenization using ClinicalBERT
tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(list(X_val), truncation=True, padding=True, max_length=128)

# Dataset class for multi-class classification
class DiseaseDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels.iloc[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

# Create train and validation datasets
train_dataset = DiseaseDataset(train_encodings, y_train.reset_index(drop=True))
val_dataset = DiseaseDataset(val_encodings, y_val.reset_index(drop=True))

# Load Pre-trained BERT model for multi-class classification
model = BertForSequenceClassification.from_pretrained(
    'emilyalsentzer/Bio_ClinicalBERT',
    num_labels=len(bert_data[bert_data.columns[0]].unique())  # Number of unique diseases
)
# Metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
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
    save_total_limit=2,
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
trainer.save_model('./my_bert_model2')  # Saves the model, tokenizer, and config
tokenizer.save_pretrained('./my_bert_model2')

import shutil

# Zip the saved model directory
model_directory = '/kaggle/working/my_bert_model2'
zip_file_path = '/kaggle/working/my_bert_model2'

# Creating a zip archive of the model directory
shutil.make_archive(zip_file_path, 'zip', model_directory)

zip_file_path
