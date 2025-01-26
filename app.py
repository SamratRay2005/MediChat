from flask import Flask, render_template, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import requests
import os
import string
# import faiss
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Load the BERT data and models
bert_data = pd.read_csv('bert.csv')
symptoms = bert_data.columns[1:]

# Step 1: Load Disease Dataset and FAISS Index
data = pd.read_csv("Disease_Info.csv")  # Ensure this file has "Disease" and "Description" columns
descriptions = data["Description"].tolist()
diseases = data["Disease"].tolist()


# # FAISS Index Path
# faiss_index_path = "faiss_index/index.faiss"
# faiss_index = faiss.read_index(faiss_index_path)


# # Embedding Model
# embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
# embedding_model = SentenceTransformer(embedding_model_name)

# Step 2: Define Query Function
def retrieve_info_from_csv(disease_name):
    """
    Retrieve the description of a disease directly from the CSV.
    """
    match = data[data["Disease"].str.lower() == disease_name.lower()]
    if not match.empty:
        description = match["Description"].iloc[0]
        return {"disease": disease_name, "description": description}
    else:
        return {"disease": None, "description": "Information not found for the given disease."}

# Example Query Test
example_query = "diabetes insipidus"
# example_result = retrieve_closest_source(example_query)
# print(f"Example Result: {example_result}")

initial_bert_dir = './models/my_bert_model'
initial_bert_model = BertForSequenceClassification.from_pretrained(initial_bert_dir)
initial_bert_tokenizer = BertTokenizer.from_pretrained(initial_bert_dir)

bert_model_dir = './models/my_bert_model2'
bert_model = BertForSequenceClassification.from_pretrained(bert_model_dir)
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
initial_bert_model.to(device)
bert_model.to(device)

# Your actual API key
api_key = "AIzaSyApvZXWXyl48mnoza945A8CPdh95I4gwg4"  # Replace with your actual API key

def clean_text(text):
    # Create a translation table that maps punctuation to spaces
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    # Remove punctuation by translating them to spaces
    text = text.translate(translator)
    # Normalize whitespace by splitting and rejoining with a single space
    return ' '.join(text.split())


# Helper functions (same as before)
def classify_symptoms(text, threshold=0.5):
    inputs = initial_bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    outputs = initial_bert_model(**inputs)
    probabilities = torch.sigmoid(outputs.logits).detach().cpu().numpy()[0]
    return [symptoms[i] for i, prob in enumerate(probabilities) if prob > threshold]

def predict_disease(detected_symptoms, top_n=10, threshold=0.7):
    symptoms_text = " ".join(detected_symptoms)
    inputs = bert_tokenizer(symptoms_text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    probabilities = torch.sigmoid(bert_model(**inputs).logits).detach().cpu().numpy()[0]
    probable_diseases = sorted(
        [(bert_data[bert_data.columns[0]].unique()[i], prob) 
         for i, prob in enumerate(probabilities) if prob > threshold],
        key=lambda x: x[1], reverse=True)[:top_n]
    return probable_diseases

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    user_input = clean_text(user_input)
    detected_symptoms = classify_symptoms(user_input)
    
    if detected_symptoms:
        probable_diseases = predict_disease(detected_symptoms)
        if probable_diseases:
            diseases_list = "\n".join([f"**{disease.upper()}**" for disease, _ in probable_diseases])
            response_text = f"Based on your symptoms, possible conditions are:\n{diseases_list}"
            print(f"{diseases_list}")

            # Query for each of the top diseases
            diseases_list2 = []
            for i in range(len(probable_diseases)):
                diseases_list2.append(probable_diseases[i])
            guidance_text = "\n\n".join(
                [
                    f"**{disease.upper()}**:\n{info['description']}" 
                    for disease, _ in probable_diseases
                    if (info := retrieve_info_from_csv(disease))["disease"] is not None
                ]
            )

            return jsonify(response=f"{response_text}\n\n<span style='font-size: 24px;'><strong>Details For Each Disease:</strong></span>\n{guidance_text}")
            
    return jsonify(response="Could you provide more details about your symptoms?")

if __name__ == '__main__':
    app.run(debug=True)
