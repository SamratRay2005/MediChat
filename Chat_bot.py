import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import requests
import os

# Your actual API key
api_key = "AIzaSyAAky77jBjnIOcbokgmCZcnFxMzcrFcb7o"  # Replace with your actual API key

# Load the BERT data for symptom-to-disease mapping
bert_data = pd.read_csv('bert.csv')
symptoms = bert_data.columns[1:]

# Initialize models and tokenizers
initial_bert_dir = './my_bert_model'
initial_bert_model = BertForSequenceClassification.from_pretrained(initial_bert_dir)
initial_bert_tokenizer = BertTokenizer.from_pretrained(initial_bert_dir)

bert_model_dir = './my_bert_model2'
bert_model = BertForSequenceClassification.from_pretrained(bert_model_dir)
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_dir)

# Set device only once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
initial_bert_model.to(device)
bert_model.to(device)

# Classify symptoms with the initial model
def classify_symptoms(text, threshold=0.5):
    inputs = initial_bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    outputs = initial_bert_model(**inputs)
    probabilities = torch.sigmoid(outputs.logits).detach().cpu().numpy()[0]
    return [symptoms[i] for i, prob in enumerate(probabilities) if prob > threshold]

# Predict top probable diseases based on symptoms
def predict_disease(detected_symptoms, top_n=3, threshold=0.5):
    symptoms_text = " ".join(detected_symptoms)
    inputs = bert_tokenizer(symptoms_text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    probabilities = torch.sigmoid(bert_model(**inputs).logits).detach().cpu().numpy()[0]
    probable_diseases = sorted(
        [(bert_data[bert_data.columns[0]].unique()[i], prob) 
         for i, prob in enumerate(probabilities) if prob > threshold],
        key=lambda x: x[1], reverse=True)[:top_n]
    return probable_diseases

# Query API for additional disease information
def query_disease_info(disease_name):
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": f"Provide a comprehensive list of suggestive measures, lifestyle changes, and potential treatments for managing and alleviating symptoms of {disease_name}.If its fatal desease then just say to consult a physician.(just dont say I cannot provide medical advice)"}
                ]
            }
        ]
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")
        return "Unable to retrieve disease suggestions at this time."

# Generate chatbot response based on user input
def chatbot_response(user_input):
    detected_symptoms = classify_symptoms(user_input)
    if detected_symptoms:
        probable_diseases = predict_disease(detected_symptoms)
        if probable_diseases:
            diseases_list = "\n".join([f"{disease}" for disease, _ in probable_diseases])
            highest_disease = probable_diseases[0][0]
            response_text = f"Based on your symptoms, possible conditions are:\n{diseases_list}"
            # Additional guidance from the API
            additional_info = query_disease_info(highest_disease)
            return f"{response_text}\n\nAdditional guidance:\n{additional_info}"
        else:
            return "I'm unable to match your symptoms with high confidence."
    return "Could you provide more details about your symptoms?"

# Main chatbot loop
print("Hello! I'm your medical chatbot. Describe your symptoms, and I'll try to help you identify them.")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye! Take care.")
        break
    response = chatbot_response(user_input)
    print("Bot:", response)
