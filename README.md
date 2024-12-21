# MediChat

MediChat is an AI-powered medical chatbot designed to assist users by providing information on various medical conditions based on user-inputted symptoms. It leverages advanced machine learning models and Retrieval-Augmented Generation (RAG) for retrieving accurate and reliable responses about diseases and treatments.

## Features

- **Symptom Analysis**: Identifies potential medical conditions based on user-described symptoms.
- **Disease Information Retrieval**: Provides detailed descriptions of diseases, including possible treatments, using RAG.
- **Interactive Chat Interface**: Engages users in a conversational manner to gather symptom information and deliver responses.
- **Retrieval-Augmented Generation (RAG)**: Enhances responses by combining FAISS-based similarity search with advanced generative capabilities for detailed and context-aware answers.

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/SamratRay2005/MediChat.git
   cd MediChat
   ```

2. **Set Up a Virtual Environment** (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:

   Ensure you have [PyTorch](https://pytorch.org/get-started/locally/) installed with CUDA support if you plan to use a GPU.

   ```bash
   pip install -r requirements.txt
   ```

4. **Download Necessary Models and Data**:

   - **BERT Models**: Download and place the BERT models in the `./models/my_bert_model` and `./models/my_bert_model2` directories.
   - **FAISS Index**: Ensure the FAISS index file is located at `faiss_index/index.faiss`.
   - **Disease Information Data**: Place the `Disease_Info.csv` file in the project root directory.

   > **Note**: You can download the dataset used for training from [Kaggle](https://www.kaggle.com/datasets/samratray05/berttraining).

## Usage

1. **Run the Flask Application**:

   ```bash
   python app.py
   ```

2. **Access the Application**:

   Open your web browser and navigate to `http://127.0.0.1:5000/` to interact with MediChat.

## Retrieval-Augmented Generation (RAG) for Disease Data

MediChat uses a RAG approach to enhance its responses:

1. **Symptom Classification**:
   Symptoms are classified using a fine-tuned BERT model, which predicts the relevant symptoms based on user input.

2. **Disease Retrieval via FAISS**:
   FAISS (Facebook AI Similarity Search) is used to find the most relevant disease descriptions from the preprocessed dataset. A query embedding is created using Sentence Transformers, and the closest match is retrieved from the FAISS index.

3. **Generative Responses**:
   The retrieved data is combined with generative capabilities to provide context-aware and detailed answers about the disease and its treatments.

## Project Structure

```
MediChat/
├── app.py                     # Main Flask application
├── requirements.txt           # Python dependencies
├── models/
│   ├── my_bert_model/         # Initial BERT model directory
│   └── my_bert_model2/        # Secondary BERT model directory
├── faiss_index/
│   └── index.faiss            # FAISS index file
│   └── index.pkl              # FAISS pikl file
├── Disease_Info.csv           # Disease information dataset
├── static/                    # Static files (CSS, JS)
└── templates/
    └── index.html             # HTML template for the web interface
```

## Used Dataset

This project uses the following dataset for training:

- [BERT Training Dataset](https://www.kaggle.com/datasets/samratray05/berttraining)
- [Disease and Description Training Dataset](https://www.kaggle.com/datasets/samratray05/traningtemporarydiseasefile)

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Kaggle Dataset: BERT Training](https://www.kaggle.com/datasets/samratray05/berttraining)
- [FAISS - A library for efficient similarity search](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
- [RAG: Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)

---

*Note: This README provides a general overview of the MediChat project. For detailed information on specific modules or functions, please refer to the source code and inline comments.*

