# 🩺 Medical Diagnosis Expert System

An intelligent expert system that diagnoses common diseases based on user-described symptoms using rule-based reasoning (via **Experta**) and Natural Language Processing (via **spaCy** and **NLTK**). The system also provides health precautions for each diagnosed disease.

---

## 📌 Description

This system simulates a basic medical diagnostic process. Users can describe their symptoms in natural language, and the system will:

1. **Understand** the input using NLP (tokenization, lemmatization, stemming).
2. **Match** the extracted symptoms with predefined rules (using **Experta**).
3. **Diagnose** the most likely disease.
4. **Recommend** suitable precautions for the diagnosed disease.

---

## 🧠 Knowledge Representation

The system uses the [**Experta**](https://github.com/nbrooks/experta) rule-based engine (based on CLIPS) to model relationships between diseases and their symptoms. Each disease is defined as a rule that is triggered when a sufficient number of its associated symptoms are identified in the user's input.

---

## 🧪 Natural Language Processing

NLP preprocessing is performed using:

- **spaCy** for:
  - Tokenization
  - Lemmatization

- **NLTK** for:
  - Stemming
  - Stopword removal
  - Symptom keyword matching

This combination ensures robust handling of user-provided natural language inputs.

---

## 🗂️ Dataset

The system uses a CSV file with the following structure:

| Column      | Description                                                |
|-------------|------------------------------------------------------------|
| `Disease`     | Name of the disease                                        |
| `Symptoms`    | Comma-separated symptoms for the disease                  |
| `Precautions` | Comma-separated list of relevant precautions              |

---

## ▶️ Run Example

### **User Input**:
I have itching and skin rash.

markdown
Copy
Edit

### **System Output**:
Diagnosis:
Based on the symptoms you've provided, it seems you might have Fungal Infection.

Precautions:

Bath twice daily.

Use Dettol or neem in your bathing water.

Keep the infected area dry.

yaml
Copy
Edit

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/medical-diagnosis-expert-system.git
cd medical-diagnosis-expert-system
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
Example requirements.txt:

nginx
Copy
Edit
experta
nltk
spacy
pandas
Make sure to download necessary NLTK and spaCy resources:

python
Copy
Edit
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# For spaCy
import spacy
spacy.cli.download("en_core_web_sm")
3. Run the System
bash
Copy
Edit
python main.py
📦 Tech Stack
🧠 Experta – rule-based expert system

🧬 spaCy – NLP preprocessing

🧪 NLTK – text analysis (stemming, stopwords)

📊 Pandas – data handling

