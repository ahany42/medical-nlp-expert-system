import pandas as pd
import spacy
from collections import Counter
import ast

# Load spaCy for NLP processing
nlp = spacy.load("en_core_web_sm")

# --- N-gram Processing Functions ---
def format_symptoms(symptoms):
    return symptoms.replace('_', ' ')

def preprocess_input(user_input):
    formatted_input = format_symptoms(user_input.lower())
    doc = nlp(formatted_input)
    remove_pos = ['PRON', 'ADP', 'AUX', 'CONJ', 'SCONJ', 'DET', 'PART']
    symptoms = [token.lemma_ for token in doc
                if token.is_alpha
                and not token.is_stop
                and token.pos_ not in remove_pos]
    return symptoms

def preprocess_input_with_ngrams(user_input, frequent_bigrams, frequent_trigrams):
    formatted_input = format_symptoms(user_input.lower())
    doc = nlp(formatted_input)
    remove_pos = ['PRON', 'ADP', 'AUX', 'CONJ', 'SCONJ', 'DET', 'PART']
    processed_tokens = [token.lemma_ for token in doc
                        if token.is_alpha
                        and not token.is_stop
                        and token.pos_ not in remove_pos]

    processed_symptoms = []
    i = 0
    while i < len(processed_tokens):
        found_ngram = False
        # Check for trigrams
        if i + 2 < len(processed_tokens):
            trigram = tuple(processed_tokens[i:i + 3])
            if trigram in frequent_trigrams:
                processed_symptoms.append('_'.join(trigram))
                i += 3
                found_ngram = True
        if found_ngram:
            continue
        # Check for bigrams
        if i + 1 < len(processed_tokens):
            bigram = tuple(processed_tokens[i:i + 2])
            if bigram in frequent_bigrams:
                processed_symptoms.append('_'.join(bigram))
                i += 2
                found_ngram = True
        if found_ngram:
            continue
        # If not part of a multi-word symptom, add the single word
        processed_symptoms.append(processed_tokens[i])
        i += 1
    return processed_symptoms

def load_disease_data_with_ngrams(df, frequent_bigrams, frequent_trigrams):
    disease_data = {}
    for _, row in df.iterrows():
        disease = row['disease']
        if isinstance(row['symptoms'], list):
            raw_symptoms = [s.strip() for s in row['symptoms']]
        else:
            raw_symptoms = [s.strip() for s in str(row['symptoms']).split(',')]

        processed_symptoms_with_ngrams = set()
        if frequent_bigrams and frequent_trigrams:
            # Join the raw symptoms into a single string for n-gram processing
            all_raw_symptoms_str = ' '.join(raw_symptoms)
            processed_symptoms_with_ngrams.update(
                preprocess_input_with_ngrams(
                    all_raw_symptoms_str,
                    frequent_bigrams,
                    frequent_trigrams
                )
            )
        else:
            # If no n-grams, apply basic preprocessing to individual symptoms
            for s in raw_symptoms:
                processed_symptoms_with_ngrams.update(preprocess_input(s))

        precautions = [p.strip() for p in str(row['precautions']).split(',') if p]

        if disease not in disease_data:
            disease_data[disease] = {'symptoms': set(), 'precautions': set()}
        disease_data[disease]['symptoms'].update(processed_symptoms_with_ngrams)
        disease_data[disease]['precautions'].update(precautions)

    for disease_entry in disease_data:
        disease_data[disease_entry]['symptoms'] = sorted(disease_data[disease_entry]['symptoms'])
        disease_data[disease_entry]['precautions'] = sorted(disease_data[disease_entry]['precautions'])

    return disease_data

if __name__ == "__main__":
    # Load your disease data
    data = pd.read_csv('Medical Diagnosis Expert System.csv')
    df = data.copy()

    # --- N-gram Processing ---
    df['processed_symptoms'] = df['symptoms'].apply(preprocess_input)

    # Identify potential multi-word symptoms
    all_processed_symptoms = [s for sublist in df['processed_symptoms'] for s in sublist]
    bigrams = [tuple(all_processed_symptoms[i:i + 2]) for i in range(len(all_processed_symptoms) - 1)]
    trigrams = [tuple(all_processed_symptoms[i:i + 3]) for i in range(len(all_processed_symptoms) - 2)]

    frequent_bigrams = Counter(bigrams).most_common(10)  # Adjust number as needed
    frequent_trigrams = Counter(trigrams).most_common(10)  # Adjust number as needed

    frequent_bigram_tuples = [bg[0] for bg in frequent_bigrams]
    frequent_trigram_tuples = [tg[0] for tg in frequent_trigrams]

    # Load data with n-grams
    disease_data = load_disease_data_with_ngrams(df, frequent_bigram_tuples, frequent_trigram_tuples)

    # Print the formatted data
    print("Formatted Disease Data:")
    for disease, data in disease_data.items():
        print(f"\nDisease: {disease}")
        print(f"  Symptoms: {data['symptoms']}")
        print(f"  Precautions: {data['precautions']}")
