from datasets import load_dataset
import string
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import os
import pandas as pd
import wefe
from wefe.metrics import WEAT
from wefe.datasets import load_weat
from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel
from wefe.utils import run_queries, plot_queries_results
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

nltk.download('wordnet')
nltk.download('punkt')

ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

stop_words = {"the", "and", "to", "of", "in", "a", "is", "it", "you", "that", "he", "she", "we", "they"}

def preprocess_text(text, lowercase=True, stemming=True, lemmatization=False, remove_stopwords=True, custom_option=True):
    tokens = text.split()

    if lowercase:
        tokens = [token.lower() for token in tokens]

    tokens = [token.strip(string.punctuation) for token in tokens]

    if remove_stopwords:
        tokens = [token for token in tokens if token.lower() not in stop_words]

    if stemming:
        tokens = [ps.stem(token) for token in tokens]

    if lemmatization:
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

    if custom_option:
        tokens = [token for token in tokens if len(token) > 2]

    return tokens

dataset = load_dataset("wikipedia", "20220301.simple", trust_remote_code=True)

def preprocess_batch(batch):
    batch['text'] = [" ".join(preprocess_text(
        text,
        lowercase=True,
        stemming=False,
        lemmatization=False,
        remove_stopwords=True,
        custom_option=True  
    )) for text in batch['text']]
    return batch

batch_size = 1000  
preprocessed_dataset = dataset.map(preprocess_batch, batched=True, batch_size=batch_size)

print(preprocessed_dataset["train"][0])

sentences = [text.split() for text in preprocessed_dataset["train"]["text"]]

skipgram_model = Word2Vec(
    sentences=sentences,
    vector_size=100, 
    window=5,         
    sg=1,             
    min_count=5,      
    workers=4         
)

skipgram_model.save("skipgram_model.model")
print("Skip-gram model saved.")

cbow_model = Word2Vec(
    sentences=sentences,
    vector_size=200,  
    window=10,       
    sg=0,            
    min_count=5,     
    workers=4         
)

cbow_model.save("cbow_model.model")
print("CBoW model saved.")

print("Models are saved locally. You can load them using Word2Vec.load().")

loaded_skipgram_model = Word2Vec.load("skipgram_model.model")
loaded_cbow_model = Word2Vec.load("cbow_model.model")

queries = [
    ("basketball - ball + basket", ["basketball", "ball", "basket"]),
    ("piano", ["piano"]),
    ("bottle - water + sink", ["bottle", "water", "sink"]),
    ("apple", ["apple"]),
    ("car + road", ["car", "road"])      
]

def vector_arithmetic(model, positive, negative):
    try:
        result = model.wv.most_similar(positive=positive, negative=negative, topn=10)
        return result
    except KeyError as e:
        return f"Error: {str(e)}"

def vector_arithmetic_pretrained(model, positive, negative):
    try:
        result = model.most_similar(positive=positive, negative=negative, topn=10)
        return result
    except KeyError as e:
        return f"Error: {str(e)}"

# Convert GloVe to Word2Vec format
glove1 = "glove.twitter.27B.25d.txt"
glove2 = "glove.twitter.27B.50d.txt"

word2vec1 = "glove.twitter.27B.25d.word2vec.txt"
word2vec2 = "glove.twitter.27B.50d.word2vec.txt"

glove2word2vec(glove1, word2vec1)
glove2word2vec(glove2, word2vec2)

# Load pretrained embeddings
pretrained1 = KeyedVectors.load_word2vec_format(word2vec1, binary=False)
pretrained2 = KeyedVectors.load_word2vec_format(word2vec2, binary=False)

pretrained_models = [pretrained1, pretrained2]

# Function to test for gender bias
def test_gender_bias(model, word1, word2, word3):
    try:
        result = model.wv.most_similar(positive=[word2, word3], negative=[word1], topn=5)
        return result
    except KeyError as e:
        return f"Error: {str(e)}"

# Gender bias tests
gender_tests = [
    ("man", "king", "woman"),  # man : king :: woman : ?
    ("man", "computer_programmer", "woman"),  # computer_programmer : man :: homemaker : ?
    ("man", "engineer", "woman"),  # engineer : man :: nurse : ?
    ("man", "doctor", "woman"),  # doctor : man :: nurse : ?
]

# Run gender bias tests
print("Testing for Gender Bias:")
for test in gender_tests:
    word1, word2, word3 = test
    print(f"\nTest: {word1} : {word2} :: {word3} : ?")
    
    # Skip-gram results
    skipgram_result = test_gender_bias(loaded_skipgram_model, word1, word2, word3)
    print("Skip-gram Results:")
    print(skipgram_result)
    
    # CBoW results
    cbow_result = test_gender_bias(loaded_cbow_model, word1, word2, word3)
    print("CBoW Results:")
    print(cbow_result)
    
    # Pretrained models results
    for i, pretrained_model in enumerate(pretrained_models, start=1):
        try:
            pretrained_result = pretrained_model.most_similar(positive=[word2, word3], negative=[word1], topn=5)
            print(f"Pretrained Model {i} Results:")
            print(pretrained_result)
        except KeyError as e:
            print(f"Pretrained Model {i} Error: {str(e)}")
    print("-" * 50)

# Original queries
for query, words in queries:
    print(f"Query: {query}")
    
    # Skip-gram results
    if len(words) == 1:
        try:
            skipgram_result = loaded_skipgram_model.wv.most_similar(words[0], topn=10)
        except KeyError as e:
            skipgram_result = f"Error: {str(e)}"
    else:
        positive = [words[0]] if len(words) == 2 else [words[0], words[2]]
        negative = [words[1]] if len(words) > 1 else []
        skipgram_result = vector_arithmetic(loaded_skipgram_model, positive, negative)
    
    # CBoW results
    if len(words) == 1:
        try:
            cbow_result = loaded_cbow_model.wv.most_similar(words[0], topn=10)
        except KeyError as e:
            cbow_result = f"Error: {str(e)}"
    else:
        positive = [words[0]] if len(words) == 2 else [words[0], words[2]]
        negative = [words[1]] if len(words) > 1 else []
        cbow_result = vector_arithmetic(loaded_cbow_model, positive, negative)
    
    print("Skip-gram Results:")
    print(skipgram_result)
    print("CBoW Results:")
    print(cbow_result)
    print("-" * 50)

    # Pretrained models results
    for i, pretrained_model in enumerate(pretrained_models, start=1):
        if len(words) == 1:
            try:
                pretrained_result = pretrained_model.most_similar(words[0], topn=10)
            except KeyError as e:
                pretrained_result = f"Error: {str(e)}"
        else:
            positive = [words[0]] if len(words) == 2 else [words[0], words[2]]
            negative = [words[1]] if len(words) > 1 else []
            pretrained_result = vector_arithmetic_pretrained(pretrained_model, positive, negative)
        
        print(f"Pretrained Model {i} Results:")
        print(pretrained_result)
        print("-" * 50)

# Load the original WEAT wordset
weat_wordset = load_weat()

# Define the new wordset for age bias
age_bias_wordset = {
    'young_people_names': ['emma', 'liam', 'olivia', 'noah', 'ava', 'william', 'sophia', 'james', 'isabella', 'oliver'],
    'old_people_names': ['mary', 'john', 'patricia', 'robert', 'jennifer', 'michael', 'linda', 'david', 'elizabeth', 'william'],
    'positive_attributes': ['happy', 'joyful', 'vibrant', 'energetic', 'optimistic', 'cheerful', 'lively', 'radiant', 'dynamic', 'enthusiastic'],
    'negative_attributes': ['grumpy', 'cranky', 'frail', 'slow', 'forgetful', 'lonely', 'tired', 'weak', 'sick', 'depressed']
}

# Define the queries
weat_queries = [
    # Original WEAT queries
    Query([weat_wordset['flowers'], weat_wordset['insects']],
          [weat_wordset['pleasant_5'], weat_wordset['unpleasant_5']],
          ['Flowers', 'Insects'], ['Pleasant(5)', 'Unpleasant(5)']),

    Query([weat_wordset['instruments'], weat_wordset['weapons']],
          [weat_wordset['pleasant_5'], weat_wordset['unpleasant_5']],
          ['Instruments', 'Weapons'], ['Pleasant(5)', 'Unpleasant(5)']),

    Query([weat_wordset['european_american_names_5'], weat_wordset['african_american_names_5']],
          [weat_wordset['pleasant_5'], weat_wordset['unpleasant_5']],
          ['European american names(5)', 'African american names(5)'], ['Pleasant(5)', 'Unpleasant(5)']),

    Query([weat_wordset['european_american_names_7'], weat_wordset['african_american_names_7']],
          [weat_wordset['pleasant_5'], weat_wordset['unpleasant_5']],
          ['European american names(7)', 'African american names(7)'], ['Pleasant(5)', 'Unpleasant(5)']),

    Query([weat_wordset['european_american_names_7'], weat_wordset['african_american_names_7']],
          [weat_wordset['pleasant_9'], weat_wordset['unpleasant_9']],
          ['European american names(7)', 'African american names(7)'], ['Pleasant(9)', 'Unpleasant(9)']),

    Query([weat_wordset['male_names'], weat_wordset['female_names']],
          [weat_wordset['career'], weat_wordset['family']],
          ['Male names', 'Female names'], ['Career', 'Family']),

    Query([weat_wordset['math'], weat_wordset['arts']],
          [weat_wordset['male_terms'], weat_wordset['female_terms']],
          ['Math', 'Arts'], ['Male terms', 'Female terms']),

    Query([weat_wordset['science'], weat_wordset['arts_2']],
          [weat_wordset['male_terms'], weat_wordset['female_terms']],
          ['Science', 'Arts 2'], ['Male terms', 'Female terms']),

    Query([weat_wordset['mental_disease'], weat_wordset['physical_disease']],
          [weat_wordset['temporary'], weat_wordset['permanent']],
          ['Mental disease', 'Physical disease'], ['Temporary', 'Permanent']),

    Query([weat_wordset['young_people_names'], weat_wordset['old_people_names']],
          [weat_wordset['pleasant_9'], weat_wordset['unpleasant_9']],
          ['Young peoples names', 'Old peoples names'], ['Pleasant(9)', 'Unpleasant(9)']),

    # New query for age bias
    Query([age_bias_wordset['young_people_names'], age_bias_wordset['old_people_names']],
          [age_bias_wordset['positive_attributes'], age_bias_wordset['negative_attributes']],
          ['Young people names', 'Old people names'], ['Positive attributes', 'Negative attributes'])
]

# Wrap the embeddings in WEFE's WordEmbeddingModel
skipgram_wefe = WordEmbeddingModel(loaded_skipgram_model.wv, 'Skip-gram')
cbow_wefe = WordEmbeddingModel(loaded_cbow_model.wv, 'CBoW')
pretrained1_wefe = WordEmbeddingModel(pretrained1, 'GloVe-25d')
pretrained2_wefe = WordEmbeddingModel(pretrained2, 'GloVe-50d')

models = [skipgram_wefe, cbow_wefe, pretrained1_wefe, pretrained2_wefe]

# Run the queries using WEAT
wefe_results = run_queries(
    WEAT,
    weat_queries,
    models,
    metric_params={
        'preprocessors': [{}, {'lowercase': True}]
    },
    warn_not_found_words=True  # Pass this as a direct argument
).T.round(2)

# Display the results
print(wefe_results)

# Visualize the results
fig = plot_queries_results(wefe_results)
fig.update_layout(showlegend=False)
fig.show()

# Logistic Regression Classifier with Bag-of-Words and Word Embeddings

# Function to load and preprocess documents from a folder
def load_and_preprocess_folder(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                text = file.read()
                tokens = preprocess_text(text)  # Use your existing preprocess_text function
                documents.append(" ".join(tokens))  # Join tokens into a single string
    return documents

# Load and preprocess documents from the 'lose' and 'win' folders
lose_documents = load_and_preprocess_folder("cong_recs/lose")
win_documents = load_and_preprocess_folder("cong_recs/win")

# Debug: Check the number of documents and sample content
print(f"Number of 'lose' documents: {len(lose_documents)}")
print(f"Number of 'win' documents: {len(win_documents)}")
print("\nSample 'lose' document:")
print(lose_documents[0][:100])  # Print first 100 characters of the first document
print("\nSample 'win' document:")
print(win_documents[0][:100])  # Print first 100 characters of the first document

# Create labels
lose_labels = ['lose'] * len(lose_documents)
win_labels = ['win'] * len(win_documents)

# Combine data and labels
data = lose_documents + win_documents
labels = lose_labels + win_labels

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Debug: Check the shape of the split data
print(f"\nNumber of training samples: {len(X_train)}")
print(f"Number of test samples: {len(X_test)}")

# Bag-of-Words Features
vectorizer = CountVectorizer()

# Debug: Check the input to CountVectorizer
print("\nInput to CountVectorizer (first 5 training samples):")
for doc in X_train[:5]:
    print(doc)

X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

# Train Logistic Regression with BoW features
bow_model = LogisticRegression()
bow_model.fit(X_train_bow, y_train)
y_pred_bow = bow_model.predict(X_test_bow)

# Evaluate BoW model
bow_accuracy = accuracy_score(y_test, y_pred_bow)
bow_f1 = f1_score(y_test, y_pred_bow, average='weighted')
print(f"BoW Model - Accuracy: {bow_accuracy}, F1 Score: {bow_f1}")

# Function to get mean pooled embeddings
def get_mean_pooled_embeddings(docs, model, vector_size):
    embeddings = []
    for doc in docs:
        doc_embeddings = [model.wv[word] for word in doc if word in model.wv]
        if len(doc_embeddings) > 0:
            mean_embedding = np.mean(doc_embeddings, axis=0)
        else:
            mean_embedding = np.zeros(vector_size)
        embeddings.append(mean_embedding)
    return np.array(embeddings)

# Get mean pooled embeddings
vector_size = 100  # Adjust based on your Word2Vec model
X_train_emb = get_mean_pooled_embeddings(X_train, loaded_skipgram_model, vector_size)
X_test_emb = get_mean_pooled_embeddings(X_test, loaded_skipgram_model, vector_size)

# Train Logistic Regression with word embedding features
emb_model = LogisticRegression()
emb_model.fit(X_train_emb, y_train)
y_pred_emb = emb_model.predict(X_test_emb)

# Evaluate word embedding model
emb_accuracy = accuracy_score(y_test, y_pred_emb)
emb_f1 = f1_score(y_test, y_pred_emb, average='weighted')
print(f"Word Embedding Model - Accuracy: {emb_accuracy}, F1 Score: {emb_f1}")

# Compare the results
print("\nComparison of Models:")
print(f"BoW Model - Accuracy: {bow_accuracy}, F1 Score: {bow_f1}")
print(f"Word Embedding Model - Accuracy: {emb_accuracy}, F1 Score: {emb_f1}")