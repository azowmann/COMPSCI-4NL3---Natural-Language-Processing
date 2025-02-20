
import os
import string
import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import pos_tag  
from collections import defaultdict
import gensim
import gensim.corpora as corpora
from gensim.models import TfidfModel 
import pyLDAvis.gensim

lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

# Function to combine text files
def combine_text_files(input_folder, output_file):
    try:
        with open(output_file, 'w') as outfile:
            for filename in os.listdir(input_folder):
                if filename.endswith('.txt'):
                    file_path = os.path.join(input_folder, filename)
                    with open(file_path, 'r') as infile:
                        outfile.write(infile.read())
                        outfile.write('\n')
        print(f"All text files combined into '{output_file}'")
    except Exception as e:
        print(f"Error: {e}")

# Function to check file contents
def check_file_contents(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            print(f"Contents of '{file_path}':\n{content}\n")
            return bool(content.strip())
    except Exception as e:
        print(f"Error reading '{file_path}': {e}")
        return False

# Function to preprocess text
def preprocess_text(text, lowercase=True, stemming=False, lemmatization=False, remove_stopwords=True, custom_option1=True, custom_option2=True, pos_filtering=False):
    tokens = text.split()
    
    if lowercase:
        tokens = [token.lower() for token in tokens]
    tokens = [token.strip(string.punctuation) for token in tokens]
    
    if remove_stopwords:
        stop_words = {"the", "and", "to", "of", "in", "a", "is", "it", "you", "that", "he", "she", "we", "they"}
        tokens = [token for token in tokens if token.lower() not in stop_words]
    
    if pos_filtering:
        # Keep only nouns and verbs
        tagged_tokens = pos_tag(tokens)
        tokens = [word for word, pos in tagged_tokens if pos.startswith('NN') or pos.startswith('VB')]
    
    if stemming:
        tokens = [ps.stem(token) for token in tokens]
    
    if lemmatization:
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    if custom_option1:
        obvious = {"win", "won", "lose", "lost", "loss", "2023", "2020", "20A24", "2025", "https"}
        tokens = [token for token in tokens if token.lower() not in obvious]

    if custom_option2:
        tokens = [token for token in tokens if len(token) > 2]
    
    return tokens

# Function to preprocess a file
def preprocess_file(input_file, pos_filtering=False):
    try:
        with open(input_file, 'r') as infile:
            text = infile.read()
            processed_tokens = preprocess_text(text, pos_filtering=pos_filtering)
            return processed_tokens
    except Exception as e:
        print(f"Error: {e}")
        return []

# Function to count words
def count_words(data):
    word_counts = defaultdict(lambda: defaultdict(int))
    total_words = defaultdict(int)
    for words, label in data:
        for word in words:
            word_counts[label][word] += 1
            total_words[label] += 1
    
    vocab = set(word for words, _ in data for word in words)
    return word_counts, total_words, vocab

# Function to calculate probability
def calculate_probability(word, word_counts, total_words, vocab_size, alpha=1):
    return (word_counts[word] + alpha) / (total_words + alpha * vocab_size)

# Function to compute log-likelihood
def compute_log_likelihood(word_counts, total_words, vocab, alpha=1):
    log_likelihoods = defaultdict(dict)

    for c in word_counts:
        for word in vocab:
            p_wc = calculate_probability(word, word_counts[c], total_words[c], len(vocab), alpha)
            p_wCo = sum(
                calculate_probability(word, word_counts[co], total_words[co], len(vocab), alpha)
                for co in word_counts if co != c
            ) / (len(word_counts) - 1)

            if p_wCo == 0:
                log_likelihoods[c][word] = np.log(p_wc + 1e-9)
            else:
                log_likelihoods[c][word] = np.log(p_wc) - np.log(p_wCo)

    return log_likelihoods

# Function to print top words
def print_top_words(log_likelihoods, top_n=10):
    for label, words in log_likelihoods.items():
        sorted_words = sorted(words.items(), key=lambda x: x[1], reverse=True)[:top_n]
        print(f"\nTop {top_n} words for class '{label}':")
        for word, score in sorted_words:
            print(f"{word}: {score:.4f}")

def extract_top_words(lda_model, num_words=25):
    topics = lda_model.show_topics(num_topics=-1, num_words=num_words, formatted=False)
    topic_data = []
    
    for topic_id, words in topics:
        topic_label = f"Topic {topic_id}" 
        word_list = [f"{word} ({prob:.4f})" for word, prob in words]
        topic_data.append([topic_label] + word_list)
    
    return topic_data

# Function to get the average topic distribution for a category
def get_average_topic_distribution(lda_model, corpus, category_docs):

    topic_sums = defaultdict(float)  # Sum of probabilities for each topic
    num_docs = len(category_docs)  # Number of documents in the category

    for doc in category_docs:
        # Get the topic distribution for the document
        doc_bow = dictionary.doc2bow(doc)  # Convert document to BoW format
        doc_topics = lda_model.get_document_topics(doc_bow)  # Get topic distribution

        for topic_id, prob in doc_topics:
            topic_sums[topic_id] += prob

    # Calculate the average probability for each topic
    avg_topic_dist = {topic_id: prob / num_docs for topic_id, prob in topic_sums.items()}
    return avg_topic_dist

def get_average_topic_distribution(lda_model, corpus, category_docs):

    topic_sums = defaultdict(float)  # Sum of probabilities for each topic
    num_docs = len(category_docs)  # Number of documents in the category

    for doc in category_docs:
        # Ensure `doc` is a list of tokens
        if isinstance(doc, str):
            doc = doc.split()  # Split the string into tokens if it's not already tokenized

        # Get the topic distribution for the document
        doc_bow = dictionary.doc2bow(doc)  # Convert document to BoW format
        doc_topics = lda_model.get_document_topics(doc_bow)  # Get topic distribution

        # Sum the probabilities for each topic
        for topic_id, prob in doc_topics:
            topic_sums[topic_id] += prob

    # Calculate the average probability for each topic
    avg_topic_dist = {topic_id: prob / num_docs for topic_id, prob in topic_sums.items()}
    return avg_topic_dist

# Function to report the top N topics for a category
def report_top_topics(avg_topic_dist, top_n=3):
    sorted_topics = sorted(avg_topic_dist.items(), key=lambda x: x[1], reverse=True)
    return sorted_topics[:top_n]

# Main script
if __name__ == '__main__':
    # Define input and output paths
    input_folderl = "cong_recs/lose"
    output_filel = "loset.txt"
    input_folderw = "cong_recs/win"
    output_filew = "wint.txt"

    # Combine text files
    combine_text_files(input_folderl, output_filel)
    combine_text_files(input_folderw, output_filew)

    # Check if input files are empty
    if not check_file_contents(output_filel):
        print(f"Warning: '{output_filel}' is empty or could not be read!")
    if not check_file_contents(output_filew):
        print(f"Warning: '{output_filew}' is empty or could not be read!")

    # Preprocess files with and without POS filtering
    print("Preprocessing without POS filtering...")
    loset_tokens = preprocess_file(output_filel, pos_filtering=False)
    wint_tokens = preprocess_file(output_filew, pos_filtering=False)

    print("Preprocessing with POS filtering...")
    loset_tokens_pos = preprocess_file(output_filel, pos_filtering=True)
    wint_tokens_pos = preprocess_file(output_filew, pos_filtering=True)

    # Save processed tokens to files
    with open("processed_loset.txt", "w") as loset_outfile:
        loset_outfile.write(" ".join(loset_tokens))

    with open("processed_wint.txt", "w") as wint_outfile:
        wint_outfile.write(" ".join(wint_tokens))

    with open("processed_loset_pos.txt", "w") as loset_outfile_pos:
        loset_outfile_pos.write(" ".join(loset_tokens_pos))

    with open("processed_wint_pos.txt", "w") as wint_outfile_pos:
        wint_outfile_pos.write(" ".join(wint_tokens_pos))

    # Create dictionary and corpus for count-based representation
    processed_texts = [loset_tokens, wint_tokens]
    dictionary = corpora.Dictionary(processed_texts)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]

    # Verify the dictionary and corpus
    print("Dictionary size:", len(dictionary))
    print("Sample dictionary items:", list(dictionary.items())[:10])
    print("Sample corpus (bag-of-words):", corpus[:5])

    # Create TF-IDF representation
    tfidf = TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    # Verify the TF-IDF corpus
    print("Sample TF-IDF corpus:", list(corpus_tfidf)[:5])

    # Run LDA topic modeling on count-based representation
    print("\nRunning LDA on count-based representation...")
    num_topics = 3
    lda_model = gensim.models.LdaMulticore(corpus, num_topics=num_topics, id2word=dictionary, passes=10, workers=2)

    # Run LDA topic modeling on TF-IDF representation
    print("\nRunning LDA on TF-IDF representation...")
    lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=num_topics, id2word=dictionary, passes=30, workers=2)

    # Visualize topics using pyLDAvis for count-based representation
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis, 'lda_visualization_count.html')
    print("LDA visualization for count-based representation saved to 'lda_visualization_count.html'")

    # Visualize topics using pyLDAvis for TF-IDF representation
    vis_tfidf = pyLDAvis.gensim.prepare(lda_model_tfidf, corpus_tfidf, dictionary)
    pyLDAvis.save_html(vis_tfidf, 'lda_visualization_tfidf.html')
    print("LDA visualization for TF-IDF representation saved to 'lda_visualization_tfidf.html'")

    # Print topics for count-based representation
    print("\nTopics from count-based representation:")
    topics = lda_model.print_topics(num_words=10)
    for idx, topic in topics:
        print(f"Topic {idx}: {topic}")

    # Print topics for TF-IDF representation
    print("\nTopics from TF-IDF representation:")
    topics_tfidf = lda_model_tfidf.print_topics(num_words=10)
    for idx, topic in topics_tfidf:
        print(f"Topic {idx}: {topic}")

    # Prepare labeled data for Naive Bayes
    labeled_data = [
        (loset_tokens, 'lose'),
        (wint_tokens, 'win')
    ]

    # Count words and compute log likelihood
    word_counts, total_words, vocab = count_words(labeled_data)
    log_likelihoods = compute_log_likelihood(word_counts, total_words, vocab)

    # Extract top words from LDA model (count-based)
    topic_table = extract_top_words(lda_model)

    # Convert to DataFrame for better visualization
    df_topics = pd.DataFrame(topic_table, columns=["Topic Label"] + [f"Word {i+1}" for i in range(25)])

    df_topics["Topic Label"] = ["slightly more political", "slightly less political", "presidential"]

    # Save to CSV
    df_topics.to_csv("lda_topics.csv", index=False)
    df_topics.to_csv("lda_topics_count.csv", index=False)
    print("Topics from count-based representation saved to 'lda_topics.csv'")

    # Extract top words from LDA model (TF-IDF)
    topic_table_tfidf = extract_top_words(lda_model_tfidf)

    # Convert to DataFrame for better visualization
    df_topics_tfidf = pd.DataFrame(topic_table_tfidf, columns=["Topic Label"] + [f"Word {i+1}" for i in range(25)])

    df_topics_tfidf["Topic Label"] = ["bad", "worse", "almost same as bad"]

    # Save to CSV
    df_topics_tfidf.to_csv("lda_topics_tfidf.csv", index=False)
    print("Topics from TF-IDF representation saved to 'lda_topics_tfidf.csv'")

    # Display tables
    print("\nTopics from count-based representation:")
    print(df_topics)

    print("\nTopics from TF-IDF representation:")
    print(df_topics_tfidf)

    # Print top words for each class
    print("\nTop words for each class (log-likelihood ratios):")
    print_top_words(log_likelihoods)

    # Calculate average topic distribution for each category
    avg_topic_dist_lose = get_average_topic_distribution(lda_model, corpus, loset_tokens)
    avg_topic_dist_win = get_average_topic_distribution(lda_model, corpus, wint_tokens)

    # Report the top 3-5 topics for each category
    top_topics_lose = report_top_topics(avg_topic_dist_lose, top_n=3)
    top_topics_win = report_top_topics(avg_topic_dist_win, top_n=3)

    # Print the results
    print("\nTop 3 topics for 'lose' category:")
    for topic_id, avg_prob in top_topics_lose:
        print(f"Topic {topic_id}: {avg_prob:.4f}")

    print("\nTop 3 topics for 'win' category:")
    for topic_id, avg_prob in top_topics_win:
        print(f"Topic {topic_id}: {avg_prob:.4f}")