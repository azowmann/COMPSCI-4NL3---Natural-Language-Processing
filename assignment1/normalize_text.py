#importing packages
import string
import sys
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer, PorterStemmer

lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

#preprocess method
def preprocess_text(text, lowercase, stemming, lemmatization, remove_stopwords, custom_option):
    #whitespace splitting
    tokens = text.split()
    print("Raw tokens:", tokens)

    #options
    if lowercase:
        tokens = [token.lower() for token in tokens]

    tokens = [token.strip(string.punctuation) for token in tokens]
    print("Tokens after punctuation removal:", tokens)

    #stop_words generated from ChatGPT
    if remove_stopwords:
        stop_words = {"the", "and", "to", "of", "in", "a", "is", "it", "you", "that", "he", "she", "we", "they"}
        tokens = [token for token in tokens if token.lower() not in stop_words]

    if stemming:
        tokens = [ps.stem(token) for token in tokens]
        print("Tokens after stemming:", tokens)

    if lemmatization:
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        print("Tokens after lemmatization:", tokens)

    #custom option to remove short tokens (2 letters or less)
    if custom_option:
        tokens = [token for token in tokens if len(token) > 2]
        print("Tokens after custom option (remove short tokens):", tokens)

    print("Tokens after preprocessing:", tokens)

    return tokens

#function to obtain frequency of tokens
def count_tokens(tokens):
    token_counts = {}
    for token in tokens:
        if token not in token_counts:
            token_counts[token] = 1
        else:
            token_counts[token] += 1
    print("Token counts:", token_counts)
    return token_counts

#function to visualizze bar graph
def plot_token_counts(token_counts, top=25, bot=25):
    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)

    top_tokens = sorted_tokens[:top]
    bot_tokens = sorted_tokens[-bot:]

    print("\nTop 25 Most Frequent Tokens:")
    print(top_tokens)

    print("\nBottom 25 Least Frequent Tokens:")
    print(bot_tokens)

    combined_tokens = top_tokens + bot_tokens

    tokens, counts = zip(*combined_tokens)
    
    plt.figure(figsize=(12, 6))

    plt.bar(range(1, len(tokens) + 1), counts, width=0.8, color='skyblue')
  
    plt.yscale('log')

    plt.xticks(range(1, len(tokens) + 1), tokens, rotation=90, fontsize=10)

    plt.title('Token Frequency vs Rank (Log-Scale Y Axis)', fontsize=15)
    plt.xlabel('Rank of Token', fontsize=12)
    plt.ylabel('Frequency of Token (log scale)', fontsize=12)

    plt.grid(True, which="both", axis="y", linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

#main function
def main():
    if len(sys.argv) < 2:
        print("Error: No file path provided.")
        return

    file_path = sys.argv[1]
    
    lowercase = 'lowercase' in sys.argv
    stemming = 'stemming' in sys.argv
    lemmatization = 'lemmatization' in sys.argv
    remove_stopwords = 'remove_stopwords' in sys.argv
    custom_option = 'custom_option' in sys.argv

    try:
        with open(file_path, "r", encoding="utf8") as file:
            text = file.read()

        tokens = preprocess_text(
            text,
            lowercase=lowercase,
            stemming=stemming,
            lemmatization=lemmatization,
            remove_stopwords=remove_stopwords,
            custom_option=custom_option
        )

        token_counts = count_tokens(tokens)

        sorted_tokens = sorted(token_counts.items(), key=lambda x : x[1], reverse=True)
        for token, count in sorted_tokens:
            print(f"{token} {count}")

        plot_token_counts(token_counts)

        raw_tokens = text.split()
        print(f"\nTotal number of tokens before preprocessing: {len(raw_tokens)}")

    #error handling
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
