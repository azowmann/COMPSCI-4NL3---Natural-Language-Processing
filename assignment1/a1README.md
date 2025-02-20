[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/axpepi8Q)

# Homework 1 Counting Tokens

## Description
##### This Python script performs text preprocessing on an input text file and visualizes the frequency distribution of tokens (words) in the text. The script allows users to configure various preprocessing steps such as lowercasing, stemming, lemmatization, stopword removal, and a custom option to filter short tokens. After preprocessing, the script counts the frequency of tokens, outputs the most and least frequent tokens, and generates a bar plot visualizing these frequencies on a log scale.

## Requirements:
##### Required libraries include string, nltk, and matplotlib
##### You can install the required libraries using pip:
##### pip install string nltk matplotlib

## How to Use
##### 1. Place the Python script in the desired directory.
##### 2. Create or select a text file that you want to process.
##### 3. Place this text file in the same directory as the Python script. (In my case, normalize_text.py and textfile.txt are in the same directory)
#####  4. Run the script from the command line with the following format:
##### python script_name.py <file_path> [options]

##### Example Usage:
#####  python normalize_text.py filetext.txt lowercase stemming remove_stopwords custom_option

## Example Output
##### -The script will print the raw tokens from the text file, the tokens after punctuation removal, and the tokens after each preprocessing step.
##### -It will display the top 25 most frequent tokens and bottom 25 least frequent tokens.
##### -A bar chart will be shown, illustrating the token frequency on a log scale.

## Error Handling
##### -If the script cannot find the provided file, it will print an error message.

## Generative AI Use
##### Generative AI was used as outlined in the Report: homework1_report.pdf. It was also used partially in creating this README file.
