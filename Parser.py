import re
import os
import numpy as np
import pandas as pd
import Load_MasterDictionary as LM


def get_all_files(base_path, filename):
    files_list = []
    for dirpath, _, filenames in os.walk(base_path):
        for file in filenames:
            if file == filename:
                files_list.append(os.path.join(dirpath, file))
    return files_list


def process_lm(TARGET_FILES, num_files):
    lm_dictionary = LM.load_masterdictionary(MASTER_DICTIONARY_FILE, True)
    negative_word_set = {key for key, values in lm_dictionary.items() if values.negative != 0}
    lm_set = set(lm_dictionary.keys())

    tf_matrix = np.zeros((len(TARGET_FILES), len(negative_word_set)))
    idf_matrix = np.zeros_like(tf_matrix)
    doc_length_matrix = np.zeros((len(TARGET_FILES), 1))
    negative_word_to_idx = {word: idx for idx, word in enumerate(negative_word_set)}

    for doc_idx, file in enumerate(TARGET_FILES):
        with open(file, 'r', encoding='UTF-8', errors='ignore') as f:
            contents = f.read()
            tokens = re.findall('\w+', contents)
            tokens = [token.upper() for token in tokens if not token.isdigit() and len(token) > 1]
            word_count = sum(1 for token in tokens if token in lm_set)
            for token in tokens:
                if token in negative_word_set:
                    word_idx = negative_word_to_idx[token]
                    tf_matrix[doc_idx, word_idx] += 1
                    idf_matrix[doc_idx, word_idx] = 1
            doc_length_matrix[doc_idx, 0] = word_count

    num_docs_containing_word = np.sum(idf_matrix, axis=0)
    w = np.zeros((tf_matrix.shape[0], 1))
    avg_word_count = np.mean(doc_length_matrix)
    for i in range(tf_matrix.shape[0]):
        sum_weights = 0
        for j in range(tf_matrix.shape[1]):
            if tf_matrix[i, j] >= 1:
                sum_weights += ((1 + np.log(tf_matrix[i, j])) / (1 + np.log(avg_word_count))) * np.log(num_files / (num_docs_containing_word[j] + 1e-10))
        w[i, 0] = sum_weights

    tfidf_score = w.sum(axis=1).reshape(-1, 1)
    proportion_weights = tf_matrix.sum(axis=1) / doc_length_matrix

    df = pd.DataFrame({
        'accession number': [os.path.basename(os.path.dirname(file)) for file in TARGET_FILES],
        'tf-idf': tfidf_score[:, 0],
        'proportion weight': proportion_weights[:, 0]
    })

    return df


def process_hd(TARGET_FILES, num_files):
    Harvard_dictionary = pd.read_excel('/Users/isabella/Desktop/inquirerbasic.xls', engine='xlrd')
    negative_words = set(Harvard_dictionary[Harvard_dictionary['Negativ'] == 'Negativ']['Entry'].str.upper().tolist())
    Hd_list = set(Harvard_dictionary['Entry'].str.upper().tolist())

    tf_matrix = np.zeros((len(TARGET_FILES), len(negative_words)))
    idf_matrix = np.zeros_like(tf_matrix)
    doc_length_matrix = np.zeros((len(TARGET_FILES), 1))
    word_to_idx = {word: idx for idx, word in enumerate(negative_words)}

    for doc_idx, file in enumerate(TARGET_FILES):
        with open(file, 'r', encoding='UTF-8', errors='ignore') as f:
            contents = f.read()
            tokens = re.findall(r'\w+', contents.upper())
            tokens = [token for token in tokens if not token.isdigit() and len(token) > 1 and token in Hd_list]
            
            word_count = len(tokens)
            for token in tokens:
                if token in negative_words:
                    word_idx = word_to_idx[token]
                    tf_matrix[doc_idx, word_idx] += 1
                    idf_matrix[doc_idx, word_idx] = 1

            doc_length_matrix[doc_idx, 0] = word_count

    num_docs_containing_word = np.sum(idf_matrix, axis=0)
    avg_word_count = np.mean(doc_length_matrix)
    
    tfidf_score = []
    for i in range(tf_matrix.shape[0]):
        sum_weights = sum(
            ((1 + np.log(tf_matrix[i, j])) / (1 + np.log(avg_word_count))) * np.log(num_files / (num_docs_containing_word[j] + 1e-10))
            for j in range(tf_matrix.shape[1]) if tf_matrix[i, j] >= 1
        )
        tfidf_score.append(sum_weights)

    tfidf_score = np.array(tfidf_score).reshape(-1, 1)
    proportion_weights = tf_matrix.sum(axis=1) / doc_length_matrix

    df_Hd = pd.DataFrame({
        'accession number': [os.path.basename(os.path.dirname(file)) for file in TARGET_FILES],
        'tf-idf': tfidf_score[:, 0],
        'proportion weight': proportion_weights[:, 0]
    })
    return df_Hd


BASE_PATH = '/Users/isabella/Desktop/tickers/sec-edgar-filings/'
TARGET_FILES = get_all_files(BASE_PATH, 'full-submission.txt')

MASTER_DICTIONARY_FILE = '/Users/isabella/Desktop/LoughranMcDonald_MasterDictionary_2018.csv'
num_files = len(TARGET_FILES)

df_LM = process_lm(TARGET_FILES, num_files)
df_Hd = process_hd(TARGET_FILES, num_files)

print(df_LM.head(5))
print(df_Hd.head(5))
