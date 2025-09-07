import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np
import spacy
from tqdm import tqdm
import pandas as pd

STOP_WORDS = stopwords.words("english")
nlp = spacy.load("en_core_web_lg")  # SpaCy large model

# ----------------- Text preprocessing -----------------
def preprocess(text):
    text = str(text).lower()
    text = text.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")\
        .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")\
        .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")\
        .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")\
        .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")\
        .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")\
        .replace("€", " euro ").replace("'ll", " will")
    
    text = re.sub(r"([0-9]+)000000", r"\1m", text)
    text = re.sub(r"([0-9]+)000", r"\1k", text)

    porter = PorterStemmer()
    pattern = re.compile(r'\W')

    if isinstance(text, str):
        text = re.sub(pattern, ' ', text)
        text = porter.stem(text)
        text = BeautifulSoup(text, "lxml").get_text()
    return text

# ----------------- Feature Engineering -----------------
def add_handcrafted_features(df):
    df['freq_qid1'] = df.groupby('qid1')['qid1'].transform('count')
    df['freq_qid2'] = df.groupby('qid2')['qid2'].transform('count')
    df['q1len'] = df['question1'].str.len()
    df['q2len'] = df['question2'].str.len()
    df['q1_n_words'] = df['question1'].apply(lambda x: len(str(x).split()))
    df['q2_n_words'] = df['question2'].apply(lambda x: len(str(x).split()))

    def word_common(row):
        w1 = set(str(row['question1']).lower().split())
        w2 = set(str(row['question2']).lower().split())
        return len(w1 & w2)
    df['word_Common'] = df.apply(word_common, axis=1)

    def word_total(row):
        w1 = set(str(row['question1']).lower().split())
        w2 = set(str(row['question2']).lower().split())
        return len(w1) + len(w2)
    df['word_Total'] = df.apply(word_total, axis=1)

    def word_share(row):
        w1 = set(str(row['question1']).lower().split())
        w2 = set(str(row['question2']).lower().split())
        return len(w1 & w2) / (len(w1) + len(w2) + 1e-6)
    df['word_share'] = df.apply(word_share, axis=1)

    df['freq_q1+q2'] = df['freq_qid1'] + df['freq_qid2']
    df['freq_q1-q2'] = abs(df['freq_qid1'] - df['freq_qid2'])
    return df

# ----------------- SpaCy embeddings -----------------
def get_spacy_vector(text):
    doc = nlp(str(text))
    return doc.vector

def add_spacy_embeddings(df):
    df['q1_feats_m'] = df['question1'].apply(get_spacy_vector)
    df['q2_feats_m'] = df['question2'].apply(get_spacy_vector)
    return df

# ----------------- Expand vector columns -----------------
def expand_vector_columns(df, col_name_prefix):
    vector_cols = pd.DataFrame(df[col_name_prefix].to_list(), index=df.index)
    vector_cols = vector_cols.add_prefix(col_name_prefix + "_")
    df = df.drop(columns=[col_name_prefix])
    df = pd.concat([df, vector_cols], axis=1)
    return df
