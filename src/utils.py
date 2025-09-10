from src.logger import logging
import matplotlib.pyplot as plt
import seaborn as sns
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np
import spacy
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    log_loss, confusion_matrix, roc_auc_score, roc_curve
)

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

# ---------------- Evaluating Model and returning the results------------



def evaluate_model(y_true, y_pred, y_proba=None, plot_cm=True, plot_roc=True):
    """
    Evaluate classification performance with multiple metrics.
    
    Parameters:
    - y_true : ground truth labels
    - y_pred : predicted labels
    - y_proba : predicted probabilities (optional, for log_loss & ROC-AUC)
    - plot_cm : whether to plot confusion matrix
    - plot_roc : whether to plot ROC curve (binary classification only)
    """
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_micro": f1_score(y_true, y_pred, average="micro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "classification_report": classification_report(y_true, y_pred, digits=4)
    }

    if y_proba is not None:
        results["log_loss"] = log_loss(y_true, y_proba)
        # ROC-AUC (only works for binary/multiclass with proba)
        try:
            if y_proba.shape[1] == 2:  # binary classification
                results["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
            else:  # multiclass
                results["roc_auc_ovr"] = roc_auc_score(y_true, y_proba, multi_class="ovr")
                results["roc_auc_ovo"] = roc_auc_score(y_true, y_proba, multi_class="ovo")
        except Exception as e:
            logging.warning(f"ROC-AUC could not be computed: {e}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    results["confusion_matrix"] = cm

    # Logging metrics
    logging.info(f"Accuracy: {results['accuracy']:.4f}")
    logging.info(f"Macro F1 Score: {results['f1_macro']:.4f}")
    logging.info(f"Micro F1 Score: {results['f1_micro']:.4f}")
    logging.info(f"Weighted F1 Score: {results['f1_weighted']:.4f}")
    if "log_loss" in results:
        logging.info(f"Log Loss: {results['log_loss']:.4f}")
    if "roc_auc" in results:
        logging.info(f"ROC-AUC: {results['roc_auc']:.4f}")
    logging.info(f"\nClassification Report:\n{results['classification_report']}")

    # Plot confusion matrix
    if plot_cm:
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()

    # Plot ROC curve (only binary)
    if plot_roc and y_proba is not None and y_proba.shape[1] == 2:
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {results['roc_auc']:.4f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.show()

    return results

