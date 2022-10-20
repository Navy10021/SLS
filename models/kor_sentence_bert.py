from transformers import AutoConfig, AutoTokenizer, AutoModel 
from torch import nn
import json
from typing import List, Dict, Optional, Union, Tuple
import os
import pandas as pd
from sentence_transformers import SentenceTransformer, models
from sentence_transformers.readers import InputExample
    
###################### 
# Make Sentence BERT #
######################
def make_sentenceBERT(model_path, tokenizer_path):
    # 1. Load KoLawBERT
    word_embedding_model = models.Transformer(model_name_or_path = model_path,
                                              tokenizer_name_or_path = tokenizer_path)
    
    # 2. Mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)
    
    # 3. Sentence-BERT
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    return model


########################### 
# KorNLI Fine-Tuning Task #
###########################
# Make KorNLI Dataset for Training
def drop_kornli(df):
    df = df.dropna(how='any')
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    return df

def make_kornli_dataset(df):
    label_dict = {"contradiction": 0, "entailment": 1, "neutral": 2}
    samples = []
    s1 = df['sentence1'].to_list()
    s2 = df['sentence2'].to_list()
    label = list(map(lambda x : label_dict[x], df['gold_label'].to_list()))
    for (s1, s2, score) in zip(s1, s2, label):
        samples.append(InputExample(texts= [s1,s2], label=score))
    return samples


########################### 
# KorSTS Fine-Tuning Task #
###########################
def drop_korsts(df):
    df = df.dropna(how='any')
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    df = df.loc[:,['score', 'sentence1', 'sentence2']]
    return df

def make_korsts_dataset(df):
    samples = []
    s1 = df['sentence1'].to_list()
    s2 = df['sentence2'].to_list()
    score = list(map(lambda i : i/5.0, df['score'].to_list()))
    for (s1, s2, score) in zip(s1, s2, score):
        samples.append(InputExample(texts= [s1,s2], label=score))
    return samples
