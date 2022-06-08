from transformers import DistilBertConfig, DistilBertTokenizerFast, DistilBertModel
from transformers import RobertaConfig, RobertaTokenizerFast, RobertaModel
from transformers import AlbertConfig, AlbertTokenizerFast, AlbertModel
from torch import nn
import json
from typing import List, Dict, Optional, Union, Tuple
import os
import pandas as pd
from sentence_transformers import SentenceTransformer, models
from sentence_transformers.readers import InputExample


class Transformer(nn.Module):
    def __init__(self, model_name_or_path: str, max_seq_length: Optional[int] = None,
                 model_args: Dict = {}, cache_dir: Optional[str] = None,
                 tokenizer_args: Dict = {}, do_lower_case: bool = False,
                 tokenizer_name_or_path : str = None,
                 bert_name : str = "bert"):
        super(Transformer, self).__init__()
        self.config_keys = ['max_seq_length', 'do_lower_case']
        self.do_lower_case = do_lower_case

        if bert_name == "roberta":
            my_config = RobertaConfig.from_pretrained(model_name_or_path)
            self.auto_model = RobertaModel(my_config)
            self.tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_name_or_path)

        elif bert_name == "albert":
            my_config = AlbertConfig.from_pretrained(model_name_or_path)
            self.auto_model = AlbertModel(my_config)
            self.tokenizer = AlbertTokenizerFast.from_pretrained(tokenizer_name_or_path)

        else: # Bert
            my_config = DistilBertConfig.from_pretrained(model_name_or_path)
            self.auto_model = DistilBertModel(my_config)
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_name_or_path)
        
        if max_seq_length is None:
            if hasattr(self.auto_model, "config") and hasattr(self.auto_model.config, "max_position_embeddings") and hasattr(self.tokenizer, "model_max_length"):
                max_seq_length = min(self.auto_model.config.max_position_embeddings, self.tokenizer.model_max_length)

        self.max_seq_length = max_seq_length

        if tokenizer_name_or_path is not None:
            self.auto_model.config.tokenizer_class = self.tokenizer.__class__.__name__


    def _load_model(self, my_config):
        """Loads the transformer model"""
        self.auto_model = DistilBertModel(my_config)

    def __repr__(self):
        return "Transformer({}) with Transformer model: {} ".format(self.get_config_dict(), self.auto_model.__class__.__name__)

    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        trans_features = {'input_ids': features['input_ids'], 'attention_mask': features['attention_mask']}
        if 'token_type_ids' in features:
            trans_features['token_type_ids'] = features['token_type_ids']

        output_states = self.auto_model(**trans_features, return_dict=False)
        output_tokens = output_states[0]

        features.update({'token_embeddings': output_tokens, 'attention_mask': features['attention_mask']})

        if self.auto_model.config.output_hidden_states:
            all_layer_idx = 2
            if len(output_states) < 3: #Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            features.update({'all_layer_embeddings': hidden_states})

        return features

    def get_word_embedding_dimension(self) -> int:
        return self.auto_model.config.hidden_size

    def tokenize(self, texts: Union[List[str], List[Dict], List[Tuple[str, str]]]):
        """
        Tokenizes a text and maps tokens to token-ids
        """
        output = {}
        if isinstance(texts[0], str):
            to_tokenize = [texts]
        elif isinstance(texts[0], dict):
            to_tokenize = []
            output['text_keys'] = []
            for lookup in texts:
                text_key, text = next(iter(lookup.items()))
                to_tokenize.append(text)
                output['text_keys'].append(text_key)
            to_tokenize = [to_tokenize]
        else:
            batch1, batch2 = [], []
            for text_tuple in texts:
                batch1.append(text_tuple[0])
                batch2.append(text_tuple[1])
            to_tokenize = [batch1, batch2]

        #strip
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

        #Lowercase
        if self.do_lower_case:
            to_tokenize = [[s.lower() for s in col] for col in to_tokenize]

        output.update(self.tokenizer(*to_tokenize, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_seq_length))
        return output


    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        self.auto_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        with open(os.path.join(output_path, 'sentence_bert_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path: str):
        #Old classes used other config names than 'sentence_bert_config.json'
        for config_name in ['sentence_bert_config.json', 'sentence_roberta_config.json', 'sentence_distilbert_config.json', 'sentence_camembert_config.json', 'sentence_albert_config.json', 'sentence_xlm-roberta_config.json', 'sentence_xlnet_config.json']:
            sbert_config_path = os.path.join(input_path, config_name)
            if os.path.exists(sbert_config_path):
                break

        with open(sbert_config_path) as fIn:
            config = json.load(fIn)
        return Transformer(model_name_or_path=input_path, **config)
    
    
###################### 
# Make Sentence BERT #
######################
def make_sentenceBERT(model_path, tokenizer_path, model_name, device):
    # 1. Load KoLawBERT
    word_embedding_model = Transformer(model_name_or_path = model_path,
                                       tokenizer_name_or_path = tokenizer_path,
                                       bert_name = model_name)
    
    # 2. Mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)
    
    # 3. Sentence-BERT
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)

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
