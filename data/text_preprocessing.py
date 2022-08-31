#! pip install hanja
#! pip install pickle5

import re
import hanja
import pandas as pd
import pickle5 as pickle

###############################
## 1. Preprocessing Function ##
###############################

# 1. Filtering sentences based on words length
def sentence_split(sent):
    if len(sent.split()) > 8:
        return sent
    else:
        pass

# 2. Filtering Chinese characters
def cleansing_chinese(sent):
    # Chinese to Korean
    sent = hanja.translate(sent, 'substitution')
    return sent

# 3. Filtering special characters and spaces
def cleansing_special(sent):
    sent = re.sub("[,,ㆍ·\'\"’‘”“!?\\‘|\<\>`\'[\◇…]", " ", sent)
    sent = re.sub("[\]]", ".", sent)
    sent = re.sub("[^.[]가-힣a-zA-Z0-9\\s]", " ", sent)
    sent = re.sub("\s+", " ", sent)
    sent = sent.strip()
    sent = sent.replace("[SEP]", "")
    return sent

# 4. Final Preprocessing
def cleansing_sent(sent):
    clean_sent = cleansing_chinese(sent)
    clean_sent = cleansing_special(clean_sent)
    return clean_sent

# 5. Pre-processing
def preprocessing(text_list):
    # convert to string
    sent = list(map(str, text_list))
    # Filtering sentence length
    sent = list(map(sentence_split, sent))
    # Filtering None
    sent = list(filter(None, sent))
    # Remove Chines and Special characters
    sent = list(map(cleansing_sent, sent))
    return sent


#############################################
## 2. Make Clean Korean Legal Corpus(CKLC) ##
#############################################
df = pd.read_csv("./data/contracts_input_final_label_fixed.csv")
with open('./data/legal_data.pickle', 'rb') as f:
    data = pickle.load(f)
law_cases = pd.DataFrame(data)

conts = df['원본'].to_list()
new_conts = df['new_conts'].to_list()
issue = law_cases['judgment_issue'].to_list()
summary = law_cases['judgment_summary'].to_list()
contents = law_cases['judgment_contents'].to_list()
contents = [row.strip("[주 문]") for row in contents]
contents = [row.replace("[이 유]", "") for row in contents]

preprocessed_conts = preprocessing(conts)
preprocessed_new = preprocessing(new_conts)
preprocessed_issue = preprocessing(issue)
preprocessed_summary = preprocessing(summary)
preprocessed_contents = preprocessing(contents)

contracts = preprocessed_conts + preprocessed_new
small_cklc = preprocessed_issue + preprocessed_summary
small_s = contracts + small_cklc 

print(">> CKLC-Small size :", len(small_s))

# List to a File line by line
with open('./data/cklc-small.txt', 'w') as fp:
    for row in small_s:
        fp.write("%s\n" % row)
    print("Write is Done.")