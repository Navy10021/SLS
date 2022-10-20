import pandas as pd
import re
import hanja
from tqdm import tqdm
import json


###################
## Legal Dataset ##
###################

# Load Dataframe
data_path = './data/law_cases(20221020).csv'
df = pd.read_csv(data_path)
df = df.dropna()
df = df.drop_duplicates(subset=['judgment_issue'])

# Law cases since 1990(your choice)
df = df.sort_values(by = "date")
df = df[df['date'] > 19900000.0]

print("Legal Dataset Size : {}".format(len(df)))


############################
## Preprocessing function ##
############################

# 1. Filtering Chinese characters
def cleansing_chinese(sent):
    # Chinese to Korean
    sent = hanja.translate(sent, 'substitution')
    return sent

# 2. Filtering special characters and spaces
def cleansing_special(sent):
    sent = re.sub("[]ㆍ·\'\"’‘”“!?\\‘|\<\>`\'[\◇○…@▶▲ⓒ]", " ", sent)
    sent = re.sub("[^.,%:()가-힣0-9\\s]", " ", sent)
    sent = re.sub("\s+", " ", sent)
    sent = sent.strip()
    return sent

# 3. Sentence Preprocessing
def cleansing_sent(sent):
    sent = str(sent)
    clean_sent = cleansing_chinese(sent)
    clean_sent = cleansing_special(clean_sent)
    return clean_sent

# 4. Precedent Preprocessing
def cleansing_contents(text_data):
    text = str(text_data)

    if re.search('【주    문】(.+?)【이    유】', text):
        main = re.search('【주    문】(.+?)【이    유】', text).group(1)
    
    elif re.search('【주  문】(.+?)【이  유】', text):
        main = re.search('【주  문】(.+?)【이  유】', text).group(1)
    
    elif re.search('【주    문】(.+?)【이  유】', text):
        main = re.search('【주    문】(.+?)【이  유】', text).group(1)

    elif re.search('[주    문](.+?)[이    유]', text):
        main = re.search('[주    문](.+?)[이    유]', text).group(1)
    
    elif re.search('【주문】(.+?)【이유】', text):
        main = re.search('【주문】(.+?)【이유】', text).group(1)

    main = main.strip()
    c_main = cleansing_sent(main)

    if re.search('【이    유】(.+?) 주문과 같이 판결한다.', text):
        reason = re.search('【이    유】(.+?) 주문과 같이 판결한다.', text).group(1) + ' 주문과 같이 판결한다.'
    
    elif re.search('【이    유】(.+?) 주문과 같이 결정한다.', text):
        reason = re.search('【이    유】(.+?) 주문과 같이 결정한다.', text).group(1) + ' 주문과 같이 결정한다.'

    elif re.search('【이    유】(.+?)(재판장)', text):
        reason = re.search('【이    유】(.+?)(재판장)', text).group(1)
        reason = reason[:-8]
    
    elif re.search('【이  유】(.+?)(재판장)', text):
        reason = re.search('【이  유】(.+?)(재판장)', text).group(1)
        reason = reason[:-8]

    elif re.search('[이    유](.+?) 주문과 같이 판결한다.', text):
        reason = re.search('[이    유](.+?) 주문과 같이 판결한다.', text).group(1) + ' 주문과 같이 판결한다.'

    elif re.search('[이    유](.+?) 주문과 같이 결정한다.', text):
        reason = re.search('[이    유](.+?) 주문과 같이 결정한다.', text).group(1) + ' 주문과 같이 결정한다.'

    elif re.search('【이  유】(.+?) 이를 기각한다.', text):
        reason = re.search('【이  유】(.+?) 이를 기각한다.', text).group(1) + ' 이를 기각한다.'


    elif re.search('【이유】(.+?)', text):
        reason = re.search('【이유】(.+?)', text).group(1)
    
    elif re.search('【이  유】(.+?)', text):
        reason = re.search('【이  유】(.+?)', text).group(1)
    
    else:
        reason =  re.search('【이    유】(.+?)', text).group(1)

    reason = reason.strip()
    c_reason = cleansing_sent(reason)
    
    return c_main, c_reason



#############################
## Make Clean legal Corpus ##
#############################

df_list = df.to_dict('records')
corpus = list()

for idx, row in enumerate(tqdm(df_list)):

    data = dict()
    data['id'] = row['case_id']
    data['law_case'] = dict()
    data['law_case']['case_name'] = row['case_name']
    data['law_case']['case_num'] = row['case_number']
    data['law_case']['issue'] = cleansing_sent(row['judgment_issue'])
    data['law_case']['summary'] = cleansing_sent(row['judgment_summary'])
    data['law_case']['contents'] = dict()

    if len(row['judgment_contents']) < 100:
        pass
    else:
        data['law_case']['contents']['main'], data['law_case']['contents']['reasons'] = cleansing_contents(row['judgment_contents'])

    corpus.append(data)
    

print("  \n === Legal Corpus Examples === \n  {}".format(corpus[:3]))



##########################
## Convert to json file ##
##########################

with open('./data/legal_corpus.json', 'w') as f:
    json.dump(corpus, f)