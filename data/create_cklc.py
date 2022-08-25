import pandas as pd
import xml.etree.ElementTree as ET
from urllib.request import urlopen
from tqdm import trange, tqdm
import os
import re
import random
import pickle


##############################################
## 1. Crawling Korean Law Cases(1954 ~ Now) ##
##############################################

url = "https://www.law.go.kr/your_url"
response = urlopen(url).read()
xtree = ET.fromstring(response)

totalCnt = int(xtree.find('totalCnt').text)
print(">> Data Size : ", totalCnt)

# Initialization
page = 1
rows = []
# Crawling Korean court precedent
for i in trange(int(totalCnt / 20)):
    try:
        items = xtree[5:]
    except:
        break
        
    for node in items:
        data_link = node.find('판례상세링크').text

        rows.append({'link': data_link})
    page += 1
    url = "https://www.law.go.kr/your_url={}".format(page)
    response = urlopen(url).read()
    xtree = ET.fromstring(response)
cases = pd.DataFrame(rows)
cases.to_csv('../data/cases(2022-08-21).csv', index=False)


####################
## 2. Extractions ##
####################

case_list = pd.read_csv('../data/cases(2022-08-21).csv')
print(">> Total number of precedents in Korean courts : {}".format(len(case_list)))

data_dict = {'judgment_issue' : [], 'judgment_summary' : [], 'judgment_contents' : []}
url = "https://www.law.go.kr"
case_url = case_list.link.to_list()
case_link = [url+case.replace('HTML', 'XML') for case in case_url]

for link in tqdm(case_link):
    response = urlopen(link).read()
    tree = ET.fromstring(response)
    # Add data_dict
    #data_dict['case_name'].append(tree.find('사건명').text)  
    #data_dict['case_number'].append(tree.find('사건번호').text)
    #data_dict['date'].append(tree.find('선고일자').text)
    #data_dict['case_code'].append(tree.find('사건종류코드').text)
    if tree.find('판시사항').text:
        #data_dict['judgment_issue'].append(preprocessing_1(tree.find('판시사항').text))
        data_dict['judgment_issue'].append(tree.find('판시사항').text)
    else:
        data_dict['judgment_issue'].append(None)
        
    if tree.find('판결요지').text:
        #data_dict['judgment_summary'].append(preprocessing_1(tree.find('판결요지').text))
        data_dict['judgment_summary'].append(tree.find('판결요지').text)
    else:
        data_dict['judgment_summary'].append(None)
    if tree.find('판례내용').text:
        #data_dict['judgment_contents'].append(preprocessing_2(tree.find('판례내용').text))
        data_dict['judgment_contents'].append(tree.find('판례내용').text)
    else:
        data_dict['judgment_contents'].append(None)

with open('original_legal_data.pickle', 'wb') as f:
    pickle.dump(data_dict, f, pickle.HIGHEST_PROTOCOL)

############################
## 3. Text Preprocessing  ##
############################
def clean_special(sent):
    # remove special characters
    sent = re.sub("[,,ㆍ·\'\"’‘”“!?\\‘|\<\>`\'[\◇…]", " ", sent)
    sent = re.sub("[\]]", ".", sent)
    # Except for Korean, English, numbers, and \s (white space)
    sent = re.sub("[^가-힣a-zA-Z0-9\\s[]()]", " ", sent)
    # keep spacing
    sent = re.sub("\s+", " ", sent)
    # Remove the first two spaces
    sent = sent.strip()
    # remove specific characters
    sent = sent.replace("a href AJAX class link onclick javascript fncLawPop", "")
    sent = sent.replace("prec", "")
    return sent

splitter = re.compile("[./?\n]")
contents_list, summary_list, issue_list = [], [], []

# 3-1. Contents
for row in data_dict['judgment_contents']:
    row = str(row)
    if re.search('【주    문】(.+?)【이    유】', row):
        found_1 = re.search('【주    문】(.+?)【이    유】', row).group(1)
    if re.search('【이    유】(.+?)(주심)', row):
        found_2 = re.search('【이    유】(.+?)(주심)', row).group(1)
    new_sent = found_1 + found_2
    # 1. Spliter
    new_sent = splitter.split(new_sent)
    # 2. Remove short sentences
    new_sent = [line for line in new_sent if len(line) > 20]
    # 3. Mapping Preprocessing function
    new_sent = list(map(clean_special, new_sent))
    # 4. Cleaned Sentences
    contents_list.extend(new_sent)

# 3-2. Summary
for row in data_dict['judgment_summary']:
    row = str(row)
    # 1. Spliter
    new_sent = splitter.split(row)
    # 2. Remove short sentences
    new_sent = [line for line in new_sent if len(line) > 20]
    # 3. Mapping Preprocessing function
    new_sent = list(map(clean_special, new_sent))
    # 4. Cleaned Sentences
    summary_list.extend(new_sent)

# 3-3. Issue
for row in data_dict['judgment_issue']:
    row = str(row)
    # 1. Spliter
    new_sent = splitter.split(row)
    # 2. Remove short sentences
    new_sent = [line for line in new_sent if len(line) > 20]
    # 3. Mapping Preprocessing function
    new_sent = list(map(clean_special, new_sent))
    # 4. Cleaned Sentences
    issue_list.extend(new_sent)

####################################
## 4. Make Text File line by line ##
####################################
small_cklc = issue_list + summary_list
large_cklc = small_cklc + contents_list
sample_cklc = small_cklc[:1000]


with open('./data/cklc-small.txt', 'w') as fp:
    for row in small_cklc:
        fp.write("%s\n" % row)
    print("Write is Done.")

with open('./data/cklc-large.txt', 'w') as fp:
    for row in large_cklc:
        fp.write("%s\n" % row)
    print("Write is Done.")

with open('./data/cklc-sample.txt', 'w') as fp:
    for row in sample_cklc:
        fp.write("%s\n" % row)
    print("Write is Done.")