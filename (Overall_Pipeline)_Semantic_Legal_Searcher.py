#! pip install transformers
#! pip install -U sentence-transformers
#! pip install sentencepiece
#! pip install faiss-gpu
#! pip install pytrends
#! pip install funcy pickle5
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer, util

###############################################
## Legal Database (Original Korean case law) ##
###############################################

# 1. Load Legal Dataset(Get Only Judicial precedent data)
df = pd.read_csv('./data/law_cases(20220829).csv')
df = df.dropna(how='any')
# 2. Make index for clustering
print(">> Judicial precedent data size : ",len(df))

# 3. Text length measurement function
def text_len_plot(dataframe, col):
    """
    dataframe : DataFrame
    col : str(Column name)
    """
    dataframe['text_len'] = dataframe[col].apply(lambda words : len(words.split()))
    #dataframe['sent_len'] = dataframe[col].apply(lambda words : len(words.split(".")))
    mean_seq_len = np.round(dataframe['text_len'].mean() + dataframe['text_len'].std()).astype(int)
    #mean_sent_len = np.round(dataframe['sent_len'].mean() + dataframe['sent_len'].std()).astype(int)
    print(">> {} Average Text length : {}".format(col, mean_seq_len))
    print(">> {} Max Text length : {}".format(col, dataframe['text_len'].max()))
    print(">> {} Text words : {}".format(col, dataframe['text_len'].sum()))

    plt.figure(figsize=(12,6))
    sns.histplot(dataframe['text_len'], kde=True, color='royalblue', label='Text_length')
    plt.axvline(x=mean_seq_len, color='r', linestyle='--', label='max_length')
    plt.title('Text length'); plt.legend()
    plt.show()
    
# 3. Text Length
# 3-1. Judgment_issue
text_len_plot(df, 'judgment_issue')
# 3-2. judgment_summary
text_len_plot(df, 'judgment_summary')
# 3-3. judgment_full-text
text_len_plot(df, 'judgment_contents')
# 3-4. case_name
text_len_plot(df, 'case_name')


#######################################################
## STEP 1 : Load My PLMs(Pre-trained Language Model) ##
#######################################################

# Load supervised or unsupervised fine-tuned models
my_model = './output/nil_sts_tsdae-bert'


#######################################################
## STEP 2 : Parallel Clustering-based Topic Modeling ##
#######################################################
from models.parallel_topic_model import *

# 1.Obtain Semantic Legal Embedding
target_column = 'judgment_summary'

cluster = ParallelCluster(
    dataframe = df,
    tgt_col = target_column,
    model_name = my_model,
    use_sentence_bert = True
    )

# 2. Parallel Embedding Clustering
clusters, unclusters = cluster.parallel_cluster(
    clusters = None,
    threshold = 0.78,
    page_size = 2500,
    iterations = 15
    )

# 3. Stack : Stack the clustered results in order of cluster size
col_list = ['case_name', 'case_number', 'date', 'case_code',
            'judgment_issue', 'judgment_summary', 'judgment_contents',
            'case_id', 'case_hits', 'case_hits_norm']

new_df = cluster.cluster_stack(
    col_list = col_list,
    clusters = clusters,
    unclusters = unclusters
    )

# 4. Topic Modeling : Extract Latent Topics (or Keywords)
top_n_words = cluster.extract_top_n_words_per_topic(
    dataframe = new_df,
    n = 20,
    en = False
    )
new_df['Topic_Modeling'] = [top_n_words[i] for i in new_df['Topic'].values]
#print(new_df.head())

# 5. Save the Parallel Clusted Dataset 
new_df.to_csv("./data/clusted_df.csv", sep=',', na_rep="NaN")


#######################################
## STEP 3 : Semantic law case search ##
#######################################
from models.semantic_law_search import *

# 1. Load Clustered DataFrame
new_df = pd.read_csv('./data/clusted_df.csv')

# 2. Obtain clustered documents embeddings from PLMs(KoLawBERT models)
law_bert = KoLawBERT(
    dataframe = new_df,
    tgt_col = 'judgment_summary',
    model_name = my_model,
    use_sentence_bert = True,
    cluster = True
    )

# 3. Build the Index
# 3-1.(Strategy 1) : Calculate Vector Similarity with All text embeddings
index_1 = law_bert.all_relevant_embedding()
# 3-2.(Strategy 2) : Calculate Vector Similarity with Centroid of embeddings
index_2 = law_bert.centroid_relevant_embedding(nlist = 200, nprobe = 6)

# 4. Search
### Enter User's query ###
# "(English) My car collided with a vehicle in the next lane while trying to avoid another vehicle changing from lane 1 to 2."
my_query = "1차선에서 2차선으로 바꾸는 차량을 피하려다가 옆 차선 차와 충돌하였습니다."

# 4-1. Search the Law cases(Strategy 1)
original_outputs, _ = law_bert.search(
    user_query = my_query,
    top_k = 10,
    index = index_1,
    print_results = True,
    )

# 4-2. Search the Law cases(Strategy 2 : More Faster)
fast_outputs, _ = law_bert.search(
    user_query = my_query,
    top_k = 10,
    index = index_2,
    print_results = True,
    )



#####################################
## STEP 4 : Dynamic Post-Filtering ##
#####################################
from models.dynamic_post_filtering import *

# 4-1. Popularity-based Filtering
p_outputs = sorted(original_outputs, key=lambda x: x['case_hits'], reverse = True)

print("\n  ========== <<  Popularity-based Filtering (Hits)  >> ========== \n")
for i, out in enumerate(p_outputs):
    #print("\n === Law Cases ===")
    print("\n   Top {} - Case name (Number) : {} ({})  \n | Cluster : {} \n | Cluster's Topics (Keywords) : {} \n | Date : {} | Judgment Issue : {} \n | Judgment Summary : {}".format(i+1, out['case_name'], out['case_number'],
                                                                                                                                                  out['Topic'], out['Topic_Modeling'],
                                                                                                                                                  out['date'], out['judgment_issue'],
                                                                                                                                                  out['judgment_summary']))

# 4-2. User-based Filtering
# 1. Load User cases hits data
user_rating = pd.read_csv('./data/user_views.csv')
view_df = new_df[['case_id', 'case_name', 'judgment_issue', 'judgment_summary', 'case_hits', 'case_hits_norm']]
user_df = pd.merge(user_rating, view_df, on = 'case_id')
print(">> Merge dataframe size : ", len(user_df))

# 2. Make Law case and User Interaction table
cf_preds_df = interaction_table(user_df)

# 3. If User Id is "laywer_10" 
user_outputs = user_based_sementic_search(
    cf_dataframe = cf_preds_df,
    view_dataframe = view_df,
    user_id = 10,
    my_model = law_bert,
    my_query = my_query,
    index_kind = index_1
    )

# 4. Sorting by scaled ratings
user_outputs = sorted(user_outputs, key = lambda x : x['scaled_rating'], reverse = True)
user_outputs = user_outputs[:10]

print("\n  ========== <<  User-based Filtering  >> ==========  \n")
for i, out in enumerate(user_outputs):
    #print("\n === Law Cases ===")
    print("\n   Top {} - Case name (Number) : {} ({})  \n | Cluster : {} \n | Cluster's Topics (Keywords) : {} \n | Date : {} | Judgment Issue : {} \n | Judgment Summary : {}".format(i+1, out['case_name'], out['case_number'],
                                                                                                                                                  out['Topic'], out['Topic_Modeling'],
                                                                                                                                                  out['date'], out['judgment_issue'],
                                                                                                                                                  out['judgment_summary']))


# 4-3. Online-based Filtering
# 1. Type of Law Cases (Based onf Korea and United state courts)
genre = ['Criminal cases',
         'Civil cases',
         'Family cases',
         'Bankrupty',
         'Marriage Dissolution',
         'Paternity',
         'Child Custody',
         'Domestic Violence cases',
         'Name Changes',
         'Guardianship',
         'Parental Rights',
         'Adoptions',
         'Juvenile cases',
         'Fines cases',
         'Community service cases',
         'Probation cases',
         'Prision',
         'Tort claims',
         'Breach of contract claims',
         'Equitable claims',
         'Landlord and tenant']

# 2. Encoding and make search vector
_, items = keyword_encoding(genre, law_bert)
search_encoded_vector = online_search_vector(genre, items)

# 3. Get original Semantic Embeddings
semantic_outputs, _ = law_bert.search(user_query = my_query, top_k = 10, index = index_1)
candidate_plots = [x['judgment_summary'] for x in semantic_outputs]
embeddings = law_bert.model.encode(candidate_plots)

# 4. Compute Cosine-sim for each sentence with search_vector
cosine_scores = util.pytorch_cos_sim(search_encoded_vector, embeddings)
for (i, score) in zip(semantic_outputs, cosine_scores.numpy()[0]):
    i['search_score'] = score

# 5. Sorting by search_score
online_outputs = sorted(semantic_outputs, key=lambda x: x['search_score'], reverse = True)

print("\n  ========== <<  Online-based Filtering  >> ==========  \n")
for i, out in enumerate(online_outputs):
   #print("\n === Law Cases ===")
   print("\n   Top {} - Case name (Number) : {} ({})  \n | Cluster : {} \n | Cluster's Topics (Keywords) : {} \n | Date : {} | Judgment Issue : {} \n | Judgment Summary : {}".format(i+1, out['case_name'], out['case_number'],
                                                                                                                                                  out['Topic'], out['Topic_Modeling'],
                                                                                                                                                  out['date'], out['judgment_issue'],
                                                                                                                                                  out['judgment_summary']))

# Final Results table
print("\n")
print("\n >> Write your case :", my_query)
print("\n")

# Case name (Case number) : Judgment issue (cluster_name) 
results = pd.DataFrame()
results['Only Semantic Search'] = [item['case_name'] + str(" (" + item['case_number']+")" " : ") + item['judgment_issue'] for item in original_outputs]
results['Popularity-based Semantic Search'] = [item['case_name'] + str(" (" + item['case_number']+")" " : ") + item['judgment_issue'] for item in p_outputs]
results['User-based Semantic Search'] = [item['case_name'] + str(" (" + item['case_number']+")" " : ") + item['judgment_issue'] for item in user_outputs]
results['Online-based Semantic Search'] = [item['case_name'] + str(" (" + item['case_number']+")" " : ") + item['judgment_issue'] for item in online_outputs]
results