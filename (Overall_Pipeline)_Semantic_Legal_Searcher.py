#! pip install transformers
#! pip install -U sentence-transformers
#! pip install sentencepiece
#! pip install faiss-gpu
#! pip install funcy pickle5

########################################
# STEP 1 : Load DataFrame & KRLawBERT  #
########################################
import pandas as pd
# 1. Load Legal Dataset(Get Judicial precedent data)
df = pd.read_csv('./data/law_cases(20221020).csv')
df = df.dropna(how = 'any')

# 2. Make index for clustering
print(">> Law cases data size : ",len(df))

# 3. Load pre-trained & fine-tuned models(KRLawBERT)
my_plms = './output/tsdae-krlawbert'

#####################################################
# STEP 2 : Parallel Clustering-based Topic Modeling #
#####################################################
from models.parallel_clustering_TM import *

# 1. Semantic Legal Embedding
target_text = 'judgment_summary'

cluster = ParallelCluster(
    dataframe = df,
    tgt_col = target_text,
    model_name = my_plms,
    use_sentence_bert = True
    )

# 2. Parallel Clustering
clusters, unclusters = cluster.parallel_cluster(
    clusters = None,
    threshold = 0.71,
    page_size = 1000,
    iterations = 20
    )

# 3. Stack : Stack the clustered results in order of cluster size
col_list = ['case_name', 'case_number', 'date', 'judgment_issue', 'judgment_summary']
new_df = cluster.cluster_stack(
    col_list = col_list,
    clusters = clusters,
    unclusters = unclusters
    )

# 4. Extract Keywords from each documents
top_n_words = cluster.extract_top_n_words_per_topic(
    dataframe = new_df,
    n = 10,
    en = False
    )
new_df['keywords'] = [', '.join(top_n_words[i]) for i in new_df['Topic'].values]

# 5. Save the Parallel Clusted Dataset 
new_df.to_csv("./data/clusted_df.csv", sep=',', na_rep="NaN")


#################################################################################
# STEP 3 : Embeddings modelization(Split-merge) and Scoring(Multi-interactions) #
#################################################################################
from models.semantic_legal_searcher import *

# Dataframe with keyword extraction from Parallel clustering-based TM
#new_df = pd.read_csv('./data/clusted_df.csv')

# 1. Obtain query, documents, keywords embeddings from KRLawBERT models
my_plms = './output/tsdae-kolawbert'
sls = SLS(
    dataframe = new_df,
    doc_col = 'judgment_summary',
    key_col = 'keywords',
    model_name = my_plms,
    use_sentence_bert = True,
    split_and_merge = True,
    multi_inter = True,
    )

# 2. Build the Index
# (Strategy 1) : All Distance Metric
all_index = sls.all_distance_metric()
# (Strategy 2) : Restricted Distance Metric
restricted_index = sls.restricted_distance_metric(nlist = 200, nprobe = 6)


##############################################
# STEP 4 : Semantic law case search with SLS #
##############################################

# 3. Semantic case law search (Legal Question-Answering)
my_query = "마약 불법거래 및 운반에 관한 판례"

outputs, _ = sls.semantic_search(
    user_query = my_query,
    top_k = 5,
    index = all_index,
    print_results = True,
    )
