#! pip install transformers
#! pip install -U sentence-transformers
#! pip install sentencepiece
#! pip install faiss-gpu
#! pip install funcy pickle5

######################################
# STEP 1 : Load Dataset & PLMs (Eng) #
######################################
import pandas as pd

# 1. Load arXiv dataset(Cornell University., 2022)
df = pd.read_csv('./data/arxiv_meta.csv')
print(">> arxiv-meta data size : ", len(df))

# 2. Load pre-trained language model on English dataset
my_plms = "all-mpnet-base-v2"


#####################################################
# STEP 2 : Parallel Clustering-based Topic Modeling #
#####################################################
from models.parallel_clustering_TM import *

# 1. Obtain Embeddings
target_text = 'abstract'

cluster = ParallelCluster(
    dataframe = df,
    tgt_col = target_text,
    model_name = my_plms,
    use_sentence_bert = True
    )

# 2. Parallel Clustering
clusters, unclusters = cluster.parallel_cluster(
    clusters = None,
    threshold = 0.52,
    page_size = 2000,
    iterations = 20
    )

# 3. Stack : Stack the clustered results in order of cluster size
col_list = ['title', 'abstract', 'year']
new_df = cluster.cluster_stack(
    col_list = col_list,
    clusters = clusters,
    unclusters = unclusters
    )

# 4. Extract Keywords from each documents
top_n_words = cluster.extract_top_n_words_per_topic(
    dataframe = new_df,
    n = 20,
    en = True
    )
new_df['keywords'] = [', '.join(top_n_words[i]) for i in new_df['Topic'].values]

# 5. Save the Parallel Clusted Dataset 
new_df.to_csv("./data/clusted_arxiv_df.csv", sep=',', na_rep="NaN")


#################################################################################
# STEP 3 : Embeddings modelization(Split-merge) and Scoring(Multi-interactions) #
#################################################################################
from models.semantic_searcher_eng import *

# 1. Load SLS framework
sls = SLS(
    dataframe = new_df,
    doc_col = 'abstract',
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
#restricted_index = sls.restricted_distance_metric(nlist = 200, nprobe = 6)


#####################################
# STEP 4 : Semantic search with SLS #
#####################################
# 3. Semantic documents search (Question-Answering)
my_query = "Research about the Transformer network architecture, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely."

outputs, _ = sls.semantic_search(
    user_query = my_query,
    top_k = 10,
    index = all_index,
    print_results = True,
    )
