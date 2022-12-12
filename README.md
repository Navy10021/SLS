![header](https://capsule-render.vercel.app/api?type=wave&color=auto&height=300&section=header&text=Semantic%20Legal%20Searcher&fontSize=70)

## Neural Information Retrieval-based Semantic Search for Case Law
<img src="https://img.shields.io/badge/Semantic Search-3776AB?style=flat-square&logo=Source Engine&logoColor=white"/> <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"/> <img src="https://img.shields.io/badge/Colab-3776AB?style=flat-square&logo=Google Colab&logoColor=white"/> 

### 1. Model Description

 In this work, we propose a ***Semantic Legal Searcher (SLS)*** which is a new conceptual search model based on neural information retrieval. ***Semantic Legal Searcher*** can find accurate legal information for users' queries, regardless of whether the user is a lawyer or not. 
 
 The architecture of ***Semantic Legal Searcher (SLS)*** is a new neural IR approach optimized for legal datasets as shown in Figure 1 (b). Unlike common methods Figure 1 (a), we extend our search model by introducing two information retrieval techniques. First, a ***split-merge*** technique is introduced to contain as much document information as possible in embeddings. In other words, we perform additional embedding modelization that splits each document into sentences and merges encoded sentence-level embeddings to minimize the loss of information in converting the whole document text into embedding. Secondly, a ***multi-interactions technique*** is introduced to improve the quality of semantic similarity measures. ***SLS*** is a search framework that combines semantic search and topic modeling to find relevant documents and simultaneously can extract keywords from each document. Therefore, it is possible to generate keyword embedding in ***SLS***. The ***multi-interactions*** paradigm is that input queries, documents, and keywords are encoded into vectors and then relevance is measured not only by two sets of vectors from queries and documents but also by keyword embeddings.
 
![F_1](https://user-images.githubusercontent.com/105137667/206842983-1a5438d0-cd1c-4d77-991b-e63cacba4e66.jpg)

 ### 2. Model Overall Pipeline
 
 The process of the ***SLS*** is divided into four steps as shown in Figure 2. In the first step, each document in the database is encoded into embeddings and then fulfilled embedding modelization called ***split-merge***. In the next step, these embeddings are parallelly clustered quickly with a **parallel clustering algorithm**, and then keywords are extracted by **our topic modeling technique**. In the third step, named ***multi-interactions***, both the relevance of the query vector to the legal document embeddings and to the keyword embeddings are estimated by distance metrics(e.i. cosine or Euclidean). Lastly, the model provides user search results based on their relevance score. 

<p align="center"><img src="https://user-images.githubusercontent.com/105137667/206843022-300ada12-e43e-4af9-b3a1-cd4d08b45cb2.jpg" width="500" height="600"/></p>

### 3. Model Usage

#### STEP 1 : Load pre-trained language models (PLMs)
 
You can use existing PLMs such as BERT or Sentence-BERT in the ***SLS*** framework. 


```python
import pandas as pd
# Load dataset(Cornell University., 2022)
df = pd.read_csv('./data/arxiv_meta.csv')

# Load pre-trained language model on English dataset
my_plms = "all-mpnet-base-v2"
```

Or you can use a language model called ***KRLawBERT*** pre-trained in Korean languages with a large scaled legal corpus.	We release a language model named KRLawBERT that pre-trained Transformer-based models to generate high-quality embeddings and better understand texts in legal domains.


```python
import pandas as pd
# Load Korean Legal Dataset(Korean Judicial precedent data)
df = pd.read_csv('./data/law_cases(20221020).csv')

# Load pre-trained & fine-tuned models (KRLawBERT)
my_plms = './output/tsdae-krlawbert'
```

#### STEP 2 : Keywords extraction with Parallel Clustering-based Topic Modeling

Topic modeling is an unsupervised method to extract latent keywords and uncover latent themes within documents. Clustering-based topic modeling is an advanced technique using various clustering frameworks with embeddings for topic modeling. We create a ***parallel clustering-based topic modeling*** technique focused on speed.

**1. Parallel Clustering class**
 - dataframe : Dataframe based table
 - tgt_col : Documents columns (str)
 - model_name : PLMs (str)
 - use_sentence_bert : Whether to generate sentence embeddings or not (bool)
 - threshold : threshold (float)
 - page_size : Number of initial centroids (int)
 - iterations : Max iterations (int)

**2. Keywords Extraction class**
 - dataframe : DataFrame including clustered documents
 - n : Number of keywords to extract (int)
 - en : Whether documents is English or not (bool)
 

```python
# import parallel_clustering-based topic modeling model
from models.parallel_clustering_TM import *

# Obtain Embeddings
target_text = 'abstract'
cluster = ParallelCluster(
    dataframe = df,
    tgt_col = target_text,
    model_name = my_plms,
    use_sentence_bert = True
    )

# Parallel Clustering
clusters, unclusters = cluster.parallel_cluster(
    clusters = None,
    threshold = 0.56,
    page_size = 2000,
    iterations = 30
    )
    
# Stack : Stack the clustered results in order of cluster size
col_list = ['title', 'abstract', 'year']
new_df = cluster.cluster_stack(
    col_list = col_list,
    clusters = clusters,
    unclusters = unclusters
    )
    
# Extract Keywords from each documents
top_n_words = cluster.extract_top_n_words_per_topic(
    dataframe = new_df,
    n = 20,
    en = True
    )
new_df['keywords'] = [', '.join(top_n_words[i]) for i in new_df['Topic'].values]
```

#### STEP 3 : Embedding Modelization, Scoring, and Indexing

We find that both the embedding modelization(***split-merge***) and scoring method(***the multi-interactions mechanisms***) help improve semantic search accuracy by 14 – 20%. It demonstrates that they are suitable approach in neural information retrieval.

**SLS class**
 - dataframe : Dataframe based table
 - doc_col : Documents columns (str)
 - key_col : Keywords columns (str)
 - model_name : PLMs name (str)
 - use_sentence_bert : Whether to generate sentence embeddings or not (bool)
 - split_and_merge : Whether to use ***split-merge*** embeddings modelization technique (bool)
 - multi_inter : Whether to use ***multi-interactions*** scoring technique (bool)


```python
# If you use the English dataset, import this semantic searcher model
from models.semantic_searcher_eng import *
# If you use the Korean case law dataset, import this semantic searcher model
from models.semantic_legal_searcher import *

# Load SLS framework
sls = SLS(
    dataframe = new_df,
    doc_col = 'abstract',
    key_col = 'keywords',
    model_name = my_plms,
    use_sentence_bert = True,
    split_and_merge = True,
    multi_inter = True,
    )

# Build the Index
# (Strategy 1) : All Distance Metric
all_index = sls.all_distance_metric()
# (Strategy 2) : Restricted Distance Metric
#restricted_index = sls.restricted_distance_metric(nlist = 200, nprobe = 6)
```

#### STEP 4 : Semantic Search

Now just enter your query and start searching for documents !

**semantic search function**
 - user_query : Your input query (str)
 - top_k : Number of documents related to your query (int)
 - index : Index (variable)
 - print_results : Whether to print search results or not (bool)


```python
# Semantic documents(arXiv) search
your_query = "Research about the Transformer network architecture, based solely on attention mechanisms."

outputs, _ = sls.semantic_search(
    user_query = your_query,
    top_k = 10,
    index = all_index,
    print_results = True,
    )
```
