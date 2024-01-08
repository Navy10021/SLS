![header](https://capsule-render.vercel.app/api?type=waving&color=auto&height=300&section=header&text=Semantic%20Legal%20Searcher&fontSize=70)

# LLMs-Based Legal Info Retrieval : A Neural Semantic Search Framework with Multi-Interaction Tech and KRLawBERT/GPT
<img src="https://img.shields.io/badge/Semantic Search-3776AB?style=flat-square&logo=Source Engine&logoColor=white"/> <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"/> <img src="https://img.shields.io/badge/Colab-3776AB?style=flat-square&logo=Google Colab&logoColor=white"/> 

### 0. Abstract
 In this work, we introduce a pioneering Neural Semantic Search Framework designed for legal information retrieval, known as ***Semantic Legal Searcher (SLS)***. The framework utilizes state-of-the-art neural information retrieval techniques to deliver precise legal information responses for user queries, irrespective of their legal background. This work encompasses a distinctive architecture, incorporating the ***split-merge technique*** for enhanced document embeddings and a ***multi-interactions*** paradigm to elevate semantic similarity measures. Additionally, the ***SLS*** framework integrates two pre-trained Legal Language Models(LLMs), ***KRLawBERT and KRLawGPT***, specifically tailored for legal texts, surpassing the capabilities of conventional NLP models like **BERT**.

### 1. Model Description

### 1.1 Semantic Legal Searcher(SLS)
The ***SLS*** is presented as a groundbreaking conceptual search model based on neural information retrieval, aiming to revolutionize legal information accessibility. Unlike common methods in Figure 1 (a), we extend our search model by introducing two information retrieval techniques shown in (b). The ***SLS*** employs an optimized neural Information Retrieval (IR) approach for legal datasets, incorporating the ***split-merge technique*** to maximize information retention in document embeddings. The ***multi-interactions technique*** enhances semantic similarity measures, establishing ***SLS*** as a comprehensive search framework that combines semantic search and topic modeling for relevance identification and keyword extraction.
 
![F_1](https://user-images.githubusercontent.com/105137667/206842983-1a5438d0-cd1c-4d77-991b-e63cacba4e66.jpg)

### 1.2. KRLawBERT and KRLawGPT
**BERT(Bidirectional Encoder Representations from Transformers)** and **GPT(Generative Pre-trained Transformer)** are both NLP models capable of understanding the semantic meaning of the text and have been applied to a variety of tasks including text classification, entity recognition, and more. 

***KRLawBERT***, developed through benchmarking Masked Language Modeling (MLM) and Transformer-based Sequential Denoising Auto-Encoder (TSDAE), outperforms conventional BERT-based models in capturing sentence text similarity for legal information retrieval. 

☞ https://github.com/Navy10021/KRLawBERT

Additionally, ***KRLawGPT***, a decoder-only transformer, is introduced for generating expressive Korean legal text, broadening the language model capabilities.

☞ https://github.com/Navy10021/KRLawGPT

 ### 2. Model Overall Pipeline(SLS Process)
 
 The process of the ***SLS*** is divided into four steps as shown in Figure 2 : document encoding and split-merge modelization, parallel clustering-based topic modeling for keyword extraction, multi-interactions for relevance estimation, and the provision of user search results based on relevance scores. Each step is intricately designed to optimize the efficiency and effectiveness of legal information retrieval.

<p align="center"><img src="https://user-images.githubusercontent.com/105137667/206843022-300ada12-e43e-4af9-b3a1-cd4d08b45cb2.jpg" width="500" height="600"/></p>

### 3. Model Usage

#### STEP 1 : Load Pre-trained Legal Language Models (LLMs)
 
Users have the flexibility to utilize existing models such as **BERT** or opt for the specialized ***KRLawBERT***, pre-trained on a large-scale legal text dataset, enhancing performance for legal information retrieval.
```python
import pandas as pd
# Load dataset(Cornell University., 2022)
df = pd.read_csv('./data/arxiv_meta.csv')

# Load pre-trained language model on English dataset
my_plms = "all-mpnet-base-v2"
```

***KRLawBERT*** is a pre-trained Transformer's encoder-based model to generate high-quality embeddings and better understand texts in legal domains.

```python
import pandas as pd
# Load Korean Legal Dataset(Korean Judicial precedent data)
df = pd.read_csv('./data/law_cases(20221020).csv')

# Load pre-trained & fine-tuned models (KRLawBERT)
my_plms = './output/tsdae-krlawbert'
```

#### STEP 2 : Keywords Extraction with Parallel Clustering-based Topic Modeling

A novel parallel clustering-based topic modeling technique is introduced, ensuring swift and efficient extraction of latent keywords from legal documents.

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
# Import parallel_clustering-based topic modeling model
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

The research highlights the significance of the ***split-merge embedding modelization technique*** and the ***multi-interactions scoring mechanism***, showcasing a substantial improvement of 14-20% in semantic search accuracy. The SLS class facilitates these processes for both English and Korean legal datasets.

**SLS class**
 - dataframe : Dataframe based table
 - doc_col : Documents columns (str)
 - key_col : Keywords columns (str)
 - model_name : PLMs name (str)
 - use_sentence_bert : Whether to ***generate sentence embeddings*** or not (bool)
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

Users can seamlessly initiate semantic searches by inputting queries, leveraging the comprehensive SLS framework to retrieve relevant legal documents.

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

### 4. About Research Paper
 - Currently, our paper is under review. It will be revealed in the future.
 
 
### 5. Development Team
 - Seoul National University NLP Labs
 - Under the guidance of Navy Lee
