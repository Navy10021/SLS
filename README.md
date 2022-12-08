# Semantic Legal Searcher: Neural Information Retrieval-based Semantic Search for Case Law

 In this work, we propose a Semantic Legal Searcher (SLS) which is a new conceptual search model based on neural information retrieval. Semantic Legal Searcher can find accurate legal information for users' queries, regardless of whether the user is a lawyer or not. 
 
 The architecture of Semantic Legal Searcher (SLS) is a new neural IR approach optimized for legal datasets as shown in (b). Unlike common methods (a), we extend our search model by introducing two information retrieval techniques. First, a split and merge technique is introduced to contain as much document information as possible in embeddings. We perform additional embedding modelization that splits each document into sentences and merges encoded sentence-level embeddings to minimize the loss of information in converting the whole document text into embedding. Secondly, a multi-interactions technique is introduced to improve the quality of semantic similarity measures. SLS is a search framework that combines semantic search and topic modeling to find relevant documents and simultaneously can extract keywords from each document. Therefore, it is possible to generate keyword embedding in SLS. The multi-interaction paradigm is that input queries, documents, and keywords are encoded into vectors and then relevance is measured not only by two sets of vectors from queries and documents but also by keyword embeddings.
 
![image](https://user-images.githubusercontent.com/105137667/206459415-f5dab41f-1185-430e-8279-4c9703b76be5.png)

 ## 1. Overall Pipeline
 
 The process of the SLS is divided into four steps. In the first step, each document in the legal database is encoded into embeddings and then fulfilled embedding modelization called split and merge. In the next step, these embeddings are parallelly clustered quickly, and then keywords are extracted by our topic modeling technique. In the third step, named multi-interactions, both the relevance of the query vector to the legal document embeddings and to the keyword embeddings are estimated by distance metrics. Lastly, the model provides user search results based on their relevance score.

![Figure_2](https://user-images.githubusercontent.com/105137667/206458930-48d18d66-868a-4cb1-beba-ee9eafb2266c.jpg)


## 2. Clean Korean Legal Corpus(CKLC)
We created a Clean Korean Legal Corpus (CKLC), a new dataset of Korean legal texts. It is a pre-processed corpus consisting of 150 thousand cases of judicial decisions from the Supreme Court of Korea and statutes published from 1954 to the current year. The total number of sentences in CKLC is 5.3 million. The dataset consists of five distinct sections for each law case: 1) case name; 2) case number; 3) judgment issue, 4) judgment summary; 5) full-text; 6) label.


## 3. KoLawBERT

We can use existing PLMs such as BERT in the SLS framework. However, this way is less competitive in the field of legal information retrieval. Therefore, we release a KoLawBERT pre-trained on CKLC (§2.) by benchmarking the popular two techniques: Masked Language Modeling (MLM) and Transformer-based Sequential Denoising Auto-Encoder (TSDAE).

![Figure_3](https://user-images.githubusercontent.com/105137667/206460566-59fb3dba-2b58-45e5-9118-68c02d8792ac.jpg)


## 4. Parallel Clustering-based Topic Modeling

Topic modeling is an unsupervised method to extract latent keywords and uncover latent themes within documents. Clustering-based topic modeling is an advanced technique using various clustering frameworks with embeddings for topic modeling. We create a simple parallel clustering-based topic modeling focused on speed.

![Figure_4](https://user-images.githubusercontent.com/105137667/206460521-576070a3-7b13-4776-9d88-83181541f211.jpg)
 

## 5. Evaluation

We conducted three different NLP downstream tasks for evaluating performance of KoLawBERT in SLS framework: 1) Korean Natural Language Inference; 2) Korean Semantic Textual Similarity; 3) Legal Question Answering.

![Table_1](https://user-images.githubusercontent.com/105137667/206461094-dc5c3884-bc42-4a43-83e6-f536f26c1e6c.jpg)
