# Semantic Legal Searcher: Faster Natural Language-based Semantic Search for Case Law

 We propose Semantic Legal Searcher which is a new conceptual case law search procedure. Our main contributions are as follows:
 
 1.	We introduce a Clean Korean Legal Corpus (CKLC). This corpus consists of 2.2 million sentences of Korean legal text published from the year of 1954 to the present year and they were pre-processed.
 2.	We create KoLawBERT, pre-trained language models trained on the Clean  Korean Legal Corpus by applying various masked language modeling (MLM) methods. Training with MLM allows language models to better understand the language in a more specific domain. KoLawBERT was shaped with three popular masking techniques: BERT static masking, Roberta dynamic masking, and ALBERT parameter-sharing.
 3.	We design the Semantic Legal Searcher framework by combining semantic document search with cluster-based topic modeling. Topic modeling is a method to extract topics within documents. Our model finds semantically similar precedents by matching the similarity between document embeddings and query embeddings, and at the same time extracts representative keywords and topics of each precedent through topic modeling.
 4.	We provide enhanced search results through Dynamic post-filtering. When a user searches for relevant case law, the system dynamically re-ranking the search results based on the search popularity, user, and search volumes.

 Semantic Legal Searcher can provide users with more substantial and diverse information regardless of whether they are lawyers or not. Moreover, we have verified experimentally the practicality of the model by testing for both lawyers and ordinary students without legal domain knowledge.

 Furthermore, the Semantic Legal Searcher framework is not limited to the Korean language and the fields of Law. This framework we designed can be applied to multilingual datasets and extended to the various areas of specialization such as medical and financial sectors because it is a vector-based architecture consisting of cluster-based topic modeling and semantic document search. By separating the process of parallel clustering, generating topic representations, semantic search, and dynamic post-filtering, flexibility can be given in the model allowing for ease of usability.
 

 ## 1. Overall Pipeline
 
 The basic process of the Semantic Legal Searcher is divided into four steps as shown in Figure 1.  In the first step, each document in the legal database is converted into the form of embeddings using PLM. In the next step, these embeddings are parallelly clustered and representations of topics and keywords are extracted from clusters using a class-based TF-IDF formula. In the third step, the relevance between the query vector and legal document embeddings is measured by Euclidean-distance or Dot-product. Lastly, to provide users with optimal search results, relevant precedents are re-ranked through dynamic post-filtering.

![figure_1](https://user-images.githubusercontent.com/105137667/185838080-acf61d17-c615-48cd-bc2d-93911dde7c8b.jpg)


## 2. Semantic Legal Embeddings

 Any other embedding techniques can be used for this step if the language model generating the document embedding is fine-tuned on semantic similarity. In this study, we created pre-trained KoLawBERT models applying various BERT-based models’ language masking techniques such as original BERT static masking (2018),  Roberta dynamic masking (2019), and ALBERT masking (2019) with cross-layer parameter sharing and factorized embedding parameterization. After this, we converted KoLawBERT into sentence-BERT by adding a pooling layer and fine-tuning on both Korean Natural Language Inference (KorNLI) and Korean Semantic Textual Similarity (KorSTS) datasets for making semantically meaningful embeddings.
 

![semantic_legal_emb](https://user-images.githubusercontent.com/105137667/185838110-2235cce2-1b06-4b3e-bdae-ef22456de863.jpg)


## 3. Cluster-based Simple Topic Modeling

 Clustering-based topic modeling is using a clustering framework with contextualized document embeddings for topic modeling. Adding a clustering-based topic modeling in the semantic document search process has two advantages. First, the user can see not only the similarity between the query input and the search results but also the semantic relationships between the search results provided. Second, the latent space representation of the search result is explainable since latent topics and keywords in clustered data are discovered through clustering-based topic modeling. We develop a simple cluster-based topic modeling focused on speed.
 
 
![topic_modeling](https://user-images.githubusercontent.com/105137667/185838139-d6ed8874-2715-48e1-98ef-1cb8317f7a19.jpg)
 
 
Experimental results demonstrate that our parallel clustering is faster and more coherent in document embeddings clustering than other famous clustering methods such as K-means, Agglomerative Clustering, DBSCAN, and HDBSCAN.


![parallel_clustering_speed](https://user-images.githubusercontent.com/105137667/172763944-19bf4646-861b-432c-8e71-84dc95bf80a5.jpg)


## 4. Dynamic Post-Filtering
 Post-filtering is meant for re-ranking the search results in response to the user’s request after measuring embeddings relevance. Dynamic post-filtering is the system that converts originally searched case law into enhanced results through the following three different post-filtering techniques: Popularity-based filtering, User-based filtering, and Online-based filtering. They dynamically filter the case law based on the precedent views, user, and search volume.
 

## 5. Evaluation

Rank-less Recommendation metrics.


![results_metrics](https://user-images.githubusercontent.com/105137667/174745498-ed65fda1-493b-4ae1-80c5-21e0e34db4ef.jpg)
