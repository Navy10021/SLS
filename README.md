# Semantic-Law-case-Search-with-KoLawBERT

 We propose the search model Semantic Law case Search and provide a new conceptual legal search framework. Our main contributions are as follows:
 
 1. We introduce a Clean Korean Legal Corpus (CKLC), a data set consisting of 100.5K pre-processed Korean legal texts from all precedents ranging from the year 1954 to the present, and contracts of large corporations.
 2. We create a KoLawBERT pre-trained language model trained with the Clean  Korean Legal Corpus (CKLC) by applying various masked language modeling (MLM) methods. Training with MLM allows language models to better understand the language in a more specific domain. KoLawBERT model was shaped with three popular masking techniques: BERT static masking, Roberta dynamic masking, and ALBERT parameter-sharing.
 3. We design a Semantic Law case Search framework by combining semantic document search with cluster-based topic modeling. Topic modeling is an unsupervised method to extract semantic themes within documents. Our model finds semantically similar precedents by matching the similarity between document embeddings and query embeddings, and at the same time extracts representative keywords and topics of each precedent through topic modeling.

 Semantic Law case Search can provide users with more substantial and diverse information regardless of whether they are lawyers or not. Moreover, we have demonstrated the practicality of the model by conducting beta testing both with lawyers and with ordinary students without legal domain knowledge.


 ## 1.Overall Pipeline
 The process of Semantic Law case Search is divided into four steps. In the first step, each document in the legal database is converted into the form of embeddings using a pre-trained language model. In the next step, these embeddings are parallelly clustered and representations of topics and keywords are extracted from clusters using a class-based TF-IDF formula. In the third step, the relevance between the query vector and legal document embeddings is measured by Euclidean distance or Dot-product. Lastly, to provide users with optimal search results, relevant law cases are re-ranked through post-filtering.

![overall_pipeline](https://user-images.githubusercontent.com/105137667/172510181-53605f35-9d36-43c5-b7b3-de1a3cfbfea6.jpg)


## 2.Legal Document Embeddings
 Any other embedding techniques can be used for this step if the language model generating the document embedding is fine-tuned on semantic similarity. In this study, we created pre-trained KoLawBERT models applying various BERT-based models’ language masking techniques such as original BERT static masking (2018),  Roberta dynamic masking (2019), and ALBERT masking (2019) with cross-layer parameter sharing and factorized embedding parameterization. After this, we converted KoLawBERT into sentence-BERT by adding a pooling layer and fine-tuning on both Korean Natural Language Inference (KorNLI) and Korean Semantic Textual Similarity (KorSTS) datasets for making semantically meaningful embeddings.
![Doc_embeddings](https://user-images.githubusercontent.com/105137667/172509527-cd1625fa-20bb-4af8-ae74-373f791c17ae.jpg)


## 3. Cluster-based Simple Topic Modeling
 Clustering-based topic modeling is using a clustering framework with contextualized document embeddings for topic modeling. Adding a clustering-based topic modeling in the semantic document search process has two advantages. First, the user can see not only the similarity between the query input and the search results but also the semantic relationships between the search results provided. Second, the latent space representation of the search result is explainable since latent topics and keywords in clustered data are discovered through clustering-based topic modeling. We develop a simple cluster-based topic modeling focused on speed.
 
![parallel_clustering](https://user-images.githubusercontent.com/105137667/172509591-2b472591-2199-45d3-b0fe-d5a8617e5a1f.jpg)
 
Experimental results demonstrate that our parallel clustering is faster and more coherent in document embeddings clustering than other famous clustering methods such as K-means, Agglomerative Clustering, DBSCAN, and HDBSCAN.

![parallel_clustering_speed](https://user-images.githubusercontent.com/105137667/172509757-2b4cda3c-fd85-4ffc-a4e0-d6369a2071d5.jpg)


## 4. Evaluation

Rank-less Recommendation metrics.
