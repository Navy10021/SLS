# Semantic-Law-case-Search-with-KoLawBERT

 We propose the search model Semantic Law case Search and provide a new conceptual legal search framework. Our main contributions are as follows:
 
1. We introduce a Clean Korean Legal Corpus (CKLC). This corpus consists of 52.9 million words of a pre-processed Korean legal text published from the year 1954 to the present and they were pre-processed.
2. We create a KoLawBERT pre-trained language model trained with the Clean  Korean Legal Corpus (CKLC) by applying various masked language modeling (MLM) methods. Training with MLM allows language models to better understand the language in a more specific domain. KoLawBERT model was shaped with three popular masking techniques: BERT static masking, Roberta dynamic masking, and ALBERT parameter-sharing.
3. We design a Semantic Law case Search framework by combining semantic document search with cluster-based topic modeling. Topic modeling is an unsupervised method to extract semantic themes within documents. Our model finds semantically similar precedents by matching the similarity between document embeddings and query embeddings, and at the same time extracts representative keywords and topics of each precedent through topic modeling.

 Semantic Law case Search can provide users with more substantial and diverse information regardless of whether they are lawyers or not. Moreover, we have verified experimentally the practicality of the model by testing for both lawyers and ordinary students without legal domain knowledge.


 ## 1. Overall Pipeline
 The process of Semantic Law case Search is divided into four steps. In the first step, each document in the legal database is converted into the form of embeddings using a pre-trained language model. In the next step, these embeddings are parallelly clustered and representations of topics and keywords are extracted from clusters using a class-based TF-IDF formula. In the third step, the relevance between the query vector and legal document embeddings is measured by Euclidean distance or Dot-product. Lastly, to provide users with optimal search results, relevant law cases are re-ranked through post-filtering.


![overall_pipeline](https://user-images.githubusercontent.com/105137667/172763770-7bfae9e2-d868-4eb8-a6d0-efcb55fc2c46.jpg)


## 2. Legal Document Embeddings
 Any other embedding techniques can be used for this step if the language model generating the document embedding is fine-tuned on semantic similarity. In this study, we created pre-trained KoLawBERT models applying various BERT-based models’ language masking techniques such as original BERT static masking (2018),  Roberta dynamic masking (2019), and ALBERT masking (2019) with cross-layer parameter sharing and factorized embedding parameterization. After this, we converted KoLawBERT into sentence-BERT by adding a pooling layer and fine-tuning on both Korean Natural Language Inference (KorNLI) and Korean Semantic Textual Similarity (KorSTS) datasets for making semantically meaningful embeddings.
 
 
![Doc_embeddings](https://user-images.githubusercontent.com/105137667/172763860-ca50c83f-a10d-4b58-9f64-2e6c86bcfbdc.jpg)


## 3. Cluster-based Simple Topic Modeling
 Clustering-based topic modeling is using a clustering framework with contextualized document embeddings for topic modeling. Adding a clustering-based topic modeling in the semantic document search process has two advantages. First, the user can see not only the similarity between the query input and the search results but also the semantic relationships between the search results provided. Second, the latent space representation of the search result is explainable since latent topics and keywords in clustered data are discovered through clustering-based topic modeling. We develop a simple cluster-based topic modeling focused on speed.
 
 
![cluster_topic_modeling](https://user-images.githubusercontent.com/105137667/172763898-f5eba72e-a60b-4acb-b9d2-f0f9a6221da8.jpg)
 
 
Experimental results demonstrate that our parallel clustering is faster and more coherent in document embeddings clustering than other famous clustering methods such as K-means, Agglomerative Clustering, DBSCAN, and HDBSCAN.


![parallel_clustering_speed](https://user-images.githubusercontent.com/105137667/172763944-19bf4646-861b-432c-8e71-84dc95bf80a5.jpg)


## 4. Evaluation

Rank-less Recommendation metrics.
