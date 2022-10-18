# Semantic Legal Searcher: Faster Natural Language-based Semantic Search for Case Law

 We propose Semantic Legal Searcher which is a new conceptual case law search procedure. Our main contributions are as follows:
 
 1)	We introduce a Clean Korean Legal Corpus (CKLC). This corpus consists of 2.2 million sentences of Korean legal text published from 1954 to the present year and they were pre-processed.
 2) We release a legal language model named KoLawBERT that pre-trained or re-trained Transformer-based models on the CKLC by applying Masked Language Modeling (MLM) methods. Training with MLM allows language models to better understand the language in a more specific domain. KoLawBERT was shaped with three popular masking techniques: BERT static masking, Roberta dynamic masking, and ALBERT parameter-sharing. We also created a sentence-level PLM fine-tuned by an unsupervised pretraining method called Transformer-based Sequential Denoising Auto-Encoder (TSDAE).
 3) We design the Semantic Legal Searcher framework by combining semantic document search with cluster-based topic modeling. Topic modeling is a method to extract topics within documents. Our model finds semantically similar precedents by matching the similarity between document embeddings and query embeddings, and at the same time extracts representative keywords and topics of each precedent through topic modeling.
 4)	We provide enhanced search results by way of Dynamic post-filtering. When a user searches for relevant case law, this system dynamically re-ranking the search results based on the search popularity, user, and search volumes.

 Semantic Legal Searcher can provide users with more substantial and diverse information regardless of whether they are lawyers or not. Moreover, we have verified experimentally the practicality of the model by testing for both lawyers and ordinary students without legal domain knowledge.
 
 Furthermore, the Semantic Legal Searcher framework is not limited to the Korean language and the fields of Law. This framework we designed can be applied to multilingual datasets and extended to the various areas of specialization such as medical and financial sectors because it is a vector-based architecture consisting of cluster-based topic modeling and semantic document search. By separating the process of parallel clustering, generating topic representations, semantic search, and dynamic post-filtering, flexibility can be given in the model allowing for ease of usability.
 

 ## 1. Overall Pipeline
 
 The basic process of the Semantic Legal Searcher is divided into four steps as shown in Figure 1.  In the first step, each document in the legal database is converted into the form of embeddings using the PLMs we designed. In the next step, these embeddings are parallelly clustered and representations of topics and keywords are extracted from clusters using a class-based TF-IDF formula. In the third step, the relevance between the query vector and legal document embeddings is measured by Euclidean-distance or Dot-product. Lastly, to provide users with optimal search results, relevant precedents are re-ranked through dynamic post-filtering.

![overall_2](https://user-images.githubusercontent.com/105137667/195964428-cd0b99ca-657b-437e-b896-ad6203c3a249.jpg)


## 2. KoLawBERT

 We can use existing PLMs such as BERT in the Semantic Legal Searcher framework. However, this way is too simple and often not competitive with other methods regarding search accuracy in the legal field. Therefore, we release pre-trained language models, KoLawBERT, trained on our legal corpus by applying the popular three masked language modeling methods: BERT, Roberta and ALBERT.

## 3. Semantic Legal Embeddings

 Any other embedding learning techniques can be used at this stage if the language model leads to generating semantically similar embeddings. In this study, We approach two classes of methods for embodying semantic embeddings: Supervised training with KoLawBERT and Unsupervised training without KoLawBERT.
 
  - Supervised Training: The first strategy for sentence embedding learning is leveraging siamese and triplet networks (Schroff et al., 2015) to derive a long text into semantic vector space efficiently. Natural Language Inference (NLI) and Semantic Textual Similarity (STS) is the most common supervised approach to fine-tuning semantic embedding. Both NLI and STS datasets contain sentence pairs labeled. The language model learns how to distinguish between similar and dissimilar sentence pairs using the optimization functions like softmax loss or cosine similarity loss (Reimers and Gurevych., 2019).
 
  - Unsupervised Training: Another approach is to perform the Transformer-based Sequential Denoising Auto-Encoder (TSDAE) pretraining method (Kexin Wang., 2021). TSDAE introduces noise to input text by removing about 60% of word-level tokens. These damaged sentences are encoded by the Transformer encoder into sentence vectors and then another decoder network attempts to predict the original input text from the damaged encoding vector. 


![semantic_legal_emb_2](https://user-images.githubusercontent.com/105137667/195964435-ddfd7782-3541-4b12-b0cf-45cea5a182a7.jpg)



## 4. Cluster-based Simple Topic Modeling

 Clustering-based topic modeling is using a clustering framework with contextualized document embeddings for topic modeling. Adding a clustering-based topic modeling in the semantic document search process has two advantages. First, the user can see not only the similarity between the query input and the search results but also the semantic relationships between the search results provided. Second, the latent space representation of the search result is explainable since latent topics and keywords in clustered data are discovered through clustering-based topic modeling. We develop a simple cluster-based topic modeling focused on speed.
 
![topic_modeling](https://user-images.githubusercontent.com/105137667/195964449-495c48b7-4695-4225-b295-abaee39d7355.jpg)

### Class-based TF-IDF Formula

![c-tf-idf](https://user-images.githubusercontent.com/105137667/195963697-5bac9855-02f6-4c82-9f9c-2e363828e419.jpg)
 

### Parallel Clustering Algorithm
![parallel_clustering_algorithm](https://user-images.githubusercontent.com/105137667/195963681-8ea7bd18-39fd-44de-9f15-c2ac2b75a559.jpg)

As a result of the parallel clustering contextual embeddings, legal documents are grouped into semantically similar documents and rearranged by cluster size. Experimental results evaluated on the MovieLens dataset demonstrate that our parallel clustering is faster and more coherent in document embedding clustering than other famous clustering methods such as K-means, Agglomerative Clustering, DBSCAN, and HDBSCAN.


![image](https://user-images.githubusercontent.com/105137667/187072253-d44af179-807b-4c99-84c7-fbd27eb67e0c.png)

## 5. Dynamic Post-Filtering
 Post-filtering is meant for re-ranking the search results in response to the user’s request after measuring embeddings relevance. Dynamic post-filtering provides the improvement over original searched results by way of the following three different post-filtering techniques: Popularity-based filtering, User-based filtering, and Online-based filtering. They dynamically filter the case law based on the precedent views, user, and search volume.
 

## 6. Evaluation

Rank-less Recommendation metrics.

![results_metrics](https://user-images.githubusercontent.com/105137667/188052022-3034afa7-a793-46f3-84ee-12e83a3fd701.jpg)



