# Semantic Legal Searcher: Neural Information Retrieval-based Semantic Search for Case Law

 We propose Semantic Legal Searcher (SLS) which is a new conceptual case law search procedure. Our main contributions are as follows:
 
 1) We introduce a Clean Korean Legal Corpus (CKLC). This corpus consists of 5.3 million sentences of Korean legal text published from 1954 to the present year and they were pre-processed. 
 2) We release a language model named KoLawBERT that pre-trained Transformer-based models on the CKLC for generating high-quality legal embedding. We benchmark a series of state-of-the-art pre-training techniques: Masked Language Modeling (Devlin et al., 2018, Liu et al., 2019) and Transformer-based Sequential Denoising Auto-Encoder (Kexin Wang., 2021). These methods allow getting the word-level and sentence-level contextualized embeddings as well as creating language models for better understanding texts in specific legal domains.
 3) We design the Semantic Legal Searcher (SLS) framework by combining semantic document search with cluster-based topic modeling. Topic modeling is a method to extract topics within documents. Our model finds semantically similar precedents by matching the similarity between queries and documents, and at the same time extracts representative keywords of each precedent through topic modeling. Moreover, we introduce a new concept of neural IR called multi-interactions that further estimate the relevance between queries and extracted keywords.
 4)	We provide enhanced search results by way of Dynamic post-filtering. When a user searches for relevant case law, this system automatically re-ranking the search results based on the search popularity, user, and search volumes.

 SLS can find accurate legal information for users' queries, whether the user is a lawyer or not. In addition, we have verified experimentally the practicality of the model by choosing for three specific tasks: Natural language inference (Bowman et al., 2015; Williams et al., 2018), semantic textual similarity (Cer et al., 2017) and legal question answering tasks (Lawyer validation).
 
 Furthermore, the Semantic Legal Searcher framework is not limited to the Korean language and the fields of Law. This framework we designed can be applied to multilingual datasets and extended to the various areas of specialization such as medical and financial sectors because it is a vector-based architecture consisting of cluster-based topic modeling and semantic document search. By separating the process of parallel clustering, generating topic representations, semantic search, and dynamic post-filtering, flexibility can be given in the model allowing for ease of usability.
 

 ## 1. Overall Pipeline
 
 The basic process of the SLS is divided into four steps as shown in Figure 2. In the first step, each document in the legal database is converted into the form of embeddings using the PLMs we created. In the next step, these embeddings are parallelly clustered and representations of keywords are extracted from clusters using a class-based TF-IDF formula (Grootendorst., 2022). In the third step, named multi-interactions, both the relevance of the query vector to the legal document embeddings and to the keyword embeddings are estimated by distance metrics such as cosine similarity or Euclidean distance. Lastly, to provide users with optimal search results, relevant precedents are re-ranked through dynamic post-filtering.

![Figure_2](https://user-images.githubusercontent.com/105137667/202361000-9abe0071-e5d6-4966-b168-57daaf7b11a1.jpg)

## 2. Clean Korean Legal Corpus(CKLC)
We created a Clean Korean Legal Corpus (CKLC), a new dataset of Korean legal texts. It is a pre-processed corpus consisting of 1.12 million cases of judicial decisions from the Supreme Court of Korea and statutes published from 1954 to the current year. The total number of sentences in CKLC is 5.3 million, and Table 1 shows the basic text statistics. The dataset consists of five distinct sections for each law case: 1) case name; 2) case number; 3) judgment issue, 4) judgment summary; 5) full-text; 6) label.

## 3. KoLawBERT

We can use existing PLMs such as BERT in the Semantic Legal Searcher framework. However, this way is too simple and often not competitive with other methods regarding search accuracy in the legal field. Therefore, we release a pre-trained language model, KoLawBERT, trained on CKLC by benchmarking the popular two techniques: Masked Language Modeling (MLM) and Transformer-based Sequential Denoising Auto-Encoder (TSDAE).

## 4. Semantic Legal Embeddings

To adapt the KoLawBERT to produce semantic legal embeddings, we typically need a more supervised fine-tuning approach. We use datasets like natural language inference (NLI) pairs, labeled semantic textual similarity (STS) data, or parallel legal data. Both NLI and STS datasets contain labeled sentence pairs. The parallel legal datasets consist of 1.2 million pairs of semantically similar legal sentences based on CKLC. Any other embedding learning techniques can be used at this stage if the language model leads to generating semantically similar embeddings. 

![Figure_3](https://user-images.githubusercontent.com/105137667/202361435-1c196824-7a26-48c5-b8e2-0d290d358904.jpg)


## 4. Parallel Clustering-based Topic Modeling

 Topic modeling is an unsupervised method to extract latent keywords and uncover latent themes within documents. Furthermore, clustering-based topic modeling is an advanced technique using various clustering frameworks with embeddings for topic modeling. Adding topic modeling in the semantic search process has distinct advantages in interpretability and search quality. First, representations of the search results are interpretable since literal topics and keywords in the latent vector space are discovered from embeddings clustering and topic modeling. Second, we can obtain not only document embeddings but also token-based keyword vector representation. Thus, we increase search accuracy by leveraging multi-interactions, which measures the relevance of not a single set of vectors from queries and documents but multi-sets of vectors by adding keywords embeddings. We create a simple parallel cluster-based topic modeling focused on speed.
 
![Figure_4](https://user-images.githubusercontent.com/105137667/202361557-69be0331-5558-4212-9d2c-00922aef87cb.jpg)

### Class-based TF-IDF Formula

![c-tf-idf](https://user-images.githubusercontent.com/105137667/202361719-507a6848-6ef3-4a95-a0cd-3e1c2b51429f.jpg)
 

### Parallel Clustering Algorithm

![c-tf-idf](https://user-images.githubusercontent.com/105137667/202361735-435e330a-9b2f-4c84-8c19-4c514220b3c4.jpg)

## 5. Dynamic Post-Filtering
 Post-filtering is meant for re-ranking the search results in response to the user’s request after measuring embeddings relevance. Dynamic post-filtering provides the improvement over original searched results by way of the following three different post-filtering techniques: Popularity-based filtering, User-based filtering, and Online-based filtering. They dynamically filter the case law based on the precedent views, user, and search volume.
 

## 6. Evaluation & Results

We conducted three different NLP downstream tasks for evaluating performance of KoLawBERT in Semantic Legal Searcher framework: 1) Korean Natural Language Inference; 2) Korean Semantic Textual Similarity; 3) Legal Question Answering.

 In Table, we show the performance of the language models on the Semantic Legal Searcher process. All of the language models we designed showed better performance than baseline. The TSDAE-based KoLawBERT achieved the highest score in NLI and STS tasks. That indicates the TSDAE-based model encodes semantically meaningful information better than others. In addition, evaluation results show that TSDAE-based KoLawBERT performs comparably well in legal question-answering tasks. Compared to the baseline, the metric scores of this KoLawBERT are dramatically up by 10 – 15% points. We also find that the multi-interactions mechanism is a good choice for information retrieval (IR) compared to any other recent popular neural IR method.
