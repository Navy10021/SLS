import time
import torch
import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, models

class SLS():
    def __init__(self, dataframe, doc_col, key_col, model_name, use_sentence_bert, split_and_merge, multi_inter):
        """
        dataframe : text Dataframe (Dataframe)
        doc_col : Document column (str)
        key_col : Extracted Keywords column (str)
        model_name : pre-trained language model (str)
        use_sentence_bert : sentence-level embedding or not (bool)
        split_and_merge : Embedding Modelization (bool)
        multi_inter : Multi-interactions or not (bool)
        """

        self.dataframe = dataframe
        self.doc_col = doc_col
        self.key_col = key_col
        self.model_name = model_name
        self.use_sentence_bert = use_sentence_bert
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # PLMs
        self.model = self.PLMs()
        # Gate formula
        self.gate_layer = torch.nn.Linear(768 * 2, 768)
        self.sigmoid = torch.nn.Sigmoid()
        # Documents embeddings
        # 1. Using merged sentence-level embeddings
        if split_and_merge:
            self.docs_embeddings = self.sent_embeddings_sum()
        # 2. Using document-level embeddings
        else:
            self.docs_embeddings = self.get_docs_embeddings()
        # Keywords embeddings
        self.keys_embeddings = self.get_keys_embeddings()
        # Finial embeddings
        # 1. using multi-interactions
        if multi_inter:
            self.fin_embeddings = self.multi_interactions()
        # 2. using single-interactions
        else:
            self.fin_embeddings = self.docs_embeddings

    ###############################
    # Pre-trained Language models #
    ###############################
    def PLMs(self):
        if self.use_sentence_bert:
            model = SentenceTransformer(self.model_name)
        else:
            word_embedding_model = models.Transformer(self.model_name)
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                           pooling_mode_mean_tokens = True,
                                           pooling_mode_cls_token = False,
                                           pooling_mode_max_tokens = False)
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=self.device)

        return model   

    ##########################
    # Embedding Modelization #
    ##########################

    # 1. Embedding Modelization with Split and Merge formula
    def gate_sum(self, vec_a, vec_b):
        concat = torch.cat((vec_a, vec_b), -1)
        context_gate = self.sigmoid(self.gate_layer(concat))
        return torch.add(context_gate * vec_a, (1.-context_gate)*vec_b)

    def sent_embeddings_sum(self):
        doc_list = self.dataframe[self.doc_col].tolist()
        split_list = [[sent.strip() + '.' for sent in doc.split('.')] for doc in doc_list]
        new_embeddings = []
        for idx, sentences_split in enumerate(split_list):
            sent_embeddings = self.model.encode(sentences_split)
            sent_embeddings = torch.from_numpy(sent_embeddings)
            embedding_sum = sent_embeddings[0]
            for emb in sent_embeddings[1:]:
                embedding_sum = self.gate_sum(embedding_sum, emb)
            new_embeddings.append(embedding_sum.detach().numpy())

        docs_embeddings = np.asarray(new_embeddings, dtype = np.float32)
        print(">> Split and Merage embeddings shape(Items x PLMs_dim) :", docs_embeddings.shape)
        return docs_embeddings
        
    # 2. Generate directly documents-level embedding
    def get_docs_embeddings(self):
        embeddings = self.model.encode(self.dataframe[self.doc_col].tolist(), show_progress_bar=True)
        embeddings = np.asarray(embeddings.astype('float32'))
        print(">> Documents embeddings shape(Items x PLMs_dim) :", embeddings.shape)

        return embeddings
    
    # 3. Generate keywords embedding
    def get_keys_embeddings(self):
        embeddings = self.model.encode([' '.join(i.split(', ')) for i in self.dataframe[self.key_col]], show_progress_bar=True)
        embeddings = np.asarray(embeddings.astype('float32'))
        print(">> Keywords embeddings shape(Items x PLMs_dim) :", embeddings.shape)
        
        return embeddings
    
    # Softmax function
    def softmax(self, vector):
        e_x = np.exp(vector - np.max(vector))
        return e_x / e_x.sum()
    
    # 4. Multi-interactions formula
    def multi_interactions(self, weight = 0.25):
        docs = self.docs_embeddings
        keys = self.keys_embeddings
        scores = np.add(keys*weight, docs*(1-weight))
        return scores


    #############################################################
    # Measure the Relevance of Embeddings with Distance Metrics #
    #############################################################

    # Strategy 1 : Calculate Similarity between query and All text of Embeddings
    def all_distance_metric(self, calculator = "L2", save = False, save_name = 'all.index'):
        """
        calculator : str('L2(Euclidean distance' or 'IP(Inner product)')
        save : bool(Save index or not)
        """
        dim = self.fin_embeddings.shape[1]
        num_items = self.fin_embeddings.shape[0]

        # Euclidean Distance
        if calculator == "L2":
            index = faiss.IndexFlatL2(dim)
            index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))  
        # Dot-porduct
        else:
            index = faiss.IndexFlatIP(dim)
            index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        # Add the abstract vectors(encoded data) and their ID mapping to the index.
        index.add_with_ids(self.fin_embeddings, np.array(range(0, num_items)))

        if save:
            faiss.write_index(index, save_name)
        
        return index

    # Strategy 2 : Calculate Similarity between query and Centroid of Embeddings
    def restricted_distance_metric(self, nlist, nprobe, save = False, save_name = 'all.index'):
        """
        nlist : The number of partition(Centroid)
        nprobe : N-closest cell from Centroid
        save : bool(Save index or not)
        * similarity calculator : Only L2-distance
        """
        dim = self.fin_embeddings.shape[1]
        num_items = self.fin_embeddings.shape[0]

        # L2-distance(Only)
        quantizer = faiss.IndexFlatL2(dim)
        # Build the index with IVFFlat
        index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        # Train sentence embeddings
        index.train(self.fin_embeddings)
        # Add the abstract vectors(encoded data) and their ID mapping to the index.
        index.add_with_ids(self.fin_embeddings, np.array(range(0, num_items)))
        # Improve accuracy by increasing the search scope. How many nearby cells to search.
        index.nprobe = nprobe

        if save:
            faiss.write_index(index, save_name)
        
        return index


    def fetch_info(self, df_idx):
        """
        df_idx : int
        cluster : bool
        """
        sim_dict = dict()
        info = self.dataframe.iloc[df_idx]
        sim_dict['Topic'] = info['Topic']
        sim_dict['keywords'] = info['keywords']
        sim_dict['case_name'] = info['case_name']
        sim_dict['case_number'] = info['case_number']
        sim_dict['date'] = info['date']
        sim_dict['judgment_issue'] = info['judgment_issue']
        sim_dict['judgment_summary'] = info['judgment_summary']
        #sim_dict['judgment_contents'] = info['judgment_contents']
        return sim_dict

    ###################
    # Semantic Search #
    ###################
    def semantic_search(self, user_query, top_k, index, print_results = False):
        """
        user_query : str(input query)
        top_k : int(top k based on similarity score)
        index : Final embeddings index from Faiss
        model : Sentence-Transformer
        cluster : bool(clustering or not)
        print : bool(print result or not)
        """
        start = time.time()
        query_vector = self.model.encode([user_query]) # (1, 768)
        # Query-Final embeddings similarity by FAISS
        top_similarity = index.search(query_vector, top_k)
        print("\n === Calculate run time : {} ms === \n".format(round((time.time()-start)*1000, 4)))
        top_similarity_ids = top_similarity[1].flatten().tolist()               # Top_query ids
        top_similarity_ids = list(np.unique(top_similarity_ids))                # Id unique cheack
        # Query-Final embeddings similarity scores
        similarity_score = top_similarity[0].flatten().tolist()                 # Similarity_Score
        result = [self.fetch_info(idx) for idx in top_similarity_ids]

        # Print
        if print_results:
            print(">> Write your case :", user_query)
            for i, out in enumerate(result):
                print("\n >> Top {} - Case name (Number) : {} ({})  \n | Cluster : {} \n | Extracted keywords : {} \n | Date : {} | Judgment Issue : {} \n | Judgment Summary : {}".format(i+1,
                                                                                                                                                                                                    out['case_name'],
                                                                                                                                                                                                    out['case_number'],
                                                                                                                                                                                                    out['Topic'],
                                                                                                                                                                                                    out['keywords'],
                                                                                                                                                                                                    out['date'],
                                                                                                                                                                                                    out['judgment_issue'],
                                                                                                                                                                                                    out['judgment_summary']))
        return result, similarity_score