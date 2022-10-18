import time
import torch
import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, models, util

class KoLawBERT():
    def __init__(self, dataframe, tgt_col, model_name, use_sentence_bert, cluster):
        
        self.dataframe = dataframe
        self.tgt_col = tgt_col
        self.model_name = model_name
        self.use_sentence_bert = use_sentence_bert
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # pre-trained LM
        self.model = self.get_models()
        # document-level embeddings
        self.embeddings = self.get_docs_embeddings()
        # cluster
        self.cluster = cluster
    
    def get_models(self):
        if self.use_sentence_bert:
            model = SentenceTransformer(self.model_name)
        else:
            word_embedding_model = models.Transformer(self.model_name)
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                           pooling_mode_mean_tokens=True,
                                           pooling_mode_cls_token=False,
                                           pooling_mode_max_tokens=False)
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=self.device)

        return model      


    def get_docs_embeddings(self):
        embeddings = self.model.encode(self.dataframe[self.tgt_col].tolist(), show_progress_bar=True)
        embeddings = np.asarray(embeddings.astype('float32'))
        print(">> Data embeddings shape(Items x Dimensionality) :", embeddings.shape)

        return embeddings

    # Strategy 1 : Calculate Similarity between query and All text of Embeddings
    def all_relevant_embedding(self, calculator="L2", save=False, save_name='all.index'):
        """
        calculator : str('L2(Euclidean distance' or 'IP(Inner product)')
        save : bool(Save index or not)
        """
        dim = self.embeddings.shape[1]
        num_items = self.embeddings.shape[0]

        # Euclidean Distance
        if calculator == "L2":
            index = faiss.IndexFlatL2(dim)
            index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))  
        # Dot-porduct
        else:
            index = faiss.IndexFlatIP(dim)
            index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        # Add the abstract vectors(encoded data) and their ID mapping to the index.
        index.add_with_ids(self.embeddings, np.array(range(0, num_items)))

        if save:
            faiss.write_index(index, save_name)
        
        return index

    # Strategy 2 : Calculate Similarity between query and Centroid of Embeddings
    def centroid_relevant_embedding(self, nlist, nprobe, save=False, save_name='all.index'):
        """
        nlist : The number of partition(Centroid)
        nprobe : N-closest cell from Centroid
        save : bool(Save index or not)
        * similarity calculator : Only L2-distance
        """
        dim = self.embeddings.shape[1]
        num_items = self.embeddings.shape[0]

        # L2-distance(Only)
        quantizer = faiss.IndexFlatL2(dim)
        # Build the index with IVFFlat
        index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        # Train sentence embeddings
        index.train(self.embeddings)
        # Add the abstract vectors(encoded data) and their ID mapping to the index.
        index.add_with_ids(self.embeddings, np.array(range(0, num_items)))
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
        if self.cluster:
            info = self.dataframe.iloc[df_idx]
            sim_dict['Topic'] = info['Topic']
            sim_dict['case_name'] = info['case_name']
            sim_dict['case_number'] = info['case_number']
            sim_dict['date'] = info['date']
            sim_dict['case_code'] = info['case_code']
            sim_dict['judgment_issue'] = info['judgment_issue']
            sim_dict['judgment_summary'] = info['judgment_summary']
            sim_dict['judgment_contents'] = info['judgment_contents']
            sim_dict['case_id'] = info['case_id']
            sim_dict['case_hits'] = info['case_hits']
            sim_dict['case_hits_norm'] = info['case_hits_norm']
            sim_dict['Topic_Modeling'] = info['Topic_Modeling']
        else:
            info = self.dataframe.iloc[df_idx]
            sim_dict['case_name'] = info['case_name']
            sim_dict['case_number'] = info['case_number']
            sim_dict['date'] = info['date']
            sim_dict['case_code'] = info['case_code']
            sim_dict['judgment_issue'] = info['judgment_issue']
            sim_dict['judgment_summary'] = info['judgment_summary']
            sim_dict['judgment_contents'] = info['judgment_contents']
            sim_dict['case_id'] = info['case_id']
            sim_dict['case_hits'] = info['case_hits']
            sim_dict['case_hits_norm'] = info['case_hits_norm']
        return sim_dict

    
    def search(self, user_query, top_k, index, print_results = False):
        """
        user_query : str(input query)
        top_k : int(top k based on similarity score)
        index : Faiss index
        model : Sentence-Transformer
        cluster : bool(clustering or not)
        print : bool(print result or not)
        """
        start = time.time()
        query_vector = self.model.encode([user_query]) # (1, 768)
        # Vector similarity from FAISS
        top_similarity = index.search(query_vector, top_k)
        print("\n === Calculate run time : {} ms === \n".format(round((time.time()-start)*1000, 4)))
        top_similarity_ids = top_similarity[1].flatten().tolist()               # Top_query ids
        top_similarity_ids = list(np.unique(top_similarity_ids))                # Id unique cheack
        similarity_score = top_similarity[0].flatten().tolist()                 # Similarity_Score
        result = [self.fetch_info(idx) for idx in top_similarity_ids]

        # Print
        if print_results:
            print(">> Write your case :", user_query)
            for i, out in enumerate(result):
                print("\n >> Top {} - Case name (Number) : {} ({})  \n | Cluster : {} \n | Cluster's Topics (Keywords) : {} \n | Date : {} | Judgment Issue : {} \n | Judgment Summary : {}".format(i+1, out['case_name'], out['case_number'],
                                                                                                                                                  out['Topic'], out['Topic_Modeling'],
                                                                                                                                                  out['date'], out['judgment_issue'],
                                                                                                                                                  out['judgment_summary']))
        return result, similarity_score