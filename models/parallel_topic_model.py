from sentence_transformers import SentenceTransformer, models, util
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import logging
import torch
from joblib import delayed
from torch import Tensor
from collections import defaultdict
from joblib import Parallel, delayed
from funcy import log_durations
import math
from sklearn.feature_extraction.text import CountVectorizer

class ParallelCluster():
    def __init__(self, dataframe, tgt_col, model_name, use_sentence_bert):
                
        dataframe['id'] = dataframe.index
        self.ids = dataframe.id
        self.dataframe = dataframe
        self.tgt_col = tgt_col
        self.model_name = model_name
        self.use_sentence_bert = use_sentence_bert
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # pre-trained LM
        self.model = self.get_models()
        # document-level embeddings
        embeddings = self.get_docs_embeddings()
        self.embeddings = {idx: embedding for idx, embedding in zip(self.ids, embeddings)}
        # cluster
    
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

    #################################
    # parallel clustering functions #
    #################################
    # 1. Help function
    def cos_sim(self, a, b):
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(np.array(a))
        if not isinstance(b, torch.Tensor):
            b = torch.tensor(np.array(b))
        if len(a.shape) == 1:
            a = a.unsqueeze(0)
        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)

        return torch.mm(a_norm, b_norm.transpose(0, 1))

    # Get ids's Embeddings
    def get_embeddings(self, ids, embeddings):
        return np.array([embeddings[idx] for idx in ids])

    # Reorder & filtering Cluster Based on "Cluster Head"
    def reorder_and_filter_cluster(self, cluster_idx, cluster, cluster_embeddings, cluster_head_embedding, threshold):

        # ReOrder Merged Cluster Heads[0] based on Cos-Sim 
        cos_scores = self.cos_sim(cluster_head_embedding, cluster_embeddings)  # Matrix with res[i][j]
        sorted_vals, indices = torch.sort(cos_scores[0], descending=True)
        # Fittering
        bigger_than_threshold = sorted_vals > threshold # tensor([True, True, False])
        indices = indices[bigger_than_threshold]        # tensor([1, 2])
        sorted_vals = sorted_vals.numpy()
        # Cluster_ids, [Cluster_head, Cos_Sim_scores]
        return cluster_idx, [(cluster[i][0], sorted_vals[i]) for i in indices]

    def get_ids(self, cluster):
        # [cluster_ids, [cluster_head, cos_sim_scores]]
        return [transaction[0] for transaction in cluster]


    # Reorder Many Clusters 
    def reorder_and_filter_clusters(self, clusters, embeddings, threshold, parallel):
        # Get Reorded Clusters useing Parallel(병렬처리)
        # parallel(n_jobs=number of Parallel)(delayed(Function)(parameters) for i in range(100))
        results = parallel(delayed(self.reorder_and_filter_cluster)(cluster_idx,
                                                                    cluster,
                                                                    self.get_embeddings(self.get_ids(cluster), embeddings),
                                                                    self.get_embeddings([cluster_idx], embeddings),
                                                                    threshold,) for cluster_idx, cluster in clusters.items())
        clusters = {k : v for k, v in results}
        return clusters

    def get_clustured_ids(self, clusters):
        clustered_ids = set([transaction[0] for cluster in clusters.values() for transaction in cluster])
        clustered_ids |= set(clusters.keys())
        return clustered_ids

    # Get ids from Many Clusters
    def get_clusters_ids(self, clusters):
        return list(clusters.keys())

    # Get ids from Non-Clusters
    def get_unclustured_ids(self, ids, clusters):
        clustered_ids = self.get_clustured_ids(clusters)
        unclustered_ids = list(set(ids) - clustered_ids)
        return unclustered_ids

    # Many Clusters(Sort based on Size)
    def sort_clusters(self, clusters):
        return dict(sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)) 

    # One Cluster(Sort based on Cos-Sim)
    def sort_cluster(self, cluster):
        return list(sorted(cluster, key=lambda x: x[1], reverse=True)) 


    def filter_clusters(self, clusters, min_cluster_size):
        return {k: v for k, v in clusters.items() if len(v) >= min_cluster_size}

    # Get Unique dictionary's keys
    def unique(self, collection):
        return list(dict.fromkeys(collection))

    # Get Unique dictionary's keys
    def unique_txs(self, collection):
        seen = set()
        return [x for x in collection if not (x[0] in seen or seen.add(x[0]))]

    # [1,2,3,4,5] ---> [[1,2],[3,4],[5]] divided by page_size == 2500 or 5000
    def chunk(self, txs, chunk_size):
        try:
            n = math.ceil(len(txs) / chunk_size)
            k, m = divmod(len(txs), n)
        except ZeroDivisionError:
            print("*** Precautions! Reduce the number of Iterations or the Threshold! ***")
        return (txs[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))   

    def nearest_cluster_chunk(self, chunk_ids, chunk_embeddings, cluster_ids, cluster_embeddings, threshold):
        # Get Cos_scores
        cos_scores = self.cos_sim(chunk_embeddings, cluster_embeddings)
        # Get Top_1 Score & Index
        top_val_large, top_idx_large = cos_scores.topk(k=1, largest=True)
        top_idx_large = top_idx_large[:, 0].tolist()
        top_val_large = top_val_large[:, 0].tolist()
        # Assignment based on threshold
        cluster_assignment = []
        for i, (score, idx) in enumerate(zip(top_val_large, top_idx_large)):
            cluster_id = cluster_ids[idx]
            if score < threshold:
                cluster_id = None
            cluster_assignment.append(((chunk_ids[i], score), cluster_id))

        # set((chunk_ids[i], socre), cluster_id)
        return cluster_assignment


    def nearest_cluster(self, transaction_ids, embeddings, clusters, parallel, threshold, chunk_size):
    
        cluster_ids = list(clusters.keys())
        if len(cluster_ids) == 0:
            return clusters
        # Get Cluster Embeddings
        cluster_embeddings = self.get_embeddings(cluster_ids, embeddings)
        # Divided by Chunk size
        c = list(self.chunk(transaction_ids, chunk_size))
        # Implement Parallel 
        with log_durations(logging.info, ">> Parallel jobs nearest cluster ..."):
            # out = list(((chunk_ids[i], score), cluster_id)...)
            out = parallel(delayed(self.nearest_cluster_chunk)(chunk_ids, self.get_embeddings(chunk_ids, embeddings), cluster_ids, cluster_embeddings, threshold) for chunk_ids in c)
        # cluster_assignment = [(chunk_ids[i], score), cluster_id, (chunk_ids[i], score), cluster_ids, ...]
            cluster_assignment = [assignment for sublist in out for assignment in sublist]
    
        # Sort in right order
        for (transaction_id, similarity), cluster_id in cluster_assignment:
            if cluster_id is None:
                continue
            clusters[cluster_id].append((transaction_id, similarity))
    
        # Sort Based on Similarity {cluster_id : cluster}
        clusters = {cluster_id : self.unique_txs(self.sort_cluster(cluster)) for cluster_id, cluster in clusters.items()}

        return clusters

    ##############################################################################
    #  Fast Clustering function : Finds in the Close embeddings All Sentences    #
    ##############################################################################
    def fast_clustering(self, ids, embeddings, threshold, min_cluster_size):

        # STEP 1) Compute Cos-sim
        cos_scores = self.cos_sim(embeddings, embeddings)

        # STEP 2) Create clusters where similarity is bigger than threshold
        bigger_than_threshold = cos_scores >= threshold
        indices = bigger_than_threshold.nonzero()
        cos_scores = cos_scores.numpy()

        extracted_clusters = defaultdict(lambda: [])
        for row, col in indices.tolist():
            extracted_clusters[ids[row]].append((ids[col], cos_scores[row, col]))
    
        extracted_clusters = self.sort_clusters(extracted_clusters)

        # STEP 3) Remove overlapping clusters(filtering)
        unique_clusters = {}
        extracted_ids = set()

        # visited each cluster
        for cluster_id, cluster in extracted_clusters.items():
            add_cluster = True

            for transaction in cluster:
                if transaction[0] in extracted_ids:
                    add_cluster = False
                    break

            if add_cluster:
                unique_clusters[cluster_id] = cluster
                for transaction in cluster:
                    extracted_ids.add(transaction[0])
    
        new_clusters = {}
        for cluster_id, cluster in unique_clusters.items():
            community_extended = []
            for idx in cluster:
                community_extended.append(idx)
            new_clusters[cluster_id] = self.unique_txs(community_extended)

        new_clusters = self.filter_clusters(new_clusters, min_cluster_size)

        return new_clusters

    def create_clusters(self, ids, embeddings, clusters,
                        parallel, min_cluster_size,
                        threshold, chunk_size):
        to_cluster_ids = np.array(ids)
        np.random.shuffle(to_cluster_ids)
        c = list(self.chunk(to_cluster_ids, chunk_size))
    
        # Implement Parallel
        with log_durations(logging.info, ">> Parallel jobs create clusters ..."):
            out = parallel(delayed(self.fast_clustering)(chunk_ids, self.get_embeddings(chunk_ids, embeddings), threshold, min_cluster_size) for chunk_ids in c)
        # Combine output {idx : cluster((idx, similarity))}
    
        new_clusters = {}
        for out_clusters in out:
            for idx, cluster in out_clusters.items():
                new_clusters[idx] = self.unique_txs(cluster + new_clusters.get(idx, []))

        # Add ids from old cluster to new cluster
        for cluster_idx, cluster in new_clusters.items():
            community_extended = []
            for (idx, similarity) in cluster:
                community_extended += [(idx, similarity)] + clusters.get(idx, [])
            new_clusters[cluster_idx] = self.unique_txs(community_extended)

        new_clusters = self.reorder_and_filter_clusters(new_clusters, embeddings, threshold, parallel)  # filter to keep only the relevant
        new_clusters = self.sort_clusters(new_clusters)

        clustered_ids = set()
        for idx, cluster_ids in new_clusters.items():
            filtered = set(cluster_ids) - clustered_ids
            cluster_ids = [cluster_idx for cluster_idx in cluster_ids if cluster_idx in filtered]
            new_clusters[idx] = cluster_ids
            clustered_ids |= set(cluster_ids)

        new_clusters = self.filter_clusters(new_clusters, min_cluster_size)
        new_clusters = self.sort_clusters(new_clusters)
        return new_clusters

    def parallel_cluster(self,
                         clusters = None,
                         threshold = 0.7,
                         min_cluster_size = 3,
                         page_size = 2500,
                         iterations = 15,
                         cores=1):
        if clusters is None:
            clusters = {}
    
        with Parallel(n_jobs=cores) as parallel:
            for iteration in range(iterations):
                print('=== Iteration {:} / {:} ==='.format(iteration + 1, iterations))
                print("\n")

                unclustered_ids = self.get_unclustured_ids(self.ids, clusters)
                cluster_ids = list(clusters.keys())

                # Get the nearest Cluster : {cluster_id : cluster}
                clusters = self.nearest_cluster(unclustered_ids,
                                                self.embeddings,
                                                clusters,
                                                parallel=parallel,
                                                threshold=threshold,
                                                chunk_size=page_size)
                unclustered_ids = self.get_unclustured_ids(self.ids, clusters)

                # Get New Clusters
                new_clusters = self.create_clusters(unclustered_ids, self.embeddings, clusters={}, min_cluster_size=min_cluster_size, 
                                                    chunk_size=page_size, threshold=threshold, parallel=parallel)
                # {cluster_id : cluster}
                new_cluster_ids = list(new_clusters.keys())

                ### Control Max_cluster_size ###
                max_clusters_size = 20000
                while True:
                    new_cluster_ids = list(new_clusters.keys())
                    old_new_cluster_ids = new_cluster_ids
                    new_clusters = self.create_clusters(new_cluster_ids, self.embeddings, new_clusters,
                                                        min_cluster_size=1,
                                                        chunk_size=max_clusters_size,
                                                        threshold=threshold,
                                                        parallel=parallel)
                    new_clusters = self.filter_clusters(new_clusters, 2)
                    new_cluster_ids = list(new_clusters.keys())
                    if len(old_new_cluster_ids) < max_clusters_size:
                        break
            
                new_clusters = self.filter_clusters(new_clusters, min_cluster_size)
                clusters = {**new_clusters, **clusters}
                print(">> Number of Total Clusters : ", len(clusters))
                clusters = self.sort_clusters(clusters)

                unclustered_ids = self.get_unclustured_ids(self.ids, clusters)
                cluster_ids = list(clusters.keys())

                clusters = self.nearest_cluster(unclustered_ids,
                                                self.embeddings,
                                                clusters,
                                                parallel=parallel,
                                                threshold=threshold,
                                                chunk_size=page_size)
                clusters = self.sort_clusters(clusters)
                unclustered_ids = self.get_unclustured_ids(self.ids, clusters)
                clustured_ids = self.get_clustured_ids(clusters)
                print(f">> Percentage clusted Doc Embeddings : {len(clustured_ids) / (len(clustured_ids) + len(unclustered_ids)) * 100:.2f}%")
                print("\n")
        return clusters, unclustered_ids

    ###################
    # Stack functions #
    ###################
    def cluster_stack(self, col_list, clusters, unclusters):
        """
        col_list : List(Dataframe column features)
        clusters : Clusted results
        unclusters : UnClusted results
        """
        L = list()
        for i, cluster in enumerate(list(clusters.values())):
            c_df = pd.DataFrame()
            for j in col_list:
                c_df[j] = self.dataframe[j][self.dataframe.id.isin([tx[0] for tx in cluster])]

            c_df['Topic'] = i
            L.append(c_df)
        clusted_df = pd.concat(L)

        # unclusted_id
        unclusted_df = pd.DataFrame()
        for k in col_list:
            unclusted_df[k] = self.dataframe[k][self.dataframe.id.isin(unclusters)]
        unclusted_df['Topic'] = -1

        # Stack clusted_id + unclusted_id
        clusted_df = pd.concat([clusted_df, unclusted_df])
        return clusted_df

    #################################
    # Cluster-based Topic Modeling  #
    #################################
    def c_tf_idf(self, dataframe, ngram_range, en):
        docs_per_topic = dataframe.groupby(['Topic'], as_index=False).agg({self.tgt_col:' '.join})
        documents = docs_per_topic[self.tgt_col].values
        m = len(dataframe)

        if en:
            stopwords = "english"
        else:
            stopwords = []
            f = open("./data/kor_stopwords.txt", 'r', encoding='utf-8')
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                stopwords.append(line)
            f.close()

        count = CountVectorizer(ngram_range=ngram_range, stop_words=stopwords).fit(documents)
        t = count.transform(documents).toarray()
        w = t.sum(axis=1)
        tf = np.divide(t.T, w)
        sum_t = t.sum(axis=0)
        idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
        tf_idf = np.multiply(tf, idf)

        return docs_per_topic, tf_idf, count
    
    def extract_top_n_words_per_topic(self, dataframe, n, ngram_range=(1,1), en=True):
        docs_per_topic, tf_idf, count = self.c_tf_idf(dataframe, ngram_range=ngram_range, en=en)
        words = count.get_feature_names()
        labels = list(docs_per_topic.Topic)
        tf_idf_transposed = tf_idf.T
        indices = tf_idf_transposed.argsort()[:, -n:]
        top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
        return top_n_words
    