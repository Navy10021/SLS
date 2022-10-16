import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

###################################
## User-Based Filtering Function ##
###################################

def interaction_table(user_dataframe, latent_factors = 10):
    """
    Make Data x User Interaction Table function
    """
    # 1. Make pivot table(user_id x case_id)
    user_case_matrix_df = user_dataframe.pivot_table(index = 'user_id', columns = 'case_id', values = 'user_hits', aggfunc = 'mean').fillna(0)
    print("\n >> Original Matrix shape : ", user_case_matrix_df.shape)
    user_case_pivot_matrix = user_case_matrix_df.to_numpy()

    # 2. Converting a NumPy sparse matrix to a SciPy compressed sparse row matrix
    users_ids = list(user_case_matrix_df.index)
    print("\n >> Total User Ids : ", len(users_ids))
    users_case_pivot_sparse_matrix = csr_matrix(user_case_pivot_matrix)

    # 3. Performs matrix factorization of the original user item matrix
    U, sigma, Vt = svds(
        users_case_pivot_sparse_matrix,
        k = latent_factors
        )
    sigma = np.diag(sigma)
    #print("\n >> U (Left singular matrix, Relationship between User and latent factors) :", U.shape)
    #print("\n >> V_transpose (Right singular matrix, Similarity between Items and latent factors) :", Vt.shape)
    #print("\n >> S (Diagonal matirx, Strength of each latent factors):", sigma.shape)

    # 4. Get Interaction matrix = U * S * Vt
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
    # 5. Min-Max Scaling [0, 1]
    all_user_predicted_ratings_norm = (all_user_predicted_ratings-all_user_predicted_ratings.min())/(all_user_predicted_ratings.max()-all_user_predicted_ratings.min())
    #print("\n >> Interaction Matrix (User x Law Cases) : {}\n".format(all_user_predicted_ratings_norm.shape))

    # 6. Converting the reconstructed matrix back to a Pandas dataframe
    cf_preds_df = pd.DataFrame(all_user_predicted_ratings_norm, columns = user_case_matrix_df.columns, index = users_ids).transpose()
    print("\n >> Interaction Matrix (Law Cases x User) : {}".format(cf_preds_df.shape))

    return cf_preds_df


def user_based_sementic_search(cf_dataframe, view_dataframe, user_id, my_model, my_query, index_kind,sementic_weight = 0.3, user_weight = 0.7):
    """
    Combined Orginal semantic search and Utility matrix with weight average of the normalized.
    """
    user_id = 'lawyer_' + str(user_id)
    # 1. Sorting the User's predictions & score
    sorted_user_preds = cf_dataframe[user_id].sort_values(ascending=False).reset_index().rename(columns={user_id : 'score'})
    # 2. prepare User table & score
    mean_score = sorted_user_preds['score'].mean()
    user_table = sorted_user_preds.merge(view_dataframe, how='left', left_on='case_id', right_on='case_id')[['case_id', 'case_name', 'judgment_issue', 'judgment_summary','score']]
    # 3. Sementic search table(S-BERT), top 100
    print("******* Original Semantic Law Cases Search *******")
    sementic_table, _ = my_model.search(user_query = my_query, top_k = 50, index = index_kind)

    # 4. Get scaled score = (weight * smentic_rating) + (weight * SVD score)
    for i in sementic_table:
        target_id = i['case_id']
        if target_id in user_table['case_id']:
            # If empty list pass
            if not (user_table[user_table.case_id == target_id].score.values):
                pass
            else:
                scaled_rating = (sementic_weight * i['case_hits_norm']) + (user_weight * float(user_table[user_table.case_id == target_id].score.values))
        else:
            scaled_rating = (sementic_weight * i['case_hits_norm']) + (user_weight * mean_score)
        i['scaled_rating'] = scaled_rating
    
    return sementic_table



#####################################
## Online-Based Filtering Function ##
#####################################
import pytrends
from pytrends.request import TrendReq
import datetime
from dateutil.relativedelta import relativedelta
import scipy.stats as stats
import matplotlib.pyplot as plt

def keyword_encoding(keyword_list, my_model):
    """
    Encoding the types of keywords(Law Cases)
    """
    items = [my_model.model.encode([item]) for item in keyword_list]
    items = [np.asarray(item.astype('float32')) for item in items]
    encoded_items = np.asarray(items)
    encoded_items = encoded_items.reshape((len(keyword_list), 768))

    return encoded_items, items


def google_trends(search, date_range):
    """
    Acquire Google search volume data for keywords.
    """
    pytrends = TrendReq(hl = "en-US", tz = 360)
    kw_list = [search]
    try :
        pytrends.build_payload(kw_list, timeframe = date_range[0], geo = 'US')
        # Retrieve the interest Over time
        trends = pytrends.interest_over_time()
        #related_queries = pytrends.related_queries()
        
    except Exception as e:
        print("\nGoogle Search Trend retrieval failed.")
        print(e)
        return
    return trends


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def online_search_vector(search_query, items, days = 16):
    
    # 1. last 2 weeks Google Trend
    now = datetime.datetime.now()
    two_week = now - relativedelta(days = days)
    date_range = ["%s %s" % (two_week.strftime('%Y-%m-%d'), now.strftime('%Y-%m-%d'))]
    print("Search date range : {}".format(date_range))
    # 2. Online Search Dataframe
    google_watch_hist = pd.DataFrame()
    #search_query = [m + " case" for m in search_query]

    for s in search_query:
        trend = google_trends(s, date_range)
        google_watch_hist[s] = trend[s].tolist()

    google_watch_hist['date'] = trend.index.tolist()
    google_watch_hist.set_index(pd.PeriodIndex(google_watch_hist.date, freq = "D"), inplace = True)
    google_watch_hist.drop('date', axis=1, inplace=True)

    # 3. Plot
    google_watch_hist.plot(kind='barh', stacked=True, figsize=(28, 12))
    plt.xlabel("Search Volume",  fontsize=15)
    plt.ylabel("Date",  fontsize=15)
    plt.legend()

    # 4. Weight Formula
    search_score = dict()
    for g in search_query:
        search_score[g] = stats.zscore(google_watch_hist[g])[-1]
    x = np.asarray([val for key, val in search_score.items()])

    # 5. Get the weight value through the Softmax function.
    keyword_weights = softmax(x)
    # Weight Formula : w_i : genre_weight, x_i : encoded genre text
    search_encoded_vector = np.asarray(sum([item * weight for item, weight in zip(items, keyword_weights)]))
    search_encoded_vector = search_encoded_vector.reshape((1, 768))
    
    return search_encoded_vector
