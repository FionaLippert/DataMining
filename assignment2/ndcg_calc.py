import numpy as np
import pandas as pd

def ndcg_calc(train_df, pred_scores):
    """
    >>ndcg_calc(train_df, pred_scores)
       train_df: pd.DataFrame with Expedia Columns: 'srch_id', 'booking_bool', 'click_bool'
       pred_scores: np.Array like vector of scores with length = num. rows in train_df
       
    Calculate Normalized Discounted Cumulative Gain for a dataset is ranked with pred_scores (higher score = higher rank).
    If 'booking_bool' == 1 then that result gets 5 points.  If 'click_bool' == 1 then that result gets 1 point (except:
    'booking_bool' = 1 implies 'click_bool' = 1, so only award 5 points total).  
    
    NDCG = DCG / IDCG
    DCG = Sum( (2 ** points - 1) / log2(rank_in_results + 1) )
    IDCG = Maximum possible DCG given the set of bookings/clicks in the training sample.
    
    """
    eval_df = train_df[['srch_id', 'booking_bool']]
    eval_df['click_bool'] = train_df['click_bool'] - train_df['booking_bool']
    eval_df['score'] = pred_scores

    logger = lambda x: np.log2(x + 1)
    # rank by the score column and calculate the denominator for each entry
    eval_df['log_rank'] = eval_df.groupby(by = 'srch_id')['score'].rank(ascending = False).map(logger)

    book_dcg = (eval_df['booking_bool'] * 31.0 / eval_df['log_rank']).sum() #where 2 ** 5 - 1.0 = 31.0
    book_idcg = (31.0 * eval_df['booking_bool']).sum()
    
    click_dcg = (eval_df['click_bool'] * (eval_df['booking_bool'] == 0) / eval_df['log_rank']).sum()
    
    # Max number of rows per srch_id in training set is 38.
    # and put in dict {num of rows: IDCG value}.
    disc = [1.0 / np.log2(i + 1) if i != 0 else 0 for i in range(38)]
    disc_dict = {i: np.array(disc).cumsum()[i] for i in range(38)}
    
    # Map the number of clicks to its IDCG and subtract off any clicks due to bookings
    # since these were accounted for in book_idcg.
    click_idcg = (eval_df.groupby(by = 'srch_id')['click_bool'].sum().map(disc_dict)).sum()

    return (book_dcg + click_dcg) / (book_idcg + click_idcg)
