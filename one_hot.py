import pandas as pd
import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np
import torch.nn as nn
import torch

column_names = ['News_id', 'Category', 'Subcategory', 'Title', 'Abstract', 'URL', 'Title_Entities', 'Abstract_Entities']
news = pd.read_csv('./train//train_news.tsv', sep='\t', names=column_names)


one_hot_news = pd.get_dummies(pd.Series(news['Subcategory'])).astype(float).values
one_hot = {news['News_id'][i]:one_hot_news[i] for i in range(len(news))}


column_names = ['Impression_id', 'User', 'Time', 'Clicked_News', 'Impressions']
behaviors = pd.read_csv('./train/train_behaviors.tsv', sep='\t', names=column_names, parse_dates=['Time'])

# User history sum
user_click = np.empty((behaviors.shape[0], one_hot_news.shape[1]))
for i in tqdm.tqdm(range(len(user_click))):
    for click in behaviors['Clicked_News'].values[i].split():
        user_click[i] += one_hot[click]

## Users impressions
impression = np.empty((behaviors.shape[0], 15, one_hot_news.shape[1]))
for i in tqdm.tqdm(range(len(behaviors['Impressions']))):
    for j in range(15):
        impress = behaviors['Impressions'].values[i].split()[j].split('-')[0]
        impression[i, j] = one_hot[impress]

soft = nn.Softmax(dim=-1)

prob = np.empty((len(behaviors), 15))
for i in tqdm.tqdm(range(len(behaviors))):
    prob[i] = soft(torch.Tensor(impression[i] @ user_click[i])).numpy()

## Ground Truth
users_impressions_news_truth = []
for j in tqdm.tqdm(range(len(behaviors))):
    user_j = behaviors['Impressions'].values[j]
        
    user_j_truth_table = [float(user_j.split()[i].split('-')[1]) for i in range(len(user_j.split()))]
    
    users_impressions_news_truth.append(user_j_truth_table)

y = np.array(users_impressions_news_truth).reshape(-1)
x = prob.reshape(-1)

print(roc_auc_score(y, x))
