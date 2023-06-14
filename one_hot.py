import pandas as pd
import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np
import torch.nn as nn
import torch

column_names = ['News_id', 'Category', 'Subcategory', 'Title', 'Abstract', 'URL', 'Title_Entities', 'Abstract_Entities']
news = pd.read_csv('./data/train_news.tsv', sep='\t', names=column_names)
news2 = pd.read_csv('./data/test_news.tsv', sep='\t', names=column_names)
news = pd.concat([news, news2], ignore_index=True)

del news2

one_hot_category = pd.get_dummies(news['Category']).astype(float).values
one_hot_category_map = {news['News_id'][i]:one_hot_category[i] for i in range(len(news))}

num_of_class = one_hot_category.shape[1]

column_names = ['Impression_id', 'User', 'Time', 'Clicked_News', 'Impressions']
# behaviors = pd.read_csv('./data/train_behaviors.tsv', sep='\t', names=column_names)
behaviors2 = pd.read_csv('./data/test_behaviors.tsv', sep='\t', names=column_names)

num_of_rows = behaviors2.shape[0]

# User history sum
user_clicked = np.empty((num_of_rows, num_of_class))
click_news = behaviors2['Clicked_News'].values
for i in tqdm.tqdm(range(num_of_rows)):
    for click in click_news[i].split():
        user_clicked[i] += one_hot_category_map[click]

## User impressions
impression = np.empty((num_of_rows, 15, num_of_class))
impression_clicked = []
impressions = behaviors2['Impressions'].values
for i in tqdm.tqdm(range(num_of_rows)):
    impression_i = impressions[i].split()
    # impression_truth = []
    for j in range(15):
        # impress_j = impression_i[j].split('-')
        impress_j = impression_i[j]
        # impress = impress_j[0]
        impression[i, j] = one_hot_category_map[impress_j]
        # impression_truth.append(impress_j[1])
    # impression_clicked.append(impression_truth)

soft = nn.Softmax(dim=-1)

prob = np.empty((num_of_rows, 15))
for i in tqdm.tqdm(range(num_of_rows)):
    prob[i] = soft(torch.Tensor(impression[i] @ user_clicked[i])).numpy()

# y = np.array(impression_clicked).reshape(-1)
# x = prob.reshape(-1)

# print(roc_auc_score(y, x))

submit = pd.DataFrame(prob, columns=[f'p{i}' for i in range(1, 16)])
submit.index.name = 'index'
submit.to_csv('./data/submit.csv')