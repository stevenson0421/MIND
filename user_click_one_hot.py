# %%
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

import numpy as np
import pandas as pd

import torch
from torch.nn import Module, Linear, ReLU, Sigmoid, BCELoss
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import time
import os
import tqdm

# %% [markdown]
# # Set Random Seed

# %%
seed=24
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic=True

# %% [markdown]
# # Dataset

# %%
class CustomDataset(Dataset):
    def __init__(self, x, y=None):
        super().__init__()

        self.x = x
        if torch.is_tensor(y):
            self.train = True
            self.y = y
        else:
            self.train = False

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        if self.train:
            return self.x[index], self.y[index]
        else:
            return self.x[index]

# %% [markdown]
# # Network

# %%
class UserClickNetwork(Module):
    def __init__(self, input_dimension, hidden_dimension, output_dimension):
        super().__init__()

        def init_parameters(layer, scale=1):
            torch.nn.init.xavier_normal_(layer.weight, gain=scale)
            torch.nn.init.zeros_(layer.bias)

        self.fc1 = Linear(in_features=input_dimension, out_features=hidden_dimension)
        init_parameters(self.fc1)
        self.fc2 = Linear(in_features=hidden_dimension, out_features=hidden_dimension*2)
        init_parameters(self.fc2)
        self.fc3 = Linear(in_features=hidden_dimension*2, out_features=hidden_dimension)
        init_parameters(self.fc3)
        self.fc4 = Linear(in_features=hidden_dimension, out_features=output_dimension)
        init_parameters(self.fc4)
        
        self.relu = ReLU()
        self.sigmoid = Sigmoid()

    def forward(self, input_vector):
        latent = self.relu(self.fc1(input_vector))
        latent = self.relu(self.fc2(latent))
        latent = self.relu(self.fc3(latent))
        output = self.sigmoid(self.fc4(latent))

        return output

# %% [markdown]
# # Load Data

# %%
# load news data
column_names = ['News_id', 'Category', 'Subcategory', 'Title', 'Abstract', 'URL', 'Title_Entities', 'Abstract_Entities']
news = pd.read_csv('./data/train_news.tsv', sep='\t', names=column_names)
news2 = pd.read_csv('./data/test_news.tsv', sep='\t', names=column_names)
news = pd.concat([news, news2], ignore_index=True)
del news2

column_names = ['Impression_id', 'User', 'Time', 'Clicked_News', 'Impressions']
behaviors = pd.read_csv('./data/train_behaviors.tsv', sep='\t', names=column_names)
behaviors_test = pd.read_csv('./data/test_behaviors.tsv', sep='\t', names=column_names)

# %% [markdown]
# # User Vector

# %%
one_hot_category = pd.get_dummies(news['Category']).astype(float).values
one_hot_category_map = {news['News_id'][i]:one_hot_category[i] for i in range(len(news))}

num_of_class = one_hot_category.shape[1]

num_of_rows = behaviors.shape[0]
num_of_rows_test = behaviors_test.shape[0]

# User history sum
user_clicked = np.empty((num_of_rows, num_of_class))
click_news = behaviors['Clicked_News'].values
for i in tqdm.tqdm(range(num_of_rows)):
    for click in click_news[i].split():
        user_clicked[i] += one_hot_category_map[click]

user_clicked_test = np.empty((num_of_rows_test, num_of_class))
click_news_test = behaviors_test['Clicked_News'].values
for i in tqdm.tqdm(range(num_of_rows_test)):
    for click in click_news_test[i].split():
        user_clicked_test[i] += one_hot_category_map[click]

## User impressions
impression = np.empty((num_of_rows, 15, num_of_class))
impression_clicked = []
impressions = behaviors['Impressions'].values
for i in tqdm.tqdm(range(num_of_rows)):
    impression_i = impressions[i].split()
    impression_label = []
    for j in range(15):
        impress_j = impression_i[j].split('-')
        # impress_j = impression_i[j]
        impress = impress_j[0]
        impression[i, j] = one_hot_category_map[impress]

        impression_label.append(int(impress_j[1]))
    impression_clicked.append(impression_label)

impression_test = np.empty((num_of_rows_test, 15, num_of_class))
impressions = behaviors_test['Impressions'].values
for i in tqdm.tqdm(range(num_of_rows_test)):
    impression_i = impressions[i].split()
    for j in range(15):
        impress_j = impression_i[j]
        impression_test[i, j] = one_hot_category_map[impress_j]

user_vector = np.empty((num_of_rows, 15))
for i in tqdm.tqdm(range(num_of_rows)):
    user_vector[i] = impression[i] @ user_clicked[i]

user_vector_test = np.empty((num_of_rows_test, 15))
for i in tqdm.tqdm(range(num_of_rows_test)):
    user_vector_test[i] = impression_test[i] @ user_clicked_test[i]

# %%
train_x = torch.FloatTensor(user_vector)
print(train_x.shape)
train_y = torch.FloatTensor(np.array(impression_clicked))
print(train_y.shape)
test_x = torch.FloatTensor(user_vector_test)
print(test_x.shape)

# %%
number_of_epoch = 1000
learning_rate = 0.001
batch_size = 128
embedding_dimension = train_x.shape[1]
hidden_dimension = 256
latent_dimension = 256
output_dimension = 15
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_, counts = train_y.reshape(-1).unique(return_counts=True)
class_weights = 1 - counts / counts.sum()

# %%
# news_embeddings = np.load('./data/news_embedding/news_embedding_vector.npy')
# user_embeddings = np.load('./data/user_embedding/user_embedding_vector.npy')

# dot_vector = np.empty((behaviors.shape[0], latent_dimension), dtype=np.float32)
# pointer = 0
# for i in range(len(behaviors['Target_News'])):
#     vec_list = []
#     user_embedding = user_embeddings[i]
#     for j in behaviors['Target_News'][i].split():
#         index = news[news['News_id']==j].index[0]
#         vec_list.append(news_embeddings[index])
#     dot_vector[pointer] = np.dot(np.mean(vec_list, axis=0), user_embedding)
#     pointer += 1

# dot_vector_tensor = torch.FloatTensor(dot_vector)

# print('user mapped with news vector (input)')

# %%
# target_clicked = np.empty((behaviors.shape[0], target_dimension), dtype=np.float32)
# pointer = 0
# for i in behaviors['Target_News_Clicked']:
#     target_clicked[pointer] = list(map(int, i.split()))

# target_clicked_tensor = torch.LongTensor(target_clicked)

# print('user clicked target (target) encoded')

# %% [markdown]
# # Train

# %%
train_dataset = CustomDataset(train_x, train_y)

whole_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

generator = torch.Generator()
data_list = random_split(train_dataset, [0.7, 0.3], generator)

train_loader = DataLoader(data_list[0], batch_size=batch_size, shuffle=True)
val_loader = DataLoader(data_list[1], batch_size=batch_size)

test_dataset = CustomDataset(test_x)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# %%
if not os.path.exists('./runs/'):
    os.makedirs('./runs/')

if not os.path.exists('./model/'):
    os.makedirs('./model/')

writer = SummaryWriter(log_dir=f'./runs/user_click_{time.strftime("%Y%m%d-%H%M%S")}')

network = UserClickNetwork(input_dimension=embedding_dimension,
                            hidden_dimension=hidden_dimension,
                            output_dimension=output_dimension).to(device)

optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
criterion = BCELoss()

start_time = time.time()
for epoch in tqdm.tqdm(range(number_of_epoch)):
    network.to(device)
    predicts = np.empty((len(train_x)*15,))
    labels = np.empty((len(train_x)*15,))
    pointer = 0
    for i, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        predict = network.forward(x)
        predict = torch.flatten(predict)
        y = torch.flatten(y)
        loss = criterion(predict, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            predict = predict.cpu().numpy()
            y = y.cpu().numpy()
            mb_size = len(predict)
            predicts[pointer:pointer+mb_size] = predict
            labels[pointer:pointer+mb_size] = y
            pointer += mb_size

        del x, y, predict

    writer.add_scalar('Loss', loss, epoch)

    auc = roc_auc_score(labels, predicts)
    predicts = np.where(predicts >= 0.5, 1, 0)
    accuracy = accuracy_score(labels, predicts)
    f1 = f1_score(labels, predicts)
    writer.add_scalar('Train AUC', auc, epoch)
    writer.add_scalar('Train Accuracy', accuracy, epoch)
    writer.add_scalar('Train F1', f1, epoch)

    network.to('cpu')
    with torch.no_grad():
        predicts = np.empty((len(train_x)*15,))
        labels = np.empty((len(train_x)*15,))
        pointer = 0
        for i, (x, y) in enumerate(val_loader):
            predict = network.forward(x)
            predict = torch.flatten(predict).numpy()
            y = torch.flatten(y).numpy()
            mb_size = len(predict)
            predicts[pointer:pointer+mb_size] = predict
            labels[pointer:pointer+mb_size] = y
            pointer += mb_size

            del x, y, predict
    auc = roc_auc_score(labels, predicts)
    predicts = np.where(predicts >= 0.5, 1, 0)
    accuracy = accuracy_score(labels, predicts)
    f1 = f1_score(labels, predicts)
    writer.add_scalar('Val AUC', auc, epoch)
    writer.add_scalar('Val Accuracy', accuracy, epoch)
    writer.add_scalar('Val F1', f1, epoch)

    if epoch % 10 == 0:
        torch.save(network.state_dict(),
                    f'./model/{epoch}_{auc:.2f}.pth')

print(f'train time: {time.time() - start_time} seconds')
writer.close()

# %%
torch.save(network.state_dict(), f'./model/100_{auc:.2f}.pth')

# %%
writer = SummaryWriter(log_dir=f'./runs/user_click_{time.strftime("%Y%m%d-%H%M%S")}')

network = UserClickNetwork(input_dimension=embedding_dimension,
                            hidden_dimension=hidden_dimension,
                            output_dimension=output_dimension).to(device)

optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
criterion = BCELoss()

start_time = time.time()
for epoch in tqdm.tqdm(range(number_of_epoch)):
    network.to(device)
    predicts = np.empty((len(train_x)*15,))
    labels = np.empty((len(train_x)*15,))
    pointer = 0
    for i, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        predict = network.forward(x)
        predict = torch.flatten(predict)
        y = torch.flatten(y)
        loss = criterion(predict, y)

        writer.add_scalar('Loss', loss, epoch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            predict = predict.cpu().numpy()
            y = y.cpu().numpy()
            mb_size = len(predict)
            predicts[pointer:pointer+mb_size] = predict
            labels[pointer:pointer+mb_size] = y
            pointer += mb_size

        del x, y, predict

    writer.add_scalar('Loss', loss, epoch)
    
    auc = roc_auc_score(labels, predicts)

    predicts = np.where(predicts >= 0.5, 1, 0)
    accuracy = accuracy_score(labels, predicts)
    f1 = f1_score(labels, predicts)
    writer.add_scalar('Train AUC', auc, epoch)
    writer.add_scalar('Train Accuracy', accuracy, epoch)
    writer.add_scalar('Train F1', f1, epoch)

    del predicts

    if epoch % 10 == 0:
        torch.save(network.state_dict(),
                    f'./model/whole_{epoch}_{auc:.2f}.pth')

print(f'train time: {time.time() - start_time} seconds')
writer.close()

torch.save(network.state_dict(), f'./model/whole_100_{auc:.2f}.pth')

# %%
network.load_state_dict(torch.load('./model/100_0.87.pth'))
network.to('cpu')

with torch.no_grad():
    predicts = np.empty((len(test_x),15))
    pointer = 0
    for i, x in enumerate(test_loader):
        predict = network.forward(x)
        predict = predict.numpy()
        mb_size = len(predict)
        predicts[pointer:pointer+mb_size] = predict
        pointer += mb_size

        del x, predict

# %%
submit = pd.DataFrame(predicts, columns=[f'p{i}' for i in range(1, 16)])
submit.index.name = 'index'
submit.to_csv('./data/submit.csv')


