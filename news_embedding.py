# %%
from gensim.models.word2vec import Word2Vec

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import numpy as np
import pandas as pd

import torch
from torch.nn import Module, Linear, ReLU, Softmax, CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, default_collate
from torch.utils.tensorboard import SummaryWriter

import time
import os
import tqdm

# %%
class NewsDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()

        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

# %%
class NewsEmbeddingNetwork(Module):
    def __init__(self, input_dimension, hidden_dimension, latent_dimension, output_dimension):
        super().__init__()

        def init_parameters(layer, scale=1):
            torch.nn.init.xavier_normal_(layer.weight, gain=scale)
            torch.nn.init.zeros_(layer.bias)

        self.fc1 = Linear(in_features=input_dimension, out_features=hidden_dimension)
        init_parameters(self.fc1)
        self.fc2 = Linear(in_features=hidden_dimension, out_features=hidden_dimension)
        init_parameters(self.fc2)
        self.fc3 = Linear(in_features=hidden_dimension, out_features=latent_dimension)
        init_parameters(self.fc3)
        self.fc4 = Linear(in_features=latent_dimension, out_features=output_dimension)
        init_parameters(self.fc4)

        self.relu = ReLU()
        self.softmax = Softmax(dim=-1)

    def forward(self, input_vector):
        latent = self.relu(self.fc1(input_vector))
        latent = self.relu(self.fc2(latent))
        latent = self.softmax(self.fc3(latent))
        output = self.softmax(self.fc4(latent))

        return latent, output

# %%
seed=24
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic=True

# %%
# load news data
column_names = ['News_id', 'Category', 'Subcategory', 'Title', 'Abstract', 'URL', 'Title_Entities', 'Abstract_Entities']
news = pd.read_csv('./data/train_news.tsv', sep='\t', names=column_names)
news2 = pd.read_csv('./data/test_news.tsv', sep='\t', names=column_names)
news = pd.concat([news, news2], ignore_index=True)
del news2

print('data loaded')

# %%
# train word to vector model with title
number_of_epoch = 1000
learning_rate = 0.001
batch_size = 128
embedding_dimension = 100
hidden_dimension = 256
latent_dimension = 256
output_dimension = len(news['Category'].unique())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


corpus = [x.split() for x in news['Title']]
model_w2v = Word2Vec(corpus, vector_size=embedding_dimension, min_count=1, workers=15)
print('word2vec model trained')

# %%
# map each news to mean of title w2v
train_x = np.empty((news.shape[0], embedding_dimension), dtype=np.float32)
pointer = 0
for i in corpus:
    vec_list = []
    for j in i:
        vec_list.append(model_w2v.wv[j])
    train_x[pointer] = np.mean(vec_list, axis=0)
    pointer += 1

# news_vector = np.load('./data/news_embedding/train_x.npy')
train_x_tensor = torch.FloatTensor(train_x)

print('news mapped with word2vec model (input)')

# %%
train_y = OneHotEncoder().fit_transform(news['Category'].values)

train_y_tensor = torch.LongTensor(train_y)

print('news category (target) encoded')

# %%
news_dataset = NewsDataset(train_x, train_y)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader = DataLoader(news_dataset, batch_size=batch_size, shuffle=True,
                          collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))

output_dimension = len(news['Category'].unique())

# %%
if not os.path.exists('./runs/'):
    os.makedirs('./runs/')

if not os.path.exists('./model/'):
    os.makedirs('./model/')

writer = SummaryWriter(log_dir=f'./runs/news_embedding_{time.strftime("%Y%m%d-%H%M%S")}')

network = NewsEmbeddingNetwork(input_dimension=embedding_dimension,
                               hidden_dimension=hidden_dimension,
                               latent_dimension=latent_dimension,
                               output_dimension=output_dimension).to(device)

optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
criterion = CrossEntropyLoss()

start_time = time.time()
for epoch in trange(number_of_epoch):
    network.to(device)
    predicts = np.empty((len(train_x),output_dimension))
    labels = np.empty((len(train_x), output_dimension))
    pointer = 0
    for i, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        latent, predict = network.forward(x)
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

    auc = roc_auc_score(labels, predicts, multi_class='ovr')

    writer.add_scalar('AUC', auc, epoch)

print(f'news embedding train time: {time.time() - start_time} seconds')
writer.close()

with torch.no_grad():
    network.to(device)
    predicts = np.empty((len(train_x),output_dimension))
    labels = np.empty((len(train_x), output_dimension))
    pointer = 0
    for i, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        latent, predict = network.forward(x)
        loss = criterion(predict, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mb_size = len(predict)
        predicts[pointer:pointer+mb_size] = predict
        labels[pointer:pointer+mb_size] = y
        pointer += mb_size

        del x, y, predict

predicts = np.argmax(predicts, axis=1)
print(np.unique(predicts))

# %%
with torch.no_grad():
    network.to(device)
    predicts = np.empty((len(train_x),output_dimension))
    labels = np.empty((len(train_x),output_dimension))
    pointer = 0
    for i, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        latent, predict = network.forward(x)
        loss = criterion(predict, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mb_size = len(predict)
        predicts[pointer:pointer+mb_size] = predict
        labels[pointer:pointer+mb_size] = y
        pointer += mb_size

        del x, y, predict

predicts = np.argmax(predicts, axis=1)
print(np.unique(predicts))

# %%
if not os.path.exists('./model/'):
    os.makedirs('./model/')
torch.save(network.state_dict(), 
           f'./model/news_embedding_{time.strftime("%Y%m%d-%H%M%S")}_{auc:.2f}.pth')