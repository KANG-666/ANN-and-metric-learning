import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

df = pd.read_excel('ANN.xlsx')
X = df.iloc[:, :6].values  # input
y = df.iloc[:, -1:].values  # gt
print(X)
print(y)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


dataset = CustomDataset(X, y)


class SiameseNetwork(nn.Module):
    def __init__(self, input_dim):
        super(SiameseNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )



    def forward(self, x1, x2):
        out1 = self.fc(x1)
        out2 = self.fc(x2)
        return out1, out2


# 度量学习
def contrastive_loss(output1, output2, label, margin=1.0):
    euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
    loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                      label * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
    return loss


def train_siamese_network(dataset, epochs=100, batch_size=4):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = SiameseNetwork(input_dim=6)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        total_loss = 0
        for x1, y1 in dataloader:
            for x2, y2 in dataloader:
                label = (torch.norm(y1 - y2, dim=1) > 0.5).float()
                optimizer.zero_grad()
                out1, out2 = model(x1, x2)
                loss = contrastive_loss(out1, out2, label)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {total_loss:.4f}")

    return model


siamese_model = train_siamese_network(dataset)


def get_embeddings(model, X):
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        embeddings = model.fc(X_tensor).numpy()
    return embeddings


X_embedded = get_embeddings(siamese_model, X)

knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_embedded, y)


# 预测
def predict(X_new):
    X_new = scaler_X.transform(X_new)  # 归一化
    X_new_embedded = get_embeddings(siamese_model, X_new)  # 获得嵌入
    y_pred = knn.predict(X_new_embedded)
    return scaler_y.inverse_transform(y_pred)


df = pd.read_excel('ANNtest.xlsx')
X = df.iloc[:, :6].values
print("Predicted output:", predict(X))
print(predict(X).shape)

