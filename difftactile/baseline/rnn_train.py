import numpy as np
import torch
import os
import pickle
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split


## load data here 

with open('rnn_train_data.pkl', 'rb') as f:
    data_list, label_list = pickle.load(f)


max_length = max(array.shape[0] for array in data_list)
padded_data = [np.pad(array, ((max_length - len(array), 0), (0, 0), (0, 0)), mode='constant') for array in data_list]
padded_data = [torch.tensor(array, dtype=torch.float32) for array in padded_data]
labels = torch.tensor(label_list, dtype=torch.int64)
X_train, X_val, y_train, y_val = train_test_split(padded_data, labels, test_size=0.3, random_state=43)
X_train = torch.stack(X_train, dim = 0)
X_val = torch.stack(X_val, dim = 0)
B, L, N, _ = X_train.shape
X_train = X_train.reshape(B, L, -1)
B, L, N, _ = X_val.shape
X_val = X_val.reshape(B, L, -1)


from torch.utils.data import DataLoader, TensorDataset
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)



batch_size = 16  
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Model define and training


import torch.nn as nn
import torch.optim as optim

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

input_size = 2* 136 
hidden_size = 128
num_layers = 2
num_classes = 2
model = RNNModel(input_size, hidden_size, num_layers, num_classes).cuda()



# Training

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def calculate_accuracy(y_true, y_pred):
    predicted = torch.argmax(y_pred, dim=1)
    correct = (predicted == y_true).float().sum()
    return correct / y_true.shape[0]

num_epochs = 10  
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_correct = 0

    for X_batch, y_batch in train_loader:
        X_batch_cuda = X_batch.cuda()
        y_batch_cuda = y_batch.cuda()
        optimizer.zero_grad()
        outputs = model(X_batch_cuda)
        
        loss = criterion(outputs, y_batch_cuda)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += (outputs.argmax(dim=1) == y_batch_cuda).sum().item()

    train_loss = total_loss / len(train_loader)
    train_acc = total_correct / len(train_dataset)

   
    model.eval()
    total_correct = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch_cuda = X_batch.cuda()
            y_batch_cuda = y_batch.cuda()

            val_outputs = model(X_batch_cuda)
            total_correct += (val_outputs.argmax(dim=1) == y_batch_cuda).sum().item()

    val_acc = total_correct / len(val_dataset)
    print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Train Acc: {train_acc}, Val Acc: {val_acc}')

torch.save(model.state_dict(), 'rnn_model.pth')





##  Deploy here

model_state = torch.load('rnn_model.pth')
model.load_state_dict(model_state)

def predict(model, new_data):
    model.eval()
    with torch.no_grad():
        new_data_tensor = torch.tensor(new_data, dtype=torch.float32).unsqueeze(0)  
        output = model(new_data_tensor)
        predicted = torch.argmax(output, dim=1)
        return predicted.item()
    





