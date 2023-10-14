import os.path
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from main import preprocessing
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from itertools import chain

PATH = './AZDModel1.pth'
BATCH_SIZE = 20
EPOCHS = 10
LEARNING_RATE = 0.001

class Data(Dataset):
  def __init__(self, x_train, y_train):
    self.X = torch.from_numpy(x_train.astype(np.float32))
    self.y = torch.from_numpy(y_train.astype(np.float32))
    self.len = self.X.shape[0]
  
  def __getitem__(self, index):
    return self.X[index], self.y[index]
  def __len__(self):
    return self.len

class Network(nn.Module):
    _hidden_layers = 5000

    def __init__(self, _output_dim, _input_dim):
        self._input_dim = _input_dim
        self._output_dim = _output_dim
        super(Network, self).__init__()
        self.linear1 = nn.Linear(self._input_dim, self._hidden_layers)
        self.linear2 = nn.Linear(self._hidden_layers, self._output_dim)

    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = self.linear2(x)
        return x

def run():
    torch.multiprocessing.freeze_support()

if __name__ == '__main__':
    run()


    if not os.path.isfile(PATH):
        X_train, X_test, y_train, y_test = preprocessing()

        traindata = Data(X_train, y_train)
        trainloader = DataLoader(traindata, batch_size=BATCH_SIZE, 
                            shuffle=True, num_workers=2)

        clf = Network(len(np.unique(y_train)), len(X_train[0]))
        class_count = [i for i in np.unique(y_train)]
        class_weights = 1./torch.tensor(class_count, dtype=torch.float) 
        criterion = nn.CrossEntropyLoss()
        # optimizer = torch.optim.SGD(clf.parameters(), lr=0.1)
        optimizer = torch.optim.Adam(clf.parameters(), lr=LEARNING_RATE)

        clf.train()
        for epoch in range(EPOCHS):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                optimizer.zero_grad()            
                # print(labels)
                outputs = clf(inputs)     # forward propagation
                # print(outputs)
                loss = criterion(outputs, torch.max(labels, 1)[1])
                loss.backward()     # backward propagation
                optimizer.step()
                running_loss += loss.item()
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.5f}')


        torch.save(clf.state_dict(), PATH)
    else: 
        y_test = pd.read_csv("./zad1/data/y_test.csv").values[:,1:]
        X_test = pd.read_csv("./zad1/data/X_test.csv").values[:,1:]

        clf = Network(len(np.unique(y_test)), len(X_test[0]))
        clf.load_state_dict(torch.load(PATH))

    testdata = Data(X_test, y_test)
    testloader = DataLoader(testdata, batch_size=BATCH_SIZE, 
                            shuffle=True, num_workers=2)
    correct, total = 0, 0
    y_pred_list = []
    with torch.no_grad():
        clf.eval()
        for data in testloader:
            inputs, labels = data
            outputs = clf(inputs)
            __, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            y_pred_list.append(predicted.cpu().numpy())
    # y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    y_pred = list(chain.from_iterable(y_pred_list))
    # confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_pred_list)).rename(columns=idx2class, index=idx2class)
    # sns.heatmap(confusion_matrix_df, annot=True)
    print(classification_report(y_test, y_pred))