from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from NN import NN_classfier
from kNNRFSVM import kNNRFSVM
from stacking import knntreesvm
import torch
import os.path
import numpy as np

def preprocessing():
    if not os.path.isfile("./zad1/data/X_train.csv"):
        df = pd.read_csv("./zad1/data/emotions.csv")
        scaler = StandardScaler()
        df_2 = df.drop(["label"], axis=1)
        X = pd.DataFrame(scaler.fit_transform(df_2))
        label_e = LabelEncoder()
        df['label']=label_e.fit_transform(df['label'])    # neutral = 0, negative=1, positive=2
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=48)
        X_train.to_csv("./zad1/data/X_train.csv")
        X_test.to_csv("./zad1/data/X_test.csv")
        y_train.to_csv("./zad1/data/y_train.csv")
        y_test.to_csv("./zad1/data/y_test.csv")
        return X_train.values, X_test.values, y_train.values, y_test.values

    else :
        y_test = pd.read_csv("./zad1/data/y_test.csv").values[:,1:]
        X_test = pd.read_csv("./zad1/data/X_test.csv").values[:,1:]
        y_train = pd.read_csv("./zad1/data/y_train.csv").values[:,1:]
        X_train = pd.read_csv("./zad1/data/X_train.csv").values[:,1:]
        y_train = np.concatenate(y_train, axis=0)
        y_test = np.concatenate(y_test, axis=0)
        return X_train, X_test, y_train, y_test



def run():
    torch.multiprocessing.freeze_support()

if __name__ == '__main__':
    run()
    X_train, X_test, y_train, y_test = preprocessing()
    print('Neural Network Classifier: \n')
    NN_classfier(X_train, X_test, y_train, y_test)
    print('Ensebly classifier no. 1: \n')
    kNNRFSVM(X_train, X_test, y_train, y_test)
    print('Ensebly classifier no. 2: \n')
    knntreesvm(X_train, X_test, y_train, y_test)
    
