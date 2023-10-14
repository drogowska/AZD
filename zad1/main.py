from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import random

def preprocessing():    
    df = pd.read_csv("./zad1/data/emotions.csv")

    scaler = StandardScaler()
    df_2 = df.drop(["label"], axis=1)

    X = pd.DataFrame(scaler.fit_transform(df_2))
    label_e = LabelEncoder()
    df['label']=label_e.fit_transform(df['label'])
    # neutral = 0, negative=1, positive=2

    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=48)
    return X_train.values, X_test.values, y_train.values, y_test.values


def preprocessing2():

    test = pd.read_csv("./zad1/mitbih_test.csv")
    train = pd.read_csv("./zad1/mitbih_train.csv")
    n = len(test.values[0]) -1
    return train.values[:,:n], test.values[:,:n], train.values[:,n:], test.values[:,n:]

def preprocessing0():
    # f1 = pd.read_csv("./zad1/data/ptbdb_abnormal.csv")
    # f2 = pd.read_csv("./zad1/data/ptbdb_normal.csv")
    # f = pd.concat([f1, f2])
    # f.to_csv("f.csv")
    f = pd.read_csv("./zad1/data/ptbdb.csv")
    # f = f.sample(frac=1) 
    # f = random.shuffle(f)                      

    scaler = StandardScaler()
    df_2 = f.iloc[:, :-1]
    X = pd.DataFrame(scaler.fit_transform(df_2))
    y = f.iloc[:, -1:]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=48)
    X_train.to_csv("./zad1/data/X_train.csv")
    X_test.to_csv("./zad1/data/X_test.csv")
    y_train.to_csv("./zad1/data/y_train.csv")
    y_test.to_csv("./zad1/data/y_test.csv")
    return X_train.values, X_test.values, y_train.values, y_test.values
