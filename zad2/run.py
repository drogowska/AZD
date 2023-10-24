import pandas as pan
import dgsc
from sklearn.preprocessing import StandardScaler
import os
import DBprep as db

"""
    path  (string) : ścieżka dostępu do katalogu zawierającego pliki baz danych
    files (array)  : lista nawzw baz danych

"""



path  = "E:\GM\src\databases\pure"
files = ["cardio", "vowels", "musk", "synthetic", "chemical", "MNIST0"]
db_path = os.path.join(path, "db")

os.mkdir(db_path)


file = files[3]
for file in files:
    print("Baza danych : " + file)

    X = pan.read_csv(path + file + "X.csv")
    X = pan.DataFrame(StandardScaler().fit_transform(X.values))
    Y = pan.read_csv(path + file + "Y.csv") 
    train, test = help.split_to_train(X, Y)
    dimensions = len(test.values[0]) - 1
    x = test.values[:,:dimensions]
    y = test.values[:,dimensions:]



    print('DGSC:')
    gm = dgsc.DGSC(dimensions)
    res_dgsc = gm.classify(train, test)
    help.show_results(res_dgsc, y, x, 'DGSC')

