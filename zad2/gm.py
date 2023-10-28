from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix
from Cluster import Cluster
from abc import ABC, abstractmethod
import pandas as pd
import seaborn as sns
# Klasa abstrakcyjna będąca reprezentacją grawitacyjnego modelu klasyfikacji, dziedzicząca po klasie ABC

class GM(ABC):
    
    """     Atrybuty klasy:
            classes     (array) : lista obiektów klasy Cluster
            dimensions  (int)   : liczba wymiarów danych
            k           (int)   : liczba klas
    """

    def __init__(self, dimensions, k=2):
        """Konstruktor klasy - metoda towrzy instancję klasy wraz z listą klas o etykietach numerycznych 

            Argumenty:
            dimensions (int)  : liczba wymiarów danych
            k          (int)  : liczba klas
        """

        self.classes = []
        self.outliers = []
        self.min_froce = 0.0
        self.k = k
        self.dimensions = dimensions
        self.out = Cluster(-1,self.dimensions)
        for i in range(1,k+1):
            self.classes.append(Cluster(i, dimensions)) 

    def core(self, d, test=False):
        """Metoda określająca etykietę przynależności obiektu d zgodnie z modelem grawitacyjnym
        
            Argumenty:
            d        (array) : wektor danych poddany klasyfikacji
           
            Wynik:
            int              : etykieta klasy 
        """    

        tab_f = []
        tab_c = []
        for clas in self.classes:
            tab_f.append(clas.force(d))
            tab_c.append(clas)   
        m = max(tab_f)
        id = tab_f.index(m)   
        self.classes[id].add(d)

        if abs(m - self.classes[id].min_force) <  0.10  and test :
            self.out.add(m)
            return self.out.label
        return tab_c[id].label

    def set_centres(self, x_data, y_data):
        """Metoda obliczająca połorzenie centroidów klas

            Argumenty:
            data        (DataFrame) : zbiór danych uczący 
        """ 
        x_data = pd.DataFrame(x_data)
        y_data = pd.DataFrame(y_data)
        data = pd.merge(x_data, y_data, left_index=True, right_index=True)
        out = self.dimensions 
        for c in range(self.k):
            clas = data[data.iloc[:, self.dimensions] == c+1]
            clas = clas.values[:, :out]
            center = []

            for i in range(len(clas[0])):
                s = 0
                for d in clas:
                    s += d[i]
                center.append(s/len(clas))
            self.classes[c].c = center
            self.classes[c].find_min_force(clas)

    @abstractmethod
    def find_masses(self, x_data, y_data):
        """Metoda abstrakcyjna obliczająca masy klas

            Argumenty:
            data        (DataFrame) : zbiór danych uczący 
        """         
        pass




    def get_cluster(self, label):
        """Metoda zwracająca klasę z listy o podanej etykiecie

            Argumenty:
            label        (int) : etykieta klasy

            Wynik:
            Cluster            : obiekt klasy  
        """ 
        for c in self.classes:
            if c.label == label:
                return c

    def find_outliers(self):
        """Metoda ustalająca klasę wyjątków 
        """ 

        self.classes.sort(key=lambda c : len(c.data), reverse=False)
        # self.classes[0].label = -1 
        

    def classify(self, x_train, x_test, y_train, y_test):
        """Metoda klasyfikująca dane

            Argumenty:
            train        (DataFrame) : zbiór danych uczący 
            test         (DataFrame) : zbiór danych testowych

            Wynik:
            Array                    : lista 
         """ 

        res = []
        self.find_masses(x_train, y_train)
        # self.find_outliers()
    
        for i in x_test:
            j = self.core(i, False)
            res.append(j)
        confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, res))
        sns.heatmap(confusion_matrix_df, annot=True)
        plt.show()
        print(classification_report(y_test, res))

        return res
    

    def show_outliers(self, x_test):
        res = []
        for i in x_test:
            j = self.core(i, True)
            res.append(j)
        self.classes.append(self.out) 
        pred = []
        for i in res: 
            if (i != -1):
                pred.append(1)
            else: 
                pred.append(i)


        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(x_test)
        tsne_df = pd.DataFrame(tsne_results, columns=['Dimension 1', 'Dimension 2'])
        tsne_df['label'] = pred
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='Dimension 1', y='Dimension 2', hue='label', data=tsne_df, palette='viridis')
        plt.show()


