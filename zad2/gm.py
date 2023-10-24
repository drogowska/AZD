from Cluster import Cluster
from abc import ABC, abstractmethod

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
        for i in range(k):
            self.classes.append(Cluster(i, dimensions)) 

    def core(self, d):
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
        # if m < min_force
        # if sila jest mniejsza niz podana jakas dozwolona to wyjatek
        # moze jak jest mniejsza od najmnieszej sily w klasie policzone podczas wyznaczania mas
        if m < self.classes[id].min_force :
            self.outliers.append(m)
        return tab_c[id].label

    def set_centres(self, data):
        """Metoda obliczająca połorzenie centroidów klas

            Argumenty:
            data        (DataFrame) : zbiór danych uczący 
        """ 

        out = self.dimensions 
        for c in range(self.k):
            clas = data[data.iloc[:, self.dimensions] == c]
            clas = clas.values[:, :out]
            center = []

            for i in range(len(clas[0])):
                s = 0
                for d in clas:
                    s += d[i]
                center.append(s/len(clas))
            self.classes[c].c = center

    @abstractmethod
    def find_masses(self, data):
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
        self.classes[0].label = -1 
        

    def classify(self, train, test):
        """Metoda klasyfikująca dane

            Argumenty:
            train        (DataFrame) : zbiór danych uczący 
            test         (DataFrame) : zbiór danych testowych

            Wynik:
            Array                    : lista 
         """ 

        res = []
        i = len(test.values[0])
        test = test.iloc[:, :i-1]
        self.find_masses(train)
        self.find_outliers()
        for i in self.classes:
            i.find_min_force()
    
        for i in test.values:
            j = self.core(i)
            res.append(j)
        return res
