import math
import random

# Klasa Cluster reprezentuje klaster utworzony jako rezultat grupowania.

class Cluster:           
         
    def __init__(self, label, dimensions):
        """Konstruktor klasy

            Argumenty:
            label       : etykieta klasy
            dimensions  : liczba wymiarów danych

            Atrybuty klasy:
            mass        : masa klasy, początkow posida wartość 0
            label       : etykieta klasy
            dimensions  : liczba wymiarów danych
            data        : zbiór danych przyprządkowanych do klasy
            size        : liczność klasy
            n           : stała o której wartość masa klasy zostaje zmodyfikowana
            m           : stała określająca wartość o jaką zostanie pomniejszona wartość m 
            c           : lista współrzędnych centroidu klasy    
        """
        self.mass = 0
        self.label = label 
        self.mass = 1
        self.dimensions = dimensions
        self.data = []
        self.size = 0
        self.n = .001 
        self.m = .0001  
        self.min_force = -999   

        c = []
        for i in range(dimensions):
            c.append(random.random())
        self.c = c
    
    def set_mass(self, m):
        self.mass = m
    
    def get_label(self):
        return self.label

    def more_mass(self):
        self.mass = self.mass + self.n
    
    def less_mass(self):
        self.mass = self.mass - self.n

    def update_ksi(self, m):
        self.n -= m
    
    def modify_offline_mass(self, b1, b2):
        self.mass = self.mass + (b1 - b2) * self.n

    def update_size(self):
        self.size = len(self.data)

    def update_centroid(self):
        center = []
        if len(self.data) != 0:
            for i in range(len(self.c)):
                s = 0
                for d in self.data:
                    s += d[i]
                center.append(s/len(self.data))
            self.c = center
        else: 
            self.c = [1] * self.x

    def add(self, d):
        self.data.append(d)
        self.size += 1

    def remove(self):
        self.data.pop()
        self.size -= 1

    def reset(self):
        self.data = []
        self.size = 0
        self.avg_sim = 0

    def sim(self, d):
        r = self.get_center_distance(d)
        if r == 0: 
            return 1
        return 1/r

    def force(self, d):
        return self.mass * self.sim(d) 
    
    def update_avg_sim(self):
        avg_sim = 0
        for i in self.data:
            avg_sim += self.force(i)
        if len(self.data) != 0:
            tmp = avg_sim / len(self.data)
            self.avg_sim = tmp
    
    def get_center_distance(self, d):
        return math.dist(d, self.c)
            
    def find_min_force(self, data): 
        tmp = []
        for i in data:
            tmp.append(self.force(i))
        self.min_force = min(tmp)
            