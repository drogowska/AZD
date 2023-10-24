import gm

class DGSC(gm.GM):
    
    def find_masses(self, data):
        self.set_centres(data)
        first = True

        n = 0
        while True:
            err = 0
            pred = []
            data = data.sample(frac=1)                       

            data_x = data.iloc[:, :self.dimensions]
            data_y = data.iloc[:, self.dimensions:].values
            x = data_x.copy().values
            tmp_mass = [c.mass for c in self.classes]

            for i in range(len(x)): 
                predicated = self.core(x[i])
                pred.append(predicated)
                if (data_y[i] != predicated):
                    id = self.classes.index(self.get_cluster(data_y[i]))
                    
                    self.classes[id].more_mass()
                    for j in range(len(self.classes)):
                        if j != id:
                            self.classes[j].less_mass()
                    err += 1
            [c.update_ksi(0.00000001) for c in self.classes]

            diff = [abs(self.classes[c].mass - tmp_mass[c]) for c in range(len(tmp_mass))]
            if all(i <= self.eps for i in diff) and not first:
                n += 1
                if n > 10:
                    return

            first = False 
            for clas in self.classes:
                clas.reset()