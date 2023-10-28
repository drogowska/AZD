import gm

class DGSC(gm.GM):
    
    def find_masses(self, x_data, y_data):
        self.set_centres(x_data, y_data)
        first = True

        n = 0
        while True:
            err = 0
            pred = []
            
            x = x_data.copy()
            tmp_mass = [c.mass for c in self.classes]

            for i in range(len(x)): 
                predicated = self.core(x[i])
                pred.append(predicated)
                if (y_data[i] != predicated):
                    id = self.classes.index(self.get_cluster(y_data[i]))
                    
                    self.classes[id].more_mass()
                    for j in range(len(self.classes)):
                        if j != id:
                            self.classes[j].less_mass()
                    err += 1
            [c.update_ksi(0.00000001) for c in self.classes]

            diff = [abs(self.classes[c].mass - tmp_mass[c]) for c in range(len(tmp_mass))]
            if all(i <= 0.1 for i in diff) and not first:
                n += 1
                if n > 10:
                    return

            first = False 
            for clas in self.classes:
                clas.reset()