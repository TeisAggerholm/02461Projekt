import numpy as np
import math

class generator: 
    def __init__(self, max_step): 
        self.max_step = max_step

    def route_generator(self): 
        
        #Antal biler: 
        

        #Hvilken fordeling anvendes til bilernes tider? 
        tider = np.random.weibull(2, self.n_cars)
        tider.sort()

        #Normalisering ift. max_steps 
        new_times = []
        for tid in tider: 
            new_times.append(tid / tider[-1] * self.max_step)

        print(new_times)



generator(200,1000).route_generator()