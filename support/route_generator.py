import numpy as np
import math

class generator: 
    def __init__(self, max_step): 
        self.max_step = max_step

    def route_generator(self): 
        
        #Antal biler: 
        n_cars = np.random.normal(self.max_step/60*66.667,30)

        #Hvilken fordeling anvendes til bilernes tider? 
        tider = np.random.weibull(2, int(n_cars))
        tider.sort()

        #Normalisering ift. max_steps 
        new_times = []
        for tid in tider: 
            new_times.append(tid / tider[-1] * self.max_step)

    


generator(200).route_generator()