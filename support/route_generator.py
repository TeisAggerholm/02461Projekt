import numpy as np
import math

class generator: 
    def __init__(self, max_step): 
        self.max_step = max_step

    def route_generator(self, percentage_straight):     
        #Antal biler: 
        n_cars = np.random.normal(self.max_step/60*66.667,30)

        #Hvilken fordeling anvendes til bilernes tider? 
        tider = np.random.weibull(2, int(n_cars))
        tider.sort()

        #Normalisering ift. max_steps 
        new_times = []
        for tid in tider: 
            new_times.append(tid / tider[-1] * self.max_step)



        with open("sumo_files/osm.passenger.trips.xml", "w") as routes:
            print("""<?xml version="1.0" encoding="UTF-8"?>

            <routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
                  
            <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />) 
            
            <route id="N2W" edges="-125514711 125514713"/>
            <route id="N2S" edges="-125514711 -125514709"/>
            <route id="N2E" edges="-125514711 -548975769"/>     
            <route id="W2N" edges="-125514713 125514711"/>
            <route id="W2E" edges="-125514713 -548975769"/>
            <route id="W2S" edges="-125514711 -125514709"/>
            <route id="S2W" edges="125514709 125514713"/>
            <route id="S2N" edges="125514709 125514711"/>
            <route id="S2E" edges="-125514711 -548975769"/>
            <route id="E2S" edges="548975769 -125514709"/>
            <route id="E2W" edges="548975769 125514713"/>
            <route id="E2N" edges="548975769 125514711"/>
            """, file=routes)

            for car, departure in enumerate(new_times): 
                if np.random.rand(1)<percentage_straight: 
                    random_int = np.random.randint(1,5)
                    if random_int == 1: 
                        print(f'<vehicle id="N2S{car}" type="standard_car" route="N2S" depart="{departure}" departLane="random" departSpeed="10" />', file=routes)
                    if random_int == 2: 
                        print(f'<vehicle id="W2E{car}" type="standard_car" route="W2E" depart="{departure}" departLane="random" departSpeed="10" />', file=routes)
                    if random_int == 3: 
                        print(f'<vehicle id="S2N{car}" type="standard_car" route="S2N" depart="{departure}" departLane="random" departSpeed="10" />', file=routes)  
                    if random_int == 4: 
                        print(f'<vehicle id="E2W{car}" type="standard_car" route="E2W" depart="{departure}" departLane="random" departSpeed="10" />', file=routes)

                else: 
                    random_int = np.random.randint(1,9)
                    if random_int == 1: 
                        print(f'<vehicle id="N2W{car}" type="standard_car" route="N2W" depart="{departure}" departLane="random" departSpeed="10" />', file=routes)
                    if random_int == 2: 
                        print(f'<vehicle id="N2E{car}" type="standard_car" route="N2E" depart="{departure}" departLane="random" departSpeed="10" />', file=routes)
                    if random_int == 3: 
                        print(f'<vehicle id="W2N{car}" type="standard_car" route="W2N" depart="{departure}" departLane="random" departSpeed="10" />', file=routes)  
                    if random_int == 4: 
                        print(f'<vehicle id="W2S{car}" type="standard_car" route="W2S" depart="{departure}" departLane="random" departSpeed="10" />', file=routes)
                    if random_int == 5: 
                        print(f'<vehicle id="S2W{car}" type="standard_car" route="S2W" depart="{departure}" departLane="random" departSpeed="10" />', file=routes)
                    if random_int == 6: 
                        print(f'<vehicle id="S2E{car}" type="standard_car" route="S2E" depart="{departure}" departLane="random" departSpeed="10" />', file=routes)
                    if random_int == 7: 
                        print(f'<vehicle id="E2N{car}" type="standard_car" route="E2N" depart="{departure}" departLane="random" departSpeed="10" />', file=routes)  
                    if random_int == 8: 
                        print(f'<vehicle id="E2S{car}" type="standard_car" route="E2S" depart="{departure}" departLane="random" departSpeed="10" />', file=routes)

            print("</routes>", file=routes)

generator(200).route_generator(0.75)