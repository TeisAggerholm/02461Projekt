import traci
import numpy as np

class CrossIntersection():
    def __init__(self, sumo_mode, min_green_phase_steps, yellow_phase_steps, red_phase_steps, max_step):
        self.sumo_path = 'sumo_files/osm.sumocfg'
        self.net_path = "sumo_files/osm.netccfg"
        self.sumo_mode = sumo_mode
        self.max_step = max_step
        self.min_green_phase_steps = min_green_phase_steps
        self.yellow_phase_steps = yellow_phase_steps
        self.red_phase_steps = red_phase_steps

        self.num_actions = 2

        self.actions_def = {
            0: {'green_phase_index': 0, 'yellow_phase_index': 1},
            1: {'green_phase_index': 3, 'yellow_phase_index': 4}
        }

        self.phases = [
            traci.trafficlight.Phase(10, "rrrrGGggrrrrGGgg"), # NS green phase
            traci.trafficlight.Phase(2, "rrrryyyyrrrryyyy"), # NS yellow phase
            traci.trafficlight.Phase(2, "rrrrrrrrrrrrrrrr"), # all red phase
            traci.trafficlight.Phase(10, "GGggrrrrGGggrrrr"), # EW green phase
            traci.trafficlight.Phase(2, "yyyyrrrryyyyrrrr"), # EW yellow phase
        ]

        self.lane = ["-125514713_0", "-125514711_0", "-548975769_0", "125514709_0"]
        self.traffic_light_system_id = "24960712"

    def run_env(self):
        self.route_generate(self.max_step, 0.75)        
        traci.start([self.sumo_mode, "-c", self.sumo_path])
        self._set_phases()

    def _get_phases(self):
        tls_definitions = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.traffic_light_system_id)
        for logic in tls_definitions:
            print(f"Traffic Light {self.traffic_light_system_id} has the following phases in program {logic.programID}:")
            for phase in logic.getPhases():
                print(f"Phase index {logic.getPhases().index(phase)}: {phase.state} with duration {phase.duration} seconds")

    def _set_phases(self):

        logic = traci.trafficlight.Logic(
            programID="newProgramID",
            type=0, 
            currentPhaseIndex=0,
            phases=self.phases
        )

        traci.trafficlight.setCompleteRedYellowGreenDefinition(self.traffic_light_system_id, logic)
  
    def set_yellow_phase(self, action):
        phase_index = self.actions_def[action]['yellow_phase_index']
        traci.trafficlight.setPhase(self.traffic_light_system_id, phase_index)

        return self.yellow_phase_steps

    def set_red_phase(self):
        phase_index = 2
        traci.trafficlight.setPhase(self.traffic_light_system_id, phase_index)
        return self.red_phase_steps

    def set_green_phase(self, action):
        phase_index = self.actions_def[action]['green_phase_index']
        traci.trafficlight.setPhase(self.traffic_light_system_id, phase_index)

        return self.min_green_phase_steps

    def route_generate(self, max_step, percentage_straight): 
        #Anvend ved kontrol: seed. 

        #Antal biler: 
        #n_cars = abs(np.random.normal(max_step/60*car_intensity_per_min,spredning))
        n_cars = 100
        
        #Hvilken fordeling anvendes til bilernes tider? 
        tider = np.random.weibull(2, int(n_cars))
        tider.sort()

        #Normalisering ift. max_steps 
        new_times = []
        for tid in tider: 
            new_times.append(tid / tider[-1] * max_step)

        #Getting the map url: 
        url = self.get_map_url()

        with open("sumo_files/osm.passenger.trips.xml", "w") as routes:
            print(f"""<?xml version="1.0" encoding="UTF-8"?>

            <routes xmlns:xsi="{url}">
   
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
                        route = "N2S"
                    if random_int == 2: 
                        route = "W2E"
                    if random_int == 3: 
                        route = "S2N"
                    if random_int == 4: 
                        route = "E2W"

                else: 
                    random_int = np.random.randint(1,9)
                    if random_int == 1: 
                        route = "N2W"
                    if random_int == 2: 
                        route = "N2E"
                    if random_int == 3: 
                        route = "W2N"
                    if random_int == 4: 
                        route = "W2S"
                    if random_int == 5: 
                        route = "S2W"
                    if random_int == 6: 
                        route = "S2E"
                    if random_int == 7: 
                        route = "E2N"
                    if random_int == 8: 
                        route = "E2S"
                    
                print(f'<vehicle id="{route}{car}" type="standard_car" route="{route}" depart="{departure}" departLane="random" departSpeed="10" />', file=routes)

            print("</routes>", file=routes)
    
    def get_map_url(self): 
        import re
        pattern = r'xmlns:xsi="([^"]+)"'

        with open(self.net_path, "r") as file:
            for line in file: 
                if "configuration xmlns:xsi=" in line: 
                    match = re.search(pattern, line)
                    if match:
                        xsi_link = match.group(1)
                        return xsi_link
                    else:
                        raise ValueError(f"Failed to extract xsi_link from link{self.net_path}")
