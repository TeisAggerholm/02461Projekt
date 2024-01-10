import traci
import os
import numpy as np
import random

class CrossIntersection():

    def __init__(self, sumo_mode, min_green_phase_steps, yellow_phase_steps, red_phase_steps, max_step, percentage_straight, car_intensity_per_min, spredning):
        self.sumo_path = 'sumo_files/osm.sumocfg'
        self.net_path = "sumo_files/osm.netccfg"
        self.sumo_mode = sumo_mode
        self.max_step = max_step
        self.percentage_straight = percentage_straight
        self.car_intensity_per_min = car_intensity_per_min
        self.spredning = spredning

        self.min_green_phase_steps = min_green_phase_steps
        self.yellow_phase_steps = yellow_phase_steps
        self.red_phase_steps = red_phase_steps

        self.num_actions = 2
        self.actions_def = {
            0: {'green_phase_index': 0, 'yellow_phase_index': 1},
            1: {'green_phase_index': 3, 'yellow_phase_index': 4}
        }

        self.phases = [
            traci.trafficlight.Phase(10000, "rrrrGGggrrrrGGgg"), # NS green phase
            traci.trafficlight.Phase(10000, "rrrryyyyrrrryyyy"), # NS yellow phase
            traci.trafficlight.Phase(10000, "rrrrrrrrrrrrrrrr"), # all red phase
            traci.trafficlight.Phase(10000, "GGggrrrrGGggrrrr"), # EW green phase
            traci.trafficlight.Phase(10000, "yyyyrrrryyyyrrrr"), # EW yellow phase
        ]

        self.lane = ["-125514713_0", "-125514711_0", "-548975769_0", "125514709_0"]
        self.traffic_light_system_id = "24960712"

        self.phaseQueue = []
        self.steps_in_current_phase = 0

    def run_env(self):
        self.route_generate(self.max_step, self.percentage_straight, self.car_intensity_per_min, self.spredning)
        traci.start([self.sumo_mode, "-c", self.sumo_path])
        self._set_phases()

    def route_generate(self, max_step, percentage_straight, car_intensity_per_min, spredning): 
        #Anvend ved kontrol: seed. 

        #Antal biler: 
        n_cars = np.random.normal(max_step/60*car_intensity_per_min,spredning)

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

    def push_yellow_phase(self, action):
        phase_index = self.actions_def[action]['yellow_phase_index']
        self.phaseQueue.append({"phase_index": phase_index, "steps_to_do": self.yellow_phase_steps})

    def push_red_phase(self):
        phase_index = 2
        self.phaseQueue.append({"phase_index": phase_index, "steps_to_do": self.red_phase_steps})

    def push_green_phase(self, action):
        phase_index = self.actions_def[action]['green_phase_index']
        self.phaseQueue.append({"phase_index": phase_index, "steps_to_do": self.min_green_phase_steps})

    def isActionable(self):
        if len(self.phaseQueue) == 0:
            return True

    def get_traffic_light_state(self):

        state_string = traci.trafficlight.getRedYellowGreenState(self.traffic_light_system_id)
        light_states = {
            'red': [],
            'yellow': [],
            'green': []
        }
        
        # Loop through the state string and populate the dictionary
        for index, state in enumerate(state_string):
            if state == 'r':
                light_states['red'].append(index)
            elif state == 'y':
                light_states['yellow'].append(index)
            elif state.lower() == 'g':  # 'G' or 'g' can be used for green
                light_states['green'].append(index)
        
        return light_states

    def increment_steps_in_current_phase(self):
        if len(self.phaseQueue) > 0:
            self.steps_in_current_phase += 1     
    
    def update_current_phase(self):
        if len(self.phaseQueue) > 0:
            current_phase = self.phaseQueue[0]
            if current_phase["steps_to_do"] == self.steps_in_current_phase:
                self.phaseQueue.pop(0)
                self.steps_in_current_phase = 0
        
    def set_lights(self):
        if len(self.phaseQueue) > 0:
            traci.trafficlight.setPhase(self.traffic_light_system_id, self.phaseQueue[0]["phase_index"])
    



        # if len(self.phaseQueue) > 0:
        #     pastPhase = self.phaseQueue[0]
        #     self.steps_in_current_phase += 1

        #     if pastPhase["steps_to_do"] < self.steps_in_current_phase:
        #         self.phaseQueue.pop()
        #         self.steps_in_current_phase = 0

        #     print(self.steps_in_current_phase)  

        #     if len(self.phaseQueue) > 0:
        #         activated_phase_index = traci.trafficlight.getPhase(self.traffic_light_system_id)
        #         currentPhase = self.phaseQueue[0]

        #         if activated_phase_index != currentPhase["phase_index"]:
        #             traci.trafficlight.setPhase(self.traffic_light_system_id, currentPhase["phase_index"])

        #     # self.steps_in_current_phase = 0
        #     # self.steps_in_current_phase += 1

        #     # activated_phase_index = traci.trafficlight.getPhase(self.traffic_light_system_id)
        #     # currentPhase = self.phaseQueue[0]

        #     # if activated_phase_index != currentPhase["phase_index"]:
        #     #     print(currentPhase["phase_index"])
        #     #     traci.trafficlight.setPhase(self.traffic_light_system_id, currentPhase["phase_index"])

if __name__ == '__main__':
    sumo_config_path = 'sumo_files/osm.sumocfg'

    traci.start(["sumo-gui", "-c", sumo_config_path])

    environment = CrossIntersection(3)
    environment._set_phases()
    environment._get_phases()

    for step in range(200):
        traci.simulationStep()
        print(step)
        environment.get_traffic_light_state()


    traci.close()
