import traci

class SmidtVækSimulationMethods:
    def __init__():
        pass
    
    def _get_vehicle_stats(self):
        # STATS
        vehicles = traci.vehicle.getIDList()
        vehicle_waiting_times = []

        for vehicle in vehicles:
            vehicle_waiting_times.append(self.get_waiting_time(vehicle))
            self.co2_emission += self.get_co2_emission(vehicle)
            self.sum_queue = sum(self.get_queue_length())
        
        return {"wait_times": vehicle_waiting_times, "total_co2": self.co2_emission, "queues": self.get_queue_length()}

    def get_co2_emission(self,vehicle):
        self.vehicle = vehicle
        co2_emission_car = traci.vehicle.getCO2Emission(self.vehicle)
        return co2_emission_car
          
    def count_halting_vehicles(self):
        total_vehicle_count = 0
        halting_count = 0
        for value in self.waiting_times.values():
            total_vehicle_count += 1
            if value != 0:
                halting_count += 1
        return total_vehicle_count, halting_count
    
    def calc_overall_score(self):
        overall_score = self.final_score_weights["total_waiting_time"] * self.stats["total_waiting_time"]
        return overall_score
    
