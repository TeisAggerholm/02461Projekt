from simulation import simulation
from route_generator import generator

max_step = 200


generator(max_step).route_generator()
print(simulation('sumo_files/osm.sumocfg', max_step, False).run())

