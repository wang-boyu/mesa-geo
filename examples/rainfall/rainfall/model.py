from mesa import Model
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation

from .raster_space import RasterLayer, RasterSpace


class RainfallModel(Model):
    def __init__(self, num_agents):
        super().__init__()
        self.num_agents = num_agents
        self.grid = RasterSpace()
        self.schedule = RandomActivation(self)

        # Create raster layers
        self.grid.add_layer(RasterLayer.from_file("rainfall/FAA_UTM18N_NAD83.tif"))

    def step(self):
        # self.datacollector.collect(self)
        self.schedule.step()
