import geopandas as gpd
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation

from .geo_space import RasterLayer, VectorLayer, GeoSpace


class NycModel(Model):
    def __init__(self, num_agents):
        super().__init__()
        self.num_agents = num_agents
        self.grid = GeoSpace()
        self.schedule = RandomActivation(self)

        # Create raster layers
        self.grid.add_layer(RasterLayer.from_file("data/NYCLandsatST20170831.tif"))
        self.grid.add_layer(VectorLayer(gpd.GeoDataFrame.from_file("data/nybb.shp")))

    def step(self):
        # self.datacollector.collect(self)
        self.schedule.step()
