from mesa_geo.visualization.ModularVisualization import ModularServer

from .RasterModule import RasterModule
from .model import RainfallModel


def agent_portrayal(agent):
    portrayal = {"Shape": "circle",
                 "Color": "red",
                 "Filled": "true",
                 "Layer": 0,
                 "r": 0.5}
    return portrayal


raster_element = RasterModule(map_height=500, map_width=500)

server = ModularServer(RainfallModel,
                       [raster_element],
                       "Rainfall Model",
                       {"num_agents": 100})
