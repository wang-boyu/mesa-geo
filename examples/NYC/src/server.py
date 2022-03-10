from mesa_geo.visualization.ModularVisualization import ModularServer

from .GeoSpaceModule import GeoSpaceModule
from .model import NycModel


def agent_portrayal(agent):
    portrayal = {"Shape": "circle",
                 "Color": "red",
                 "Filled": "true",
                 "Layer": 0,
                 "r": 0.5}
    return portrayal


geospace_element = GeoSpaceModule(map_height=500, map_width=500)

server = ModularServer(NycModel,
                       [geospace_element],
                       "NYC Model",
                       {"num_agents": 100})
