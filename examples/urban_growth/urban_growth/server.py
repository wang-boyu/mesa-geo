from __future__ import annotations
from typing import Dict

import mesa

from mesa_geo.visualization.ModularVisualization import ModularServer
from mesa_geo.visualization.modules import MapModule
from .model import UrbanGrowth
from .space import UrbanCell


def cell_portrayal(cell: UrbanCell) -> Dict[str, str | float]:
    portrayal = {}
    if cell.urban:
        if cell.new_urbanized:
            portrayal["color"] = "Red"
        else:
            portrayal["color"] = "Blue"
    else:
        portrayal["opacity"] = 0.0
    return portrayal


class UrbanizedText(mesa.visualization.TextElement):
    def render(self, model):
        return f"Percentage Urbanized: {model.pct_urbanized:.2f}%"


model_params = {
    "max_coefficient": mesa.visualization.NumberInput("max_coefficient", 100),
    "dispersion_coefficient": mesa.visualization.Slider(
        "dispersion_coefficient", 20, 0, 100, 1
    ),
    "spread_coefficient": mesa.visualization.Slider(
        "spread_coefficient", 27, 0, 100, 1
    ),
    "breed_coefficient": mesa.visualization.Slider("breed_coefficient", 5, 0, 100, 1),
    "rg_coefficient": mesa.visualization.Slider("rg_coefficient", 10, 0, 100, 1),
    "slope_coefficient": mesa.visualization.Slider("slope_coefficient", 50, 0, 100, 1),
    "critical_slope": mesa.visualization.Slider("critical_slope", 25, 0, 100, 1),
    "road_influence": mesa.visualization.Choice(
        "road_influence", False, choices=[True, False]
    ),
}


map_module = MapModule(
    portrayal_method=cell_portrayal,
    view=[12.904598815296707, -8.027435210420451],
    zoom=12.1,
    map_height=394,
    map_width=531,
)
urbanized_text = UrbanizedText()
urbanized_chart = mesa.visualization.ChartModule(
    [
        {"Label": "Percentage Urbanized", "Color": "Black"},
    ]
)

server = ModularServer(
    UrbanGrowth,
    [map_module, urbanized_text, urbanized_chart],
    "UrbanGrowth Model",
    model_params,
)
