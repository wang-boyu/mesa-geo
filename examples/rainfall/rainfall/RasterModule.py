from mesa.visualization.ModularVisualization import VisualizationElement
from folium.utilities import image_to_url


class RasterModule(VisualizationElement):
    package_includes = ["leaflet.js"]
    local_includes = ["rainfall/RasterModule.js"]

    def __init__(self, view=[0, 0], zoom=10, map_height=500, map_width=500):
        super().__init__()

        self.map_height = map_height
        self.map_width = map_width
        self.view = view
        new_element = "new RasterModule({}, {}, {}, {})"
        new_element = new_element.format(view, zoom, map_width, map_height)
        self.js_code = "elements.push(" + new_element + ");"

    def render(self, model):
        layers = {
            "rasters": [image_to_url(layer.values.transpose([1, 2, 0])) for layer in model.grid.layers],
            "bounds": model.grid.bounds
        }
        return layers
