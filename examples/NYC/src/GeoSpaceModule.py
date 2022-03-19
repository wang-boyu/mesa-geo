from mesa.visualization.ModularVisualization import VisualizationElement
from folium.utilities import image_to_url

from .geo_space import RasterLayer, VectorLayer


class GeoSpaceModule(VisualizationElement):
    package_includes = ["leaflet.js"]
    local_includes = ["src/GeoSpaceModule.js"]

    def __init__(self, portrayal_method, view=[0, 0], zoom=10, map_height=500, map_width=500):
        super().__init__()

        self.portrayal_method = portrayal_method
        self.map_height = map_height
        self.map_width = map_width
        self.view = view
        new_element = "new GeoSpaceModule({}, {}, {}, {})"
        new_element = new_element.format(view, zoom, map_width, map_height)
        self.js_code = "elements.push(" + new_element + ");"

    def render_layers(self, model):
        layers = {
            "rasters": [image_to_url(layer.values.transpose([1, 2, 0]))
                        for layer in model.grid.layers if isinstance(layer, RasterLayer)],
            "vectors": [layer.__geo_interface__ for layer in model.grid.layers if isinstance(layer, VectorLayer)],
            # longlat to latlong
            "bounds": [list(reversed(model.grid.bounds[0])), list(reversed(model.grid.bounds[1]))]
        }
        return layers

    def render(self, model):
        featurecollection = dict(type="FeatureCollection", features=[])
        for _, agent in enumerate(model.grid.agents):
            shape = agent.__geo_interface__()
            portrayal = self.portrayal_method(agent)
            for key, value in portrayal.items():
                shape["properties"][key] = value
                featurecollection["features"].append(shape)
        return featurecollection
