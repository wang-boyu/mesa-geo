from folium.utilities import image_to_url
from shapely.geometry import mapping
from mesa.visualization.ModularVisualization import VisualizationElement

from mesa_geo.geospace import RasterLayer, VectorLayer


class MapModule(VisualizationElement):
    """A MapModule for Leaflet maps."""

    package_includes = ["leaflet.js", "LeafletMap.js"]
    local_includes = []

    def __init__(
        self, portrayal_method=None, view=[0, 0], zoom=10, map_height=500, map_width=500
    ):
        self.portrayal_method = portrayal_method
        self.map_height = map_height
        self.map_width = map_width
        self.view = view
        new_element = "new MapModule({}, {}, {}, {})"
        new_element = new_element.format(view, zoom, map_width, map_height)
        self.js_code = "elements.push(" + new_element + ");"

    def render_layers(self, model):
        layers = {"rasters": [], "vectors": [], "bounds": []}
        for layer in model.space.layers:
            if isinstance(layer, RasterLayer):
                layers["rasters"].append(
                    image_to_url(layer.values.transpose([1, 2, 0]))
                )
            elif isinstance(layer, VectorLayer):
                layers["vectors"].append(layer.__geo_interface__)
        # longlat to latlong
        if model.space.bounds:
            layers["bounds"] = [
                list(reversed(model.space.bounds[0])),
                list(reversed(model.space.bounds[1])),
            ]
        return layers

    def render(self, model):
        feature_collection = {"type": "FeatureCollection", "features": []}
        for agent in model.space.agents:
            transformed_geometry = agent.get_transformed_geometry(
                model.space.Transformer
            )
            properties = self.portrayal_method(agent) if self.portrayal_method else {}
            feature_collection["features"].append(
                {
                    "type": "Feature",
                    "geometry": mapping(transformed_geometry),
                    "properties": properties,
                }
            )
        return feature_collection
