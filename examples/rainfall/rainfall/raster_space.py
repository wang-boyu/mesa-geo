from __future__ import annotations
from typing import List, Optional
from itertools import zip_longest

import numpy as np
import pyproj
import rasterio
from pyproj import Transformer


class RasterLayer:
    _name: Optional[str]
    _values: np.ndarray
    _crs: Optional[pyproj.CRS]
    _bounds: List[List[float]]  # [[min_lat, min_lon], [max_lat, max_lon]]

    def __init__(self, name, values, crs, bounds):
        self._name = name
        self._values = values
        self._crs = pyproj.CRS(crs) if crs else None
        self._bounds = bounds

    @property
    def shape(self):
        return self.values.shape

    @property
    def name(self) -> Optional[str]:
        return self._name

    @property
    def crs(self) -> Optional[pyproj.CRS]:
        return self._crs

    @property
    def bounds(self) -> List[List[float]]:
        return self._bounds

    @property
    def values(self) -> np.ndarray:
        return self._values

    @classmethod
    def from_file(cls, raster_file: str, layer_name: str = None) -> RasterLayer:
        with rasterio.open(raster_file, "r") as dataset:
            values = dataset.read()
            min_lon, min_lat, max_lon, max_lat = dataset.bounds
            # converting to lat/lon from lon/lat
            bounds = [[min_lat, min_lon], [max_lat, max_lon]]
            crs = dataset.crs["init"].upper() if dataset.crs else None
            return cls(name=layer_name, values=values, crs=crs, bounds=bounds)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, crs={self.crs}, bounds={self.bounds}, " \
               f"values.shape={self.values.shape})"


class RasterSpace:
    _crs: pyproj.CRS
    _layers: List[RasterLayer]
    _bounds: List[List[float]]  # [[min_lat, min_lon], [max_lat, max_lon]]

    def __init__(self):
        self._crs = pyproj.CRS("EPSG:4326")
        self._layers = []
        self._bounds = []

    @property
    def crs(self) -> pyproj.CRS:
        return self._crs

    @property
    def layers(self) -> List[RasterLayer]:
        return self._layers

    @property
    def bounds(self) -> List[List[float]]:
        return self._bounds

    def add_layer(self, layer: RasterLayer) -> None:
        self._layers.append(layer)
        proj = Transformer.from_crs(layer.crs, self.crs, always_xy=True) if layer.crs else None
        new_bounds = []
        for layer_bound, space_bound in zip_longest(layer.bounds, self.bounds):
            if proj:
                transformed_layer_lon, transformed_layer_lat = proj.transform(layer_bound[1], layer_bound[0])
            else:
                transformed_layer_lat, transformed_layer_lon = layer_bound
            if space_bound:
                new_bounds.append([max(transformed_layer_lat, space_bound[0]), (transformed_layer_lon, space_bound[1])])
            else:
                new_bounds.append([transformed_layer_lat, transformed_layer_lon])
        self._bounds = new_bounds
