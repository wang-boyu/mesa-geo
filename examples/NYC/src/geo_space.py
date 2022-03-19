from __future__ import annotations
from typing import List, Optional, Union
from itertools import zip_longest

import numpy as np
import geopandas as gpd
import pyproj
import rasterio
from pyproj import Transformer


class RasterLayer:
    _name: Optional[str]
    _values: np.ndarray
    _crs: Optional[pyproj.CRS]
    _transform: rasterio.transform.TransformerBase
    _bounds: List[List[float]]  # [[min_x, min_y], [max_x, max_y]]

    def __init__(self, name, values, crs, transform, bounds):
        self._name = name
        self._values = values
        self._crs = pyproj.CRS(crs) if crs else None
        self._transform = transform
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

    @property
    def transform(self) -> rasterio.transform.TransformerBase:
        return self._transform

    @classmethod
    def from_file(cls, raster_file: str, layer_name: str = None) -> RasterLayer:
        with rasterio.open(raster_file, "r") as dataset:
            values = dataset.read()
            bounds = [[dataset.bounds.left, dataset.bounds.bottom], [dataset.bounds.right, dataset.bounds.top]]
            crs = dataset.crs["init"].upper() if dataset.crs else None
            transform = dataset.transform
            return cls(name=layer_name, values=values, crs=crs, transform=transform, bounds=bounds)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, crs={self.crs}, bounds={self.bounds}, " \
               f"transform={repr(self.transform)}, values={repr(self.values)})"


class VectorLayer:
    _name: Optional[str]
    _data: gpd.GeoDataFrame

    # TODO replace self.data with self.crs, self.bounds, etc, and map gdf columns to attributes using setattr
    # _crs: pyproj.CRS
    # _bounds: List[List[float]]  # [[min_x, min_y], [max_x, max_y]]

    def __init__(self, gdf, name=None):
        self._name = name
        self._data = gdf

    @property
    def name(self) -> Optional[str]:
        return self._name

    @property
    def crs(self) -> Optional[pyproj.CRS]:
        # return self._crs
        return self._data.crs

    @property
    def bounds(self) -> List[List[float]]:
        # return self._bounds
        return [list(self._data.geometry.total_bounds[:2]), list(self._data.geometry.total_bounds[-2:])]

    @property
    def __geo_interface__(self) -> dict:
        return self._data.to_crs("EPSG:4326").__geo_interface__


class GeoSpace:
    _crs: pyproj.CRS
    _layers: List[Union[RasterLayer, VectorLayer]]
    _bounds: List[List[float]]  # [[min_x, min_y], [max_x, max_y]]

    def __init__(self):
        self._crs = pyproj.CRS("EPSG:4326")
        self._layers = []
        self._bounds = []

    @property
    def crs(self) -> pyproj.CRS:
        return self._crs

    @property
    def layers(self) -> List[Union[RasterLayer, VectorLayer]]:
        return self._layers

    @property
    def bounds(self) -> List[List[float]]:
        return self._bounds

    @property
    def agents(self) -> List:
        return []

    def add_layer(self, layer: Union[RasterLayer, VectorLayer]) -> None:
        self._layers.append(layer)
        proj = Transformer.from_crs(layer.crs, self.crs, always_xy=True) if layer.crs else None
        new_bounds = []
        for layer_bound, space_bound in zip_longest(layer.bounds, self.bounds):
            transformed_layer_x, transformed_layer_y = proj.transform(*layer_bound) if proj else layer_bound
            if space_bound:
                new_bounds.append([max(transformed_layer_x, space_bound[0]), max(transformed_layer_y, space_bound[1])])
            else:
                new_bounds.append([transformed_layer_x, transformed_layer_y])
        self._bounds = new_bounds
