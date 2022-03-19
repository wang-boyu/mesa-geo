var GeoSpaceModule = function (view, zoom, map_width, map_height) {
    // Create the map tag:
    var map_tag = "<div style='width:" + map_width + "px; height:" + map_height + "px;border:1px dotted' id='mapid'></div>"
    // Append it to body:
    var div = $(map_tag)[0]
    $('#elements').append(div)

    // Create Leaflet map and agent layer
    var Lmap = L.map('mapid', {zoomSnap: 0.1}).setView(view, zoom)
    var agentLayer = L.geoJSON().addTo(Lmap)

    this.renderLayers = function (layers) {
        layers.rasters.forEach(function (layer) {
            L.imageOverlay(layer, layers.bounds).addTo(Lmap)
        })
        layers.vectors.forEach(function (layer) {
            L.geoJSON(layer).addTo(Lmap)
        })
        Lmap.fitBounds(layers.bounds)
    }

    this.render = function (data) {
        agentLayer.remove()
        agentLayer = L.geoJSON(data, {
            onEachFeature: PopUpProperties,
            style: function (feature) {
                return {color: feature.properties.color};
            },
            pointToLayer: function (feature, latlang) {
                return L.circleMarker(latlang, {radius: feature.properties.radius, color: feature.properties.color});
            }
        }).addTo(Lmap)
    }

    this.reset = function () {
        agentLayer.remove()
    }
}

function PopUpProperties(feature, layer) {
    var popupContent = '<table>'
    if (feature.properties) {
        for (var p in feature.properties) {
            popupContent += '<tr><td>' + p + '</td><td>' + feature.properties[p] + '</td></tr>'
        }
    }
    popupContent += '</table>'
    layer.bindPopup(popupContent)
}