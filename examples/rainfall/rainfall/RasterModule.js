var RasterModule = function (view, zoom, map_width, map_height) {
    // Create the map tag:
    var map_tag = "<div style='width:" + map_width + "px; height:" + map_height + "px;border:1px dotted' id='mapid'></div>"
    // Append it to body:
    var div = $(map_tag)[0]
    $('#elements').append(div)

    // Create Leaflet map and raster layers
    var Lmap = L.map('mapid', {zoomSnap: 0.1}).setView(view, zoom)
    var dummyUrl = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNgYAAAAAMAASsJTYQAAAAASUVORK5CYII='
    var rasterLayers = [L.imageOverlay(dummyUrl, [[0, 0], [0, 0]]).addTo(Lmap)]

    this.render = function (imageOverlay) {
        rasterLayers.forEach(function (layer) {
            layer.remove()
        })
        rasterLayers = []
        imageOverlay.urls.forEach(function (layer) {
            rasterLayers.push(L.imageOverlay(layer, imageOverlay.bounds).addTo(Lmap))
        })
        Lmap.fitBounds(imageOverlay.bounds)
    }

    this.reset = function () {
        rasterLayers.forEach(function (layer) {
            layer.remove()
        })
        rasterLayers = [L.imageOverlay(dummyUrl, [[0, 0], [0, 0]]).addTo(Lmap)]
    }
}
