import geopandas as gpd
import geoplot as gplt
import numpy as np

def get_world():
    world = gpd.read_file(gplt.datasets.get_path('world'))
    geometries = []
    for g in world.geometry:
        if 'geoms' in dir(g): 
            g = g.geoms[np.argmax([i.area for i in g.geoms])]
        geometries.append(g)
    
    world['geometry'] = geometries
    return world

