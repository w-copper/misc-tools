import pyproj
import numpy as np
proj1 = pyproj.CRS("EPSG:4326")
proj2 = pyproj.CRS("EPSG:32650")

trans = pyproj.Transformer.from_crs(proj1, proj2, always_xy=True)

p1 = np.array(trans.transform(114.32265, 30.516063, ))
p2 = np.array(trans.transform(114.321504,30.515517, ))

print(p2 - p1)

c = np.array((243605.09988933665, 3379509.491427537))

print(p2 - p1 + c)