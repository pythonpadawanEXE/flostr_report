from pathlib import Path

import numpy as np

import dmsh
import skfem
import skfem.io.json

length = 2.2
height = 0.41
radius = 0.05
centre = (0.2, 0.2)

geo = dmsh.Difference(
    dmsh.Rectangle(0.0, length, 0.0, height), dmsh.Circle(centre, radius)
)
geo.feature_points = np.append(
    geo.feature_points, centre + radius * np.outer([-1, 1], [1, 0]), axis=0
)

points, triangles = dmsh.generate(geo, radius / 2, tol=1e-9)
mesh = skfem.MeshTri(points.T, triangles.T)
mesh.define_boundary("inlet", lambda x: x[0] == 0.0)
mesh.define_boundary("outlet", lambda x: x[0] == length)

skfem.io.json.to_file(mesh, Path(__file__).with_suffix(".json"))
