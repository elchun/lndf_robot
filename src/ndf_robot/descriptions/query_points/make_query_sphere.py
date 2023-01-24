import numpy as np
import trimesh

from airobot.utils import common

sphere = trimesh.load('1250_polygon_sphere_100mm.STL')

scale_factor = 0.5
scaler = np.eye(4)
scaler[:3, :3] *= scale_factor 


print(sphere.extents)
sphere.apply_scale(0.5)
# sphere.apply_transform(scaler)
print('---')
centering_transform = trimesh.bounds.oriented_bounds(sphere)[0]
sphere.apply_transform(centering_transform)
print(sphere.extents)


sphere.export('sphere.obj')