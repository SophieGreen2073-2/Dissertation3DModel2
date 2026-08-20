from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np

class AreaModel():
    def __init__(self, sim, height, width, num_ugvs, resolution):
        print("Build area model")

        self.walls = []
        self.resolution = resolution
        self.overlap_area = np.zeros((height * resolution, width * resolution, num_ugvs))

        # Get parent object
        parent_name = "/Walls"
        parent_handle = sim.getObject(parent_name)

        self.wall_collection_handle = sim.createCollection(0)
        sim.addItemToCollection(self.wall_collection_handle, sim.handle_tree, parent_handle, 0)

        # Get all shapes inside this branch
        wall_handles = sim.getObjectsInTree(parent_handle, sim.object_shape_type)

        self.walls = []

        for handle in wall_handles:
            name = sim.getObjectAlias(handle)

            # Get 3D world position
            # -1 = World frame
            pos = sim.getObjectPosition(handle, -1)

            # Local bounding box dimensions
            min_x = sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_min_x)
            max_x = sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_max_x)
            min_y = sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_min_y)
            max_y = sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_max_y)
            min_z = sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_min_z)
            max_z = sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_max_z)

            wall_info = {
                'name': name,
                'min_x': pos[0] + min_x,
                'max_x': pos[0] + max_x,
                'min_y': pos[1] + min_y,
                'max_y': pos[1] + max_y,
                'min_z': pos[2] + min_z,
                'max_z': pos[2] + max_z
            }

            self.walls.append(wall_info)

        return