from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np

class AreaModel():
    def __init__(self, sim, height, width, num_ugvs, resolution):
        print("Build area model")

        self.walls = []
        self.resolution = resolution

        self.x_min, self.x_max = -width/2, width/2
        self.y_min, self.y_max = -height/2, height/2
        self.resolution = resolution

        self.width = int((self.x_max - self.x_min)/ resolution)
        self.height = int((self.y_max - self.y_min) / resolution)

        self.walls_grid = np.zeros((self.height, self.width), dtype=np.float32)
        self.overlap_area = np.zeros((self.height, self.width, num_ugvs))

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

    # Convert world coordinates into grid belief position
    def world_to_grid(self, wx, wy):
        gx = np.floor((wx - self.x_min) / self.resolution).astype(int)
        gy = np.floor((self.y_max - wy)/ self.resolution).astype(int)
        return gx, gy

    # Create grid of walls
    def create_grid_of_walls(self):
        for wall in self.walls:
            g_xmin, g_ymin_idx = self.world_to_grid(wall["min_x"], wall["max_y"])
            g_xmax, g_ymax_idx = self.world_to_grid(wall["max_x"], wall["min_y"])
            
            # Ensure proper ordering for grid slicing (min to max)
            gx_start = max(0, min(g_xmin, g_xmax))
            gx_end = min(self.width, max(g_xmin, g_xmax) + 1)
            
            gy_start = max(0, min(g_ymin_idx, g_ymax_idx))
            gy_end = min(self.height, max(g_ymin_idx, g_ymax_idx) + 1)
            
            # Mark the wall region as occupied in the grid
            self.walls_grid[gy_start:gy_end, gx_start:gx_end] = 1.0
