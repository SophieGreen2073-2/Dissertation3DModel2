import numpy as np
import struct
from collections import deque

class UAVModel():
    def __init__(self, x, y, z, top_speed, danger_speed, start_speed,
                 acceleration, battery_life, charge_time, robot_id,
                 alias, drone_base, drone_target, current_path,
                 grid_width, grid_height, sim, fov_deg, max_range,
                    resolution = 0.2):
        print("Create new UAV")

        self.sim = sim

        # Robot status
        self.completed = False

        # Robot Position
        self.pos = (x, y)
        self.directions = {'north': [0, -1], 'south': [0, 1], 'east': [1, 0], 'west': [-1, 0], 'stay': [0,0], 'north_east': [1, -1], 'south_east': [1, 1], 'south_west': [-1, 1], 'north_west': [-1, -1]}

        # Robot Velocity and Acceleration
        self.top_speed = top_speed
        self.danger_speed = danger_speed
        self.start_speed = start_speed
        self.acceleration = acceleration

        # Robot battery life
        self.battery_life = battery_life
        self.charge_time = charge_time

        # Robot ID
        self.robot_id = robot_id
        self.alias = alias
        self.drone_base = drone_base

        # Robot and attributes paths (target, vision sensor etc.)
        self.drone_target = drone_target
        self.current_path = current_path
        self.cam_handle = self.sim.getObject('/Quadcopter/visionSensor')

        # Belief grid
        self.occupancy_grid = OccupancyBeliefGrid(resolution, grid_width, grid_height)
        
        # Camera model
        self.sensor = UAVSensor(fov_deg, max_range, self.cam_handle)

    # Move robot into next position
    def step_robot(self):
        next_wp = self.current_path[0]
        target_wx, target_wy = self.occupancy_grid.grid_to_world(next_wp[0], next_wp[1])
        
        # Send target to next waypoint
        self.sim.setObjectPosition(self.drone_target, -1, [target_wx, target_wy, 2.5])

        grid_pos = self.get_grid_pos()

        # Get grid pos
        # grid_pos = self.occupancy_grid.world_to_grid(self.pos[0], self.pos[1])

        # Check if drone reached waypoint
        distance = ((grid_pos[0] - next_wp[0])**2 + (grid_pos[1] - next_wp[1])**2)**0.5

        if distance < 0.2:
            self.current_path.pop(0)

        self.sensor.get_lidar_points(self.sim)
        self.occupancy_grid.update_belief(self.sensor.wall_points, self.sim, self.drone_base)


    def scan(self):
        self.sensor.get_lidar_points(self.sim)
        self.occupancy_grid.update_belief(self.sensor.wall_points, self.sim, self.drone_base)


    def get_grid_pos(self):
        uav_pos = self.sim.getObjectPosition(self.drone_base, -1)
        return self.occupancy_grid.world_to_grid(uav_pos[0], uav_pos[1])

    # Basic yamauchi move (move to the closest free square, no search for frontiers)
    def yamauchi_move(self):
        # Want to find closest frontier position (unobserved space)
        # Ony want to look at all edges that are not visted directly next to visited
        # Implementing a breadth first search
        
        # Return current position if the position is not scanned
        dest_location = []

        current_grid_pos = self.get_grid_pos()

        directions = ['north', 'south', 'east', 'west']
        queue = deque([current_grid_pos])
        visited = [[False for _ in range(self.occupancy_grid.width)] for _ in range(self.occupancy_grid.height)]
        visited[current_grid_pos[1]][current_grid_pos[0]] = True
        parent = {current_grid_pos: None}
        wall_belief = self.occupancy_grid.get_probability_grid()

        # Go through each position until there is an unknown space (frontier)
        while len(queue) != 0:
            cc, cr = queue.popleft()

            if 0.4 < wall_belief[cr, cc] < 0.6:
                dest_location = (cc, cr)
                break

            for dir in directions:
                dr = self.directions[dir][1]
                dc = self.directions[dir][0]
                grid_val = wall_belief[cr + dr, cc + dc]

                if 0 <= cc + dc < self.occupancy_grid.width and 0 <= cr + dr < self.occupancy_grid.height and not visited[cr + dr][cc + dc] and grid_val < 0.7:
                    visited[cr + dr][cc + dc] = True
                    queue.append((cc + dc, cr + dr))
                    parent[(cc + dc, cr + dr)] = (cc, cr)
        
        # Change this to do all steps towards frontier instead of just one to reduce calculation
        if len(dest_location) != 0:
            # Find the next step the UAV should take to get to the free space selected
            next_step = dest_location
            path_nodes = []
            while next_step is not None:
                path_nodes.append(next_step)
                next_step = parent[next_step]

            path_nodes.reverse()
            
            self.current_path = path_nodes
        else:
            self.moved = False


class OccupancyBeliefGrid():
    def __init__(self, resolution, grid_width, grid_height):
        # Grid belief
        self.x_min, self.x_max = -grid_width/2, grid_width/2
        self.y_min, self.y_max = -grid_height/2, grid_height/2
        self.resolution = resolution

        self.width = int((self.x_max - self.x_min)/ resolution)
        self.height = int((self.y_max - self.y_min) / resolution)

        self.belief_grid = np.zeros((self.height, self.width), dtype=np.float32)

        self.l_occ = 0.85   # Increase belief when wall is detected
        self.l_free = -0.4  # Decrease belief along free space
        self.l_max =  5     # Max possible belief
        self.l_min = -5     # Min possible belief


    # Convert world coordinates into grid belief position
    def world_to_grid(self, wx, wy):
        gx = np.floor((wx - self.x_min) / self.resolution).astype(int)
        gy = np.floor((wy - self.y_min)/ self.resolution).astype(int)
        return gx, gy

    # Convert grid coordinates into world coordinates
    def grid_to_world(self, gx, gy):
        wx = self.x_min + (gx + 0.5) * self.resolution
        wy = self.y_min + (gy + 0.5) * self.resolution
        return wx, wy

    # Update the belief grid based on the scanned wall_points
    def update_belief(self, wall_points, sim, drone_handle):
        # Get current drone position
        uav_pos = sim.getObjectPosition(drone_handle, -1)

        # Update current UAV position to be a belief of -5
        # Drone must be in a free space to be in that location
        uav_x, uav_y = uav_pos[0], uav_pos[1]
        uav_col, uav_row = self.world_to_grid(uav_x, uav_y)
        self.belief_grid[uav_row, uav_col] = -5 

        if len(wall_points) == 0:
            return

        # Convert wall to grid points
        wall_cols, wall_rows = self.world_to_grid(wall_points[:, 0], wall_points[:, 1])

        # Check that the wall is within the bounds of the grid
        valid_mask = (wall_cols >= 0) & (wall_cols < self.width) & \
                     (wall_rows >= 0) & (wall_rows < self.height)

        valid_cols = wall_cols[valid_mask]
        valid_rows = wall_rows[valid_mask]

        # Mark detected walls as occupied
        for r, c in zip(valid_rows, valid_cols):
            # Increase belief that there is a wall in that location
            self.belief_grid[r, c] += self.l_occ  

            self.clear_space(uav_col, uav_row, c, r)

        np.clip(self.belief_grid, self.l_min, self.l_max, out=self.belief_grid)


    # Mark space between drone and wall as clear
    # Distinguish between unscanned and unoccupied space 
    def clear_space(self, x_start, y_start, x_end, y_end):
        dx = abs(x_end - x_start)
        dy = abs(y_end - y_start)
        dir_x = 1 if x_start < x_end else -1
        dir_y = 1 if y_start < y_end else -1
        err = dx - dy

        curr_x, curr_y = x_start, y_start
        while curr_x != x_end or curr_y != y_end:
            if 0 <= curr_x < self.width and 0 <= curr_y < self.height:
                self.belief_grid[curr_y, curr_x] += self.l_free

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                curr_x += dir_x
            if e2 < dx:
                err += dx
                curr_y += dir_y

    # Get grid of probabilities that the space has a wall in it
    def get_probability_grid(self):
        return 1.0 / (1.0 + np.exp(self.belief_grid))


class UAVSensor():
    def __init__(self, fov_deg, max_range, cam_handle):
        print("sensor")

        self.fov_deg = fov_deg
        self.max_range = max_range
        self.cam_handle = cam_handle


    # Get LiDAR point cloud from CopelliaSim
    def get_lidar_points(self, sim):
        # Get raw depth matrix (1 = metres)
        depth_bytes, resolution = sim.getVisionSensorDepth(self.cam_handle, 1)

        if not depth_bytes or len(depth_bytes) == 0:
            return np.empty((0, 3), dtype=np.float32)

        width, height = resolution[0], resolution[1]
        depth_map = np.frombuffer(depth_bytes, dtype=np.float32).reshape(height, width)

        # Camera model
        fov_rad = np.radians(self.fov_deg)
        fx = width / (2.0 * np.tan(fov_rad / 2.0))
        fy = fx
        cx = width / 2.0
        cy = height / 2.0

        # Create pixel coordinate grid
        u, v = np.meshgrid(np.arange(width), np.arange(height))

        # Filter out background clipping plane and empty depth points
        valid_mask = (depth_map > 0.08) & (depth_map < (self.max_range * 0.99))

        z = depth_map[valid_mask]
        u_val = u[valid_mask]
        v_val = v[valid_mask]

        if len(z) == 0:
            return np.empty((0, 3), dtype=np.float32)

        # Project 2D pixels (u, v, z) to 3D camera local frame (x, y, z)
        x = (u_val - cx) * z / fx
        y = (v_val - cy) * z / fy
        local_pts = np.column_stack((x, y, z))

        # Transform camera local points into world coordinates
        cam_matrix = np.array(sim.getObjectMatrix(self.cam_handle)).reshape(3, 4) 
        R = cam_matrix[:, :3]
        T = cam_matrix[:, 3]

        world_pts = np.dot(local_pts, R.T) + T

        self.wall_points = self.extract_wall_points(world_pts)

        if len(self.wall_points) > 0:
            print(f"Captured {len(self.wall_points)} wall points! Sample point: {self.wall_points[0]}")


    def extract_wall_points(self, world_points, min_z=0.2, max_z=2.0):
        if len(world_points) == 0:
            return np.empty((0, 3), dtype=np.float32)

        wall_mask = (world_points[:, 2] >= min_z) & (world_points[:, 2] <= max_z)
        return world_points[wall_mask]

