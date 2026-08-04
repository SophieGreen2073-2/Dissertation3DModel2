import numpy as np
import struct

class UAVModel():
    def __init__(self, x, y, z, top_speed, danger_speed, start_speed,
                 acceleration, battery_life, charge_time, robot_id,
                 alias, drone_base, drone_target, current_path,
                 grid_width, grid_height, sim, fov_deg, max_range,
                    resolution = 0.2):
        print("Create new UAV")

        self.sim = sim

        # Robot Position
        self.x_pos = x
        self.y_pos = y
        self.z_pos = z

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

        # Grid belief
        self.x_min, self.x_max = -grid_width/2, grid_width/2
        self.y_min, self.y_max = -grid_height/2, grid_height/2
        self.resolution = resolution

        self.width = int((self.x_max - self.x_min)/ resolution)
        self.height = int((self.y_max - self.y_min) / resolution)

        self.belief_grid = np.full((self.width, self.height), 0.5, dtype=np.float32)

        # Camera model
        self.fov_deg = fov_deg
        self.max_range = max_range


    # Move robot into next position
    def step_robot(self):
        next_wp = self.current_path[0]
        
        # Send target to next waypoint
        self.sim.setObjectPosition(self.drone_target, -1, [next_wp[0], next_wp[1], next_wp[2]])

        # Check if drone reached waypoint
        distance = ((self.pos[0] - next_wp[0])**2 + (self.pos[1] - next_wp[1])**2 + (self.pos[2] - next_wp[2])**2)**0.5

        if distance < 0.2:
            self.current_path.pop(0)

        self.get_lidar_points()


    # Convert world coordinates into grid belief position
    def world_to_grid(self, wx, wy):
        gx = int((wx - self.x_min) / self.resolution)
        gy = int((wy - self.y_min)/ self.resolution)
        return gx, gy


    # Get LiDAR point cloud from CopelliaSim
    def get_lidar_points(self):
        # Get raw depth matrix (1 = metres)
        depth_bytes, resolution = self.sim.getVisionSensorDepth(self.cam_handle, 1)

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
        cam_matrix = np.array(self.sim.getObjectMatrix(self.cam_handle)).reshape(3, 4) 
        R = cam_matrix[:, :3]
        T = cam_matrix[:, 3]

        world_pts = np.dot(local_pts, R.T) + T

        wall_points = self.extract_wall_points(world_pts)

        if len(wall_points) > 0:
            print(f"Captured {len(wall_points)} wall points! Sample point: {wall_points[0]}")

        # signal_name = f"measuredDataAtThisTime"

        # # try:
        # raw_buffer = self.sim.getStringSignal(signal_name)
        # if raw_buffer is None:
        #     return []

        # # Reshape into list of 3D points [[x, y, z], ...]
        # num_floats = len(raw_buffer) // 4
        # floats = struct.unpack(f'{num_floats}f', raw_buffer)
        
        # # 3. Group into [x, y, z] points
        # points = [[floats[i], floats[i+1], floats[i+2]] for i in range(0, len(floats), 3)]
        # return points
        # except Exception:
            # return []


    def extract_wall_points(self, world_points, min_z=0.2, max_z=2.0):
        if len(world_points) == 0:
            return np.empty((0, 3), dtype=np.float32)

        wall_mask = (world_points[:, 2] >= min_z) & (world_points[:, 2] <= max_z)
        return world_points[wall_mask]

    # Basic yamauchi move (move to the closest free square, no search for frontiers)
    # def yamauchi_move(self, area: AreaModel, robot_start_id):
    #     # Want to find closest frontier position (unobserved space)
    #     # Ony want to look at all edges that are not visted directly next to visited
    #     # Implementing a breadth first search
        
    #     # Return current position if the position is not scanned
    #     dest_location = []

    #     current_grid_pos = self.get_grid_pos()

    #     directions = ['north', 'south', 'east', 'west']
    #     queue = deque([current_grid_pos])
    #     visited = [[False for _ in range(self.scanned_grid.width)] for _ in range(self.scanned_grid.height)]
    #     visited[current_grid_pos[1]][current_grid_pos[0]] = True
    #     parent = {current_grid_pos: None}

    #     # Go through each position until there is an unknown space (frontier)
        
    #     # Could try and add something that checks walls etc. however maybe not because that is not the algorithm
    #     while len(queue) != 0:
    #         cc, cr = queue.popleft()

    #         if self.scanned_grid.grid[cr, cc] == 0:
    #             dest_location = (cc, cr)
    #             break

    #         for dir in directions:
    #             dr = self.directions[dir][1]
    #             dc = self.directions[dir][0]
    #             grid_val = self.scanned_grid.grid[cr + dr, cc + dc]

    #             if 0 <= cc + dc < self.scanned_grid.width and 0 <= cr + dr < self.scanned_grid.height and not visited[cr + dr][cc + dc] and grid_val != 1:
    #                 visited[cr + dr][cc + dc] = True
    #                 queue.append((cc + dc, cr + dr))
    #                 parent[(cc + dc, cr + dr)] = (cc, cr)
        
    #     # Change this to do all steps towards frontier instead of just one to reduce calculation
    #     if len(dest_location) != 0:
    #         # Find the next step the UAV should take to get to the free space selected
    #         next_step = dest_location
    #         while dest_location != current_grid_pos:
    #             next_step = dest_location
    #             dest_location = parent[dest_location]

    #         step_dir = [next_step[0] - dest_location[0], next_step[1] - dest_location[1]]
    #         self.step(step_dir, area, robot_start_id)
    #     else:
    #         self.moved = False