import math
import numpy as np

class OccupancyBeliefGrid():
    def __init__(self, resolution, grid_width, grid_height):
        # Grid belief
        self.x_min, self.x_max = -grid_width/2, grid_width/2
        self.y_min, self.y_max = -grid_height/2, grid_height/2
        self.resolution = resolution

        self.width = int((self.x_max - self.x_min)/ resolution)
        self.height = int((self.y_max - self.y_min) / resolution)

        self.belief_grid = np.zeros((self.height, self.width), dtype=np.float32)

        self.l_occ = 0.85   # Increase belief when wall is detected by forward lidar
        self.l_free = -0.4  # Decrease belief along free space
        self.l_max =  5     # Max possible belief
        self.l_min = -5     # Min possible belief


    # Convert world coordinates into grid belief position
    def world_to_grid(self, wx, wy):
        gx = np.floor((wx - self.x_min) / self.resolution).astype(int)
        gy = np.floor((self.y_max - wy)/ self.resolution).astype(int)
        return gx, gy

    # Convert grid coordinates into world coordinates
    def grid_to_world(self, gx, gy):
        wx = self.x_min + (gx + 0.5) * self.resolution
        wy = self.y_max - (gy + 0.5) * self.resolution
        return wx, wy

    # Update the belief grid based on the scanned wall_points
    def get_wall_points(self, wall_points):
        if len(wall_points) == 0:
            return set()

        # Convert wall to grid points
        wall_cols, wall_rows = self.world_to_grid(wall_points[:, 0], wall_points[:, 1])

        # Check that the wall is within the bounds of the grid
        valid_mask = (wall_cols >= 0) & (wall_cols < self.width) & \
                     (wall_rows >= 0) & (wall_rows < self.height)

        valid_cols = wall_cols[valid_mask]
        valid_rows = wall_rows[valid_mask]

        return set(zip(valid_rows, valid_cols))


    def update_belief(self, sim, drone_handle, fov_deg, max_range, wall_points):
        uav_pos = sim.getObjectPosition(drone_handle, -1)
        uav_orient = sim.getObjectOrientation(drone_handle, -1)
        uav_x, uav_y, uav_yaw = uav_pos[0], uav_pos[1], uav_orient[2]
        
        uav_col, uav_row = self.world_to_grid(uav_x, uav_y)
        self.belief_grid[uav_row, uav_col] = -5 
        
        # Define FOV half-angle
        half_fov = math.radians(fov_deg / 2.0)
        angle_start = uav_yaw - half_fov
        angle_end = uav_yaw + half_fov
        curr_angle = angle_start

        valid_walls = self.get_wall_points(wall_points, sim, drone_handle)

        angular_resolution = math.radians(1.0)
        step_size = self.resolution * 0.5

        prob_grid = self.get_probability_grid()

        # Loop through a bounding box around the drone corresponding to max_range
        while curr_angle <= angle_end:
            dist = self.resolution
            while dist <= max_range:
                # Convert grid cell back to world coordinates
                wx = uav_x + dist * math.cos(curr_angle)
                wy = uav_y + dist * math.sin(curr_angle)
                c, r = self.world_to_grid(wx, wy)
                
                # Distance check
                if not (0 <= c < self.width and 0 <= r < self.height):
                    break

                # CHeck if location is a wall, if yes move to next ray
                if (r, c) in valid_walls or prob_grid[r, c] > 0.5:
                    self.belief_grid[r, c] += self.l_occ
                    break
                else:
                    if self.belief_grid[r, c] <= 0:
                        self.belief_grid[r, c] += self.l_free

                dist += step_size

            curr_angle += angular_resolution

        # Ensure belief stays within bounds [l_min, l_max]
        np.clip(self.belief_grid, self.l_min, self.l_max, out=self.belief_grid)


    # Check if target has wall in the way
    # Might be able to merge this with free space update
    def is_line_of_sight(self, x_start, y_start, x_end, y_end):
        dx = abs(x_end - x_start)
        dy = abs(y_end - y_start)
        dir_x = 1 if x_start < x_end else -1
        dir_y = 1 if y_start < y_end else -1
        err = dx - dy

        curr_x, curr_y = x_start, y_start
        wall_belief = self.get_probability_grid()

        # max_steps = dx + dy + 2
        # steps = 0

        while (curr_x != x_end or curr_y != y_end): #and steps < max_steps:
            # steps += 1
            if 0 <= curr_x < self.width and 0 <= curr_y < self.height:
                if wall_belief[curr_y, curr_x] > 0.6:
                    return False

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                curr_x += dir_x
            if e2 < dx:
                err += dx
                curr_y += dir_y

        return True

    # Get grid of probabilities that the space has a wall in it
    def get_probability_grid(self):
        return 1.0 - (1.0 / (1.0 + np.exp(self.belief_grid)))


class UGVOccupancyGrid(OccupancyBeliefGrid):
    def __init__(self, resolution, grid_width, grid_height):
        OccupancyBeliefGrid.__init__(self, resolution, grid_width, grid_height)

        self.l_occ_lidar = 1.2   # Increase belief when wall is detected by forward lidar
        self.l_occ_vision = 0.7   # Increase belief when wall is detected by 360 degree vision cams 

    def update_belief(self, sim, robot_handle, vision_fov_deg, 
                      vision_max_range, forward_lidar_wall_points, 
                      vision_cam_wall_points, robot_pos, robot_orient):
        robot_x, robot_y, robot_yaw = robot_pos[0], robot_pos[1], robot_orient[2]
        
        # robot_col, robot_row = self.world_to_grid(robot_x, robot_y)
        self.belief_grid[robot_y, robot_x] = -5 

        prob_grid = self.get_probability_grid()

        self.update_vision_cam_belief(vision_fov_deg, vision_max_range, robot_x,
                                      robot_y, robot_yaw, sim, robot_handle, vision_cam_wall_points,
                                      prob_grid)

        self.update_lidar_belief(sim, robot_handle, forward_lidar_wall_points)

        # Ensure belief stays within bounds [l_min, l_max]
        np.clip(self.belief_grid, self.l_min, self.l_max, out=self.belief_grid)

    def update_lidar_belief(self, sim, robot_handle, wall_points):
        if len(wall_points) > 0:
            valid_wall = self.world_to_grid(wall_points[0], wall_points[1])
            self.belief_grid[valid_wall[1], valid_wall[0]] += self.l_occ_lidar


    # def update_vision_cam_belief(self, fov_deg, max_range, gx, gy, yaw, sim, robot_handle, wall_points, prob_grid):
    #     # Define FOV half-angle
    #     fov_deg = 360
    #     half_fov = math.radians(fov_deg / 2.0)
    #     angle_start = yaw - half_fov
    #     angle_end = yaw + half_fov
    #     curr_angle = angle_start

    #     x, y = self.grid_to_world(gx, gy)

    #     valid_walls = self.get_wall_points(wall_points)

    #     angular_resolution = math.radians(1.0)
    #     step_size = self.resolution * 0.5

    #     # Loop through a bounding box around the drone corresponding to max_range
    #     while curr_angle <= angle_end:
    #         dist = step_size
    #         while dist <= max_range:
    #             # Convert grid cell back to world coordinates
    #             wx = x + dist * math.cos(curr_angle)
    #             wy = y + dist * math.sin(curr_angle)
    #             c, r = self.world_to_grid(wx, wy)
                
    #             # Distance check
    #             if not (0 <= c < self.width and 0 <= r < self.height):
    #                 break

    #             # CHeck if location is a wall, if yes move to next ray
    #             if (r, c) in valid_walls or prob_grid[r, c] > 0.5:
    #                 self.belief_grid[r, c] += self.l_occ_vision
    #                 break
    #             else:
    #                 if self.belief_grid[r, c] <= 0:
    #                     self.belief_grid[r, c] += self.l_free

    #             dist += step_size

    #         curr_angle += angular_resolution

    import math
import numpy as np

class OccupancyBeliefGrid():
    def __init__(self, resolution, grid_width, grid_height):
        # Grid belief
        self.x_min, self.x_max = -grid_width/2, grid_width/2
        self.y_min, self.y_max = -grid_height/2, grid_height/2
        self.resolution = resolution

        self.width = int((self.x_max - self.x_min)/ resolution)
        self.height = int((self.y_max - self.y_min) / resolution)

        self.belief_grid = np.zeros((self.height, self.width), dtype=np.float32)

        self.l_occ = 0.85   # Increase belief when wall is detected by forward lidar
        self.l_free = -0.4  # Decrease belief along free space
        self.l_max =  5     # Max possible belief
        self.l_min = -5     # Min possible belief


    # Convert world coordinates into grid belief position
    def world_to_grid(self, wx, wy):
        gx = np.floor((wx - self.x_min) / self.resolution).astype(int)
        gy = np.floor((self.y_max - wy)/ self.resolution).astype(int)
        return gx, gy

    # Convert grid coordinates into world coordinates
    def grid_to_world(self, gx, gy):
        wx = self.x_min + (gx + 0.5) * self.resolution
        wy = self.y_max - (gy + 0.5) * self.resolution
        return wx, wy

    # Update the belief grid based on the scanned wall_points
    def get_wall_points(self, wall_points):
        if len(wall_points) == 0:
            return set()

        # Convert wall to grid points
        wall_cols, wall_rows = self.world_to_grid(wall_points[:, 0], wall_points[:, 1])

        # Check that the wall is within the bounds of the grid
        valid_mask = (wall_cols >= 0) & (wall_cols < self.width) & \
                     (wall_rows >= 0) & (wall_rows < self.height)

        valid_cols = wall_cols[valid_mask]
        valid_rows = wall_rows[valid_mask]

        return set(zip(valid_rows, valid_cols))


    def update_belief(self, sim, drone_handle, fov_deg, max_range, wall_points):
        uav_pos = sim.getObjectPosition(drone_handle, -1)
        uav_orient = sim.getObjectOrientation(drone_handle, -1)
        uav_x, uav_y, uav_yaw = uav_pos[0], uav_pos[1], uav_orient[2]
        
        uav_col, uav_row = self.world_to_grid(uav_x, uav_y)
        self.belief_grid[uav_row, uav_col] = -5 
        
        # Define FOV half-angle
        half_fov = math.radians(fov_deg / 2.0)
        angle_start = uav_yaw - half_fov
        angle_end = uav_yaw + half_fov
        curr_angle = angle_start

        valid_walls = self.get_wall_points(wall_points, sim, drone_handle)

        angular_resolution = math.radians(1.0)
        step_size = self.resolution * 0.5

        prob_grid = self.get_probability_grid()

        # Loop through a bounding box around the drone corresponding to max_range
        while curr_angle <= angle_end:
            dist = self.resolution
            while dist <= max_range:
                # Convert grid cell back to world coordinates
                wx = uav_x + dist * math.cos(curr_angle)
                wy = uav_y + dist * math.sin(curr_angle)
                c, r = self.world_to_grid(wx, wy)
                
                # Distance check
                if not (0 <= c < self.width and 0 <= r < self.height):
                    break

                # CHeck if location is a wall, if yes move to next ray
                if (r, c) in valid_walls or prob_grid[r, c] > 0.5:
                    self.belief_grid[r, c] += self.l_occ
                    break
                else:
                    if self.belief_grid[r, c] <= 0:
                        self.belief_grid[r, c] += self.l_free

                dist += step_size

            curr_angle += angular_resolution

        # Ensure belief stays within bounds [l_min, l_max]
        np.clip(self.belief_grid, self.l_min, self.l_max, out=self.belief_grid)


    # Check if target has wall in the way
    # Might be able to merge this with free space update
    def is_line_of_sight(self, x_start, y_start, x_end, y_end):
        dx = abs(x_end - x_start)
        dy = abs(y_end - y_start)
        dir_x = 1 if x_start < x_end else -1
        dir_y = 1 if y_start < y_end else -1
        err = dx - dy

        curr_x, curr_y = x_start, y_start
        wall_belief = self.get_probability_grid()

        # max_steps = dx + dy + 2
        # steps = 0

        while (curr_x != x_end or curr_y != y_end): #and steps < max_steps:
            # steps += 1
            if 0 <= curr_x < self.width and 0 <= curr_y < self.height:
                if wall_belief[curr_y, curr_x] > 0.6:
                    return False

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                curr_x += dir_x
            if e2 < dx:
                err += dx
                curr_y += dir_y

        return True

    # Get grid of probabilities that the space has a wall in it
    def get_probability_grid(self):
        return 1.0 - (1.0 / (1.0 + np.exp(self.belief_grid)))


class UGVOccupancyGrid(OccupancyBeliefGrid):
    def __init__(self, resolution, grid_width, grid_height):
        OccupancyBeliefGrid.__init__(self, resolution, grid_width, grid_height)

        self.l_occ_lidar = 1.2   # Increase belief when wall is detected by forward lidar
        self.l_occ_vision = 0.7   # Increase belief when wall is detected by 360 degree vision cams 

    def update_belief(self, sim, robot_handle, 
                      vision_max_range, forward_lidar_wall_points, 
                      vision_cam_wall_points, robot_pos, robot_orient, area_model):
        robot_x, robot_y, robot_yaw = robot_pos[0], robot_pos[1], robot_orient[2]
        
        # robot_col, robot_row = self.world_to_grid(robot_x, robot_y)
        prob_grid = self.get_probability_grid()

        self.update_vision_cam_belief(vision_max_range, robot_x,
                                      robot_y, robot_yaw, sim, robot_handle, vision_cam_wall_points,
                                      prob_grid, area_model)

        self.update_lidar_belief(sim, robot_handle, forward_lidar_wall_points, area_model)

        self.belief_grid[robot_y, robot_x] = -5 

        # Ensure belief stays within bounds [l_min, l_max]
        np.clip(self.belief_grid, self.l_min, self.l_max, out=self.belief_grid)

    def line_of_sight(self, x0, y0, x1, y1):
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return points


    def update_lidar_belief(self, sim, robot_handle, wall_points, area_model, robot_id):
        if len(wall_points) == 0:
            return

        pos = sim.getObjectPosition(robot_handle, -1)
        robot_grid = self.world_to_grid(pos[0], pos[1])
        
        valid_wall = self.world_to_grid(wall_points[0], wall_points[1])
        points = self.line_of_sight(robot_grid[0], robot_grid[1], valid_wall[0], valid_wall[1])

        for p in points:
            self.belief_grid[p[1], p[0]] += self.l_free
            area_model.overlap_area[p[1], p[0], robot_id] += 1

        self.belief_grid[valid_wall[1], valid_wall[0]] += self.l_occ_lidar
        area_model.overlap_area[valid_wall[1], valid_wall[0], robot_id] += 1


    def update_vision_cam_belief(self, max_range, gx, gy, yaw, sim, robot_handle, wall_points, prob_grid, area_model, robot_id):
        # Define FOV half-angle
        fov_deg = 360
        half_fov = math.radians(fov_deg / 2.0)
        angle_start = yaw - half_fov
        angle_end = yaw + half_fov
        curr_angle = angle_start

        x, y = self.grid_to_world(gx, gy)

        valid_walls = self.get_wall_points(wall_points)

        angular_resolution = math.radians(1.0)
        step_size = self.resolution * 0.5

        # Loop through a bounding box around the drone corresponding to max_range
        while curr_angle <= angle_end:
            dist = step_size
            while dist <= max_range:
                # Convert grid cell back to world coordinates
                wx = x + dist * math.cos(curr_angle)
                wy = y + dist * math.sin(curr_angle)
                c, r = self.world_to_grid(wx, wy)
                
                # Distance check
                if not (0 <= c < self.width and 0 <= r < self.height):
                    break

                # CHeck if location is a wall, if yes move to next ray
                if (r, c) in valid_walls:
                    self.belief_grid[r, c] += self.l_occ_vision
                    area_model.overlap_area[r, c, robot_id] += 1
                    break
                else:
                    if self.belief_grid[r, c] <= 2.5:
                        self.belief_grid[r, c] += self.l_free
                        area_model.overlap_area[r, c, robot_id] += 1 

                dist += step_size

            curr_angle += angular_resolution

    # def update_vision_cam_belief(self, max_range, gx, gy, yaw, sim, robot_handle, wall_points, prob_grid):
    #     uav_x, uav_y = self.grid_to_world(gx, gy)
        
    #     # Define your proximity sensor angles relative to the robot's heading (e.g., 8 sensors spaced every 45 degrees)
    #     num_sensors = 8
    #     sensor_angles = [yaw + i * (2.0 * math.pi / num_sensors) for i in range(num_sensors)]

    #     valid_walls = self.get_wall_points(wall_points)
    #     step_size = self.resolution * 0.5

    #     # Cast a ray for each proximity sensor direction
    #     for angle in sensor_angles:
    #         ray_hit = False
    #         hit_cell = None
            
    #         # Step along this specific ray
    #         dist = step_size
    #         while dist <= max_range:
    #             wx = uav_x + dist * math.cos(angle)
    #             wy = uav_y + dist * math.sin(angle)
    #             c, r = self.world_to_grid(wx, wy)
                
    #             # Distance boundary check
    #             if not (0 <= c < self.width and 0 <= r < self.height):
    #                 break

    #             # Check if this cell contains a wall detected by our sensors or belief grid
    #             if (r, c) in valid_walls or prob_grid[r, c] > 0.5:
    #                 hit_cell = (r, c)
    #                 ray_hit = True
    #                 break
    #             else:
    #                 # Only mark free space along this active ray if not already a confirmed wall
    #                 if self.belief_grid[r, c] <= 0:
    #                     self.belief_grid[r, c] += self.l_free

    #             dist += step_size

    #         # If the ray hit a wall, update that specific endpoint
    #         if ray_hit and hit_cell:
    #             r, c = hit_cell
    #             self.belief_grid[r, c] += self.l_occ_vision