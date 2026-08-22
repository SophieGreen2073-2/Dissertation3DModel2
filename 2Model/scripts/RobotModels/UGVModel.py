from collections import deque
import numpy as np
from RobotModels.OccupancyBeliefGrid import UGVOccupancyGrid
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

class UGVModel():
    def __init__(self, robot_handle, left_motor, right_motor, sim, speed_params,
                 sensor_params, battery_params, wall_danger_zone, grid_height,
                 grid_width, resolution, area_model, robot_id):

        # Simulation
        self.sim = sim

        # Robot Status
        self.completed = False
        self.steps_completed = True
        self.is_returning_home = False
        self.is_scanning_in_place = False
        self.moved = False
        self.is_locked_dir = False
        self.robot_id = robot_id

        # Robot handle names
        self.robot_handle = robot_handle
        self.left_motor = left_motor
        self.right_motor = right_motor

        # Robot velocities
        self.top_speed = speed_params["Top"]
        self.danger_speed = speed_params["Danger"]
        self.start_speed = speed_params["Start"]
        self.acceleration = speed_params["Acceleration"]
        self.steering_gain = 1

        # Area model
        self.area = area_model

        # Robot paths
        self.steps_queue = deque()
        self.prev_target = None
        self.locked_sign = 1
        self.failed_frontiers = set()
        self.previous_loc = None

        # Battery
        self.battery = Battery(battery_params)

        # Camera models
        self.sensors = Sensors(sensor_params, self.robot_handle, self.sim)
        self.max_vision_range = sensor_params["VisionSensor"]["MaxRange"]
        self.min_vision_range = sensor_params["VisionSensor"]["MinRange"]

        # Occupancy grid
        self.occupancy_grid = UGVOccupancyGrid(resolution, grid_width, grid_height)

        # Utility function
        self.util_cost_weight = 1
        self.util_penalty_weight = 300
        self.util_wall_weight = 10
        self.frontier_count = 20

        # Wall avoidance
        self.wall_danger_zone = wall_danger_zone

        # Local UAVs
        self.localUAVs = []

        # Robot directions
        self.directions = {'north': [0, -1], 'south': [0, 1], 'east': [1, 0], 'west': [-1, 0], 'stay': [0,0], 'north_east': [1, -1], 'south_east': [1, 1], 'south_west': [-1, 1], 'north_west': [-1, -1]}

        # Step count
        self.step_count = 0
        self.dt = self.sim.getSimulationTimeStep()
        self.path_taken = []
        self.paths_planned = []


    # Check if target should be moved to next target 
    def get_lookahead_target(self, curr_pos, lookahead_dist):
        while len(self.steps_queue) > 1:
            wp = self.steps_queue[0]
            wx, wy = self.occupancy_grid.grid_to_world(wp[0], wp[1])
            dist = math.hypot(wx - curr_pos[0], wy - curr_pos[1])

            # If we're closer than lookahead_dist to current waypoint, pop it and move to next
            if dist < lookahead_dist:
                self.steps_queue.popleft()
            else:
                break

        if len(self.steps_queue) == 0:
            return None, None

        # Return the target position of the active waypoint
        target_wp = self.steps_queue[0]
        if not hasattr(self, 'prev_target'):
            self.prev_target = None

        if not self.prev_target == (target_wp[0], target_wp[1]):
            self.prev_target = (target_wp[0], target_wp[1])

        return self.occupancy_grid.grid_to_world(target_wp[0], target_wp[1])
    

    def step_robot(self, area_model):
        curr_pos = self.sim.getObjectPosition(self.robot_handle, -1)
        curr_orient = self.sim.getObjectOrientation(self.robot_handle, -1)
        curr_grid_pos = self.occupancy_grid.world_to_grid(curr_pos[0], curr_pos[1])

        # Log the path taken by the robot
        if curr_grid_pos != self.previous_loc:
            self.path_taken.append(curr_grid_pos)
            self.previous_loc = curr_grid_pos

        wall_belief = self.occupancy_grid.get_probability_grid()

        # If the robot is at the charge point the increase charge time
        if curr_grid_pos == tuple(self.battery.recharge_point) and self.is_returning_home:
            self.battery.charge_time_elapsed += self.dt
            if self.battery.charge_time_elapsed >= self.battery.charge_time:
                self.battery.mission_time = 0
                self.steps_queue.clear()
                self.steps_completed = True
                self.is_returning_home = False
            return

        # Increment mission time
        self.battery.mission_time += self.dt

        # Check that the robot has enough battery life to continue
        if not self.is_returning_home:
            path = self.do_a_star(curr_grid_pos, tuple(self.battery.recharge_point), False)
            if path == None: return
            is_battery_remaining = self.battery.check_battery_remaining(self.danger_speed, len(path) * self.area.resolution)
            if not is_battery_remaining:
                self.steps_queue.clear()
                self.steps_queue = deque(path)
                self.steps_completed = False
                self.is_returning_home = True

        # If steps queue empty then create new path
        if not self.steps_queue:
            self.steps_completed = True
            return

        # Check if the final point is still a frontier point
        final_wp = self.steps_queue[-1]
        if not self.check_frontier(self.directions, final_wp[0], final_wp[1]):
            self.steps_completed = True
            self.steps_queue.clear()
            return
        
        target_wx, target_wy = self.get_lookahead_target(curr_pos, lookahead_dist=1)

        # Compute distance to target        
        dx = target_wx - curr_pos[0]
        dy = target_wy - curr_pos[1]
        distance = math.hypot(dx, dy)

        # Check if the robot is close enough to target to move to next target
        if distance < 1:
            if len(self.steps_queue) == 1:
                final_wp = self.steps_queue[0]
                if self.check_frontier(self.directions, final_wp[0], final_wp[1]):
                    self.failed_frontiers.add(final_wp)
            self.steps_completed = True
            self.steps_queue.clear()
            return

        desired_angle = math.atan2(dy, dx)
        
        heading_error = math.atan2(math.sin(desired_angle - curr_orient[2]), 
                                math.cos(desired_angle - curr_orient[2]))
        
        if abs(heading_error) > 0.35:  # Roughly 20 degrees threshold
            # Pure rotation in place
            turn_speed = max(-self.top_speed, min(self.top_speed, heading_error * 1.5))
            left_speed = -turn_speed
            right_speed = turn_speed
        else:
            # Move forward while making minor steering corrections
            steering_gain = 1.5
            steer_correction = heading_error * steering_gain
            
            left_speed = self.top_speed + steer_correction
            right_speed = self.top_speed - steer_correction
            
            # Proportionally scale down if either wheel exceeds top_speed
            max_mag = max(abs(left_speed), abs(right_speed))
            if max_mag > self.top_speed:
                left_speed = (left_speed / max_mag) * self.top_speed
                right_speed = (right_speed / max_mag) * self.top_speed

        self.sim.setJointTargetVelocity(self.left_motor, left_speed)
        self.sim.setJointTargetVelocity(self.right_motor, right_speed)

        # Scan and update belief
        if self.step_count % self.sensors.scan_frequency == 0:
            self.sensors.get_points(self.sim, self.robot_handle)
            self.occupancy_grid.update_belief(self.sim, self.robot_handle,
                                            self.max_vision_range, 
                                            self.sensors.forward_lidars,
                                            self.sensors.lidar_spin.wall_points,
                                            curr_grid_pos, curr_orient, area_model, self.robot_id)

        # Increase number of steps
        self.step_count += 1


    # Basic yamauchi move (move to the closest free square, no search for frontiers)
    def yamauchi_move(self):
        dest_location = []

        current_grid_pos = self.get_grid_pos()

        directions = ['north', 'south', 'east', 'west', 'north_west', 'north_east', 'south_west', 'south_east']
        queue = deque([current_grid_pos])
        visited = [[False for _ in range(self.occupancy_grid.width)] for _ in range(self.occupancy_grid.height)]
        visited[current_grid_pos[1]][current_grid_pos[0]] = True
        parent = {current_grid_pos: None}
        wall_belief = self.occupancy_grid.get_probability_grid()

        # Go through each position until there is an unknown space (frontier)
        while len(queue) != 0:
            cc, cr = queue.popleft()
            grid_val = wall_belief[cr, cc]

            # Get current drone and target position
            current_world_pos = self.sim.getObjectPosition(self.robot_handle, -1)
            target_world_position = self.occupancy_grid.grid_to_world(cc, cr)

            # Check direction and distance to target
            step_dir = (target_world_position[0] - current_world_pos[0], target_world_position[1] - current_world_pos[1])
            dist_to_target = math.hypot(step_dir[0], step_dir[1])

            # Check if the grid value is unscanned, not the current position, and within the scan area and not the same as the previous target
            if 0.4 < grid_val < 0.6 and (cc, cr) != current_grid_pos and dist_to_target > self.min_vision_range:# and self.prev_end_target != (cc, cr):
                dest_location = (cc, cr)
                self.prev_end_target = dest_location
                break

            # Check if grid value is current position or free space
            if grid_val < 0.6 or (cc, cr) == current_grid_pos:
                # Check each direction
                for dir in directions:
                    dr = self.directions[dir][1]
                    dc = self.directions[dir][0]

                    # Check if location within grid bounds
                    if 0 <= cc + dc < self.occupancy_grid.width and 0 <= cr + dr < self.occupancy_grid.height:
                        # Check if location already checked
                        if not visited[cr + dr][cc + dc]:
                            neighbour_val = wall_belief[cr + dr, cc + dc]

                            # Corner-cutting guard for diagonal steps
                            if dc != 0 and dr != 0:
                                if wall_belief[cr, cc + dc] >= 0.6 or wall_belief[cr + dr, cc] >= 0.6:
                                    continue

                            # Check if neighbour is currently unscanned/free position
                            if neighbour_val < 0.6:
                                visited[cr + dr][cc + dc] = True
                                queue.append((cc + dc, cr + dr))
                                parent[(cc + dc, cr + dr)] = (cc, cr)
                                    
        # Change this to do all steps towards frontier instead of just one to reduce calculation
        if dest_location:
            # Find the next step the UAV should take to get to the free space selected
            next_step = dest_location
            path_nodes = []
            while next_step is not None:
                path_nodes.append(next_step)
                next_step = parent[next_step]

            # Create path to follow
            path_nodes.reverse()
            pruned = self.prune_path(path_nodes)
            self.steps_queue = pruned if pruned else path_nodes
            self.moved = True

            # If target within scan zone then scan in place
            if dist_to_target < self.max_vision_range:
                self.is_scanning_in_place = self.occupancy_grid.is_line_of_sight(current_grid_pos[0], current_grid_pos[1], self.steps_queue[-1][0], self.steps_queue[-1][1])
            else:
                self.is_scanning_in_place = False
        else:
            self.moved = False


    # Build full length of frontier
    def build_frontier(self, queue_frontier, MapCloseList, FrontierCloseList, directions, NewFrontier, FrontierOpenList, current_grid_pos, wall_belief):
        # While there are frontier points that have not been checked
        while len(queue_frontier) != 0:
            # Pick unchecked frontier point
            fc, fr = queue_frontier.popleft()
            # If q has not been checked
            if (fc, fr) in MapCloseList or (fc, fr) in FrontierCloseList:
                continue
            
            # Check if the point in the queue is a frontier point
            frontier_point = self.check_frontier(directions, fc, fr)
            distance = self.heuristic_function(current_grid_pos, (fc, fr))
            if distance > self.max_vision_range and len(NewFrontier) != 0:
                frontier_point = False

            # If point in frontier check list is a frontier 
            if frontier_point:
                NewFrontier.append((fc, fr))
                
                # Check all adjacent points to the frontier
                for dir in directions:
                    dr = self.directions[dir][1]
                    dc = self.directions[dir][0]
                    w = (fc + dc, fr + dr)
                    
                    # If w is not checked then add it to the queue
                    if 0 <= w[0] < self.occupancy_grid.width and 0 <= w[1] < self.occupancy_grid.height:
                        if w not in FrontierOpenList and w not in FrontierCloseList and w not in MapCloseList:
                            queue_frontier.append(w)
                            FrontierOpenList.add(w)
            FrontierCloseList.add((fc, fr))

        # Close the current point
        new_frontier_away_from_walls = []
        for p in NewFrontier:
            # Check if the point in the frontier is too close to a wall
            close_to_wall = self.check_close_wall(p, wall_belief)

            # Move the UAV away from the wall if within 3 squares
            if not close_to_wall:
                new_frontier_away_from_walls.append(p)
            MapCloseList.add(p)

        # If there are no points in the frontier away from the wall use the normal frontier
        if len(new_frontier_away_from_walls) == 0:
            new_frontier_away_from_walls = NewFrontier

        return MapCloseList, FrontierCloseList, FrontierOpenList, NewFrontier, new_frontier_away_from_walls


    # Works out cost of destination chosen based on how close to wall and other UAVs 
    def utility_function(self, p, directions, current_grid_pos):
        # Current drones distance to the frontier point
        current_to_p = self.heuristic_function(current_grid_pos, p)
        if current_to_p == 0:
            current_to_p = 0.1

        cost = current_to_p * self.util_cost_weight

        # Distance of frontier point
        walls_to_p = float('inf')

        # Converts belief grid to belief that there is a wall
        wall_belief = self.occupancy_grid.get_probability_grid()

        # Check each direction for a wall to penalise wall distance
        for dir in directions:
            dir_val = self.directions[dir]
            for i in range(self.wall_danger_zone):
                # Get position we are checking for wall
                scaled_dir_val = tuple(item * (i+1) for item in dir_val)
                curr_x = p[0] + scaled_dir_val[0]
                curr_y = p[1] + scaled_dir_val[1]

                # Chceck if selected position is within grid bounds
                if curr_x < 0 or curr_x >= self.occupancy_grid.width or curr_y < 0 or curr_y >= self.occupancy_grid.height:
                    break

                # Check grid position
                grid_val = wall_belief[curr_y, curr_x]
                if grid_val > 0.6:
                    if walls_to_p == float('inf'):
                        walls_to_p = 0
                    walls_to_p += self.heuristic_function((curr_x, curr_y), p)
                    break
        
        if walls_to_p == float('inf') or walls_to_p == 0:
            wall_penalty = 0
        else:
            # Closer to wall = larger penalty
            wall_penalty = self.util_wall_weight / (walls_to_p ** 2)

        uav_penalty = 0
        # Calculate distance to frontier point from each other UAV
        for uav in self.localUAVs:
            uav_to_p = self.heuristic_function((uav.x_pos, uav.y_pos), p)
            if uav_to_p == 0:
                uav_to_p = 0.1
                
            # If another UAV is closer to this target than we are, heavily penalize it
            if uav_to_p < current_to_p:
                # Scale penalty exponentially when close to minimize shared frontiers
                uav_penalty += self.util_penalty_weight / (uav_to_p ** 2)
            else:
                uav_penalty += (self.util_penalty_weight * 0.5) / (uav_to_p ** 2)

        return -cost - wall_penalty - uav_penalty


    # Yamauchi frontier algorithm that uses a utility function to choose the target point
    def yamauchi_move_utility_function(self, area_model):
        current_world_pos = self.sim.getObjectPosition(self.robot_handle, -1)
        current_grid_pos = self.occupancy_grid.world_to_grid(current_world_pos[0], current_world_pos[1])
        curr_orient = self.sim.getObjectOrientation(self.robot_handle, -1)

        dest_location = tuple()
        frontiers_found = []
        directions = ['north', 'south', 'east', 'west', 'north_east', 'south_east', 'south_west', 'north_west']
        queue = deque([current_grid_pos])

        MapOpenList = {current_grid_pos}
        MapCloseList = set()

        # if wall_belief[current_grid_pos[1], current_grid_pos[0]] == 0.5:
        self.sensors.get_points(self.sim, self.robot_handle)
        self.occupancy_grid.update_belief(self.sim, self.robot_handle,
                                            self.max_vision_range, 
                                            self.sensors.forward_lidars,
                                            self.sensors.lidar_spin.wall_points,
                                            current_grid_pos, curr_orient, area_model, 
                                            self.robot_id)

        wall_belief = self.occupancy_grid.get_probability_grid()

        inflated_wall_belief = self.inflate_walls(wall_belief)

        if inflated_wall_belief[current_grid_pos[1], current_grid_pos[0]] == 0.5:
            return

        # if current_grid_pos == (55, 50):
            # print("test")

        # Go through each position until frontier found
        while len(queue) != 0 and len(frontiers_found) <= self.frontier_count:
            cc, cr = queue.popleft()
            # self.prev_end_target = (cc, cr)

            # If p has not been visited, or likely a wall
            if (cc, cr) in MapCloseList or inflated_wall_belief[cr, cc] >= 0.4:
                continue

            # Check if the frontier is in the failed list
            if (cc, cr) in self.failed_frontiers:
                continue
            
            if (cc, cr) != current_grid_pos:
                # Get current position and target position
                target_world_position = self.occupancy_grid.grid_to_world(cc, cr)
                
                # Check direction and distance to target
                step_dir = (target_world_position[0] - current_world_pos[0], target_world_position[1] - current_world_pos[1])
                dist_to_target = math.hypot(step_dir[0], step_dir[1])

                if dist_to_target > self.min_vision_range:
                    if inflated_wall_belief[current_grid_pos[1], current_grid_pos[0]] == 0.5:
                        frontiers_found.append((cc, cr))
                        break

                    # If p is a frontier point
                    is_frontier = self.check_frontier(directions, cc, cr)

                    # If the point is a frontier point add to the list of frontiers
                    if is_frontier:
                        frontiers_found.append((cc, cr))

            # Add adjacent points to the check queue
            for dir in directions:
                dr = self.directions[dir][1]
                dc = self.directions[dir][0]
                adj_point = (cc + dc, cr + dr)

                # Check if each adjacent point has not been checked and is within bounds and is not a wall
                if 0<= adj_point[0] < self.occupancy_grid.width and 0 <= adj_point[1] < self.occupancy_grid.height:
                    if adj_point not in MapOpenList and adj_point not in MapCloseList:
                        
                        if inflated_wall_belief[adj_point[1], adj_point[0]] < 0.5:
                            queue.append(adj_point)
                            MapOpenList.add(adj_point)
            
            MapCloseList.add((cc, cr))

        # Calculate the best position (highest utility value)
        best_cost_val = float('-inf')
        for p in frontiers_found:
            # Calculate utility cost of location
            util_val = self.utility_function(p, directions, current_grid_pos)

            # If value is better than current best value then choose destination
            if util_val > best_cost_val:
                best_cost_val = util_val
                dest_location = p

        # If no destination found then grid is completed
        if len(dest_location) == 0:
            self.completed = True
            return

        # Get current position and target position
        target_world_position = self.occupancy_grid.grid_to_world(dest_location[0], dest_location[1])
        
        # Check direction and distance to target
        step_dir = (target_world_position[0] - current_world_pos[0], target_world_position[1] - current_world_pos[1])
        dist_to_target = math.hypot(step_dir[0], step_dir[1])

        self.do_a_star(current_grid_pos, dest_location, True)

        if len(self.steps_queue) != 0:
            self.steps_completed = False


    # Frontier based search
    def yamauchi_move_create_full_frontier(self, area_model):
        robot_world_pos = self.sim.getObjectPosition(self.robot_handle, -1)
        current_grid_pos = self.occupancy_grid.world_to_grid(robot_world_pos[0], robot_world_pos[1])
        curr_orient = self.sim.getObjectOrientation(self.robot_handle, -1)
        
        dest_location = []

        directions = ['north', 'south', 'east', 'west', 'north_east', 'south_east', 'south_west', 'north_west']
        queue = deque([current_grid_pos])

        MapOpenList = {current_grid_pos}
        MapCloseList = set()
        FrontierOpenList = set()
        FrontierCloseList = set()

        wall_belief = self.occupancy_grid.get_probability_grid()

        # Check if the current position of the UAV is unscanned
        if wall_belief[current_grid_pos[1], current_grid_pos[0]] == 0.5:
            dest_location = current_grid_pos
            self.occupancy_grid.belief_grid[current_grid_pos[1], current_grid_pos[0]] = -5
            self.sensors.get_points(self.sim, self.robot_handle)
            self.occupancy_grid.update_belief(self.sim, self.robot_handle,
                                            self.max_vision_range, 
                                            self.sensors.forward_lidars,
                                            self.sensors.lidar_spin.wall_points,
                                            current_grid_pos, curr_orient, area_model, self.robot_id)
            
            return

        # Go through each position until there is an unknown space (frontier)
        while len(queue) != 0 and len(dest_location) == 0:
            cc, cr = queue.popleft()

            # If p has not been visited
            if (cc, cr) in MapCloseList or wall_belief[cr, cc] > 0.6:
                continue

            # If p is a frontier point
            is_frontier = self.check_frontier(directions, cc, cr)

            if is_frontier:
                # Add p to the frontier queue
                queue_frontier = deque([(cc, cr)])
                NewFrontier = []
                FrontierOpenList.add((cc, cr))

                MapCloseList, FrontierCloseList, FrontierOpenList, NewFrontier, new_frontier_away_from_walls = self.build_frontier(
                    queue_frontier, MapCloseList, FrontierCloseList, directions, NewFrontier, FrontierOpenList, current_grid_pos,
                    wall_belief)

                # Find centroid in New Frontier list
                total_x, total_y = 0, 0
                for val in new_frontier_away_from_walls:
                    total_x += val[0]
                    total_y += val[1]

                x_target = total_x // len(new_frontier_away_from_walls)
                y_target = total_y // len(new_frontier_away_from_walls)
                dest_location = (x_target, y_target)

                # If centroid is the current position then send to first discovered frontier point (should be closest one)
                if dest_location == current_grid_pos or wall_belief[y_target, x_target] > 0.6:
                    dest_location = next(
                        (p for p in new_frontier_away_from_walls if p != current_grid_pos), 
                        new_frontier_away_from_walls[0]  # Fallback value if every single point matches current_loc
                    )

                    if dest_location == current_grid_pos:
                        # Run a raw, unfiltered search to find the absolute closest open frontier cell
                        raw_frontier_backup = []
                        for r in range(self.occupancy_grid.height):
                            for c in range(self.occupancy_grid.width):
                                if self.check_frontier(directions, c, r):
                                    raw_frontier_backup.append((c, r))
                        
                        if raw_frontier_backup:
                            # Target the closest raw frontier cell, ignoring the wall safety padding
                            dest_location = min(
                                raw_frontier_backup,
                                key=lambda p: (p[0] - current_grid_pos[0])**2 + (p[1] - current_grid_pos[1])**2
                            )
                break

            # Add adjacent points to the check queue
            for dir in directions:
                dr = self.directions[dir][1]
                dc = self.directions[dir][0]
                adj_point = (cc + dc, cr + dr)

                # Check if each adjacent point has not been checked and is within bounds
                if 0<= adj_point[0] < self.occupancy_grid.width and 0 <= adj_point[1] < self.occupancy_grid.height:
                    if adj_point not in MapOpenList and adj_point not in MapCloseList:
                        
                        if wall_belief[adj_point[1], adj_point[0]] <= 0.6:
                            queue.append(adj_point)
                            MapOpenList.add(adj_point)
            
            MapCloseList.add((cc, cr))

        # Generate path to target
        self.do_a_star(current_grid_pos, dest_location, True)

        if len(self.steps_queue) != 0:
            # self.scanned_grid.grid[dest_location[1], dest_location[0]] = 2
            self.steps_completed = False

    
    # Eucliden distance
    def heuristic_function(self, current_pos, target_pos):
        return ((target_pos[0] - current_pos[0])**2 + (target_pos[1] - current_pos[1])**2)**(1/2)


    # Create final path list to return once target has been reached
    def generate_path(self, preceeding_nodes, current, is_find_destination):
        path = deque()
        
        while(current in preceeding_nodes):
            if is_find_destination:
                self.steps_queue.appendleft(current)
            else:
                path.appendleft(current)

            current = preceeding_nodes[current]

        # Log the paths that have been planned
        if is_find_destination:
            self.paths_planned.append(self.steps_queue)
        else:
            self.paths_planned.append(path)

        return path


    # Inflate walls by two squares in each direction to keep the robot away from it
    def inflate_walls(self, wall_belief):
        height, width = wall_belief.shape

        new_wall_belief = wall_belief.copy()
        inflation_radius = 1

        for r in range(height):
            for c in range(width):
                if wall_belief[r, c] > 0.6:
                    for dr in range(-inflation_radius, inflation_radius + 1):
                        for dc in range(-inflation_radius, inflation_radius + 1):
                            nr, nc = r + dr, c + dc

                            if 0 <= nr < height and 0 <= nc < width:
                                if new_wall_belief[nr, nc] < 0.6:
                                    new_wall_belief[nr, nc] = 0.85

        return new_wall_belief

    # A* algorithm
    def do_a_star(self, start, end, is_find_destination):
        # Get the size of the grid
        open_nodes = [start]    # Currently open nodes, starting with the start node
        preceeding_nodes = {}   # Dictionary of the nodes preceeding the one selected, preceeding_nodes[n] = node that came before n in current cheapest path

        g_score = [[float('inf') for _ in range(self.occupancy_grid.width)] for _ in range(self.occupancy_grid.height)]  # Currently known cheapest path from start to n, set to infinity for every position initially
        f_score = [[float('inf') for _ in range(self.occupancy_grid.width)] for _ in range(self.occupancy_grid.height)]  # f_score(n) = g_score(n) + heuristic_function(n), representing current best guess of how cheap a path could be from start to finish through n, infinity for each position initially

        # Initialise the start position g_score and f_score
        g_score[start[1]][start[0]] = 0                 
        f_score[start[1]][start[0]] = self.heuristic_function(start, end)

        directions = ['north', 'south', 'east', 'west', 'north_east', 'south_east', 'south_west', 'north_west']

        wall_belief = self.occupancy_grid.get_probability_grid()

        inflated_wall_belief = self.inflate_walls(wall_belief)

        while(open_nodes):
            # Get open node with current lowest f score
            current_node = min(open_nodes, key=lambda node: f_score[node[1]][node[0]])

            # Check if the target has been reached, if so generate path and return
            if (current_node == end):
                path = self.generate_path(preceeding_nodes, end, is_find_destination)
                if is_find_destination:
                    # self.steps_queue = self.prune_path(self.steps_queue)
                    self.steps_queue = deque(self.steps_queue)

                return self.prune_path(deque(path))
            
            # Remove current node from open nodes
            open_nodes.remove(current_node)

            # Iterate through each direction from the current node
            for dir in directions:
                dx, dy = self.directions[dir]
                # Get position of neighbour in that direction
                neighbour_node = (current_node[0] + dx, current_node[1] + dy)

                # If the robot is returning home then only return along already scanned paths
                if (neighbour_node[0] < 0 or neighbour_node[1] < 0 or 
                    neighbour_node[0] >= self.occupancy_grid.width or 
                    neighbour_node[1] >= self.occupancy_grid.height):
                    continue

                cell_belief = inflated_wall_belief[neighbour_node[1]][neighbour_node[0]]
                if cell_belief > 0.4:
                    continue

                move_cost = 1.414 if (dx != 0 and dy != 0) else 1.0

                # Penalise points close to a wall
                wall_pen = 0.3
                clearance_cells = max(1, int(round(wall_pen / self.occupancy_grid.resolution)))
                near_wall = False
                for ny in range(max(0, neighbour_node[1] - clearance_cells), 
                                  min(self.occupancy_grid.height, neighbour_node[1] + clearance_cells + 1)):
                    for nx in range(max(0, neighbour_node[0] - clearance_cells), 
                                      min(self.occupancy_grid.width, neighbour_node[0] + clearance_cells + 1)):
                        if inflated_wall_belief[ny][nx] > 0.4:
                            near_wall = True
                            break
                    if near_wall:
                        break

                if near_wall:
                    move_cost *= 15.0
                curr_g_score = g_score[current_node[1]][current_node[0]] + move_cost

                # If calculated g score is better than current g score for the selected neighbour, update position
                if (curr_g_score < g_score[neighbour_node[1]][neighbour_node[0]]):
                    # Set preceeding node to be current open node
                    preceeding_nodes[neighbour_node] = current_node

                    # Update g and f score for the selected neighbour
                    g_score[neighbour_node[1]][neighbour_node[0]] = curr_g_score
                    f_score[neighbour_node[1]][neighbour_node[0]] = curr_g_score + self.heuristic_function(neighbour_node, end)

                    # Add neighbour to open nodes if not already there
                    if(neighbour_node not in open_nodes):
                        open_nodes.append(neighbour_node)


    # Change path to be be end points of full direction movement
    def prune_path(self, path):
        if len(path) <= 2:
            return path

        pruned = [path[0]]
        curr_dx = path[1][0] - path[0][0]
        curr_dy = path[1][1] - path[0][1]

        for i in range(1, len(path) - 1):
            next_dx = path[i + 1][0] - path[i][0]
            next_dy = path[i + 1][1] - path[i][1]

            if next_dx != curr_dx or next_dy != curr_dy:
                pruned.append(path[i])
                curr_dx = next_dx
                curr_dy = next_dy

        pruned.append(path[-1])
        return pruned
    
     # Check if the selected position is a frontier point (at least one discovered neigbour)
    def check_frontier(self, directions, cc, cr):
        wall_belief = self.occupancy_grid.get_probability_grid()

        if wall_belief[cr, cc] >= 0.5:
            return False
        
        for dir in directions:
            dr = self.directions[dir][1]
            dc = self.directions[dir][0]
            check_c = cc + dc
            check_r = cr + dr
            if 0 <= check_c < self.occupancy_grid.width and 0 <= check_r < self.occupancy_grid.height:
                neighbour_val = wall_belief[check_r, check_c]
                if neighbour_val == 0.5:
                    if not self.check_corner((cc, cr), (check_c, check_r), wall_belief):
                        return True
        
        return False
    

    # Check if the frontiers lateral free space is behind a corner
    def check_corner(self, pos, free_pos, wall_belief):
        wall_x, wall_y = pos[0], pos[1]
        free_x, free_y = free_pos[0], free_pos[1]

        corner_a = wall_belief[free_y, wall_x]
        corner_b = wall_belief[wall_y, free_x]
        
        if corner_a > 0.6 and corner_b > 0.6:
            return True
            
        return False


    # Check if the robot is going close to a wall and should slow down
    def check_close_wall(self, p, wall_belief):
        directions = ['north', 'south', 'east', 'west', 'north_east', 'south_east', 'south_west', 'north_west']

        for dir in directions:
            dir_val = self.directions[dir]
            for i in range(self.wall_danger_zone):
                # Get position we are checking copelliasim jobsfor wall
                scaled_dir_val = tuple(item * (i+1) for item in dir_val)
                curr_x = p[0] + scaled_dir_val[0]
                curr_y = p[1] + scaled_dir_val[1]

                # Chceck if selected position is within grid bounds
                if curr_x < 0 or curr_x >= self.occupancy_grid.width or curr_y < 0 or curr_y >= self.occupancy_grid.height:
                    break

                # Check grid position
                grid_val = wall_belief[curr_y, curr_x]
                if grid_val > 0.6:
                    return True
        
        return False


class Battery():
    def __init__(self, battery_params):
        self.life = battery_params["Life"]
        self.charge_time = battery_params["ChargeTime"]
        self.recharge_point = battery_params["RechargePoint"]

        self.mission_time = 0
        self.charge_time_elapsed = 0

    def check_battery_remaining(self, speed, distance):
        steps_time = distance / speed
        if steps_time > self.life - self.mission_time - 60:
            return False
        return True


class Sensors():
    def __init__(self, sensor_params, robot_handle, sim):
        # Scan frequency
        self.scan_frequency = sensor_params["ScanFrequency"]

        # Forward lidar
        # self.lidar_wall_points = []
        self.num_sensors = 32
        self.forward_lidars = []
        for i in range(self.num_sensors):
            self.forward_lidars.append(ForwardLiDAR(sensor_params["LiDAR"], robot_handle, sim, i))

        self.lidar_spin = UGVPerception(robot_handle, sim)


    def get_points(self, sim, robot_handle):
        rclpy.spin_once(self.lidar_spin, timeout_sec=0.0)
        
        # Get the points directly ahead of the forward lidar (accurate sensing)
        for sensor in self.forward_lidars:
            sensor.get_lidar_point(sim)

        return


class ForwardLiDAR():
    def __init__(self, LiDAR_params, robot_handle, sim, index):
        # LiDAR range
        self.min_range = LiDAR_params["MinRange"]
        self.max_range = LiDAR_params["MaxRange"]

        # Camera handle
        self.cam_handle = sim.getObject(f'/PioneerP3DX/proximitySensor[{index}]')

        self.wall_points = None

        self.orient = sim.getObjectOrientation(self.cam_handle, robot_handle)
        self.pos = sim.getObjectPosition(self.cam_handle, robot_handle)

    def get_lidar_point(self, sim):
        res, dist, detected_point, obj_handle, normal_vector = sim.checkProximitySensor(self.cam_handle, sim.handle_all)

        if res > 0:
            sensor_matrix = sim.getObjectMatrix(self.cam_handle, -1)
            self.wall_points = sim.multiplyVector(sensor_matrix, detected_point)
            return

        self.wall_points = None

class UGVPerception(Node):
    def __init__(self, robot_handle, sim):
        super().__init__('ugv_perception_node')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.listener_callback,
            1
        )
        self.wall_points = np.empty((0, 2), dtype=np.float32)
        self.robot_handle = robot_handle
        self.sim = sim

        self.prev_pos = self.sim.getObjectPosition(self.robot_handle, -1)
        self.prev_orient = self.sim.getObjectOrientation(self.robot_handle, -1)

 
    def listener_callback(self, msg):
        ranges = np.array(msg.ranges, dtype=np.float32)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))
        
        min_valid_range = 0.8
        valid_mask = (ranges > min_valid_range) & (ranges < msg.range_max)
        
        if not np.any(valid_mask):
            self.wall_points = np.empty((0, 2), dtype=np.float32)
            return

        r_valid = ranges[valid_mask]
        a_valid = angles[valid_mask]
        
        lx = r_valid * np.cos(a_valid)
        ly = r_valid * np.sin(a_valid)
        local_points = np.stack((lx, ly), axis=-1)

        pos = self.sim.getObjectPosition(self.robot_handle, -1)
        orient = self.sim.getObjectOrientation(self.robot_handle, -1)
        yaw = orient[2]

        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        
        global_points = np.dot(local_points, R.T) + np.array([pos[0], pos[1]], dtype=np.float32)
        self.wall_points = global_points

        # if len(self.wall_points):
            # print(f"Robot Pos: {pos[0]:.2f}, {pos[1]:.2f} | First Wall Point Global: {self.wall_points[0]}")


