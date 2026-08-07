import numpy as np
import struct
from collections import deque
import math

class UAVModel():
    def __init__(self, x, y, z, top_speed, danger_speed, start_speed,
                 acceleration, battery_life, charge_time, robot_id,
                 alias, drone_base, drone_target, current_path,
                 grid_width, grid_height, sim, fov_deg, max_range,
                 scan_frequency, min_range, wall_danger_zone, resolution = 0.2):
        print("Create new UAV")

        self.sim = sim

        # Robot status
        self.completed = False

        # Robot Position
        # self.pos = (x, y)
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
        self.prev_end_target = None
        self.cam_handle = self.sim.getObject('/Quadcopter/visionSensor')

        # Robot paths
        self.steps_queue = deque()
        self.steps_completed = True
        self.is_returning_home = False
        self.current_path = current_path

        # Belief grid
        self.occupancy_grid = OccupancyBeliefGrid(resolution, grid_width, grid_height)
        
        # Camera model
        self.sensor = UAVSensor(fov_deg, max_range, self.cam_handle)
        self.scan_frequency = scan_frequency
        self.step_count = 0
        self.min_range = min_range
        self.max_range = max_range
        self.is_scanning_in_place = False

        # Utility function
        self.util_cost_weight = 1
        self.util_penalty_weight = 300
        self.util_wall_weight = 10
        self.frontier_count = 20

        # Wall avoidance
        self.wall_danger_zone = wall_danger_zone

        # Local UAVs
        self.localUAVs = []


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
        return self.occupancy_grid.grid_to_world(target_wp[0], target_wp[1])


    # Move robot into next position
    def step_robot(self):
        # Get grid position and orientation of drone
        curr_pos = self.sim.getObjectPosition(self.drone_base, -1)
        curr_orient = self.sim.getObjectOrientation(self.drone_base, -1)

        if not self.steps_queue:
            self.steps_completed = True
            return

        # Wall emergency exit
        next_wp = self.steps_queue[0]
        wall_belief = self.occupancy_grid.get_probability_grid()
        if 0 <= next_wp[0] < self.occupancy_grid.width and 0 <= next_wp[1] < self.occupancy_grid.height:
            if wall_belief[next_wp[1], next_wp[0]] > 0.6:
                self.steps_queue.clear()
                self.steps_completed = True
                return

        target_wx, target_wy = self.get_lookahead_target(curr_pos, lookahead_dist=0.6)
        if target_wx is None:
            self.steps_completed = True
            return

        # Calculate distance to final target for arrival braking
        final_wp = self.steps_queue[-1] if len(self.steps_queue) > 0 else None
        if final_wp:
            final_wx, final_wy = self.occupancy_grid.grid_to_world(final_wp[0], final_wp[1])
            dist_to_final = math.hypot(final_wx - curr_pos[0], final_wy - curr_pos[1])
        else:
            dist_to_final = float('inf')

        # Get change in orientation 
        step_dir = (target_wx - curr_pos[0], target_wy - curr_pos[1])
        dist_to_target = math.hypot(step_dir[0], step_dir[1])

        target_angle = math.atan2(step_dir[1], step_dir[0])
        angle_diff = math.atan2(math.sin(target_angle - curr_orient[2]), math.cos(target_angle - curr_orient[2]))

        # Alter angle if difference is a big enough angle
        if dist_to_target > 0.05:
            max_turn_rate = 0.005
            p_gain = 0.3

            if abs(angle_diff) < 0.8:  # Within ~17 degrees
                # Proportional slowdown (shrinks as angle approaches 0)
                dynamic_turn_rate = max_turn_rate * (abs(angle_diff) / 0.3)
            else:
                dynamic_turn_rate = max_turn_rate

            clamped_turn = max(-dynamic_turn_rate, min(dynamic_turn_rate, angle_diff * p_gain))
            commanded_yaw = curr_orient[2] + clamped_turn
 
            if self.is_scanning_in_place:
                self.sim.setObjectPosition(self.drone_target, -1, [curr_pos[0], curr_pos[1], 2.5])
                self.sim.setObjectOrientation(self.drone_target, -1, [0.0, 0.0, commanded_yaw])
            else:
                # Slow speed if angle not aligned with target
                alignment = max(0.03, math.cos(angle_diff))

                # Check if close to end of path
                arrival_braking = min(1.0, dist_to_final / 0.8)
                step_dist = min(dist_to_target, 0.25 * alignment * arrival_braking)
    
                step_x = curr_pos[0] + (step_dir[0] / dist_to_target) * step_dist
                step_y = curr_pos[1] + (step_dir[1] / dist_to_target) * step_dist

                # Spin in place until close to correct angle
                # if abs(angle_diff) >= 0.01:
                    # self.sim.setObjectPosition(self.drone_target, -1, [curr_pos[0], curr_pos[1], 2.5])
                    # self.sim.setObjectOrientation(self.drone_target, -1, [0.0, 0.0, commanded_yaw])
                # else:
                self.sim.setObjectPosition(self.drone_target, -1, [step_x, step_y, 2.5])
                self.sim.setObjectOrientation(self.drone_target, -1, [0.0, 0.0, commanded_yaw])

                           
        if self.is_scanning_in_place:
            if abs(angle_diff) < 0.01:
                self.steps_queue.clear()
                self.steps_completed = True
        else:
            final_wp = self.steps_queue[-1]
            final_wx, final_wy = self.occupancy_grid.grid_to_world(final_wp[0], final_wp[1])
            if math.hypot(final_wx - curr_pos[0], final_wy - curr_pos[1]) < 0.25:
                self.steps_queue.clear()
                self.steps_completed = True

        # if len(self.steps_queue) > 0:
        #     final_wp = self.steps_queue[-1]
        #     final_wx, final_wy = self.occupancy_grid.grid_to_world(final_wp[0], final_wp[1])
        #     if math.hypot(final_wx - curr_pos[0], final_wy - curr_pos[1]) < 0.25:
        #         self.steps_queue.clear()
        #         self.steps_completed = True

        if self.step_count % self.scan_frequency == 0:
            self.sensor.get_lidar_points(self.sim)
            self.occupancy_grid.update_belief(self.sim, self.drone_base, self.sensor.fov_deg, self.max_range, self.sensor.wall_points)

        # Increase number of steps
        self.step_count += 1

    def scan(self):
        self.sensor.get_lidar_points(self.sim)
        self.occupancy_grid.update_belief(self.sim, self.drone_base, self.sensor.fov_deg, self.max_range, self.sensor.wall_points)

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
            current_world_pos = self.sim.getObjectPosition(self.drone_base, -1)
            target_world_position = self.occupancy_grid.grid_to_world(cc, cr)

            # Check direction and distance to target
            step_dir = (target_world_position[0] - current_world_pos[0], target_world_position[1] - current_world_pos[1])
            dist_to_target = math.hypot(step_dir[0], step_dir[1])

            # Check if the grid value is unscanned, not the current position, and within the scan area and not the same as the previous target
            if 0.4 < grid_val < 0.6 and (cc, cr) != current_grid_pos and dist_to_target > self.min_range and self.prev_end_target != (cc, cr):
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
            if dist_to_target < self.max_range:
                self.is_scanning_in_place = self.occupancy_grid.is_line_of_sight(current_grid_pos[0], current_grid_pos[1], self.current_path[-1][0], self.current_path[-1][1])
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
            if distance > self.max_range and len(NewFrontier) != 0:
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
    def yamauchi_move_utility_function(self):
        current_grid_pos = self.get_grid_pos()
        dest_location = tuple()
        frontiers_found = []
        directions = ['north', 'south', 'east', 'west', 'north_east', 'south_east', 'south_west', 'north_west']
        queue = deque([current_grid_pos])

        MapOpenList = {current_grid_pos}
        MapCloseList = set()

        wall_belief = self.occupancy_grid.get_probability_grid()

        if wall_belief[current_grid_pos[1], current_grid_pos[0]] == 0.5:
            self.scan()
            return

        # Go through each position until frontier found
        while len(queue) != 0 and len(frontiers_found) <= self.frontier_count:
            cc, cr = queue.popleft()
            self.prev_end_target = (cc, cr)

            current_world_pos = self.sim.getObjectPosition(self.drone_base, -1)

            # If p has not been visited, or likely a wall
            if (cc, cr) in MapCloseList or wall_belief[cr, cc] >= 0.4:
                continue
            
            if (cc, cr) != current_grid_pos:
                # Get current position and target position
                target_world_position = self.occupancy_grid.grid_to_world(cc, cr)
                
                # Check direction and distance to target
                step_dir = (target_world_position[0] - current_world_pos[0], target_world_position[1] - current_world_pos[1])
                dist_to_target = math.hypot(step_dir[0], step_dir[1])

                if dist_to_target > self.min_range:
                    if wall_belief[current_grid_pos[1], current_grid_pos[0]] == 0.5:
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
                        
                        if wall_belief[adj_point[1], adj_point[0]] <= 0.4:
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
        current_world_pos = self.sim.getObjectPosition(self.drone_base, -1)
        target_world_position = self.occupancy_grid.grid_to_world(dest_location[0], dest_location[1])
        
        # Check direction and distance to target
        step_dir = (target_world_position[0] - current_world_pos[0], target_world_position[1] - current_world_pos[1])
        dist_to_target = math.hypot(step_dir[0], step_dir[1])

        is_line_of_sight = self.occupancy_grid.is_line_of_sight(current_grid_pos[0], current_grid_pos[1], dest_location[0], dest_location[1])

        # If the target is within the max scan range then scan in place
        if dist_to_target < self.max_range * 0.98 and is_line_of_sight:
            self.is_scanning_in_place = True
            self.steps_queue.append(dest_location)
        else:
            # Generate path to target
            self.do_a_star(current_grid_pos, dest_location, True)
            self.is_scanning_in_place = False

        if len(self.steps_queue) != 0:
            self.steps_completed = False


    # Frontier based search
    def yamauchi_move_create_full_frontier(self):
        current_grid_pos = self.get_grid_pos()
        
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
            self.scan()
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
                                key=lambda p: (p[0] - current_grid_pos[0])**2 + (p[1] - current_grid_pos[0])**2
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

        return path
    

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

        directions = ['north', 'south', 'east', 'west']

        wall_belief = self.occupancy_grid.get_probability_grid()

        while(open_nodes):
            # Get open node with current lowest f score
            current_node = min(open_nodes, key=lambda node: f_score[node[1]][node[0]])

            # Check if the target has been reached, if so generate path and return
            if (current_node == end):
                path = self.generate_path(preceeding_nodes, end, is_find_destination)
                return path
            
            # Remove current node from open nodes
            open_nodes.remove(current_node)

            # Iterate through each direction from the current node
            for dir in directions:
                dx, dy = self.directions[dir]
                # Get position of neighbour in that direction
                neighbour_node = (current_node[0] + dx, current_node[1] + dy)

                # If the robot is returning home then only return along already scanned paths
                if is_find_destination:
                    # Check that the neighbour is within the grid and not a wall
                    if (neighbour_node[0] < 0 or neighbour_node[1] < 0 or neighbour_node[0] >= self.occupancy_grid.width or neighbour_node[1] >= self.occupancy_grid.height or wall_belief[neighbour_node[1]][neighbour_node[0]] > 0.4):
                        continue
                else:
                    if (neighbour_node[0] < 0 or neighbour_node[1] < 0 or neighbour_node[0] >= self.occupancy_grid.width or neighbour_node[1] >= self.occupancy_grid.height or wall_belief[neighbour_node[1]][neighbour_node[0]] <= 0.6):
                        continue

                # Calculate current g score for neighbour from selected node
                curr_g_score = g_score[current_node[1]][current_node[0]] + 1

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


    # Check if the UAV needs to return to charge
    def check_battery_remaining(self, recharge_point):
        current_grid_pos = self.get_grid_pos()
        path = self.do_a_star(current_grid_pos, tuple(recharge_point), False)
        if path == None: return
        steps_time = len(path)
        if steps_time > self.battery_life - self.mission_time - 60:
            self.steps_queue.clear()
            self.steps_queue = path
            self.steps_completed = False
            self.is_returning_home = True


    # Check if the selected position is a frontier point (at least one discovered neigbour)
    def check_frontier(self, directions, cc, cr):
        wall_belief = self.occupancy_grid.get_probability_grid()

        if wall_belief[cr, cc] == 0.5:
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


    # Change path to be be end points of full direction movement
    def prune_path(self, path):
        if len(path) <= 2:
            return path

        pruned = [path[0]]
        for i in range(1, len(path) - 1):
            prev = path[i - 1]
            curr = path[i]
            next_node = path[i + 1]

            dir1 = (curr[0] - prev[0], curr[1] - prev[1])
            dir2 = (next_node[0] - curr[0], next_node[1] - curr[1])

            len1 = math.hypot(dir1[0], dir1[1])
            len2 = math.hypot(dir2[0], dir2[1])

            norm1 = (round(dir1[0] / len1, 2), round(dir1[1] / len1, 2)) if len1 > 0 else (0, 0)
            norm2 = (round(dir2[0] / len2, 2), round(dir2[1] / len2, 2)) if len2 > 0 else (0, 0)

            if norm1 != norm2:
                pruned.append(curr)

        pruned.append(path[-1])
        return pruned

        # # Work out the next robot step 
    # def robot_next_step(self, start_robot_ids, dt, area, time_step, recharge_point):
    #     # Get current grid position
    #     current_grid_pos = self.get_grid_pos()

    #     if current_grid_pos == tuple(recharge_point) and self.is_returning_home:
    #         self.charge_time_elapsed += time_step
    #         if self.charge_time_elapsed >= self.charge_time:
    #             self.mission_time = 0
    #             self.steps_queue.clear()
    #             self.target = None
    #             self.steps_completed = True
    #             self.is_returning_home = False
    #         return

        
    #     # Increment how long the robots mission has been
    #     self.mission_time += time_step

    #     if not self.is_returning_home:
    #         # Check if the robot need to head back to recharge
    #         self.check_battery_remaining(recharge_point)

    #     # Check if the robot should be moved
    #     if self.steps_completed:
    #         return

    #     # Get next step and start position
    #     start = (self.x_pos, self.y_pos)
    #     end = self.target if len(self.steps_queue) == 0 else self.steps_queue[len(self.steps_queue) - 1] 
    #     if not self.target:
    #         self.target = self.steps_queue.popleft()

    #     # Step robot into the next position
    #     step_dir = (self.target[0] - start[0], self.target[1] - start[1])
    #     distance = np.hypot(step_dir[0], step_dir[1])

    #     # If robot is close to target then move target to next step
    #     if distance < 0.01:
    #         if len(self.steps_queue) > 0:
    #             self.target = self.steps_queue.popleft()
    #             step_dir = (self.target[0] - start[0], self.target[1] - start[1])
    #             distance = np.hypot(step_dir[0], step_dir[1])
    #         else:
    #             self.steps_completed = True
    #             return

    #     # If the next step is into a wall then clear the steps queue and move to next robot
    #     if self.scanned_grid.grid[self.target[1], self.target[0]] == 1:
    #         self.steps_queue.clear()
    #         self.target = None
    #         self.steps_completed = True
    #         return

    #     # Check if the robot is entering a tight space so should be at a slower speed
    #     is_tight_space = self.check_close_wall()

    #     # If the robot is close to the wall then slow down
    #     target_speed = self.top_speed if not is_tight_space else self.danger_speed

    #     # Accelerate/decelerate
    #     acc = self.acceleration * dt
    #     speed_diff = target_speed - self.current_speed
    #     self.current_speed += np.clip(speed_diff, -acc, acc)

    #     step_distance = self.current_speed * dt
    #     if step_distance >= distance:
    #         self.x_pos, self.y_pos = float(self.target[0]), float(self.target[1])
    #         if len(self.steps_queue) == 0:
    #             self.target = None
    #             self.steps_completed = True
    #             return
    #     else:
    #         self.x_pos += (step_dir[0] / distance) * step_distance
    #         self.y_pos += (step_dir[1] / distance) * step_distance
    #     self.simulate_lidar(area, start_robot_ids)

    #     if not self.is_returning_home:
    #         # Check if the target is still a frontier, if not then reset the target
    #         directions = ['north', 'south', 'east', 'west', 'north_east', 'south_east', 'south_west', 'north_west']
    #         current_grid_pos = self.get_grid_pos()
    #         if not self.check_frontier(directions, end[0], end[1]):
    #             self.steps_queue.clear()
    #             self.target = None
    #             self.steps_completed = True
    #             return

    #     # If this was the last step mark the robot as reached destination
    #     if len(self.steps_queue) == 0 and self.x_pos == float(self.target[0]) and self.y_pos == float(self.target[1]):
    #         self.target = None
    #         self.steps_completed = True


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
        wx_clipped = np.clip(wx, self.x_min, self.x_max - 1e-4)
        wy_clipped = np.clip(wy, self.y_min + 1e-4, self.y_max)

        gx = np.floor((wx - self.x_min) / self.resolution).astype(int)
        gy = np.floor((self.y_max - wy)/ self.resolution).astype(int)
        return gx, gy

    # Convert grid coordinates into world coordinates
    def grid_to_world(self, gx, gy):
        wx = self.x_min + (gx + 0.5) * self.resolution
        wy = self.y_max - (gy + 0.5) * self.resolution
        return wx, wy

    # Update the belief grid based on the scanned wall_points
    def get_wall_points(self, wall_points, sim, drone_handle):
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

        self.world_pts = np.dot(local_pts, R.T) + T

        self.wall_points = self.extract_wall_points(self.world_pts)

        if len(self.wall_points) > 0:
            print(f"Captured {len(self.wall_points)} wall points! Sample point: {self.wall_points[0]}")


    def extract_wall_points(self, world_points, min_z=0.2, max_z=2.0):
        if len(world_points) == 0:
            return np.empty((0, 3), dtype=np.float32)

        wall_mask = (world_points[:, 2] >= min_z) & (world_points[:, 2] <= max_z)
        return world_points[wall_mask]

