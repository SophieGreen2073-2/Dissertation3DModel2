from collections import deque
import numpy as np
from RobotModels.OccupancyBeliefGrid import UGVOccupancyGrid
import math

class UGVModel():
    def __init__(self, robot_handle, left_motor, right_motor, sim, speed_params,
                 sensor_params, battery_params, wall_danger_zone, grid_height,
                 grid_width, resolution):

        # Simulation
        self.sim = sim

        # Robot Status
        self.completed = False
        self.steps_completed = True
        self.is_returning_home = False

        # Robot handle names
        self.robot_handle = robot_handle
        self.left_motor = left_motor
        self.right_motor = right_motor

        # Robot velocities
        self.top_speed = speed_params["TopSpeed"]
        self.danger_speed = speed_params["DangerSpeed"]
        self.start_speed = speed_params["StartSpeed"]
        self.acceleration = speed_params["Acceleration"]
        self.steering_gain = 2.5

        # Robot paths
        self.steps_queue = deque()

        # Battery
        self.battery = Battery(battery_params)

        # Camera models
        self.sensors = Sensors(sensor_params)

        # Occupancy grid
        self.occupancy_grid = UGVOccupancyGrid(resolution, grid_height, grid_width)

        # Utility function
        self.util_cost_weight = 1
        self.util_penalty_weight = 300
        self.util_wall_weight = 10
        self.frontier_count = 20

        # Wall avoidance
        self.wall_danger_zone = wall_danger_zone

        # Local UAVs
        self.localUAVs = []

        # Robot target
        self.target_pos = None
        self.target_orient = None


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
        return self.occupancy_grid.grid_to_world(target_wp[0], target_wp[1])


    # Update robot target and speed
    def step_target(self):
        curr_pos = self.sim.getObjectPosition(self.robot_handle, -1)
        curr_orient = self.sim.getObjectOrientation(self.robot_handle, -1)

        if not self.steps_queue:
            self.steps_completed = True
            return

        target_wx, target_wy = self.get_lookahead_target(curr_pos, lookahead_dist=1)

        if target_wx is None:
            self.sim.setJointTargetVelocity(self.left_motor, 0)
            self.sim.setJointTargetVelocity(self.right_motor, 0)
            return 

        step_dir = (target_wx - curr_pos[0], target_wy - curr_pos[1])
        dist_to_target = math.hypot(step_dir[0], step_dir[1])
        
        target_angle = math.atan2(step_dir[1], step_dir[0])
        angle_diff = math.atan2(math.sin(target_angle - curr_orient[2]), math.cos(target_angle - curr_orient[2]))

        left_speed = self.top_speed + (angle_diff * self.steering_gain)
        right_speed = self.top_speed - (angle_diff * self.steering_gain)

        self.sim.setJointTargetVelocity(self.left_motor, left_speed)
        self.sim.setJointTargetVelocity(self.right_motor, right_speed)

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

        if self.step_count % self.sensors.scan_frequency == 0:
            self.sensor.get_points(self.sim)
            self.occupancy_grid.update_belief(self.sim, self.drone_base, self.sensor.fov_deg, self.max_range, self.sensor.wall_points)

        # Increase number of steps
        self.step_count += 1


class Battery():
    def __init__(self, battery_params):
        self.life = battery_params["Life"]
        self.charge_time = battery_params["ChargeTime"]


class Sensors():
    def __init__(self, sensor_params):
        # Scan frequency
        self.scan_frequency = sensor_params["ScanFrequency"]

        # Forward lidar
        self.forward_lidar = ForwardLiDAR(sensor_params["LiDAR"])

        # 360 SLAM
        self.slam_sensors = []
        for _ in range(4): self.slam_sensors.append(ForwardLiDAR(sensor_params["VisionSensor"]))


    def get_points(self):
        return

class ForwardLiDAR():
    def __init__(self, LiDAR_params):
        # LiDAR range
        self.min_range = LiDAR_params["MinRange"]
        self.max_range = LiDAR_params["MaxRange"]


class VisionSensor():
    def __init__(self, vision_params):
        # Vision range
        self.min_range = vision_params["MinRange"]
        self.max_range = vision_params["MaxRange"]

        # FoV
        self.fov = vision_params["FovDegrees"]


