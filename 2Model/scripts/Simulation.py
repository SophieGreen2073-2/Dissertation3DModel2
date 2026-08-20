import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import os
import json
import math
from Record import RecordRedundancy, RecordTime, RecordScannedGrid
from RobotModels.UAVModel import UAVModel
from RobotModels.UGVModel import UGVModel
from AreaModel import AreaModel

class Simulation():
    def __init__(self):
        ("Create Simulation")
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')

        self.client.setStepping(True)

        self.GetParams()
        self.CalculateTotalLinkBudget()

        for run in self.simulations:
            # Start simulation
            self.sim.startSimulation()

            self.num_uavs = run["NumUAVs"]
            self.num_ugvs = run["NumUGVs"]
            num_legged = run["NumLegged"]

            # Get walls within the area
            self.area = AreaModel(self.sim, self.Grid["Height"], self.Grid["Width"], self.num_ugvs, self.Grid["Resolution"])

            # Create list of active UAVs
            self.CreateQuadcopterList()

            # Create list of active UGVs
            self.CreateUGVList()

            # self.time_elapsed = 0
            record_time = RecordTime()
            record_redundancy = RecordRedundancy()
            record_scanned_grid = RecordScannedGrid()

            self.client.step()

            # try:
            while True:
                self.completed = True
                self.client.step()

                # Check if simulation was stopped manually from GUI or script
                if self.sim.getSimulationState() == self.sim.simulation_stopped:
                    break

                self.sim_time = self.sim.getSimulationTime()

                for ugv in self.UGVs:
                    # uav.scan()
                    if not ugv.steps_queue:
                        ugv.yamauchi_move_utility_function(self.area)
                    else:
                        ugv.step_robot(self.area)
                    self.completed &= ugv.completed

                if self.completed:
                    break
            # except Exception:
            #     print(Exception)
            # finally:
            #     self.sim.stopSimulation()
            #     record_time.record_time_elapsed(self.num_ugvs, self.sim.getSimulationTime(), self.UGVParams)
            #     record_redundancy.record_overlap(self.area.overlap_area, self.num_ugvs, self.UGVParams)
            #     record_scanned_grid.save_final_grids(self.UGVs)
        

    # Get simulation parameters
    def GetParams(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, "JSONFiles/SimulationParams.JSON")
        
        with open(json_path) as f:
            d = json.load(f)
            self.simulations = d["Simulations"]
            self.startRobotIDs = d["StartRobotIDs"]
            self.time_step = d["TimeStep"]
            self.recharge_point = d["RechargePoint"]
            self.is_comms_modelled = d["IsCommsModelled"] == 1
            self.UAVParams = d["UAVParams"]
            self.UGVParams = d["UGVParams"]
            self.Grid = d["Grid"]


    # Calculate total dBm for communication between robots
    def CalculateTotalLinkBudget(self):
        comms_params = self.UGVParams["Communications"]

        # Get params used to model wifi communication
        transmit_power = comms_params["TransmitPower"]
        receiver_sensitivity = comms_params["ReceiverSensitivity"]
        antennae_gains = comms_params["AntennaeGains"]
        interference_margin = comms_params["InterferenceMargin"]

        # Calculate if the total communication budget is bigger than the amount needed to communicate
        self.total_link_budget = transmit_power + antennae_gains - receiver_sensitivity - interference_margin


    # Create the list of UAVs in the area
    def CreateQuadcopterList(self):
        self.UAVs = []

        # Loop through each quadcopter
        for index in range(self.num_uavs):
            drone_base = None
            drone_target = None

            if self.num_uavs == 1:
                drone_base = self.sim.getObject('/Quadcopter')
                drone_target = self.sim.getObject('/target')
            else:
                drone_base = self.sim.getObject(f'/Quadcopter[{index}]')
                drone_target = self.sim.getObject(f'/target[{index}]')

            pos = self.sim.getObjectPosition(drone_base, -1)
            alias = self.sim.getObjectAlias(drone_base)

            uav = UAVModel(pos[0], pos[1], pos[2], self.UAVParams["TopSpeed"],
                        self.UAVParams["DangerSpeed"], self.UAVParams["StartSpeed"],
                        self.UAVParams["Acceleration"], self.UAVParams["BatteryLife"],
                        self.UAVParams["ChargeTime"], len(self.UAVParams) + self.startRobotIDs,
                        alias, drone_base, drone_target, [], self.Grid["Width"],
                        self.Grid["Height"], self.sim, self.UAVParams["FOVDeg"],
                        self.UAVParams["MaxRange"], self.UAVParams["ScanFrequency"],
                        self.UAVParams["MinRange"], self.UAVParams["WallDangerZone"])

            self.UAVs.append(uav)


    # Create the list of UAVs in the area
    def CreateUGVList(self):
        self.UGVs = []

        # Loop through each quadcopter
        for index in range(self.num_ugvs):
            robot_handle = None
            left_motor = None
            right_motor = None

            if self.num_ugvs == 1:
                robot_handle = self.sim.getObject('/PioneerP3DX')
                left_motor = self.sim.getObject('/PioneerP3DX/leftMotor')
                right_motor = self.sim.getObject('/PioneerP3DX/rightMotor')
            else:
                robot_handle = self.sim.getObject(f'/PioneerP3DX[{index}]')
                left_motor = self.sim.getObject(f'/PioneerP3DX[{index}]/leftMotor')
                right_motor = self.sim.getObject(f'/PioneerP3DX[{index}]/rightMotor')

            pos = self.sim.getObjectPosition(robot_handle, -1)
            # alias = self.sim.getObjectAlias(drone_base)

            ugv = UGVModel(robot_handle, left_motor, right_motor, self.sim,
                           self.UGVParams["Speed"], self.UGVParams["Sensors"],
                           self.UGVParams["Battery"], self.UGVParams["WallDangerZone"],
                           self.Grid["Height"], self.Grid["Width"], self.Grid["Resolution"],
                           self.area, index)

            self.UGVs.append(ugv)

    def ShareRobotData(self):
        for ugv in self.UGVs:
            ugv.localUGVs = []
            for ugv2 in self.UGVs:
                if ugv == ugv2:
                    continue

                if self.is_comms_modelled:
                    transmission_possible = self.Is_Transmission_Possible(ugv, ugv2)
                else:
                    transmission_possible = True

                if transmission_possible:
                    if not ugv2 in ugv.localUGVs:
                        ugv.localUGVs.append(ugv2)
                    for row in range(ugv.occupancy_grid.height):
                        for col in range(ugv.occupancy_grid.width):
                            if ugv.occupancy_grid.grid[row, col] == 0:
                                ugv.occupancy_grid.grid[row, col] = ugv2.occupancy_grid.grid[row, col]



    # Calculate if transmission is possible between two UAVs
    def Is_Transmission_Possible(self, ugv1: UGVModel, ugv2: UGVModel):
        # Check params
        step = 0.1
        epsilon = 1e-9

        # Difference between uav1 and uav2
        curr_pos_1 = self.sim.getObjectPosition(ugv1.robot_handle, -1)
        curr_grid_pos_1 = self.area.world_to_grid(curr_pos_1[0], curr_pos_1[1])

        curr_pos_2 = self.sim.getObjectPosition(ugv2.robot_handle, -1)
        curr_grid_pos_2 = self.area.world_to_grid(curr_pos_2[0], curr_pos_2[1])

        dx = curr_grid_pos_2[0] - curr_grid_pos_1[0]
        dy = curr_grid_pos_2[1] - curr_grid_pos_1[1]
        total_dist = math.hypot(dx, dy)

        # Drones are at the exact same point
        if total_dist < 1e-6:
            return True

        # Distance from uav1 to check point
        step = 0.1  # Step resolution in grid units
        num_steps = int(math.ceil(total_dist / step))
        
        # Normalized direction vector per step
        step_x = (dx / total_dist) * step
        step_y = (dy / total_dist) * step
        x_pos = curr_grid_pos_1[0]
        y_pos = curr_grid_pos_1[1]

        # Moniter the amount of free space and wall space between the drones
        free_space = 0
        wall_space = 0

        # Check along line until reaching other uav2
        for i in range(num_steps):
            grid_x = int(round(x_pos + epsilon))
            grid_y = int(round(y_pos + epsilon))

            # Boundary check safeguard before array lookup
            if 0 <= grid_y < self.area.width * self.area.resolution and 0 <= grid_x < self.area.height * self.area.resolution:
                if self.area.walls_grid[grid_y, grid_x] == 1:
                    wall_space += step
                else:
                    free_space += step
            else:
                # Out of bounds grid cell treated as free space (or wall depending on your sim rules)
                free_space += step

            # Advance along ray
            x_pos += step_x
            y_pos += step_y

        comms_params = self.UGVParams["Communications"]

        # Get params used to model wifi communication
        frequency = comms_params["Frequency"]
        concrete_loss = comms_params["ConcreteLoss"]
        
        total_required = self.ConcreteLoss(wall_space, concrete_loss) + self.CalculateFreeSpaceLoss(free_space, frequency)

        return self.total_link_budget > total_required


    # Calculate signal loss through the amount of wall
    def ConcreteLoss(self, wall_space, concrete_loss):
        return wall_space * concrete_loss

    # Calculate signal loss through the free space
    def CalculateFreeSpaceLoss(self, free_space_dist, frequency):
        if free_space_dist <= 0.001:
            return 0.0

        return 20 * math.log10(free_space_dist / 1000) + 20 * math.log10(frequency) + 32.45