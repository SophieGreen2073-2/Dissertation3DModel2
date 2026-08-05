import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import os
import json
from Record import RecordRedundancy, RecordTime
from RobotModels.UAVModel import UAVModel
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
            num_ugvs = run["NumUGVs"]
            num_legged = run["NumLegged"]

            # Get walls within the area
            self.area = AreaModel(self.sim)

            # Create list of active UAVs
            self.CreateQuadcopterList()

            # self.time_elapsed = 0
            record_time = RecordTime()
            record_redundancy = RecordRedundancy()

            self.client.step()

            # try:
            # 3. Step FIRST, then inspect simulation state safely
            while True:
                self.completed = True
                # Advance the simulation step (CRITICAL)
                self.client.step()

                # Check if simulation was stopped manually from GUI or script
                if self.sim.getSimulationState() == self.sim.simulation_stopped:
                    break

                self.sim_time = self.sim.getSimulationTime()

                for uav in self.UAVs:
                    # uav.scan()
                    if not uav.current_path:
                        uav.yamauchi_move()
                    else:
                        uav.step_robot()
                    self.completed &= uav.completed

                if self.completed:
                    break

            # except Exception as e:
            #     print(f"Simulation run interrupted: {e}")

            # finally:
                # 4. Clean up between runs
            self.sim.stopSimulation()
                    

            # while(True):
            #     self.completed = True
            #     for uav in self.UAVs:
            #         if not uav.released and round(self.time_elapsed, 1) == round((uav.robot_id - self.startRobotIDs), 1) * self.UAVParams["ReleaseDelay"]:
            #             uav.released = True
            #         if uav.steps_completed and uav.released:
            #             uav.yamauchi_move_utility_function(self.area, self.startRobotIDs)
            #         self.completed &= uav.completed

            #     if self.completed:
            #         break
            #     self.StepRobots()

            # record_time.record_time_elapsed(num_uavs, self.time_elapsed, self.UAVParams)
            # record_redundancy.record_overlap(self.area.overlap_area, num_uavs, self.UAVParams)

            # sim.stopSimulation()
        

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
            self.Grid = d["Grid"]


    # Calculate total dBm for communication between robots
    def CalculateTotalLinkBudget(self):
        comms_params = self.UAVParams["Communications"]

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
                        self.UAVParams["MinRange"])

            self.UAVs.append(uav)