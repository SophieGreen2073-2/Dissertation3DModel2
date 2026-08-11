import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
from Simulation import Simulation
import rclpy

def main():
    rclpy.init()

    try:
        sim = Simulation()
    finally:
        print("done")

main()