import numpy as np
import csv
import pandas as pd

class RecordTime():
    def record_time_elapsed(self, num_robots, time_elapsed, uav_params):
        with open('dissertation_time_record.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            
            # 1. Start with your base variables safely converted
            row = [int(num_robots), f"{time_elapsed:.6f}"]
            
            # 2. Extract and iterate through the dict values, skipping the keys
            # .values() gives you the actual parameters (e.g., 2.5, "Mavic", True)
            for param in uav_params.values():
                if isinstance(param, float):
                    row.append(f"{param:.6f}") # Safely format floats to 6 decimal places
                elif isinstance(param, (int, bool)):
                    row.append(int(param))     # Write integers/booleans cleanly without decimal drift
                else:
                    row.append(str(param))     # Write text, strings, or labels exactly as they are
            
            # 3. Append the mixed data row directly to the CSV
            writer.writerow(row)


class RecordRedundancy():
    def record_overlap(self, overlap_area, numUAVs, uav_params):
        with open('dissertation_redundancy_record.csv', 'a') as f:
            writer = csv.writer(f)
            
            # 1. Start with your base variables safely converted
            row = [int(numUAVs)]

            for val in overlap_area.ravel():
                row.append(f"{float(val):.6f}")
            
            # 2. Extract and iterate through the dict values, skipping the keys
            # .values() gives you the actual parameters (e.g., 2.5, "Mavic", True)
            for param in uav_params.values():
                if isinstance(param, float):
                    row.append(f"{param:.6f}") # Safely format floats to 6 decimal places
                elif isinstance(param, (int, bool)):
                    row.append(int(param))     # Write integers/booleans cleanly without decimal drift
                else:
                    row.append(str(param))     # Write text, strings, or labels exactly as they are
            
            # 3. Append the mixed data row directly to the CSV
            writer.writerow(row)

class RecordScannedGrid():
    def save_final_grids(self, ugvs, ugv_params):
        with open('dissertation_scanned_grids_record.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            
            for i, ugv in enumerate(ugvs):
                # 1. Start the row with the robot's identifier/index
                row = [int(i)]
                
                # 2. Append the flattened grid values
                flattened_grid = ugv.occupancy_grid.belief_grid.flatten()
                for val in flattened_grid:
                    row.append(f"{float(val):.6f}")
                
                # 3. Extract and append parameters matching the other record classes
                for param in ugv_params.values():
                    if isinstance(param, float):
                        row.append(f"{param:.6f}")
                    elif isinstance(param, (int, bool)):
                        row.append(int(param))
                    else:
                        row.append(str(param))
                
                # 4. Write this robot's complete row to the CSV
                writer.writerow(row)