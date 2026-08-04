import numpy as np
import csv

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
        