import json
import os
import shutil
import subprocess
import pandas as pd
import pdb
from pathlib import Path
from functions import load_config
import locale
from natsort import natsorted
locale.setlocale(locale.LC_TIME, "en_US.UTF-8")

# Function to modify the config file
def modify_config(config_file, starttime, endtime, tracklength):
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    track_path_ = config['track_path']
    config['starttime'] = starttime.strftime("%Y-%m-%d %H:%M")
    config['endtime'] = endtime.strftime("%Y-%m-%d %H:%M")
    config['tracklength'] = tracklength
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent='')

# Function to rename the output folder
def rename_output_folder(output_path, eventdate, HIobs, count):
    old_folder_name = f"{eventdate}_{HIobs}"
    new_folder_name = f"{eventdate}_{HIobs}_{count}"
    old_path = os.path.join(output_path, old_folder_name)
    new_path = os.path.join(output_path, new_folder_name)
    
    if os.path.exists(new_path):
        shutil.rmtree(new_path)
        
    shutil.move(old_path, new_path)

# Main loop
def main():
    output_path = "/Users/tanja/ELEvoHI/predictions/"
    config = load_config('config.json')
    eventdate = config['eventdate']
    HIobs = config['HIobs']
    track_path = config['track_path']
    count = 0

    folders = natsorted([f for f in os.listdir(track_path) if os.path.isdir(os.path.join(track_path, f)) and f.startswith("timestep_")]) 
     
    for folder in folders:
        all_dates = []
        count += 1
        folder_path = Path(track_path) / folder
        tracklength = f"timestep_{count-1}/"
        config_file = "config.json"
        
        for csv_file in folder_path.glob("*.csv"):          
            df = pd.read_csv(csv_file)            
            df["TRACK_DATE"] = pd.to_datetime(df["TRACK_DATE"], format="%Y-%m-%d %H:%M:%S.%f")            
            all_dates.extend(df["TRACK_DATE"])   

        starttime = min(all_dates)
        endtime = max(all_dates)
            
        modify_config(config_file, starttime, endtime, tracklength)
        
        # Run ELEvoHI
        #subprocess.run(["python", "ELEvoHI.py"], check=True)
        
        # Run ELEvoHI with error handling
        try:
            subprocess.run(["python", "ELEvoHI.py"], check=True)
            #rename_output_folder(output_path, eventdate, HIobs, count)
        except subprocess.CalledProcessError as e:
            print(f"ELEvoHI failed for timestep {count}: {e}. Skipping to next iteration.")
        
        # Rename the output folder
        rename_output_folder(output_path, eventdate, HIobs, count)

if __name__ == "__main__":
    main()
