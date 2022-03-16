"""
With this module you prepare data to be ingested into the model
"""
import os
import json
import pandas as pd
from datetime import datetime
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Open the config.json file get the paths variables
with open('./config.json', 'r') as conf:
    config = json.load(conf)

input_folder = config["input_data_folder"]
output_folder = config["output_data_folder"]



def merge_and_deduplicate(input_folder, output_folder):
    """
    This function will take all data available on "all_data" folder,
    merge it and drop duplicates. After that it will create a log
    .txt file with the names of all data used
    """
    filenames = os.listdir(input_folder)
    final_df = pd.DataFrame()
    for file in filenames:
        df = pd.read_csv(input_folder + "/" + file)
        final_df = pd.concat([final_df, df], ignore_index=True)
    # All dataframes from "all_data" in one
    final_df.drop_duplicates(inplace=True)
    # Save it to use to train the model
    final_df.to_csv(output_folder + "/final_data.csv", index=False)

    # Log the names of all dataframes used on the final_data.csv
    with open(output_folder + "/data_logs.txt", "w") as logs:
        for i in filenames:
            logs.write(str(i))
            logs.write("\n")


if __name__ == "__main__":
    merge_and_deduplicate(input_folder, output_folder)
