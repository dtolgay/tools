import sys 
sys.path.append("/scratch/m/murray/dtolgay")
from tools import constants 

import numpy as np 
import pandas as pd 


def read_training_data(base_file_dir, main_directory, file_name, base_line_names, epsilon=1e-30):

    #################################################
    # Get the trained data

    print("Training data is started to be read.")

    line_names = []
    for line_name in base_line_names:
        line_names.append(f"I_{line_name}")

    column_names = [
        "log_metallicity",
        "log_hden",
        "log_turbulence",
        "log_isrf",
        "log_radius",
    ]  + line_names

    # Read file
    path2TrainingData = f"{base_file_dir}/{main_directory}/{file_name}"
    unprocessed_train_data = pd.DataFrame(
        np.loadtxt(fname=path2TrainingData),
        columns=column_names,
    )

    ############## Process the cloudy data 
    # Discard all nan values 
    print("Dropping NaN containing lines")
    unprocessed_train_data = unprocessed_train_data.dropna()

    ### Check if all intensities are positive and set 0 values to epsilon
    print(f"Check if all intensities are positive. Then set 0 values to {epsilon}")
    # Check if there exist negative flux values
    negative_columns_exist = (unprocessed_train_data[line_names]).all().all()
    print(f"negative column exists: {negative_columns_exist}. Setting them to epsilon")
    
    all_positive_columns = (unprocessed_train_data[line_names] >= 0).all().all()
    if all_positive_columns:
        print(f"All of the intensity values are non-negative. Continuing...")
    else:
        # Set values smaller or equal to zero to epsilon in specified columns
        for col in line_names:
            unprocessed_train_data[col] = unprocessed_train_data[col].map(lambda x: epsilon if x <= 0 else x)
        print(f"Not all intensities are positive. Some is zero. Setting them to epsilon")


    line_names_with_log = []
    for column in line_names:
        unprocessed_train_data[f"log_{column}"] = np.log10(unprocessed_train_data[column])
        line_names_with_log.append(f"log_{column}") # Store the new line names


    train_data_df = unprocessed_train_data[[
        "log_metallicity",
        "log_hden",
        "log_turbulence",
        "log_isrf",
        "log_radius",
        ] + line_names_with_log]  # Only use the log of the line luminosities    

    # # Double check if there is any NaN
    # if (np.isnan(train_data_df.values).any()):
    #     print("Still there are NaN values. Exiting with code 1...")
    #     exit(1)
    # elif (np.isinf(train_data_df.values).any()):
    #     print("Still there are inf values. Exiting with code 2...")
    #     exit(2)

    ######
    # Add the column density data to interpolate that too 
    train_data_df['log_column_density'] = np.log10(
        (10**train_data_df['log_hden'] / constants.cm2pc**3) * (10**train_data_df['log_radius']) * (constants.mu_h * constants.proton_mass * constants.kg2Msolar)
    ) # Msolar / pc^2

    print(f"{path2TrainingData} is read.")


    return train_data_df, line_names_with_log
