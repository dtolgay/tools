import sys
sys.path.append("/mnt/raid-cita/dtolgay/FIRE/post_processing_fire_outputs")
from tools import readsnap, functions # type: ignore

import pandas as pd 
import numpy as np 


def sfr_calculator(star_df: pd.DataFrame, within_how_many_Myr:float):
    # Calculate star formation happened in the last 10 Myr
    indices = np.where(star_df["age"] <= within_how_many_Myr)[0] 
    sfr_star = np.sum(star_df.iloc[indices]["mass"]) / (within_how_many_Myr * 1e6)  # Msolar / year
    return sfr_star


def halo_mass(galaxy_name, galaxy_type, redshift):
    

    if galaxy_type == "zoom_in":
        
        if redshift == "0.0":
            snapshot_number = 600     # z = 0.0
        elif redshift == "0.5":
            snapshot_number = 381     # z = 0.5
        elif redshift == "1.0":
            snapshot_number = 277     # z = 1.0
        elif redshift == "2.0":
            snapshot_number = 172     # z = 2.0
        elif redshift == "3.0":
            snapshot_number = 120     # z = 3.0
        else:
            print(f"Exiting... Redshift is wrong. The given galaxy type is {redshift}")
            sys.exit(2)
        
        if galaxy_name[0:3] == "m12":
            halo_finder_file_path = f'/mnt/raid-project/murray/FIRE/FIRE_2/Fei_analysis/md/{galaxy_name}/rockstar_dm/catalog'
            snap_dir_file_path = f'/mnt/raid-project/murray/FIRE/FIRE_2/Fei_analysis/md/{galaxy_name}/output'

        elif galaxy_name[0:3] == "m11":
            halo_finder_file_path = f'/fs/lustre/project/murray/scratch/tolgay/metal_diffusion/{galaxy_name}/halo/rockstar_dm/catalog'
            snap_dir_file_path = f'/fs/lustre/project/murray/scratch/tolgay/metal_diffusion/{galaxy_name}/output'
            
        header_info = readsnap.readsnap(snap_dir_file_path, snapshot_number, 0, header_only=1)
    
        hubble      = header_info['hubble']
        time      = header_info['time']
        
        
        mass_of_MMH, x_MMH_center, y_MMH_center, z_MMH_center, vx_MMH_center, vy_MMH_center, vz_MMH_center, ID, DescID = functions.halo_with_most_particles_rockstar(
            rockstar_snapshot_dir=halo_finder_file_path, 
            snapshot_number=snapshot_number,
            time = time,
            hubble = hubble
        )    
    
    elif galaxy_type == "firebox":
        
        redshift = f"{float(redshift):.3f}" # Convert redshift 0 -> 0.000 as a string
        
        galaxy_number = int(galaxy_name.replace("gal", ""))
            
        fdir = "/fs/lustre/project/murray/scratch/lliang/FIRE_CO/FIREbox/AHF"

        halos = np.loadtxt(fname = f"{fdir}/FB15N1024.z{redshift}.AHF_halos", skiprows=1) 
        
        mass_of_MMH = halos[galaxy_number][3]
        
        
    elif galaxy_type == "particle_split":
        # Read halos for the particle splitted FIREBox simualation
        import h5py
        
        if redshift == "0.0":
            snapshot_number = 600     # z = 0.0
        elif redshift == "0.5":
            snapshot_number = 381     # z = 0.5
        elif redshift == "1.0":
            snapshot_number = 277     # z = 1.0
        elif redshift == "2.0":
            snapshot_number = 172     # z = 2.0
        elif redshift == "3.0":
            snapshot_number = 120     # z = 3.0
        else:
            print(f"Exiting... Redshift is wrong. The given galaxy type is {redshift}")
            sys.exit(2)
            
        halo_finder_file_path = f"/fs/lustre/project/murray/FIRE/FIRE_2/{galaxy_name}/halos"
        hf = h5py.File(f"{halo_finder_file_path}/halo_{snapshot_number}.hdf5", 'r')
        virilized_mass = np.array(hf.get("mass.bound"))
        
        mass_of_MMH = max(virilized_mass)

        hf.close() # Close the file        
        
    else: 
        mass_of_MMH = np.nan # TODO: 
    
    return mass_of_MMH


def mass_average(property_array, mass):
    return sum(property_array * mass) / sum(mass)
