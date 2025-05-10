import sys
sys.path.append("/mnt/raid-cita/dtolgay/FIRE/post_processing_fire_outputs")
from tools import readsnap, functions, readsnap_FIREBox # type: ignore

import pandas as pd 
import numpy as np 


def sfr_calculator(star_df: pd.DataFrame, within_how_many_Myr:float):
    # Calculate star formation happened in the last 10 Myr
    indices = np.where(star_df["age"] <= within_how_many_Myr)[0] 
    sfr_star = np.sum(star_df.iloc[indices]["mass"]) / (within_how_many_Myr * 1e6)  # Msolar / year
    return sfr_star


def halo_mass_calculator(galaxy_name, galaxy_type, redshift):
    
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

        halo_mass = mass_of_MMH
    
    elif galaxy_type == "firebox":

        if redshift == "0.0":
            snapshot_number = 1200     # z = 0.0
            snap_dir_file_path = '/fs/lustre/project/murray/scratch/lliang/FIRE_CO/FIREbox'
            # snap_dir_file_path = '/fs/lustre/project/murray/scratch/tolgay/firebox/FB15N1024/seperated_galaxies'
        elif redshift == "0.5":
            print(f"Exiting... Currently there are no z=0.5 galaxies... {redshift}")
            sys.exit(2)                
        elif redshift == "1.0":
            snapshot_number = 554     # z = 1.0
            snap_dir_file_path = '/fs/lustre/project/murray/scratch/tolgay/firebox/FB15N1024/seperated_galaxies/z1.0'
        elif redshift == "2.0":
            snapshot_number = 344     # z = 2.0
            snap_dir_file_path = '/fs/lustre/project/murray/scratch/tolgay/firebox/FB15N1024/seperated_galaxies/z2.0'
        elif redshift == "3.0":
            snapshot_number = 240     # z = 3.0
            # snap_dir_file_path = '/fs/lustre/project/murray/scratch/lliang/FIRE_CO/FIREbox' # TODO. Where is the path?
        else:
            print(f"Exiting... Redshift is wrong. The given galaxy type is {redshift}")
            sys.exit(2)         
        
        
        galaxy_number = int(galaxy_name.replace("gal", ""))
        
        print(f"redshift: {redshift}")

        if redshift in ["0.0", "3.0"]:
            
            print("Reading header information...")
            header_info = readsnap_FIREBox.readsnap(
                sdir=snap_dir_file_path, 
                snum=snapshot_number, 
                simulation_name="snapshot", 
                file_number=galaxy_number, 
                ptype=0, 
                header_only=1
            )

            hubble      = header_info['hubble']
            time        = header_info['time']  

            redshift_to_read_ahf_files = f"{float(redshift):.3f}" # Convert redshift 0 -> 0.000 as a string
            fdir = "/fs/lustre/project/murray/scratch/lliang/FIRE_CO/FIREbox/AHF"
            ahf_file_name = f"FB15N1024.z{redshift_to_read_ahf_files}.AHF_halos"

            column_names = [
                "ID", "hostHalo", "numSubStruct", "Mvir", "npart", "Xc", "Yc", "Zc", "VXc", "VYc", "VZc",
                "Rvir", "Rmax", "r2", "mbp_offset", "com_offset", "Vmax", "v_esc", "sigV", "lambda", "lambdaE",
                "Lx", "Ly", "Lz", "b", "c", "Eax", "Eay", "Eaz", "Ebx", "Eby", "Ebz", "Ecx", "Ecy", "Ecz",
                "ovdens", "nbins", "fMhires", "Ekin", "Epot", "SurfP", "Phi0", "cNFW", "n_gas", "M_gas",
                "lambda_gas", "lambdaE_gas", "Lx_gas", "Ly_gas", "Lz_gas", "b_gas", "c_gas", "Eax_gas",
                "Eay_gas", "Eaz_gas", "Ebx_gas", "Eby_gas", "Ebz_gas", "Ecx_gas", "Ecy_gas", "Ecz_gas",
                "Ekin_gas", "Epot_gas", "n_star", "M_star", "lambda_star", "lambdaE_star", "Lx_star",
                "Ly_star", "Lz_star", "b_star", "c_star", "Eax_star", "Eay_star", "Eaz_star", "Ebx_star",
                "Eby_star", "Ebz_star", "Ecx_star", "Ecy_star", "Ecz_star", "Ekin_star", "Epot_star"
            ]
            halo_fdir = f"{fdir}/{ahf_file_name}"
            halos = pd.DataFrame(
                np.loadtxt(halo_fdir),
                columns=column_names
            )
            # Change the ID column data type to int
            halos['ID'] = halos['ID'].astype(int)
            halos['hostHalo'] = halos['hostHalo'].astype(int)
            halos['numSubStruct'] = halos['numSubStruct'].astype(int)
            halos['npart'] = halos['npart'].astype(int)                                    
            
            # Get the Mvir of the halo that has the same ID with the galaxy number 
            condition = halos['ID'] == galaxy_number
            mvir = halos[condition]['Mvir'].values[0]
            mvir = mvir * 1/hubble # [Msolar]

            halo_mass = mvir

        else: 
            print(f"Implement other redshifts for FIREBox simulation. I am in the halo_mass function.")
            sys.exit(1)
        
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

        halo_mass = mass_of_MMH
        
    else: 
        halo_mass = np.nan # TODO: 
    
    return halo_mass

def get_halo_mass_from_previously_calculated_file(galaxy_name, galaxy_type, redshift):

    print(f" ---------- I am in the get_halo_mass_from_previously_calculated_file function for galaxy: {galaxy_name}. ---------- ")

    # TODO: 
    if galaxy_type == "zoom_in":
        fdir = "/mnt/raid-cita/dtolgay/FIRE/post_processing_fire_outputs/skirt/python_files/analyze_hden_metallicity_turbulence_isrf_radius/data/2025_04_18_wrongSfrCalculator_trustOnlySFR_10Myr"
        file_name = f"galactic_properties_smoothingLength_hybridInterpolator_z{float(redshift):.0f}_usingIvalues.csv"

        # Read the csv file 
        df = pd.read_csv(f"{fdir}/{file_name}")
        # Get the halo mass
        halo_mass = df[df["name"] == galaxy_name]["halo_mass"].values[0]
    else:
        # Not implemented. Exit
        print("Exiting... The galaxy type is not implemented yet.")
        sys.exit(1)
        return np.nan
    
    print(f"Finished reading halo mass from the file. The halo mass is {halo_mass:.2e} Msun.")

    return halo_mass

if __name__ == "__main__":
    galaxy_name = "m12b_res7100_md"
    galaxy_type = "zoom_in"
    redshift = "0.0"
    halo_mass = halo_mass_calculator(galaxy_name, galaxy_type, redshift)

def mass_average(property_array, mass):
    return sum(property_array * mass) / sum(mass)
