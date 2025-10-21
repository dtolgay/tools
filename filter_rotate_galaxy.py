import os
from . import functions
from . import constants

import pandas as pd 
import numpy as np 



def finding_the_average_velocity_vector(vx:np.ndarray, 
                                        vy:np.ndarray,
                                        vz:np.ndarray)->list:
    
    '''This function finds the average velocity vector for given particles.
    
    Arguments:
    ----------
    vx : np.ndarray
    Velocities of the particles along the X direction

    vy : np.ndarray
    Velocities of the particles along the Y direction
    
    vz : np.ndarray
    Velocities of the particles along the Z direction    

    Returns:
    ----------
    [vx_average, vy_average, vz_average]: list
    Average velocity vectors for X, Y, and Z direction. The units are the same
    with the units of the inputted velocity arrays.
    
    '''
    vx_average = np.sum(vx)/len(vx)
    vy_average = np.sum(vy)/len(vy)
    vz_average = np.sum(vz)/len(vz)
    
    return [vx_average, vy_average, vz_average]

############################################################################################################################

def shifting_velocities(vx:np.ndarray,
                        vy:np.ndarray,
                        vz:np.ndarray,
                        v_average:list)->(np.ndarray,
                                          np.ndarray,
                                          np.ndarray):
    
    '''This function shifts the velocities according to the average velocities list found above.
    
    Arguments:
    ----------
    vx : np.ndarray
    Velocities of the particles along the X direction

    vy : np.ndarray
    Velocities of the particles along the Y direction
    
    vz : np.ndarray
    Velocities of the particles along the Z direction    

    v_average : list
    Average velocities of the particles in X, Y and Z direction    

    Returns:
    ----------
    vx : np.ndarray
    Velocities of the particles along the X direction after shifting.
    Units are the same with the inputted velocity units.

    vy : np.ndarray
    Velocities of the particles along the Y direction after shifting.
    Units are the same with the inputted velocity units.
    
    vz : np.ndarray
    Velocities of the particles along the Z direction after shifting.
    Units are the same with the inputted velocity units.
    
    '''
    
    vx -= v_average[0] 
    vy -= v_average[1] 
    vz -= v_average[2] 
    
    return (vx, vy, vz)

############################################################################################################################

def rotate_galaxy(gas_particles_df, star_particles_df):
    
    L_gas = functions.net_angular_momentum(
        mass=gas_particles_df["mass"], 
        rx=gas_particles_df["x"],
        ry=gas_particles_df["y"], 
        rz=gas_particles_df["z"], 
        vx=gas_particles_df["vx"], 
        vy=gas_particles_df["vy"], 
        vz=gas_particles_df["vz"]
    ) 
    # lx, ly, and lz are the indices 0, 1, and 2 respectively the net angular momentum of gas particles
    # L unit: [1e10 M☉ kpc km / sec]


        # Finding the angles between coordinate axis and net angular momentum
    theta, phi = functions.finding_the_angles_between_current_coordinate_system_and_net_angular_momentum(L=L_gas)
    # theta [radian]
    # phi   [radian]


        # Rotating the coordinate system such that z axis of the net angular momentum coincides with the positive z axis of the new coordinate system
    x_star, y_star, z_star = functions.rotating_coordinate_system_along_net_angular_momentum(
        theta=theta, 
        phi=phi, 
        vectorx=star_particles_df["x"], 
        vectory=star_particles_df["y"], 
        vectorz=star_particles_df["z"]
    )


    vx_star, vy_star, vz_star = functions.rotating_coordinate_system_along_net_angular_momentum(
        theta=theta, 
        phi=phi, 
        vectorx=star_particles_df["vx"], 
        vectory=star_particles_df["vy"], 
        vectorz=star_particles_df["vz"]
    )


    x_gas, y_gas, z_gas = functions.rotating_coordinate_system_along_net_angular_momentum(
        theta=theta, 
        phi=phi,
        vectorx=gas_particles_df["x"],
        vectory=gas_particles_df["y"],
        vectorz=gas_particles_df["z"]
    )

    vx_gas, vy_gas, vz_gas = functions.rotating_coordinate_system_along_net_angular_momentum(
        theta=theta, 
        phi=phi,
        vectorx=gas_particles_df["vx"],
        vectory=gas_particles_df["vy"],
        vectorz=gas_particles_df["vz"]
    )


    # Editing the dataframes 
    gas_particles_df["x"] = x_gas
    gas_particles_df["y"] = y_gas
    gas_particles_df["z"] = z_gas
    gas_particles_df["vx"] = vx_gas
    gas_particles_df["vy"] = vy_gas
    gas_particles_df["vz"] = vz_gas

    star_particles_df["x"] = x_star
    star_particles_df["y"] = y_star
    star_particles_df["z"] = z_star
    star_particles_df["vx"] = vx_star
    star_particles_df["vy"] = vy_star
    star_particles_df["vz"] = vz_star    
    
    return gas_particles_df, star_particles_df

############################################################################################################################

def filter_rotate_galaxy(
    galaxy_name,    
    galaxy_type,    
    header_info, 
    gas_particles, 
    star_particles,   
    snapshot_number = None, # Not important if the galaxy is firebox     
    R_max = 20.0, # kpc    
    cluster_name = "cita" # cita, niagara or trillium. The cluster where the data is stored.
):
    
    hubble      = header_info['hubble']
    redshift    = header_info['redshift']   
    time        = header_info['time']   

    print(f"redshift: {redshift}")

    # In[4]:


    # Assuming gas_particles and star_particles are dictionaries containing the relevant data

    print("Creating DataFrame for gas and star particles")
    # Create dataframe for gas particles
    gas_particles_df = pd.DataFrame({
        'x': gas_particles['p'][:,0],
        'y': gas_particles['p'][:,1],
        'z': gas_particles['p'][:,2],
        'vx': gas_particles['v'][:,0],
        'vy': gas_particles['v'][:,1],
        'vz': gas_particles['v'][:,2],
        'mass': gas_particles['m'],
        'density': gas_particles['rho'],
        'smoothing_length': gas_particles['h'],
        'star_formation_rate': gas_particles['sfr'],
        'internal_energy': gas_particles['u'] * 1e6,  # Converted to [m^2 s^-2]
        'neutral_hydrogen_fraction': gas_particles['nh'],
        'electron_abundance': gas_particles['ne'],
        'metallicity': gas_particles['z'][:,0],
        'He_mass_fraction': gas_particles['z'][:,1],
        'C_mass_fraction': gas_particles['z'][:,2],
        'N_mass_fraction': gas_particles['z'][:,3],
        'O_mass_fraction': gas_particles['z'][:,4],
        'Ne_mass_fraction': gas_particles['z'][:,5],
        'Mg_mass_fraction': gas_particles['z'][:,6],
        'Si_mass_fraction': gas_particles['z'][:,7],
        'S_mass_fraction': gas_particles['z'][:,8],
        'Ca_mass_fraction': gas_particles['z'][:,9],
        'Fe_mass_fraction': gas_particles['z'][:,10],
        # You can add other fractions as needed
    })

    # Create dataframe for star particles
    star_particles_df = pd.DataFrame({
        'x': star_particles['p'][:,0],
        'y': star_particles['p'][:,1],
        'z': star_particles['p'][:,2],
        'vx': star_particles['v'][:,0],
        'vy': star_particles['v'][:,1],
        'vz': star_particles['v'][:,2],
        'metallicity': star_particles['z'][:,0],
        'scale_factor': star_particles['age'],
        'mass': star_particles['m']
    })

    # Now you have two dataframes: gas_particles_df and star_particles_df


    # In[5]:
    # Reading the MMH information
    print("Reading MMH")

    if galaxy_type == "particle_split":
        # Read halos for the particle splitted FIREBox simualation
        import h5py
        halo_finder_file_path = f"/fs/lustre/project/murray/FIRE/FIRE_2/{galaxy_name}/halos"
        print(f"halo_finder_file_path: {halo_finder_file_path}")
        hf = h5py.File(f"{halo_finder_file_path}/halo_{snapshot_number}.hdf5", 'r')
        # print(f"hf.keys(): {hf.keys()}")
        virilized_mass = np.array(hf.get("mass.bound"))
        # mass is used
        # mass.vir is used 
        positions = np.array(hf.get("position"))
        velocities = np.array(hf.get("velocity"))
        index = np.argmax(virilized_mass) 

        x_MMH_center = positions[index][0] # kpc
        y_MMH_center = positions[index][1] # kpc 
        z_MMH_center = positions[index][2] # kpc


        vx_MMH_center = velocities[index][0] # km/s
        vy_MMH_center = velocities[index][1] # km/s
        vz_MMH_center = velocities[index][2] # km/s

        mass_of_MMH = virilized_mass[index] # Msolar

        hf.close() # Close the file
        
    elif galaxy_type == "zoom_in":
        """
        Get the information of the MMH. This serves two purposses: 
        1. Find the center of the halo and filter the gas and star particles such that only the particles within the given kpc distance 
        away from the center will be considered 
        2. Rotate the galaxy by findinf the average velocity vectors. 
        """

        halo_finder_file_path = f'/mnt/raid-project/murray/FIRE/FIRE_2/Fei_analysis/md/{galaxy_name}/rockstar_dm/catalog'
        print(f"halo_finder_file_path: {halo_finder_file_path}")

        mass_of_MMH, x_MMH_center, y_MMH_center, z_MMH_center, vx_MMH_center, vy_MMH_center, vz_MMH_center, ID, DescID = functions.halo_with_most_particles_rockstar(
            rockstar_snapshot_dir=halo_finder_file_path, 
            snapshot_number=snapshot_number,
            time = time,
            hubble = hubble
        )

        # Units
        # ----
        # mass_of_MMH [M☉]
        # x_MMH_center [kpc]
        # y_MMH_center [kpc]
        # z_MMH_center [kpc]
        # vx_MMH_center [km/s]
        # vy_MMH_center [km/s]
        # vz_MMH_center [km/s]    

    elif galaxy_type == "zoom_in_tolgay":
        """
        Get the information of the MMH. This serves two purposses: 
        1. Find the center of the halo and filter the gas and star particles such that only the particles within the given kpc distance 
        away from the center will be considered 
        2. Rotate the galaxy by findinf the average velocity vectors. 
        """

        halo_finder_file_path = f'/fs/lustre/project/murray/scratch/tolgay/metal_diffusion/{galaxy_name}/halo/rockstar_dm/catalog'
        print(f"halo_finder_file_path: {halo_finder_file_path}")

        mass_of_MMH, x_MMH_center, y_MMH_center, z_MMH_center, vx_MMH_center, vy_MMH_center, vz_MMH_center, ID, DescID = functions.halo_with_most_particles_rockstar(
            rockstar_snapshot_dir=halo_finder_file_path, 
            snapshot_number=snapshot_number,
            time = time,
            hubble = hubble
        )

        # Units
        # ----
        # mass_of_MMH [M☉]
        # x_MMH_center [kpc]
        # y_MMH_center [kpc]
        # z_MMH_center [kpc]
        # vx_MMH_center [km/s]
        # vy_MMH_center [km/s]
        # vz_MMH_center [km/s]  

    elif galaxy_type == "firebox":
        print("It is a firebox galaxy, it is already centered. I am only getting the mass of the MMH")

        redshift_4_digits = '{0:.3f}'.format(redshift)
            
        galaxy_number = int(galaxy_name.replace("gal", ""))

        # Determine the fdir
        if cluster_name == "cita":
            fdir = "/fs/lustre/project/murray/scratch/lliang/FIRE_CO/FIREbox/AHF"
        elif cluster_name == "niagara":
            fdir = "/scratch/m/murray/dtolgay/firebox/FB15N1024/analysis/AHF/halo_new"
        elif cluster_name == "trillium":
            fdir = "/scratch/dtolgay/firebox/FB15N1024/analysis/AHF/halo_new"

        
        # cita clusters
        try: 

            halos = np.loadtxt(fname = f"{fdir}/FB15N1024.z{redshift_4_digits}.AHF_halos", skiprows=1) 
        
        # niagara and trillium
        except: 

            if redshift_4_digits == "0.000":
                epoch_num = "1200"
            elif redshift_4_digits == "1.000":
                epoch_num = "554"
            elif redshift_4_digits == "2.000":
                epoch_num = "344"
            elif redshift_4_digits == "3.000":
                epoch_num = "240"

            halos = np.loadtxt(fname = f"{fdir}/{epoch_num}/FB15N1024.z{redshift_4_digits}.AHF_halos", skiprows=1)         

        mass_of_MMH = halos[galaxy_number][3]        
        pass

    else:
        print("Something is wrong! MMH information couldn't be read")
        pass


    # In[6]:


    print("Starting to change origin")
    """
    Put coordinate axis on the center of the halo
    """
    if galaxy_type in ["zoom_in", "particle_split", "zoom_in_tolgay"]:
        # Chancing the position of the origin. The new origin position is the center of mass of the most massive halo
        x_gas, y_gas, z_gas = functions.change_origin(
            x=gas_particles_df["x"], 
            y=gas_particles_df["y"], 
            z=gas_particles_df["z"], 
            x_halo_center=x_MMH_center, 
            y_halo_center=y_MMH_center, 
            z_halo_center=z_MMH_center
        )

        vx_gas, vy_gas, vz_gas = functions.change_origin(
            x=gas_particles_df["vx"], 
            y=gas_particles_df["vy"], 
            z=gas_particles_df["vz"], 
            x_halo_center=vx_MMH_center, 
            y_halo_center=vy_MMH_center, 
            z_halo_center=vz_MMH_center
        )



        x_star, y_star, z_star = functions.change_origin(
            x=star_particles_df["x"], 
            y=star_particles_df["y"], 
            z=star_particles_df["z"], 
            x_halo_center=x_MMH_center, 
            y_halo_center=y_MMH_center, 
            z_halo_center=z_MMH_center
        )

        vx_star, vy_star, vz_star = functions.change_origin(
            x=star_particles_df["vx"], 
            y=star_particles_df["vy"], 
            z=star_particles_df["vz"], 
            x_halo_center=vx_MMH_center, 
            y_halo_center=vy_MMH_center, 
            z_halo_center=vz_MMH_center
        )


        # Editing the dataframes 
        gas_particles_df["x"] = x_gas
        gas_particles_df["y"] = y_gas
        gas_particles_df["z"] = z_gas
        gas_particles_df["vx"] = vx_gas
        gas_particles_df["vy"] = vy_gas
        gas_particles_df["vz"] = vz_gas

        star_particles_df["x"] = x_star
        star_particles_df["y"] = y_star
        star_particles_df["z"] = z_star
        star_particles_df["vx"] = vx_star
        star_particles_df["vy"] = vy_star
        star_particles_df["vz"] = vz_star
        
    elif galaxy_type == "firebox":
        v_average = finding_the_average_velocity_vector(
            vx=gas_particles_df["vx"],
            vy=gas_particles_df["vy"],
            vz=gas_particles_df["vz"]
        )


        # The average velocity is subtracted from the velocities of the particles individually
        vx_gas, vy_gas, vz_gas = shifting_velocities(
            vx=gas_particles_df["vx"], 
            vy=gas_particles_df["vy"], 
            vz=gas_particles_df["vz"], 
            v_average=v_average
        )    
        
        # The average velocity is subtracted from the velocities of the particles individually
        vx_star, vy_star, vz_star = shifting_velocities(
            vx=star_particles_df["vx"], 
            vy=star_particles_df["vy"], 
            vz=star_particles_df["vz"], 
            v_average=v_average
        )        
        
        
        gas_particles_df["vx"] = vx_gas
        gas_particles_df["vy"] = vy_gas
        gas_particles_df["vz"] = vz_gas  
        
        star_particles_df["vx"] = vx_star
        star_particles_df["vy"] = vy_star
        star_particles_df["vz"] = vz_star  

    print("Origin changed")


    # In[7]:


    #############################################################################################################################################
    '''
    Find the indices of the star and gas particles that are within the 20 kpc from the center of the halo 
    '''
    # Filtering the code to increase its speed
        # To increase the speed of the code I will only consider 20 kpc radius

    print(f"Considering only {R_max} kpc from the center of the MMH.")
    print(f"Before: len(gas_particles_df): {len(gas_particles_df)} --- len(star_particles_df): {len(star_particles_df)}")

        # Finding the distance of gas particles from the center of the MMH
    R_gas   = np.sqrt(np.power(gas_particles_df["x"],2) + np.power(gas_particles_df["y"],2) + np.power(gas_particles_df["z"],2))
    R_star  = np.sqrt(np.power(star_particles_df["x"],2) + np.power(star_particles_df["y"],2) + np.power(star_particles_df["z"],2))

        # Determining the indices that satisfy R_gas < R_max
    R_gas_smaller_than_Rmax_indices = np.where(R_gas < R_max)[0]

        # Determining the indices that satisfy R_star < R_max
    R_star_smaller_than_Rmax_indices = np.where(R_star < R_max)[0]

    # """
    # Filter the gas and star particles such that at the end you will end up with only the particles within the 20 kpc from the center
    # of the halo
    # """

    gas_particles_df = gas_particles_df.iloc[R_gas_smaller_than_Rmax_indices].reset_index(drop=True)
    star_particles_df = star_particles_df.iloc[R_star_smaller_than_Rmax_indices].reset_index(drop=True)

    print(f"After: len(gas_particles_df): {len(gas_particles_df)} --- len(star_particles_df): {len(star_particles_df)}")

    # Rotate the galaxy
    print(f"Rotating galaxy")
    gas_particles_df, star_particles_df = rotate_galaxy(
        gas_particles_df.copy(), 
        star_particles_df.copy()
    )

    if (redshift > 0.3):
        print(f"Since redshift={redshift} > 0.3 I will consider all gas particles within the {R_max} rather than considering only the gas particles in the disc. There is no disc either!")
        pass
    # elif (galaxy_type == "zoom_in_tolgay"):
    #     print(f"Since galaxy_type == {galaxy_type}, I will consider all gas particles within the {R_max} rather than considering only the gas particles in the disc. There is no disc either!")        
    #     pass
    elif (mass_of_MMH < 3e11):
        print(f"Halo mass is {mass_of_MMH / 1e11}e11 Msolar. I am considering all gas particles within the {R_max} kpc.")
        pass
    else:
        # Use only the galactic disc
        gas_disc_indices = np.where(abs(gas_particles_df["z"]) < 5)[0]
        star_disc_indices = np.where(abs(star_particles_df["z"]) < 5)[0]

        gas_particles_df = gas_particles_df.iloc[gas_disc_indices].reset_index(drop=True)
        star_particles_df = star_particles_df.iloc[star_disc_indices].reset_index(drop=True)

        print(f"Only the disc indices are considered")

    
    return gas_particles_df, star_particles_df