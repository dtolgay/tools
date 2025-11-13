from astropy.io import fits
import astropy.units as units
from matplotlib import pyplot as plt
import matplotlib
import numpy as np 
import pandas as pd 
import sys
from tools_tolgay import readsnap, functions, readsnap_FIREBox, constants # type: ignore
import h5py

def sfr_calculator(star_df: pd.DataFrame, within_how_many_Myr:float):
    # Calculate star formation happened in the last 10 Myr
    indices = np.where(star_df["age"] <= within_how_many_Myr)[0] 
    sfr_star = np.sum(star_df.iloc[indices]["mass"]) / (within_how_many_Myr * 1e6)  # Msolar / year
    return sfr_star

def halo_mass_calculator(galaxy_name, galaxy_type, redshift):

    print(f" ---------- I am in the halo_mass_calculator function for galaxy: {galaxy_name}. ---------- ") # TODO: Delete 
    
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
            halo_finder_file_path = f"/fs/lustre/project/murray/scratch/tolgay/metal_diffusion/{galaxy_name}/rockstar_dm/catalog"
            snap_dir_file_path = f'/fs/lustre/project/murray/scratch/tolgay/metal_diffusion/{galaxy_name}/output'

        elif galaxy_name[0:3] == "m11":
            halo_finder_file_path = f'/fs/lustre/project/murray/scratch/tolgay/metal_diffusion/{galaxy_name}/halo/rockstar_dm/catalog'
            snap_dir_file_path = f'/fs/lustre/project/murray/scratch/tolgay/metal_diffusion/{galaxy_name}/output'
        
        try:
            header_info = readsnap.readsnap(snap_dir_file_path, snapshot_number, 0, header_only=1)
        
            # TODO: Delete 
            print(f"header_info keys: {header_info.keys()}")

            hubble      = header_info['hubble']
            time      = header_info['time']


            mass_of_MMH, x_MMH_center, y_MMH_center, z_MMH_center, vx_MMH_center, vy_MMH_center, vz_MMH_center, ID, DescID = functions.halo_with_most_particles_rockstar(
                rockstar_snapshot_dir=halo_finder_file_path, 
                snapshot_number=snapshot_number,
                time = time,
                hubble = hubble
            )    

            halo_mass = mass_of_MMH

        except Exception as e:
            print(f"Exception occured in the halo mass calculation for zoom-in galaxy: {e}")
            print(f"Because Fei's snapshots are deleted, I couldn't read the header info. I will use the previously calculated halo masses.")
            mapping = {"0.0": "0", "1.0": "1", "2.0": "2", "3.0": "3"}
            redshift_temp = mapping.get(redshift, redshift)

            temp_read_file = f"/mnt/raid-cita/dtolgay/FIRE/post_processing_fire_outputs/skirt/python_files/analyze_hden_metallicity_turbulence_isrf_radius/data/2025_11_08/galactic_properties_smoothingLength_RBFInterpolator_z{redshift_temp}_using_luminosity_per_mass_values_voronoi_1e5.csv"
            df_temp = pd.read_csv(temp_read_file)
            halo_mass = df_temp[df_temp["name"] == galaxy_name]["halo_mass"].values[0]
            print(f"Read halo mass from the file: {halo_mass:.2e} Msun.")
        
    
    elif galaxy_type == "firebox":
        snap_dir_file_path_base = "/fs/lustre/project/murray/scratch/tolgay/firebox/FB15N1024/seperated_galaxies"
        if redshift == "0.0":
            snapshot_number = 1200     # z = 0.0
            # snap_dir_file_path = '/fs/lustre/project/murray/scratch/lliang/FIRE_CO/FIREbox'
            # snap_dir_file_path = '/fs/lustre/project/murray/scratch/tolgay/firebox/FB15N1024/seperated_galaxies'
            snap_dir_file_path = f'{snap_dir_file_path_base}/z0.0'
        elif redshift == "0.5":
            print(f"Exiting... Currently there are no z=0.5 galaxies... {redshift}")
            sys.exit(2)                
        elif redshift == "1.0":
            snapshot_number = 554     # z = 1.0
            snap_dir_file_path = f'{snap_dir_file_path_base}/z1.0'
        elif redshift == "2.0":
            snapshot_number = 344     # z = 2.0
            snap_dir_file_path = f'{snap_dir_file_path_base}/z2.0'
        elif redshift == "3.0":
            snapshot_number = 240     # z = 3.0
            snap_dir_file_path = f'{snap_dir_file_path_base}/z3.0'
        else:
            print(f"Exiting... Redshift is wrong. The given galaxy type is {redshift}")
            sys.exit(2)         
        
        
        galaxy_number = int(galaxy_name.replace("gal", ""))
        

        # Read halos_used.csv
        all_halos = pd.read_csv(f"{snap_dir_file_path}/halos_used.csv")
        # Find the halo mass for the given galaxy number
        condition = all_halos['ID_new'] == galaxy_number
        # halo mass is Mvir 
        halo_mass_ = all_halos[condition]['Mvir'].values[0] # [Msolar/h]

        try: 
            ## I used AHF so I need to divide it to hubble to get the mass in Msolar unit
            # Read the header of the hdf5 file. Files are like gal_990.hdf5
            header_info = h5py.File(f"{snap_dir_file_path}/gal_{galaxy_number}.hdf5", 'r')['header']
            hubble = header_info['hubble'][()]
        except Exception as e:
            print(f"Exception occurred while finding the hubble constant in the file. I am using h=0.6774)")
            hubble = 0.6774 

        halo_mass = halo_mass_ / hubble # [Msolar]

        
    elif galaxy_type == "particle_split":
        # Read halos for the particle splitted FIREBox simualation
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

def mass_average(property_array, mass):
    return sum(property_array * mass) / sum(mass)

def calculate_coordinates_from_fits_file(hdul):
    
    # Read the header and the data 
    fits_data = hdul[0].data    
    header = hdul[0].header
    
    # pprint(header)
    
    ## Find the coordinates 
    # Extract spatial axis parameters from the header
    crpix1, crval1, cdelt1 = header['CRPIX1'], header['CRVAL1'], header["CDELT1"]  # X-axis
    crpix2, crval2, cdelt2 = header['CRPIX2'], header['CRVAL2'], header["CDELT2"]  # Y-axis
    
    # Generate X and Y coordinates
    x_coords = (np.arange(fits_data.shape[1]) + 1 - crpix1) * cdelt1 + crval1
    y_coords = (np.arange(fits_data.shape[2]) + 1 - crpix2) * cdelt2 + crval2
    #### Convert coordinates to pc
    # arcseconds -> radians 
    arcseconds2radians = 4.84814e-6
    x_coords *= arcseconds2radians
    y_coords *= arcseconds2radians
    # radians -> pc 
    distance = 10 # Mpc
    Mpc2parsec = 1e6 # Mpc -> pc
    x_coords *= distance * Mpc2parsec  # pc
    y_coords *= distance * Mpc2parsec  # pc
    
    # Meshgrid to calculate the coordinate of each pixel.
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    X = X.ravel() # pc
    Y = Y.ravel() # pc
    
    return X, Y

def seperate_into_annuli(data):
    
    Rgal = np.sqrt(data['x']**2 + data['y']**2)
    bins = np.linspace(0, 20e3, num=101) # The boundaries for 100 annuli's are created
    
    # Calculate digitized bins 
    digitized_indices_for_gas_bins = np.digitize(Rgal, bins)
    
    # Calculate center Rgal for every annuli
    annuli_center_Rgas = (bins[:-1] + bins[1:])/2  
    
    return digitized_indices_for_gas_bins, annuli_center_Rgas 

def half_light_radius(band, i0_total_fits_file_path, skirt_wavelengths_file_path, plot=False):
    
    # Read the fits emission file 
    hdul = fits.open(i0_total_fits_file_path)
    fits_data = hdul[0].data
    
    # Read the wavelengths file 
    wavelengths_unfiltered = np.loadtxt(skirt_wavelengths_file_path)[:,0] * units.micron
    
    if plot:
        plt.hist(np.log10(wavelengths_unfiltered.to(units.nm).value), bins=100)
        plt.axvline(np.log10(band['max'].value), color='red', linestyle='--', label='Max')
        plt.axvline(np.log10(band['min'].value), color='orange', linestyle='--', label='Min')
        plt.legend()
        plt.show()

    # Find the indices of the wavelengths inside the band 
    condition1 = wavelengths_unfiltered <= band['max']
    condition2 = wavelengths_unfiltered >= band['min']
    indices = np.where(condition1 & condition2)[0]
    # Filter the wavelengths 
    wavelengths = wavelengths_unfiltered[indices].value
    
    # Filter the fits file 
    fits_data = fits_data[indices,:,:]
    
    # Integrate the emission data 
    integrated_data = np.trapz(fits_data, wavelengths, axis=0) # W/m²/arcsec²

    # Create df
    X, Y = calculate_coordinates_from_fits_file(hdul = hdul)
    size_of_a_rectangle = (Y[1] - Y[0])**2 * constants.pc2m**2 # m2 -- The size is the same 
    data = pd.DataFrame(np.array([X,Y,integrated_data.ravel()]).T, columns=["x", "y", "flux"]) 
    data["luminosity"] = data["flux"] * size_of_a_rectangle # Watts
    # Luminosity bulunmadan da sadece flux ile half light radius bulunulabilirdi. Her pixel icin alan sabit oldugunda, flux 
    # sabit bir sayi ile carpiliyor. Dolayisiyla, flux veya luminosity kullanildiginda degisen bir sey olmuyor.
    
    if plot:
        plt.hist2d(data['x'], data['y'], bins=[1024, 1024], weights=np.log10(data['flux']), cmap='inferno')
        plt.colorbar(label='(W/m²/arcsec²)')
        plt.xlabel('X (pc)')
        plt.ylabel('Y (pc)')
        plt.show()    
    
    
    # Data is created. Now seperate data into annuli
    digitized_indices_for_gas_bins, annuli_center_Rgas = seperate_into_annuli(data = data)
    
    # Calculate flux in each annuli
    luminosity_in_annuli = []
    for annuli_number in range(min(digitized_indices_for_gas_bins), max(digitized_indices_for_gas_bins)):
        indices = np.where(digitized_indices_for_gas_bins == annuli_number)
        luminosity_in_annuli.append(sum(data.iloc[indices]['luminosity']))
    
    if plot: 
        plt.scatter(annuli_center_Rgas, luminosity_in_annuli)
        plt.xlabel("Rgal [pc]")
        plt.ylabel("Luminosity [Watts]")
        plt.show()
        
    # Find half light radius
    total_luminosity = sum(data["luminosity"])
    integrated_luminosity_upto_certain_r = 0
    for i in range(len(luminosity_in_annuli)):
        integrated_luminosity_upto_certain_r += luminosity_in_annuli[i]
        if integrated_luminosity_upto_certain_r > total_luminosity/2:
            R_half_life = annuli_center_Rgas[i] # [pc] half light radius is where the half of the total luminosity is reached.
            break
    
    return R_half_life 

def filter_particles(particles_df, condition):
    
    r = np.sqrt(particles_df['x']**2 + particles_df['y']**2)
    conditions = (r < condition['r_max']) & (abs(particles_df['z']) < condition['z_max'])
    
    return particles_df[conditions].copy()

def disk_pressure_calculator(gas_particles, star_particles, filtering_condition, R_half_light, is_plot=False):
        
    # Filter the particles
    gas_particles = filter_particles(particles_df = gas_particles, condition = filtering_condition)
    star_particles = filter_particles(particles_df = star_particles, condition = filtering_condition)
    
    
    if is_plot:
        particles = star_particles.copy()
        Rmax = 20e3
        plt.hist2d(
            x = particles['x'],
            y = particles['y'],
            bins = 500,
            weights = particles['mass'],
            norm=matplotlib.colors.LogNorm(),        
            range = [[-Rmax, Rmax], [-Rmax, Rmax]]
        )
        plt.xlabel('x [pc]')
        plt.ylabel('y [pc]')
        plt.show()

        plt.hist2d(
            x = particles['x'],
            y = particles['z'],
            bins = 500,
            weights = particles['mass'],
            norm=matplotlib.colors.LogNorm(),        
            range = [[-Rmax, Rmax], [-Rmax, Rmax]]
        )    
        plt.xlabel('x [pc]')
        plt.ylabel('z [pc]')
        plt.show()
    
    
    #### Calculate the pressure 
    area = np.pi * filtering_condition['r_max']**2 # pc2
    # volume = area * filtering_condition['z_max'] #pc3 # Not used now. Ellison's method is used
    ## Column Density calculation
    sigma_gas = sum(gas_particles['mass']) / area # Msolar / pc2
    ## Star Density calculation
    # rho_star_dtolgay = sum(star_particles['mass']) / volume # Msolar / pc3 # Not used now. Ellison's method is used
    sigma_star = sum(star_particles['mass']/ area) # Msolar / pc2
 
    # Rstar eqn. 4. in Ellison
    R_star = R_half_light / 1.68 # [pc]
    rho_star = sigma_star / (0.54 * R_star) # [Msolar / pc2] eqn. 3 in Ellison
    
    ### Disk pressure calculation
    ## Convering to SI units 
    sigma_gas_unit_converted = sigma_gas * constants.Msolar2kg / constants.pc2m**2
    rho_star_unit_converted = rho_star * constants.Msolar2kg / constants.pc2m**3
    ## Gas pressure 
    Pgas = np.pi * constants.gravitational_constant * sigma_gas_unit_converted**2 / 2 # [N/m2 = J/m3]
    Pgas *= 1 / constants.kb / constants.m2cm**3 # [K/cm3]
    
    ## Star pressure 
    # Assuming constant σ_z 
    velocity_dispersion_in_z_direction = 11*1e3 # 11 km/s -- Ellison 2024
    Pstar = sigma_gas_unit_converted * np.sqrt(2 * constants.gravitational_constant * rho_star_unit_converted) * velocity_dispersion_in_z_direction # [N/m2]
    Pstar *= 1 / constants.kb / constants.m2cm**3 # [K/cm3] 
    
    ## Total pressure 
    Ptotal = Pgas + Pstar # [K/cm3]
    
    return Pgas, Pstar, Ptotal

def half_mass_radius(particles_df):
    # Calculate the total mass 
    total_mass = sum(particles_df['mass'])
                                  
    # Calculate the distance that corresponds to the half of the total mass
    digitized_bins, distance_to_annuli_center = seperate_into_annuli(data = particles_df)
    mass_in_annuli = []
    for annuli_number in range(min(digitized_bins), max(digitized_bins)):
        indices = np.where(digitized_bins == annuli_number)
        mass_in_annuli.append(sum(particles_df.iloc[indices]['mass']))

    for i in range(len(mass_in_annuli)):
        if sum(mass_in_annuli[:i]) > total_mass/2:
            R_half_mass = distance_to_annuli_center[i]
            break

    return R_half_mass

