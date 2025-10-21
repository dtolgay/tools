import sys
from tools_tolgay import constants # type: ignore

import pandas as pd 
import numpy as np 


def read_cloudy_gas_particles(
        galaxy_name, 
        galaxy_type, 
        redshift, 
        directory_name,
        base_fdir = f"/mnt/raid-cita/dtolgay/FIRE/post_processing_fire_outputs/skirt/runs_hden_radius"
    ):
    
    cloudy_gas_particles_file_directory = f"{base_fdir}/{galaxy_type}/z{redshift}/{galaxy_name}/{directory_name}"
    
    # Define the column names based on your description
    gas_column_names = [
        "x",                                    # pc 
        "y",                                    # pc 
        "z",                                    # pc 
        "smoothing_length",                     # pc 
        "mass",                                 # Msolar
        "metallicity",                          # Zsolar
        "temperature",                          # Kelvin
        "vx",                                   # km/s
        "vy",                                   # km/s
        "vz",                                   # km/s
        "hden",                                 # 1/cm3
        "radius",                               # pc 
        "sfr",                                  # Msolar / year
        "turbulence",                           # km/s
        "density",                              # gr/cm3
        "mu_theoretical",                       # 1
        "average_sobolev_smoothingLength",      # pc 
        "index",                                # 1
        "isrf",                                 # G0
    ]
    
    cloudy_gas_particles = pd.read_csv(
        f"{cloudy_gas_particles_file_directory}/cloudy_gas_particles.txt",
        delim_whitespace=True,
        comment="#",
        names=gas_column_names,
    )    
    
    return cloudy_gas_particles

def read_comprehensive_gas_particles(
        galaxy_name, 
        galaxy_type, 
        redshift, 
        directory_name, 
        base_fdir = f"/mnt/raid-cita/dtolgay/FIRE/post_processing_fire_outputs/skirt/runs_hden_radius",
    ):

    dir_path = f"{base_fdir}/{galaxy_type}/z{redshift}/{galaxy_name}/{directory_name}"
    gas_columns = [
        "x",
        "y",
        "z",
        "smoothing_length",
        "mass",
        "metallicity",
        "temperature",
        "vx",
        "vy", 
        "vz",
        "hden",
        "radius",
        "star_formation_rate",
        "turbulence",
        "density",
        "mu_theoretical",
        "average_sobolev_smoothingLength", 
    ]        

    gas_data = pd.DataFrame(
        np.loadtxt(f"{dir_path}/comprehensive_gas.txt"),
        columns = gas_columns
    )

    return gas_data

def read_comprehensive_star_particles(
        galaxy_name, 
        galaxy_type, 
        redshift, 
        directory_name, 
        base_fdir = f"/mnt/raid-cita/dtolgay/FIRE/post_processing_fire_outputs/skirt/runs_hden_radius",
    ):

    dir_path = f"{base_fdir}/{galaxy_type}/z{redshift}/{galaxy_name}/{directory_name}"


    star_columns = [
        'x',            # (pc)
        'y',            # (pc)
        'z',            # (pc)
        'vx',           # (km/s)
        'vy',           # (km/s)
        'vz',           # (km/s)
        'metallicity',  # (1)
        'mass',         # (Msolar)
        'age'           # (Myr)
    ]


    star_data = pd.DataFrame(
        np.loadtxt(f"{dir_path}/comprehensive_star.txt"),
        columns = star_columns
    )    
    return star_data

def read_skirt_sed_file(galaxy_name, galaxy_type, redshift, directory_name, inclination):

    fdir = f"/mnt/raid-cita/dtolgay/FIRE/post_processing_fire_outputs/skirt/runs_hden_radius/{galaxy_type}/z{redshift}/{galaxy_name}/{directory_name}"    

    columns = [
        "wavelength",                 # micron
        "total_flux",                 # W/m2/micron
        "transparent_flux",           # W/m2/micron
        "direct_primary_flux",        # W/m2/micron
        "scattered_primary_flux",     # W/m2/micron
        "direct_secondary_flux",      # W/m2/micron
        "scattered_secondary_flux",   # W/m2/micron
        "transparent_secondary_flux"  # W/m2/micron
    ]


    sed = pd.DataFrame(
        np.loadtxt(f"{fdir}/{galaxy_name}_i{inclination}_sed.dat"),
        columns=columns
    )
    return sed

def read_otherProperties(
        galaxy_name, 
        galaxy_type, 
        redshift, 
        directory_name, 
        file_name,
        base_fdir = f"/mnt/raid-cita/dtolgay/FIRE/post_processing_fire_outputs/skirt/runs_hden_radius"
    ):
    
    column_names = [
        'x',                # (pc)
        'y',                # (pc)
        'z',                # (pc)
        'smoothing_length', # (pc)
        'mass',             # (Msolar)
        'metallicity',      # (Zsolar)
        'temperature',      # (K)
        'vx',               # (km/s)
        'vy',               # (km/s)
        'vz',               # (km/s)
        'hden',             # (cm^-3)
        'radius',           # (pc)
        'sfr',              # (Msolar/yr)
        'turbulence',       # (km/s)
        'density',          # (gr/cm^-3)
        'mu_theoretical',   # (1)
        'average_sobolev_smoothingLength',  # (pc)
        'index',            # [1]
        'isrf',             # [G0]
        'fh2',              # [1] 
        'fCO',              # [1]
    ]

    ####
    data = pd.DataFrame(
        np.loadtxt(f'{base_fdir}/{galaxy_type}/z{redshift}/{galaxy_name}/{directory_name}/{file_name}'),
        columns = column_names    
    )

    volume = data['mass'] / (data['density'] * constants.gr2Msolar) # cm^3
    h_mass = data['hden'] * volume * constants.proton_mass * constants.kg2Msolar # Msolar
    
    data['h2_mass'] = data['fh2'] * h_mass    # Msolar
    data['co_mass'] = data['fCO'] * h_mass    # Msolar
    
    
    return data

def read_interpolated_Lline(galaxy_name, galaxy_type, redshift, directory_name, file_name, base_dir="/mnt/raid-cita/dtolgay/FIRE/post_processing_fire_outputs/skirt/runs_hden_radius"):
     
    lines = [
        "L_ly_alpha",  # [erg s^-1]
        "L_h_alpha", # [erg s^-1]
        "L_h_beta", # [erg s^-1]
        "L_co_10", # [K km s^-1 pc^2] 
        "L_co_21", # [K km s^-1 pc^2] 
        "L_co_32", # [K km s^-1 pc^2] 
        "L_co_43", # [K km s^-1 pc^2] 
        "L_co_54", # [K km s^-1 pc^2] 
        "L_co_65", # [K km s^-1 pc^2] 
        "L_co_76", # [K km s^-1 pc^2] 
        "L_co_87", # [K km s^-1 pc^2] 
        "L_13co",  # [K km s^-1 pc^2] 
        "L_c2", # [erg s^-1]
        "L_o3_88", # [erg s^-1]
        "L_o3_5006", # [erg s^-1]
        "L_o3_4958", # [erg s^-1] 
    ]

    gas_column_names = [
        "x",                                    # pc 
        "y",                                    # pc 
        "z",                                    # pc 
        "smoothing_length",                     # pc 
        "mass",                                 # Msolar
        "metallicity",                          # Zsolar
        "temperature",                          # Kelvin
        "vx",                                   # km/s
        "vy",                                   # km/s
        "vz",                                   # km/s
        "hden",                                 # 1/cm3
        "radius",                               # pc 
        "sfr",                                  # Msolar / year
        "turbulence",                           # km/s
        "density",                              # gr/cm3
        "mu_theoretical",                       # 1
        "average_sobolev_smoothingLength",      # pc 
        "index",                                # 1
        "isrf",                                 # G0
    ] + lines


    fdir = f'{base_dir}/{galaxy_type}/z{redshift}/{galaxy_name}/{directory_name}/{file_name}'

    gas = pd.DataFrame(
        np.loadtxt(fname=fdir), 
        columns=gas_column_names
    )

    return gas, lines

def read_interpolated_files_usingFilePath(path, interpolation_type):
     
    if interpolation_type == "line_emissions":
        file_specific_columns = [
            "L_ly_alpha",  # [erg s^-1]
            "L_h_alpha", # [erg s^-1]
            "L_h_beta", # [erg s^-1]
            "L_co_10", # [K km s^-1 pc^2] 
            "L_co_21", # [K km s^-1 pc^2] 
            "L_co_32", # [K km s^-1 pc^2] 
            "L_co_43", # [K km s^-1 pc^2] 
            "L_co_54", # [K km s^-1 pc^2] 
            "L_co_65", # [K km s^-1 pc^2] 
            "L_co_76", # [K km s^-1 pc^2] 
            "L_co_87", # [K km s^-1 pc^2] 
            "L_13co",  # [K km s^-1 pc^2] 
            "L_c2", # [erg s^-1]
            "L_o3_88", # [erg s^-1]
            "L_o3_5006", # [erg s^-1]
            "L_o3_4958", # [erg s^-1] 
        ]
    elif interpolation_type == "abundance":
        file_specific_columns = [
            'fh2',              # [1] 
            'fCO',              # [1]
        ]
    elif interpolation_type == "temperature":
        file_specific_columns = ["Th2", "Tco", "T"]

    else:
        raise ValueError("interpolation_type must be one of 'line_emissions', 'abundance', 'temperature")

    gas_column_names = [
        "x",                                    # pc 
        "y",                                    # pc 
        "z",                                    # pc 
        "smoothing_length",                     # pc 
        "mass",                                 # Msolar
        "metallicity",                          # Zsolar
        "temperature",                          # Kelvin
        "vx",                                   # km/s
        "vy",                                   # km/s
        "vz",                                   # km/s
        "hden",                                 # 1/cm3
        "radius",                               # pc 
        "sfr",                                  # Msolar / year
        "turbulence",                           # km/s
        "density",                              # gr/cm3
        "mu_theoretical",                       # 1
        "average_sobolev_smoothingLength",      # pc 
        "index",                                # 1
        "isrf",                                 # G0
    ] + file_specific_columns


    gas = pd.DataFrame(
        np.loadtxt(fname=path), 
        columns=gas_column_names
    )

    return gas, file_specific_columns


def read_interpolated_files_usingFilePath2(path, interpolation_type):
    """
        base_fdir = "/mnt/raid-cita/dtolgay/FIRE/post_processing_fire_outputs/skirt/runs_hden_radius/"
        path = f"{base_fdir}/{galaxy_type}/z{redshift}/{galaxy_name}/{directory_name}"
    """


    if interpolation_type == "line_emissions":
        file_specific_columns = [
            "L_ly_alpha",  # [erg s^-1]
            "L_h_alpha", # [erg s^-1]
            "L_h_beta", # [erg s^-1]
            "L_co_10", # [K km s^-1 pc^2] 
            "L_co_21", # [K km s^-1 pc^2] 
            "L_co_32", # [K km s^-1 pc^2] 
            "L_co_43", # [K km s^-1 pc^2] 
            "L_co_54", # [K km s^-1 pc^2] 
            "L_co_65", # [K km s^-1 pc^2] 
            "L_co_76", # [K km s^-1 pc^2] 
            "L_co_87", # [K km s^-1 pc^2] 
            "L_13co",  # [K km s^-1 pc^2] 
            "L_c2", # [erg s^-1]
            "L_o3_88", # [erg s^-1]
            "L_o3_5006", # [erg s^-1]
            "L_o3_4958", # [erg s^-1] 
        ]
    elif interpolation_type == "abundance":
        file_specific_columns = [
            'fh2',              # [1] 
            'fCO',              # [1]
            'fCii',             # [1]
            'fOiii',            # [1]
            'mco_over_mh2',     # [1]
            'visual_extinction_point', # [mag]
            'visual_extinction_extended', # [mag]
        ]
    elif interpolation_type == "temperature":
        file_specific_columns = ["Th2", "Tco", "T", "Tcii", "Toiii"]

    else:
        raise ValueError("interpolation_type must be one of 'line_emissions', 'abundance', 'temperature")

    gas_column_names = [
        "x",                                    # pc 
        "y",                                    # pc 
        "z",                                    # pc 
        "smoothing_length",                     # pc 
        "mass",                                 # Msolar
        "metallicity",                          # Zsolar
        "temperature",                          # Kelvin
        "vx",                                   # km/s
        "vy",                                   # km/s
        "vz",                                   # km/s
        "hden",                                 # 1/cm3
        "radius",                               # pc 
        "sfr",                                  # Msolar / year
        "turbulence",                           # km/s
        "density",                              # gr/cm3
        "mu_theoretical",                       # 1
        "average_sobolev_smoothingLength",      # pc 
        "index",                                # 1
        "isrf",                                 # G0
    ] + file_specific_columns


    gas = pd.DataFrame(
        np.loadtxt(fname=path), 
        columns=gas_column_names
    )

    return gas, file_specific_columns

def read_all_interpolated_files_usingFilePath(path: str, files_info: dict, verbose: bool = False):

    """
    base_fdir = "/mnt/raid-cita/dtolgay/FIRE/post_processing_fire_outputs/skirt/runs_hden_radius/"
    path = f"{base_fdir}/{galaxy_type}/z{redshift}/{galaxy_name}/{directory_name}"

    files_info = {
        "abundance": {
            "interpolation_type": "abundance",
            "file_name": "abundance_RBFInterpolator_smoothingLength.txt", 
        },  
        "temperature": {
            "interpolation_type": "temperature",
            "file_name": "temperature_RBFInterpolator_smoothingLength.txt",     
        },
        "line_emissions": {
            "interpolation_type": "line_emissions",
            "file_name": "line_emissions_RBFInterpolator_smoothingLength.txt",          
        }

    }    
    """

    # Create empty DataFrame to store all data
    data = pd.DataFrame()

    for counter, key in enumerate(files_info.keys()):
        file_name = files_info[key]['file_name']
        interpolation_type = files_info[key]['interpolation_type']
        path2dir = f"{path}/{file_name}"

        if verbose:
            print(f"Reading file: {path2dir}")

        # Read the file 
        gas, file_specific_columns = read_interpolated_files_usingFilePath2(path2dir, interpolation_type)
        if counter == 0:
            data = gas.copy()
        else:
            gas = gas[["index"] + file_specific_columns]
            data = pd.merge(data, gas, on="index", how="inner", validate="1:1") 

    return data


def read_semianalytical_file(galaxy_name, galaxy_type, redshift, directory_name, file_name):

    # Read semi_analytical_average_sobolev_smoothingLength.txt
    semi_analytical_column_names = [
        "x",
        "y",
        "z",
        "smoothing_length",
        "mass", 
        "metallicity",
        "temperature",
        "vx",
        "vy",
        "vz",
        "hden",
        "radius",
        "sfr",
        "turbulence",
        "density",
        "mu_theoretical",
        "average_sobolev_smoothingLength",
        "index",
        "isrf",
        "h2_mass",
        "Xco",
        "L_co_10"
    ]    
    
    base_dir = "/mnt/raid-cita/dtolgay/FIRE/post_processing_fire_outputs/skirt/runs_hden_radius"
    fdir = f'{base_dir}/{galaxy_type}/z{redshift}/{galaxy_name}/{directory_name}/{file_name}'  
    
    semi_analytical = pd.DataFrame(
        np.loadtxt(fdir),
        columns=semi_analytical_column_names
    )    

    return semi_analytical

def read_semianalytical_file_2(galaxy_name, galaxy_type, redshift, directory_name, file_name):

    # Read semi_analytical_average_sobolev_smoothingLength.txt

    semi_analytical_column_names = [
        "index",                # [1] 
        "fh2",                  # [1]
        "Mh2",                  # [Msolar]
        "dust_optical_depth",   # [1]
        "alpha_co",             # [Msolar / (K-km s^-1 pc^2)]
        "L_co"                  # [K-km s^-1 pc^2]
    ]        

    base_dir = "/mnt/raid-cita/dtolgay/FIRE/post_processing_fire_outputs/skirt/runs_hden_radius"
    fdir = f'{base_dir}/{galaxy_type}/z{redshift}/{galaxy_name}/{directory_name}/{file_name}'  
    
    semi_analytical = pd.DataFrame(
        np.loadtxt(fdir),
        columns=semi_analytical_column_names
    )    

    return semi_analytical

def read_galactic_properties(file_path, file_name):
    # Read the DataFrame, skipping the header lines
    galaxies = pd.read_csv(
        f"{file_path}/{file_name}", 
        sep=',', 
    )


    galaxies['alpha_co_cloudy'] = galaxies['h2_mass_cloudy'] / galaxies['L_co_10'] 
    galaxies['X_co_cloudy'] = galaxies['alpha_co_cloudy'] * 6.3e19 


    galaxies['alpha_co_semi_analytical'] = galaxies['h2_mass_semi_analytical'] / galaxies['L_co_10']
    galaxies['X_co_semi_analytical'] = galaxies['alpha_co_semi_analytical'] * 6.3e19 

    return galaxies 

def read_csv_files(galaxy_name, galaxy_type, redshift, directory_name, file_name, base_fdir):

    # Get the path to file 
    fdir = f'{base_fdir}/{galaxy_type}/z{redshift}/{galaxy_name}/{directory_name}/{file_name}'

    # Read the csv file 
    data = pd.read_csv(fdir)
    return data

def create_Lc2_sfr_relations():
    """
    Values are taken from: https://ui.adsabs.harvard.edu/abs/2024MNRAS.528..499L/abstract 
    Liang, Murray, Tolgay
    """

    # deLooze+ 2011
    A_delooze = 7.31 
    B_delooze = 0.93
    one_sigma_scatter_delooze = 0.26
    delooze = pd.DataFrame()
    delooze['log_sfr'] = np.linspace(np.log10(0.02), np.log10(88), num=1000)
    delooze['log_L_c2'] = A_delooze + delooze['log_sfr']*B_delooze
    delooze['log_L_c2_upper'] = delooze['log_L_c2'] + one_sigma_scatter_delooze
    delooze['log_L_c2_lower'] = delooze['log_L_c2'] - one_sigma_scatter_delooze

    # herrera+ 2015
    A_herrera = 7.63
    B_herrera = 0.97
    one_sigma_scatter_herrera = 0.21
    herrera = pd.DataFrame()
    herrera['log_sfr'] = np.linspace(np.log10(1e-3), np.log10(9.6), num=1000)
    herrera['log_L_c2'] = A_herrera + herrera['log_sfr']*B_herrera
    herrera['log_L_c2_upper'] = herrera['log_L_c2'] + one_sigma_scatter_herrera
    herrera['log_L_c2_lower'] = herrera['log_L_c2'] - one_sigma_scatter_herrera

    return herrera, delooze


################################

def _prep_round_unique(df, ndigits=0, x='x', y='y', z='z'):
    """Round coords and ensure one row per rounded triplet by
    keeping the row closest to the rounded point."""
    out = df.copy()

    for c in (x, y, z):
        out[f'{c}_r'] = out[c].round(ndigits)

    # distance from original point to its rounded point
    dr = np.sqrt((out[x] - out[f'{x}_r'])**2 +
                 (out[y] - out[f'{y}_r'])**2 +
                 (out[z] - out[f'{z}_r'])**2)
    out['_dr'] = dr

    # keep the row with minimal rounding distance for each rounded cell
    out = (out.sort_values('_dr')
               .drop_duplicates([f'{x}_r', f'{y}_r', f'{z}_r'], keep='first')
               .drop(columns=['_dr']))

    return out

def merge_on_rounded_coords(df_left, df_right, ndigits=0, suffixes=('_gas', '_other')):
    L = _prep_round_unique(df_left, ndigits=ndigits)
    R = _prep_round_unique(df_right, ndigits=ndigits)

    merged = pd.merge(
        L, R,
        on=['x_r', 'y_r', 'z_r'],
        how='inner',
        validate='one_to_one',
        suffixes=suffixes
    )

    # Change the column names back to original
    merged = merged.rename(columns={
        'x_gas': 'x',
        'y_gas': 'y',
        'z_gas': 'z'
    })

    # Drop the duplicate original coordinate columns from the right DataFrame
    merged = merged.drop(columns=['x_other', 'y_other', 'z_other'])
    # Drop the _r columns 
    merged = merged.drop(columns=['x_r', 'y_r', 'z_r'])

    return merged

################################
