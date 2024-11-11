import sys
sys.path.append("/mnt/raid-cita/dtolgay/FIRE/post_processing_fire_outputs")
from tools import constants # type: ignore

import pandas as pd 
import numpy as np 


def read_cloudy_gas_particles(galaxy_name, galaxy_type, redshift, directory_name):
    
    cloudy_gas_particles_file_directory = f"/mnt/raid-cita/dtolgay/FIRE/post_processing_fire_outputs/skirt/runs_hden_radius/{galaxy_type}/z{redshift}/{galaxy_name}/{directory_name}"
    
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

def read_comprehensive_star_star_particles(galaxy_name, galaxy_type, redshift, directory_name):
    
    dir_path = f"/mnt/raid-cita/dtolgay/FIRE/post_processing_fire_outputs/skirt/runs_hden_radius/{galaxy_type}/z{redshift}/{galaxy_name}/{directory_name}"
    
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

def read_otherProperties(base_dir, galaxy_name, galaxy_type, redshift, directory_name, file_name):
    
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
        np.loadtxt(f'{base_dir}/{galaxy_type}/z{redshift}/{galaxy_name}/{directory_name}/{file_name}'),
        columns = column_names    
    )

    volume = data['mass'] / (data['density'] * constants.gr2Msolar) # cm^3
    h_mass = data['hden'] * volume * constants.proton_mass * constants.kg2Msolar # Msolar
    
    data['h2_mass'] = data['fh2'] * h_mass    # Msolar
    data['co_mass'] = data['fCO'] * h_mass    # Msolar
    
    
    return data