# Import modules 
from astropy.io import fits
from astropy.table import Table
import numpy as np
import pandas as pd
from tools_tolgay import constants

################################################################################################################################################
# Constants 
Mpc2meter = 3.086e+22  # meter / Mpc
watt2ergs_per_second = 1e7 # erg s^-1 / watt
Lsolar2erg_per_s = 3.826e33 # erg s^-1 / Lsolar 


################################################################################################################################################

def converting_log12_plus_log_O_over_H_to_solar_units(X):
    """
    This function is used to convert the metallicity in 12 + log(O/H) units to solar units.
    
    12 + log([O/H]_solar) = 8.69 is the solar metallicity. (Asplund et al. 2009)
    Question is if: 
    12 + log([O/H]_galaxy) = X
    What is the [O/H]_galaxy in solar units?

    (O/H) / (O/H)_Zsolar = 10**(X - 12) / 10**(-3.31) = 10**(X + 3.31 - 12) = 10**(X - 8.69)
    
    """

    X_solar = 10**(X - 8.69) # X is in 12 + log(O/H) units

    return X_solar


################################################################################################################################################

def laura_eyeballing_data_df_creator():

    print("I am in the function laura_eyeballing_data_df_creator")

    """This function is used to read the XCOLDGASS Survey results 
    
    Arguments:
    ----------
    

    Returns:
    ----------
    laura_eyeballing_data_df: pandas data_frame
        Return the eyeballed values from the Laura's paper. 

    
    References:
    ----------
    arXiv:2001.08197
    Reproducing the CO-to-H2 conversion factor in cosmological simulations of Milky Way-mass galaxies
    Laura Keating
    """ 
    
    Lco_m12i_Lsm = 1e10         # [K-km s^-1 pc^2] 
    M_H2_m12i_Lsm = 2e9         # [M☉]
    
    alpha_CO_m12i_Lsm = M_H2_m12i_Lsm / Lco_m12i_Lsm  # [M☉ (K km s^-1 pc^2)^-1]
    X_CO_m12i_Lsm = alpha_CO_m12i_Lsm * 6.3e19        # [cm^-2 (K km s^-1)^-1]

    SFR_m12i_Lsm = 9.210632     # [M☉ year^-1]

    data = {
        "Lco": Lco_m12i_Lsm,
        "Xco": X_CO_m12i_Lsm,
        "Mh2": M_H2_m12i_Lsm,
        "SFR": SFR_m12i_Lsm 
    }

    laura_eyeballing_data_df = pd.DataFrame(data, index=pd.Index([0])) 

    return laura_eyeballing_data_df

################################################################################################################################################

def XCold_Gass_data_reading(filedir):

    print("I am in the function XCold_Gass_data_reading")

    """This function is used to read the XCOLDGASS Survey results 
    
    Arguments:
    ----------
    file_dir: string
        Indicator of the file directory to read the star particle information

    Returns:
    ----------
    ID_XCOLDGASS: numpy.ndarray 
        ID of the galaxies. These are numbers assigned to galaxies.
        Galaxies with six-digit IDs are part of COLD GASS-low.
    
    LCO_corrected_XCOLDGASS: numpy.ndarray - float
        CO(1-0) luminosity of the galaxies. Aperture correction is applied. 
        Unit: K-km s^-1 pc^2

    XCO_XCOLDGASS: numpy.ndarray - float
        X_CO value of the galaxies.
        Unit: cm^-2 (K-km s^-1)^-1

    MH2_XCOLDGASS: numpy.ndarray - float
        Total molecular gas mass of the galaxies.
        Unit: M⊙

    SFR_XCOLDGASS: numpy.ndarray - float
        Total molecular gas mass of the galaxies.
        Unit: M⊙

    References: 
    -----------
    xCOLD GASS The Complete IRAM 30 m Legacy Survey of Molecular Gas for Galaxy Evolution Studies

    """ 

    with fits.open(filedir) as hdu:
        data_table_XCOLDGASS = Table(hdu[1].data)
        
        
    # GASS catalog ID
    ID_XCOLDGASS                = data_table_XCOLDGASS['ID'].data

    # CO(1-0) line luminosity, aperture corrected [K km/s pc^2]
    LCO_uncorrected             = data_table_XCOLDGASS['LCO'].data  # K-km s^-1 pc^2
    LCO_corrected_XCOLDGASS     = data_table_XCOLDGASS['LCO_COR'].data  # K-km s^-1 pc^2

    # Other CO lines 
    LCO_21                      = data_table_XCOLDGASS['LCO21'].data  # K-km s^-1 pc^2

    #  Constant Galactic CO-to-H2 conversion factor  
    XCO_XCOLDGASS               = data_table_XCOLDGASS['XCO_A17'].data  # 1e20 cm^-2 (K-km s^-1)^-1 This is the Accurso+2017 conversion value
    XCO_XCOLDGASS               = XCO_XCOLDGASS*1e20                    # cm^-2 (K-km s^-1)^-1

    # Print the contents of data_table_XCOLDGASS
    # from pprint import pprint
    # pprint(data_table_XCOLDGASS.columns)

    LOGMH2_XCOLDGASS            = data_table_XCOLDGASS['LOGMH2'].data
    MH2_XCOLDGASS               = 10**LOGMH2_XCOLDGASS                  # M⊙
    MH2_upperLimits             = 10**data_table_XCOLDGASS['LIM_LOGMH2'].data 

    alphaCO2Xco                 = 6.3e19 # cm^-2 (K-km s^-1)^-1
    alphaCO                     = XCO_XCOLDGASS / alphaCO2Xco # M⊙ (K-km s^-1 pc^2)^-1
    Mh2_Xco                     = LCO_corrected_XCOLDGASS * alphaCO  # M⊙, total molecular gas mass calculated by using the Xco value. This is not used in the paper but it is good to have it.

    ratio_mh2 = Mh2_Xco / MH2_XCOLDGASS # Checking the Mh2_Xco and Mh2_XCOLDGASS values. They should be equal.

    # Molecular gas mass fraction 
    molecular_gas_mass_fraction = 10**data_table_XCOLDGASS['LOGMH2MS'].data # M_H2 / M_star, where M_star is the stellar mass of the galaxy.

    # SFR from WISE + GALEX when detected in both data sets
    LOGSFR_XCOLDGASS            = data_table_XCOLDGASS['LOGSFR_BEST'].data
    SFR_XCOLDGASS               = 10**LOGSFR_XCOLDGASS                  # M⊙/year

    Lir = 10**data_table_XCOLDGASS['LOGLIR_IRAS'].data # LIR in Lsolar units

    # half_light_radius
    R50 = data_table_XCOLDGASS['R50KPC'].data # kpc    

    # Metallicity 
    metallicity_in_solar_units_XCOLDGASS = converting_log12_plus_log_O_over_H_to_solar_units(X = data_table_XCOLDGASS['Z_MZR'].data) # 12 + log(O/H) = Z_MZR
    Z_PP04_N2 = data_table_XCOLDGASS['Z_PP04_N2'].data
    Z_MZR = data_table_XCOLDGASS['Z_MZR'].data
    Z_PP04_O3N2 = data_table_XCOLDGASS['Z_PP04_O3N2'].data

    # Detection or upper limit 
    FLAG_CO = data_table_XCOLDGASS['FLAG_CO'].data 
    FLAG_CO21 = data_table_XCOLDGASS['FLAG_CO21'].data

    # Stellar mass 
    Mstar = 10**data_table_XCOLDGASS['LOGMSTAR'].data
    stellar_mass_surface_density = 10**data_table_XCOLDGASS['LOGMUST'].data # M*/(2*pi*r50_z^2) # Msolar/kpc^2


    # NUV-r color 
    NUV_r_color = data_table_XCOLDGASS['NUVR'].data

    # Redshift 
    redshift_SDSS = data_table_XCOLDGASS['Z_SDSS'].data

    data = {
        # Galaxy ID (integer values, safe to store as 64-bit integers)
        "Id": np.array(ID_XCOLDGASS, dtype=np.int64),

        # CO(1-0) luminosity (aperture-corrected), units: K km s^-1 pc^2
        "Lco": np.array(LCO_corrected_XCOLDGASS, dtype=np.float64),

        # CO(1-0) luminosity (uncorrected), same units
        "Lco_uncorrected": np.array(LCO_uncorrected, dtype=np.float64),

        # CO(2-1) luminosity, units: K km s^-1 pc^2
        "Lco_21": np.array(LCO_21, dtype=np.float64),

        # CO-to-H2 conversion factor X_CO, units: cm^-2 (K km s^-1)^-1
        "Xco": np.array(XCO_XCOLDGASS, dtype=np.float64),

        # α_CO derived from X_CO, units: M☉ (K km s^-1 pc^2)^-1
        "alphaCO": np.array(alphaCO, dtype=np.float64),

        # Molecular hydrogen mass from the catalog, units: M☉
        "Mh2": np.array(MH2_XCOLDGASS, dtype=np.float64),

        # Upper limits on molecular hydrogen mass, units: M☉
        "Mh2_upperLimits": np.array(MH2_upperLimits, dtype=np.float64),

        # H₂ mass computed using X_CO (not used in the paper, but useful for cross-check)
        "Mh2_Xco": np.array(Mh2_Xco, dtype=np.float64),

        # Ratio between Mh2_Xco and the catalog mass (for internal consistency check)
        "ratio_mh2": np.array(ratio_mh2, dtype=np.float64),

        # Star formation rate, units: M☉ yr^-1
        "SFR": np.array(SFR_XCOLDGASS, dtype=np.float64),

        # Total infrared luminosity (L_IR), units: L☉
        "Lir": np.array(Lir, dtype=np.float64),

        # NUV−r color (dimensionless)
        "NUV_r_color": np.array(NUV_r_color, dtype=np.float64),

        # Gas-phase metallicities based on different calibrations (12 + log(O/H))
        "Z_PP04_N2": np.array(Z_PP04_N2, dtype=np.float64),
        "Z_MZR": np.array(Z_MZR, dtype=np.float64),
        "Z_PP04_O3N2": np.array(Z_PP04_O3N2, dtype=np.float64),

        # Metallicity in solar units, converted from log(O/H)
        "metallicity": np.array(metallicity_in_solar_units_XCOLDGASS, dtype=np.float64),

        # Detection flags: 1 = detection, 2 = non-detection (use upper limit)
        "FLAG_CO": np.array(FLAG_CO, dtype=np.int16),
        "FLAG_CO21": np.array(FLAG_CO21, dtype=np.int16),

        # Stellar mass, units: M☉
        "Mstar": np.array(Mstar, dtype=np.float64),

        # Stellar mass surface density, units: M☉ kpc^-2
        "stellar_mass_surface_density": np.array(stellar_mass_surface_density, dtype=np.float64),

        # Half-light radius, units: kpc
        "R50": np.array(R50, dtype=np.float64),

        # Spectroscopic redshift from SDSS
        "redshift_SDSS": np.array(redshift_SDSS, dtype=np.float64),
    }

    # Finally, create a pandas DataFrame from the cleaned, native-endian data
    XCOLDGASS = pd.DataFrame(data)

    XCOLDGASS['FLAG_CO'] = np.array(XCOLDGASS['FLAG_CO'], dtype=np.int16)
    XCOLDGASS['FLAG_CO21'] = np.array(XCOLDGASS['FLAG_CO21'], dtype=np.int16)
    XCOLDGASS['Id'] = np.array(XCOLDGASS['Id'], dtype=np.int64)


    # For FLAG_CO == 2 change Mh2 with Mh2_upperLimits
    XCOLDGASS.loc[XCOLDGASS['FLAG_CO'] == 2, 'Mh2'] = XCOLDGASS['Mh2_upperLimits']

    # Drop Mh2_upperLimits column as it is not needed anymore
    XCOLDGASS.drop(columns=['Mh2_upperLimits'], inplace=True)


    XCOLDGASS['alphaCO_fromData'] = XCOLDGASS['Mh2'] / XCOLDGASS['Lco'] # M⊙ (K-km s^-1 pc^2)^-1, alpha_CO calculated from the data. It is not used in the paper but it is good to have it.

    return XCOLDGASS


def allsmog_cicone2017_data_reading(filedir):

    '''
    CO(2-1) detections. 
    Targets were selected such that 
        1. They are star forming galaxies (based on the BPT diagram; Baldwin et al. 1981)
        2. They have stellar masses in the range 10^9 M⊙ < M* < 10^10.5 M⊙
        3. They have redshifts 0.01 < z < 0.03
        4. They have metallicities 12 + log(O/H) > 8.5 (based on the Tremonti et al. 2004 mass-metallicity relation)
    '''


    fpath = f"{filedir}/allsmog_cicone_2017/cicone_all_data.tsv"

    ### Table 1 
    # Samp;Seq;Name;RAJ2000;DEJ2000;zopt;DL;i;d25;logM*;e_logM*;logSFR;e_logSFR;MPA-JHU;N2PP04;N2M13;O3N2PP04;O3N2M13;l_logMHI;logMHI;l_logMHIc;logMHIc;e_logMHIc;Tel;Ref;Note;SimbadName
    columns_table1 = [
        "Samp",
        "Seq",
        "Name",
        "RAJ2000",
        "DEJ2000",
        "zopt",
        "DL",
        "i",
        "d25",
        "logMstar",
        "e_logMstar",
        "logSFR",
        "e_logSFR",
        "MPA-JHU",
        "N2PP04",
        "N2M13",
        "O3N2PP04",
        "O3N2M13",
        "l_logMHI",
        "logMHI",
        "l_logMHIc",
        "logMHIc",
        "e_logMHIc",
        "Tel",
        "Ref",
        "Note",
        "SimbadName"
    ]

    # Read the file using that header
    table1 = pd.read_csv(
        fpath,
        sep=';',
        names=columns_table1,
        skiprows=62,
        nrows = 159 - 62
    )

    ### Table 2 
    # Samp;Seq;Name;tON;rms;v0;e_v0;sigma;e_sigma;Speak;e_Speak;l_Int;Int;e_Int
    columns_table2 = [
        "Samp",
        "Seq",
        "Name",
        "tON",
        "rms",
        "v0",
        "e_v0",
        "sigma",
        "e_sigma",
        "Speak",
        "e_Speak",
        "l_Int",
        "Int",
        "e_Int"
    ]
    table2 = pd.read_csv(
        fpath,
        sep=';',
        names=columns_table2,
        skiprows=180,
        nrows = 279 - 180
    )
    # Drop columns ['l_Int', 'Int', 'e_Int'] from table2 as they are not needed. They are going to be taken from table3
    table2 = table2.drop(columns=['l_Int', 'Int', 'e_Int'])

    ## Table 3 
    # Samp;Seq;Name;l_Int;Int;e_Int;Beam;e_Beam;l_L_CO;L_CO;e_L_CO
    columns_table3 = [
        "Samp",
        "Seq",
        "Name",
        "l_Int",
        "Int",
        "e_Int",
        "Beam",
        "e_Beam",
        "l_L_CO",
        "L_CO", # 10+8 K.km/s.pc+2
        "e_L_CO" # # 10+8 K.km/s.pc+2
    ]

    table3 = pd.read_csv(
        fpath,
        sep=';',
        names=columns_table3,
        skiprows=297,
        nrows = 395 - 297
    ) 

    table3['L_CO'] = table3['L_CO'] * 1e8  # K.km/s.pc+2
    table3['e_L_CO'] = table3['e_L_CO'] * 1e8  # K.km/s.pc+2

    ###
    IRAM_observations_ids = [89, 90, 91, 92, 93, 94, 95, 96, 97]
    # Delete Seq values that match with IRAM_observations_ids
    table1 = table1[~table1['Seq'].isin(IRAM_observations_ids)]
    table2 = table2[~table2['Seq'].isin(IRAM_observations_ids)]
    table3 = table3[~table3['Seq'].isin(IRAM_observations_ids)]

    ## There are two NGC 2936 values in table 2. So I am deleting all of them. Seq == 35. After this deletion they have the same length
    problematic_ids = [35]
    table1 = table1[~table1['Seq'].isin(problematic_ids)]
    table2 = table2[~table2['Seq'].isin(problematic_ids)]
    table3 = table3[~table3['Seq'].isin(problematic_ids)]

    ## Merge dataframes based on 'Samp', 'Seq', and 'Name' columns
    # Samp - A : Arecibo CO(2-1) -- I1 IRAM CO(1-0) -- I2 IRAM CO(2-1)
    merged_df = table1.merge(table2, on=['Samp', 'Seq', 'Name'])
    merged_df = merged_df.merge(table3, on=['Samp', 'Seq', 'Name'])

    ## Set the detection_flag based on 'l_Int' column
    upper_limit_cond = merged_df['l_Int'] == '<'
    merged_df['detection_flag'] = np.where(upper_limit_cond, 2, 1)


    return merged_df

if __name__ == "__main__":
    fdir = 

###############################################################################################################################################

def PHIBSS2_data_reading(filedir):


    print("I am in the function PHIBSS2_data_reading")

    """This function is used to read the PHIBBS2 Survey results 
    
    Arguments:
    ----------
    file_dir: string
        Indicator of the file directory to read the star particle information

    Returns:
    ----------
    ID_PHIBBS2: numpy.ndarray - string
        ID of the galaxies. These are the name of the galaxies 

    ID_Number_PHIBBS: numpy.ndarray - float 
        Numbers assigned to galaxies. 
        
    LCO_10_PHIBBS2: numpy.ndarray - float
        CO(1-0) luminosity of the galaxies.
        Unit: K-km s^-1 pc^2

    M_H2_PHIBBS2: numpy.ndarray - float
        Total molecular gas mass of the galaxies.
        Unit: M⊙

    SFR_PHIBBS2: numpy.ndarray - float
        Total molecular gas mass of the galaxies.
        Unit: M⊙

    X_CO_PHIBBS2: numpy.ndarray - float
        X_CO value of the galaxies.
        Unit: cm^-2 (K-km s^-1)^-1


    References: 
    -----------
    PHIBSS2 survey design and z=0.5-0.8 results. Molecular gas reservoirs during the winding-down of star formation

    """ 

    data_PHIBBS2 = pd.read_excel(filedir)

    # ID of the PHIBBS2 galaxies
    ID_PHIBBS2 = data_PHIBBS2["ID"].to_numpy()

    # ID_Number of the PHIBBS2 galaxies
    ID_Number_PHIBBS = (data_PHIBBS2["#"].to_numpy()).astype(float)

    # CO(2-1) Luminosity of the PHIBBS2 survey
    LCO_21_PHIBBS2 = (data_PHIBBS2["L CO(2−1)"].to_numpy()).astype(float)

    # r_21 is assumed to be 0.77. It is given in the "PHIBSS2 survey design and z=0.5-0.8 results. Molecular gas reservoirs during the winding-down of star formation" paper at page 9
    r21 = 0.77
    LCO_10_PHIBBS2 = LCO_21_PHIBBS2 / r21

    # Total Molecular gas mass of PHIBB2 survey. It looks like it is calculated by assuming alfa_co = 4.36. Page 8 notes for the table. 
    M_H2_PHIBBS2 = (data_PHIBBS2["M gas"].to_numpy()).astype(float)

    # SFR of the PHIBBS2 survey
    SFR_PHIBBS2 = data_PHIBBS2["SFR"].to_numpy().astype(float)

    # Mstar  
    Mstar_PHIBBS2 = (data_PHIBBS2["M_stellar"].to_numpy()).astype(float)

    # zoptical
    z_optical_PHIBBS2 = (data_PHIBBS2["zoptical"].to_numpy()).astype(float)

    # Mgas 
    Mgas_PHIBBS2 = (data_PHIBBS2["M gas"].to_numpy()).astype(float)

    # mu gas 
    mugas_PHIBBS2 = (data_PHIBBS2["μgas"].to_numpy()).astype(float) # mu_gas = M_gas / M_star

    # fgas
    fgas_PHIBBS2 = (data_PHIBBS2["fgas"].to_numpy()).astype(float) # f_gas = M_gas / (M_gas + M_star)

    # tdepl
    tdep_PHIBBS2 = (data_PHIBBS2["tdepl"].to_numpy()).astype(float) # Gyr -- M_gas / SFR

    # Derived alfa_co 
    alfa_CO_PHIBBS2 = M_H2_PHIBBS2 / LCO_10_PHIBBS2
    # Derived X_co, using eqn 3 in the paper "A General Model for the CO-H2 Conversion Factor in Galaxies with Applications to the Star Formation Law"
    X_CO_PHIBBS2 = alfa_CO_PHIBBS2 * 6.3e19

    data = {
        "Id": ID_PHIBBS2,
        "Id_Number": ID_Number_PHIBBS,
        "z_optical": z_optical_PHIBBS2,
        "Mh2": M_H2_PHIBBS2,
        "SFR": SFR_PHIBBS2,
        "Mstar": Mstar_PHIBBS2,
        "Mgas": Mgas_PHIBBS2,
        "mugas": mugas_PHIBBS2,
        "fgas": fgas_PHIBBS2,
        "tdep": tdep_PHIBBS2,
        "Xco": X_CO_PHIBBS2,
        "Lco10": LCO_10_PHIBBS2,
        "Lco21": LCO_21_PHIBBS2,
    }

    PHIBBS2 = pd.DataFrame(data)

    return PHIBBS2



###############################################################################################################################################

def ALMA_2019_Data_Reading(filedir):

    print("I am in the function ALMA_2019_Data_Reading")

    """This function is used to read the PHIBBS2 Survey results 
    
    Arguments:
    ----------
    file_dir: string
        Indicator of the file directory to read the star particle information

    Returns:
    ----------
    ID_ALMA_2019: numpy.ndarray - string
        ID of the galaxies. These are the name of the galaxies  
        
    L_CO_ALMA_2019: numpy.ndarray - float
        CO(1-0) luminosity of the galaxies.
        Unit: K-km s^-1 pc^2

    M_H2_ALMA_2019: numpy.ndarray - float
        Total molecular gas mass of the galaxies.
        Unit: M⊙

    SFR_ALMA_2019: numpy.ndarray - float
        Total molecular gas mass of the galaxies.
        Unit: M⊙

    X_CO_ALMA_2019: numpy.ndarray - float
        X_CO value of the galaxies.
        Unit: cm^-2 (K-km s^-1)^-1


    References: 
    -----------
    The ALMA Spectroscopic Survey in the HUDF Nature and Physical Properties of Gas-mass Selected Galaxies Using MUSE Spectroscopy

    """ 

    # Reading the excel file 
    data_ALMA_2019 = pd.read_excel(filedir)

    # Name of the galaxies 
    ID_ALMA_2019 = data_ALMA_2019["ID"].to_numpy()

    # redshift of the galaxies 
    z_ALMA_2019 = data_ALMA_2019["z_CO"].to_numpy().astype(float)

    # L_co of the galaxies 
    L_CO_ALMA_2019 = data_ALMA_2019["L_CO(1-0)"].to_numpy().astype(float)       # 1e9 K km s^-1 pc^2
    L_CO_ALMA_2019 = L_CO_ALMA_2019 * 1e9                                       # K km s^-1 pc^2

    # Error_LCO of the galaxies 
    Error_LCO_ALMA_2019 = data_ALMA_2019["Error_L_CO(1-0)"].to_numpy().astype(float)
    # 1e9 K km s^-1 pc^2    

    # Total Molecular gas mass of the galaxies 
    M_H2_ALMA_2019 = data_ALMA_2019["Mmol"].to_numpy().astype(float)        # 1e10 M_solar
    M_H2_ALMA_2019 = M_H2_ALMA_2019 * 1e10                                  # M_solar


    # Error on the total molecular gas mass of the galaxies 
    Error_MH2_ALMA_2019 = data_ALMA_2019["Error_Mmol"].to_numpy().astype(float)     # 1e10 M_solar
    Error_MH2_ALMA_2019 = Error_MH2_ALMA_2019 * 1e10                                # M_solar


    # Depletion time of the galaxies 
    t_dep_ALMA_2019 = data_ALMA_2019["t_depl"].to_numpy().astype(float)
    # Gyr

    # Star mass 
    M_star_ALMA_2019 = (10**data_ALMA_2019["log_M_star"]).to_numpy().astype(float)       # Msolar

    # Visual attenuation
    A_V_ALMA_2019 = data_ALMA_2019["Av"].to_numpy().astype(float)                       # mag

    # Metallicity 
    metallicity_in_solar_units_ALMA_2019 = converting_log12_plus_log_O_over_H_to_solar_units(X=data_ALMA_2019["12 + log(O/H)"].to_numpy().astype(float)) # 12 + log(O/H) = Z_MZR


    # SFR of the galaxies 
    SFR_ALMA_2019 = M_H2_ALMA_2019 / t_dep_ALMA_2019        # M_solar/Gyr
    SFR_ALMA_2019 = SFR_ALMA_2019 / 1e9                             # M_solar/year

    # X_CO of the galaxies 
    alfa_CO_ALMA_2019 = M_H2_ALMA_2019 / L_CO_ALMA_2019  # M⊙ pc^-2 (K-km s^-1)^-1
    # Derived X_co, using eqn 3 in the paper "A General Model for the CO-H2 Conversion Factor in Galaxies with Applications to the Star Formation Law"
    X_CO_ALMA_2019 = alfa_CO_ALMA_2019 * 6.3e19


    # Other CO transitions 
    J_up = data_ALMA_2019["J_up"].to_numpy().astype(int)
    L_line = data_ALMA_2019["L_line"].to_numpy().astype(float) * 1e9       # K km s^-1 pc^2

    # redshift 
    z_CO = data_ALMA_2019["z_CO"].to_numpy().astype(float)

    data = {
        "Id": ID_ALMA_2019,
        "redshift": z_CO,
        "Lco": L_CO_ALMA_2019,
        "Mh2": M_H2_ALMA_2019,
        "SFR": SFR_ALMA_2019,
        "Xco": X_CO_ALMA_2019,
        "Mstar": M_star_ALMA_2019,
        "Av": A_V_ALMA_2019,
        "12 + log(O/H)": data_ALMA_2019["12 + log(O/H)"].to_numpy().astype(float),
        "metallicity": metallicity_in_solar_units_ALMA_2019,
        "J_up": J_up,
        "L_line": L_line,
    }

    ALMA = pd.DataFrame(data)

    return ALMA

###############################################################################################################################################

def Leroy_Data_Reading(filedir):

    """This function is used to read the Leroy paper that shows Kennicutt like relation between SFR_surface_density and molecular 
    gas surface density   
    
    Arguments:
    ----------
    file_dir: string
        Indicator of the file directory to read the star particle information

    Returns:
    ----------
    
    Leroy_df: pandas dataframe 
        Contains information of 
        
        galaxy_name_LEROY: 
            Name of the galaxies
            string

        sigma_MH2_LEROY: 
            Hi + H2 gas surface density 
            Unit: M☉ pc^-2

        sigma_SFR_LEROY: numpy.ndarray - float
            Star formation rate surface density
            Unit: M☉ yr^-1 kpc^-2 


    average_radius_r25_times_075_LEROY: float
        Average radius of galaxies that sigma_sfr and sigma_MH2 is calculated

    References: 
    -----------
    
    Molecular Gas and Star Formation in Nearby Disk Galaxies
    arXiv:1301.2328v1

    """     
    print("I am in the function Leroy_data_reading")

    # Reading the excel file
    data_Leroy = pd.read_excel(filedir)

    # Name of the galaxies 
    galaxy_name_LEROY = data_Leroy["Galaxy"].to_numpy()

    # Stellar Mass 
    log10_stellar_mass_LEROY = data_Leroy["log(M*)"].to_numpy().astype(float)
    M_stellar_LEROY = 10**log10_stellar_mass_LEROY
    # Unit: M☉ 

    #r25 and 0.75*r25
    r25_LEROY = data_Leroy["r25"].to_numpy().astype(float)
    r25_times_075_LEROY = data_Leroy["0.75r25"].to_numpy().astype(float)
    # Unit: kpc

    # Metallicity in 12+log[O/H]
    metallicity_LEROY = data_Leroy["z"].to_numpy().astype(float)
    # Unit 

    # Molecular gas mass 
    sigma_MH2_LEROY = data_Leroy["〈ΣH i+H2〉"].to_numpy().astype(float)
    # Unit: M☉ pc^-2

    # SFR
    sigma_SFR_LEROY = data_Leroy["〈ΣSFR〉"].to_numpy().astype(float)
    sigma_SFR_LEROY = sigma_SFR_LEROY * 1e-3
    # Unit: M☉ yr^-1 kpc^-2 

    average_radius_r25_times_075_LEROY = np.sum(r25_times_075_LEROY)/len(r25_times_075_LEROY)


    Leroy_df = pd.DataFrame({"galaxy_name":galaxy_name_LEROY,
                             "sigma_MH2":sigma_MH2_LEROY,
                             "sigma_SFR":sigma_SFR_LEROY})  

    return (Leroy_df,
            average_radius_r25_times_075_LEROY) 
###############################################################################################################################################

def leroy_2011_alphaCO(filedir):
    """
    https://ui.adsabs.harvard.edu/abs/2011ApJ...737...12L/abstract

    The CO-to-H2 Conversion Factor from Infrared Dust Emission across the Local Group
    Leroy, Adam K. search by orcid ; Bolatto, Alberto search by orcid ; Gordon, Karl search by orcid ; Sandstrom, Karin search by orcid ; Gratier, Pierre search by orcid ; Rosolowsky, Erik ; Engelbracht, Charles W. ; Mizuno, Norikazu ; Corbelli, Edvige search by orcid ; Fukui, Yasuo search by orcid ; Kawamura, Akiko
    """

    # Read the data 
    data = pd.read_csv(f"{filedir}/leroy_2011.csv", sep=";")
    data['metallicity'] = converting_log12_plus_log_O_over_H_to_solar_units(data['12 + log(O/H)'].to_numpy())

    ## Convert alpha_CO to Xco 
    data['Xco'] = data['alpha_CO'] * 6.3e19 # cm^-2 (K-km s^-1)^-1

    return data

###############################################################################################################################################

def miville_deschenes_2017(fdir):


    # Define column specs from the description
    colspecs = [
        (0, 4),    # Cloud
        (5, 8),    # Ncomp
        (9, 12),   # Npix
        (13, 25),  # A
        (26, 39),  # l
        (40, 52),  # e_l
        (53, 66),  # b
        (67, 79),  # e_b
        (80, 93),  # theta
        (94, 106), # WCO
        (107, 119),# NH2
        (120, 132),# Sigma
        (133, 146),# vcent
        (147, 159),# sigmav
        (160, 172),# Rmax
        (173, 185),# Rmin
        (186, 198),# Rang
        (199, 211),# Rgal
        (212, 213),# INF
        (214, 226),# Dn
        (227, 239),# Df
        (240, 253),# zn
        (254, 267),# zf
        (268, 280),# Sn
        (281, 293),# Sf
        (294, 306),# Rn
        (307, 319),# Rf
        (320, 332),# Mn
        (333, 345) # Mf
    ]

    # Define column names (from description)
    names = [
        "Cloud", "Ncomp", "Npix", "A", "l", "e_l", "b", "e_b", "theta", 
        "WCO", "NH2", "Sigma", "vcent", "sigmav", "Rmax", "Rmin", "Rang", 
        "Rgal", "INF", "Dn", "Df", "zn", "zf", "Sn", "Sf", "Rn", "Rf", "Mn", "Mf"
    ]

    # Read the file, skipping header lines
    data = pd.read_fwf(f"{fdir}/miville_deschenes_2017.txt", colspecs=colspecs, names=names, skiprows=45)

    data['Lco'] = data['WCO'] * data['Sn'] # Near derived CO luminosity in K km s^-1 pc^2
    data['hden'] = data['NH2'] / (data['Rn'] * constants.pc2cm)
    # data['metallicity'] = 1.0 # TODO:For now 
    data['isrf'] = 1.0 # TODO:For now
    data['radius'] = data['Rn'] # Near radius in pc
    data['turbulence'] = data['sigmav'] # Velocity dispersion in km/s

    # From "https://iopscience.iop.org/article/10.1086/498869"   X(R) = X(R=8.5kpc) * (R - 8.5kpc) * (-0.006), where X(R=8.5kpc) = 8.67

    data['12 + log(O/H)'] = 8.67 + (data['Rgal'] - 8.5) * (-0.06)

    # Converting 12 + log(O/H) to metallicity in solar units
    data['metallicity'] = converting_log12_plus_log_O_over_H_to_solar_units(data['12 + log(O/H)'].to_numpy())

    # Get the mass of the cloud in solar masses 
    data['mass'] = data['Mn'] # Near derived mass in Msolar

    return data 


def twelve_plus_logOH_to_solar_metallicity(X1, X_solar=8.69):
    """
    X1: float
        12 + log(O/H)
    X_solar: float
        12 + log(O/H) at the solar value.
        Default is 8.69  
        By following: https://en.wikipedia.org/wiki/Abundance_of_the_chemical_elements
    """
    
    
    mO_over_mH_solar = 1.04 / 100 / 0.74
    
    molecular_mass_ratio_H_over_O = (1.00784 / 15.999)
    
    nO_over_nH_solar = mO_over_mH_solar * molecular_mass_ratio_H_over_O 
    nO_over_nH = nO_over_nH_solar * ( 10**(X1 - 12)  / 10**(X_solar - 12) )
    
    mO_over_mH = nO_over_nH / molecular_mass_ratio_H_over_O
    
    Z = mO_over_mH / mO_over_mH_solar
    
    return Z

twelve_plus_logOH_to_solar_metallicity(X1 = 8)

if __name__ == "__main__":
    fdir = "/home/dtolgay/Observations"
    miville_deschenes_2017(fdir)


###############################################################################################################################################


def Li_model(galaxy_name):

    print("I am in the function Li_model")

    """Lco calcuation from the SFR using the Li model
    
    Arguments:
    ----------
    galaxy_name: pandas DataFrane
        DataFrame that contains information about the galaxies  

    Returns:
    ----------
    Li_model: pandas DataFrane
        DataFrame that contains output of the Li Model

    References: 
    -----------
    arXiv:1503.08833
    
    """ 

    # defining parameters
    delta_MF, alpha, beta = (1.0, 1.37, -1.74)

    # Calculating infrared luminosity
    Lir = galaxy_name["sfr_array"] * 1e10               # M⊙

    # Calculating CO luminosity 
    log_Lco = ( np.log10(Lir) - beta ) / alpha              # log(K km s^-1 pc^2)   Units are not same with the Lir but it is fine. It is also same in the Cariilli and Walter and also in the Li et al.

    log_Lco_in_observer_units = log_Lco                     # log(K km s^-1 pc^2)
    Lco_in_observer_units = 10**log_Lco_in_observer_units   # K km s^-1 pc^2
    Lco = 10**log_Lco * 4.9e-5                              # L_solar
    log_Lco = np.log10(Lco)                                 # log(L_solar)

    # Molecular gas mass calculated by assuming constant CO to H2 factor
    alpha_co = 3.2
    Mh2 = Lco_in_observer_units * alpha_co

    Li_model = pd.DataFrame(data=None)
    Li_model["L_co_total_array_in_observer_units"] =  Lco_in_observer_units             # K km s^-1 pc^2
    Li_model["log_L_co_total_array_in_observer_units"] = log_Lco_in_observer_units      # log(K km s^-1 pc^2)
    Li_model["log_L_co_total_array"] = log_Lco                                          # log(L_solar)
    Li_model["L_co_total_array"] = Lco                                                  # L_solar
    Li_model["young_sfr_array"] = galaxy_name["young_sfr_array"]                        # M_solar/year
    Li_model["log_young_sfr_array"] = np.log10(Li_model["young_sfr_array"])             # log(M_solar/year)
    Li_model["total_mass_h2_array"] = Mh2                                               # M⊙
    Li_model["redshift_array"] = galaxy_name["redshift_array"]

    Li_model["sfr_array"] = galaxy_name["sfr_array"]                                    # M⊙/year
    Li_model["log_sfr_array"] = np.log10(galaxy_name["sfr_array"])                      # log(M⊙/year)

    return Li_model



def Li_model_sfr_input(sfr):

    print("I am in the function Li_model")

    """Lco calcuation from the SFR using the Li model
    
    Arguments:
    ----------
    galaxy_name: pandas DataFrane
        DataFrame that contains information about the galaxies  

    Returns:
    ----------
    Li_model: pandas DataFrane
        DataFrame that contains output of the Li Model

    References: 
    -----------
    arXiv:1503.08833
    
    """ 

    # defining parameters
    delta_MF, alpha, beta = (1.0, 1.37, -1.74)

    # Calculating infrared luminosity
    Lir = sfr * 1e10               # M⊙

    # Calculating CO luminosity 
    log_Lco = ( np.log10(Lir) - beta ) / alpha              # log(K km s^-1 pc^2)   Units are not same with the Lir but it is fine. It is also same in the Cariilli and Walter and also in the Li et al.
    log_Lco_in_observer_units = log_Lco                     # log(K km s^-1 pc^2)

    Li_model = pd.DataFrame(data=None)
    Li_model["log_L_co_total_array_in_observer_units"] = log_Lco_in_observer_units      # log(K km s^-1 pc^2)
    Li_model["log_sfr_array"] = np.log10(sfr)                                           # log(M⊙/year)

    return Li_model


################################################################################################################################################

def rachford_2002_number_column_density(filedir):

    # Define the column names
    columns = [
        "name", "logN(CH)", "Reference", "log(N(CH+))", "Reference2", 
        "log(N(CN))", "Reference3", "log(N(CO))", "Reference4", "logN(H2)", 
        "unc logN(H2)", "logN(0)", "unc logN(0)", "logN(1)", "unc logN(1)", 
        "T_kin", "unc T_kin", "logN(HI)", "unc log(HI)", "Reference5", "f_H2", "unc f_H2"
    ]

    # File path
    fdir = f"{filedir}/rachford_2022.csv"  # Adjust path if needed

    # Read CSV while handling spaces properly
    data = pd.read_csv(fdir, sep=";", names=columns)

    return data 


def crenny_2004_number_column_density(filedir):

    """
    https://ui.adsabs.harvard.edu/abs/2004ApJ...605..278C/abstract
    Reanalysis of Copernicus Measurements of Interstellar Carbon Monoxide
    Diffuse molecular clouds in MW. Data points corresponds to measurements of molecular abundances along line of sight lines 
    toward background stars in the MW. 
    """

    # columns = [
    #     "Star", "Nco", "Nh2", "Nch", "Nch+"
    # ]

    # Read CSV with default comma separator
    df = pd.read_csv(f"{filedir}/crenny_2004.csv", sep=";")

    # Make Nco, Nh2, Nch and Nch+ columns float64
    columns = ["Nco", "Nh2", "Nch", "Nch+"]
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df 

def sheffer_2008_number_column_density(filedir):
    """
    https://ui.adsabs.harvard.edu/abs/2008ApJ...687.1075S/abstract

    Observing diffuse molecular clouds along the molecular Galactic sidelines. 
    """

    # columns = [
    #     "Star",
    #     "logH2",
    #     "References",
    #     "logCO",
    #     "References_1",
    #     "logCH^+",
    #     "References_2",
    #     "logCH",
    #     "References_3",
    #     "logCN",
    #     "References_4"
    # ]

    # Read the CSV
    df = pd.read_csv(f"{filedir}/sheffer_2008.csv", sep=";")

    # Convert relevant columns to float
    numeric_columns = ["logH2", "logCO", "logCH^+", "logCH", "logCN"]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df 

# Define a helper function to split the "value ± error" format
def split_plus_minus(val):
    if isinstance(val, str) and '±' in val:
        value, error = val.split('±')
        return float(value.strip()), float(error.strip())
    return val, None  # Leave original value if no ±

def burgh_2010_number_column_density(filedir):

    columns = ['StarName', 'Sp.Type', 'E(B − V )', 'A_V', 'Ref.', 'logN(Hi)', 'Ref._1',
       'logN(H2)', 'Uncertainty logN(H2)', 'T_01', 'Ref._2', 'f^N',
       'logN(CO)']

    # Read the CSV file
    df = pd.read_csv(f"{filedir}/burgh_2010.csv", sep=";")

    # Choose columns that have "±" values (update based on your data)
    columns_with_pm = ['logN(Hi)', 'logN(H2)', 'logN(CO)']  # Replace with real column names or indices

    # Process each column
    for col in columns_with_pm:
        df[[f"{col}_value", f"{col}_error"]] = df[col].apply(lambda x: pd.Series(split_plus_minus(x)))

    # Now change the _value column with the columns_with_pm and drop the _value columns 
    for col in columns_with_pm:
        df[col] = df[f"{col}_value"]
        df.drop(columns=[f"{col}_value"], inplace=True)

    # Change the type of the columns_with_pm to float
    for col in columns_with_pm:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

################################################################################################################################################

# Lya functions
def hetdex_data_reading(fdir):

    """

    """


    print(f"I am in the function hetdex_data_reading!")

    table = Table.read(fdir)

    data = table[[
        "source_id",        # [1]
        "z_hetdex",         # [1]    
        "source_type",      # [1]  
        "z_hetdex_conf",    # [1] Between -1 and 1
        "lum_lya",          # [erg s^-1]
        "lum_lya_err"       # [erg s^-1]
    ]].to_pandas()                           

    return data    


def HAYES_2014_data_reading(filedir):

    # Reading the excel file 
    data = pd.read_excel(filedir)

    """
    Returned Units:
    z = reshdift [1]
    2 x r_p20 [kpc]
    Luminosities: [erg/s]
    W [Armstrong]
    Age [Myr]
    SfR [Msolar / year]
    E_B_V [mag]
    mass [Msolar]

    References:
    ---------------
    The Lyman Alpha Reference Sample. II. Hubble Space Telescope Imaging Results, Integrated Properties, and Trends by Hayes+ 2014    
    """

    print(f"I am in the function HAYES_2014_data_reading!")


    data.rename(columns={
        'LARS': 'ID', 
        'Mass': 'mass', 
        'Age': 'age',
    }, inplace=True)

    data.rename(columns={'LARS': 'ID'}, inplace=True)


    # Unit adjustment
    data["L_Lya"] *= 1e42
    data["L_FUV"] *= 1e40
    data["L_Ha"] *= 1e42
    data["L_Hb"] *= 1e42
    data["mass"] *= 1e9
    
    return data


def cowie_2011_lya_reading(filedir):

    """
    Returned Units:
    z = reshdift [1]
    log_L [log(erg/s)]
    EW [armstrong]
    L_Lya [erg/s]

    References:
    -------------
    Lyα EMITTING GALAXIES AS EARLY STAGES IN GALAXY FORMATION by Cowie+ 2011
    """

    print(f"I am in the function cowie_2011_lya_reading!")

    data = pd.read_excel(filedir)

    data["L_Lya"] = 10**data["log L_lya"] 
    
    return data



################################################################################################################################################
 
 # H alpha functions 

def young_1996(fdir): 

    # Using paper: https://ui.adsabs.harvard.edu/abs/1996AJ....112.1903Y/abstract

    print(f"I am in the function young_1996!")

    # Read the Excel file
    table_df = pd.read_excel(fdir)

    # Calculate SFR inferred from Lir. Conversion is being done using the formula in the 3rd paragraph of Cool gas in high redshift galaxies by Carilli and Walter
    table_df["ir_inferrred_sfr"] = 10**(table_df["log_L_ir"] - 10 + np.log10(1.3))



    # Converting Lsolar to erg s^-1 
    columns = [
        # "log_L_B",
        "log_L_h_alpha",
        "log_L_ir",
    ]

    table_df["L_h_alpha"] = 10**table_df["log_L_h_alpha"] * Lsolar2erg_per_s    

    return table_df


def james_2024(fdir):

    # Using paper: https://www.aanda.org/articles/aa/abs/2004/04/aah4129/aah4129.html

    column_names = [
        # "galaxy_name",
        "galaxy_name_1",
        "galaxy_name_2",
        "galaxy_classification", 
        "recession_velocity",
        "galaxy_distance",   
        "major_axis_size",
        "major_to_minor_axis_ratio",
        "R_magnitude",
        "error_R_magnitude",
        "F_h_alpha_NII",
        "error_F_h_alpha_NII",
        "EW_h_alpha_NII",
        "error_EW_h_alpha_NII",
        "sfr",
        "error_sfr",
        "surface_brightness_h_alpha_NII",
        "error_surface_brightness_h_alpha_NII",
    ]

    # Read the CSV file
    df = pd.read_csv(fdir, sep='\s+', names=column_names, header=None)    

    # Calculating luminosity 


    # Luminsoity is calculated assuming there is no extinction
    # 1e-16 factor is because of the normalization value in the table. Check the paper 
    df["L_h_alpha_NII"] = df["F_h_alpha_NII"] * 1e-16* 4 * np.pi * (df["galaxy_distance"]*Mpc2meter)**2 * watt2ergs_per_second # erg s^-1
    df["error_L_h_alpha_NII"] = df["error_F_h_alpha_NII"] * 1e-16* 4 * np.pi * (df["galaxy_distance"]*Mpc2meter)**2 * watt2ergs_per_second # erg s^-1


    return df


###############################################################################################################################################


# Importing all CO observations 
def read_CO_observations(base_dir="/mnt/raid-cita/dtolgay/Observations"):
    
    ### C0
    xCOLDGASS_file_path = f"{base_dir}/xCOLDGASS_PubCat.fits"
    XCOLDGASS_df = XCold_Gass_data_reading(filedir=xCOLDGASS_file_path)

    PHIBSS2_file_path = f"{base_dir}/PHIBSS2_data.xlsx" 
    PHIBBS2_df = PHIBSS2_data_reading(filedir=PHIBSS2_file_path)

    ALMA_2019_file_path = f"{base_dir}/ALMA_2019_data.xlsx"
    ALMA_df = ALMA_2019_Data_Reading(filedir=ALMA_2019_file_path)

    Leroy_file_path = f"{base_dir}/Leroy_data.xlsx"
    Leroy_df, average_radius_r25_times_075_LEROY = Leroy_Data_Reading(filedir=Leroy_file_path)

    cicone_df = allsmog_cicone2017_data_reading(filedir=f"{base_dir}")   
    
    return XCOLDGASS_df, PHIBBS2_df, ALMA_df, Leroy_df, cicone_df


###############
# Importing C2 observations 
def read_Cii_observations(base_dir="/mnt/raid-cita/dtolgay/Observations"):

    herrera_2015 = pd.DataFrame(
        np.loadtxt(f"{base_dir}/Herrera2015.txt"),
        columns=['sfr', 'unc_sfr', 'L_c2']
    )

    delooze = pd.DataFrame(
        np.loadtxt(f"{base_dir}/deLooze2011_corr.txt"),
        columns=['sfr', 'unc_sfr', 'log_L_c2', 'unc_Lc2']
    )
    delooze['L_c2'] = 10**delooze['log_L_c2']    

    return herrera_2015, delooze 

    ###############


###############################################################################################################################################

# Reading Kamenetzky+ 2016 results. 

def meters_to_Ghz_calculator(wavelength_in_meters):
    c = 299792458  # m/s
    frequency_in_Ghz = c / wavelength_in_meters * 1e-9
    return frequency_in_Ghz

def calculate_Lco_in_observer_units(df):

    """
    Table contains the information about the measurements and galaxies. This information includes: 
    1. reshift [1]
    2. measured flux [Jy km s^-1]
    3. luminosity distance [Mpc]
    4. Upper J level of the emission [1]
    """
    table = df.copy() # Copy the table to not accidentally change the values in the original dataframe
    
    # Change the unit of the calculated CO luminosities 
    Jup_and_wavelengths = {
        "1": 2600.05e-6,  # meter
        "2": 1300.05e-6,
        "3": 866.727e-6,
        "4": 650.074e-6,
        "5": 520.08e-6,
        "6": 433.438e-6,
        "7": 371.549e-6,
        "8": 325.137e-6,
        "9": 289.199e-6,
        "10": 260.005e-6,
        "11": 236.368e-6,
        "12": 216.672e-6,
        "13": 200.003e-6,  
    }    
    
    # Calculate Lco in observer units 
    observed_frequency_in_Ghz = []
    for _, gal in table.iterrows():
        rest_frequency_in_Ghz = meters_to_Ghz_calculator(Jup_and_wavelengths[str(gal['Jup'])]) 
        redshift = gal['redshift']
        observed_frequency_in_Ghz.append(rest_frequency_in_Ghz / (1 + redshift))

    table['f_obs'] = observed_frequency_in_Ghz # Ghz 

    Lco = 3.25e7 * table["Flux"] * table["d_luminosity"]**2 * (1 + table["redshift"])**(-3) * table['f_obs']**(-2)     
    
    log_Lco = np.log10(Lco)
    
    return log_Lco

def average_the_flux_values_kametzsky(kamenetzsky_data):

    data = []
    
    # Calculate r values
    galaxy_ids = kamenetzsky_data['ID'].unique()

    for galaxy_id in galaxy_ids:
        condition = kamenetzsky_data['ID'] == galaxy_id
        filtered_df = kamenetzsky_data[condition]

        for Jup in filtered_df['Jup'].unique():
            condition2 = filtered_df['Jup'] == Jup
            df = filtered_df[condition2]

            # Average the values and only report that 
            average_Flux = np.sum(df['Flux']) / len(df)

            data.append({
                'ID': df.iloc[0]['ID'],
                'Flux': float(average_Flux),
                'd_luminosity': df.iloc[0]['d_luminosity'],
                'Jup': Jup,
                'log_Lfir': df.iloc[0]['log_Lfir'], # No need to average.
                'redshift': float(df.iloc[0]['redshift'])
            })
            
    data = pd.DataFrame(data)
            
    return data

def kamenetzky_2016(fdir, Lir_lower_limit=None, Lir_higher_limit=None): 

    from astropy.cosmology import Planck18 as cosmo  # or use your specific cosmology
    from astropy import units as u
    from astropy.cosmology import z_at_value


    # Define a list to store all rows as dictionaries
    table1 = []

    with open(f"{fdir}/table1.txt") as f:
        # Skip the first 30 lines
        for _ in range(31):
            next(f)
        
        # Process each remaining line
        for row in f:
            # Extract data from specific columns
            ID = row[:16].strip()   # Galaxy identifier    
            f_ID = np.nan if row[17:18].strip() == '' else row[17:18].strip()  # [a] Flag on ID (1)
            RAh = row[19:21].strip() # Hour of Right Ascension (J2000)                
            RAm = row[22:24].strip() # Minute of Right Ascension (J2000)                 
            RAs = row[25:30].strip() # Second of Right Ascension (J2000)                
            DE_sign = row[31:32].strip() # Sign of the Declination (J2000)
            DEd = row[32:34].strip() # Degree of Declination (J2000)                
            DEm = row[35:37].strip() # Arcminute of Declination (J2000)                
            DEs = row[38:42].strip() # Arcsecond of Declination (J2000)               
            logLFIR = row[43:47].strip() or np.nan  # ? Log Far-IR (40-120 um) luminosity
            D = row[48:52].strip() or np.nan        # Luminosity distance 
            ndet = row[53:55].strip() or np.nan    # Number of 3{sigma} detections (2) 
            nul = row[56:58].strip() or np.nan    #  Number of upper limits (2)  
            FTS = row[59:69].strip()                #  Herschel SPIRE FTS Observation ID
            Phot = row[70:81].strip()           # Herschel SPIRE Photometer Observation ID    
            
            # Append the parsed row as a dictionary to the data list
            table1.append({
                "ID": ID,
    #             "f_ID": f_ID,
    #             "RAh": RAh,
    #             "RAm": RAm,
    #             "RAs": RAs,
    #             "DE_sign": DE_sign,
    #             "DEd": DEd,
    #             "DEm": DEm,
    #             "DEs": DEs,
                "log_Lfir": float(logLFIR),
                "d_luminosity": float(D),
    #             "ndet": ndet,
    #             "nul": nul,
    #             "FTS": FTS,
    #             "Phot": Phot
            })

    # Convert the list of dictionaries to a DataFrame
    table1 = pd.DataFrame(table1)

    # Calculate the redshift using the luminosity distance 
    luminosity_distances = table1['d_luminosity'].to_numpy().astype(float) * u.Mpc
    redshifts = np.array(z_at_value(cosmo.luminosity_distance, luminosity_distances, verbose=False))
    table1['redshift'] = redshifts


    ################################################################################

    # Define a list to store all rows as dictionaries
    table2 = []

    with open(f"{fdir}/table2.txt") as f:
        # Skip the first 30 lines (if necessary, adjust based on your file structure)
        for _ in range(19):
            next(f)
        
        # Process each remaining line
        for row in f:
            # Extract data from specific columns based on fixed-width positions
            ID = row[0:16].strip()                   # Galaxy identifier
            Line = row[17:24].strip()                # Line identifier
            f_Line = np.nan if row[25:26].strip() == '' else row[25:26].strip()  # Line resolved indicator
            Flux = row[27:35].strip() or np.nan      # Median line flux
            b_Flux = row[36:44].strip() or np.nan    # Lower 1σ boundary for Flux
            B_Flux = row[45:53].strip() or np.nan    # Upper 1σ boundary for Flux
            UpLim = row[54:62].strip() or np.nan     # 3σ upper limit on Line flux
            
            # Append the parsed row as a dictionary to the data list
            table2.append({
                "ID": ID,
                "Line": Line,
    #             "f_Line": f_Line,
                "Flux": float(Flux),
                "b_Flux": float(b_Flux),
                "B_Flux": float(B_Flux),
    #             "UpLim": UpLim
            })

    # Convert the list of dictionaries to a DataFrame
    table2 = pd.DataFrame(table2)

    # Use only CO
    condition = table2["Line"].str.startswith("CO")
    table2 = table2[condition].copy()
    # Extract the Jup value from the 'Line' column
    table2['Jup'] = table2['Line'].str.extract(r'CO(\d+)-')[0].astype(int)

    ################################################################################

    # Define a list to store all rows as dictionaries
    table3 = []

    with open(f"{fdir}/table3.txt") as f:
        # Skip the first 30 lines (adjust if needed)
        for _ in range(69):
            next(f)
        
        # Process each remaining line
        for row in f:
            # Extract data from specific columns based on fixed-width positions
            ID = row[0:19].strip()                  # Galaxy identifier
            Jup = row[20:21].strip() or np.nan      # Upper J level
            RFlux = row[22:30].strip()              # Reported line flux
            sigmam = row[31:39].strip()             # Measurement error in RFlux
            sigmac = row[40:48].strip()             # Calibration error in RFlux
            x_RFlux = row[49:54].strip()            # Units of RFlux
            dv = row[55:58].strip() or np.nan       # Line velocity FWHM
            Omegab = row[59:61].strip() or np.nan   # Beam size FWHM
            Flux = row[62:70].strip() or np.nan     # This analysis line flux
            e_Flux = row[71:79].strip() or np.nan   # Total uncertainty in Flux
            r_RFlux = row[80:82].strip() or np.nan  # Reference for RFlux
            
            # Append the parsed row as a dictionary to the data list
            table3.append({
                "ID": ID,
                "Jup": int(Jup),
    #             "RFlux": RFlux,
    #             "sigmam": sigmam,
    #             "sigmac": sigmac,
    #             "x_RFlux": x_RFlux,
    #             "dv": dv,
    #             "Omegab": Omegab,
                "Flux": float(Flux),
                "e_Flux": float(e_Flux),
                "reference": r_RFlux
            })

    # Convert the list of dictionaries to a DataFrame
    table3 = pd.DataFrame(table3)


    high_J_CO = table1.merge(table2, on='ID', how='right') # merge table 2 with table 1. Table 1 stores the information about galaxies
    low_J_CO = table1.merge(table3, on='ID', how='right') # merge table 3 with table 1. Table 1 stores the information about galaxies


    ### Filter the galaxies according to the upper and lower Lfir 
    # Set the boundaries 
    if Lir_lower_limit is None:
        Lir_lower_limit = 1       # Allow all lower values
    if Lir_higher_limit is None:
        Lir_higher_limit = 1e100    # Essentially no upper limit
    # Filter 
    condition = (np.log10(Lir_lower_limit) < high_J_CO['log_Lfir']) &  (high_J_CO['log_Lfir'] < np.log10(Lir_higher_limit))
    high_J_CO = high_J_CO[condition].copy()

    condition = (np.log10(Lir_lower_limit) < low_J_CO['log_Lfir']) &  (low_J_CO['log_Lfir'] < np.log10(Lir_higher_limit))
    low_J_CO = low_J_CO[condition].copy()    
    ###

    high_J_CO['log_Lco'] = calculate_Lco_in_observer_units(df = high_J_CO.copy())
    low_J_CO['log_Lco'] = calculate_Lco_in_observer_units(df = low_J_CO.copy())

    # 
    averaged_data_low_J = average_the_flux_values_kametzsky(kamenetzsky_data = low_J_CO)
    averaged_data_high_J = average_the_flux_values_kametzsky(kamenetzsky_data = high_J_CO)

    averaged_data = pd.concat([averaged_data_low_J, averaged_data_high_J])

    averaged_data['log_Lco'] = calculate_Lco_in_observer_units(averaged_data.copy())



    return high_J_CO, low_J_CO, averaged_data
