# Import modules 
from astropy.io import fits
from astropy.table import Table
import numpy as np
import pandas as pd

################################################################################################################################################
# Constants 
Mpc2meter = 3.086e+22  # meter / Mpc
watt2ergs_per_second = 1e7 # erg s^-1 / watt
Lsolar2erg_per_s = 3.826e33 # erg s^-1 / Lsolar 


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
        
        
    print("data_table_XCOLDGASS: ", data_table_XCOLDGASS.keys())

    # GASS catalog ID
    ID_XCOLDGASS                = data_table_XCOLDGASS['ID'].data

    # CO(1-0) line luminosity, aperture corrected [K km/s pc^2]
    LCO_corrected_XCOLDGASS     = data_table_XCOLDGASS['LCO_COR'].data  # K-km s^-1 pc^2

    #  Constant Galactic CO-to-H2 conversion factor  
    XCO_XCOLDGASS               = data_table_XCOLDGASS['XCO'].data      # 1e20 cm^-2 (K-km s^-1)^-1
    XCO_XCOLDGASS               = XCO_XCOLDGASS*1e20                    # cm^-2 (K-km s^-1)^-1

    # Total molecular gas mass [log Msun]
    LOGMH2_XCOLDGASS            = data_table_XCOLDGASS['LOGMH2'].data
    MH2_XCOLDGASS               = 10**LOGMH2_XCOLDGASS                  # M⊙


    # SFR from WISE + GALEX when detected in both data sets
    LOGSFR_XCOLDGASS            = data_table_XCOLDGASS['LOGSFR_BEST'].data
    SFR_XCOLDGASS               = 10**LOGSFR_XCOLDGASS                  # M⊙/year


    # Metallicity 
    # At solar metallicity 12 + log(O/H) = 8.69
    Zsolar = 10**(-3.31)
    metalicity_in_12_plus_log_O_over_H_units = data_table_XCOLDGASS['Z_MZR'].data
    log_12_plus_log_O_over_H = metalicity_in_12_plus_log_O_over_H_units - 12 
    metallicity_in_solar_units = 10**log_12_plus_log_O_over_H / Zsolar

    # Detection or upper limit 
    FLAG_CO = data_table_XCOLDGASS['FLAG_CO'].data 

    # Stellar mass 
    Mstar = 10**data_table_XCOLDGASS['LOGMSTAR'].data
    
    # Creating pandas dataframe
    data = {
        "Id": ID_XCOLDGASS,
        "Lco": LCO_corrected_XCOLDGASS,
        "Xco": XCO_XCOLDGASS,
        "Mh2": MH2_XCOLDGASS,
        "SFR": SFR_XCOLDGASS, 
        "metallicity": metallicity_in_solar_units,
        "FLAG_CO": FLAG_CO,
        "Mstar": Mstar,
    }

    XCOLDGASS = pd.DataFrame(data)


    return XCOLDGASS


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

    # Derived alfa_co 
    alfa_CO_PHIBBS2 = M_H2_PHIBBS2 / LCO_10_PHIBBS2
    # Derived X_co, using eqn 3 in the paper "A General Model for the CO-H2 Conversion Factor in Galaxies with Applications to the Star Formation Law"
    X_CO_PHIBBS2 = alfa_CO_PHIBBS2 * 6.3e19

    data = {
        "Id": ID_PHIBBS2,
        "Id_Number": ID_Number_PHIBBS,
        "Mh2": M_H2_PHIBBS2,
        "SFR": SFR_PHIBBS2,
        "Xco": X_CO_PHIBBS2,
        "Lco": LCO_10_PHIBBS2
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

    # SFR of the galaxies 
    SFR_ALMA_2019 = M_H2_ALMA_2019 / t_dep_ALMA_2019        # M_solar/Gyr
    SFR_ALMA_2019 = SFR_ALMA_2019 / 1e9                             # M_solar/year

    # X_CO of the galaxies 
    alfa_CO_ALMA_2019 = M_H2_ALMA_2019 / L_CO_ALMA_2019  # M⊙ pc^-2 (K-km s^-1)^-1
    # Derived X_co, using eqn 3 in the paper "A General Model for the CO-H2 Conversion Factor in Galaxies with Applications to the Star Formation Law"
    X_CO_ALMA_2019 = alfa_CO_ALMA_2019 * 6.3e19


    data = {
        "Id": ID_ALMA_2019,
        "Lco": L_CO_ALMA_2019,
        "Mh2": M_H2_ALMA_2019,
        "SFR": SFR_ALMA_2019,
        "Xco": X_CO_ALMA_2019
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

    print("average_radius_r25_times_075_LEROY: ", average_radius_r25_times_075_LEROY)


    Leroy_df = pd.DataFrame({"galaxy_name":galaxy_name_LEROY,
                             "sigma_MH2":sigma_MH2_LEROY,
                             "sigma_SFR":sigma_SFR_LEROY})  

    return (Leroy_df,
            average_radius_r25_times_075_LEROY) 
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