solar_metallicity 	= 0.02 			# solar mass fraction

def reading_output_csv_files(which_fire,
							 galaxy_name,
							 start_snapshot,
							 end_snapshot,
							 clumping_factor):

	print("I am in the function reading_output_csv_files for ", which_fire, " - ", galaxy_name)

	"""This function is used to read the XCOLDGASS Survey results 
	
	Arguments:
	----------
	which_fire: string
		fire_1 or fire_2 catalog

	galaxy_name: string
		full name of the galaxy

	start_snapshot: int
		First snapshot number

	end_snapshot: string
		Second snapshot number

	Returns:
	----------
	df: pandas data_frame
		data_frame for output for the fire galaxy of interest

	"""	

	import numpy as np
	import pandas
	import os.path

	col_list = ["#snapshot_number_array",									# 
				"redshift_array",											# 
				
				"L_co_total_array",											# [K-km s^-1 pc^2]
				"L_co_total_solar_metallicity_array",						# [K-km s^-1 pc^2]
				
				"average_metallicity_array",	 							# This will be normalized to solar metallicity - look below
				"max_metallicity_array",		 							# This will be normalized to solar metallicity - look below
				
				"sfr_array",						 						# [M☉/year]
				"young_sfr_array", 											# [M☉/year]
				
				"x_MMH_center_array",										# [kpc]
				"y_MMH_center_array",										# [kpc]
				"z_MMH_center_array",										# [kpc]
				"vx_MMH_center_array",										# [km/s]
				"vy_MMH_center_array",										# [km/s]
				"vz_MMH_center_array",										# [km/s]
				"mass_of_MMH_array",										# [M☉]
				
				"number_of_particles_within_Rmax_array",					#
				
				"total_mass_h2_array",										# Initially [1e10 M☉] but the unit will be changed to -> [M☉] 
				
				"X_co_average_array",										# [cm^-2 /K-km s^-1]
				"X_co_solar_metallicity_array",								# [cm^-2 /K-km s^-1]
				"alfa_co_average_array", 									# [M☉ pc^-2 (K-km s^-1)^-1]
				"alfa_co_solar_metallicity_average_array", 					# [M☉ pc^-2 (K-km s^-1)^-1]
				
				"average_gas_smoothing_length_array",			 			# float - [kpc]
				
				"sigma_sfr_dividing_total_area_array", 	  					# float - [M☉ year^-1 kpc^-2]				
				"sigma_mass_h2_total_area_array", 							# float - [M☉ pc^-2]
				"sigma_sfr_dividing_individual_area_array", 				# float - [M☉ year^-1 kpc^-2]
				"sigma_mass_h2_individual_area_array",  					# float - [M☉ pc^-2]
				"star_formation_rate_surface_density_mesh_average_array", 	# float - [M☉ year^-1 kpc^-2]
				"mass_h2_surface_density_mesh_average_array",  				# float - [M☉ pc^-2]
				
				"average_chi_array", 										# float - unitless
				"average_s_array", 											# float - unitless
				"average_tau_c_array"										# float - unitless
				]

	file_path = os.path.dirname(__file__)
	file_path += "/../../fire_output/"
	 
	file_name = file_path + galaxy_name + "_output_data_startsnap" + str(start_snapshot) + "_endsnap" + str(end_snapshot) + "_cf_" + str(clumping_factor) + ".csv"
	# print("file_name: ", file_name)

	df = pandas.read_csv(file_name, usecols=col_list)

	df.rename(columns={"#snapshot_number_array": 				"snapshot_number_array"}, inplace=True)



	######### Unit conversions
	df["average_metallicity_array"] = df["average_metallicity_array"]/solar_metallicity  	# [z☉]
	df["max_metallicity_array"] = df["max_metallicity_array"]/solar_metallicity				# [z☉]

	df["total_mass_h2_array"] = df["total_mass_h2_array"]*1e10								# [M☉]
	df["log_L_co_total_array"] =  np.log10(df["L_co_total_array"] * 4.9e-5)  				# [log(L☉)]
	df["log_sfr_array"] =  np.log10(df["sfr_array"])						  				# [log(M☉/year)]
	df["log_young_sfr_array"] =  np.log10(df["young_sfr_array"])						  	# [log(M☉/year)]

	return df


def reading_output_csv_files_firebox(which_fire,
									 galaxy_name,
									 snapshot_number,
									 clumping_factor):

	print("I am in the function reading_output_csv_files for ", which_fire, " - ", galaxy_name, " - ", snapshot_number)

	"""This function is used to read the XCOLDGASS Survey results 
	
	Arguments:
	----------
	which_fire: string
		fire_1 or fire_2 catalog

	galaxy_name: string
		full name of the galaxy

	start_snapshot: int
		First snapshot number

	end_snapshot: string
		Second snapshot number

	Returns:
	----------
	df: pandas data_frame
		data_frame for output for the fire galaxy of interest

	"""	

	import numpy as np
	import pandas
	import os.path

	col_list = ["#snapshot_number_array",									# 
				"redshift_array",											# 
				
				"L_co_total_array",											# [K-km s^-1 pc^2]
				"L_co_total_solar_metallicity_array",						# [K-km s^-1 pc^2]
				
				"average_metallicity_array",	 							# This will be normalized to solar metallicity - look below
				"max_metallicity_array",		 							# This will be normalized to solar metallicity - look below
				
				"sfr_array",						 						# [M☉/year]
				"young_sfr_array", 											# [M☉/year]
				
				# "x_MMH_center_array",										# [kpc]
				# "y_MMH_center_array",										# [kpc]
				# "z_MMH_center_array",										# [kpc]
				# "vx_MMH_center_array",										# [km/s]
				# "vy_MMH_center_array",										# [km/s]
				# "vz_MMH_center_array",										# [km/s]
				# "mass_of_MMH_array",										# [M☉]
				
				"number_of_particles_within_Rmax_array",					#
				
				"total_mass_h2_array",										# Initially [1e10 M☉] but the unit will be changed to -> [M☉] 
				
				"X_co_average_array",										# [cm^-2 /K-km s^-1]
				"X_co_solar_metallicity_array",								# [cm^-2 /K-km s^-1]
				"alfa_co_average_array", 									# [M☉ pc^-2 (K-km s^-1)^-1]
				"alfa_co_solar_metallicity_average_array", 					# [M☉ pc^-2 (K-km s^-1)^-1]
				
				"average_gas_smoothing_length_array",			 			# float - [kpc]
				
				"sigma_sfr_dividing_total_area_array", 	  					# float - [M☉ year^-1 kpc^-2]				
				"sigma_mass_h2_total_area_array", 							# float - [M☉ pc^-2]
				"sigma_sfr_dividing_individual_area_array", 				# float - [M☉ year^-1 kpc^-2]
				"sigma_mass_h2_individual_area_array",  					# float - [M☉ pc^-2]
				"star_formation_rate_surface_density_mesh_average_array", 	# float - [M☉ year^-1 kpc^-2]
				"mass_h2_surface_density_mesh_average_array",  				# float - [M☉ pc^-2]
				
				"average_chi_array", 										# float - unitless
				"average_s_array", 											# float - unitless
				"average_tau_c_array"										# float - unitless
				]

	file_path = os.path.dirname(__file__)
	file_path += "/../../fire_output/"
	 
	file_name = file_path + galaxy_name + "_output_data_numsnap_" + str(snapshot_number) + "_cf_" + str(clumping_factor) + ".csv"
	print("file_name: ", file_name)

	df = pandas.read_csv(file_name, usecols=col_list)

	df.rename(columns={"#snapshot_number_array": 				"snapshot_number_array"}, inplace=True)



	######### Unit conversions
	df["average_metallicity_array"] = df["average_metallicity_array"]/solar_metallicity  	# [z☉]
	df["max_metallicity_array"] = df["max_metallicity_array"]/solar_metallicity				# [z☉]

	df["total_mass_h2_array"] = df["total_mass_h2_array"]*1e10								# [M☉]
	df["log_L_co_total_array"] =  np.log10(df["L_co_total_array"] * 4.9e-5)  				# [log(L☉)]
	df["log_sfr_array"] =  np.log10(df["sfr_array"])						  				# [log(M☉/year)]
	df["log_young_sfr_array"] =  np.log10(df["young_sfr_array"])						  	# [log(M☉/year)]

	return df


def add_galaxy_name_into_DataFrame(galaxy, galaxy_name_string):

	import pandas as pd 
	import numpy as np

	galaxy_name_array = np.empty(len(galaxy['snapshot_number_array'].to_numpy()), dtype=object)
	for i in range (len(galaxy_name_array)):
		galaxy_name_array = galaxy_name_string

	galaxy['galaxy_name'] = galaxy_name_array

	return galaxy

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
	
	import pandas as pd

	Lco_m12i_Lsm = 1e10 		# [K-km s^-1 pc^2] 
	M_H2_m12i_Lsm = 2e9 		# [M☉]
	
	alpha_CO_m12i_Lsm = M_H2_m12i_Lsm / Lco_m12i_Lsm  # [M☉ (K km s^-1 pc^2)^-1]
	X_CO_m12i_Lsm = alpha_CO_m12i_Lsm * 6.3e19  	  # [cm^-2 (K km s^-1)^-1]

	SFR_m12i_Lsm = 9.210632 	# [M☉ year^-1]

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

	from astropy.io import fits
	from astropy.table import Table
	import numpy as np
	import pandas as pd


	with fits.open(filedir) as hdu:
		data_table_XCOLDGASS = Table(hdu[1].data)

	# GASS catalog ID
	ID_XCOLDGASS 				= data_table_XCOLDGASS['ID'].data

	# CO(1-0) line luminosity, aperture corrected [K km/s pc^2]
	LCO_corrected_XCOLDGASS 	= data_table_XCOLDGASS['LCO_COR'].data  # K-km s^-1 pc^2

	#  Constant Galactic CO-to-H2 conversion factor  
	XCO_XCOLDGASS 				= data_table_XCOLDGASS['XCO'].data  	# 1e20 cm^-2 (K-km s^-1)^-1
	XCO_XCOLDGASS				= XCO_XCOLDGASS*1e20					# cm^-2 (K-km s^-1)^-1

	# Total molecular gas mass [log Msun]
	LOGMH2_XCOLDGASS 			= data_table_XCOLDGASS['LOGMH2'].data
	MH2_XCOLDGASS  				= 10**LOGMH2_XCOLDGASS					# M⊙


	# SFR from WISE + GALEX when detected in both data sets
	LOGSFR_XCOLDGASS 			= data_table_XCOLDGASS['LOGSFR_BEST'].data
	SFR_XCOLDGASS 				= 10**LOGSFR_XCOLDGASS 					# M⊙/year


	data = {
		"Id": ID_XCOLDGASS,
		"Lco": LCO_corrected_XCOLDGASS,
		"Xco": XCO_XCOLDGASS,
		"Mh2": MH2_XCOLDGASS,
		"SFR": SFR_XCOLDGASS 
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

	import pandas as pd 
	import numpy as np

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

	import pandas as pd 
	import numpy as np

	# Reading the excel file 
	data_ALMA_2019 = pd.read_excel(filedir)

	# Name of the galaxies 
	ID_ALMA_2019 = data_ALMA_2019["ID"].to_numpy()

	# redshift of the galaxies 
	z_ALMA_2019 = data_ALMA_2019["z_CO"].to_numpy().astype(float)

	# L_co of the galaxies 
	L_CO_ALMA_2019 = data_ALMA_2019["L_CO(1-0)"].to_numpy().astype(float)		# 1e9 K km s^-1 pc^2
	L_CO_ALMA_2019 = L_CO_ALMA_2019 * 1e9										# K km s^-1 pc^2

	# Error_LCO of the galaxies 
	Error_LCO_ALMA_2019 = data_ALMA_2019["Error_L_CO(1-0)"].to_numpy().astype(float)
	# 1e9 K km s^-1 pc^2	

	# Total Molecular gas mass of the galaxies 
	M_H2_ALMA_2019 = data_ALMA_2019["Mmol"].to_numpy().astype(float)		# 1e10 M_solar
	M_H2_ALMA_2019 = M_H2_ALMA_2019 * 1e10									# M_solar


	# Error on the total molecular gas mass of the galaxies 
	Error_MH2_ALMA_2019 = data_ALMA_2019["Error_Mmol"].to_numpy().astype(float) 	# 1e10 M_solar
	Error_MH2_ALMA_2019 = Error_MH2_ALMA_2019 * 1e10								# M_solar


	# Depletion time of the galaxies 
	t_dep_ALMA_2019 = data_ALMA_2019["t_depl"].to_numpy().astype(float)
	# Gyr

	# SFR of the galaxies 
	SFR_ALMA_2019 = M_H2_ALMA_2019 / t_dep_ALMA_2019 		# M_solar/Gyr
	SFR_ALMA_2019 = SFR_ALMA_2019 / 1e9 							# M_solar/year

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

	import pandas as pd 
	import numpy as np 

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

	import numpy as np
	import pandas as pd

	# defining parameters
	delta_MF, alpha, beta = (1.0, 1.37, -1.74)

	# Calculating infrared luminosity
	Lir = galaxy_name["sfr_array"] * 1e10   			# M⊙

	# Calculating CO luminosity 
	log_Lco = ( np.log10(Lir) - beta ) / alpha     			# log(K km s^-1 pc^2)   Units are not same with the Lir but it is fine. It is also same in the Cariilli and Walter and also in the Li et al.

	log_Lco_in_observer_units = log_Lco						# log(K km s^-1 pc^2)
	Lco_in_observer_units = 10**log_Lco_in_observer_units  	# K km s^-1 pc^2
	Lco = 10**log_Lco * 4.9e-5								# L_solar
	log_Lco = np.log10(Lco)  								# log(L_solar)

	# Molecular gas mass calculated by assuming constant CO to H2 factor
	alpha_co = 3.2
	Mh2 = Lco_in_observer_units * alpha_co

	Li_model = pd.DataFrame(data=None)
	Li_model["L_co_total_array_in_observer_units"] =  Lco_in_observer_units 	 		# K km s^-1 pc^2
	Li_model["log_L_co_total_array_in_observer_units"] = log_Lco_in_observer_units  	# log(K km s^-1 pc^2)
	Li_model["log_L_co_total_array"] = log_Lco  								 		# log(L_solar)
	Li_model["L_co_total_array"] = Lco 								 					# L_solar
	Li_model["young_sfr_array"] = galaxy_name["young_sfr_array"] 						# M_solar/year
	Li_model["log_young_sfr_array"] = np.log10(Li_model["young_sfr_array"])      		# log(M_solar/year)
	Li_model["total_mass_h2_array"] = Mh2 												# M⊙
	Li_model["redshift_array"] = galaxy_name["redshift_array"]

	Li_model["sfr_array"] = galaxy_name["sfr_array"]									# M⊙/year
	Li_model["log_sfr_array"] = np.log10(galaxy_name["sfr_array"])  					# log(M⊙/year)

	return Li_model

###############################################################################################################################################

def average_FIRE2_calculator(galaxy_name_1,
							 galaxy_name_1_string,
							 galaxy_name_2,
							 galaxy_name_2_string,
							 galaxy_name_3,
							 galaxy_name_3_string,
							 galaxy_name_4,
							 galaxy_name_4_string,
							 galaxy_name_5,
							 galaxy_name_5_string,
							 galaxy_name_6,
							 galaxy_name_6_string):
	
	import pandas as pd 

	all_fire2 = pd.concat([galaxy_name_1, galaxy_name_2, galaxy_name_3, galaxy_name_4, galaxy_name_5, galaxy_name_6], \
						keys=[galaxy_name_1_string, galaxy_name_2_string, galaxy_name_3_string, galaxy_name_4_string, \
						galaxy_name_5_string, galaxy_name_6_string], axis=0)

	mean_fire2 = all_fire2.groupby('snapshot_number_array', as_index=False).mean()

	return all_fire2, mean_fire2