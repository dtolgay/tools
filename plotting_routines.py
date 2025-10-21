
solar_metallicity 	= 0.02 			# solar mass fraction

# Functions for snapshot_to_Lco.py

def plot_hist2d_face_on_density_gas(x_coordinates_array, 
									y_coordinates_array, 
									density_gas,
									R_max,
									redshift,
									snapshot_number,
									galaxy_name,
									do_you_want_to_save_fig=0,
									do_you_want_to_show_fig=0):
	
	print("I am in the function plot_hist2d_face_on_density_gas")

	"""This function is written to plot the 2D histogram of the x and y coordinates of the galaxy. This function is 
	useful when you want the pictorialize the galaxy at certain redshift

	Arguments:
	----------
	x_coordinates_array : array-like

	y_coordinates_array : array-like

	density_gas: array-like 
		density of the gas is being used as a weight 
		1e10M☉/kpc^2

	R_max : integer
		Maximum R to limit x and y axis

	redshift : float 
		For title

	snapshot_number : int
		For naming the output file is saved

	do_you_want_to_save_fig : int
		Naming is clear. If 1 saves, if 0 does not save

	Returns:
	----------
	
	It may save the figure depending on the value of do_you_want_to_save_fig
	
	"""

	import matplotlib
	import matplotlib.pyplot as plt
	import os 

	size_of_the_font = 32

	if (do_you_want_to_save_fig == 1):

		directory_to_save_figures 		= "post_processing_outputs/fire_2/" + str(galaxy_name) + "/hist2d_face_on_density_gas"
		save_name_of_the_figure			= "snapshot_number_" + str(snapshot_number) + ".png"

		#  Check if file exists. If not create one.
		isFile_existing = os.path.exists(directory_to_save_figures)
		if (isFile_existing == False):
			print("New directory is being created: ", directory_to_save_figures)
			os.makedirs(directory_to_save_figures)

		path_and_name_of_the_figure 	= directory_to_save_figures + "/" + save_name_of_the_figure


	#Chancing the unit of the gas particle densities
	density_gas_2 = density_gas * 1e10  					# M☉/kpc^{2}

	fig, ax = plt.subplots(figsize =(12, 10))
	ax.hist2d(
	    x_coordinates_array,
	    y_coordinates_array,
	    bins = 256,
	    weights=density_gas_2,
	    norm = matplotlib.colors.LogNorm(),
	    range = [[-R_max, R_max], [-R_max, R_max]]
	    )

	ax.set_xlabel("x [kpc]", fontsize=size_of_the_font)
	ax.set_ylabel("y [kpc]", fontsize=size_of_the_font)
	plt.title("redshift is: "+ '{0:1.4f}'.format(redshift), fontsize=size_of_the_font)
	ax.set_aspect('equal')

	if (do_you_want_to_save_fig == 1):
		plt.savefig(path_and_name_of_the_figure)
		print("Figure is saved as: ", path_and_name_of_the_figure)

	if (do_you_want_to_show_fig == 1):
		plt.show()
	

	return 0

####################################################################################################################################

def plot_hist2d_edge_on_density_gas(z_coordinates_array, 
									y_coordinates_array, 
									density_gas,
									R_max, 
									redshift, 
									snapshot_number, 
									galaxy_name,
									do_you_want_to_save_fig=0,
									do_you_want_to_show_fig=0):
	

	print("I am in the function plot_hist2d_face_on_density_gas")

	"""This function is written to plot the 2D histogram of the x and y coordinates of the galaxy. This function is 
	useful when you want the pictorialize the galaxy at certain redshift

	Arguments:
	----------
	x_coordinates_array : array-like

	y_coordinates_array : array-like

	density_gas: array-like 
		Weight for the color plot 
		1e10M☉/kpc^[2]

	R_max : integer
		Maximum R to limit x and y axis

	redshift : float 
		For title

	snapshot_number : int
		For naming the output file is saved

	do_you_want_to_save_fig : int
		Naming is clear. If 1 saves, if 0 does not save

	Returns:
	----------
	
	It may save the figure depending on the value of do_you_want_to_save_fig
	
	"""

	import matplotlib
	import matplotlib.pyplot as plt
	import os 

	size_of_the_font = 32

	if (do_you_want_to_save_fig == 1):

		directory_to_save_figures 		= "post_processing_outputs/fire_2/" + str(galaxy_name) + "/hist2d_edge_on_density_gas"
		save_name_of_the_figure			= "snapshot_number_" + str(snapshot_number) + ".png"

		#  Check if file exists. If not create one.
		isFile_existing = os.path.exists(directory_to_save_figures)
		if (isFile_existing == False):
			print("New directory is being created: ", directory_to_save_figures)
			os.makedirs(directory_to_save_figures)

		path_and_name_of_the_figure 	= directory_to_save_figures + "/" + save_name_of_the_figure


	#Chancing the unit of the gas particle densities
	density_gas_2 = density_gas * 1e10  					# M☉/kpc^{2}

	fig, ax = plt.subplots(figsize =(12, 10))
	ax.hist2d(
	    z_coordinates_array,
	    y_coordinates_array,
	    bins = 256,
	    weights=density_gas_2,
	    norm = matplotlib.colors.LogNorm(),
	    range = [[-R_max, R_max], [-R_max, R_max]]
	    )

	ax.set_xlabel("z [kpc]", fontsize=size_of_the_font)
	ax.set_ylabel("y [kpc]", fontsize=size_of_the_font)
	plt.title("redshift is: "+ '{0:1.4f}'.format(redshift), fontsize=size_of_the_font)
	ax.set_aspect('equal')

	if (do_you_want_to_save_fig == 1):
		plt.savefig(path_and_name_of_the_figure)
		print("Figure is saved as: ", path_and_name_of_the_figure)

	if (do_you_want_to_show_fig == 1):
		plt.show()
	

	return 0	

####################################################################################################################################

def plot_hist2d_face_on_and_edge_on_density_gas(x_coordinates_array,
												y_coordinates_array, 
												z_coordinates_array, 
												density_gas,
												R_max, 
												redshift, 
												snapshot_number, 
												galaxy_name,
												do_you_want_to_save_fig=0,
												do_you_want_to_show_fig=0):
	

	print("I am in the function plot_hist2d_edge_on_and_face_on_density_gas")

	"""This function is written to plot the 2D histogram of the x and y coordinates of the galaxy. This function is 
	useful when you want the pictorialize the galaxy at certain redshift

	Arguments:
	----------
	x_coordinates_array : array-like

	y_coordinates_array : array-like

	z_coordinates_array : array-like

	density_gas: array-like 
		Weight for the color plot 
		1e10M☉/kpc^[2]

	R_max : integer
		Maximum R to limit x and y axis

	redshift : float 
		For title

	snapshot_number : int
		For naming the output file is saved

	do_you_want_to_save_fig : int
		Naming is clear. If 1 saves, if 0 does not save

	Returns:
	----------
	
	It may save the figure depending on the value of do_you_want_to_save_fig
	
	"""

	import matplotlib
	import matplotlib.pyplot as plt
	import os 

	size_of_the_font = 25

	if (do_you_want_to_save_fig == 1):

		directory_to_save_figures 		= "post_processing_outputs/fire_2/" + str(galaxy_name) + "/hist2d_face_on_and_edge_on_density_gas"
		save_name_of_the_figure			= "snapshot_number_" + str(snapshot_number) + ".png"

		#  Check if file exists. If not create one.
		isFile_existing = os.path.exists(directory_to_save_figures)
		if (isFile_existing == False):
			print("New directory is being created: ", directory_to_save_figures)
			os.makedirs(directory_to_save_figures)

		path_and_name_of_the_figure 	= directory_to_save_figures + "/" + save_name_of_the_figure


	#Chancing the unit of the gas particle densities
	density_gas_2 = density_gas * 1e10  					# M☉/kpc^{2}

	fig = plt.figure(figsize=(15, 8)) # width, height
	fig.tight_layout()

	ax1 = fig.add_subplot(1,2,1)
	ax1.hist2d(
	    x_coordinates_array,
	    y_coordinates_array,
	    bins = 256,
	    weights=density_gas_2,
	    norm = matplotlib.colors.LogNorm(),
	    range = [[-R_max, R_max], [-R_max, R_max]]
	    )

	ax1.set_xlabel("x [kpc]", fontsize=size_of_the_font)
	ax1.set_ylabel("y [kpc]", fontsize=size_of_the_font)

	ax2 = fig.add_subplot(1,2,2)
	ax2.hist2d(
	    y_coordinates_array,
	    z_coordinates_array,
	    bins = 256,
	    weights=density_gas_2,
	    norm = matplotlib.colors.LogNorm(),
	    range = [[-R_max, R_max], [-R_max, R_max]]
	    )

	ax2.set_xlabel("y [kpc]", fontsize=size_of_the_font)
	ax2.set_ylabel("z [kpc]", fontsize=size_of_the_font)
	

	plt.suptitle("redshift is: "+ '{0:1.4f}'.format(redshift), fontsize=size_of_the_font)
	# ax.set_aspect('equal')

	if (do_you_want_to_save_fig == 1):
		plt.savefig(path_and_name_of_the_figure)
		print("Figure is saved as: ", path_and_name_of_the_figure)

	if (do_you_want_to_show_fig == 1):
		plt.show()
	

	return 0	

####################################################################################################################################

def d3_plot_of_particles(x_coordinates_array, y_coordinates_array, z_coordinates_array, title):

	print("I am in the function d3_plot_of_particles")

	import numpy as np
	import matplotlib.pyplot as plt

	fig, ax = plt.subplots(figsize =(12, 10))
	ax = plt.axes(projection='3d')

	ax.scatter3D(x_coordinates_array, y_coordinates_array, z_coordinates_array)
	plt.title(title)
	plt.show()

	return 0



####################################################################################################################################

def plot_histogram_plots(L_co,
						 density_gas,
						 mass_h2,
						 star_formation_rate,
						 R_max,
						 gas_x_coordinates_array,
						 gas_y_coordinates_array,
						 star_x_coordinates_array,
						 star_y_coordinates_array,
						 redshift,
						 snapshot_number,
						 galaxy_name,
						 do_you_want_to_save_fig,
						 do_you_want_to_show_fig):

	print("I am in the function plot_histogram_plots") 

	import numpy as np
	import matplotlib.pyplot as plt 
	from matplotlib import colors
	import os

	# Changing the units
	mass_h2_2 = mass_h2 * 1e10  							# M☉
	density_gas_2 = density_gas * 1e10  					# M☉/kpc^{2}

	size_of_the_font = 32
	size_of_the_font_2 = 24

	fig = plt.figure(figsize=(12,12))
	fig.tight_layout()
	
	ax1 = fig.add_subplot(2,2,1)
	ax1.set_aspect('equal')
	hh1 = ax1.hist2d(gas_x_coordinates_array, gas_y_coordinates_array, bins=256, weights=density_gas_2, norm=colors.LogNorm(vmin=1e4, vmax=1e14), range = [[-R_max, R_max], [-R_max, R_max]])
	fig.colorbar(hh1[3], ax=ax1)
	ax1.set_ylabel("y (kpc)", fontsize=size_of_the_font)
	# plt.title("$log(ρ_{gas})$ $log(M☉/kpc^{2})$", fontsize=size_of_the_font_2)
	plt.title("$log(ρ_{gas}/(M☉/kpc^{2}))$", fontsize=size_of_the_font_2)
	# plt.title("$log(ρ_{gas})$", fontsize=size_of_the_font)

	ax2 = fig.add_subplot(2,2,2)
	ax2.set_aspect('equal')
	hh2 = ax2.hist2d(gas_x_coordinates_array, gas_y_coordinates_array, bins=256, weights=mass_h2_2, norm=colors.LogNorm(vmin=1, vmax=1e8), range = [[-R_max, R_max], [-R_max, R_max]])
	fig.colorbar(hh2[3], ax=ax2)
	# ax3.set_xlabel("x position", fontsize=size_of_the_font)
	plt.title("$log(M_{H_2}/M☉)$", fontsize=size_of_the_font_2)
	# plt.title("$log(M_{H_2})$", fontsize=size_of_the_font)

	ax3 = fig.add_subplot(2,2,3)
	ax3.set_aspect('equal')
	hh3 = ax3.hist2d(gas_x_coordinates_array, gas_y_coordinates_array, bins=256, weights=L_co, norm=colors.LogNorm(vmin=1e-3, vmax=1e8), range = [[-R_max, R_max], [-R_max, R_max]])
	fig.colorbar(hh3[3], ax=ax3)
	ax3.set_ylabel("y (kpc)", fontsize=size_of_the_font)	
	ax3.set_xlabel("x (kpc)", fontsize=size_of_the_font)
	plt.title("$log(Lco/{(K-kms^{-1}pc^{2})})$", fontsize=size_of_the_font_2)
	# plt.title("log(Lco)", fontsize=size_of_the_font)

	ax4 = fig.add_subplot(2,2,4)
	ax4.set_aspect('equal')
	hh4 = ax4.hist2d(gas_x_coordinates_array, gas_y_coordinates_array, bins=256, weights=star_formation_rate, norm=colors.LogNorm(vmin=1e-5, vmax=5), range= [[-R_max, R_max], [-R_max, R_max]])
	fig.colorbar(hh4[3], ax=ax4)
	ax4.set_xlabel("x (kpc)", fontsize=size_of_the_font)
	plt.title("$log(SFR/(M☉ year^{-1})$)", fontsize=size_of_the_font_2)
	# plt.title("log(SFR)", fontsize=size_of_the_font)



	title = "redshift = "
	title += str(round(redshift,4))  
	fig.suptitle(title, fontsize=size_of_the_font)

	if (do_you_want_to_save_fig == 1):

		directory_to_save_figures 		= "post_processing_outputs/fire_2/" + str(galaxy_name) + "/histogram_plots"
		save_name_of_the_figure			= "snapshot_number_" + str(snapshot_number) + ".png"

		#  Check if file exists. If not create one.
		isFile_existing = os.path.exists(directory_to_save_figures)
		if (isFile_existing == False):
			print("New directory is being created: ", directory_to_save_figures)
			os.makedirs(directory_to_save_figures)

		path_and_name_of_the_figure 	= directory_to_save_figures + "/" + save_name_of_the_figure

		plt.savefig(path_and_name_of_the_figure)
		print("Figure is saved as: ", path_and_name_of_the_figure)
		
		


	if (do_you_want_to_show_fig == 1):
		plt.show()
	
	return 0 


####################################################################################################################################

def plot_sfr_and_star_particles(mass_star,
								star_formation_rate,
								star_x_coordinates_array,
								star_y_coordinates_array,
								gas_x_coordinates_array,
								gas_y_coordinates_array,
								R_max,
								redshift,
								snapshot_number,
								plotting_output_file_path,
								do_you_want_to_save_fig=0,
								do_you_want_to_show_fig=0):

	import numpy as np 
	import matplotlib.pyplot as plt 
	from matplotlib import colors

	print("I am in the function plot_sfr_and_star_particles") 

	# Chancing the units
	mass_star = mass_star * 1e10  # M☉


	size_of_the_font = 28
	size_of_the_font_2 = 24

	fig = plt.figure(figsize=(12,12))
	fig.tight_layout()

	ax1 = fig.add_subplot(2,1,1)
	ax1.set_aspect('equal')
	hh1 = ax1.hist2d(star_x_coordinates_array, star_y_coordinates_array, bins=256, weights=mass_star, norm=colors.LogNorm(vmin=1e3, vmax=1e9), range= [[-R_max, R_max], [-R_max, R_max]])	
	fig.colorbar(hh1[3], ax=ax1)
	# ax1.set_xlabel("x (kpc)", fontsize=size_of_the_font)
	ax1.set_ylabel("y (kpc)", fontsize=size_of_the_font)	
	plt.title("log(M*/M☉))", fontsize=size_of_the_font_2)


	ax2 = fig.add_subplot(2,1,2)
	ax2.set_aspect('equal')
	hh2 = ax2.hist2d(gas_x_coordinates_array, gas_y_coordinates_array, bins=256, weights=star_formation_rate, norm=colors.LogNorm(vmin=1e-5, vmax=5), range= [[-R_max, R_max], [-R_max, R_max]])
	fig.colorbar(hh2[3], ax=ax2)
	ax2.set_xlabel("x (kpc)", fontsize=size_of_the_font)
	ax2.set_ylabel("y (kpc)", fontsize=size_of_the_font)	
	plt.title("$log(SFR/(M☉ year^{-1})$)", fontsize=size_of_the_font_2)
	# plt.title("log(SFR)", fontsize=size_of_the_font)

	title = "redshift = "
	title += str(round(redshift,4))  
	fig.suptitle(title, fontsize=size_of_the_font)


	if (do_you_want_to_save_fig == 1): 		
		directory_in_params_file   		= plotting_output_file_path
		directory_to_save_figures 		= directory_in_params_file + "/sfr_and_star_particles"
		save_name_of_the_figure			= "snapshot_number_" + str(snapshot_number) + ".png"
		path_and_name_of_the_figure 	= directory_to_save_figures + "/" + save_name_of_the_figure
		plt.savefig(path_and_name_of_the_figure)
		print("Figure is saved as: ", path_and_name_of_the_figure)


	if (do_you_want_to_show_fig == 1):
		plt.show()
	

	return 0


####################################################################################################################################################################

def plot_temperature_vs_Rgas(temperature_gas,
							 R_gas):

	import matplotlib.pyplot as plt 

	fig = plt.figure(figsize=(8,8))

	ax1 = fig.add_subplot(1,1,1)
	ax1.plot(R_gas, temperature_gas, "r.")
	ax1.set_xlabel("Rgas (kpc)")
	ax1.set_ylabel("Gas Temperature (K)")
	plt.show()
	
	return 0

####################################################################################################################################################################

def plot_fh2_vs_Rgas_for_two_different_calculations(fh2_original, fh2_temperature, R_gas):

	import matplotlib.pyplot as plt 

	fig = plt.figure(figsize=(8,8))

	ax1 = fig.add_subplot(1,2,1)
	ax2 = fig.add_subplot(1,2,2)

	ax1.plot(R_gas, fh2_original, "r.")

	ax2.plot(R_gas, fh2_temperature, "b.")

	ax1.set_xlabel("Rgas (kpc)")
	ax2.set_xlabel("Rgas (kpc)")
	
	ax1.set_ylabel("fh2")

	plt.show()
	
	return 0

####################################################################################################################################################################

def plot_fh2_vs_temperature(fh2_krumholz, fh2_norm, temperature_gas):

	import matplotlib.pyplot as plt 

	fig = plt.figure(figsize=(8,8))

	ax1 = fig.add_subplot(1,2,1)
	ax1.plot(temperature_gas, fh2_krumholz, "r.")
	ax1.set_ylabel("fh2_krumholz")
	ax1.set_xlabel("Gas Temperature (K)")
	plt.xlim([0,1e5])

	ax2 = fig.add_subplot(1,2,2)
	ax2.plot(temperature_gas, fh2_norm, "r.")
	ax2.set_ylabel("fh2_norm")
	ax2.set_xlabel("Gas Temperature (K)")
	plt.xlim([0,1e5])
	

	plt.show()
	
	return 0

####################################################################################################################################################################

def plot_smoothing_length_vs_Rgal(smoothing_length_gas, R_gas):


	import matplotlib.pyplot as plt 

	size_of_the_font = 24

	fig = plt.figure(figsize=(8,8))

	ax1 = fig.add_subplot(1,1,1)
	ax1.plot(R_gas, smoothing_length_gas, "r.")
	ax1.set_ylabel("smoothing_length_gas (kpc)", fontsize=size_of_the_font)
	ax1.set_xlabel("R_gas (kpc)", fontsize=size_of_the_font)
	plt.xlim([0,20])

	plt.show()
	
	return 0

####################################################################################################################################################################

def plot_smoothing_length_vs_density(smoothing_length_gas, density_gas):

	import matplotlib.pyplot as plt 

	fig = plt.figure(figsize=(8,8))

	size_of_the_font = 24

	ax1 = fig.add_subplot(1,1,1)
	ax1.plot(density_gas, smoothing_length_gas, "r.")
	ax1.set_ylabel("smoothing_length_gas (kpc)", fontsize=size_of_the_font)
	ax1.set_xlabel("density_gas (1e10M☉ kpc^-3)", fontsize=size_of_the_font)
	plt.xlim([0,20])

	plt.show()
	
	return 0

####################################################################################################################################################################

def d3_plot_of_particles(x_coordinates_array, y_coordinates_array, z_coordinates_array, title):

	import numpy as np
	import matplotlib.pyplot as plt

	fig, ax = plt.subplots(figsize = (12, 10))
	ax = plt.axes(projection='3d')

	ax.scatter3D(x_coordinates_array, y_coordinates_array, z_coordinates_array)
	plt.title(title)
	plt.show()

	return 0

####################################################################################################################################################################
####################################################################################################################################################################
####################################################################################################################################################################
####################################################################################################################################################################
####################################################################################################################################################################

# Functions for reading_and_plotting_snapshot_to_Lco.py


def plot_position_of_the_center_of_mmh_vs_redshift(x_MMH_center_array, y_MMH_center_array, z_MMH_center_array, redshift_array): 

	print("I am in the function plot_position_of_the_center_of_mmh_vs_redshift")
	import matplotlib.pyplot as plt 

	plt.plot(redshift_array, x_MMH_center_array, 'bo')
	plt.plot(redshift_array, x_MMH_center_array, 'b--')

	plt.plot(redshift_array, y_MMH_center_array, 'ro')
	plt.plot(redshift_array, y_MMH_center_array, 'r--')

	plt.plot(redshift_array, z_MMH_center_array, 'go')
	plt.plot(redshift_array, z_MMH_center_array, 'g--')


	plt.xlabel("redshift")
	plt.ylabel("kpc")
	plt.legend(["x position", "", "y position", "", "z position", ""])
	plt.show()

	return 0

####################################################################################################################################################################

def plot_velocity_of_the_center_of_mmh_vs_redshift(vx_MMH_center_array, vy_MMH_center_array, vz_MMH_center_array, redshift_array): 

	print("I am in the function plot_velocity_of_the_center_of_mmh_vs_redshift")
	import matplotlib.pyplot as plt 

	plt.plot(redshift_array, vx_MMH_center_array, 'bo')
	plt.plot(redshift_array, vx_MMH_center_array, 'b--')

	plt.plot(redshift_array, vy_MMH_center_array, 'ro')
	plt.plot(redshift_array, vy_MMH_center_array, 'r--')

	plt.plot(redshift_array, vz_MMH_center_array, 'go')
	plt.plot(redshift_array, vz_MMH_center_array, 'g--')


	plt.xlabel("redshift")
	plt.ylabel("km/s")
	plt.legend(["x velocity", "", "y velocity", "", "z velocity", ""])
	plt.show()

	return 0

####################################################################################################################################################################

def plot_number_of_particles_vs_redshift(number_of_particles_within_Rmax, redshift_array): 

	print("I am in the function plot_number_of_particles_vs_redshift")
	import matplotlib.pyplot as plt 

	plt.plot(redshift_array, number_of_particles_within_Rmax, 'bo')
	plt.plot(redshift_array, number_of_particles_within_Rmax, 'b--')

	plt.xlabel("redshift")
	plt.ylabel("number_of_particles_within_Rmax")
	plt.show()

	return 0


####################################################################################################################################################################

def plot_log10_Lco_and_SFR_vs_redshift(L_co_total_divided_by_Lsun_array, SFR_array,  SFR_array_label, redshift_array):

	print("I am in the function plot_log10_Lco_and_SFR_vs_redshift")

	import numpy as np
	import matplotlib.pyplot as plt 

	L_co_total_divided_by_Lsun_log10_array = np.zeros([len(L_co_total_divided_by_Lsun_array)])
	L_co_total_divided_by_Lsun_log10_array = np.log10(L_co_total_divided_by_Lsun_array)	

	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	ax1.plot(redshift_array, L_co_total_divided_by_Lsun_log10_array, 'ro')
	ax1.plot(redshift_array, L_co_total_divided_by_Lsun_log10_array, 'r--')
	ax1.set_ylabel('log10(L_co / L_sun)', color="red")

	ax2 = ax1.twinx()
	ax2.plot(redshift_array, SFR_array, 'bo')
	ax2.plot(redshift_array, SFR_array, 'b--')
	ax2.set_ylabel(SFR_array_label, color="blue")	

	plt.xlabel("redshift")
	plt.show()

	return 0	

####################################################################################################################################################################

def plot_log10_Lco_and_2_different_SFR_calculation_vs_redshift(L_co_total_array, SFR_array_1,  SFR_array_1_label,\
 SFR_array_2,  SFR_array_2_label, redshift_array, title_string, clumping_factor, do_you_want_to_save_fig = 0):

	print("I am in the function plot_log10_Lco_and_2_different_SFR_calculation_vs_redshift")
	
	import numpy as np
	import matplotlib.pyplot as plt 


	L_co_total_array = L_co_total_array * 4.9e-5  # Lsun Units

	# TO save figure: 
	import os

	current_directory   			= os.getcwd()
	directory_to_save_figures 		= current_directory + "/figures_and_data"
	save_name_of_the_figure 		= "z"  + '{0:1.1f}'.format(redshift_array[0]) + "_to_" + "z" + '{0:1.1f}'.format(redshift_array[len(redshift_array)-1]) + "_cf" + str(clumping_factor) + ".png"
	path_and_name_of_the_figure 	= directory_to_save_figures + "/" + save_name_of_the_figure


	L_co_total_divided_by_Lsun_log10_array = np.zeros([len(L_co_total_array)])
	L_co_total_divided_by_Lsun_log10_array = np.log10(L_co_total_array)	

	L_co_total_divided_by_Lsun_log10_array[L_co_total_divided_by_Lsun_log10_array<(-10)] = float("nan")

	fig = plt.figure(figsize=(12, 8))
	ax1 = fig.add_subplot(111)
	ax1.plot(redshift_array, L_co_total_divided_by_Lsun_log10_array, 'ro')
	ax1.plot(redshift_array, L_co_total_divided_by_Lsun_log10_array, 'r--')
#	ax1.invert_xaxis()
	ax1.set_ylabel('log$_{10}$($L_{co}$ / $L_{⊙}$)', color="red")
	ax1.set_xlabel("z")
#	plt.ylim([2,6])

	ax2 = ax1.twinx()
	ax2.plot(redshift_array, SFR_array_1, 'bo')
	ax2.plot(redshift_array, SFR_array_1, 'b--')
#	ax2.invert_xaxis()
	ax2.set_ylabel("SFR ($M_{⊙}$/year)", color="blue")	

	ax2.plot(redshift_array, SFR_array_2, 'go')
	ax2.plot(redshift_array, SFR_array_2, 'g--')
#	ax2.invert_xaxis()

	plt.legend([SFR_array_1_label, "", SFR_array_2_label, ""])
	
	plt.title(title_string)


	if (do_you_want_to_save_fig==1):
		plt.savefig(path_and_name_of_the_figure)
		print("Figure is saved as: ", path_and_name_of_the_figure)

	plt.show()


	return 0

####################################################################################################################################################################

def plot_Lco_and_2_different_SFR_calculation_vs_redshift(L_co_total_array, SFR_array_1,  SFR_array_1_label,\
 SFR_array_2,  SFR_array_2_label, redshift_array, title_string, clumping_factor, do_you_want_to_save_fig = 0):

	print("I am in the function plot_log10_Lco_and_2_different_SFR_calculation_vs_redshift")
	
	import numpy as np
	import matplotlib.pyplot as plt 


	L_co_total_array = L_co_total_array * 4.9e-5  # Lsun Units

	# TO save figure: 
	import os

	current_directory   			= os.getcwd()
	directory_to_save_figures 		= current_directory + "/figures_and_data"
	save_name_of_the_figure 		= "z"  + '{0:1.1f}'.format(redshift_array[0]) + "_to_" + "z" + '{0:1.1f}'.format(redshift_array[len(redshift_array)-1]) + "_cf" + str(clumping_factor) 
	save_name_of_the_figure 		+= "_Lco_and_2_different_SFR_calculation_vs_redshift.png"
	path_and_name_of_the_figure 	= directory_to_save_figures + "/" + save_name_of_the_figure


	# L_co_total_divided_by_Lsun_log10_array[L_co_total_divided_by_Lsun_log10_array<(-10)] = float("nan")

	fig = plt.figure(figsize=(12, 8))
	ax1 = fig.add_subplot(111)
	ax1.plot(redshift_array, L_co_total_array, 'ro')
	ax1.plot(redshift_array, L_co_total_array, 'r--')
#	ax1.invert_xaxis()
	ax1.set_ylabel('$L_{co}$ / $L_{⊙}$', color="red", fontsize=18)
	ax1.set_xlabel("z", fontsize=18)
	ax1.set_yscale('log', base=10)
	plt.ylim([1e-4,1e6])
	plt.grid(True)

	ax2 = ax1.twinx()
	ax2.plot(redshift_array, SFR_array_1, 'bo')
	ax2.plot(redshift_array, SFR_array_1, 'b--')
#	ax2.invert_xaxis()
	ax2.set_ylabel("SFR ($M_{⊙}$/year)", color="blue", fontsize=18)	
	ax2.plot(redshift_array, SFR_array_2, 'go')
	ax2.plot(redshift_array, SFR_array_2, 'g--')
#	ax2.invert_xaxis()
	ax2.set_yscale('log', base=10)
	plt.ylim([1e-4,1e6])
	plt.grid(True)
	plt.xlim([0,np.max(redshift_array)])

	plt.legend([SFR_array_1_label, "", SFR_array_2_label, ""])
	
	#TODO Uncomment below
	# plt.title(title_string)


	if (do_you_want_to_save_fig==1):
		plt.savefig(path_and_name_of_the_figure)
		print("Figure is saved as: ", path_and_name_of_the_figure)

	plt.show()

	return 0

####################################################################################################################################################################

def plot_Lco_and_SFR_and_metallicity_calculation_vs_redshift(L_co_total_array, 
															 SFR_array,  
															 metallicity_array,   
															 redshift_array, 
															 title_string, 
															 clumping_factor, 
															 do_you_want_to_save_fig = 0):

	print("I am in the function plot_log10_Lco_and_2_different_SFR_calculation_vs_redshift")
	
	import numpy as np
	import matplotlib.pyplot as plt 


	L_co_total_array = L_co_total_array * 4.9e-5  # Lsun Units
	normalized_metallicity = metallicity_array / solar_metallicity

	# TO save figure: 
	import os

	current_directory   			= os.getcwd()
	directory_to_save_figures 		= current_directory + "/figures_and_data"
	save_name_of_the_figure 		= "z"  + '{0:1.1f}'.format(redshift_array[0]) + "_to_" + "z" + '{0:1.1f}'.format(redshift_array[len(redshift_array)-1]) + "_cf" + str(clumping_factor) 
	save_name_of_the_figure 		+= "_Lco_SFR_metallicity_calculation_vs_redshift.png"
	path_and_name_of_the_figure 	= directory_to_save_figures + "/" + save_name_of_the_figure


	# L_co_total_divided_by_Lsun_log10_array[L_co_total_divided_by_Lsun_log10_array<(-10)] = float("nan")

	fig = plt.figure(figsize=(12, 20))

	# Lco and SFR vs Redshift
	ax1 = fig.add_subplot(211)
	ax1.plot(redshift_array, L_co_total_array, 'ro')
	ax1.plot(redshift_array, L_co_total_array, 'r--')
#	ax1.invert_xaxis()
	ax1.set_ylabel('$L_{co}$ / $L_{⊙}$', color="red", fontsize=18)
	# ax1.set_xlabel("z", fontsize=18)
	ax1.set_yscale('log', base=10)
	plt.ylim([1e-4,1e6])
	plt.grid(True)

	ax2 = ax1.twinx()
	ax2.plot(redshift_array, SFR_array, 'bo')
	ax2.plot(redshift_array, SFR_array, 'b--')
#	ax2.invert_xaxis()
	ax2.set_ylabel("SFR ($M_{⊙}$/year)", color="blue", fontsize=18)	
#	ax2.invert_xaxis()
	ax2.set_yscale('log', base=10)
	plt.ylim([1e-4,1e6])
	plt.grid(True)
	plt.xlim([0,np.max(redshift_array)])
	plt.title(title_string)	

	# Lco and metallicity vs Redshift
	ax3 = fig.add_subplot(212)
	ax3.plot(redshift_array, L_co_total_array, 'ro')
	ax3.plot(redshift_array, L_co_total_array, 'r--')
#	ax1.invert_xaxis()
	ax3.set_ylabel('$L_{co}$ / $L_{⊙}$', color="red", fontsize=18)
	ax3.set_xlabel("redshift", fontsize=18)
	ax3.set_yscale('log', base=10)
	plt.ylim([1e-4,1e6])
	plt.grid(True)

	ax4 = ax3.twinx()
	ax4.plot(redshift_array, normalized_metallicity, 'go')
	ax4.plot(redshift_array, normalized_metallicity, 'g--')	
	ax4.set_ylabel("Metallicity", color="green", fontsize=18)
	ax4.set_yscale('log', base=10)
	plt.xlim([0,np.max(redshift_array)])




	if (do_you_want_to_save_fig==1):
		plt.savefig(path_and_name_of_the_figure)
		print("Figure is saved as: ", path_and_name_of_the_figure)

	plt.show()

	return 0



####################################################################################################################################################################


def plot_Lco_for_different_clumping_factors_and_ratio_between(Lco_cf1_array, Lco_cf2_array, Lco_cf1_label, Lco_cf2_label, redshift_array):

	print("I am in the function plot_Lco_for_different_clumping_factors_and_ratio_between")

	ratio = Lco_cf1_array/Lco_cf2_array

	import numpy as np
	import matplotlib.pyplot as plt 

	L_co_total_divided_by_Lsun_log10_array_cf1 = np.zeros([len(Lco_cf1_array)])
	L_co_total_divided_by_Lsun_log10_array_cf1 = np.log10(Lco_cf1_array)		


	L_co_total_divided_by_Lsun_log10_array_cf2 = np.zeros([len(Lco_cf2_array)])
	L_co_total_divided_by_Lsun_log10_array_cf2 = np.log10(Lco_cf2_array)		


	ratio_array = np.zeros([len(ratio)])
	ratio_array = np.log10(ratio)

	axis2_string = "log10(" + Lco_cf1_label + "/" + Lco_cf2_label + ")"

	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	ax1.plot(redshift_array, L_co_total_divided_by_Lsun_log10_array_cf1, 'ro')
	ax1.plot(redshift_array, L_co_total_divided_by_Lsun_log10_array_cf1, 'r--')

	ax1.plot(redshift_array, L_co_total_divided_by_Lsun_log10_array_cf2, 'bo')
	ax1.plot(redshift_array, L_co_total_divided_by_Lsun_log10_array_cf2, 'b--')
	ax1.set_xlabel("redshift")
	ax1.set_ylabel("log10()")

	plt.legend([Lco_cf1_label,"",Lco_cf2_label,""])

	ax2 = ax1.twinx()
	ax2.plot(redshift_array, ratio_array, 'go')
	ax2.plot(redshift_array, ratio_array, 'g--')
	ax2.set_ylabel(axis2_string, color="green")	

	plt.show()

	return 0

####################################################################################################################################################################

def plot_response_function_SFRs_vs_Lco(L_co_total_divided_by_Lsun_array,
									   SFR_array_1,  
									   SFR_array_1_label,
									   SFR_array_2,  
									   SFR_array_2_label, 
									   title_string, 
									   redshift_array, 
									   clumping_factor, 
									   do_you_want_linear_fit=0, 
									   do_you_want_to_save_fig=0):

	print("I am in the function plot_response_function_SFRs_vs_Lco")

	import numpy as np
	import matplotlib.pyplot as plt 


	#############
	# TO save figure: 
	import os

	current_directory   			= os.getcwd()
	directory_to_save_figures 		= current_directory + "/figures_and_data"
	save_name_of_the_figure 		= "response_function_SFR_Lco_" + "z"  + '{0:1.1f}'.format(redshift_array[0]) + "_to_" + "z" + '{0:1.1f}'.format(redshift_array[len(redshift_array)-1]) + "_cf" + str(clumping_factor) + ".png"
	path_and_name_of_the_figure 	= directory_to_save_figures + "/" + save_name_of_the_figure
	#############

	#############
	# Data cleaning
	L_co_total_divided_by_Lsun_array[np.log10(L_co_total_divided_by_Lsun_array)<(-10)] = float("nan")
	#############

	L_co_total_divided_by_Lsun_log10_array = np.zeros([len(L_co_total_divided_by_Lsun_array)])
	L_co_total_divided_by_Lsun_log10_array = np.log10(L_co_total_divided_by_Lsun_array)	

	log10_SFR_array_1 = np.zeros([len(SFR_array_1)])
	log10_SFR_array_1 = np.log10(SFR_array_1)	

	log10_SFR_array_2 = np.zeros([len(SFR_array_2)])
	log10_SFR_array_2 = np.log10(SFR_array_2)	

	# # TODO - Delete below
	# print("L_co_total_divided_by_Lsun_log10_array[valid]: ", L_co_total_divided_by_Lsun_log10_array[valid])
	# print("log10_SFR_array_1[valid]: ", log10_SFR_array_1[valid])
	# print("fit_coefficients: ", fit_coefficients)
	# print("a: ", a)
	# print("b: ", b)

	fig = plt.figure(figsize=(12, 12))
	fig.suptitle(title_string, fontsize=20)
	ax1 = fig.add_subplot(2,1,1)
	ax1.plot(L_co_total_divided_by_Lsun_log10_array, log10_SFR_array_1, 'bo', label="FIRE Output")
	# ax1.set_ylabel('$log_{10}$(SFR)', color="red")
	if (do_you_want_linear_fit==1):

		valid = ~(np.isnan(L_co_total_divided_by_Lsun_log10_array) | 
				  np.isnan(log10_SFR_array_1) | 
				  np.isinf(L_co_total_divided_by_Lsun_log10_array) |
				  np.isinf(log10_SFR_array_1))


		fit_coefficients  = np.polyfit(x=L_co_total_divided_by_Lsun_log10_array[valid], y=log10_SFR_array_1[valid], deg=1)
		a = fit_coefficients[0]
		b = fit_coefficients[1]
		log10_L_co_fitted = np.linspace(np.min(L_co_total_divided_by_Lsun_log10_array[valid]),
								 		np.max(L_co_total_divided_by_Lsun_log10_array[valid]), 
										1000)
		log10_SFR_fitted = a * log10_L_co_fitted + b
		SFR_label_1 = "$log_{10}$(" + SFR_array_1_label + ")"
		linear_fit_label_1 = SFR_label_1 + " = " + '{0:1.5E}'.format(a) + " * $log_{10}$($L_{co}$ / $L_{⊙})$ " + '{0: 1.5f}'.format(b)
		lns1 = ax1.plot(log10_L_co_fitted, log10_SFR_fitted, 'k--', label=linear_fit_label_1)
		ax1.set_ylabel('$log_{10}$(SFR ($M_{⊙}$/year))', fontsize=18)
		# ax1.set_xlabel('$log_{10}$($L_{co}$ / $L_{⊙})$')
		plt.ylim([-3, 3])
		plt.legend(loc="upper left")

	ax2 = fig.add_subplot(2,1,2)
	ax2.plot(L_co_total_divided_by_Lsun_log10_array, log10_SFR_array_2, 'go', label="FIRE Output")
	# ax2.set_ylabel(SFR_array_2_label, color="blue")
	if (do_you_want_linear_fit==1):

		valid = ~(np.isnan(L_co_total_divided_by_Lsun_log10_array) | 
				  np.isnan(log10_SFR_array_2) | 
				  np.isinf(L_co_total_divided_by_Lsun_log10_array) |
				  np.isinf(log10_SFR_array_2))


		fit_coefficients  = np.polyfit(x=L_co_total_divided_by_Lsun_log10_array[valid], y=log10_SFR_array_2[valid], deg=1)
		a = fit_coefficients[0]
		b = fit_coefficients[1]
		log10_L_co_fitted = np.linspace(np.min(L_co_total_divided_by_Lsun_log10_array[valid]),
								 		np.max(L_co_total_divided_by_Lsun_log10_array[valid]), 
										1000)
		log10_SFR_fitted = a * log10_L_co_fitted + b

		SFR_label_2 = "$log_{10}$(" + SFR_array_2_label + ")"
		linear_fit_label_2 = SFR_label_2 + " = " + '{0:1.5E}'.format(a) + " * $log_{10}$($L_{co}$ / $L_{⊙})$ " + '{0: 1.5f}'.format(b)
		lns2 = ax2.plot(log10_L_co_fitted, log10_SFR_fitted, 'k--', label=linear_fit_label_2)
		ax2.set_ylabel('$log_{10}$(SFR ($M_{⊙}$/year))', fontsize=18)		
		ax2.set_xlabel('$log_{10}$($L_{co}$ / $L_{⊙})$', fontsize=18)
		plt.ylim([-3, 3])
		plt.legend(loc="upper left")
	
	if (do_you_want_to_save_fig==1):
		plt.savefig(path_and_name_of_the_figure)
		print("Figure is saved as: ", path_and_name_of_the_figure)


	plt.show()

	return 0

####################################################################################################################################################################

def plot_SFR_vs_Mhalo(SFR_array_1,  SFR_array_1_label, SFR_array_2,  SFR_array_2_label, \
	mass_of_mmh, mass_of_mmh_label, title_string, redshift_array, clumping_factor, do_you_want_to_save_fig=0):

	print("I am in the function plot_SFR_vs_Mhalo")

	##############
	# TO save figure: 
	import os
	#To name the saved file
	current_directory   			= os.getcwd()
	directory_to_save_figures 		= current_directory + "/figures_and_data"
	save_name_of_the_figure 		= "SFRs_vs_Mhalo_for_" + "z"  + '{0:1.1f}'.format(redshift_array[0]) + "_to_" + "z" + '{0:1.1f}'.format(redshift_array[len(redshift_array)-1]) + "_cf" + str(clumping_factor) + ".pdf"
	path_and_name_of_the_figure 	= directory_to_save_figures + "/" + save_name_of_the_figure
	##############

	import numpy as np
	import matplotlib.pyplot as plt 

	fig = plt.figure(figsize=(12, 8))
	ax1 = fig.add_subplot(111)
	ax1.plot(mass_of_mmh, SFR_array_1, 'ro')
	ax1.plot(mass_of_mmh, SFR_array_1, 'r--')
	ax1.set_ylabel(SFR_array_1_label, color="red")
	ax1.set_xlabel(mass_of_mmh_label)

	ax2 = ax1.twinx()
	ax2.plot(mass_of_mmh, SFR_array_2, 'bo')
	ax2.plot(mass_of_mmh, SFR_array_2, 'b--')
	ax2.set_ylabel(SFR_array_2_label, color="blue")

	plt.title(title_string)


	if (do_you_want_to_save_fig==1):
		plt.savefig(path_and_name_of_the_figure)
		print("Figure is saved as: ", path_and_name_of_the_figure)


	plt.show()

	return 0

####################################################################################################################################################################


def plot_metallicity_vs_redshift(metallicity_array_1,  metallicity_array_1_label, metallicity_array_2,  metallicity_array_2_label, \
	redshift_array, reshift_label, title_string, clumping_factor, do_you_want_to_save_fig=0):

	print("I am in the function plot_metallicity_vs_redshift")

	##############
	# TO save figure: 
	import os
	#To name the saved file
	current_directory   			= os.getcwd()
	directory_to_save_figures 		= current_directory + "/figures_and_data"
	save_name_of_the_figure 		= "metallicity_for_" + "z"  + '{0:1.1f}'.format(redshift_array[0]) + "_to_" + "z" + '{0:1.1f}'.format(redshift_array[len(redshift_array)-1]) + "_cf" + str(clumping_factor) + ".pdf"
	path_and_name_of_the_figure 	= directory_to_save_figures + "/" + save_name_of_the_figure
	##############

	import numpy as np
	import matplotlib.pyplot as plt 

	fig = plt.figure(figsize=(12, 8))
	ax1 = fig.add_subplot(111)
	ax1.plot(redshift_array, metallicity_array_1, 'ro')
	ax1.plot(redshift_array, metallicity_array_1, 'r--')
	ax1.set_ylabel(metallicity_array_1_label, color="red")
	ax1.set_xlabel(reshift_label)

	ax2 = ax1.twinx()
	ax2.plot(redshift_array, metallicity_array_2, 'bo')
	ax2.plot(redshift_array, metallicity_array_2, 'b--')
	ax2.set_ylabel(metallicity_array_2_label, color="blue")
	ax2.invert_xaxis()

	plt.title(title_string)


	if (do_you_want_to_save_fig==1):
		plt.savefig(path_and_name_of_the_figure)
		print("Figure is saved as: ", path_and_name_of_the_figure)


	plt.show()

	return 0


####################################################################################################################################################################

def plot_Lco_vs_metallicity(L_co_total_divided_by_Lsun_array, metallicity_array, metallicity_label,\
	title_string, redshift_array, clumping_factor, do_you_want_to_save_fig=0):

	import numpy as np
	import matplotlib.pyplot as plt 


	print("I am in the function plot_Lco_vs_metallicity")


	##############
	# TO save figure: 
	import os
	#To name the saved file
	current_directory   			= os.getcwd()
	directory_to_save_figures 		= current_directory + "/figures_and_data"
	save_name_of_the_figure 		= "Lco_vs_metallicity_for_" + "z"  + '{0:1.1f}'.format(redshift_array[0]) + "_to_" + "z" + '{0:1.1f}'.format(redshift_array[len(redshift_array)-1]) + "_cf" + str(clumping_factor) + ".pdf"
	path_and_name_of_the_figure 	= directory_to_save_figures + "/" + save_name_of_the_figure
	##############

	#Taking the logarithim of the Lco 
	L_co_total_divided_by_Lsun_log10_array = np.zeros([len(L_co_total_divided_by_Lsun_array)])
	L_co_total_divided_by_Lsun_log10_array = np.log10(L_co_total_divided_by_Lsun_array)	

	fig = plt.figure(figsize=(12, 8))
	ax1 = fig.add_subplot(111)
	ax1.plot(metallicity_array, L_co_total_divided_by_Lsun_log10_array, 'ro')
#	ax1.plot(metallicity_array, L_co_total_divided_by_Lsun_log10_array, 'r--')
	ax1.set_ylabel('$log_{10}$($L_{co}$ / $L_{⊙})$')
	ax1.set_xlabel(metallicity_label)

	plt.title(title_string)

	if (do_you_want_to_save_fig==1):
		plt.savefig(path_and_name_of_the_figure)
		print("Figure is saved as: ", path_and_name_of_the_figure)


	plt.show()

	return 0


####################################################################################################################################################################

def plot_comparison_FIRE_and_observations_according_to_SFR(L_co_total_array,
														   mass_h2,
														   X_CO_average,
														   SFR,

														   LCO_XCOLDGASS,
														   MH2_XCOLDGASS,
														   XCO_XCOLDGASS,
														   SFR_XCOLDGASS,

														   LCO_10_PHIBBS2,
														   M_H2_PHIBBS2, #M_H2 is the molecular gas mass
														   X_CO_PHIBBS2,
														   SFR_PHIBBS2,

														   L_CO_ALMA_2019,
														   M_H2_ALMA_2019,
														   X_CO_ALMA_2019,
														   SFR_ALMA_2019,

														   redshift_array,
														   clumping_factor,
														   do_you_want_to_save_fig=0):


	import matplotlib.pyplot as plt
	import numpy as np


	##############
	# TO save figure: 
	import os
	#To name the saved file
	current_directory   			= os.getcwd()
	directory_to_save_figures 		= current_directory + "/figures_and_data"
	save_name_of_the_figure 		= "Lco_FIRE_and_Observations_vs_SFR_" + "z"  + '{0:1.1f}'.format(redshift_array[0]) + "_to_" + "z" + '{0:1.1f}'.format(redshift_array[len(redshift_array)-1]) + "_cf" + str(clumping_factor) + ".png"
	path_and_name_of_the_figure 	= directory_to_save_figures + "/" + save_name_of_the_figure
	##############


	fig = plt.figure(figsize=(15,12))
	ax1 = fig.add_subplot(3,2,1)
	ax1.plot(SFR_XCOLDGASS, LCO_XCOLDGASS,'o', mfc='none')
	ax1.plot(SFR_PHIBBS2, LCO_10_PHIBBS2, 'x', mfc='none')
	ax1.plot(SFR_ALMA_2019, L_CO_ALMA_2019, 's', mfc='none')
	ax1.plot(SFR, L_co_total_array,'r.')
	ax1.set_yscale('log', base=10)
	ax1.set_xscale('log', base=10)
	# plt.xlabel('SFR [M$_{⊙}$/year]')
	plt.ylim([1e2,1e11])
	plt.ylabel('Lco [K km/s pc^2]')
	plt.grid(True)


	SFR_XCOLDGASS_greater_than_zero_indices = np.where(SFR_XCOLDGASS>1)
	SFR_XCOLDGASS_greater_than_zero_indices = SFR_XCOLDGASS_greater_than_zero_indices[0]


	SFR_greater_than_zero_indices = np.where(SFR>1)
	SFR_greater_than_zero_indices = SFR_greater_than_zero_indices[0]


	ax2 = fig.add_subplot(3,2,2)
	ax2.plot(SFR_XCOLDGASS, LCO_XCOLDGASS,'o', mfc='none')
	ax2.plot(SFR_PHIBBS2, LCO_10_PHIBBS2, 'x', mfc='none')
	ax2.plot(SFR_ALMA_2019, L_CO_ALMA_2019, 's', mfc='none')	
	ax2.plot(SFR, L_co_total_array,'r.')
	ax2.set_xlim([1, 1e2])
	plt.ylim([1e6,1e11])
	ax2.set_yscale('log', base=10)
	ax2.set_xscale('log', base=10)
	# plt.xlabel('SFR [M$_{⊙}$/year]')
	# plt.ylabel('Lco [K km/s pc^2]')
	plt.grid(True)


	# TODO: Delete after correcting the unit in snapshot_to_Lco.py
	mass_h2 = mass_h2 * 1e10

	ax3 = fig.add_subplot(3,2,3)
	ax3.plot(SFR_XCOLDGASS, MH2_XCOLDGASS, 'o', mfc='none')
	ax3.plot(SFR_PHIBBS2, M_H2_PHIBBS2, 'x', mfc='none')
	ax3.plot(SFR_ALMA_2019, M_H2_ALMA_2019, 's', mfc='none')	
	ax3.plot(SFR, mass_h2, 'r.')
	plt.ylim([1e7,1e12])
	ax3.set_yscale('log', base=10)
	ax3.set_xscale('log', base=10)
	# plt.xlabel('SFR [M$_{⊙}$/year]')
	plt.ylabel('M$_{H2}$ [M$_{⊙}$]')
	plt.legend(["XCOLDGASS Survey (0.01 < z < 0.05)", 
				"PHIBBS2 Survey (0.5 < z < 0.8)", 
				"ALMA 2019 Survey (1.0 < z < 3.6)", 
				"FIRE Simulation (0.0 < z <3.8)"], 
				loc='upper left')	
	plt.grid(True)

	ax4 = fig.add_subplot(3,2,4)
	ax4.plot(SFR_XCOLDGASS, MH2_XCOLDGASS, 'o', mfc='none')
	ax4.plot(SFR_PHIBBS2, M_H2_PHIBBS2, 'x', mfc='none')
	ax4.plot(SFR_ALMA_2019, M_H2_ALMA_2019, 's', mfc='none')		
	ax4.plot(SFR, mass_h2, 'r.')
	plt.ylim([1e7,1e12])
	ax4.set_xlim([1, 1e2])
	ax4.set_yscale('log', base=10)
	ax4.set_xscale('log', base=10)
	# plt.xlabel('SFR [M$_{⊙}$/year]')
	# plt.ylabel('M$_{H2}$ [M$_{⊙}$]')
	plt.grid(True)


	# Expressing X_CO values in a unit of 1e20 cm^-2 (K-km s^-1)^-1: 
	X_CO_average = X_CO_average / 1e20
	XCO_XCOLDGASS = XCO_XCOLDGASS / 1e20
	X_CO_PHIBBS2 = X_CO_PHIBBS2 / 1e20
	X_CO_ALMA_2019 = X_CO_ALMA_2019 / 1e20 

	ax5 = fig.add_subplot(3,2,5)
	ax5.plot(SFR_XCOLDGASS, XCO_XCOLDGASS, 'o', mfc='none')
	ax5.plot(SFR_PHIBBS2, X_CO_PHIBBS2, 'x', mfc='none')
	ax5.plot(SFR_ALMA_2019, X_CO_ALMA_2019, 's', mfc='none')	
	ax5.plot(SFR, X_CO_average, 'r.')
	ax5.set_yscale('log', base=10)
	ax5.set_xscale('log', base=10)
	plt.xlabel('SFR [M$_{⊙}$/year]')
	plt.ylabel('X$_{CO}$ [10$^{20}$ cm$^{-2}$ K$^{-1}$ km$^{-1}$ s]' )
	plt.grid(True)

	ax6 = fig.add_subplot(3,2,6)
	ax6.plot(SFR_XCOLDGASS, XCO_XCOLDGASS, 'o', mfc='none')
	ax6.plot(SFR_PHIBBS2, X_CO_PHIBBS2, 'x', mfc='none')
	ax6.plot(SFR_ALMA_2019, X_CO_ALMA_2019, 's', mfc='none')
	ax6.plot(SFR, X_CO_average, 'r.')
	plt.ylim([1e-1,1e2])
	ax6.set_xlim([1, 1e2])
	ax6.set_yscale('log', base=10)
	ax6.set_xscale('log', base=10)
	plt.xlabel('SFR [M$_{⊙}$/year]')
	# plt.ylabel('X$_{CO}$ [10$^{20}$ cm$^{-2}$ K$^{-1}$ km$^{-1}$ s]' )
	plt.grid(True)


	if (do_you_want_to_save_fig==1):
		plt.savefig(path_and_name_of_the_figure)
		print("Figure is saved as: ", path_and_name_of_the_figure)

	plt.show()

	return 0

####################################################################################################################################################################

def plot_comparison_FIRE_and_observations_according_to_SFR_with_different_redshift_regimes(L_co_total_array,
														   						 		   mass_h2,
														   						 		   X_CO_average,
														   						 		   SFR,

														   						 		   LCO_XCOLDGASS,
														   						 		   MH2_XCOLDGASS,
														   						 		   XCO_XCOLDGASS,
														   						 		   SFR_XCOLDGASS,

														   						 		   LCO_10_PHIBBS2,
														   						 		   M_H2_PHIBBS2, #M_H2 is the molecular gas mass
														   						 		   X_CO_PHIBBS2,
														   						 		   SFR_PHIBBS2,

														   						 		   L_CO_ALMA_2019,
														   						 		   M_H2_ALMA_2019,
														   						 		   X_CO_ALMA_2019,
														   						 		   SFR_ALMA_2019,

														   						 		   redshift_array,
														   						 		   clumping_factor,
														   						 		   cut_off_redshift,
														   						 		   title_string,														   						 		   
														   						 		   do_you_want_to_save_fig=0):


	import matplotlib.pyplot as plt
	import numpy as np


	##############
	# TO save figure: 
	import os
	#To name the saved file
	current_directory   			= os.getcwd()
	directory_to_save_figures 		= current_directory + "/figures_and_data"
	save_name_of_the_figure 		= "Lco_FIRE_and_Observations_vs_SFR_" + "z"  + '{0:1.1f}'.format(redshift_array[0]) + "_to_" + "z" + '{0:1.1f}'.format(redshift_array[len(redshift_array)-1]) + "_cf" + str(clumping_factor) + ".png"
	path_and_name_of_the_figure 	= directory_to_save_figures + "/" + save_name_of_the_figure
	##############

	redshift_array_1 = redshift_array - cut_off_redshift
	redshift_smaller_than_cut_off_redshift_indices = np.where(redshift_array_1<0)
	redshift_smaller_than_cut_off_redshift_indices = redshift_smaller_than_cut_off_redshift_indices[0] 

	fig = plt.figure(figsize=(15,12))
	fig.suptitle(title_string, fontsize=20)
	ax1 = fig.add_subplot(3,2,1)
	ax1.plot(SFR_XCOLDGASS, LCO_XCOLDGASS,'o', mfc='none')
	ax1.plot(SFR_PHIBBS2, LCO_10_PHIBBS2, 'x', mfc='none')
	ax1.plot(SFR_ALMA_2019, L_CO_ALMA_2019, 's', mfc='none')
	ax1.plot(SFR[redshift_smaller_than_cut_off_redshift_indices], L_co_total_array[redshift_smaller_than_cut_off_redshift_indices],'r.')
	ax1.set_yscale('log', base=10)
	ax1.set_xscale('log', base=10)
	# plt.xlabel('SFR [M$_{⊙}$/year]')
	plt.ylim([1e2,1e12])
	plt.ylabel('Lco [K km/s pc^2]', fontsize=18)
	plt.grid(True)

	ax2 = fig.add_subplot(3,2,2)
	ax2.plot(SFR_XCOLDGASS, LCO_XCOLDGASS,'o', mfc='none')
	ax2.plot(SFR_PHIBBS2, LCO_10_PHIBBS2, 'x', mfc='none')
	ax2.plot(SFR_ALMA_2019, L_CO_ALMA_2019, 's', mfc='none')	
	ax2.plot(SFR, L_co_total_array,'r.')
	# ax2.set_xlim([1, 1e2])
	plt.ylim([1e2,1e12])
	ax2.set_yscale('log', base=10)
	ax2.set_xscale('log', base=10)
	# plt.xlabel('SFR [M$_{⊙}$/year]')
	# plt.ylabel('Lco [K km/s pc^2]')
	plt.grid(True)


	# TODO: Delete after correcting the unit in snapshot_to_Lco.py
	mass_h2 = mass_h2 * 1e10

	ax3 = fig.add_subplot(3,2,3)
	ax3.plot(SFR_XCOLDGASS, MH2_XCOLDGASS, 'o', mfc='none')
	ax3.plot(SFR_PHIBBS2, M_H2_PHIBBS2, 'x', mfc='none')
	ax3.plot(SFR_ALMA_2019, M_H2_ALMA_2019, 's', mfc='none')	
	ax3.plot(SFR[redshift_smaller_than_cut_off_redshift_indices], mass_h2[redshift_smaller_than_cut_off_redshift_indices], 'r.')
	plt.ylim([1e6,1e12])
	ax3.set_yscale('log', base=10)
	ax3.set_xscale('log', base=10)
	# plt.xlabel('SFR [M$_{⊙}$/year]')
	plt.ylabel('M$_{H2}$ [M$_{⊙}$]', fontsize=18)
	legend_string_small_redshift = "FIRE Simulation (0.0 < z < " + str(cut_off_redshift) + ")"
	plt.legend(["XCOLDGASS Survey (0.01 < z < 0.05)", 
				"PHIBBS2 Survey (0.5 < z < 0.8)", 
				"ALMA 2019 Survey (1.0 < z < 3.6)", 
				legend_string_small_redshift], 
				loc='upper left')	
	plt.grid(True)

	ax4 = fig.add_subplot(3,2,4)
	ax4.plot(SFR_XCOLDGASS, MH2_XCOLDGASS, 'o', mfc='none')
	ax4.plot(SFR_PHIBBS2, M_H2_PHIBBS2, 'x', mfc='none')
	ax4.plot(SFR_ALMA_2019, M_H2_ALMA_2019, 's', mfc='none')		
	ax4.plot(SFR, mass_h2, 'r.')
	plt.ylim([1e6,1e12])
	# ax4.set_xlim([1, 1e2])
	ax4.set_yscale('log', base=10)
	ax4.set_xscale('log', base=10)
	# plt.xlabel('SFR [M$_{⊙}$/year]')
	# plt.ylabel('M$_{H2}$ [M$_{⊙}$]')
	plt.legend(["XCOLDGASS Survey (0.01 < z < 0.05)", 
				"PHIBBS2 Survey (0.5 < z < 0.8)", 
				"ALMA 2019 Survey (1.0 < z < 3.6)", 
				"FIRE Simulation (0.0 < z <3.8)"], 
				loc='upper left')	
	plt.grid(True)


	# Expressing X_CO values in a unit of 1e20 cm^-2 (K-km s^-1)^-1: 
	X_CO_average = X_CO_average / 1e20
	XCO_XCOLDGASS = XCO_XCOLDGASS / 1e20
	X_CO_PHIBBS2 = X_CO_PHIBBS2 / 1e20
	X_CO_ALMA_2019 = X_CO_ALMA_2019 / 1e20 

	ax5 = fig.add_subplot(3,2,5)
	ax5.plot(SFR_XCOLDGASS, XCO_XCOLDGASS, 'o', mfc='none')
	ax5.plot(SFR_PHIBBS2, X_CO_PHIBBS2, 'x', mfc='none')
	ax5.plot(SFR_ALMA_2019, X_CO_ALMA_2019, 's', mfc='none')	
	ax5.plot(SFR[redshift_smaller_than_cut_off_redshift_indices], X_CO_average[redshift_smaller_than_cut_off_redshift_indices], 'r.')
	plt.ylim([1e-1,1e2])
	ax5.set_yscale('log', base=10)
	ax5.set_xscale('log', base=10)
	plt.xlabel('SFR [M$_{⊙}$/year]', fontsize=18)
	plt.ylabel('X$_{CO}$ \n[10$^{20}$ cm$^{-2}$ K$^{-1}$ km$^{-1}$ s]', fontsize=18)
	plt.grid(True)

	ax6 = fig.add_subplot(3,2,6)
	ax6.plot(SFR_XCOLDGASS, XCO_XCOLDGASS, 'o', mfc='none')
	ax6.plot(SFR_PHIBBS2, X_CO_PHIBBS2, 'x', mfc='none')
	ax6.plot(SFR_ALMA_2019, X_CO_ALMA_2019, 's', mfc='none')
	ax6.plot(SFR, X_CO_average, 'r.')
	plt.ylim([1e-1,1e2])
	# ax6.set_xlim([1, 1e2])
	ax6.set_yscale('log', base=10)
	ax6.set_xscale('log', base=10)
	plt.xlabel('SFR [M$_{⊙}$/year]', fontsize=18)
	# plt.ylabel('X$_{CO}$ [10$^{20}$ cm$^{-2}$ K$^{-1}$ km$^{-1}$ s]' )
	plt.grid(True)


	if (do_you_want_to_save_fig==1):
		plt.savefig(path_and_name_of_the_figure)
		print("Figure is saved as: ", path_and_name_of_the_figure)

	plt.show()

	return 0


####################################################################################################################################################################

