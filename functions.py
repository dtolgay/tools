import glob
import numpy as np
from .constants import *
from . import constants
import pandas as pd
import sys

def return_snapshot_path_and_number(galaxy_type, galaxy_name, redshift, operating_cluster):

	# redshift = "2.0"  # Never compare float numbers...
	# Determine the snapshot number 
	if galaxy_type in ["zoom_in", "zoom_in_tolgay", "particle_split"]:
		# Determine snapshot_number
		if redshift == "0.0":
			snapshot_number = 600     # z = 0.0
		elif redshift == "0.998":
			snapshot_number = 590	  # z = 0.998
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
		# Determine snap_dir_file_path
		if galaxy_type == "zoom_in":
			if redshift == "0.998":
				if operating_cluster == "cita":
					snap_dir_file_path = f'/fs/lustre/project/murray/FIRE/FIRE_2/{galaxy_name}/output'
			else: 
				if operating_cluster == "cita":
					snap_dir_file_path = f'/fs/lustre/project/murray/scratch/tolgay/metal_diffusion/{galaxy_name}/output'
				elif operating_cluster == "niagara":
					sys.exit("Exiting... Niagara is not supported for zoom_in galaxies. Please use cita cluster.")
				else:
					sys.exit("Exiting... Please set the operating cluster to cita, niagara or trillium.")
		elif galaxy_type == "zoom_in_tolgay":
			if operating_cluster == "cita":
				snap_dir_file_path = f'/fs/lustre/project/murray/scratch/tolgay/metal_diffusion/{galaxy_name}/output'
			elif operating_cluster == "niagara":
				sys.exit("Exiting... Niagara is not supported for zoom_in_tolgay galaxies. Please use cita cluster.")
			else:
				sys.exit("Exiting... Please set the operating cluster to cita, niagara or trillium.")
		elif galaxy_type == "particle_split":
			if operating_cluster == "cita":
				snap_dir_file_path = f"/fs/lustre/project/murray/FIRE/FIRE_2/{galaxy_name}/output"
			elif operating_cluster == "niagara":
				sys.exit("Exiting... Niagara is not supported for particle_split galaxies. Please use cita cluster.")
			else:
				sys.exit("Exiting... Please set the operating cluster to cita, niagara or trillium.")
		else:
			print(f"Exiting... Galaxy type is wrong. The given galaxy type is {galaxy_type}")

	# TODO: Fix the path to file directories
	elif galaxy_type in ["firebox"]:
		if redshift == "0.0":
			if operating_cluster == "cita":
				snap_dir_file_path = '/fs/lustre/project/murray/scratch/lliang/FIRE_CO/FIREbox'
			elif operating_cluster == "niagara":
				snap_dir_file_path = f'/scratch/m/murray/dtolgay/post_processing_fire_outputs/firebox_halo_finder/z{redshift}'
			elif operating_cluster == "trillium":
				snap_dir_file_path = f'/scratch/dtolgay/post_processing_fire_outputs/firebox_halo_finder/z{redshift}'
			else:
				sys.exit("Exiting... Please set the operating cluster to cita, niagara or trillium.")
			snapshot_number = 1200     # z = 0.0
		elif redshift == "0.5":
			print(f"Exiting... Currently there are no z=0.5 galaxies... {redshift}")
			sys.exit(2)                
		elif redshift == "1.0":
			if operating_cluster == "cita":
				snap_dir_file_path = '/fs/lustre/project/murray/scratch/tolgay/firebox/FB15N1024/seperated_galaxies/z1.0'
			elif operating_cluster == "niagara":
				snap_dir_file_path = f'/scratch/m/murray/dtolgay/post_processing_fire_outputs/firebox_halo_finder/z{redshift}'
			elif operating_cluster == "trillium":
				snap_dir_file_path = f'/scratch/dtolgay/post_processing_fire_outputs/firebox_halo_finder/z{redshift}'
			else:
				sys.exit("Exiting... Please set the operating cluster to cita, niagara or trillium.")
			snapshot_number = 554     # z = 1.0
		elif redshift == "2.0":
			if operating_cluster == "cita":
				snap_dir_file_path = '/fs/lustre/project/murray/scratch/tolgay/firebox/FB15N1024/seperated_galaxies/z2.0'
			elif operating_cluster == "niagara":
				snap_dir_file_path = f'/scratch/m/murray/dtolgay/post_processing_fire_outputs/firebox_halo_finder/z{redshift}'
			elif operating_cluster == "trillium":
				snap_dir_file_path = f'/scratch/dtolgay/post_processing_fire_outputs/firebox_halo_finder/z{redshift}'
			else:
				sys.exit("Exiting... Please set the operating cluster to cita, niagara or trillium.")
			snapshot_number = 344     # z = 2.0
		elif redshift == "3.0":
			if operating_cluster == "cita":
				# snap_dir_file_path = '/fs/lustre/project/murray/scratch/lliang/FIRE_CO/FIREbox' # TODO. Where is the path?
				sys.exit("Exiting... There is no z=3.0 galaxy in the firebox directory.")
			elif operating_cluster == "niagara":
				snap_dir_file_path = f'/scratch/m/murray/dtolgay/post_processing_fire_outputs/firebox_halo_finder/z{redshift}'
			elif operating_cluster == "trillium":
				snap_dir_file_path = f'/scratch/dtolgay/post_processing_fire_outputs/firebox_halo_finder/z{redshift}'
			else:
				sys.exit("Exiting... Please set the operating cluster to cita, niagara or trillium.")
			snapshot_number = 240     # z = 3.0
		else:
			print(f"Exiting... Redshift is wrong. The given galaxy type is {redshift}")
			sys.exit(2)        

	else:
		print(f"Exiting... Galaxy type is wrong. The given galaxy type is {galaxy_type}")
		sys.exit(1)


	return snap_dir_file_path, snapshot_number


def finding_mass_and_coordinates_of_the_halo_with_most_particle_ahf(snapshot_dir, snapshot_number):
	print("I am in the function finding_mass_and_coordinates_of_the_halo_with_most_particle_ahf") 

	"""This function is written to obtain the mass and the coordinate of the halo with most particles. It is assumed to
	be the most massive halo.
	
	Arguments:
	----------
	snapshot_dir : string
	
	snapshot_number : int

	Returns:
	----------
	mass: array_like
		mass of the most massive halo
		[M☉/h]

	x_halo_center: array_like
		x position of the most massive halo
		[kpc/h]

	y_halo_center: array_like
		y position of the most massive halo
		[kpc/h]

	z_halo_center: array_like
		z position of the most massive halo
		[kpc/h]

	"""


	# Finding the file 
	filein = glob.glob(snapshot_dir+"/snap"+str(snapshot_number)+"Rpep*_halos")
		# glob searches for specific file pattern that is creted by wildcards. For example astreiks(*) is a wildcard

	print("filein[0]: ", filein[0])

	# Reading data from the file
	data   = np.loadtxt(filein[0])

	# Reading the mass of the MMH
	mass = data[0,3]	# Halo finder code works such a way that initial row belongs to the center of mass of the most massive halo 

	# Reading Coordinates that belong to center of mass of most massive halo
	x_halo_center	= data[0,5]
	y_halo_center	= data[0,6]
	z_halo_center	= data[0,7]

	return mass, x_halo_center, y_halo_center, z_halo_center


def finding_velocity_of_the_halo_with_most_particle_ahf(snapshot_dir, snapshot_number):
# This function is written for controll purposes. It has nothing to do with the Lco calculation
# This function is written to obtain the velocity of the most massive halo
	print("I am in the function finding_velocity_of_the_halo_with_most_particle_ahf") 

	"""This function is written to obtain the velocity of the of the halo with most particles. It is assumed to be most massive halo. 
		
		This function is written to check the motion of the halo with most particles. It is not being used to calculate Lco. 

	Arguments:
	----------
	snapshot_dir : string
	
	snapshot_number : int

	Returns:
	----------
	vx_halo_center: array_like
		velocity along x direction of the most massive halo
		[km/s]

	vy_halo_center: array_like
		velocity along y direction of the most massive halo
		[km/s]

	vz_halo_center: array_like
		velocity along z direction of the most massive halo
		[km/s]

	"""



	# Finding the file 
	filein = glob.glob(snapshot_dir+"/snap"+str(snapshot_number)+"Rpep*_halos")
		# glob searches for specific file pattern that is creted by wildcards. For example astreiks(*) is a wildcard
		
	# Reading data from the file
	data   = np.loadtxt(filein[0])

	# Reading velocities that belong to center of mass of most massive halo
	vx_halo_center	= data[0,8]
	vy_halo_center	= data[0,9]
	vz_halo_center	= data[0,10]

	return vx_halo_center, vy_halo_center, vz_halo_center


def halo_with_most_particles_rockstar(rockstar_snapshot_dir, snapshot_number, time, hubble):

#This function is written to find the halo with most gas particles.

	print("I am in the function halo_with_most_particles_rockstar") 


	"""This function is written to obtain the positin and velocity of the halo with most particle
		
		The position outputs are used to calculate Lco, however the velocity arguments are not used 
		to calculate Lco. Velocity and ID arguments are output to check the motion of the most massive halo.

	Arguments:
	----------
	rockstar_snapshot_dir : string
	
	snapshot_number : int

	Returns:
	----------
	x_mph: float
		position along x direction of the halo with most particles

	y_mph: float
		position along y direction of the halo with most particles

	z_mph: float
		position along z direction of the halo with most particles

	vx_mph: float
		velocity along x direction of the halo with most particles

	vy_mph: float
		velocity along y direction of the halo with most particles

	vz_mph: float
		velocity along z direction of the halo with most particles

	ID: integer
		ID of the halo with most particles

	DescID: integer
		DescID along z direction of the halo with most particles

	"""


	# Finding the file name 
	filein = "out_" + str(snapshot_number) + ".list"
	file_path = rockstar_snapshot_dir + "/" + filein
#	print("file_path: ", file_path)

	# The Units of the Rockstar Halo Finder code: 
	# ----------	
	# Units: Masses in Msun / h
	# Units: Positions in Mpc / h (comoving)
	# Units: Velocities in km / s (physical, peculiar)
	# Units: Halo Distances, Lengths, and Radii in kpc / h (comoving)
	# Units: Angular Momenta in (Msun/h) * (Mpc/h) * km/s (physical)
	# Units: Spins are dimensionless

	data 					= np.loadtxt(file_path)
	# Finding the halo that contains most particles
	Np 						= data[:,7] 
	index_of_the_max_Np 	= np.argmax(Np)

	# Mass of the halo with most particles: 
	#"mph" stands for halo with most particles
	mvir_mph = data[index_of_the_max_Np,20]

	#Position of the halo with most particles:
	#"mph" stands for halo with most particles  
	x_mph =  data[index_of_the_max_Np,8] * 1e3 		# in kpc/h
	y_mph =  data[index_of_the_max_Np,9] * 1e3 		# in kpc/h
	z_mph =  data[index_of_the_max_Np,10] * 1e3 	# in kpc/h

	#Velocity of the halo with most particles: 
	vx_mph =  data[index_of_the_max_Np,11]
	vy_mph =  data[index_of_the_max_Np,12]
	vz_mph =  data[index_of_the_max_Np,13]

	#ID of the halo with most particles: 
	ID 		= data[index_of_the_max_Np,0]
	DescID 	= data[index_of_the_max_Np,1]  


	# The positions and mass of the Halo is in comoving units. To make everything on physical units the below code is written
	mvir_mph 	= mvir_mph * 1./hubble    # [M☉]
	x_mph 		= x_mph * time/hubble     # [kpc]
	y_mph 		= y_mph * time/hubble     # [kpc]
	z_mph 		= z_mph * time/hubble     # [kpc]  
	# Velocity is in peculiar components [km/s]

	return mvir_mph, x_mph, y_mph, z_mph, vx_mph, vy_mph, vz_mph, ID, DescID

##########################################################################################################################################################################################################


def change_origin(x, y, z, x_halo_center, y_halo_center, z_halo_center):

	print("I am in the function change_origin") 


	"""This function is written to transform the position of the origin. The origin of the coordinate system is transformed to the halo center

	Arguments:
	----------
	x : array_like - float
		x position that needs to be expressed in new coordinate system
		[kpc]
	
	y : array_like - float
		y position that needs to be expressed in new coordinate system	
		[kpc]

	z : array_like - float
		z position that needs to be expressed in new coordinate system
		[kpc]

	x_halo_center : array_like - float
		x component of the center position of the most massive halo
		[kpc]

	y_halo_center : array_like - float
		y component of the center position of the most massive halo
		[kpc]

	z_halo_center : array_like - float
		z component of the center position of the most massive halo
		[kpc]

	Returns:
	----------
	x_new: 	array_like - float
			Transformed x coordinate
			[kpc]

	y_new: 	array_like - float
			Transformed y coordinate
			[kpc]

	z_new: 	array_like - float
			Transformed z coordinate
			[kpc]

	"""

	x_new = x - x_halo_center 
	y_new = y - y_halo_center 
	z_new = z - z_halo_center 

	return x_new, y_new, z_new


def net_angular_momentum(mass, rx, ry, rz, vx, vy, vz):

	print("I am in the function net_angular_momentum") 
	
# This function is written in order to calculate the net angular momentum of the particles. All variables are vectors

	"""This function is written in order to calculate the net angular momentum of the particles

	Arguments:
	----------
	mass : array_like
		mass of the particles
	
	rx : array_like
		x position of the particles

	ry : array_like
		y position of the particles

	rz : array_like
		z position of the particles

	vx : array_like
		velocity along x direction of the particles

	vy : array_like
		velocity along y direction of the particles

	vz : array_like
		velocity along z direction of the particles

	Returns:
	----------
	L: 1x3 sized vector
		Net angular momentum corresponding to x, y and z directions
		[1e10 M☉ kpc km / sec]

	"""	

	Lx = mass * ((ry * vz) - (rz * vy))
	Ly = mass * ((rz * vx) - (rx * vz))
	Lz = mass * ((rx * vy) - (ry * vx))

	L = [np.sum(Lx), np.sum(Ly), np.sum(Lz)]

	return (L)

def finding_the_angles_between_current_coordinate_system_and_net_angular_momentum(L):

	print("I am in the function finding_the_angles_between_current_coordinate_system_and_net_angular_momentum")

	import math

	"""This function is used in order the find the angles that net angular momentum vector does with the coordinate axes

	Arguments:
	----------
	L : 1x3 sized vector
		Net angular momentum corresponding to x, y and z directions of gas particles
	
	Returns:
	----------
	theta: double
		angle around z axis 
		[radian]

	phi: double
		angle in the x-y plane
		[radian]

	"""	

	magnitude_of_the_L = np.sqrt(np.power(L[0],2) + np.power(L[1],2) + np.power(L[2],2))

	# # I used Gunjan's method to calculate angles
	theta  	= math.acos(L[2] / magnitude_of_the_L)  						# theta = Lz / L 		 - Angle in radians
	phi		= math.atan2(L[1] , L[0]) 										# phi = Ly / Lx 		 - Angle in radians

	return theta, phi


def rotating_coordinate_system_along_net_angular_momentum(theta, 
														  phi, 
														  vectorx, 
														  vectory, 
														  vectorz):

	print("I am in the function rotating_coordinate_system_along_net_angular_momentum") 

	"""This function is used in order to allign the z axis of the net angular momentum with the positive z axis of the coordinate system, such that
	we can observe the halos in more proper way. x-y axis corresponds to the mid-plane of the halo.

	Arguments:
	----------
	theta: double
		Angle with z axis 
		[radian]

	phi: double
		Angle in the x-y plane 
		[radian]

	vectorx: array_like - float
		Unrotated x positions of the gas particles

	vectory: array_like - float
		Unrotated y positions of the gas particles

	vectorz: array_like - float
		Unrotated z positions of the gas particles

	Returns:
	----------
	vectorx_new: array_like - float
		Rotated x positions of the gas particles

	vectory_new: array_like - float
		Rotated y positions of the gas particles

	vectorz_new: array_like - float
		Rotated z positions of the gas particles

	"""	

	import numpy as np 


	# Initially rotating around z axis. This will project angular momentum vector to the x-z plane. 
	# How the code below works is as follows: It basically takes the length of the vector in the x-y plane and sets x_new = length_xy, although it looks more complicated! 
	# length_xy = sqrt(x^2 + y^2)
	# Then x_new = length_xy; 
	#      y_new = 0; 
	#	   z_new = z_old

	vectorx_new = np.cos(phi) * vectorx + \
	              np.sin(phi) * vectory + \
	              0 * vectorz

	vectory_new = -np.sin(phi) * vectorx + \
				  np.cos(phi) * vectory + \
				  0 * vectorz

	vectorz_new = 0 * vectorx + \
				  0 * vectory + \
				  1 * vectorz

	# This is being done because otherwise I am having problems in the rotation around y axis below.
	vectorx = vectorx_new
	vectory = vectory_new
	vectorz = vectorz_new

	# print("\n\n")
	# print("After projecting in the x-z plane")
	# print("vectorx: ", vectorx)
	# print("vectory: ", vectory)
	# print("vectorz: ", vectorz)
	# print("\n\n")

	# Now rotating around the y axis. This will put new coordinate frame of the angular momentum onto the z axis of the angular momentum.
	vectorx_new = np.cos(theta) * vectorx + \
				  0 * vectory + \
				  -np.sin(theta) * vectorz  

	vectory_new = 0 * vectorx + \
				  1 * vectory + \
				  0 * vectorz


	vectorz_new = np.sin(theta) * vectorx + \
					0 * vectory + \
					np.cos(theta) * vectorz

	# print("\n\n")
	# print("After projecting onto the z axis")
	# print("vectorx_new: ", vectorx_new)
	# print("vectory_new: ", vectory_new)
	# print("vectorz_new: ", vectorz_new)
	# print("\n\n")

	# Now the z direction of angular momentum coincides with the z axis of the coordinate frame.

	return vectorx_new, vectory_new, vectorz_new


##########################################################################################################################################################################################################

def local_density_scale_height_calculator(density_gas, x_gas, y_gas, z_gas, smoothing_length_gas):

	"""This function is used to calculate the local density scale height (h=density/grad(density)):
	A Comparison of Methods for Determining the Molecular Content of Model Galaxies by Krumholz, and Gnedin (2011)

	Arguments:
	----------
	density_gas: array_like
		Density of the gas particles
		[1e10 M☉/kpc^3]

	x_gas: array_like
		x positon of the gas particles
		[kpc]

	y_gas: array_like
		y positon of the gas particles
		[kpc]

	z_gas: array_like
		z positon of the gas particles
		[kpc]	

	smoothing_length_gas: array_like 
		smoothing length of the gas particles
		[kpc]


	Returns:
	----------
	local_density_scale_height: array_like
		local scale density height
		[kpc]


	References: 
	-----------
	A Comparison of Methods for Determining the Molecular Content of Model Galaxies by Krumholz, and Gnedin (2011)

	"""	

	print("I am in the function local_density_scale_height_calculator") 

	import numpy as np
	from scipy.spatial import cKDTree

	# create a k-d tree for the gas particle positions. It creates a 3xlen(x_gas) matrix.
	tree = cKDTree(np.column_stack((x_gas, y_gas, z_gas)))

	# query the k-d tree for the nearest neighbors of each gas particle within a radius of 10 kpc
	distances, indices = tree.query(tree.data, k=2, distance_upper_bound=10)

	# tree.query functions returns two closest nearest neighbourhood because k=2. The closest particle to our particle is the particle
	# itself. indices[i,0] is the i'th particle. The closest nearest neighboor therefore is the indices[i,1].   
	index_of_the_closest_nnp = indices[:,1]				# indices of the closest particles
	distance_of_the_closest_nnp = distances[:,1] 		# distances between the closest particle and the ith particle

	# Calculating the density gradient:
	grad_density_gas = np.abs(density_gas[:] - density_gas[index_of_the_closest_nnp]) / distance_of_the_closest_nnp 	# [1e10 M☉/kpc^4] 

	local_density_scale_height = density_gas / grad_density_gas 	# [kpc]


	# # TODO: Delete
	# difference_between_h_and_scale_length = (smoothing_length_gas - local_density_scale_height) 
	# percantage_difference_between_h_and_scale_length = difference_between_h_and_scale_length / smoothing_length_gas

	# # Averaging for printing purposes:
	# average_difference_between_h_and_scale_length = np.sum(difference_between_h_and_scale_length)/len(difference_between_h_and_scale_length)
	# average_percantage_difference_between_h_and_scale_length = np.sum(percantage_difference_between_h_and_scale_length)/len(percantage_difference_between_h_and_scale_length)


	# print("\ndifference_between_h_and_scale_length: ", difference_between_h_and_scale_length)
	# print("\naverage_difference_between_h_and_scale_length: ", average_difference_between_h_and_scale_length)


	# print("\n\n\npercantage_difference_between_h_and_scale_length: ", percantage_difference_between_h_and_scale_length)
	# print("\naverage_percantage_difference_between_h_and_scale_length: ", average_percantage_difference_between_h_and_scale_length)

	# for i in range(0, 1000):
	# 	print('smoothing_length_gas[i]: ', smoothing_length_gas[i], '		local_density_scale_height[i]: ', local_density_scale_height[i],)

	return local_density_scale_height


###############


def h2_mass_fraction_calculator(local_density_scale_height, 
								density, 
								metallicity, 
								clumping_factor):

	print("I am in the function h2_mass_fraction_calculator") 


	"""This function is used to calculate the H2 mass fraction by using the formula 1 in the paper:
	A Comparison of Methods for Determining the Molecular Content of Model Galaxies by Krumholz, and Gnedin (2011)

	Arguments:
	----------
	local_density_scale_height: array_like
		In this equation smooting length of the gas is assumed to be accurate estimation of the local density scale height. 
		Therefore smooting length is used instead of local density scale height
		[kpc]

	density: array_like
		Density of the gas particles
		[1e10 M☉ / kpc^3]

	metallicity: array_like
		metallicity of the gas particles 
		It is the direct output of the fire simulation. It is not normalized to solar metallicity

	clumping_factor: double or int
		It is a parameter to boost the h2 mass fraction and therefore h2 column density and CO luminosity
		[unitless]

	Returns:
	----------
	h2_mass_fraction: array_like
		h2_mass_fraction = h2_gas_mass / total_gas_mass
		[unitless]

	column_density: array_like
		It is the column density considering all elements in the gas particle
		[gr/cm^2]

	dust_optical_depth: array_like	
		tau_c in the reference paper. It was being output in order to control the code
		[unitless]

	References: 
	-----------
	A Comparison of Methods for Determining the Molecular Content of Model Galaxies by Krumholz, and Gnedin (2011)

	"""	

	# The unit of density in snapshots is [10^10 M_sun / kpc^3]
	density = density * constants.ten_to_ten_times_Msun_to_Msun 			    		# [M_sun / kpc^3]
	density = density * constants.M_sun2gr / constants.kpc2cm**3  					 	# [gr / cm^3]

	# The units of local_density_scale_height is kpc 
	local_density_scale_height = local_density_scale_height * constants.kpc2cm 	# [cm]


	# Calculation of column density 
	column_density = density * local_density_scale_height 	# [gr / cm^2]
	# Column density is the summation sign in the paper
	# Gunjan assumed that local density scale height is same with the smoooting length of the gas particles. I will continue on this assumption
	# but I don't know how it works

	# Calculation of dust cross section (sigma_d in the paper)
	normalized_metallicity = metallicity / constants.solar_metallicity
	dust_cross_section_per_H_nucleus_normalized_to_1eminus21 = normalized_metallicity
	dust_cross_section = dust_cross_section_per_H_nucleus_normalized_to_1eminus21 * 1e-21	# [cm^2]


	# Calculation of dust optical depth (tau_c in the paper)
	# mu_h is the mean mass per H nucleus
	mu_h = 2.3e-24	# [gr] 
	# clumping factor is used to increase the H2 formation to account for density inhomogeneities that are unresolved on the computational grid
	# since the H2 formation rate varies as the square of density, these inhomogeneities increase the overall rate
	dust_optical_depth = column_density * dust_cross_section / mu_h 	# [dimensionless]	
	# In the original equation there is no clumping factor! 
	dust_optical_depth[dust_optical_depth==0] = 1e-30
	# The above operation was done because at regions since H_mass_fraction is zero, column density is also zero. Therefore dust_optical_depth 
	# also yields zero that creates a warning for the scaled_radiation_field calculation.  


	# Calculation for scaled radiation field (chi in the paper) Eq 4 
	# This scaled radiation field will not likely to hold cell-by-cell every time step, but it should hold on average
	# clumping factor is used to boost the formation rate of the H2 molecules on dust grains (the R term)	
	scaled_radiation_field = 3.1 * (1 + 3.1 * normalized_metallicity**0.365) / (4.1 * clumping_factor)  # [dimensionless]

	# Calculation for s in the paper (Eq 2)
	s = np.log(1 + 0.6*scaled_radiation_field + 0.01 * np.power(scaled_radiation_field,2)) / ( 0.6 * dust_optical_depth )

	# Calculation for the H2 mass fraction (f_H2 in the paper Eq 1)
	h2_mass_fraction = 1 - (3/4) * (s / (1 + 0.25*s))	# [dimensionless]
	h2_mass_fraction[h2_mass_fraction < 0] = 0 		# If the result is negative set it to zero


	return h2_mass_fraction, column_density, dust_optical_depth, scaled_radiation_field, s, dust_optical_depth


##########################################################################################################################################################################################################

def h2_column_density_calculator(h2_mass_fraction, gas_column_density, digitize_gas_bins):
	print("I am in the function h2_column_density")		

	"""This function is created in order to calculate the molecular gas column density 
	A Comparison of Methods for Determining the Molecular Content of Model Galaxies by Krumholz, and Gnedin (2011)

	Arguments:
	----------
	h2_mass_fraction: array_like
		h2_mass_fraction = h2_gas_mass / total_gas_mass
		[unitless]

	gas_column_density: array_like
		gas_column_density is the same thing with the column density output of the h2_mass_fraction_calculator function
		All the elements in the gas contributes the gas column density
		[gr/cm^2]

	digitize_gas_bins: array_like
		It is a identifier of which gas particle belongs to which annulus. In the code the gas particles are categorized 
		according to their distance from the center of the most massive halo, digitize_gas_bins shows the indices of these 
		categorization to take the average of the gas column density in each annulus. However since h2_column_density_annulus
		is not used in the snapshot_to_Lco.py code, it is meaningless to use it. TODO


	Returns:
	----------
	h2_column_density: array_like
		h2_column_density is the h2 column density in each gas particle positions. It is being used in the code in order to find the 
		mass average of the h2_column_density in each annulus
		[gr/cm^2]

	h2_column_density_annulus: array_like
		h2_column_density averaged to each annulus. It is used only to compare with the code written by Gunjan Lakhlani.
		It is not being used in the snapshot_to_Lco.py code. TODO
		[gr/cm^2]

	"""	

	h2_column_density = h2_mass_fraction * gas_column_density  # [gr/cm^2]	


	###### Calculating h2_column_density_annulus
	h2_column_density_annulus = np.asarray([np.nansum(h2_column_density[digitize_gas_bins==i]) / len(h2_column_density[digitize_gas_bins==i]) \
		for i in range(1, max(digitize_gas_bins)+1)])		# [gr / cm^2]


	return h2_column_density, h2_column_density_annulus




def h2_column_density_by_finding_mass_and_dividing_to_area(h2_mass_fraction, gas_mass, digitize_gas_bins, annulus_area):

	# TODO: Comment h2_column_density_by_finding_mass_and_dividing_to_area function

	print("I am in the function h2_column_density_by_finding_mass_and_dividing_to_area")

	# In here gas_mass and H_mass_fraction must be for R < Rmax

	mass_h2 = gas_mass * h2_mass_fraction   		

	h2_column_density_annulus = np.asarray(np.asarray([mass_h2[digitize_gas_bins == i].sum()/annulus_area[i-1] \
		for i in range(1, len(annulus_area)+1)]))	# [1e10 * M_star / kpc^2]

	# Below is the unit conversion for Gunjan's h2 column density
	# The units of the fire simulation are in kpc for distance and 10^10 M_star for mass. I will use cgs units therefore some conversion will be 
	# done in below lines
	h2_column_density_annulus = h2_column_density_annulus * constants.ten_to_ten_times_Msun_to_Msun * constants.M_sun2gr / (constants.kpc2cm)**2  # [gr/cm^2]

	return h2_column_density_annulus

###########################################################################################################################################################
def take_mass_average(vector_that_needs_to_be_mass_averaged, h2_mass):
	print("I am in the function take_mass_average")	


	"""This function is created to calculate the mean surface density in all of the different regions (annulus). 
	The output can be considered as the surface density at which most of the mass resides.


	Arguments:
	----------
	vector_that_needs_to_be_mass_averaged: array_like
		Name is self explanatory. It is the inital array that is defined for every point that gas particles occupy. 
		This vector will be mass averaged
		vector_that_needs_to_be_mass_averaged = (without <>)

	h2_mass: array_like
		h2 mass at the every gas particle location
		h2_mass = M_H2

	digitize_gas_bins: array_like
		It is a identifier of which gas particle belongs to which annulus. In the code the gas particles are categorized 
		according to their distance from the center of the most massive halo, digitize_gas_bins shows the indices of these 
		categorization to take the average of the gas column density in each annulus. It is being used to categorize the 
		regions and mass average the vector only in these regions one by one

	h2_mass_annulus: vector
		It is the total mass of h2 in each annulus


	Returns:
	----------
	mass_averaged_vector_annulus: vector 
		It is the mass averaged vector for each annulus


	Notes: 
	-----------	
	Bins and R_gas_smaller_than_Rmax is required to seperate different regions between each other. The mean surface density will be calculated
	for each of the different bins.	

	References: 
	-----------
	A general model for the CO-H2 conversion factor in galaxies with applications to the star formation law. (Narayanan et al. 2012)

	"""	

	# Following Eq 5. 
	# mass_averaged_vector_annulus = np.asarray([np.sum(vector_that_needs_to_be_mass_averaged[digitize_gas_bins == i]*h2_mass[digitize_gas_bins == i])/h2_mass_annulus[i-1] \
	# 	for i in range(1, max(digitize_gas_bins)+1)])


	dummy = vector_that_needs_to_be_mass_averaged * h2_mass 
	mass_averaged_vector = np.nansum(dummy) / np.nansum(h2_mass)


	return mass_averaged_vector


def take_mass_average_annulus(vector_that_needs_to_be_mass_averaged, h2_mass, digitize_gas_bins, h2_mass_annulus):
	print("I am in the function take_mass_average_annulus")	


	"""This function is created to calculate the mean surface density in all of the different regions (annulus). 
	The output can be considered as the surface density at which most of the mass resides.


	Arguments:
	----------
	vector_that_needs_to_be_mass_averaged: array_like
		Name is self explanatory. It is the inital array that is defined for every point that gas particles occupy. 
		This vector will be mass averaged
		vector_that_needs_to_be_mass_averaged = (without <>)

	h2_mass: array_like
		h2 mass at the every gas particle location
		h2_mass = M_H2

	digitize_gas_bins: array_like
		It is a identifier of which gas particle belongs to which annulus. In the code the gas particles are categorized 
		according to their distance from the center of the most massive halo, digitize_gas_bins shows the indices of these 
		categorization to take the average of the gas column density in each annulus. It is being used to categorize the 
		regions and mass average the vector only in these regions one by one

	h2_mass_annulus: vector
		It is the total mass of h2 in each annulus


	Returns:
	----------
	mass_averaged_vector_annulus: vector 
		It is the mass averaged vector for each annulus


	Notes: 
	-----------	
	Bins and R_gas_smaller_than_Rmax is required to seperate different regions between each other. The mean surface density will be calculated
	for each of the different bins.	

	References: 
	-----------
	A general model for the CO-H2 conversion factor in galaxies with applications to the star formation law. (Narayanan et al. 2012)

	"""	

	# Following Eq 5. 
	mass_averaged_vector_annulus = np.asarray([np.sum(vector_that_needs_to_be_mass_averaged[digitize_gas_bins == i]*h2_mass[digitize_gas_bins == i])/h2_mass_annulus[i-1] \
		for i in range(1, max(digitize_gas_bins)+1)])

	return mass_averaged_vector_annulus


###########################################################################################################################################################

def X_co_calculator(h2_column_density, metallicity, h2_column_density_annulus_mass_averaged_annulus, metallicity_gas_annulus_mass_averaged):
	print("I am in the function X_co_calculator")

	"""This function is being used in order to calculate the X_co for each annulus

	Arguments:
	----------
	h2_column_density: array-like
		h2_column_density for each particle 
		[gr/cm^2]

	metallicity: array-like
		Not normalized to solar metallicity
		[unitless]

	h2_column_density_annulus_mass_averaged_annulus: array-like
		h2_column_density for each particle 
		[gr/cm^2]

	metallicity_gas_annulus_mass_averaged: array-like
		Not normalized to solar metallicity
		[unitless]

	Returns:
	----------
	X_co: vector 
		CO conversion factor for each particle
		[cm^-2 /K-km s^-1]

	X_co_annulus: array-like
		CO conversion factor for each annulus
		[cm^-2 /K-km s^-1]		

	References: 
	-----------
	A general model for the CO-H2 conversion factor in galaxies with applications to the star formation law. (Narayanan et al. 2012)

	"""	

# Unit conversion needs to be done:

	normalized_metallicity = metallicity/constants.solar_metallicity
	normalized_metallicity[normalized_metallicity==0]=0

	h2_column_density = h2_column_density * constants.gr2M_sun / (constants.cm2pc)**2	
	h2_column_density[h2_column_density==0]=0
	# h2_column_density [M_sun/pc^2]

# A general model for the CO-H2 conversion factor in galaxies with applications to the star formation law. (Narayanan et al. 2012)
# Eq 6 

	X_co = 1.3e21/(normalized_metallicity * h2_column_density**0.5) #[cm^-2 /K-km s^-1]

	# X_co[X_co>1e30] = "NaN"

	# X_co_annulus = np.asarray([np.nansum(X_co[digitize_gas_bins==i]) / len(X_co[digitize_gas_bins==i]) \
	# 	for i in range(1, max(digitize_gas_bins)+1)])		# [gr / cm^2]

	# X_co_annulus calculation
	metallicity_gas_annulus_mass_averaged_normalized = metallicity_gas_annulus_mass_averaged/constants.solar_metallicity
	metallicity_gas_annulus_mass_averaged_normalized[metallicity_gas_annulus_mass_averaged_normalized==0]="NaN"

	h2_column_density_annulus_mass_averaged_annulus = h2_column_density_annulus_mass_averaged_annulus * constants.gr2M_sun / (constants.cm2pc)**2
	h2_column_density_annulus_mass_averaged_annulus[h2_column_density_annulus_mass_averaged_annulus==0]="NaN"

	X_co_annulus = 1.3e21/(metallicity_gas_annulus_mass_averaged_normalized * h2_column_density_annulus_mass_averaged_annulus**0.5)

	####	

	X_co_solar_metallicity = 1.3e21/(constants.solar_metallicity * h2_column_density**0.5) #[cm^-2 /K-km s^-1]

	return X_co, X_co_annulus, X_co_solar_metallicity


###########################################################################################################################################################

def L_co_calculator(X_co, X_co_solar_metallicity, h2_mass):

	print("I am in the function L_co_calculator")

	"""This function is used to calculate the alfa_co and L_co for each particle 

	Arguments:
	----------
	X_co: array_like
		X_co value for particle

	h2_mass: array_like
		h2 mass for each particle
		Unit: M_solar


	Returns:
	----------
	alfa_co: vector 
		CO conversion factor (alfa_co) for each particle 

	L_co_annulus: vector 
		CO luminosity in each annulus in units of [K-km s^-1 pc^2]

	References: 
	-----------
	A general model for the CO-H2 conversion factor in galaxies with applications to the star formation law. (Narayanan et al. 2012)

	"""	

# I followed the below methadology:
# A general model for the CO-H2 conversion factor in galaxies with applications to the star formation law. (Narayanan et al. 2012)
	
	# Since we have M_gas information and want to get L_co, alfa_co should be used. 
	# Following the Narayanan et. al, 2012. Eq 3 
	alfa_co = X_co / 6.3e19 # [M_solar/pc^-2 (K-km s^-1)^-1]

	alfa_co_solar_metallicity = X_co_solar_metallicity / 6.3e19 # [M_solar/pc^-2 (K-km s^-1)^-1]

	# Following the Narayanan et al. 2012, eq 2
	L_co = constants.ten_to_ten_times_Msun_to_Msun * h2_mass / alfa_co     # There is 1e10 because the unit of h2_mass_annulus is 1e10M_solar
	# L_co [K-km s^-1 pc^2]
	# L_co[L_co > 1e24] = float("nan")    # This is done because I of the fact that I set the values equal to zero to be equal to 1e30 for 
	# computational purposes. This assumption results with a large L_co in the outer region of the molecular gas

	L_co_solar_metallicity = constants.ten_to_ten_times_Msun_to_Msun * h2_mass / alfa_co_solar_metallicity     # There is 1e10 because the unit of h2_mass_annulus is 1e10M_solar


	L_co_total = np.nansum(L_co) # [K-km s^-1 pc^2]
	L_co_total_solar_metallicity = np.nansum(L_co_solar_metallicity) # [K-km s^-1 pc^2]
	#Converting L_co_total to L_sun units:
	# L_co_total_divided_by_Lsun = L_co_total * 4.9e-5    # TODO: I took it from Gunjan. I was not able to find this constant in any paper. 

	return alfa_co, alfa_co_solar_metallicity, L_co, L_co_total, L_co_total_solar_metallicity

###########################################################################################################################################################

def SFR_h_alfa_calculator(mass_star,
						  scale_factor_star,   
						  scale_factor_galaxy): 

	print("I am in the function SFR_h_alfa_calculator")

	"""This function is used to calculate the SFR by only looking at the stars that are formed in the last 5Myr's and within R_max distance from center of MMH
	This type of SFR calculation is also known as h_alfa emission calculation. Observers are looking h_alfa emission to guess SFR
	
	Arguments:
	----------
	mass_star = array like
		mass of the star particles within Rmax

	scale_factor_galaxy = double
		scale factor of the galaxy

	Returns:
	----------
	SFR_h_alfa: double 
		SFR by looking at the h_alfa emission. Or in other words the rate of formed stars within 5Myr's and within R_max distance from the center of MMH

	References: 
	-----------
	Norman Murray 

	"""	

	# First related properties of star particles are readed. Then I changed the origin and only looked star particles that are within R_max[kpc] distance from the center of MMH and formed within 5Myr's.
	# Then I summed all star particle masses that satisfy the above conditions and divided to 5Myr to find the star formation rate. 

	import numpy as np 
	import readsnap
	import os 


	#####
	def find_nearest(scale_factor_of_the_object_of_interest, scale_factor_in_snapshot_times_file_array, time_in_snapshot_times_file_array):
		
		"""This function is used to calculate the time of the galaxy and star particles
		
		Arguments:
		----------
		scale_factor_of_the_object_of_interest = array like or float
			scale factor of the stars or galaxy

		scale_factor_in_snapshot_times_file_array = array like - double
			scale factor in the snapshot_times.txt file

		time_in_snapshot_times_file_array = array like - double 
			time in the snapshot_times.txt file 

		Returns:
		----------
		time_in_snapshot_times_file_array[idxs]: array like or float 
			time of the star particles or galaxy

		"""

		# Use np.searchsorted() to find the indices of the closest scale factor in the file for each particle. 
		idxs = np.searchsorted(scale_factor_in_snapshot_times_file_array, scale_factor_of_the_object_of_interest)

		# Use np.clip() to ensure the indices are within the valid range of the file.
		idxs = np.clip(idxs, 1, len(scale_factor_in_snapshot_times_file_array)-1)				

		# Use left_scale_factors and right_scale_factors to determine which closest scale factor should be used for each particle.
		left_scale_factors = scale_factor_in_snapshot_times_file_array[idxs-1]
		right_scale_factors = scale_factor_in_snapshot_times_file_array[idxs]

		# The resulting boolean array compares these two arrays and checks whether the difference between scale_factor_star and left_scale_factors is less than
		# the difference between right_scale_factors and scale_factor_star. Then, the resulting boolean array is subtracted from idxs, 
		# which will either keep the same value or subtract 1 from the value depending on whether the boolean value is True or False.
		idxs -= scale_factor_of_the_object_of_interest - left_scale_factors < right_scale_factors - scale_factor_of_the_object_of_interest

		return time_in_snapshot_times_file_array[idxs]

	#####



	#I will use snapshot_times.txt file to convert the scale_factor to time
	current_directory = os.getcwd()
	snapshot_times_file_dir = current_directory
	file_name 			= snapshot_times_file_dir + "/snapshot_times.txt"
	snapshot_times_file = np.loadtxt(file_name)

	time_in_snapshot_times_file 		= snapshot_times_file[:,3]				# in Gyr
	scale_factor_in_snapshot_times_file = snapshot_times_file[:,1] 


	# For GALAXY
	simulation_time = find_nearest(scale_factor_of_the_object_of_interest =scale_factor_galaxy,
								   scale_factor_in_snapshot_times_file_array = scale_factor_in_snapshot_times_file,
								   time_in_snapshot_times_file_array = time_in_snapshot_times_file)

	# For Star Particles 
	formation_age_star = find_nearest(scale_factor_of_the_object_of_interest = scale_factor_star,
		    						  scale_factor_in_snapshot_times_file_array = scale_factor_in_snapshot_times_file,
		    						  time_in_snapshot_times_file_array = time_in_snapshot_times_file)



	# Only considering star particles within R_max distance from the most massive halo and that is born in last 5Myr
	R_star_smaller_than_Rmax_indices_and_formed_within_5myr_indices = [np.where((simulation_time-formation_age_star)<5e-3)]  # 5e-3 because time unit of the GIZMO is Gyr
	R_star_smaller_than_Rmax_indices_and_formed_within_5myr_indices = R_star_smaller_than_Rmax_indices_and_formed_within_5myr_indices[0] # indices are the zeroth element of the np.where() function


	# Calculation the young star formation rate:
	five_million_years = 5e6


	# Unit conversion is done by multiplying with 1e10. Don't forget that mass unit of GIZMO is 1e10*M_sun
	SFR_h_alfa = np.sum(mass_star[R_star_smaller_than_Rmax_indices_and_formed_within_5myr_indices]) * constants.ten_to_ten_times_Msun_to_Msun / five_million_years 	# [M_sun / five_million_years]
	# I am not sure if the unit is [M_sun / five_million_years], it can be [M_sun / year]

	return SFR_h_alfa


####################################################################################################################################################################

def gas_temperature_calculator(He_mass_fraction,
							   electron_abundace_gas,
							   internal_energy_gas):

	print("I am in the function gas_temperature_calculator")

	"""This function is used to calculate the SFR by only looking at the stars that are formed in the last 5Myr's and within R_max distance from center of MMH
	This type of SFR calculation is also known as h_alfa emission calculation. Observers are looking h_alfa emission to guess SFR
	
	Arguments:
	----------
	He_mass_fraction: array-like - float
		Helium mass fraction - between 0 and 1 

	electron_abundace_gas: array-like - float
		mean free electron number per proton; averaged over the mass of the gas particle

	internal_energy_gas = array-like - float
		Internal energy of gas that is used to calculate the temperature of the gas 
		Units: [m^2 s^-2]

	Returns:
	----------
	temperature_gas: array-like - float 
		Gas temperature
		Units: K

	References: 
	-----------
	 http://www.tapir.caltech.edu/~phopkins/Site/GIZMO_files/gizmo_documentation.html

	"""	

	y_helium = He_mass_fraction / (4*(1-He_mass_fraction))

	mu = (1 + 4*y_helium) / (1+y_helium+electron_abundace_gas)  # ignoring small corrections from the metals 

	proton_mass = 1.67262192 * 1e-27 # kg
	mean_molecular_weight = mu * proton_mass

	k_Boltzman = 1.380649 * 1e-23  # m2 kg s-2 K-1
	gamma = 5/3 # Adiabatic index
	temperature_gas = mean_molecular_weight * (gamma-1)  * internal_energy_gas / k_Boltzman

	return temperature_gas


####################################################################################################################################################################

def fh2_calculation_using_gas_temperature(temperature_gas,
										  neutral_hydrogen_fraction_gas):

	print("I am in the function fh2_calculation_using_gas_temperature")

	"""This function is used to calculate the molecular gas fraction using the temperature of the particles.
	Hydrogen has three states in the ISM. These are ionized hydrogen, neutral hydrogen and molecular hydrogen. If temperature is below 1000K it is assumed that there
	is no inoized hydrogen, then the molecular gas fraction is assumed to be 1-neutral_hydrogen_fraction.
	
	Arguments:
	----------
	temperature_gas: array-like - float
		temperature of the gas particles 
		Unit: K 

	neutral_hydrogen_fraction_gas: array-like - float
		This is the output of snapshots. It is the neutal hydrogen mass fraction of the gas particles 
		Unitless

	Returns:
	----------
	fh2: array-like - float 
		Molecular gas fraction
		Unitless

	References: 
	-----------
	 Norman Murray

	"""

	import numpy as np

	# Finding the indices where gas temperature smaller than 1000K 
	indices_gas_temperature_smaller_than_1000 = np.where(temperature_gas<1000)
	indices_gas_temperature_smaller_than_1000 = indices_gas_temperature_smaller_than_1000[0]

	print(indices_gas_temperature_smaller_than_1000)

	# Calculating the molecular gas fraction
	fh2 = np.zeros(len(temperature_gas))
	fh2[indices_gas_temperature_smaller_than_1000] = 1 - neutral_hydrogen_fraction_gas[indices_gas_temperature_smaller_than_1000]

	return fh2


####################################################################################################################################################################

def surface_density_calculator(star_formation_rate, mass_h2, smoothing_length_gas, R_gas, cut_off_radius):

	print("I am in the function surface_density_calculator")

	"""This function is used to calculate the surface densities of molecular gas and sfr. Two different surface densities is calculated. This function needs to be 
	revized and the appropiate calculation has to selected.
	
	Arguments:
	----------
	star_formation_rate: array-like - float
		star formation rate of gas particles
		Unit: M☉/year

	mass_h2: array-like - float
		Molecular gas mass  
		Unit: 10^10 M☉

	smoothing_length_gas = array-like float
		Smoothing length of the gas particles
		Unit: kpc

	R_gas = array-like - float
		Distance of gas particles from the center of the galaxy
		Unit: kpc

	cut_off_radius= array-like - float
		Maximum radius where surface densities will be calculated to 
		Unit: kpc

	Returns:
	----------
	sigma_sfr_dividing_total_area: array-like - float 
		sfr surface density calculated by summing up the sfr until cut_off_radius and dividing to the area
		Unit: M☉ year^-1 kpc^-2

	sigma_mass_h2_total_area: array-like - float 
		Molecular gas mass surface density calculated by summing up the sfr until cut_off_radius and dividing to the area
		Unit: M☉ pc^-2

	sigma_sfr_dividing_individual_area: array-like - float 
		sfr surface density calculated by taking the average of all the sfr of gas particles and dividing to smoothing_length**2 of 
		each gas particle and taking the average of all results. 
		Unit: M☉ year^-1 kpc^-2

	sigma_mass_h2_individual_area: array-like - float 
		Molecular gas mass surface density calculated by taking the average of all the Mh2 of gas particles and dividing to smoothing_length**2 of 
		each gas particle and taking the average of all results. 
		Unit: M☉ pc^-2		 

	References: 
	-----------

	"""
    
	import numpy as np

	# Units:
	# star_formation_rate: M☉/year
	# Lco: K-km s^-1 pc^2
	# mass_h2: 1e10 M☉
	# smoothing_length_gas: kpc
	# cut_off_radius: kpc

	R_gas_smaller_than_cut_off_radius = np.where(R_gas < cut_off_radius)
	R_gas_smaller_than_cut_off_radius = R_gas_smaller_than_cut_off_radius[0] 

	# Filtering the data such that I will only have particles within the cut-off radius from the center of the MMH
	star_formation_rate = star_formation_rate[R_gas_smaller_than_cut_off_radius]
	mass_h2 = mass_h2[R_gas_smaller_than_cut_off_radius] 
	smoothing_length_gas = smoothing_length_gas[R_gas_smaller_than_cut_off_radius]

	# Converting mass_h2 to M☉
	mass_h2 = mass_h2 * constants.ten_to_ten_times_Msun_to_Msun 
	# mass_h2: M☉

	# 1st method: 
	# Summing up all the star_formation_rate and mass_h2 and dividing into the area of a sphere.
	area_of_sphere = np.pi * cut_off_radius**2

	sigma_sfr_dividing_total_area = np.sum(star_formation_rate)/area_of_sphere
	# sigma_sfr_dividing_total_area: M☉ year^-1 kpc^-2  

	sigma_mass_h2_total_area = np.sum(mass_h2)/(area_of_sphere * constants.kpc2pc**2)
	# sigma_mass_h2_total_area: M☉ pc^-2

	# 2nd method: 
	# Diving star_formation_rate and mass_h2 to smoothing_length**2 and averaging sigma_star_formation_rate and sigma_mass_h2
	sigma_sfr_dividing_individual_area = star_formation_rate/smoothing_length_gas**2
	sigma_sfr_dividing_individual_area = np.sum(sigma_sfr_dividing_individual_area)/len(sigma_sfr_dividing_individual_area) 
	# sigma_sfr_dividing_individual_area: M☉ year^-1 kpc^-2	

	sigma_mass_h2_individual_area = mass_h2/(smoothing_length_gas*constants.kpc2pc)**2
	sigma_mass_h2_individual_area = np.sum(sigma_mass_h2_individual_area)/len(sigma_mass_h2_individual_area)
	# sigma_mass_h2_individual_area: M☉ pc^-2

	return (sigma_sfr_dividing_total_area, 
	       sigma_mass_h2_total_area, 
	       sigma_sfr_dividing_individual_area, 
	       sigma_mass_h2_individual_area)


####################################################################################################################################################################

def surface_density_calculator3(x_gas, y_gas, array_that_surface_density_needs_to_be_calculated, cut_off_radius):
    print("I am in the function surface_density_calculator3")

    """This function is used to calculate the surface densities of molecular gas and sfr. 
    First it creates a 2D mesh for all points between 0 < x < cut_off_radius and 0 < y < cut_off_radius. 
    Then, if the gas particles are in that mesh, it calculates the surface density in each grid. 
    Finally, it calculates the mean density.
    
    Arguments:
    ----------
    array_that_surface_density_needs_to_be_calculated: array-like - float
        Star formation rate of gas particles or mass_h2
        Unit: Depends on the input

    cut_off_radius: float
        Maximum radius where surface densities will be calculated to 
        Unit: kpc

    Returns:
    ----------
    mean_surface_density: float 
        Calculated average surface density for the inputted array
        Unit: .... kpc^-2
    """

    import numpy as np

    square_size = 1.0  # kpc

    x = np.arange(-1 * cut_off_radius, cut_off_radius + square_size, square_size)
    y = x 

    xx, yy = np.meshgrid(x, y)
    squares = np.dstack((xx, yy))

    # Define a function to calculate the SFR/area of a given square region
    def surface_density_of_every_region_calculator(x_gas, y_gas, array_that_surface_density_needs_to_be_calculated, square):
        # This outputs a boolean array. If the particle is within the square size distance/2 from the center of the square, 
        # then the boolean returns True; otherwise, it returns False.
        indices = ((x_gas - square[0])**2 + (y_gas - square[1])**2) <= (square_size / 2)**2   
        area = np.pi * (square_size / 2)**2
        # If Boolean returns True, then sfr_surface_density is calculated in that square for the corresponding particles. 
        return np.sum(array_that_surface_density_needs_to_be_calculated[indices]) / area	

    surface_densities = []
    for square in squares.reshape(-1, 2):
        surface_density_value_of_every_region = surface_density_of_every_region_calculator(
            x_gas=x_gas,
            y_gas=y_gas,
            array_that_surface_density_needs_to_be_calculated=array_that_surface_density_needs_to_be_calculated,
            square=square
        )
        surface_densities.append(surface_density_value_of_every_region)

    # Calculating the average surface density
    mean_surface_density = np.mean(surface_densities)

    return mean_surface_density


####################################################################################################################################################################


# I used the functions in the PFH/pfh_python/gadget_lib/cosmo.py

def quick_lookback_time(z,h=0.71,Omega_M=0.27):
    ## exact solution for a flat universe
    a=1./(1.+z) 
    x=Omega_M/(1.-Omega_M) / (a*a*a)
    t=(2./(3.*np.sqrt(1.-Omega_M))) * np.log( np.sqrt(x) / (-1. + np.sqrt(1.+x)) )
    t *= 13.777 * (0.71/h) ## in Gyr
    return t


def calculate_stellar_age(scale_factor_star:np.ndarray,
                          time:float,
						  h:float=0.71)->np.ndarray:
    
    a_form=scale_factor_star
    a_now=time    
    
    z_form=1./a_form-1.
    t_form=quick_lookback_time(z_form, h=h)
    
    z_now=1./a_now-1.
    t_now=quick_lookback_time(z_now, h=h)
    
    ages = (t_now - t_form); # should be in gyr    
        
    return ages


####################################################################################################################################################################

def sfr_calculator(star_df: pd.DataFrame, within_how_many_Myr:float):
    # Calculate star formation happened in the last 10 Myr
    indices = np.where(star_df["age"] <= within_how_many_Myr)[0]
    sfr_star = np.sum(star_df.iloc[indices]["mass"]) / (within_how_many_Myr * 1e6)  # Msolar / year
    return sfr_star


def SFR_luminosity(SFR):
    
    # Convert SFR to kg/second 
    epsilon_ff = 8e-4
    
    SFR_unit_converted = SFR * constants.Msolar2kg / constants.year2seconds # kg / sec
    L_sfr = SFR_unit_converted * constants.c**2 * epsilon_ff # Watts
    
    return L_sfr * constants.w2ergs # erg / s


def calculate_luminosity_from_sed(sed, min_wavelength, max_wavelenght, distance_in_Mpc):

	from scipy.integrate import simpson # This is put inside the function because it raises an error when used in Niagara
    
	condition = (sed['wavelength'] > min_wavelength) & (sed['wavelength'] < max_wavelenght) 
	filtered_sed = sed[condition].copy()    

	distance_meter = distance_in_Mpc * constants.Mpc2meter

	# Integrate the fir luminosity 
	bolometric_flux = simpson(
		y = filtered_sed['total_flux'],
		x = filtered_sed['wavelength']
	) # W / m2    

	lum = bolometric_flux * 4 * np.pi * distance_meter**2 # Watts

	return lum * constants.w2ergs # erg / s 
