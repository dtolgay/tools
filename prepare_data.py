import glob
import numpy as np
import constants
import pandas as pd

##########################################################################################################################################################################################################

def crete_df_for_particles(particle_dict: dict,
                           ptype: int)->pd.DataFrame:
    
	particle_dict['p_x'] = particle_dict['p'][:,0]
	particle_dict['p_y'] = particle_dict['p'][:,1]
	particle_dict['p_z'] = particle_dict['p'][:,2]

	particle_dict['v_x'] = particle_dict['v'][:,0]
	particle_dict['v_y'] = particle_dict['v'][:,1]
	particle_dict['v_z'] = particle_dict['v'][:,2]    


	z = particle_dict['z'][:,0] 
	del particle_dict['z']
	particle_dict['z'] = z


	del particle_dict['p']
	del particle_dict['v']    

	if (ptype==0):          
		gas_particle_df = pd.DataFrame(data=particle_dict)
		gas_particle_df['m'] = gas_particle_df['m']  

		#Units: 
		# h:   kpc 
		# sfr: M☉/year
		# p_x: kpc 
		# p_y: kpc
		# p_z: kpc
		# v_x: km/sec
		# v_y: km/sec
		# v_z: km/sec
		# z:   mass fraction   

		return gas_particle_df
        
	if (ptype==4):
		star_particle_df = pd.DataFrame(data=particle_dict)
		# star_particle_df['h'] = star_particle_df['z']*2000

		#Units: 
		# age: scale factor 
		# p_x: kpc 
		# p_y: kpc
		# p_z: kpc
		# v_x: km/sec
		# v_y: km/sec
		# v_z: km/sec
		# z:   mass fraction

		return star_particle_df
    
	else: 
		print('Enter a valid ptype!')
		return 99


##########################################################################################################################################################################################################
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


	MMH_dict = {'mass': mvir_mph,
	            'p_x': x_mph,
	            'p_y': y_mph,
	            'p_z': z_mph,
	            'v_x': vx_mph,
	            'v_y': vy_mph,
	            'v_z': vz_mph
	#             'ID': ID,
	#             'DescID': DescID
	            }


	MMH = pd.DataFrame(data=[MMH_dict])	

	# return mvir_mph, x_mph, y_mph, z_mph, vx_mph, vy_mph, vz_mph, ID, DescID
	return MMH

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

