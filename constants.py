# Constants and conversion factors 

# The units of the fire simulation are in kpc for distance and 10^10 M_star for mass. I will use cgs units therefore some conversion will be 
# done in below lines
M_sun2gr            				= 1.99e33           # M☉ -> gr
Msolar2kg                           = 1.989e+30         # M☉ -> kg
kg2Msolar                           = 1/Msolar2kg       # kg -> M☉ 
gr2M_sun            				= 1/M_sun2gr        # gr -> M☉
gr2Msolar                           = gr2M_sun          # Same 
kg2g                                = 1e3               # kg -> gr
kg2Msun                             = 1/Msolar2kg       # kg -> Msolar

ten_to_ten_times_Msun_to_Msun     	= 1e10              # 1e10M☉ -> M☉

pc2cm               				= 3.08e18           # pc -> cm 
pc2m                                = pc2cm * 1e-2      # pc -> m
kpc2m                               = 1e3 * pc2m        # kpc -> m
kpc2cm              				= pc2cm*1e3         # kpc -> cm 
kpc2pc              				= 1e3               # kpc -> pc
pc2kpc              				= 1/kpc2pc          # pc -> kpc
Mpc2meter                           = 3.086e+22
cm2pc            				   	= 1/pc2cm           # cm -> pc
cm2kpc                              = cm2pc * 1e-3      # cm -> kpc
m2cm                                = 100               # m -> cm
cm2m                                = 1/m2cm            # cm -> m

year2gigaYear                       = 1e-9              # 1 year -> 1 gigayear
gigaYear2Megayear                   = 1e3               # 1 gigayear -> 1e3 million year
# Solar Metallicity information is taken from paper: 
# SOLAR METALLICITY DERIVED FROM IN SITU SOLAR WIND COMPOSITION by R. von Steiger and T. H. Zurbuchen, 2015
solar_metallicity 	= 0.02 			        # solar mass fraction
OH_solar            = 4.89e-4               # Assuming 12 + log(O/H solar) = 8.69 Accurso+2017 -- Deriving a multivariate αCO conversion function using the [C II]/CO (1-0) ratio and its application to molecular gas scaling relations
proton_mass         = 1.67262192e-27        # kg
proton_mass_gr      = proton_mass * kg2g    # gr
h                   = 4.1357e-15            # Plank's constant in eV s
c                   = 299792458             # m/s 
u_HAB               = 5.29e-15              # Habing energy density J/m^3
u_HAB_energy_density__erg_per_cm3   = 5.29e-14              # Habing energy density erg/cm^3
G0                  = 1.69                  # Habing radiation field Unitless
w2ergs              = 1e7                   # W -> erg/s
Lsolar2ergs         = 3.826e33              # Lsolar -> erg/s. I think the value that I am using right now is true  # 3.839e33??? $ 3.85e33 in lequeux the interstellar medium
ergs2Lsolar         = 1/Lsolar2ergs         # erg/s -> Lsolar
ev2K                = 11606                 # ev -> K
year2seconds        = 3.154e7               # year -> seconds 
kb                  = 1.380649e-23          # J/K  --- Boltzmann Constant
gravitational_constant = 6.6743e-11         # N m^2 kg^-2

mu_h                = 2.3e-24 / (proton_mass * kg2g)  # Effect of the helium atoms on the total mass ~1.37
dust2metal          = 0.4                  # metallicity to dust ratio
# From A COMPARISON OF METHODS FOR DETERMINING THE MOLECULAR CONTENT OF MODEL GALAXIES. Krumholz fh2 methadology paper


# Constants for atomic data
avogadro_number     = 6.022e23            # Avogadro's number 
mco_molecular_mass  = 28.01              # g/mol
mh2_molecular_mass  = 2.016              # g/mol
mh_molecular_mass   = 1.008              # g/mol
mc_molecular_mass   = 12.01              # g/mol
mo_molecular_mass   = 16.00              # g/mol

mh_gr               = mh_molecular_mass / avogadro_number  # g
mco_gr              = mco_molecular_mass / avogadro_number  # g
mh2_gr              = mh2_molecular_mass / avogadro_number  # g
mcii_gr             = mc_molecular_mass / avogadro_number  # g
moiii_gr            = mo_molecular_mass / avogadro_number  # g

# Cosmology 
eV2K                = 11606             # eV -> K