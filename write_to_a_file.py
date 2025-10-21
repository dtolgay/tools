import numpy as np
import os

def writing_gas_particles_into_txt_file(x_gas:np.ndarray,
                                        y_gas:np.ndarray,
                                        z_gas:np.ndarray,
                                        smoothing_length_gas:np.ndarray,
                                        mass_gas:np.ndarray,
                                        metallicity_gas:np.ndarray,
                                        snapshot_number:str,
                                        path:str):
    
    print("I am in function writing_gas_particles_into_txt_file")

    # Converting Units
    x_gas *= 1e3                 # in pc
    y_gas *= 1e3                 # in pc 
    z_gas *= 1e3                 # in pc 
    smoothing_length_gas *= 1e3  # in pc
    mass_gas = mass_gas * 1e10   # in Msun
    
    is_file_exist = os.path.isdir(path)
    
    if is_file_exist:
        file_name = path + '/gas_' + str(snapshot_number) + '.txt'
    else: 
        os.makedirs(path)
        print(f'{path} is created')
        file_name = path + '/gas_' + str(snapshot_number) + '.txt'
        
    print(len(x_gas), ' lines will be written')
    with open(file_name, 'w') as f:
        f.write('#Gas particles for a FIRE galaxy\n')
        f.write('#\n')
        f.write('#Column 1: x-coordinate (pc)\n')
        f.write('#Column 2: y-coordinate (pc)\n')
        f.write('#Column 3: z-coordinate (pc)\n')
        f.write('#Column 4: smoothing length (pc)\n')
        f.write('#Column 5: gas mass (Msun)\n')
        f.write('#Column 6: metallicity (1)\n')
        f.write('#\n')

        buffer = []
        for i in range(len(x_gas)):
            line = "{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                x_gas[i], y_gas[i], z_gas[i], smoothing_length_gas[i], mass_gas[i], metallicity_gas[i]
            )
            buffer.append(line)

            # Flush buffer every 1000 lines
            if i % 1000 == 0:
                f.write(''.join(buffer))
                buffer = []
                
            if i % 1e6 == 0:
                print(i, ' lines are written so far')
        # Write any remaining lines in the buffer
        if buffer:
            f.write(''.join(buffer))

    print(f'All {len(x_gas)} lines were written to ', file_name)        
    return 0


def writing_star_particles_into_txt_file(x_star:np.ndarray,
                                         y_star:np.ndarray,
                                         z_star:np.ndarray,
                                         smoothing_length_star:np.ndarray,
                                         metallicity_star:np.ndarray,
                                         age_star:np.ndarray,
                                         initial_mass_star_value:int,
                                         snapshot_number:int,
                                         path:str):
    
    print("I am in function writing_star_particles_into_txt_file")


    initial_mass_star = np.ones(len(x_star))*initial_mass_star_value # in Msun

    # Converting Units 
    x_star *= 1e3                       # in pc
    y_star *= 1e3                       # in pc
    z_star *= 1e3                       # in pc
    smoothing_length_star *= 1e3        # in pc
    age_star *= 1e9                     # in yr
    
    is_file_exist = os.path.isdir(path)
    
    if is_file_exist:
        file_name = path + '/star_' + str(snapshot_number) + '.txt'
    else: 
        os.makedirs(path)
        print(f'{path} is created')
        file_name = path + '/star_' + str(snapshot_number) + '.txt'
        
        
    print(len(x_star), ' lines will be written')
    with open(file_name, 'w') as f:
        f.write('#Star particles for a FIRE galaxy\n')
        f.write('#\n')
        f.write('#Column 1: x-coordinate (pc)\n')
        f.write('#Column 2: y-coordinate (pc)\n')
        f.write('#Column 3: z-coordinate (pc)\n')
        f.write('#Column 4: smoothing length (pc)\n')
        f.write('#Column 5: initial mass (Msun)\n')        
        f.write('#Column 6: metallicity (1)\n')
        f.write('#Column 7: age (yr)\n')
        f.write('#\n')

        buffer = []
        for i in range(len(x_star)):
            line = "{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                x_star[i], y_star[i], z_star[i], smoothing_length_star[i], initial_mass_star[i], metallicity_star[i], age_star[i]
            )
            buffer.append(line)

            # Flush buffer every 1000 lines
            if i % 1000 == 0:
                f.write(''.join(buffer))
                buffer = []
                
            if i % 1e6 == 0:
                print(i, ' lines are written so far')
        # Write any remaining lines in the buffer
        if buffer:
            f.write(''.join(buffer))

    print(f'All {len(x_star)} lines were written to ', file_name)                
    return 0