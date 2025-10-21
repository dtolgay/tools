import sys


def main(path2dir, center):

    file_name = f"hden{center['log_hden']:.5f}_metallicity{center['log_metallicity']:.5f}_turbulence{center['log_turbulence']:.5f}_isrf{center['log_isrf']:.5f}_radius{center['log_radius']:.5f}"
        
    print("file_name: \n", file_name)

    return None



if __name__ == '__main__':

    log_metallicity = float(sys.argv[1])
    log_hden = float(sys.argv[2])
    log_turbulence = float(sys.argv[3])
    log_isrf = float(sys.argv[4])
    log_radius = float(sys.argv[5])

    center = {
        "log_metallicity": log_metallicity,
        "log_hden": log_hden,
        "log_turbulence": log_turbulence,
        "log_isrf": log_isrf,
        "log_radius": log_radius
    }    

    path2dir = "/scratch/m/murray/dtolgay/cloudy_runs/z_0/cr_1_CO87_CII_H_O3/cr_1_CO87_CII_H_O3_metallicity_above_minus_2/python_files"

    main(path2dir, center)