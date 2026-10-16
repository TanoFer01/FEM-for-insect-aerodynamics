
import sys
sys.path.append("/home/ferra/wing_dynamics/modules")
import plot as pltmd

directory = "runs/2026-10-08_15-14-43_alpha50_phi100_Re100_res0.01_dt0.07_L6.0/"#"Output_no_rotation_Re20/"#"Output_Re120_200/"
pltmd.plot_simulation(directory + "aerodynamic_history.txt", directory + "POD_reduced.npz", m=6)