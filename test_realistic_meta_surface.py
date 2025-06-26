from meta_surf import *
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.constants import c  # Speed of light in vacuum

freq = 5.2e+9
wavelgth = c / freq

#----------------------------------------------------------------------------#
# Create a realistic unit cell

angular_spacing = "5deg" # "2deg"

uc = unit_cell(wavelgth = wavelgth)

if angular_spacing == "2deg":

    # define radiation patterns and scattering matrix for the phase state 0. rad
    rp_in_0 = radiation_pattern("radpat_in_0.csv")
    rp_out_0 = radiation_pattern("radpat_out_0.csv", flip_ud=True)
    uc.set_rad_pat(rp_in_0, rp_out_0, "transmission_coef_state_0.s4p", 
                   phase_state=0.)
    
    # define radiation patterns and scattering matrix for the phase state pi rad
    rp_in_pi = radiation_pattern("radpat_in_pi.csv")
    rp_out_pi = radiation_pattern("radpat_out_pi.csv", flip_ud=True)
    uc.set_rad_pat(rp_in_pi, rp_out_pi, "transmission_coef_state_pi.s4p", 
                   phase_state=np.pi)
    
elif angular_spacing == "5deg":

    # define radiation patterns and scattering matrix for the phase state 0. rad
    rp_in_0 = radiation_pattern("radpat_in_0_5deg.csv")
    rp_out_0 = radiation_pattern("radpat_out_0_5deg.csv", flip_ud=True)
    uc.set_rad_pat(rp_in_0, rp_out_0, "transmission_coef_state_0.s4p", 
                   phase_state=0.)
    
    # define radiation patterns and scattering matrix for the phase state pi rad
    rp_in_pi = radiation_pattern("radpat_in_pi_5deg.csv")
    rp_out_pi = radiation_pattern("radpat_out_pi_5deg.csv", flip_ud=True)
    uc.set_rad_pat(rp_in_pi, rp_out_pi, "transmission_coef_state_pi.s4p", 
                   phase_state=np.pi)

#----------------------------------------------------------------------------#
# create a horn source

horn = simplified_horn_source(wavelgth = wavelgth)

#----------------------------------------------------------------------------#
# create transmit arrays containing 20 x 20 cells illuminated by a horn source

n_cells = 10

# a transmit array with simple cells
ucs = simple_unit_cell(wavelgth=wavelgth)
tas = transmit_array(n_cells, n_cells, ucs, horn)
tas.set_pahse_mask_alternate_lines()
tas.plot_phase_mask()

tas.plot_field(plane="xz", side=0.06)
# tas.plot_field(plane="yz", side=0.06)

# a transmit array with realistic cells
ta = transmit_array(n_cells, n_cells, uc, horn)
ta.set_pahse_mask_alternate_lines()

# test execution time of radiation pattern interpolation
start_time = time.time()
for idx in range(0,10000):
    rp_in_0.value(0., 0.)
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

start_time = time.time()
ta.plot_field(plane="xz", side=0.06)
# ta.plot_field(plane="yz", side=0.06)
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")