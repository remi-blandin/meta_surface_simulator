from meta_surf import *
import numpy as np
import matplotlib.pyplot as plt
import time

#----------------------------------------------------------------------------#
# Create a realistic unit cell

uc = unit_cell()

# # define radiation patterns and scattering matrix for the phase state 0. rad
# rp_in_0 = radiation_pattern("radpat_in_0.csv")
# rp_out_0 = radiation_pattern("radpat_out_0.csv", flip_ud=True)
# uc.set_rad_pat(rp_in_0, rp_out_0, "transmission_coef_state_0.s4p", 
#                phase_state=0.)

# # define radiation patterns and scattering matrix for the phase state pi rad
# rp_in_pi = radiation_pattern("radpat_in_pi.csv")
# rp_out_pi = radiation_pattern("radpat_out_pi.csv", flip_ud=True)
# uc.set_rad_pat(rp_in_pi, rp_out_pi, "transmission_coef_state_pi.s4p", 
#                phase_state=np.pi)

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

horn = simplified_horn_source()

#----------------------------------------------------------------------------#
# create a transmit array containing 20 x 20 cells illuminated by a horn source

n_cells = 10

ucs = simple_unit_cell()
tas = transmit_array(n_cells, n_cells, ucs, horn)

tas.plot_field(plane="xz")

ta = transmit_array(n_cells, n_cells, uc, horn)
ta.field([point(0., 0., 0.5)])

start_time = time.time()
for idx in range(0,10000):
    rp_in_0.value(0., 0.)
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

start_time = time.time()
ta.plot_field(plane="xz")
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")