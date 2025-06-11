from meta_surf import *
import numpy as np
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------#
# Create a realistic unit cell

uc = unit_cell()

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

#----------------------------------------------------------------------------#
# create a horn source

horn = simplified_horn_source()

#----------------------------------------------------------------------------#
# create a transmit array containing 20 x 20 cells illuminated by a horn source

ta = transmit_array(20, 20, uc, horn)