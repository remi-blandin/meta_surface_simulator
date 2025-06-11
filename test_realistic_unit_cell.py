from meta_surf import *
import numpy as np

#----------------------------------------------------------------------------#
# Test interpolation of the radiation patterns

rp_in = radiation_pattern("radpat_in_0.csv")

theta = [0.1, 0.3, 0.9, 2.3]
phi = [0.2, 1.6, 2.7, -3.1]

rp_in.value(theta, phi, plot=True)

#----------------------------------------------------------------------------#
# Test radiation pattern flipping

rp_out = radiation_pattern("radpat_out_0.csv")
rp_out.plot()

rp_out_ud = radiation_pattern("radpat_out_0.csv", flip_ud=True)
rp_out_ud.plot()

#----------------------------------------------------------------------------#
# create the unit cell using the imported radiation patterns

uc = unit_cell()
uc.set_rad_pat(rp_in, rp_out_ud, phase_state=0.)

# add radiation patterns for the phase state pi
rp_in_pi = radiation_pattern("radpat_in_pi.csv")
rp_out_pi = radiation_pattern("radpat_out_pi.csv", flip_ud=True)
uc.set_rad_pat(rp_in_pi, rp_out_pi, phase_state=np.pi)

uc.plot_rad_pats()