from meta_surf import *

rp_in = radiation_pattern("radpat_in_0.csv")

theta = [0.1, 0.3, 0.9, 2.3]
phi = [0.2, 1.6, 2.7, -3.1]

rp_in.value(theta, phi, plot=True)

rp_out = radiation_pattern("radpat_out_0.csv")



# uc = unit_cell()

# p, t, dp = uc.set_input_dirpat("test_directivity.csv", phase_state=0.)