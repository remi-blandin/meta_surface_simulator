from meta_surf import *
import numpy as np
import matplotlib.pyplot as plt

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
# create the unit cell using the imported radiation patterns and the saved
# scattering matrices 

uc = unit_cell()
uc.set_rad_pat(rp_in, rp_out_ud, "transmission_coef_state_0.s4p", 
               phase_state=0.)

# add radiation patterns for the phase state pi
rp_in_pi = radiation_pattern("radpat_in_pi.csv")
rp_out_pi = radiation_pattern("radpat_out_pi.csv", flip_ud=True)
uc.set_rad_pat(rp_in_pi, rp_out_pi, "transmission_coef_state_pi.s4p", 
               phase_state=np.pi)

uc.plot_rad_pats()
uc.plot_scat_mats()

#----------------------------------------------------------------------------#
# Compute the input signal of the unit cell

theta = np.linspace(0., np.pi/2., 101)
phi = np.zeros(101)
phase = 0.1 * np.ones(101)
dist = 0.05 * np.ones(101)

a0 = uc.input_sig(1., theta, phi, phase)
b0 = uc.output_sig(a0, phase)
rad_field = uc.field_from_sig(b0, dist, theta, phi, phase)

plt.figure()
plt.plot(theta*180./np.pi, a0, label="a0")
plt.plot(theta*180./np.pi, np.abs(b0), label="b0")
plt.plot(theta*180./np.pi, np.abs(rad_field), label="rad_field")
plt.legend()
plt.xlabel("theta (deg)")
plt.ylabel("Ampplitude")
plt.show()

#----------------------------------------------------------------------------#
# plot the radiated field

uc.plot_field(plane="xz", side=0.03)
uc.plot_field(plane="yz", side=0.03)
uc.plot_field(plane="xy", side=0.03)