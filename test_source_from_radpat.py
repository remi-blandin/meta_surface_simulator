from meta_surf import *
import numpy as np
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------#
# Test radiation pattern importation

rp = radiation_pattern("horn_realized_gain_y.csv", flip_ud=True)

theta = [0.1, 0.3, 0.9, 2.3]
phi = [0.2, 1.6, 2.7, -3.1]

# rp.value(theta, phi, plot=True)

#----------------------------------------------------------------------------#
# Test source modelling from radiation pattern

src = source_from_radpat(rp)

src.plot_field(plane="xz", corner_pt=point(-0.3,0.,-0.6), 
               nb_side_pts = 100, dB=True)
src.plot_field(plane="yz", corner_pt=point(0.,-0.3,-0.6), 
               nb_side_pts = 100, dB=True)
src.plot_field(plane="xy", corner_pt=point(-0.3,-0.3,0.), 
               nb_side_pts = 100, dB=True)

#----------------------------------------------------------------------------#
# Test rotation

rp.rotate(30*np.pi/180, -45*np.pi/180)

src = source_from_radpat(rp)

src.plot_field(plane="xz", corner_pt=point(-0.3,0.,-0.6), 
               nb_side_pts = 100, dB=True)
src.plot_field(plane="yz", corner_pt=point(0.,-0.3,-0.6), 
               nb_side_pts = 100, dB=True)
src.plot_field(plane="xy", corner_pt=point(-0.3,-0.3,0.), 
               nb_side_pts = 100, dB=True)