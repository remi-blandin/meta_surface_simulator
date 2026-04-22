from meta_surf import *
import numpy as np
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------#
# Test radiation pattern importation

rp = radiation_pattern("horn_realized_gain_y_6GHz.csv")

theta = [0.1, 0.3, 0.9, 2.3]
phi = [0.2, 1.6, 2.7, -3.1]

rp.value(theta, phi, plot=False)

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

#----------------------------------------------------------------------------#
# Test metasurface illumination at different distances

rp.rotate(60*np.pi/180, -45*np.pi/180)

ucell = simple_unit_cell()
ta = transmit_array(8, 12, ucell, src)

ta.plot()

ta_length = 0.03*12

for idx,z in enumerate([-0.5,-0.75,-1]):
    
    ta.source.set_position(point(0.,0.,z))
    
    input_signals = 20.*np.log10(np.abs(ta.input_signals()))
    input_signals = input_signals - np.max(input_signals)
    
    if idx == 0:
        vmin = np.min(input_signals)
    
    plt.figure()
    plt.imshow(input_signals, vmin=vmin)
    plt.colorbar()
    plt.show()
    
    
    fig, axes = src.plot_field(plane="xy", corner_pt=point(-ta_length/2,-ta_length/2,0.), 
                  side=ta_length, nb_side_pts = 100, dB=True)
    ta.plot(fig, axes[0])
