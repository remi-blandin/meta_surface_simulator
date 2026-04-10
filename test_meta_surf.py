import numpy as np
import matplotlib.pyplot as plt

from meta_surf import *

#----------------------------------------------------------------------------#
# parameters

unit_cell_side_lgth = 0.03
dist_src = -0.5
# src_type = "plane_wave"
src_type = "horn"
wavelgth = 0.06
power = 1
n_cells_x = 8
n_cells_y = 12
# phase_mask = "uniform"
# phase_mask = "devided_in_half"
# phase_mask = "random"
phase_mask = "focal_point"
# phase_mask = "beam"

theta_beam = np.pi/2
phi_beam = np.pi/4

focal_point = point(10, 30*np.pi/180, 30*np.pi/180, spherical_coord=True)

quant = False

#----------------------------------------------------------------------------#
# Initialise metasuurface

ucell = simple_unit_cell(unit_cell_side_lgth)
if src_type == "plane_wave":
    src = plane_wave()
elif src_type == "horn":
    src = simplified_horn_source(5, position=point(0.,0.,dist_src))

ta = transmit_array(n_cells_x, n_cells_y, ucell, src)

# generate phase mask
if phase_mask == "random":
    ta.set_random_phase_mask()
elif phase_mask == "devided_in_half":
    ta.set_phase_mask_devided_in_half()
elif phase_mask == "beam":
    ta.set_phase_mask_beam(theta_beam, phi_beam, quantize=quant)
elif phase_mask == "focal_point":
    ta.set_phase_mask_focal_point(focal_point, quantize=quant)
    
# compute input and output signals
input_signals = ta.input_signals()
output_sig = ta.output_signals()

# plot phase mask, input and ouput signals
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                                    
ax1.imshow(ta.phase_mask.reshape(n_cells_x, n_cells_y))
ax1.set_title('Phase mask')

ax2.imshow(np.real(input_signals))
ax2.set_title("Input signal")

ax3.imshow(np.real(output_sig))
ax3.set_title("Output signal")

#----------------------------------------------------------------------------#
# Compute the radiated pattern

ta.radiation_pattern()

#----------------------------------------------------------------------------#
# compute the radiated field on point grids

res = 20

ta.plot_field(plane="xz", nb_side_pts=res)
# ta.plot_field(plane="yz", nb_side_pts=res)
# ta.plot_field(plane="xy", nb_side_pts=res)

# ta.plot_field(plane="xz", nb_side_pts=res, corner_pt=point(0.3,0., 0.3))
