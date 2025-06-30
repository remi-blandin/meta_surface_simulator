import numpy as np
import matplotlib.pyplot as plt

from meta_surf import *

#----------------------------------------------------------------------------#
# parameters

unit_cell_side_lgth = 0.03
dist_src = 10
wavelgth = 0.06
power = 1
n_cells_x = 20
n_cells_y = 20
# phase_mask = "devided_in_half"
phase_mask = "random"

# phase_mask = "beam"
theta_beam = 2*np.pi/3
phi_beam = 0

# phase_mask = "focal_point"
focal_point = point(0., 0., 0.3)

quant = False

#----------------------------------------------------------------------------#
# Initialise metasuurface

ucell = simple_unit_cell(unit_cell_side_lgth)
horn = simplified_horn_source(5, position=point(0.,0.,dist_src))

ta = transmit_array(n_cells_x, n_cells_y, ucell, horn)

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
# compute the radiated field on point grids

ta.plot_field(plane="xz")
ta.plot_field(plane="yz")
ta.plot_field(plane="xy")