import numpy as np
import matplotlib.pyplot as plt
import os

from meta_surf import *

#----------------------------------------------------------------------------#
# parameters

unit_cell_side_lgth = 0.03
dist_src = 0.5
n_cells_x = 20
n_cells_y = 20

source_type = "plane_wave"
# source_type = "horn"

# phase_mask = "devided_in_half"
# phase_mask = "random"

# phase_mask = "beam"
theta_beam = 2*np.pi/3
phi_beam = 0

phase_mask = "focal_point"
focal_point = point(0., 0., 0.3)

quant = False


#----------------------------------------------------------------------------#
# Initialise metasuurface

ucell = simple_unit_cell(unit_cell_side_lgth)

if source_type == "horn":
    source = simplified_horn_source(5, position=point(0.,0.,dist_src))
elif source_type == "plane_wave":
    source = plane_wave(position=point(0.,0.,dist_src))

ta = transmit_array(n_cells_x, n_cells_y, ucell, source)

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
# Create desordered medium
    
dm = desordered_medium(ta)

file_name = "scatterers_position.csv"
if os.path.exists(file_name):
    dm.create_scat_from_csv(file_name)
else:
    dm.generate_random_scatterers(50)
    dm.save_scat_pos(file_name)

dm.plot_scatterers()

#----------------------------------------------------------------------------#
# compute the radiated field

dm.plot_field(plane="xz", nb_side_pts=100)
dm.plot_field(plane="yz")
dm.plot_field(plane="xy")
