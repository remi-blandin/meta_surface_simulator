import numpy as np
import matplotlib.pyplot as plt
import time

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
horn = simplified_horn_source(5)

ta = transmit_array(n_cells_x, n_cells_y, ucell, horn, dist_src)

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
                                    
ax1.imshow(ta.get_phase_mask())
ax1.set_title('Phase mask')

ax2.imshow(np.real(input_signals))
ax2.set_title("Input signal")

ax3.imshow(np.real(output_sig))
ax3.set_title("Output signal")

#----------------------------------------------------------------------------#
# compute the radiated field on point grids

ta_width = n_cells_x * unit_cell_side_lgth
n_points = 50
side_length = 1*ta_width

# # In the xy plane
bottom_corner = point(0., side_length/2, 0.)
g = point_grid_2d("yz", side_length, bottom_corner \
, nb_points_per_side = n_points)
g.plot()

rad_field_yz = np.empty(n_points * n_points, dtype=np.complex128)
for idx, pt in enumerate(g.points):
    rad_field_yz[idx] = ta.field(pt)

rad_field_yz = rad_field_yz.reshape(n_points, n_points)

# In the xz plane
bottom_corner = point(side_length/2, 0., 0.)
g = point_grid_2d("xz", side_length, bottom_corner \
, nb_points_per_side = n_points)
g.plot()

rad_field_xz = np.empty(n_points * n_points, dtype=np.complex128)
for idx, pt in enumerate(g.points):
    rad_field_xz[idx] = ta.field(pt)

rad_field_xz = rad_field_xz.reshape(n_points, n_points)

# in the yz plane
bottom_corner = point(side_length/2, side_length/2, 0.3)
g = point_grid_2d("xy", side_length, bottom_corner \
, nb_points_per_side = n_points)
g.plot()

rad_field_xy = np.empty(n_points * n_points, dtype=np.complex128)
for idx, pt in enumerate(g.points):
    rad_field_xy[idx] = ta.field(pt)

rad_field_xy = rad_field_xy.reshape(n_points, n_points)

#----------------------------------------------------------------------------#
# plot radiated field

fig, (ax1, ax2) = plt.subplots(1, 2)

max_value = np.max([np.abs(rad_field_yz).max(), \
                   np.abs(rad_field_xz).max()])

im = ax1.imshow(np.abs(rad_field_yz), vmax=max_value)
ax1.set_title("yz plane")

ax2.imshow(np.abs(rad_field_xz), vmax=max_value)
ax2.set_title("xz plane")

cbar = fig.colorbar(im, ax=[ax1, ax2], 
                   location='right', 
                   pad=0.02, 
                   shrink=0.5)
cbar.set_label('|E|')