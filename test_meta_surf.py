import numpy as np
import matplotlib.pyplot as plt
import time

from meta_surf import *

#----------------------------------------------------------------------------#
# parameters

unit_cell_side_lgth = 0.03
dist_src = 0.25
wavelgth = 0.03
power = 1
n_cells_x = 20
n_cells_y = 20
# phase_mask = "devided_in_half"
phase_mask = "random"

#----------------------------------------------------------------------------#
# Initialise metasuurface

ucell = simple_unit_cell(unit_cell_side_lgth)
horn = simplified_horn_source(5)

ta = transmit_array(n_cells_x, n_cells_y, ucell, horn, dist_src)

if phase_mask == "random":
    ta.set_random_phase_shift()
elif phase_mask == "devided_in_half":
    ta.set_phase_shift_devided_in_half()

#FIXME: test and optimise this function
coords, ds, theta_in, phi_in, incoming_wave, output_sig = ta.output_signals(
    wavelgth, power)

# plot phase mask, input and ouput signals
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                                    
ax1.imshow(ta.phase_shifts())
ax1.set_title('Phase mask')

ax2.imshow(np.real(incoming_wave))
ax2.set_title("Input signal")

ax3.imshow(np.real(output_sig))
ax3.set_title("Output signal")

#----------------------------------------------------------------------------#
# compute the radiated field

# generate a grid of points on which to compute the radiated field
ta_width = n_cells_x * unit_cell_side_lgth
n_x = 50
n_y = 50
n_z = 50
x_rad = np.linspace(0, ta_width, n_x) - ta_width/2
y_rad = np.linspace(0, ta_width, n_y) - ta_width/2
z_rad = np.linspace(0, ta_width, n_y)

# In the xy plane
rad_field_yz = np.empty(n_x*n_y, dtype=np.complex128)
idx = 0
for z in z_rad:
    for y in y_rad:
        rad_field_yz[idx] = ta.radiated_field(0., y, z, wavelgth)
        idx = idx + 1  
rad_field_yz = rad_field_yz.reshape(n_y, n_z)

# In the xz plane
rad_field_xz = np.empty(n_x*n_y, dtype=np.complex128)
start_time = time.time() 
idx = 0
for z in z_rad:
    for x in x_rad:
        rad_field_xz[idx] = ta.radiated_field(x, 0., z, wavelgth)
        idx = idx + 1
end_time = time.time() 
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
rad_field_xz = rad_field_xz.reshape(n_x, n_z)

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