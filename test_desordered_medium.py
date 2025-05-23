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

nb_scat = 3

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
    ta.set_phase_mask_beam(theta_beam, phi_beam, wavelgth, quantize=quant)
elif phase_mask == "focal_point":
    ta.set_phase_mask_focal_point(focal_point, wavelgth, quantize=quant)
    
# compute input and output signals
input_signals = ta.inout_signals(wavelgth, power)
output_sig = ta.output_signals(wavelgth, power)

# plot phase mask, input and ouput signals
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                                    
ax1.imshow(ta.get_phase_mask())
ax1.set_title('Phase mask')

ax2.imshow(np.real(input_signals))
ax2.set_title("Input signal")

ax3.imshow(np.real(output_sig))
ax3.set_title("Output signal")

#----------------------------------------------------------------------------#
# Create desordered medium

nb_scat = 25
random_coordinates = np.random.rand(nb_scat, 2)/4.
scat_pos = [None] * nb_scat
for idx in range (0, nb_scat):
    scat_pos[idx] = point(random_coordinates[idx,0], 0., random_coordinates[idx, 1])
# scat_pos = [point(0.1, 0., 0.1), point(0.2, 0., 0.1), point(0.1, 0., 0.2)]
dm = desordered_medium(scat_pos, ta)

#----------------------------------------------------------------------------#
# compute the radiated field

obs_pts = [point(0., 0., 0.4), point(0., 0., 0.3)]
nb_pts = 50
g = g = point_grid_2d("xz", 0.5, point(0.15, 0., 0.), nb_pts)

rad_field = dm.field(g.points, wavelgth)

rad_field = rad_field.reshape((nb_pts, nb_pts))

plt.figure()
plt.imshow(np.abs(rad_field))
plt.show()

print(rad_field)