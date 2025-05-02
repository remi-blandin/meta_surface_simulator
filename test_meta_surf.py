import numpy as np
import matplotlib.pyplot as plt

from meta_surf import *

# parameters
unit_cell_side_lgth = 0.03
dist_src = 0.5
wavelgth = 0.06
power = 1
n_cells_x = 20
n_cells_y = 20

ucell = simple_unit_cell(unit_cell_side_lgth)
horn = simplified_horn_source(5)

ta = transmit_array(n_cells_x, n_cells_y, ucell, horn, dist_src)
ta.set_random_phase_shift()

plt.figure()
plt.imshow(np.abs(ta.output_sigs()))
plt.show()

coords, ds, theta_in, phi_in, incoming_wave, output_sig = ta.output_signals(
    wavelgth, power)

plt.figure()
plt.imshow(np.abs(ta.output_sigs()))
plt.show()

rad_field = ta.radiated_field(0., 0., 0.2, wavelgth)
print(rad_field)

plt.figure()
plt.imshow(np.abs(ta.output_sigs()))
plt.show()

# plot coordinates
# plt.figure()
# for c in coords:
#     plt.plot(c[0], c[1], 'o')
# plt.show()

# plt.figure()
# plt.plot(ds)
# plt.show()

# plt.figure()
# plt.plot(ds, theta_in*180./np.pi)
# plt.plot(ds, phi_in*180./np.pi)
# plt.show()

# plot phase mask

plt.figure()
plt.imshow(ta.phase_shifts())
plt.show()

plt.figure()
plt.imshow(np.real(incoming_wave))
plt.show()

plt.figure()
plt.imshow(np.real(output_sig))
plt.show()

# plt.figure()
# plt.imshow(np.abs(incoming_wave))
# plt.show()

# plt.figure()
# plt.imshow(np.abs(output_sig))
# plt.show()

# generate a grid of points on which to compute the radiated field
ta_width = 0.6 #n_cells_x * unit_cell_side_lgth
n_x = 10
n_y = 10
x_rad = np.linspace(0, ta_width, n_x) - ta_width/2
y_rad = np.linspace(0, ta_width, n_y) - ta_width/2

rad_field = np.empty(n_x*n_y, dtype=np.complex128)

idx = 0
for x in x_rad:
    for y in y_rad:
        rad_field[idx] = ta.radiated_field(x, y, 0.2, wavelgth)
        idx = idx + 1
        
rad_field = rad_field.reshape(n_x, n_y)
        
plt.figure()
plt.imshow(np.abs(rad_field))