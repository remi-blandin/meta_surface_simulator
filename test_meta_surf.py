import numpy as np
import matplotlib.pyplot as plt

from meta_surf import *

# parameters
unit_cell_side_lgth = 0.03
dist_src = 0.5
wavelgth = 0.06
power = 1
n_cells_x = 2
n_cells_y = 1

ucell = simple_unit_cell(unit_cell_side_lgth)
horn = simplified_horn_source(5)

ta = transmit_array(n_cells_x, n_cells_y, ucell, horn, dist_src)
# ta.set_random_phase_shift()
ta.set_specific_phase_shift()

# plt.figure()
# plt.imshow(np.abs(ta.output_sigs()))
# plt.show()

coords, ds, theta_in, phi_in, incoming_wave, output_sig = ta.output_signals(
    wavelgth, power)

# plt.figure()
# plt.imshow(np.abs(ta.output_sigs()))
# plt.show()

rad_field = ta.radiated_field(0., 0., 0.2, wavelgth)
print(rad_field)

# plt.figure()
# plt.imshow(np.abs(ta.output_sigs()))
# plt.show()

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
# ta_width = 0.6 
ta_width = n_cells_x * unit_cell_side_lgth
n_x = 50
n_y = 50
n_z = 50
x_rad = np.linspace(0, ta_width, n_x) - ta_width/2
y_rad = np.linspace(0, ta_width, n_y) - ta_width/2
z_rad = np.linspace(0, wavelgth, n_y)

rad_field_yz = np.empty(n_x*n_y, dtype=np.complex128)

idx = 0
for z in z_rad:
    for y in y_rad:
        rad_field_yz[idx] = ta.radiated_field(0., y, z, wavelgth)
        idx = idx + 1
        
rad_field_yz = rad_field_yz.reshape(n_y, n_z)
        
plt.figure()
# plt.imshow(np.abs(rad_field))
plt.imshow(np.real(rad_field_yz))
# plt.imshow(np.abs(np.real(rad_field)))

rad_field_xz = np.empty(n_x*n_y, dtype=np.complex128)

idx = 0
for z in z_rad:
    for x in x_rad:
        rad_field_xz[idx] = ta.radiated_field(x, 0., z, wavelgth)
        idx = idx + 1
        
rad_field_xz = rad_field_xz.reshape(n_x, n_z)
        
plt.figure()
plt.imshow(np.real(rad_field_xz))