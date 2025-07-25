import numpy as np
import matplotlib.pyplot as plt

from meta_surf import *

# parameters
phi = 0.
theta = np.linspace(-np.pi/2., np.pi/2., 101)
dist = 0.5
power = 1.


# create a unit cell
ucell = simple_unit_cell(0.03*0.03)
directivity = ucell.directivity(theta, phi)
input_sig = ucell.input_sig(1., theta, phi)

# plot directivity
plt.figure()
plt.plot(theta, directivity)
plt.show()

horn_order = range(1, 6)
plt.figure()

for order in horn_order:
    horn = simplified_horn_source(order)
    dir_horn = horn.directivity(theta, phi)
    
    # plot horn parameters
    plt.plot(theta, 20*np.log10(dir_horn), label=str(order))
plt.legend()
plt.show()

# test horn radiation
horn = simplified_horn_source(5)
points = []
x = np.linspace(-0.15, 0.15, 101)
for idx in range(0, 101):
    points.append(point(x[idx], 0., 0.15))
rad_field = horn.field(points)

plt.figure()
plt.plot(x, np.abs(rad_field)[0])
plt.show()