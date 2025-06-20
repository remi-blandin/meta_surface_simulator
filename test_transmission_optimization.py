from meta_surf import *
import numpy as np
import os

obs_pt = point(0.,0.,0.5)

# create a horn source
horn = simplified_horn_source(position=point(0.,0.,-0.5))
horn.plot_field(plane="xz", side=1.)

field_1m = horn.field(obs_pt)

print("Field from horn: " + str(np.abs(field_1m)))

# create a desordered medium
dm = desordered_medium(horn)

# if they don't already exist, create scatterers 
file_name = "scatterers_position.csv"
if os.path.exists(file_name):
    dm.create_scat_from_csv(file_name)
else:
    dm.generate_random_scatterers(50)
    dm.save_scat_pos(file_name)

fig, ax = dm.plot_scatterers()
field_1m_dm = dm.field(obs_pt)
print("Field from horn + desordered medium: " + str(np.abs(field_1m_dm[2])))

dm.plot_field(plane="yz", side=1.)

# add a transmit array
uc = simple_unit_cell()
ta = transmit_array(10, 10, uc, horn)
ta.plot(fig, ax)
dm = desordered_medium(ta)
dm.create_scat_from_csv(file_name)
field_1m_dm2 = dm.field(obs_pt)
print("Field from horn + transmit array + desordered medium: " + str(np.abs(field_1m_dm2[2])))

dm.plot_field(plane="yz", side=1.)
