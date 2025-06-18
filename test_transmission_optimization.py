from meta_surf import *
import numpy as np

obs_pt = point(0.,0.,1.)

# create a horn source
horn = simplified_horn_source(position=point(0.,0.,0.))
horn.plot_field(plane="xz", side=1.)

field_1m = horn.field(obs_pt)

print(np.abs(field_1m))

# create a desordered medium
dm = desordered_medium(horn)

dm.plot_scatterers()
field_1m_dm = dm.field(obs_pt)
print(np.abs(field_1m_dm[2]))

dm.plot_field(plane="yz", side=1.)

# add a transmit array
uc = simple_unit_cell()
ta = transmit_array(10, 10, uc, horn)
dm = desordered_medium(ta)

dm.plot_field(plane="yz", side=1.)