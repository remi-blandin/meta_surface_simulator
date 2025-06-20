from meta_surf import *
import numpy as np
import os
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------#
# Parameters

obs_pt = point(0.,0.,0.5)
side_dm = 0.2
side_field_view = 0.5
renew_scat = True
nb_scat = 100

#-----------------------------------------------------------------------------#
# create a horn source

horn = simplified_horn_source(position=point(0.,0.,-0.5))
horn.plot_field(plane="xz", side=side_field_view)

field_1m = horn.field(obs_pt)

print("Field from horn: " + str(np.abs(field_1m)))

#-----------------------------------------------------------------------------#
# create a desordered medium

dm = desordered_medium(horn)

# if they don't already exist, create scatterers 
file_name = "scatterers_position.csv"
if os.path.exists(file_name) and not(renew_scat):
    dm.create_scat_from_csv(file_name)
else:
    dm.generate_random_scatterers(nb_scat,
      bounding_box = [-side_dm, side_dm, -side_dm, side_dm, 0.1, 0.2])
    dm.save_scat_pos(file_name)

fig, ax = dm.plot_scatterers()
field_1m_dm = dm.field(obs_pt)
print("Field from horn: " + str(np.abs(field_1m_dm[0])))
print("Field from horn + desordered medium: " + str(np.abs(field_1m_dm[2])))

dm.plot_field(plane="yz", side=side_field_view)

#-----------------------------------------------------------------------------#
# add a transmit array

uc = simple_unit_cell()
ta = transmit_array(10, 10, uc, horn)
ta.plot(fig, ax)
dm = desordered_medium(ta)
dm.create_scat_from_csv(file_name)
field_1m_dm2 = dm.field(obs_pt)
print("Field from horn + transmit array: " + str(np.abs(field_1m_dm2[0])))
print("Field from horn + transmit array + desordered medium: " + str(np.abs(field_1m_dm2[2])))

dm.plot_field(plane="yz", side=side_field_view)

#-----------------------------------------------------------------------------#
# optimize a transmit array

phase_states = dm.source.phase_mask
nb_cells = len(phase_states)

field_obs = dm.field(obs_pt)
field_abs = np.empty(nb_scat)

for idx in range(0, nb_cells):
    phase_states[idx] = np.pi
    dm.source.set_phase_mask(phase_states)
    field = dm.field(obs_pt)
    
    field_abs[idx] = np.abs(field[2])
    print("iteration " + str(idx) + ": " + str(field_abs[idx]) )
    
    if np.abs(field[2]) < np.abs(field_obs[2]):
        phase_states[idx] = 0
    else:
        field_obs = field
    
plt.figure()
plt.plot(field_abs)
        
dm.source.plot_phase_mask()
dm.plot_field(plane="yz", side=side_field_view)
