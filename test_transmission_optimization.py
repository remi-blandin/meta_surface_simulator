from meta_surf import *
import numpy as np
import os
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------#
# Parameters

# obs_pt = [point(0.,0.15,0.5), point(0.,-0.15,0.5) ]
# obs_pt = [point(0.,0.15,0.2), point(0.,-0.15,0.3) ]
# obs_pt = point(0.,0.,0.5)
# obs_pt = [point(0.,-0.02,0.5), point(0.,0.02,0.5), point(0.,-0.02,0.48), point(0.,0.02,0.48)]
obs_pt = [point(0.,0.,0.5), point(0.,0.,0.25) ]


side_dm = 0.2
side_field_view = 0.5
renew_scat = False
nb_cells_side = 14
nb_scat = 100

#-----------------------------------------------------------------------------#
# create a horn source

horn = simplified_horn_source(position=point(0.,0.,-0.5))
# horn.plot_field(plane="xz", side=side_field_view)

# field_1m = horn.field(obs_pt)

# print("Field from horn: " + str(np.abs(field_1m)))

#-----------------------------------------------------------------------------#
# create a desordered medium

dm = desordered_medium(horn)

# if they don't already exist, create scatterers 
file_name = "scatterers_position.csv"
if os.path.exists(file_name) and not(renew_scat):
    dm.create_scat_from_csv(file_name)
else:
    dm.generate_random_scatterers(nb_scat,
      bounding_box = [-side_dm, side_dm, -side_dm, side_dm, 0.05, 0.1])
    dm.save_scat_pos(file_name)

# fig, ax = dm.plot_scatterers()
# field_1m_dm = dm.field(obs_pt)
# print("Field from horn: " + str(np.abs(field_1m_dm[0])))
# print("Field from horn + desordered medium: " + str(np.abs(field_1m_dm[2])))

# dm.plot_field(plane="yz", side=side_field_view)

#-----------------------------------------------------------------------------#
# add a transmit array

uc = simple_unit_cell()
ta = transmit_array(nb_cells_side, nb_cells_side, uc, horn)
# ta.plot(fig, ax)

dm = desordered_medium(ta)
dm.create_scat_from_csv(file_name)
dm.set_polarizability(0.001)
# dm.source.set_random_phase_mask()

field_1m_dm2 = dm.field(obs_pt)
print("Field from horn + transmit array: " + str(np.abs(field_1m_dm2[0]).sum()))
print("Field from horn + transmit array + desordered medium: " 
      + str(np.abs(field_1m_dm2[2]).sum()))

dm.plot_field(plane="yz", side=side_field_view)

#-----------------------------------------------------------------------------#
# optimize a transmit array

nb_repeat = 1

phase_states = dm.source.phase_mask
nb_cells = len(phase_states)

field_obs = dm.field(obs_pt)
field_abs = np.empty(nb_cells * nb_repeat)

for r in range(0, nb_repeat):
    for idx in range(0, nb_cells):
        phase_states[idx] = np.pi
        dm.source.set_phase_mask(phase_states)
        field = dm.field(obs_pt)
        
        field_abs[r*nb_cells + idx] = np.abs(field[2]).sum()
        print("iteration " + str(idx + 1) + ": " 
              + str(field_abs[r*nb_cells + idx]) )
        
        # if np.abs(field[2]).sum() < np.abs(field_obs[2]).sum():
        if (np.abs(field[2][0,0]) < np.abs(field_obs[2][0,0])) and \
        (np.abs(field[2][0,1]) > np.abs(field_obs[2][0,1])):
            phase_states[idx] = 0
        else:
            field_obs = field
            
        # if idx % 20 == 00:
        #     dm.plot_field(plane="yz", side=side_field_view)
    
plt.figure()
plt.plot(field_abs)
        
dm.source.plot_phase_mask()
dm.plot_field(plane="yz", side=side_field_view)
