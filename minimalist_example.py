from meta_surf import *

# create a unite cell of side length 0.03 m
ucell = simple_unit_cell()

# create a horn source
horn = simplified_horn_source()

# create a transmit array containing 20 x 20 cells illuminated by a horn source
# placed at 0.5 m from the metasurface
ta = transmit_array(20, 20, ucell, horn)

# create a desordered medium illuminated by the transmit array
dm = desordered_medium(ta)

# generate 10 randomly placed scatterers
dm.generate_random_scatterers(10)
dm.plot_scatterers()

# plot the electrical field in the xz plane
dm.plot_field(plane="xz", nb_side_pts=50)