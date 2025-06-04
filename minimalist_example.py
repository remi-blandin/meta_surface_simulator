from meta_surf import *

# wavelength
wavelgth = 0.06

# create a unite cell of side length 0.03 m
ucell = simple_unit_cell(0.03)

# create a horn source
horn = simplified_horn_source()

# create a transmit array containing 20 x 20 cells illuminated by a horn source
# placed at 0.5 m from the metasurface
ta = transmit_array(20, 20, ucell, horn, 0.5)

# generate a phase mask to focus waves at a point located 0.3 m in front
ta.set_phase_mask_focal_point(point(0., 0., 0.15), wavelgth, quantize=False)

input_signals = ta.inout_signals(wavelgth, 1.)
output_sig = ta.output_signals(wavelgth, 1.)

# create a desordered medium illuminated by the transmit array
dm = desordered_medium(ta)

# generate 10 randomly placed scatterers
dm.generate_random_scatterers(10)
dm.plot_scatterers()

# plot the electrical field in the xz plane
dm.plot_field(wavelgth, plane="xz", nb_side_pts=100)