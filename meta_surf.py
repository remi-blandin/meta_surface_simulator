import numpy as np
from typing import Union
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Slider
import csv
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates
from scipy.constants import c  # Speed of light in vacuum
import skrf as rf

__all__ = ["point", "point_grid_2d", "simple_unit_cell", "unit_cell",
           "simplified_horn_source", "transmit_array", "desordered_medium",
           "radiation_pattern", "field_calculator"]

##############################################################################

class point:
    
    """A cartesian point"""
    
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        
    def __repr__(self):
        return f"Point(x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f})"
        
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def distance_to(self, other_point):
        return np.sqrt((self.x - other_point.x)**2 + \
                       (self.y - other_point.y)**2 + \
                       (self.z - other_point.z)**2)
            
##############################################################################

class point_grid_2d:
    
    """A 2d squarred grid of cartesian points"""
    
    def __init__(self, orientation : str, side_length,\
                 bottom_corner= point(0., 0., 0.) \
                 , nb_points_per_side=50):
        
        self.bottom_corner = bottom_corner
        self.nb_pts = nb_points_per_side * nb_points_per_side
        self.points = [None] * self.nb_pts
        self.bounding_box = [None] * 4
        
        side_coord = np.linspace(0., side_length, nb_points_per_side)
        
        idx = 0
        for i in range(0, nb_points_per_side):
            for j in range(0, nb_points_per_side):
                
                if orientation == "xy":
                    self.points[idx] = point(side_coord[i] - bottom_corner.x,\
                                             side_coord[j] - bottom_corner.y,\
                                             bottom_corner.z)
                    self.bounding_box = [- bottom_corner.x, \
                                         side_length - bottom_corner.x, \
                                         - bottom_corner.y, \
                                         side_length - bottom_corner.y]
                        
                elif orientation == "xz":
                    self.points[idx] = point(side_coord[i] - bottom_corner.x,\
                                             bottom_corner.y,\
                                             side_coord[j] - bottom_corner.z)
                    self.bounding_box = [- bottom_corner.x, \
                                         side_length - bottom_corner.x, \
                                         - bottom_corner.z, \
                                         side_length - bottom_corner.z]
                        
                elif orientation == "yz":
                    self.points[idx] = point(bottom_corner.x, \
                                             side_coord[i] - bottom_corner.y,\
                                             side_coord[j] - bottom_corner.z)
                    self.bounding_box = [- bottom_corner.y, \
                                         side_length - bottom_corner.y, \
                                         - bottom_corner.z, \
                                         side_length - bottom_corner.z]
                idx = idx + 1
                
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def x(self):
        return [point.x for point in self.points]
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def y(self):
        return [point.y for point in self.points]
    
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def z(self):
        return [point.z for point in self.points]

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.x(), self.y(), self.z())
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
        
##############################################################################

class radiation_pattern:
    
    """A pattern defined over 2 angular spherical coordinates"""
    
    def __init__(self, csv_file, flip_ud=False):
        
        self.load_pattern_from_csv(csv_file, flip_ud)
        
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def load_pattern_from_csv(self, csv_file, flip_ud):
        self.phi = []
        all_phis_picked = False
        self.theta = []
        all_theta_picked = False
        dir_pat = []
        
        with open(csv_file, mode='r') as file:
            reader = csv.reader(file)
            
            # skip header
            next(reader) 
            
            # extract first row
            row = next(reader)
            self.phi.append(float(row[0]))
            self.theta.append(float(row[1]))
            dir_pat.append(float(row[2]))
            
            for row in reader:
                
                # extract phi coordinate
                p = float(row[0])
                if p < self.phi[-1]:
                    all_phis_picked = True
                if not(all_phis_picked) and p != self.phi[-1]:
                    self.phi.append(p)
                
                # extract theta coordinate
                t = float(row[1])
                if t < self.theta[-1]:
                    all_theta_picked = True
                if not(all_theta_picked) and t != self.theta[-1]:
                    self.theta.append(t)
                
                dir_pat.append(float(row[2]))
                
        self.n_phi = len(self.phi)
        self.n_theta = len(self.theta)
        
        # convert to radians
        self.phi = np.array(self.phi) * np.pi / 180.
        self.d_phi = self.phi[1] - self.phi[0]
        self.theta = np.array(self.theta) * np.pi / 180.
        self.d_theta = self.theta[1] - self.theta[0]

        dir_pat = np.array(dir_pat)
        
        if flip_ud:
            dir_pat = np.flipud(dir_pat)
        
        self.rad_pat = dir_pat.reshape((self.n_theta, self.n_phi))
        
        # self.interpolator = RegularGridInterpolator(
        #     (self.theta, self.phi),  # Grid points
        #     self.rad_pat,                  # Grid values
        #     method='linear'          # 'linear', 'nearest', 'slinear', 'cubic'
        # )
        
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def plot(self):
        phi, theta = np.meshgrid(self.phi, self.theta)
        
        # convert to radian
        phi = phi
        theta = theta
        
        # convert to cartesian coordinates
        x = self.rad_pat * np.sin(theta) * np.cos(phi)
        y = self.rad_pat * np.sin(theta) * np.sin(phi)
        z = self.rad_pat * np.cos(theta)
        
        max_val = self.rad_pat.max()
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z, 
                        facecolors=plt.cm.viridis(self.rad_pat / max_val)
                        )
        ax.set_box_aspect([1, 1, 1])
        
        return fig, ax
        
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def value(self, theta, phi, plot=False):
            
        # prefilter Nans as it can make the interpolation crash
        nan_mask = np.isnan(theta)
        theta = np.where(nan_mask, 0, theta)
        nan_mask = np.isnan(phi)
        phi = np.where(nan_mask, 0, phi)
        
        if theta.ndim == 0:
            theta = np.array([theta])
            
        if phi.ndim == 0:
            phi = np.array([phi])
        
        querry_pts = np.array([(theta - self.theta[0])/self.d_theta ,
            (phi - self.phi[0]) / self.d_phi])
        
        # interp_val = self.interpolator(np.array([theta, phi]).T)
        
        interp_val = map_coordinates(
            self.rad_pat,
            querry_pts,  # Transpose to (2, N) shape
            order=1,         # Linear interpolation
            mode='nearest'   # Handle out-of-bounds
        )
        
        # --- Draw spheres at interpolated points ---
        def add_sphere(ax, x, y, z, radius=0.05, color='red'):
            """Add a 3D sphere to the axis."""
            # Generate sphere vertices
            u = np.linspace(0, 2 * np.pi, 10)
            v = np.linspace(0, np.pi, 10)

            x_sphere = x + radius * np.outer(np.cos(u), np.sin(v)).ravel()
            y_sphere = y + radius * np.outer(np.sin(u), np.sin(v)).ravel()
            z_sphere = z + radius * np.outer(np.ones(np.size(u)), np.cos(v)).ravel()
            
            # Combine into (N, 3) array
            verts = np.column_stack([x_sphere, y_sphere, z_sphere])
            verts = verts.reshape(-1, 3)
            
            sphere = Poly3DCollection([verts], alpha=1.0, linewidths=0.5, 
                                      edgecolors='black', facecolors=color)
            
            ax.add_collection3d(sphere)
        
        if plot:
            fig, ax = self.plot()
            
            # convert to cartesian coordinates
            x = interp_val * np.sin(theta) * np.cos(phi)
            y = interp_val * np.sin(theta) * np.sin(phi)
            z = interp_val * np.cos(theta)
            
            for xi, yi, zi in zip(x, y, z):
                add_sphere(ax, xi, yi, zi, radius=0.1, color='red')
        
        return interp_val

##############################################################################

class simple_unit_cell:
    """A simple model for a unit cell"""
    
    def __init__(self, side_length=0.03, wavelgth=0.06):
        self.side_length = side_length
        self.area = np.square(side_length)
        self.wavelgth = wavelgth
        
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        
    def directivity(self, theta, phi):
        return 4.*np.pi * self.area * np.square(np.cos(theta)) \
    / np.square(self.wavelgth)
    
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    
    def input_sig(self, incoming_wave, theta, phi):
        return incoming_wave * self.directivity(theta, phi)
    
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    
    def output_sig(self, input_sig, phase_shift):
        return input_sig * np.exp(-1j * phase_shift)
            
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
            
    def field(self, incoming_wave, theta, phi, phase_shift, \
                       dist, theta_out, phi_out):
        input_sig = self.input_sig(incoming_wave, theta, phi)
        return self.output_sig(input_sig, phase_shift) \
            * self.directivity(theta_out, phi_out) \
                * self.wavelgth * np.exp(-1j * 2. * np.pi * dist / self.wavelgth) \
                    /4. / np.pi / dist
                    
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
                    
    def field_from_sig(self, output_sig, dist,\
                                theta_out, phi_out, phase_shift):
        return output_sig \
            * self.directivity(theta_out, phi_out) \
                * self.wavelgth * np.exp(-1j * 2. * np.pi * dist / self.wavelgth) \
                    /4. / np.pi / dist
    
##############################################################################

class unit_cell:
    """A unit cell whose characteristics are defined from simulations"""
    
    def __init__(self, side_length=0.03, wavelgth=0.06):
        self.side_length = side_length
        self.area = np.square(side_length)
        self.wavelgth = wavelgth
        self.rad_pats = []
        self.scat_mats = []
        self.phase_states = []
        
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def set_rad_pat(self, input_rad_pat, output_rad_pat, scat_mat_file,
                   phase_state=0.):
        
        self.scat_mats.append(rf.Network(scat_mat_file))
        self.rad_pats.append((input_rad_pat, output_rad_pat))
        self.phase_states.append(phase_state)
        
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def plot_rad_pats(self):
        
        for rp, pt in zip(self.rad_pats, self.phase_states):
            
            # plot input radiation pattern
            fig, ax = rp[0].plot()
            ax.set_title("Input radiation pattern, phase state " + str(pt))
            
            # plot output radiation pattern
            fig, ax = rp[1].plot()
            ax.set_title("Output radiation pattern, phase state " + str(pt))
            
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def plot_scat_mats(self):
        
        for sm, pt in zip(self.scat_mats, self.phase_states):
            
            plt.figure()
            
            sm.plot_s_db(0,0)
            sm.plot_s_db(2,2)
            sm.plot_s_db(0,2)
            sm.plot_s_db(2,0)
            
            plt.title("Input scattering matrix, phase state " + str(pt))
            
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def idx_phase_state(self, phase):
        
        min_phase_diff = np.abs(np.array(self.phase_states) - 
                                phase % (2. * np.pi))
        return np.argmin(min_phase_diff)
            
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
            
    def input_sig(self, incoming_wave, theta, phi, phase):
        
        nb_phase = max(phase.shape)
        
        if nb_phase > 1:
            input_sigs = np.empty(nb_phase, dtype=np.complex128)
            
            for idx, (t, p, pt) in enumerate(zip(theta, phi, phase)):
                idx_pt = self.idx_phase_state(pt)
                input_sigs[idx] = incoming_wave * \
                    self.rad_pats[idx_pt][0].value(t, p)
        else:
            idx_pt = self.idx_phase_state(phase)
            input_sigs = incoming_wave * \
                self.rad_pats[idx_pt][0].value(theta, phi)
            
        return input_sigs
    
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    
    def output_sig(self, input_sig, phase):
        
        freq = c / self.wavelgth
        
        nb_sigs = max(input_sig.shape)
        
        if nb_sigs > 1:
        
            output_sigs = np.empty(nb_sigs, dtype=np.complex128)
            
            for idx, (sig, pt) in enumerate(zip(input_sig, phase)):
                
                idx_pt = self.idx_phase_state(pt)
                ntwk = self.scat_mats[idx_pt]
                
                idx_f = np.argmin(np.abs(ntwk.frequency.f - freq))
                
                # get the transmission coefficcient at this specific frequency
                transmission_coef = ntwk.s[idx_f][0,2]
                
                output_sigs[idx] = transmission_coef * sig
                
        else:
            
            idx_pt = self.idx_phase_state(phase)
            ntwk = self.scat_mats[idx_pt]
            
            idx_f = np.argmin(np.abs(ntwk.frequency.f - freq))
            
            # get the transmission coefficcient at this specific frequency
            transmission_coef = ntwk.s[idx_f][0,2]
            
            output_sigs = transmission_coef * input_sig
            
        
        return output_sigs
    
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    
    def field_from_sig(self, output_sig, dist,\
                                theta_out, phi_out, phase):
        
        nb_sigs = max(output_sig.shape)
        
        if nb_sigs > 1:
            
            output_fields = np.empty(nb_sigs, dtype=np.complex128)
            
            for idx, (b0, d, t, p, pt) in enumerate(zip(output_sig, dist, 
                        theta_out, phi_out, phase)):
                
                idx_pt = self.idx_phase_state(pt)
                
                output_fields[idx] = b0 * \
                    self.rad_pats[idx_pt][1].value(t, p) * \
                    self.wavelgth * np.exp(-1j * 2. * np.pi * d / \
                   self.wavelgth) /4. / np.pi / d
                        
        else:
            
            idx_pt = self.idx_phase_state(phase)
                
            output_fields = output_sig \
                * self.rad_pats[idx_pt][1].value(theta_out, phi_out) \
                * self.wavelgth * np.exp(-1j * 2. * np.pi * dist / \
                 self.wavelgth) /4. / np.pi / dist
        
        return output_fields
    
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def field(self, points):
                
        nb_points = len(points)
        rad_field = np.empty(nb_points, dtype=np.complex128)
        
        for idx, point in enumerate(points):
        
            dp = np.sqrt(np.square(point.x) + \
                         np.square(point.y) +\
                         np.square(point.z))
            theta_out = np.acos(point.z / dp)
            phi_out = np.acos((point.y) / \
                           np.sqrt(np.square(point.x) \
                           + np.square(point.y)))
            
            # FIXME: add the possibility to set the amplitude and the phase
            rad_field[idx] = self.field_from_sig(
                    np.array([1.]), dp, theta_out, phi_out, 0.)
        
        return [rad_field]
    
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def plot_field(self, plane="xz", side = -1, \
                   corner_pt = point(1.e9,0.,0.), nb_side_pts=50,
                   plot_grid=False):
        
        fc = field_calculator(self)
        fc.field_in_plane(plane, side, corner_pt, nb_side_pts, plot_grid)
        
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def field_labels(self):
        return ["Field radiated by a unit cell"]

##############################################################################
    
class simplified_horn_source:
    """A simple model for a horn source"""
    
    def __init__(self, order=5, wavelgth=0.06):
        self.order = order
        self.wavelgth = wavelgth
        
    def directivity(self, theta, phi):
        return 2.*(self.order + 1) * np.pow(np.cos(theta), self.order)
    
    def field(self, theta, phi, power, dist):
        return np.sqrt(power) * self.wavelgth * \
            np.exp(-1j * 2. * np.pi * dist /self.wavelgth) * \
                self.directivity(theta, phi) / 4. / np.pi / dist
                
##############################################################################

class transmit_array:
    """A simple transmit array model"""
    
    def __init__(self, n_cell_x, n_cell_y, unit_cell : 'simple_unit_cell', \
                 source: 'simplified_horn_source', dist_src=0.5, \
                 wavelgth=0.06):
        
        self.n_cell_x = n_cell_x
        self.n_cell_y = n_cell_y
        self.nb_cell = n_cell_x * n_cell_y
        self.unit_cell = unit_cell
        self.phase_mask = np.zeros(self.nb_cell)
        self.source = source
        self.dist_src = dist_src
        self.wavelgth = wavelgth
        self.input_sig = np.zeros(self.nb_cell, dtype=np.complex128)
        self.output_sig = np.zeros(self.nb_cell, dtype=np.complex128)

        
        # generate the coordinates of the centers of the cells
        x_min = -self.unit_cell.side_length * (self.n_cell_x - 1)/2. 
        x_max = -x_min
        self.x = np.linspace(x_min, x_max, self.n_cell_x)
        
        y_min = -self.unit_cell.side_length * (self.n_cell_y - 1)/2. 
        y_max = -y_min
        self.y = np.linspace(y_min, y_max, self.n_cell_y)
        
        # Generate lists containing the coordinates ordered by pairs
        self.x_ordered = np.empty(self.nb_cell)
        self.y_ordered = np.empty(self.nb_cell)
        idx = 0
        for x in self.x:
            for y in self.y:
                self.x_ordered[idx] = x
                self.y_ordered[idx] = y
                idx = idx + 1
                
        self.input_signals()
        self.output_signals()
        
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        
    def set_phase_mask(self, value):
        self.phase_mask.fill(value)
        
        # update input and ouput signals 
        self.input_signals()
        self.output_signals()
        
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        
    def set_random_phase_mask(self):
        self.phase_mask = np.pi * np.random.randint(0, 2, self.nb_cell)
        
        # update input and ouput signals 
        self.input_signals()
        self.output_signals()
        
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        
    def set_phase_mask_devided_in_half(self):
        idx = 0
        for idx_x in range(0, self.n_cell_x):
            for idx_y in range(0, self.n_cell_y):
                if idx_x < self.n_cell_x / 2:
                    self.phase_mask[idx] = np.pi
                idx = idx + 1
                
        # update input and ouput signals 
        self.input_signals()
        self.output_signals()
                
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def set_phase_mask_beam(self, theta_beam, phi_beam, \
                             quantize=True ):
        for idx in range(0, self.nb_cell):
            self.phase_mask[idx] = \
            -2.* np.pi * np.sin(theta_beam) * ( \
            np.cos(phi_beam) * self.x_ordered[idx] \
            + np.sin(phi_beam) * self.y_ordered[idx] \
                ) / self.wavelgth 
            if quantize:
                self.phase_mask[idx]  = \
                    round((self.phase_mask[idx] % np.pi) / np.pi) * np.pi
                    
        # update input and ouput signals 
        self.input_signals()
        self.output_signals()
                    
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def set_phase_mask_focal_point(self, focal_point, quantize=True):
        for idx in range(0, self.nb_cell):
            self.phase_mask[idx] = \
                (-2. * np.pi * (np.sqrt(np.square(focal_point.z) + \
             np.square(self.x_ordered[idx] - focal_point.x) + \
             np.square(self.y_ordered[idx] - focal_point.y)) \
                    - focal_point.z) / self.wavelgth)  % (2. * np.pi)
            if quantize:
                self.phase_mask[idx]  = \
                    round(((self.phase_mask[idx]) % np.pi) / np.pi) * np.pi
                    
        # update input and ouput signals 
        self.input_signals()
        self.output_signals()
                
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        
    def get_phase_mask(self):
        return self.phase_mask.reshape((self.n_cell_x, self.n_cell_y))

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    
    def output_sigs(self):
        return self.output_sig.reshape(self.n_cell_x, self.n_cell_y)
    
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def input_coords(self):
        ds = np.sqrt(np.square(self.x_ordered) + \
                          np.square(self.y_ordered)\
                          + np.square(self.dist_src))
        theta_in = np.acos(self.dist_src / ds)
        phi_in = np.acos(self.y_ordered / np.sqrt(np.square(self.x_ordered) + \
                                    + np.square(self.y_ordered)))
            
        return ds, theta_in, phi_in

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def input_signals(self, power=1.):
        
        ds, theta_in, phi_in = self.input_coords()

        input_signals = self.source.field(
            theta_in, phi_in, power, ds
            )
        
        self.input_sig = input_signals
        
        input_signals = input_signals.reshape((self.n_cell_x, self.n_cell_y))
        
        return input_signals
        

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        
    def output_signals(self, power=1.):
            
        ds, theta_in, phi_in = self.input_coords()
        
        output_sig = self.unit_cell.output_sig(
            self.input_sig, self.phase_mask)
                
        self.output_sig = output_sig
                
        output_sig = output_sig.reshape((self.n_cell_x, self.n_cell_y))
        
        return output_sig
    
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    
    def field(self, points):
                
        nb_points = len(points)
        rad_field = np.empty(nb_points, dtype=np.complex128)
        
        for idx, point in enumerate(points):
        
            dp = np.sqrt(np.square(self.x_ordered - point.x) + \
                         np.square(self.y_ordered - point.y) +\
                         np.square(point.z))
            theta_out = np.acos(point.z / dp)
            phi_out = np.acos((point.y - self.y_ordered) / \
                           np.sqrt(np.square(point.x - self.x_ordered) \
                           + np.square(point.y - self.y_ordered)))
                    
            field_from_cells = self.unit_cell.field_from_sig(
                    self.output_sig, dp,
                    theta_out, phi_out, self.phase_mask)
        
            rad_field[idx] = field_from_cells.sum()
        
        return [rad_field]
    
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def field_labels(self):
        return ["Radiated field from transmit array"]
    
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def plot_field(self, plane="xz", side = -1, \
                   corner_pt = point(1.e9,0.,0.), nb_side_pts=50,
                   plot_grid=False):
        
        fc = field_calculator(self)
        fc.field_in_plane(plane, side, corner_pt, nb_side_pts, plot_grid)
                
##############################################################################

sourceType = Union[transmit_array, simplified_horn_source]

class desordered_medium:
    """A simple disordered model"""
    
    def __init__(self, source: sourceType, scat_pos=None, wavelgth=0.06):
        
        if scat_pos == None:
            nb_scat = 25
        else:
            nb_scat = len(scat_pos)
            
        self.initialize(nb_scat)
        self.wavelgth = wavelgth
        self.scat_pos = scat_pos
        self.source = source
        self.T = np.zeros(1, dtype=np.complex128)

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def initialize(self, nb_scat):
        self.nb_scat = nb_scat
        self.polarizability = 1.*np.ones(nb_scat)
        self.Gin = np.empty((1, nb_scat), dtype=np.complex128)
        self.Gout = np.empty((nb_scat, 1), dtype=np.complex128)
        self.Gdd = np.zeros((nb_scat, nb_scat), dtype=np.complex128)
        
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def generate_random_scatterers(self, nb_scat=None, 
            bounding_box = [-0.2, 0.2, -0.2, 0.2, 0.05, 0.25]):
        
        if nb_scat == None:
            nb_scat = self.nb_scat
        else:
            self.initialize(nb_scat)
            
        self.scat_pos = [None] * nb_scat
        
        delta_x = bounding_box[1] - bounding_box[0]
        delta_y = bounding_box[3] - bounding_box[2]
        delta_z = bounding_box[5] - bounding_box[4]
        random_coordinates = np.random.rand(nb_scat, 3)
        
        for idx in range(nb_scat):
            self.scat_pos[idx] = point(
                random_coordinates[idx, 0] * delta_x + bounding_box[0],
                random_coordinates[idx, 1] * delta_y + bounding_box[2],
                random_coordinates[idx, 2] * delta_z + bounding_box[4],
                )
            
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def save_scat_pos(self, file_name = "scatterers_position.csv"):
        
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            for scat in self.scat_pos:
                writer.writerow([scat.x, scat.y, scat.z])    
                
        print(f"Scatterers coordinates saved to {file_name}")

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def create_scat_from_csv(self, file_name):
        
        scat_pos = []
        
        with open(file_name, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                scat_pos.append(
                    point(float(row[0]), float(row[1]), float(row[2]))
                    )
        self.scat_pos = scat_pos
        nb_scat = len(scat_pos)
        self.initialize(nb_scat)

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def plot_scatterers(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter([point.x for point in self.scat_pos],
            [point.y for point in self.scat_pos], 
            [point.z for point in self.scat_pos])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
        
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        
    def field(self, obs_pts):
        
        # wavenumber
        k = 2.*np.pi / self.wavelgth
        two_sqrt_pi = 2.*np.sqrt(np.pi)
        
        nb_obs_pts = len(obs_pts)
        self.Gout = np.empty((self.nb_scat, nb_obs_pts), dtype=np.complex128)
        
        # compute direct field
        Gdir = np.empty((1, nb_obs_pts), dtype=np.complex128)
        Gdir[0,:] = self.source.field(obs_pts)[0]
        
        # compute input and output Green functions
        self.Gin[0,:] = self.source.field(self.scat_pos)[0]
        for idx, scat in enumerate(self.scat_pos):
            # self.Gin[0,idx] = self.source.field(scat)
            
            for idx2, obs in enumerate(obs_pts):
                d_scat_obs = scat.distance_to(obs)
                self.Gout[idx, idx2] = self.wavelgth * np.exp(1j * k * d_scat_obs) \
                / d_scat_obs / two_sqrt_pi
                
        # compute between scatterers coupling Green functions
        for i in range(0, self.nb_scat):
            for j in range(0, self.nb_scat):
                if i != j:
                    d_scat = self.scat_pos[i].distance_to(self.scat_pos[j])
                    self.Gdd[i,j] = self.wavelgth * np.exp(1j * k * d_scat) / d_scat \
                        / two_sqrt_pi
                else:
                    self.Gdd[i,j] = 0.
                
        # compute transmission matrix or coefficient     
        self.T = np.matmul(np.matmul(self.Gin, \
                      np.linalg.solve((np.eye(self.nb_scat) - \
                       np.matmul(self.Gdd, np.diag(self.polarizability))\
                       ).T, \
                        np.diag(self.polarizability).T).T),\
                      self.Gout)
            
        return [Gdir, self.T, Gdir + self.T]
        
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def plot_field(self, plane="xz", side = -1, \
                   corner_pt = point(1.e9,0.,0.), nb_side_pts=50,
                   plot_grid=False):
        
        fc = field_calculator(self)
        fc.field_in_plane(plane, side, corner_pt, nb_side_pts, plot_grid)
        
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    
    def field_labels(self):
        return ["Direct field", "Scattered field", "Total field"]

##############################################################################

class field_calculator:
    
    """Computes field generated by diverse sources"""
    
    def __init__(self, source):
        self.source = source
        
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def field_in_plane(self, plane="xz", side = -1, \
                   corner_pt = point(1.e9,0.,0.), nb_side_pts=50,
                   plot_grid=False):
        
        # set the axis labels corresponding to the chosen plane
        if plane == "xz":
            xlabel = "x (m)"
            ylabel = "z (m)"
            
        elif plane == "yz":
            xlabel = "y (m)"
            ylabel = "z (m)"
            
        elif plane == "xy":
            xlabel = "x (m)"
            ylabel = "y (m)"
        
        # set the grid parameters
        if side == -1:
            side = 10 * self.source.wavelgth
        if corner_pt.x == 1.e9:
            if plane == "xz":
                corner_pt = point(side/2., 0., 0.)
                
            elif plane == "yz":
                corner_pt = point(0., side/2., 0.)
                
            elif plane == "xy":
                corner_pt = point(side/2, side/2., side/2)
            
        g = point_grid_2d(plane, side, corner_pt, nb_side_pts)
        
        if plot_grid:
            g.plot()
            
        # compute field on the grid
        fields = self.source.field(g.points)
        nb_fields = len(fields)
        field_labels = self.source.field_labels()
        
        # reshape the field data so that they correspond to the grid and get
        # the overall maximal value
        max_fields = []
        for idx, f in enumerate(fields):
            fields[idx] = np.abs(f.reshape((nb_side_pts, nb_side_pts)).T)
            inf_mask = np.isinf(fields[idx])
            fields[idx] = np.where(inf_mask, 0., fields[idx])
            max_fields.append(fields[idx].max())
        max_value = np.max(max_fields)
        
        # Prevent garbage collection
        global slider, fig
        
        fig, axes = plt.subplots(1, nb_fields)
        
        # if there is only one plot, make the axes iterrable so that the loops
        # can work
        if nb_fields == 1:
            axes = [axes]
        
        fig.suptitle("Plane " + plane)
        
        images = []
        for f, ax, label in zip(fields, axes, field_labels):
            images.append(
                ax.imshow(np.abs(f), vmax=max_value, 
                          extent=g.bounding_box, origin='lower')
                )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(label)

        cbar = fig.colorbar(images[0], ax=axes, 
                           location='right', 
                           pad=0.02, 
                           shrink=0.5)
        cbar.set_label('|E|')
        
        # create a slider to adjust the maximal color value 
        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])  # [left, bottom, width, height]
        slider = Slider(
            ax=ax_slider,
            label='Max Color Value',
            valmin=0.,
            valmax=max_value,
            valinit=max_value,
            valstep=0.01
        )
        
        # Function to update vmax
        def update(val):
            for img in images:
                img.set_clim(vmin=0, vmax=slider.val)
            cbar.update_normal(images[0])
            fig.canvas.draw_idle()
            
        slider.on_changed(update)
        
        plt.show()
        