import numpy as np
from typing import Union
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

#----------------------------------------------------------------------------#

class point:
    
    """A cartesian point"""
    
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def distance_to(self, other_point):
        return np.sqrt((self.x - other_point.x)**2 + \
                       (self.y - other_point.y)**2 + \
                       (self.z - other_point.z)**2)
            
#----------------------------------------------------------------------------#

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

#----------------------------------------------------------------------------#

class simple_unit_cell:
    """A simple model for a unit cell"""
    
    def __init__(self, side_length):
        self.side_length = side_length
        self.area = np.square(side_length)
        
    def directivity(self, theta, phi, wavelgth):
        return 4.*np.pi * self.area * np.square(np.cos(theta)) \
    / np.square(wavelgth)
    
    def input_sig(self, incoming_wave, theta, phi, wavelgth):
        return incoming_wave * self.directivity(theta, phi, wavelgth)
    
    def output_sig(self, incoming_wave, theta, phi, wavelgth, phase_shift):
        return self.input_sig(incoming_wave, theta, phi, wavelgth) * \
            np.exp(-1j * phase_shift)
            
    def field(self, incoming_wave, theta, phi, wavelgth, phase_shift, \
                       dist, theta_out, phi_out):
        return self.output_sig(incoming_wave, theta, phi, wavelgth, phase_shift) \
            * self.directivity(theta_out, phi_out, wavelgth) \
                * wavelgth * np.exp(-1j * 2. * np.pi * dist / wavelgth) \
                    /4. / np.pi / dist
                    
    def field_from_sig(self, output_sig, wavelgth, dist,\
                                theta_out, phi_out):
        return output_sig \
            * self.directivity(theta_out, phi_out, wavelgth) \
                * wavelgth * np.exp(-1j * 2. * np.pi * dist / wavelgth) \
                    /4. / np.pi / dist
    
#----------------------------------------------------------------------------#
    
class simplified_horn_source:
    """A simple model for a horn source"""
    
    def __init__(self, order):
        self.order = order
        
    def directivity(self, theta, phi, wavelgth):
        return 2.*(self.order + 1) * np.pow(np.cos(theta), self.order)
    
    def field(self, theta, phi, wavelgth, power, dist):
        return np.sqrt(power) * wavelgth * \
            np.exp(-1j * 2. * np.pi * dist /wavelgth) * \
                self.directivity(theta, phi, wavelgth) / 4. / np.pi / dist
                
#----------------------------------------------------------------------------#

class transmit_array:
    """A simple transmit array model"""
    
    def __init__(self, n_cell_x, n_cell_y, unit_cell : 'simple_unit_cell', \
                 source: 'simplified_horn_source', dist_src):
        self.n_cell_x = n_cell_x
        self.n_cell_y = n_cell_y
        self.nb_cell = n_cell_x * n_cell_y
        self.unit_cell = unit_cell
        self.phase_mask = np.zeros(self.nb_cell)
        self.source = source
        self.dist_src = dist_src
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
        
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        
    def set_phase_mask(self, value):
        self.phase_mask.fill(value)
        
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        
    def set_random_phase_mask(self):
        self.phase_mask = np.pi * np.random.randint(0, 2, self.nb_cell)
        
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        
    def set_phase_mask_devided_in_half(self):
        idx = 0
        for idx_x in range(0, self.n_cell_x):
            for idx_y in range(0, self.n_cell_y):
                if idx_x < self.n_cell_x / 2:
                    self.phase_mask[idx] = np.pi
                idx = idx + 1
                
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def set_phase_mask_beam(self, theta_beam, phi_beam, wavelgth, \
                             quantize=True ):
        for idx in range(0, self.nb_cell):
            self.phase_mask[idx] = \
            -2.* np.pi * np.sin(theta_beam) * ( \
            np.cos(phi_beam) * self.x_ordered[idx] \
            + np.sin(phi_beam) * self.y_ordered[idx] \
                ) / wavelgth 
            if quantize:
                self.phase_mask[idx]  = \
                    round((self.phase_mask[idx] % np.pi) / np.pi) * np.pi
                    
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def set_phase_mask_focal_point(self, focal_point, \
                                    wavelgth, quantize=True):
        for idx in range(0, self.nb_cell):
            self.phase_mask[idx] = \
                (-2. * np.pi * (np.sqrt(np.square(focal_point.z) + \
             np.square(self.x_ordered[idx] - focal_point.x) + \
             np.square(self.y_ordered[idx] - focal_point.y)) \
                    - focal_point.z) / wavelgth)  % (2. * np.pi)
            if quantize:
                self.phase_mask[idx]  = \
                    round(((self.phase_mask[idx]) % np.pi) / np.pi) * np.pi
                
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

    def inout_signals(self, wavelgth, power):
        
        ds, theta_in, phi_in = self.input_coords()

        input_signals = self.source.field(
            theta_in, phi_in, wavelgth, power, ds
            )
        
        self.input_sig = input_signals
        
        input_signals = input_signals.reshape((self.n_cell_x, self.n_cell_y))
        
        return input_signals
        

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        
    def output_signals(self, wavelgth, power):
            
        ds, theta_in, phi_in = self.input_coords()
        
        output_sig = self.unit_cell.output_sig(
            self.input_sig, theta_in, phi_in, wavelgth, 
            self.phase_mask)
                
        self.output_sig = output_sig
                
        output_sig = output_sig.reshape((self.n_cell_x, self.n_cell_y))
        
        return output_sig
    
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    
    def field(self, point, wavelgth):
                
        dp = np.sqrt(np.square(self.x_ordered - point.x) + \
                     np.square(self.y_ordered - point.y) +\
                     np.square(point.z))
        theta_out = np.acos(point.z / dp)
        phi_out = np.acos((point.y - self.y_ordered) / \
                       np.sqrt(np.square(point.x - self.x_ordered) \
                       + np.square(point.y - self.y_ordered)))
                    
        rad_field = self.unit_cell.field_from_sig(
                self.output_sig, wavelgth, dp,
                theta_out, phi_out)
        
        rad_field = rad_field.sum()
        
        return rad_field
                
#----------------------------------------------------------------------------#

sourceType = Union[transmit_array, simplified_horn_source]

class desordered_medium:
    """A simple disordered model"""
    
    def __init__(self, scat_pos: list[point], source: sourceType):
        nb_scat = len(scat_pos)
        self.nb_scat = nb_scat
        self.scat_pos = scat_pos
        self.source = source
        self.polarizability = 0.1*np.ones(nb_scat)
        self.Gin = np.empty((1, nb_scat), dtype=np.complex128)
        self.Gout = np.empty((nb_scat, 1), dtype=np.complex128)
        self.Gdd = np.zeros((nb_scat, nb_scat), dtype=np.complex128)
        self.T = np.zeros(1, dtype=np.complex128)
        
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
        
    def field(self, obs_pts, wavelgth):
        
        # wavenumber
        k = 2.*np.pi / wavelgth
        
        nb_obs_pts = len(obs_pts)
        self.Gout = np.empty((self.nb_scat, nb_obs_pts), dtype=np.complex128)
        
        # compute direct field
        Gdir = np.empty((1, nb_obs_pts), dtype=np.complex128)
        for idx, obs in enumerate(obs_pts):
            Gdir[0,idx] = self.source.field(obs, wavelgth)
        
        # compute input and output Green functions
        for idx, scat in enumerate(self.scat_pos):
            self.Gin[0,idx] = self.source.field(\
                scat, wavelgth)
            
            for idx2, obs in enumerate(obs_pts):
                d_scat_obs = scat.distance_to(obs)
                self.Gout[idx, idx2] = np.exp(1j * k * d_scat_obs) / d_scat_obs
                
        # compute between scatterers coupling Green functions
        for i in range(0, self.nb_scat):
            for j in range(0, self.nb_scat):
                if i != j:
                    d_scat = self.scat_pos[i].distance_to(self.scat_pos[j])
                    self.Gdd[i,j] = np.exp(1j * k * d_scat) / d_scat
                else:
                    self.Gdd[i,j] = 0.
                
        # compute transmission matrix or coefficient     
        self.T = np.matmul(np.matmul(self.Gin, \
                      np.linalg.solve((np.eye(self.nb_scat) - \
                       np.matmul(self.Gdd, np.diag(self.polarizability))\
                       ).T, \
                        np.diag(self.polarizability).T).T),\
                      self.Gout)
            
        return Gdir, self.T
        
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def plot_field(self, wavelgth, plane="xz", side = -1, \
                   corner_pt = point(0.,0.,.0), nb_side_pts=50,
                   plot_grid=False):
        
        if side == -1:
            side = 10 * wavelgth
            if plane == "xz":
                corner_pt = point(side/2., 0., 0.)
                xlabel = "x (m)"
                ylabel = "z (m)"
                
            elif plane == "yz":
                corner_pt = point(0., side/2., 0.)
                xlabel = "y (m)"
                ylabel = "z (m)"
                
            elif plane == "xy":
                corner_pt = point(side/2, side/2., side/2)
                xlabel = "x (m)"
                ylabel = "y (m)"
            
        g = point_grid_2d(plane, side, corner_pt, nb_side_pts)
        
        if plot_grid:
            g.plot()
        
        dir_field, scat_field = self.field(g.points, wavelgth)
        
        dir_field = dir_field.reshape((nb_side_pts, nb_side_pts)).T
        scat_field = scat_field.reshape((nb_side_pts, nb_side_pts)).T
        tot_field = dir_field + scat_field
        
        # Prevent garbage collection
        global slider, fig
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        
        max_value = np.max([np.abs(dir_field).max(), \
                           np.abs(scat_field).max(), \
                               np.abs(tot_field).max()])
            
        im1 = ax1.imshow(np.abs(dir_field), vmax=max_value, 
                        extent=g.bounding_box, origin='lower')
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.set_title("Direct field")

        im2 = ax2.imshow(np.abs(scat_field), vmax=max_value, \
                   extent=g.bounding_box, origin='lower')
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)
        ax2.set_title("Scattered field")
        
        im3 = ax3.imshow(np.abs(tot_field), vmax=max_value, \
                   extent=g.bounding_box, origin='lower')
        ax3.set_xlabel(xlabel)
        ax3.set_ylabel(ylabel)
        ax3.set_title("Total field")

        cbar = fig.colorbar(im1, ax=[ax1, ax2, ax3], 
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
            valstep=0.05
        )
        
        # Function to update vmax
        def update(val):
            im1.set_clim(vmin=0, vmax=slider.val)
            im2.set_clim(vmin=0, vmax=slider.val)
            im3.set_clim(vmin=0, vmax=slider.val)
            cbar.update_normal(im1)
            fig.canvas.draw_idle()
            
        slider.on_changed(update)
        
        plt.show()
        