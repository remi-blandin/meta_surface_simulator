import numpy as np
from typing import Union

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

    def set_phase_mask_focal_point(self, x_focal, y_focal, z_focal, \
                                    wavelgth, quantize=True):
        for idx in range(0, self.nb_cell):
            self.phase_mask[idx] = \
                (-2. * np.pi * (np.sqrt(np.square(z_focal) + \
             np.square(self.x_ordered[idx] - x_focal) + \
             np.square(self.y_ordered[idx] - y_focal)) \
                    - z_focal) / wavelgth)  % (2. * np.pi)
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
    
    def field(self, x_rad, y_rad, z_rad, wavelgth):
                
        dp = np.sqrt(np.square(self.x_ordered - x_rad) + \
                     np.square(self.y_ordered - y_rad) +\
                     np.square(z_rad))
        theta_out = np.acos(z_rad / dp)
        phi_out = np.acos((y_rad - self.y_ordered) / \
                       np.sqrt(np.square(x_rad - self.x_ordered) \
                       + np.square(y_rad - self.y_ordered)))
                    
        rad_field = self.unit_cell.field_from_sig(
                self.output_sig, wavelgth, dp,
                theta_out, phi_out)
        
        rad_field = rad_field.sum()
        
        return rad_field
                
#----------------------------------------------------------------------------#

sourceType = Union[transmit_array, simplified_horn_source]

class desordered_medium:
    """A simple disordered model"""
    
    def __init__(self, scat_pos: np.ndarray, source: sourceType):
        nb_scat = scat_pos.shape[0]
        self.nb_scat = nb_scat
        self.source = source
        self.scat_pos = scat_pos
        self.polarizability = np.ones(nb_scat)
        self.Gin = np.empty((1, nb_scat), dtype=np.complex128)
        self.Gout = np.empty((nb_scat, 1), dtype=np.complex128)
        self.Gdd = np.zeros((nb_scat, nb_scat), dtype=np.complex128)
        self.T = np.zeros(1, dtype=np.complex128)
        
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        
    def field(self, obs_pt, wavelgth):
        
        # wavenumber
        k = 2.*np.pi / wavelgth
        
        # compute input and output Green functions
        for idx, scat in enumerate(self.scat_pos):
            self.Gin[0,idx] = self.source.field(\
                scat[0], scat[1], scat[2], wavelgth)
            
            d_scat_obs = np.sqrt(np.square(self.scat_pos[idx,:] - obs_pt).sum())
            self.Gout[idx] = np.exp(1j * k * d_scat_obs) / d_scat_obs
                
        # compute between scatterers coupling Green functions
        for i in range(0, self.nb_scat):
            for j in range(0, self.nb_scat):
                if i != j:
                    d_scat = np.sqrt(np.square(self.scat_pos[i,:] - \
                                      self.scat_pos[j,:]).sum())
                    self.Gdd[i,j] = np.exp(1j * k * d_scat) / d_scat
                else:
                    self.Gdd[i,j] = 0.
                
        # compute transmission matrix or coefficient              
        self.T = np.matmul(np.matmul(self.Gin, \
                      np.linalg.solve((np.eye(self.nb_scat) - self.Gdd).T, \
                        np.diag(self.polarizability).T).T),\
                      self.Gout)
            
        return self.T
        
            
        