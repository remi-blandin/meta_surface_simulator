import numpy as np

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
            
    def radiated_field(self, incoming_wave, theta, phi, wavelgth, phase_shift, \
                       dist, theta_out, phi_out):
        return self.output_sig(incoming_wave, theta, phi, wavelgth, phase_shift) \
            * self.directivity(theta_out, phi_out, wavelgth) \
                * wavelgth * np.exp(-1j * 2. * np.pi * dist / wavelgth) \
                    /4. / np.pi / dist
                    
    def radiated_field_from_sig(self, output_sig, wavelgth, dist,\
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
    
    def radiated_field(self, theta, phi, wavelgth, power, dist):
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
        self.phase_shift = np.zeros(self.nb_cell)
        self.source = source
        self.dist_src = dist_src
        
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
        
        self.output_sig = np.zeros(self.nb_cell, dtype=np.complex128)
        
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        
    def set_phase_shift(self, value):
        self.phase_shift.fill(value)
        
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        
    def set_random_phase_shift(self):
        self.phase_shift = np.pi * np.random.randint(0, 2, self.nb_cell)
        
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        
    def set_specific_phase_shift(self):
        idx = 0
        for idx_x in range(0, self.n_cell_x):
            for idx_y in range(0, self.n_cell_y):
                if idx_x < self.n_cell_x / 2:
                    self.phase_shift[idx] = np.pi
                idx = idx + 1
                
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        
    def phase_shifts(self):
        return self.phase_shift.reshape((self.n_cell_x, self.n_cell_y))

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    
    def output_sigs(self):
        return self.output_sig.reshape(self.n_cell_x, self.n_cell_y)
    
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        
    def output_signals(self, wavelgth, power):
        
        # generate coords
        coords = np.empty((self.nb_cell, 2))
        ds = np.empty(self.nb_cell)
        theta_in = np.empty(self.nb_cell)
        phi_in = np.empty(self.nb_cell)
        incoming_wave = np.empty(self.nb_cell, dtype=np.complex128)
        output_sig = np.empty(self.nb_cell, dtype=np.complex128)
        
        idx = 0
        for x in self.x:
            for y in self.y:
                coords[idx, 0] = x
                coords[idx, 1] = y
                ds[idx] = np.sqrt(x * x + y * y + np.square(self.dist_src))
                theta_in[idx] = np.acos(self.dist_src / ds[idx])
                phi_in[idx] = np.acos(y / np.sqrt(x * x + y * y))
                
                incoming_wave[idx] = self.source.radiated_field(
                    theta_in[idx], phi_in[idx], wavelgth, power, ds[idx]
                    )
                
                output_sig[idx] = self.unit_cell.output_sig(
                    incoming_wave[idx], theta_in[idx], phi_in[idx], wavelgth, 
                    self.phase_shift[idx])
                
                idx = idx + 1
                
        self.output_sig = output_sig
                
        incoming_wave = incoming_wave.reshape((self.n_cell_x, self.n_cell_y))
        output_sig = output_sig.reshape((self.n_cell_x, self.n_cell_y))
        
                
        return coords, ds, theta_in, phi_in, incoming_wave, output_sig
    
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    
    def radiated_field(self, x_rad, y_rad, z_rad, wavelgth):
                
        dp = np.sqrt(np.square(self.x_ordered - x_rad) + \
                     np.square(self.y_ordered - y_rad) +\
                     np.square(z_rad))
        theta_out = np.acos(z_rad / dp)
        phi_out = np.acos((y_rad - self.y_ordered) / \
                       np.sqrt(np.square(x_rad - self.x_ordered) \
                       + np.square(y_rad - self.y_ordered)))
                    
        rad_field = self.unit_cell.radiated_field_from_sig(
                self.output_sig, wavelgth, dp,
                theta_out, phi_out)
        
        rad_field = rad_field.sum()
        
        return rad_field
                
                
        