"""
A python library to generate FDI attack
Inherited from the state estimation tool run_AC_SE
Author: W XU 
"""

from run_AC_SE import SE
from mea_idx import define_mea_idx_noise
from se_config import se_config
from pypower.api import *
import numpy as np
import random
import matplotlib.pyplot as plt
import warnings

class FDI(SE):
    # Inherit from the Class SE
    def __init__(self, case, noise_sigma, idx, fpr):
        super().__init__(case, noise_sigma, idx, fpr)
    
    def gen_att(self, v_est, att_spec):
        """
        Generate a single FDI attack based on the Given state
        Only the non-reference bus can be attacked
        v_est: estimated state
        att_spec: a dictionary contains the attack information
        'ang_posi': 
        'mag_posi':
        'ang_str':
        'mag_str':
        """
        
        ang_posi = att_spec['ang_posi']
        ang_str = att_spec['ang_str']
        mag_posi = att_spec['mag_posi']
        mag_str = att_spec['mag_str']
        
        # Raise an error if the reference bus is attacked
        if self.ref_index[0] in ang_posi or self.ref_index[0] in mag_posi:
            warnings.warn('The reference bus is attacked. Consider reconfiguring the attack positions.')
        
        vang_est = np.angle(v_est)
        vmag_est = np.abs(v_est)
        
        vang_att = vang_est.copy()
        vmag_att = vmag_est.copy()
        
        vang_att[ang_posi] = vang_est[ang_posi] * (1 + ang_str)
        vmag_att[mag_posi] = vmag_est[mag_posi] * (1 + mag_str)
        
        v_att = vmag_att * np.exp(1j*vang_att)
        
        return v_att        

"""
An example
"""
if __name__ == "__main__":
    case = case14()
    # Define measurement idx
    mea_idx, no_mea, noise_sigma = define_mea_idx_noise(case, 'full')
    # Instance the Class FDI
    fdi = FDI(case, noise_sigma=noise_sigma, idx=mea_idx)
    # run opf
    result = fdi.run_opf()
    # Construct the measurement
    z, z_noise, vang_ref, vmag_ref = fdi.construct_mea(result) # Get the measurement
    # Run AC-SE
    se_config['verbose'] = 1
    v_est = fdi.ac_se_pypower(z_noise, vang_ref, vmag_ref, config = se_config)
    
    # Attack specification
    att_spec = {}
    att_spec['ang_posi'] = random.sample(range(fdi.no_bus), 1)
    att_spec['ang_str'] = 0.2
    att_spec['mag_posi'] = []
    att_spec['mag_str'] = 0
    # Generate FDI attack
    v_att = fdi.gen_att(v_est, att_spec)
    