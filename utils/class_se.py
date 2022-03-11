"""
A simple realization of MATPOWER AC state estimation using PyPower
which can be used alongside with Python based power system steady state analysis
Sources:
    1. MATPOWER: https://github.com/MATPOWER/matpower
    2. MATPOWER SE: https://github.com/MATPOWER/mx-se
    3. PYPOWER: https://github.com/rwl/PYPOWER

The original repo for this code can be found in my GitHub: https://github.com/xuwkk/steady-state-power-system

Some code is removed if not related to the this task

Author: W XU
"""

import numpy as np
from pypower.api import *
from pypower.idx_bus import *
from pypower.idx_brch import *
from pypower.idx_gen import *
import scipy
from configs.config_mea_idx import define_mea_idx_noise
from configs.config import se_config, opt
import copy
from scipy.stats.distributions import chi2

class SE:
    def __init__(self, case, noise_sigma, idx, fpr):
        
        """
        case: the instances case by calling from pypower api, e.g. case = case14()
        noise_sigma = A 1D array contains the noise std of the measurement, please refer to the format in mea_idx
        tol: the tolerance on the minimum jacobian matrix norm changes before considered as convegent
        max_it: maximum iteration
        verbose: description settings on the output
        measurement: the measurement given by each measurement type
        idx: the measurement index given by each measurement type, please refer to the format in mea_idx
        
        measurement type (the order matters)
        1. z = [pf, pt, pg, vang, qf, qt, qg, vmag]  (in MATPOWER-SE)
        2. z = [pf, pt, pi, vang, qf, qt, qi, vmag]  (in our current settings)
        In the future version, the selection on 1 and 2 should be added
        """
        
        """
        Define the grid parameter
        """
        
        # Case
        self.case = case
        case_int = ext2int(case)                                                         # Covert the start-1 to start-0 in python                     
        self.case_int = case_int
        
        # Numbers
        self.no_bus = len(case['bus'])
        self.no_brh = len(case['branch'])
        self.no_gen = len(case['gen'])
        self.no_mea = 0
        for key in idx.keys():
            self.no_mea = self.no_mea + len(idx[key])
        
        # Determine the bus type
        self.ref_index, pv_index, pq_index = bustypes(case_int['bus'], case_int['gen'])  # reference bus (slack bus), pv bus, and pq (load bus)
        self.non_ref_index = list(pq_index) + list(pv_index)                             # non reference bus
        self.non_ref_index.sort()
        
        """
        Define matrices related to measurement noise
        """
        self.noise_sigma = noise_sigma                     # std
        self.R = np.diag(noise_sigma**2)                   # R
        self.R_inv = np.diag(1/self.noise_sigma**2)        # R^-1 
        
        DoF = self.no_mea - 2*(self.no_bus - 1)            # Degree of Freedom
        self.bdd_threshold = chi2.ppf(1-fpr, df = DoF)     # BDD detection threshold
        
        """
        Incidence Matrix
        """
        
        # Branch Incidence Matrix
        f_bus = case_int['branch'][:, 0].astype('int')        # list of "from" buses
        t_bus = case_int['branch'][:, 1].astype('int')        # list of "to" buses
        self.Cf = np.zeros((self.no_brh,self.no_bus))         # "from" bus incidence matrix
        self.Ct = np.zeros((self.no_brh,self.no_bus))         # "to" bus incidence matrix
        for i in range(self.no_brh):
            self.Cf[i,f_bus[i]] = 1
            self.Ct[i,t_bus[i]] = 1
        
        # Generator Incidence Matrix
        self.Cg = np.zeros((self.no_bus,self.no_gen))
        for i in range(self.no_gen):
            self.Cg[int(case_int['gen'][i,0]),i] = 1
        
        # Measurement incidence Matrix
        self.idx = idx
        no_idx_all = 4*self.no_brh + 4*self.no_bus
        self.IDX = np.zeros((self.no_mea, no_idx_all))
        _cache1 = 0
        _cache2 = 0
        for key in idx.keys():
            for _idx, _value in enumerate(idx[key]):
                self.IDX[_cache1+_idx, _cache2+_value] = 1                 
            _cache1 = _cache1 + len(idx[key])
            if key == 'pf' or key == 'pt' or key == 'qf' or key == 'qt':
                _cache2 = _cache2 + self.no_brh
            else:
                _cache2 = _cache2 + self.no_bus
        
        # Calculate the admittance matrix
        self._admittance_matrix()
    
    def update_reactance(self,x_new):
        """
        Update reactance in self.case
        """
        self.case['branch'][:,BR_X] = x_new
        self.case_int = ext2int(self.case)

        # Update the admittance matrix
        self._admittance_matrix()
    
    def _admittance_matrix(self):
        """
        Calculate the Admittance matrix according to the current admittance in self.case
        """
        Ybus, Yf, Yt = makeYbus(self.case_int['baseMVA'], self.case_int['bus'], self.case_int['branch'])
        self.Ybus = scipy.sparse.csr_matrix.todense(Ybus).getA()
        self.Yf = scipy.sparse.csr_matrix.todense(Yf).getA()
        self.Yt = scipy.sparse.csr_matrix.todense(Yt).getA()

        self.Gsh = self.case['bus'][:,GS]/self.case['baseMVA']
        self.Bsh = self.case['bus'][:,BS]/self.case['baseMVA']
    
    def run_opf(self, **kwargs):
        """
        Run the optimal power flow
        """
        
        case_opf = copy.deepcopy(self.case)
        if 'load_active' in kwargs.keys():
            # if a new load condition is given
            case_opf['bus'][:,PD] = kwargs['load_active']
            case_opf['bus'][:,QD] = kwargs['load_reactive']
        else:
            # Use the default load condition in the case file
            pass
        
        result = runopf(case_opf, opt)
        
        return result
        
    def construct_mea(self, result):
        """
        Given the OPF result, construct the measurement vector
        z = [pf, pt, pi, vang, qf, qt, qi, vmag] in the current setting
        """
        pf = result['branch'][:,PF]/self.case['baseMVA']
        pt = result['branch'][:,PT]/self.case['baseMVA']
        pi = (self.Cg@result['gen'][:,PG] - result['bus'][:,PD])/self.case['baseMVA']
        vang = result['bus'][:, VA]*np.pi/180             # In radian
        
        qf = result['branch'][:,QF]/self.case['baseMVA']
        qt = result['branch'][:,QT]/self.case['baseMVA']
        qi = (self.Cg@result['gen'][:,QG] - result['bus'][:,QD])/self.case['baseMVA']
        vmag = result['bus'][:, VM]
        
        z = np.concatenate([pf, pt, pi, vang, qf, qt, qi, vmag], axis = 0)
        
        z = self.IDX@z   # Select the measurement
        #print(z.shape)
        #print(self.R.shape)
        
        z_noise = z + np.random.multivariate_normal(mean = np.zeros((self.no_mea,)), cov = self.R)
        z = np.expand_dims(z, axis = 1)
        z_noise = np.expand_dims(z_noise, axis = 1)
        
        vang_ref = vang[self.ref_index]
        vmag_ref = vmag[self.ref_index]
        
        return z, z_noise, vang_ref, vmag_ref
    
    def construct_v_flat(self, vang_ref, vmag_ref):
        
        """
        Construct a flat start voltage Given the reference bus voltage
        vmag_ref: the reference bus voltage magnitude from result
        vang: the reference bus voltage phase angle from result
        """
        vang_flat = np.zeros((self.no_bus,))
        vmag_flat = np.ones((self.no_bus,))
        
        vang_flat[self.ref_index] = vang_ref    
        vmag_flat[self.ref_index] = vmag_ref
        
        return vang_flat, vmag_flat
        
    def h_x_pypower(self, v):
        """
        Estimate the measurement from the state: z_est = h(v)
        v is complex power
        z = [pf, pt, pi, vang, qf, qt, qi, vmag] in the current setting
        """
        #print(( np.diag(self.Cf@v)).shape)
        #print((np.conj(self.Yf)).shape)
        #print((np.conj(v).shape))

        # print(np.linalg.norm(self.Yf, 2))
        
        sf = np.diag(self.Cf@v)@np.conj(self.Yf)@np.conj(v)    # "from" complex power flow
        st = np.diag(self.Ct@v)@np.conj(self.Yt)@np.conj(v)    # "to" complex power flow
        si = np.diag(v)@np.conj(self.Ybus)@np.conj(v)          # complex power injection
        vang = np.angle(v)
        vmag = np.abs(v)
        
        pf = np.real(sf)
        pt = np.real(st)
        pi = np.real(si)
        qf = np.imag(sf)
        qt = np.imag(st)
        qi = np.imag(si)

        z_est = np.concatenate([pf, pt, pi, vang, qf, qt, qi, vmag], axis = 0)
        z_est = np.expand_dims(z_est, axis = 1)
        z_est = self.IDX@z_est
        
        return z_est
        
    
    def jacobian(self, v_est):
        """
        Given the stationary state, Calculate the Jacobian matrix.
        Return: the Jacobian matrix of full measurement
        """
        # 
        # Compute the Jacobian matrix
        [dsi_dvmag, dsi_dvang] = dSbus_dV(self.Ybus, v_est)   # si w.r.t. v
        [dsf_dvang, dsf_dvmag, dst_dvang, dst_dvmag, _, _] = dSbr_dV(self.case_int['branch'], self.Yf, self.Yt, v_est)  # sf w.r.t. v

        dpf_dvang = np.real(dsf_dvang)
        dqf_dvang = np.imag(dsf_dvang)
        dpf_dvmag = np.real(dsf_dvmag)
        dqf_dvmag = np.imag(dsf_dvmag)
        
        dpt_dvang = np.real(dst_dvang)
        dqt_dvang = np.imag(dst_dvang)   
        dpt_dvmag = np.real(dst_dvmag)
        dqt_dvmag = np.imag(dst_dvmag)  
        
        dpi_dvang = np.real(dsi_dvang)
        dqi_dvang = np.imag(dsi_dvang)
        dpi_dvmag = np.real(dsi_dvmag)
        dqi_dvmag = np.imag(dsi_dvmag)
        
        dvang_dvang = np.eye(self.no_bus)
        dvang_dvmag = np.zeros((self.no_bus, self.no_bus))
        
        dvmag_dvang = np.zeros((self.no_bus, self.no_bus))
        dvmag_dvmag = np.eye(self.no_bus)

        # z = [pf, pt, pi, vang, qf, qt, qi, vmag] in the current setting
        J = np.block([
            [dpf_dvang,    dpf_dvmag],
            [dpt_dvang,    dpt_dvmag],
            [dpi_dvang,    dpi_dvmag],
            [dvang_dvang,  dvang_dvmag],
            [dqf_dvang,    dqf_dvmag],
            [dqt_dvang,    dqt_dvmag],
            [dqi_dvang,    dqi_dvmag],
            [dvmag_dvang,  dvmag_dvmag]
        ])
        # # Remove the reference bus
        # J = np.block([
        #     [dpf_dvang[:,self.non_ref_index],    dpf_dvmag[:,self.non_ref_index]],
        #     [dpt_dvang[:,self.non_ref_index],    dpt_dvmag[:,self.non_ref_index]],
        #     [dpi_dvang[:,self.non_ref_index],    dpi_dvmag[:,self.non_ref_index]],
        #     [dvang_dvang[:,self.non_ref_index],  dvang_dvmag[:,self.non_ref_index]],
        #     [dqf_dvang[:,self.non_ref_index],    dqf_dvmag[:,self.non_ref_index]],
        #     [dqt_dvang[:,self.non_ref_index],    dqt_dvmag[:,self.non_ref_index]],
        #     [dqi_dvang[:,self.non_ref_index],    dqi_dvmag[:,self.non_ref_index]],
        #     [dvmag_dvang[:,self.non_ref_index],  dvmag_dvmag[:,self.non_ref_index]]
        # ])
        
        J = np.array(J)         # Force convert to numpy array

        return J     # This J is not full column rank as the reference bus is not removed.

    def ac_se_pypower(self, z_noise, vang_ref, vmag_ref, is_honest = True, **kwargs):
        """
        AC-SE based on pypower
        v_initial: initial gauss on the 
        is_honest: 
        If True, then honest state estimation is used where the Jacobian matrix is updated at each iteration
        If False, then dishonest state estimation is used where the Jacobian matrix is fixed at the initial point.
        """
        """
        Verbose
        """
        if len(kwargs.keys()) == 0:
            # Default verbose
            tol = 1e-5,    
            max_it = 100     
            verbose = 0
            
        else:
            tol = kwargs['config']['tol']    
            max_it = kwargs['config']['max_it']
            verbose = kwargs['config']['verbose']
        
        """
        Initialization
        """
        is_converged = 0
        ite_no = 0
        vang_est, vmag_est = self.construct_v_flat(vang_ref, vmag_ref)      # Flat start state
        v_est = vmag_est*np.exp(1j*vang_est)             # (no_bus, )
        
        # For the first run and also the dishonest Jacobian, the below value will never change
        J = self.jacobian(v_est)       # Jacobian matrix on the flat state 
        # Remove the reference columns (for both angle and magnitude)
        J = np.delete(J, [self.ref_index, self.ref_index+self.no_bus], 1)
        J = self.IDX@J          # Select the measurement
        # Update rule: x := x_0 + (Jx0^T * R^-1 * Jx0)^-1 * Jx0^T * R^-1 * (z-h(x_0))
        G = J.T@self.R_inv@J
        G_inv = np.linalg.inv(G)

        """
        Gauss-Newton Iteration
        """
        
        while is_converged == False and ite_no < max_it:
            # Update iteration counter
            ite_no += 1

            # Compute estimated measurement
            z_est = self.h_x_pypower(v_est)       # z is 2D array (no_mea, 1) 
            
            if is_honest == False:
                # No need to update the Jacobian
                #print('not update')
                pass
            else:
                #print('update')
                # Update the Jacobian
                J = self.jacobian(v_est)         # It is repeated for the first run on v flat!
                J = np.delete(J, [self.ref_index, self.ref_index+self.no_bus], 1)
                J = self.IDX@J          # Select the measurement
                # Update rule: x := x_0 + (Jx0^T * R^-1 * Jx0)^-1 * Jx0^T * R^-1 * (z-h(x_0))
                G = J.T@self.R_inv@J
                G_inv = np.linalg.inv(G)
            
            # Test observability
            rankG = np.linalg.matrix_rank(G)
            if rankG < G.shape[0]:
                print(f'The current measurement setting is not observable.')
                break
            
            F = J.T@self.R_inv@(z_noise-z_est)
            dx = (G_inv@F).flatten()  # Note that the voltages are 1D array
            normF = np.linalg.norm(F, np.inf) 
            
            if verbose == 0:
                pass
            else:
                print(f'iteration {ite_no} norm of mismatch: {np.round(normF,6)}')
            
            # Terminate condition
            if normF < tol:
                is_converged = True
            
            # Update
            vang_est[self.non_ref_index] = vang_est[self.non_ref_index] + dx[:len(self.non_ref_index)]
            vmag_est[self.non_ref_index] = vmag_est[self.non_ref_index] + dx[len(self.non_ref_index):]
            v_est = vmag_est*np.exp(1j*vang_est)
        
        return v_est

    def bdd_residual(self, z_noise, v_est):
        """
        Find the residual of chi^2 detector given the estimated state
        """
        # Find z_est
        z_est = self.h_x_pypower(v_est)
        
        return ((z_noise-z_est).T@self.R_inv@(z_noise-z_est))[0,0]
        
        
"""
An example
"""
if __name__ == "__main__":
    case = case14()
    
    # Define measurement idx
    mea_idx, no_mea, noise_sigma = define_mea_idx_noise(case, 'FULL')
    # Instance the state estimation class
    se = SE(case, noise_sigma=noise_sigma, idx=mea_idx, fpr = 0.02)
    
    # Run OPF to get the measurement
    opt = ppoption()              # OPF options
    opt['VERBOSE'] = 0
    opt['OUT_ALL'] = 0
    opt['OPF_FLOW_LIM'] = 1       # Constraint on the active power flow
    result = runopf(case, opt)

    print(result['success'])
    
    # Construct the measurement
    z, z_noise, vang_ref, vmag_ref = se.construct_mea(result) # Get the measurement
    
    # Run AC-SE
    se_config['verbose'] = 1
    v_est, _ = se.ac_se_pypower(z_noise, vang_ref, vmag_ref, config = se_config)
    residual = se.bdd_residual(z_noise, v_est)    
    print(f'BDD threshold: {se.bdd_threshold}')
    print(f'residual: {residual}')
    
    
        
        