"""
This file realizes the linear AC state estimation in paper:
'Linear State Estimation and Bad Data Detection for Power Systems with RTU and PMU Measurements'
"""

import numpy as np
from pypower.api import case14, ppoption, runopf
from run_AC_SE import SE
from mea_idx import define_mea_idx_noise
from se_config import se_config
from pypower.idx_bus import VM, VA, PD, QD
from pypower.idx_gen import PG, QG
from pypower.idx_brch import PF, PT, QF, QT

"""
Conventional state estimation
"""

# Initialize the case
case = case14()
# Define measurement idx
mea_idx, no_mea, noise_sigma = define_mea_idx_noise(case, 'full')

# Instance the state estimation class
se = SE(case, noise_sigma=noise_sigma, idx=mea_idx, fpr = 0.02)

# Run OPF to get the measurement
opt = ppoption()              # OPF options
opt['VERBOSE'] = 0
opt['OUT_ALL'] = 0
opt['OPF_FLOW_LIM'] = 1       # Constraint on the active power flow
result = se.run_opf()

# Define matrices
# Construct the measurement
z, z_noise, vang_ref, vmag_ref = se.construct_mea(result) # Get the measurement

# Run AC-SE
se_config['verbose'] = 1
v_est = se.ac_se_pypower(z_noise, vang_ref, vmag_ref, config = se_config)
residual = se.bdd_residual(z_noise, vang_ref, vmag_ref)    
print(f'BDD threshold: {se.bdd_threshold}')
print(f'residual: {residual}')


"""
Linear state estimation
"""

# voltage
vmag = result['bus'][:,VM]
vang = result['bus'][:,VA]*np.pi/180
v = vmag*np.exp(1j*vang)
vreal = np.real(v)
vimag = np.imag(v)

x = np.concatenate([vreal,vimag],axis=0)

print(f'vmag: {vmag} = {np.sqrt(vreal**2+vimag**2)}')

# power
pbus = (se.Cg@result['gen'][:,PG] - result['bus'][:,PD])/se.case['baseMVA']
qbus = (se.Cg@result['gen'][:,QG] - result['bus'][:,QD])/se.case['baseMVA']
pf = result['branch'][:,PF]/se.case['baseMVA']
qf = result['branch'][:,QF]/se.case['baseMVA']
pt = result['branch'][:,PT]/se.case['baseMVA']
qt = result['branch'][:,QT]/se.case['baseMVA']

# Admittance matrix
Ybus = se.Ybus
Yf = se.Yf
Yt = se.Yt
Gf = np.real(Yf)
Gt = np.real(Yt)
Bf = np.imag(Yf)
Bt = np.imag(Yt)

# 
pbusv = np.diag(pbus/vmag**2)
qbusv = np.diag(qbus/vmag**2)

ybusr_ = se.Cf.T@Gf + se.Ct.T@Gt+np.diag(se.Gsh)
ybusi_ = se.Cf.T@Bf + se.Ct.T@Bt+np.diag(se.Bsh)
ybusr = np.real(Ybus)
ybusi = np.imag(Ybus)

pfv = np.diag(pf/(se.Cf@vmag**2))@se.Cf
qfv = np.diag(qf/(se.Cf@vmag**2))@se.Cf
ptv = np.diag(pt/(se.Ct@vmag**2))@se.Ct
qtv = np.diag(qt/(se.Ct@vmag**2))@se.Ct

H = np.array(np.block([
    [pbusv - ybusr, qbusv + ybusi],
    [qbusv + ybusi, ybusr - pbusv],

    [pfv - Gf, qfv + Bf],
    [qfv+Bf, Gf - pfv],
    [ptv-Gt, qtv + Bt],
    [qtv + Bt, Gt - ptv]
]))

# Remove the column indexed by the reference bus
href = H[:,[se.ref_index[0], se.no_bus+se.ref_index[0]]]
xref = x[[se.ref_index[0], se.no_bus+se.ref_index[0]]]
print(href)
print(xref)
zref = href@xref
print(zref)
Hr = np.delete(H,[se.ref_index[0], se.no_bus+se.ref_index[0]],1)
xr = np.delete(x,[se.ref_index[0], se.no_bus+se.ref_index[0]])

print(f'Hr shape: {Hr.shape}')
print(f'Hr rank: {np.linalg.matrix_rank(Hr)}')

z = Hr@xr + zref
print(f'z norm: {np.linalg.norm(z, np.inf)}')
u,s,vh = np.linalg.svd(Hr)
print(f'singular value: {s}')
#print(f'u : {u}')
#print(f'vh: {vh}')

# Truncated singular value decomposition
r = np.sum(s >= 1e-8)
print(r)

zpse = np.zeros((Hr.shape[0],))
xest = np.linalg.inv(Hr.T@Hr)@Hr.T@(zpse-zref)
print(xest)
print(xr)
print(f'Estimation deviation: {np.linalg.norm(xr - xest, np.inf)}')
print(f'Gf: {Gf}')
print(f'Bf: {Bf}')
