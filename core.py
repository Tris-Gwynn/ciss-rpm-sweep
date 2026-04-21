# ciss_rpm/core.py
import numpy as np
import qutip as qt
import scipy.linalg as sc
import scipy.sparse as sp
from .constants import GYRO_E

class RPMSystem:
    """
    Manages the dimensions, basis states, and static operators of the Radical Pair System.
    Initialized ONCE to save computational time.
    """
    def __init__(self, donor_spin=0.5, acceptor_spin=0, n_shelving=5): 
        # UPDATED: Default n_shelving set to 5
        self.d_dim = int(2 * donor_spin + 1)
        self.a_dim = int(2 * acceptor_spin + 1)
        self.n_uc_mult = self.d_dim * self.a_dim  
        
        # System Dimensions
        self.sys_dim = 4 * self.n_uc_mult + n_shelving
        self.n_shelving = n_shelving  # CRITICAL: Store this for factories
        
        # Shelving Indices
        self.idx_S = self.sys_dim - 5
        self.idx_Tm = self.sys_dim - 4
        self.idx_T0 = self.sys_dim - 3
        self.idx_Tp = self.sys_dim - 2
        self.idx_R = self.sys_dim - 1

        # Pre-compute Identity matrices
        self.I2 = qt.qeye(2)
        self.Id = qt.qeye(self.d_dim)
        self.Ia = qt.qeye(self.a_dim)
        
        # Spin Operators
        self.SD, self.SA = self._build_spin_ops()
        self.ID, self.IA = self._build_nuc_ops(donor_spin, acceptor_spin)
        
        # Shelving Projectors
        self.P_shelf_S  = qt.projection(self.sys_dim, self.idx_S, self.idx_S)
        self.P_shelf_Tm = qt.projection(self.sys_dim, self.idx_Tm, self.idx_Tm)
        self.P_shelf_T0 = qt.projection(self.sys_dim, self.idx_T0, self.idx_T0)
        self.P_shelf_Tp = qt.projection(self.sys_dim, self.idx_Tp, self.idx_Tp)
        self.P_shelf_R  = qt.projection(self.sys_dim, self.idx_R, self.idx_R)
        
        # Composite Projectors
        self.P_shelf_T = self.P_shelf_Tm + self.P_shelf_T0 + self.P_shelf_Tp
        self.P_shelf_F = self.P_shelf_S + self.P_shelf_T
        
        # Basis Vectors
        self.ket_shelf_S  = qt.basis(self.sys_dim, self.idx_S)
        self.ket_shelf_Tm = qt.basis(self.sys_dim, self.idx_Tm)
        self.ket_shelf_T0 = qt.basis(self.sys_dim, self.idx_T0)
        self.ket_shelf_Tp = qt.basis(self.sys_dim, self.idx_Tp)
        self.ket_shelf_R  = qt.basis(self.sys_dim, self.idx_R)

    def _build_spin_ops(self):
        Sx, Sy, Sz = 0.5 * qt.sigmax(), 0.5 * qt.sigmay(), 0.5 * qt.sigmaz()
        SD = {
            'x': qt.tensor(Sx, self.I2, self.Id, self.Ia),
            'y': qt.tensor(Sy, self.I2, self.Id, self.Ia),
            'z': qt.tensor(Sz, self.I2, self.Id, self.Ia)
        }
        SA = {
            'x': qt.tensor(self.I2, Sx, self.Id, self.Ia),
            'y': qt.tensor(self.I2, Sy, self.Id, self.Ia),
            'z': qt.tensor(self.I2, Sz, self.Id, self.Ia)
        }
        return SD, SA

    def _build_nuc_ops(self, d_spin, a_spin):
        # Donor Nucleus
        Ix = qt.jmat(d_spin, 'x')
        Iy = qt.jmat(d_spin, 'y')
        Iz = qt.jmat(d_spin, 'z')
        ID = {
            'x': qt.tensor(self.I2, self.I2, Ix, self.Ia),
            'y': qt.tensor(self.I2, self.I2, Iy, self.Ia),
            'z': qt.tensor(self.I2, self.I2, Iz, self.Ia)
        }
        
        # Acceptor Nucleus
        if a_spin == 0:
            z0 = qt.qzero(self.a_dim)
            IA = {
                'x': qt.tensor(self.I2, self.I2, self.Id, z0),
                'y': qt.tensor(self.I2, self.I2, self.Id, z0),
                'z': qt.tensor(self.I2, self.I2, self.Id, z0)
            }
        else:
            Ix = qt.jmat(a_spin, 'x')
            Iy = qt.jmat(a_spin, 'y')
            Iz = qt.jmat(a_spin, 'z')
            IA = {
                'x': qt.tensor(self.I2, self.I2, self.Id, Ix),
                'y': qt.tensor(self.I2, self.I2, self.Id, Iy),
                'z': qt.tensor(self.I2, self.I2, self.Id, Iz)
            }
        return ID, IA

# --- Hamiltonian Factories (UPDATED TO USE sys.n_shelving) ---

def r_y(a):
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])

def r_z(a):
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])

def get_rotation_matrix(theta, phi):
    return r_z(theta) @ r_y(phi)

def get_zeeman_hamiltonian(sys: RPMSystem, B0, theta, phi):
    Bx = GYRO_E * B0 * np.sin(theta) * np.cos(phi)
    By = GYRO_E * B0 * np.sin(theta) * np.sin(phi)
    Bz = GYRO_E * B0 * np.cos(theta)
    
    H_z = (Bx * (sys.SD['x'] + sys.SA['x']) + 
           By * (sys.SD['y'] + sys.SA['y']) + 
           Bz * (sys.SD['z'] + sys.SA['z']))
    
    # FIX: Use sys.n_shelving instead of hardcoded 3
    return qt.Qobj(sp.block_diag([H_z.full(), np.zeros((sys.n_shelving, sys.n_shelving))],format="csr"))

def get_hyperfine_hamiltonian(sys: RPMSystem, Axx, Ayy, Azz, theta, phi):
    PAX = GYRO_E * np.array([[Axx, 0, 0], [0, Ayy, 0], [0, 0, Azz]])
    R = get_rotation_matrix(theta, phi)
    LAB = R @ PAX @ R.T 
    
    H_accum = 0
    dirs = ['x', 'y', 'z']
    for i, d1 in enumerate(dirs):
        for j, d2 in enumerate(dirs):
            if LAB[i, j] != 0:
                H_accum += LAB[i, j] * sys.SD[d1] * sys.ID[d2]
    
    if isinstance(H_accum, int) and H_accum == 0:
        active_dim = sys.SD['x'].shape[0]
        H_mat = np.zeros((active_dim, active_dim), dtype=complex)
    else:
        H_mat = H_accum.full()

    # FIX: Use sys.n_shelving instead of hardcoded 3
    return qt.Qobj(sp.block_diag([H_mat, np.zeros((sys.n_shelving, sys.n_shelving))],format="csr"))

def get_dipolar_hamiltonian(sys: RPMSystem, Dxx, Dyy, Dzz, theta, phi):
    PAX = GYRO_E * np.array([[Dxx, 0, 0], [0, Dyy, 0], [0, 0, Dzz]])
    R = get_rotation_matrix(theta, phi)
    LAB = R @ PAX @ R.T
    
    H_accum = 0
    dirs = ['x', 'y', 'z']
    for i, d1 in enumerate(dirs):
        for j, d2 in enumerate(dirs):
            if LAB[i, j] != 0:
                H_accum += LAB[i, j] * sys.SD[d1] * sys.SA[d2]
    
    if isinstance(H_accum, int) and H_accum == 0:
        active_dim = sys.SD['x'].shape[0]
        H_mat = np.zeros((active_dim, active_dim), dtype=complex)
    else:
        H_mat = H_accum.full()
    
    # FIX: Use sys.n_shelving instead of hardcoded 3
    return qt.Qobj(sp.block_diag([H_mat, np.zeros((sys.n_shelving, sys.n_shelving))],format="csr"))

def get_exchange_hamiltonian(sys: RPMSystem, J):
    dot = (sys.SD['x'] * sys.SA['x'] + 
           sys.SD['y'] * sys.SA['y'] + 
           sys.SD['z'] * sys.SA['z'])
    
    ident = qt.tensor(sys.I2, sys.I2, sys.Id, sys.Ia)
    H_ex = -J * (2 * dot + 0.5 * ident)
    
    # FIX: Use sys.n_shelving instead of hardcoded 3
    return qt.Qobj(sp.block_diag([H_ex.full(), np.zeros((sys.n_shelving, sys.n_shelving))],format="csr"))