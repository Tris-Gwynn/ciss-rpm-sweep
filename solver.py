# ciss_rpm/solver.py
import numpy as np
import qutip as qt
import scipy.linalg as sc
import scipy.sparse as sp
from .core import RPMSystem
from .constants import DEFAULT_STEPS

class CISSSolver:
    def __init__(self, system: RPMSystem):
        self.sys = system
        
        S_state = qt.singlet_state()
        T_states = qt.triplet_states()
        Tm, T0, Tp = T_states[0], T_states[1], T_states[2]
        
        self.basis_S = []
        self.basis_T0 = []
        self.basis_Tm = []
        self.basis_Tp = []
        
        self.P_singlet = 0
        self.P_tm = 0
        self.P_t0 = 0
        self.P_tp = 0
        
        for i in range(self.sys.n_uc_mult):
            nuc = qt.basis(self.sys.n_uc_mult, i)
            
            psi_S = qt.tensor(S_state, nuc)
            psi_T0 = qt.tensor(T0, nuc)
            psi_Tm = qt.tensor(Tm, nuc)
            psi_Tp = qt.tensor(Tp, nuc)
            
            self.basis_S.append(psi_S)
            self.basis_T0.append(psi_T0)
            self.basis_Tm.append(psi_Tm)
            self.basis_Tp.append(psi_Tp)
            
            # Accumulate Projectors
            self.P_singlet += self._pad_to_shelf(psi_S * psi_S.dag())
            self.P_t0 += self._pad_to_shelf(psi_T0 * psi_T0.dag())
            self.P_tm += self._pad_to_shelf(psi_Tm * psi_Tm.dag())
            self.P_tp += self._pad_to_shelf(psi_Tp * psi_Tp.dag())

    def _pad_to_shelf(self, qobj_input):
        mat = qobj_input.full()
        n_shelf = self.sys.n_shelving

        if mat.shape[1] == 1:
            return qt.Qobj(np.concatenate([mat, np.zeros((n_shelf, 1))]))
        else:
            return qt.Qobj(sp.block_diag([mat, np.zeros((n_shelf, n_shelf))],format="csr"))

    def get_initial_rho(self, chi, initial_nuc_state='thermal'):
        """
        initial_nuc_state: 
            'thermal' (Default) -> Returns 1/N * Identity (Mixed)
            'up'                -> Returns |Up><Up| (Polarized)
            'down'              -> Returns |Down><Down| (Polarized)
        """
        c = np.cos(chi / 2.0)
        s = np.sin(chi / 2.0)
        
        # Helper to make the density matrix for a specific nuclear index
        def make_rho_for_index(idx):
            psi = c * self.basis_S[idx] + s * self.basis_T0[idx]
            full_psi = self._pad_to_shelf(psi)
            return full_psi * full_psi.dag()

        # CASE 1: Thermal (1/2, 1/2) - matches 1/2(1 0, 0 1)
        if initial_nuc_state == 'thermal':
            rho_accum = 0
            for i in range(self.sys.n_uc_mult):
                rho_accum += make_rho_for_index(i)
            return rho_accum / self.sys.n_uc_mult

        # CASE 2: Polarized UP - matches (1 0, 0 0)
        elif initial_nuc_state == 'up':
            return make_rho_for_index(0) # Index 0 is standard for Up
            
        # CASE 3: Polarized DOWN - matches (0 0, 0 1)
        elif initial_nuc_state == 'down':
            # Assumes spin-1/2 (2 states). 
            # If larger spin, this should be self.sys.n_uc_mult - 1
            return make_rho_for_index(1) 
            
        else:
            raise ValueError("state must be 'thermal', 'up', or 'down'")

    def get_collapse_ops(self, chi_recomb, ks, kt, kr):
        c = np.cos(chi_recomb / 2.0)
        s = np.sin(chi_recomb / 2.0)
        c_ops = []
        for i in range(self.sys.n_uc_mult):
            # Singlet Decay
            op_S = self.sys.ket_shelf_S * self._pad_to_shelf(self.basis_S[i]).dag()
            c_ops.append(np.sqrt(ks) * op_S)
            
            # Triplet Decay
            op_Tm = self.sys.ket_shelf_Tm * self._pad_to_shelf(self.basis_Tm[i]).dag()
            op_T0 = self.sys.ket_shelf_T0 * self._pad_to_shelf(self.basis_T0[i]).dag()
            op_Tp = self.sys.ket_shelf_Tp * self._pad_to_shelf(self.basis_Tp[i]).dag()
            c_ops.append(np.sqrt(kt) * op_Tm)
            c_ops.append(np.sqrt(kt) * op_T0)
            c_ops.append(np.sqrt(kt) * op_Tp)
            
            # Backscatter (Recombination)
            psi_back = c * self.basis_S[i] - s * self.basis_T0[i]
            op_R = self.sys.ket_shelf_R * self._pad_to_shelf(psi_back).dag()
            c_ops.append(np.sqrt(kr) * op_R)
        return c_ops

    def solve(self, H, chi_init, chi_recomb, ks, kt, kr, method='dop853'):
        t_max = 5.0 / ks
        times = np.linspace(0, t_max, 250) # Assuming DEFAULT_STEPS is 250
        
        rho0 = self.get_initial_rho(chi_init)
        c_ops = self.get_collapse_ops(chi_recomb, ks, kt, kr)
        
        e_ops = [
            self.sys.P_shelf_S, 
            self.sys.P_shelf_Tp,
            self.sys.P_shelf_T0,
            self.sys.P_shelf_Tm, 
            self.sys.P_shelf_R, 
            self.P_singlet,
            self.P_tm,
            self.P_t0,
            self.P_tp
        ]
        
        opts = {
            "store_states": False, 
            "method": method, 
            "nsteps": 50000, 
            "atol": 1e-8, 
            "rtol": 1e-6
        }
        
        result = qt.mesolve(H, rho0, times, c_ops, e_ops=e_ops, options=opts)
        return np.real(result.expect)