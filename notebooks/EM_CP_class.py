"""
Importing the pAGN module from Gangardt et al. 2024. We use the Thompson model for AGN disks as used in the Tagawa et al. 2023 paper
"""

from pagn import Thompson
from pagn import Sirko
import numpy as np
import matplotlib.pyplot as plt
import pagn.constants as ct

import pandas as pd
import math
from scipy.optimize import brentq, bisect
import random


class AGN_model():
    def __init__(self, msmbh = 1e8, epsilon=1e-3, m=0.5, xi=1, Mdot_out=10*ct.MSun/ct.yr, Rout=100*ct.pc, Rin=1e-3*ct.pc, 
                 opacity="combined", disk_model='Thompson', seed=100):
        self.c=3e10
        self.h=6.6261e-27
        self.kb=1.3807e-16
    
        self.msmbh = msmbh
        self.epsilon = epsilon
        self.m = m
        self.xi = xi
        self.Mdot_out = Mdot_out
        self.Rout = Rout
        self.Rin = Rin
        self.opacity = opacity
        self.disk_model = disk_model
        random.seed(seed)
        #print("The following are the parameters set for the AGN and BBH system.")
        #print(f"Mass of the supermassive black hole at the centre of the AGN disk {self.msmbh} M☉")
        #print(f"Mass of the binary black hole remnant {self.m_rem} M☉")
        #print(f"Model of the AGN disk chosen {self.disk_model}")
        #print(f"Model of the AGN disk chosen {self.disk_model}")
        #print(f"Opening angle at the base of the jet {self.disk_model}")

    def solve_AGN_disk_prop(self):
        """
        The Thompson model has the following input parameters: SMBH mass, velocity dispersion value (sigma), star formation efficiancy (epsilon),
        the radiative efficiency of supernovae (xi), angular momentum transfer parameter (m), gas inflow rate at the outer boundary (Mdot_out/Min),
        Radius of inner boundary of AGN disc (Rin), the outer disk boundary (Rout) (this is also the radius beyond which disk accretion rate is no longer constant).
        Out of these, either sigma or SMBH mass is calculated using the M-sigma relation.
        If Rin is None the package uses 6 timesgravitational radii of SMBH, if Rout is None 1e7 times Schwarzchild radii is used
        """
        if self.disk_model == 'Thompson':
            self.tho = Thompson.ThompsonAGN(Mbh = self.msmbh*ct.MSun, epsilon = self.epsilon, m = self.m, xi = self.xi,
                      Mdot_out= self.Mdot_out, Rout = self.Rout, Rin = self.Rin, opacity = self.opacity)
        else:
            print("Other models are not available at the moment.")

        self.tho.solve_disk(N=1e4)


    def retrieve_props(self):
        if not hasattr(self, "tho"):
            raise RuntimeError("AGN disk not solved yet. Call solve_AGN_disk_prop() first.")
        params_list = ["rho", "T", "h", "tauV", "cs"]
        #for name in params_list:
        #    setattr(self, name, getattr(self.tho, name).copy())
        print("The values of density, temperature, disk height, optical depth at the mid-plane and the speed of sound respectively (all in SI Units)")
        return {"rho": self.tho.rho, "T": self.tho.T, "h": self.tho.h, "tauV": self.tho.tauV, "cs": self.tho.cs, "R": self.tho.R}

    def plot_AGN_disk_properties(self):
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))

        ax[0][0].plot(self.tho.R/ct.pc, self.tho.rho*ct.SI_to_gcm3, '-', color='tab:blue')
        ax[0][0].set_xscale('log')
        ax[0][0].set_yscale('log')
        ax[0][0].set_xlim([1e-3, 10])
        ax[0][0].grid(True)
        #ax.axvspan(22, 27.5, color='blue', alpha=0.1)
        #ax.legend()
        ax[0][0].set_xlabel(r"R [pc]", fontsize=10)
        ax[0][0].set_ylabel(r"$\rho \, [\mathrm{g cm^{-3}}]$", fontsize=10)

        ax[0][1].plot(self.tho.R/ct.pc, self.tho.h/ct.pc, '-', color='tab:blue')
        ax[0][1].set_xscale('log')
        ax[0][1].set_yscale('log')
        ax[0][1].set_xlim([1e-3, 10])
        ax[0][1].grid(True)
        #ax.axvspan(22, 27.5, color='blue', alpha=0.1)
        #ax.legend()
        ax[0][1].set_xlabel(r"R [pc]", fontsize=10)
        ax[0][1].set_ylabel(r"h [pc]", fontsize=10)

        ax[1][0].plot(self.tho.R/ct.pc, self.tho.T, '-', color='tab:blue')
        ax[1][0].set_xscale('log')
        ax[1][0].set_yscale('log')
        ax[1][0].set_xlim([1e-3, 10])
        ax[1][0].grid(True)
        #ax.axvspan(22, 27.5, color='blue', alpha=0.1)
        #ax.legend()
        ax[1][0].set_xlabel(r"R [pc]", fontsize=10)
        ax[1][0].set_ylabel(r"T [K]", fontsize=10)

        ax[1][1].plot(self.tho.R/ct.pc, self.tho.tauV, '-', color='tab:blue')
        ax[1][1].set_xscale('log')
        ax[1][1].set_yscale('log')
        ax[1][1].set_xlim([1e-3, 10])
        ax[1][1].grid(True)
        #ax.axvspan(22, 27.5, color='blue', alpha=0.1)
        #ax.legend()
        ax[1][1].set_xlabel(r"R [pc]", fontsize=10)
        ax[1][1].set_ylabel(r"$\tau_V$", fontsize=10)
        
        fig.tight_layout()
        plt.show()
        


class EM_model():
    def __init__(self, msmbh, m_rem, eta_j=0.5, alpha_AGN=0.1, f_acc=15, theta_0=0.3, mag_ampl=0.1,
                 elec_frac=0.3, p=2.5, AGN_model=None, calc_disk_prop=False, disk_prop_vars=None):
        self.m_rem = m_rem
        self.eta_j = eta_j
        self.alpha_AGN = alpha_AGN
        self.f_acc = f_acc
        self.theta_0 = theta_0
        self.mag_ampl = mag_ampl
        self.elec_frac = elec_frac
        self.p = p
        self.c=3e10
        self.h=6.6261e-27
        self.kb=1.3807e-16
        self.msmbh = msmbh
        #self.disk_prop = disk_prop
        if calc_disk_prop:
            if disk_prop_vars is None:
                raise ValueError("disk_prop_vars must be provided to calculate AGN disk properties")
            self.AGN_disk = AGN_model(msmbh = self.msmbh, epsilon = 1e-3, m=0.5, xi=1, Mdot_out=disk_prop_vars['Mdot_out'], Rout = disk_prop_vars['Rout'], Rin = disk_prop_vars['Rin'], 
                 opacity="combined", seed=100)
            self.AGN_disk.solve_AGN_disk_prop()
            
        else:
            try:
                self.AGN_disk = AGN_model
            except AttributeError:
                raise AttributeError("Solve for the AGN disk properties first using AGN_model.solve_AGN_disk_prop()")
    
    def L_j(self, rho_AGN, H_AGN, R):
        #print(self.m_rem, self.msmbh, self.eta_j, self.f_acc)
        self.mdot_cap = 3e-4*(H_AGN/0.003)*(R)**(1/2)*(rho_AGN/4e-17)*(self.m_rem/10)**(2/3)*(self.msmbh/1e6)**(-1/6) #Calculate mdot_cap from eq 1 in Tagawa et al. 2022
        return 1e42*(self.mdot_cap/3e-4)*(self.eta_j/0.5)*(self.f_acc/0.1)
    
    def L_tilde(self, rho_AGN, H_AGN, L_jet):
        def L_tilde_eqtn(L_tilde_var, rho_AGN, H_AGN, L_jet):
            eqnt_val_branch = ((25/9)*(L_jet)/((1+L_tilde_var**(-1/2))**2*self.theta_0**4*self.c**3*rho_AGN*(ct.pc*100*H_AGN)**2))**(2/5)
            return L_tilde_var - eqnt_val_branch
        L_tilde_final = np.zeros_like(L_jet)
        ids_no_sol=[]
        for id_val in np.arange(len(L_jet)):
            try:
                L_tilde_final[id_val] = bisect(L_tilde_eqtn, 0.001, 100, args=(rho_AGN[id_val], H_AGN[id_val], L_jet[id_val]))
            except ValueError:
                ids_no_sol.append(id_val)
                continue
        print(f"For L_tilde, we get no solutions for {len(ids_no_sol)} values of R")
        return L_tilde_final
        
    def beta_func(self, L_tilde_var):
        return (1+L_tilde_var**(-1/2))**(-1)
    
    def gamma_from_beta(self, beta):
        """
        Calculates the Lorentz factor from the velocity.
        """
        return (1-beta**2)**(-1/2)
    
    def beta_from_gamma(self, gamma):
        """
        Calculates the velocity from the Lorentz factor.
        """
        return (1-1/gamma**2)**(1/2)
    
    
    """
    The following is the calculation of possible oberservable properties of the EM counterpart.
    Most of the equations are taken from the EM counterpart model from the Tagawa et al. 2023 paper.
    Mostly the variables used are the properties of the AGN disk like scale height of the AGN disk, 
    and the velocity of the waves shock front and the parameters/functions defined above.
    Disclaimer: In all the functions below and above, H_AGN is inputed and considered in parsecs, whereas rho_AGN is in g/cm^3.
    """
    
    def t_delay(self, H_AGN, beta_FS):
        """
        This is the time delay between the gravitational wave emission and the breakout emission.
        We use Equation 4 from the Tagawa et al. 2023. 
        Technically the value should be lowered by the time needed by the GW to scale the height,
        but since the speed of the shock is low (nonrelativitic), we can neglect that value.
        """
        return 0.3*(3.086e18*H_AGN/5e16)*(beta_FS/0.1)**(-1)  #returns in yr
    
    def t_break(self, H_AGN, beta_FS):
        """
        Same as the function above. This is actually the time of the breakout emission, but due to the reasons mentioned above,
        this also becomes t_delay in the nonrelativistic case.
        """
        return 0.3*((3.086e18*H_AGN)/5e16)*(beta_FS/0.1)**(-1)  #returns in yr
    
    def T_breakout_low(self, rho_AGN, beta_FS):
        """
        The thermal equilibrium temperature at shock breakout for the low velocity Newtonian case.
        We use Equation 2 from Tagawa et al. 2023 for this.
        """
        return 1e4*(rho_AGN/1e-16)**(1/4)*(beta_FS/0.02)**(1/2)
    
    def t_breakout(self, beta_FS, rho_AGN):
        """
        This is the duration of emission from the breakout shell. 
        We define breakout the moment when the time required for photons to diffue out is equal to the time
        for the shock to expand the same distance to surface of the AGN disk.
        This time is the time required for the last few photons to be released. Thus also making this the duration of the breakout emission.
        We use Equation 3 from Tagawa et al. 2023 for this.
        """
        return 3*(rho_AGN/1e-16)**(-1)*(beta_FS/0.1)**(-2)    #returns in yr
    
    def t_duration(self, beta_FS, rho_AGN):
        """
        For reasons explained above, same equation as above.
        For the relativistic case, things are different because of production/annhilation of pairs (refer to Nakar & Sari 2012).
        """
        return 3*(rho_AGN/1e-16)**(-1)*(beta_FS/0.1)**(-2)    #returns in yr
    
    def L_sh(self, H_AGN, rho_AGN, beta_FS, R):
        """
        The kinetic power of the shock, 
        which is the rate at which energy is taken up by the shock and converted to internal energy of the matter.
        We consider a cone of half angle theta_j being the base of the cylinder which produces the emission.
        This becomes the volume of this cylinder * density * velocity **2.
        volume=A*(v*t), A=pi*(theta*H)**2
        We use Equation 5 from Tagawa et al. 2023
        """
        theta_j = (self.L_j(rho_AGN, H_AGN, R)*self.theta_0**6*beta_FS**2/(rho_AGN*(3.086e18*H_AGN) **2*self.c**3))**(1/10)
        return 5e43*(theta_j/0.05)**2*((3.086e18*H_AGN)/5e16)**2*(rho_AGN/1e-16)*(beta_FS/0.1)**3    #returns in erg/s
    
    def L_breakout(self, H_AGN, rho_AGN, beta_FS, R):
        """
        The rate at which energy trapped inside as radiation escapes at breakout.
        Typically a fraction of the internal energy stored, however in the nonrelativistic case,
        most of the kinetic power escapes out as breakout luminosity.
        """
        theta_j = (self.L_j(rho_AGN, H_AGN, R)*self.theta_0**6*beta_FS**2/(rho_AGN*(3.086e18*H_AGN)**2*self.c**3))**(1/10)
        return 5e43*(theta_j/0.05)**2*((3.086e18*H_AGN)/5e16)**2*(rho_AGN/1e-16)*(beta_FS/0.1)**3    #returns in erg/s
    
    def T_breakout_high(self, beta_FS, rho_AGN):
        """
        The temperature at breakout for the slightly higher velocity nonrelativistic case.
        We use Equation 6 from Tagawa et al. 2023 to calculate this.
        """
        return 1e4*10**(0.975+1.735*(beta_FS/0.1)**(1/2)+(0.26-0.08*(beta_FS/0.1)**(1/2))*np.log10(rho_AGN/1.673e-15))    #returns in K
    
    def sed_wien(self, nu, temp_breakout, L_breakout):
        """
        The spectral distribution of luminosity for the relatively high velocity nonrelatvisitic case.
        As mentioned in Tagawa et al 2023. in this regime, the radiation is not in thermal equilibrium as the photons
        produced by free-free emission is less than required. The distribution is characterised by a Wien distribution spectrum.
        We calculate the distribution of the thermal luminosity by distributing the power of the breakout luminosity accross the frequency domain.
        We use the idea that the integral of L_nu from 0 to infinity should be equal to the the breakout luminosity.
        """
        spect_rad = (2*self.h/self.c**2)*nu**3*(np.exp(-(self.h*nu)/(self.kb*temp_breakout)))  ## The spectral radiance formula for Wien's distribution spectrum
        norm_factor = (2*self.h/self.c**2)*6*(self.kb*temp_breakout/self.h)**4   ## Normalisation factor of the spectral radiance obtained by integrating the spectral radiance from 0 to inf wrt frequency
        return nu*L_breakout*spect_rad/norm_factor   ## returns in erg/s
        #return L_breakout*(h*nu/(kb*temp_breakout))**4*(np.exp(-(h*nu/(kb*temp_breakout))))/6
    
    def sed_bb(self, nu, temp_breakout, L_breakout):
        """
        The spectral distribution of luminosity for the low velocity nonrelativistic case.
        In this case, as the velocity of the forward is shock is low, there is time to produce
        enough photons via free-free emission for the radiation to be in thermal equilibrium.
        Therefore, the radiation can be characterised by a blackbody radiation and thus estimated
        using the Planck's distribution spectrum.
        We calculate the distribution using the same logic as above.
        """
        spect_rad = (2*self.h/self.c**2)*nu**3*(1/(np.exp((self.h*nu)/(self.kb*temp_breakout))-1))  ## The spectral radiance formula for Planck 's distribution spectrum
        norm_factor = (2*self.h/self.c**2)*(self.kb*temp_breakout/self.h)**4*(np.pi**4/15)   ## Normalisation factor of the spectral radiance obtained by integrating the spectral radiance from 0 to inf wrt frequency
        return nu*L_breakout*spect_rad/norm_factor
    
    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx


    def emission_properties_retrieve(self):
        """
        We first calculate the properties of each regime for the whole disk regardless of whether it belongs to that regime or not.
        We will later find the final values of these properties based on the velocity criteria.
        Our regimes are separated as follows:
        beta_FS*gamma_FS <= 0.03, nonrelativisitic low velocity regime
        0.03 <= beta_FS*gamma_FS <= 1, relatively high velocity nonrelativistic regime
        1 < beta_FS*gamma_FS, relativistic regime
        I rewrote the code to use numpy vectorisation techniques instead of for loops following Om's suggestion.
        This should make things faster.
        """
        
        R_vals=self.AGN_disk.tho.R/ct.pc
        rho_AGN_vals=self.AGN_disk.tho.rho*ct.SI_to_gcm3
        H_AGN_vals=self.AGN_disk.tho.h/ct.pc
        
        self.L_j_vals = self.L_j(rho_AGN_vals, H_AGN_vals, R_vals)
        self.L_tilde_vals = self.L_tilde(rho_AGN_vals, H_AGN_vals, self.L_j_vals)
            
        self.beta_vals = self.beta_func(self.L_tilde_vals)
        self.gamma_vals = self.gamma_from_beta(self.beta_vals)
        self.beta_gamma_vals = self.beta_vals*self.gamma_vals
        
            
        self.t_delay_vals = self.t_delay(H_AGN_vals, self.beta_vals)
        self.t_duration_vals = self.t_duration(self.beta_vals, rho_AGN_vals)
        self.L_sh_vals = self.L_sh(H_AGN_vals, rho_AGN_vals, self.beta_vals, R_vals)
        self.L_breakout_vals = self.L_breakout(H_AGN_vals, rho_AGN_vals, self.beta_vals, R_vals)
        
        T_breakout_low_vals = self.T_breakout_low(self.beta_vals, rho_AGN_vals)
        T_breakout_high_vals = self.T_breakout_high(self.beta_vals, rho_AGN_vals)

        
        self.T_breakout_vals = np.where(self.beta_gamma_vals<0.03, T_breakout_low_vals, T_breakout_high_vals)

        print("Calculated L_j, L_tilde, beta, gamma, beta_gamma, t_delay, t_duration, L_sh, L_breakout and T_breakout")

    def spectral_distribution(self, nu_min, nu_max, dist_type, r_val):
        self.idx_val = self.find_nearest(self.AGN_disk.tho.R/ct.pc, r_val)
        self.nu_vals = np.logspace(np.log10(nu_min), np.log10(nu_max), 100000) ## This is the frequency range where we can observe the peak of the spectrum in the data
        if dist_type=='wien':
            self.sed_wien_vals = self.sed_wien(self.nu_vals, self.T_breakout_vals[self.idx_val], self.L_breakout_vals[self.idx_val]) ## Wien's spectrum distribution at breakout temperature and breakout Luminosity
        elif dist_type=='blackbody':
            self.sed_bb_vals = self.sed_bb(self.nu_vals, self.T_breakout_vals[self.idx_val], self.L_breakout_vals[self.idx_val]) ## Wien's spectrum distribution at breakout temperature and breakout Luminosity

    def plot_em_props(self, r_val):
        R_vals = self.AGN_disk.tho.R/ct.pc
        self.idx_val = self.find_nearest(self.AGN_disk.tho.R/ct.pc, r_val)
                
        fig, ax = plt.subplots(2, 3, figsize=(18, 12))

        ax[0][0].plot(R_vals, self.t_delay_vals*365*24*60*60, '-', color='tab:blue')
        ax[0][0].axhline(y=18*24*60*60, color='red', linestyle='--', lw=1.0)
        ax[0][0].plot(R_vals[self.idx_val], self.t_delay_vals[self.idx_val]*365*24*60*60, '.', ms=12, color='black')
        ax[0][0].set_xscale('log')
        ax[0][0].set_yscale('log')
        ax[0][0].set_xlim([1e-3, 10])
        ax[0][0].set_ylim([1, 1e8])
        ax[0][0].grid(True)
        ax[0][0].set_xlabel(r"R [pc]", fontsize=10)
        ax[0][0].set_ylabel(r"$\mathrm{t}_{delay} [\mathrm{s}]$", fontsize=10)
        
        
        ax[0][1].plot(R_vals, self.t_duration_vals*365*24*60*60, '-', color='tab:blue', label=r"$\mathrm{t}_{\mathrm{duration}}$")
        ax[0][1].plot(R_vals[self.idx_val], self.t_duration_vals[self.idx_val]*365*24*60*60, '.', ms=12, color='black')
        ax[0][1].axhline(y=28*24*60*60, color='red', linestyle='--', lw=1.0)
        ax[0][1].set_xscale('log')
        ax[0][1].set_yscale('log')
        ax[0][1].set_xlim([1e-3, 10])
        ax[0][1].set_ylim([1.5e-4, 1e8])
        ax[0][1].grid(True)
        ax[0][1].legend()
        ax[0][1].set_xlabel(r"R [pc]", fontsize=10)
        ax[0][1].set_ylabel(r"$\mathrm{t}_{\mathrm{duration}} [\mathrm{s}]$", fontsize=10)
        
        
        #ax[0][2].plot(R_vals, self.L_breakout_vals, '-', color='tab:blue', label=r"$\mathrm{L}_{\mathrm{breakout}}$")
        ax[0][2].plot(R_vals, self.L_sh_vals, '-', color='tab:blue', label=r"$\mathrm{L}_{\mathrm{kin}}$")
        ax[0][2].plot(R_vals, self.L_j_vals, '--', color='black', label=r"$\mathrm{L}_{\mathrm{jet}}$")
        #ax[0][2].plot(R_vals, self.L_nonthermal_vals, '--', color='grey', label=r"$\mathrm{L}_{\mathrm{nonthermal}}$")
        ax[0][2].set_xscale('log')
        ax[0][2].set_yscale('log')
        ax[0][2].set_xlim([1e-3, 10])
        ax[0][2].set_ylim([1e42, 1e51])
        ax[0][2].grid(True)
        ax[0][2].legend()
        ax[0][2].set_xlabel(r"R [pc]", fontsize=10)
        ax[0][2].set_ylabel(r"$\mathrm{L} [\mathrm{erg/s}]$", fontsize=10)        

        ax[1][1].axis('off')
        #ax[1][0].plot(R_vals, self.gamma_max_prime_vals, '-', color='tab:blue', label=r"$\mathrm{\gamma}_{\mathrm{max}}$")
        #ax[1][0].plot(R_vals, self.gamma_min_prime_vals, '--', color='black', label=r"$\mathrm{\gamma}_{\mathrm{m}}$")
        #ax[1][0].plot(R_vals, gamma_a_prime_vals, '--', color='grey', label=r"$\mathrm{\gamma}_{\mathrm{a}}$")
        #ax[1][0].set_xscale('log')
        #ax[1][0].set_yscale('log')
        #ax[1][0].set_xlim([1e-3, 10])
        #ax[1][0].set_ylim([1, 2.5e6])
        #ax[1][0].grid(True)
        #ax[1][0].legend()
        #ax[1][0].set_xlabel(r"R [pc]", fontsize=10)
        
        

        ax[1][0].plot(R_vals, self.T_breakout_vals, '-', color='tab:blue', label=r"$\mathrm{T}_{\mathrm{breakout}}$")
        #ax[1][0].plot(R_vals, self.h*nu_min_vals/kb, '-', color='black', label=r"$\mathrm{h \nu_{min}/k_b}$")
        ax[1][0].set_xscale('log')
        ax[1][0].set_yscale('log')
        ax[1][0].set_xlim([1e-3, 10])
        ax[1][0].set_ylim([1e-4, 1e10])
        ax[1][0].grid(True)
        ax[1][0].legend()
        ax[1][0].set_xlabel(r"R [pc]", fontsize=10)
        ax[1][0].set_ylabel(r"$\mathrm{T}~[\mathrm{K}]$", fontsize=10)

        ax[1][2].plot(R_vals, self.beta_vals, '-', color='tab:blue', label=r"$\mathrm{\beta}_{FS}$")
        ax[1][2].plot(R_vals, self.gamma_vals, '--', color='tab:grey', label=r"$\mathrm{\gamma}_{FS}$")
        ax[1][2].plot(R_vals, self.beta_gamma_vals, '-.', color='black', label=r"$\mathrm{\beta}_{FS} \mathrm{\gamma}_{FS}$")
        ax[1][2].set_xscale('log')
        ax[1][2].set_yscale('log')
        ax[1][2].set_xlim([1e-3, 10])
        ax[1][2].set_ylim([1e-1, 10])
        ax[1][2].grid(True)
        ax[1][2].legend()
        ax[1][2].set_xlabel(r"R [pc]", fontsize=10)
        
        fig.tight_layout()
        plt.show()

    def plot_sed(self, luminosity_distance):
        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(self.nu_vals, self.sed_wien_vals/(4*np.pi*(luminosity_distance*3.086e18)**2)/10**12, '-', color='maroon', label=r"$\mathrm{Thermal}$")
        #ax.plot(nu_vals1, nu_L_nu_nonthermal_vals/(4*np.pi*(5300*3.086e18)**2)/10**12, '-', color='black', label=r"$\mathrm{Non~Thermal}$")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim([1e10, 1e22])
        ax.grid(True)
        ax.legend()
        ax.set_xlabel(r"$\mathrm{\nu}~[\mathrm{Hz}]$", fontsize=10)
        ax.set_ylabel(r"$\mathrm{\nu F_{\nu}}~[\mathrm{erg/cm^2/s}]$", fontsize=10)
        
        plt.show()
