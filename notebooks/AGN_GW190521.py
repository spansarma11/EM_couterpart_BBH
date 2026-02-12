from EM_CP_class import AGN_model, EM_model
import pagn.constants as ct

import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import lal
from ligo.skymap.io import read_sky_map
from astropy.io import fits

from ligo.skymap.postprocess import find_greedy_credible_levels
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

from ligo.skymap.io import fits as ligofits

from astropy.cosmology import Planck18

import ligo.skymap.plot

import pickle

fits_file = "/Users/jhuma/Lahar_work/multimessenger_project/data_store/GW_skymap/IGWN-GWTC2p1-v2-GW190521_030229_PEDataRelease_cosmo_reweight_C01:IMRPhenomXPHM.fits"
contour_levels = [90]

skymap, metadata = read_sky_map(fits_file, nest=None, distances=True)
cls = 100 * find_greedy_credible_levels(skymap)

from ligo.skymap.distance import marginal_pdf, marginal_ppf

d_low, d_high = marginal_ppf([0.05, 0.95], skymap[0], skymap[1], skymap[2], skymap[3])

nside_190521 = hp.get_nside(skymap)
pix90_190521 = np.where(cls <= 90)[0]

fits_file_open = fits.open("/Users/jhuma/Lahar_work/multimessenger_project/data_store/AGN_catalog/dr16q_prop_May01_2024.fits")
data_prop = fits_file_open[1].data
fits_file_open.close()

mask_z = data_prop['Z_DR16Q'] < -1
data_prop = data_prop[~mask_z]
pix_val_AGN = hp.ang2pix(512, np.radians(90 - data_prop['DEC']), np.radians(data_prop['RA']), nest=True)
L_d_val_AGN = np.asarray(Planck18.luminosity_distance(data_prop['Z_DR16Q']))
mask_pix = np.isin(pix_val_AGN, pix90_190521)
mask_L_d = (np.isfinite(L_d_val_AGN) & (L_d_val_AGN >= d_low) & (L_d_val_AGN <= d_high))
mask_BBH = data_prop['LOGMBH']>1
mask_both = mask_pix & mask_L_d & mask_BBH
idx = np.where(mask_both)[0]

data_prop_mass = data_prop['LOGMBH']

#print(data_prop['LOGMBH'][438660], data_prop['SDSS_NAME'][438660])

del data_prop, mask_both, mask_BBH, mask_L_d, mask_pix, L_d_val_AGN, pix_val_AGN, mask_z, pix90_190521, nside_190521, skymap, metadata, cls

AGN_store_dict = {}
for idx_val in idx:
    AGN_props = AGN_model(msmbh = 10**data_prop_mass[idx_val], epsilon=1e-3, m=0.5, xi=1, Mdot_out=None, Rout=None, Rin=None, 
                 opacity="combined", disk_model='Thompson', seed=100)
    try:
        AGN_props.solve_AGN_disk_prop()
        AGN_store_dict[idx_val] = AGN_props
        
    except Exception as e:
        print(f"[WARNING] AGN model failed for idx={idx_val}: {e}")
        continue
    print()

with open('/Users/jhuma/Lahar_work/multimessenger_project/data_store/AGN_disk_data/AGN_mod_store_new.pkl', 'rb') as f:
    AGN_data = pickle.load(f)

store_em_props = {}

for key in AGN_data.keys():
    em_props = EM_model(msmbh=10**data_prop_mass[key], m_rem=150, eta_j=0.5, alpha_AGN=0.1, f_acc=15, theta_0=0.3, mag_ampl=0.1,
                 elec_frac=0.3, p=2.5, AGN_model=AGN_data[key], calc_disk_prop=False, disk_prop_vars=None)
    em_props.emission_properties_retrieve()
    store_em_props[key] = em_props

with open('/Users/jhuma/Lahar_work/multimessenger_project/data_store/AGN_disk_data/GW_EM_prop_store_new.pkl', 'wb') as f:
    pickle.dump(store_em_props, f)
