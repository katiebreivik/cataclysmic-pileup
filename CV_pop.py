import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import argparse
from astropy.coordinates import SkyCoord
from scipy.interpolate import interp1d
import pandas as pd
import scipy.stats as stats
import sys

def parse_commandline():
    """Parse the command line arguments"""
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-distance", help="What is the maximum distance of the population in parsecs?", required=False, type=float, default=300.0)
    parser.add_argument("--mu-m1", help="What is the mean of a Gaussian for the accretor mass?", required=False, type=float, default=0.7)
    parser.add_argument("--sigma-m1", help="What is the dispersion of a Gaussian for the accretor mass?", required=False, type=float, default=0.05)
    parser.add_argument("--sigma-m2", help="What is the dispersion of a Gaussian for the donor mass?", required=False, type=float, default=0.01)
    
    args = parser.parse_args()
    return args

#def sample_porb_from_Knigge_2011(nCV, log_t_cut=1):
#    # Since the data points are evenly spaced, we can just create a kde of the periods
#    # and they will be sampled according to how probable a source is to exist at that period
#    #load knigge data to sample orbital periods; enforce limits of distribution
#    dat = pd.read_csv('kniggeTable.csv')
#    porb_kde = stats.gaussian_kde(dat.loc[dat.logt > log_t_cut]['Per'])
#    porb = porb_kde.resample(nCV)[0]
#    porb[porb < min(dat.loc[dat.logt > log_t_cut]['Per'])] = np.random.uniform(min(dat.loc[dat.logt > log_t_cut]['Per']), 1.38, len(porb[porb < min(dat.loc[dat.logt > log_t_cut]['Per'])]))
#    porb[porb > max(dat.loc[dat.logt > log_t_cut]['Per'])] = np.random.uniform(2.0, max(dat.loc[dat.logt > log_t_cut]['Per']), len(porb[porb > max(dat.loc[dat.logt > log_t_cut]['Per'])]))
#    
#    return porb
#
def sample_porb_from_Pala_2020(nCV):
    dat = pd.read_hdf('Pala_2020_dat_combo.h5', key='dat')
    #porb_kde = stats.gaussian_kde(dat.porb.values / 60, bw_method=0.2) # convert minutes to hours

    tt = dat.porb.values[dat.porb.values/60 <6] /60
    porb_kde = stats.gaussian_kde(tt, bw_method=0.3) # convert minutes to hours

    porb = porb_kde.resample(nCV)[0]
    
    porb[porb < min(dat.porb/60)] = np.random.uniform( min(dat.porb/60), 1.38, len(porb[porb < min(dat.porb/60)]))
    return porb


def sample_position_from_Pala_2020(rho_0=4.8e-6, h=280, dist_max=600):
    # the Pala+2020 model is just a cylinder with an exponential decay in z
    # this means we can assign x, y randomly in a circle and z with the exponential decay
    #N_sample_positive = rho_0 * h * (1 - np.exp(-(dist_max/2) / h))

    N_sample_positive = rho_0 * np.pi *dist_max**2 * h * (1 - np.exp(-((dist_max)/h)))

    N_sample_total = 2 * N_sample_positive
    #print(N_sample_total)
    #print(N_sample_total)

    # we will do a rejection sample to get the correct number of sources.
    # we will sample 5 times the number we need and then downsample
    
    extraFactor = 5



    # determine if we add the remainder of the decimal as a source
    prob_extra = np.random.uniform(0, 1)
    remainder = N_sample_total - int(N_sample_total)
    if prob_extra < remainder:
        N_sample_total = int(N_sample_total) + 1
    else:
        N_sample_total = int(N_sample_total)
    
    # uniform in a disk around the sun 
    # kb is lazy and will do a rejection sample
    x = np.random.uniform(-dist_max, dist_max, extraFactor*N_sample_total)
    y = np.random.uniform(-dist_max, dist_max, extraFactor*N_sample_total)
    r = np.sqrt(x**2 + y**2)
    ind_keep, = np.where(r < dist_max)
    x = x[:N_sample_total]
    y = y[:N_sample_total]
    
    z = np.random.exponential(scale=h, size=5*N_sample_total)
    z = z[z<dist_max]
    z = z[:N_sample_total]
    plane_sample = np.random.uniform(0, 1, N_sample_total)
    z[plane_sample < 0.5] = -z[plane_sample < 0.5]
    
    # now place the final volume limit
    ind_volume_limit, = np.where(np.sqrt(x**2 + y**2 + z**2) < dist_max)
    x = x[ind_volume_limit]
    y = y[ind_volume_limit]
    z = z[ind_volume_limit]

    return x/1000, y/1000, z/1000
    

def calculate_m2_from_porb(porb):
    # calculate the m2 mass from the orbital period by interpolating 
    # to find the time at P and m2 at that time
    dat = pd.read_csv('kniggeTable.csv')
    t_interp = interp1d(dat['Per'], dat['logt'], fill_value = 'extrapolate')
    t_bin = t_interp(porb)
    m2_interp = interp1d(dat['logt'], dat['M2'], fill_value = 'extrapolate')
    m2 = m2_interp(t_bin)
    m2[porb < min(dat["Per"])] = min(m2)
    return m2
    
def get_Pala_sample(mu_m1, sigma_m1, sigma_m2):
    pala2020 = pd.read_hdf('Pala_2020_dat_combo.h5', key='dat')
    c = SkyCoord(pala2020.ra.values * u.deg, pala2020.dec.values * u.deg, distance=pala2020.distance.values * u.pc)
    c = c.transform_to(frame='galactic')
    x = c.cartesian.x.value
    y = c.cartesian.y.value
    z = c.cartesian.z.value
    porb = pala2020['porb'].values / 60 #convert mins to hours
    m2 = calculate_m2_from_porb(porb)
    m2_err = np.random.normal(loc=0, scale=sigma_m2, size=len(porb))
    m2 = m2 + m2_err
    m1 = np.random.normal(mu_m1, sigma_m1, len(porb))
    inclination = np.arccos(np.random.uniform(-1, 1, len(porb)))
    return m1, m2, porb, x/1000, y/1000, z/1000, inclination





if __name__ == '__main__':
    
    # Parse the command line
    args = parse_commandline()
    
    # first sample the population
    # sample the population positions and size based on Pala+2020 distribution & space density
    x, y, z = sample_position_from_Pala_2020(rho_0=4.8e-6, h=280, dist_max=args.max_distance)
    
    d = np.sqrt(x**2 + y**2 + z**2) * u.kpc
    ind_check, = np.where(d<0.15*u.kpc)
    while len(ind_check) < 54:
        print("We need 54 sources within 150pc. Generating new population!")
        x, y, z = sample_position_from_Pala_2020(rho_0=4.8e-6, h=280, dist_max=args.max_distance)
        d = np.sqrt(x**2 + y**2 + z**2) * u.kpc
        ind_check, = np.where(d<0.15*u.kpc)



    # assign a random inclination
    inclination = np.arccos(np.random.uniform(-1, 1, len(x)))
    
    # sample the primary mass with normal distribution supplied by user
    m1 = np.random.normal(loc=args.mu_m1, scale=args.sigma_m1, size=len(x))
    
    # get the orbital periods by sampling from the Pala+2020 table
    porb = sample_porb_from_Pala_2020(nCV=len(x))
    f_gw = 2/(porb * 3600) # this is simple because the binaries are circular and porb is in hrs

    # get the matching donor mass from the Knigge+2011 table
    m2 = calculate_m2_from_porb(porb)
    m2_err = np.random.normal(loc=0, scale=args.sigma_m2, size=len(x))
    m2 = m2 + m2_err
    Pala_reassign = np.zeros(len(x))
    dat = np.vstack([m1, m2, f_gw, inclination, x, y, z, Pala_reassign]).T
    
    #print(x.shape)
    #print(Pala_reassign.shape)

    # next reassign some of the sources to match the Pala data exactly
    m1_P, m2_P, porb_P, x_P, y_P, z_P, inc_P = get_Pala_sample(args.mu_m1, args.sigma_m1, args.sigma_m2)

    #ind_150, = np.where(np.sqrt(x**2 + y**2 + z**2) < 0.150)
    
    d = np.sqrt(dat[:,4]**2 + dat[:,5]**2 + dat[:,6]**2) * u.kpc
    ind_150, = np.where(d<0.15*u.kpc)
    #print("ind_150", ind_150.shape)

    # Some haking required here. Pala sample is 42 sources, and we should fix the entire 150pc sample to have 54 sources.
    # So we need to randomly select 42 sources from the 150pc sample and replace with the Pala sample.
    # But we also need to make sure that we don't replace the same source twice.
    # We also need to ensure that there is a total of 54 sources in the 150pc sample.
    # Check how many sources are in the 150pc sample. If more than 54, then we need to randomly select 54 from the 150pc sample and delete the rest from dat
    # If less than 54, then we need to run the code again.

    if len(ind_150) > 54:
        print(f"We have {len(ind_150)} sources")
        print("We're good! More than 54 sources in 150pc sample. Deleting some sources.")
        ind_remove = np.random.choice(ind_150, len(ind_150)-54, replace=False)
        dat = np.delete(dat, ind_remove, axis=0)
        d = np.sqrt(dat[:,4]**2 + dat[:,5]**2 + dat[:,6]**2) * u.kpc
        ind_150, = np.where(d<0.15*u.kpc)
    elif len(ind_150) < 54: # this should never happen
        print("Less than 54 sources in 150pc sample. Run Again!") 
        sys.exit()

    ind_Pala = np.random.choice(ind_150, len(m2_P), replace=False)   
    dat[ind_150, 7] = 2*np.ones(len(ind_150))

    dat[ind_Pala, 0] = m1_P
    dat[ind_Pala, 1] = m2_P
    dat[ind_Pala, 2] = 2/(porb_P*3600)
    dat[ind_Pala, 3] = inc_P
    dat[ind_Pala, 4] = x_P
    dat[ind_Pala, 5] = y_P
    dat[ind_Pala, 6] = z_P
    dat[ind_Pala, 7] = np.ones(len(m1_P))
    
    ind, = np.where(dat[:,7] > 0)

    c = SkyCoord(dat[:, 4], dat[:, 5], dat[:, 6], unit=u.kpc, frame='galactic', representation_type='cartesian')
    
    c_gal = c.transform_to('galactocentric')
    
    dat[:, 4] = c_gal.x
    dat[:, 5] = c_gal.y
    dat[:, 6] = c_gal.z

    # save the data
    np.savetxt(f"dat_maxDistance_{int(args.max_distance)}.txt", dat, delimiter=',', header="m1[Msun], m2[Msun], f_gw[Hz], inclination[rad], x_gal[kpc], y_gal[kpc], z_gal[kpc], Pala_reassigned", fmt='%.10f')
