import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import argparse
from astropy.coordinates import SkyCoord


def parse_commandline():
    """Parse the command line arguments"""
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--nCV", help="How many CVs are in the local volume?", required=False, type=int, default=100)
    parser.add_argument("--max-distance", help="What is the maximum distance of the population in parsecs?", required=False, type=float, default=300.0)
    parser.add_argument("--bump-width", help="What is the width of the bump caused by the orbital period pileup in units of seconds?", required=False, type=float, default=100)
    parser.add_argument("--mu-m1", help="What is the mean of a Gaussian for the accretor mass?", required=False, type=float, default=0.7)
    parser.add_argument("--sigma-m1", help="What is the dispersion of a Gaussian for the accretor mass?", required=False, type=float, default=0.05)
    parser.add_argument("--mu-m2", help="What is the mean of a Gaussian for the donor mass?", required=False, type=float, default=0.1)
    parser.add_argument("--sigma-m2", help="What is the dispersion of a Gaussian for the donor mass?", required=False, type=float, default=0.01)
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    # Parse the command line
    args = parse_commandline()
    
    # initialize the arrays containing the CV properties
    m1 = np.random.normal(loc=args.mu_m1, scale=args.sigma_m1, size=args.nCV)
    m2 = np.random.normal(loc=args.mu_m2, scale=args.sigma_m2, size=args.nCV)
    porb_center = 5150 # Based on Fig 14 of Pala+2020
    porb = np.random.uniform(porb_center - args.bump_width, porb_center + args.bump_width, size=args.nCV)
    f_gw = 2/porb
    inclination = np.arccos(np.random.uniform(-1, 1, args.nCV))
    norm = 1/(args.max_distance**3/3)
    dist = (3 * np.random.uniform(0, 1, args.nCV) / norm)**(1./3.) / 1000 * u.kpc
    phi = np.random.uniform(0, 2 * np.pi, args.nCV)
    theta = np.arccos(np.random.uniform(-1, 1, args.nCV))
    
    x = dist * np.cos(phi) * np.sin(theta)
    y = dist * np.sin(phi) * np.sin(theta)
    z = dist * np.cos(theta)
    coord = SkyCoord(x, y, z, representation_type='cartesian', frame='barycentrictrueecliptic')
    coord_galactocentric = coord.transform_to('galactocentric')
    
    dat = np.vstack([m1, m2, f_gw, inclination, coord_galactocentric.x.value, coord_galactocentric.y.value, coord_galactocentric.z.value]).T
    np.savetxt(f"dat_nCV_{args.nCV}_maxDistanc_{int(args.max_distance)}.txt", dat, header="m1[Msun], m2[Msun], f_gw[Hz], inclination[rad], x_galcen[kpc], y_galcen[kpc], z_galcen[kpc]")
