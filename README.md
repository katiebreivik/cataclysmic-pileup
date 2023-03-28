# cataclysmic-pileup

The parameters are:
nCV: the size of the CV population
max_distance: the maximum distance of the population where the positions are distributed spherically symmetrically but with an r^2 distance (this follows the local single WD population)
bump_width: the width of the pileup in the orbital period distribution in seconds centered at log(forb/Hz) = -3.68 
mu_m1: center of a gaussian for the accretor mass in Msun
sigma_m1: dispersion of gaussian for accretor mass in Msun
mu_m2: center of a gaussian for the donor mass in Msun
sigma_m2: dispersion of gaussian for donor mass in Msun

I've set some defaults so that the script will run without any input, but if you want to change the values you can do so on the command line like:
>>> python CV_pop.py --nCV 100 --max-distance 300 --bump-width 50 --mu-m1 0.7 --sigma-m1 0.05 --mu-m2 0.1 --sigma-m2 0.01

The script writes a simple text file containing the masses, GW frequency, inclination, and galactocentric cartesian coordinates (with a header), so should be pretty easy to work with.
