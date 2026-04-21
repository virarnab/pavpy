import pandas as pd
import numpy as np

from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time
import astropy.units as units
from astroquery.vizier import Vizier
from dustmaps.bayestar import BayestarQuery, BayestarWebQuery
from scipy.interpolate import griddata
import importlib_resources
from functools import lru_cache
from astroquery.simbad import Simbad
from astroquery.gaia import Gaia



bayestar = BayestarQuery(version='bayestar2019')

def estimate_theta_vk(vmag,kmag):
    # Calculate angular diameter from Boyajian et al. (2014) relation
    logtheta = 0.26382*(vmag-kmag) + 0.53246 - 0.2 * vmag
    
    return np.power(10,logtheta)

@lru_cache(None)
def get_coords(star):
    if star.endswith("150680"):
        star = "Gaia DR3 1312665361415345920"
    sim = Simbad()
    sim.add_votable_fields('ids')

    res = sim.query_object(star)
    if res is None:
        raise ValueError(f"{star} not found in SIMBAD")

    ids = res['IDS'][0]
    if isinstance(ids, bytes):
        ids = ids.decode()

    gaia_id2 = None
    gaia_id3 = None
    for item in ids.split('|'):
        if "Gaia DR3" in item:
            gaia_id3 = item.split()[-1]
        elif "Gaia DR2" in item:
            gaia_id2 = item.split()[-1]
            
    gaia_id = gaia_id3 if gaia_id3 is not None else gaia_id2

    if gaia_id is None:
        raise ValueError(f"No Gaia DR3 ID for {star}")

    query = f"""
    SELECT ra, dec, parallax
    FROM gaiadr3.gaia_source
    WHERE source_id = {gaia_id}
    """

    job = Gaia.launch_job(query)
    tab = job.get_results()

    if len(tab) == 0:
        raise ValueError(f"Gaia query failed for {star}")

    row = tab[0]

    return SkyCoord(
        ra=row["ra"] * units.deg,
        dec=row["dec"] * units.deg,
        distance=(1000.0 / row["parallax"]) * units.pc,
        frame="icrs"
    ).transform_to("galactic")

def get_extinction(coords):
    return bayestar(coords, mode='median')

    
def deredden(vmag,kmag,ebv):
    # Calculate dereddened v and k magnitudes assuming O'Donnell (1994) law
    Rv = 3.1
    av = Rv * ebv
    ak = (0.148 - 0.099/Rv) * av
    v = vmag - av
    k = kmag - ak
    
    return v,k

@lru_cache(None)
def get_vkmags(star):
    if star.endswith("150680"):
        star = "Gaia DR3 1312665361415345920"
    # Get V and K mags for a given star, and convert Tycho V mag to Johnson V
    try:
        v = Vizier(columns=["BTmag","VTmag"], catalog="I/259/tyc2")
        result = v.query_object(star)
        bt_minus_vt = result[0]['BTmag'][0]-result[0]['VTmag'][0]
        
        tycho = pd.read_fwf(importlib_resources.files('pavpy') / 'Tycho_BV_Bessel2000.dat',skiprows=2, names=['BtVt','VVt','dBV','VHp'])
        
        vmag = np.interp(bt_minus_vt, tycho.BtVt, tycho.VVt) + result[0]['VTmag'][0]
    
    except:    
        sim = Simbad()
        sim.add_votable_fields('flux(V)')
        vmag = sim.query_object(star)['FLUX_V'].value[0]
        
    try:
    
        v = Vizier(columns=["Kmag"], catalog="II/246/out")
        result = v.query_object(star)
        
        kmag = np.min(result[0]['Kmag'])
    
    except:
        sim = Simbad()
        sim.add_votable_fields('flux(K)')
        kmag = sim.query_object(star)['FLUX_K'].value[0]

    return vmag,kmag

def get_uv(df):
    #Calculate u, v, projected baseline and position angle for each observation in a dataframe

    #Location of the observatory
    chara = EarthLocation.of_site('CHARA')
    lat = chara.lat.radian #Convert latitutde to radians
        
    #Relative location of telescopes
    tel_locations = pd.read_csv(importlib_resources.files('pavpy') / 'tel_locations.txt', delimiter=' ', names=['telname','e_offset','n_offset','z_offset'])
    tel_locations = tel_locations.set_index('telname')
    
    summary = df.groupby(df.File).first()
    
    summary = summary.assign(ra = summary.filter(['Star']).applymap(lambda x : SkyCoord.from_name(x).ra.radian),
                             dec = summary.filter(['Star']).applymap(lambda x : SkyCoord.from_name(x).dec.radian),
                             lst = Time(summary.JD + 2451545, format='jd', location=chara).sidereal_time('mean').radian)
    summary = summary.assign(ha = summary.lst - summary.ra,
                             e_offset = summary.filter(['T1']).applymap(lambda x : tel_locations.loc[x].e_offset).values - 
                                           summary.filter(['T2']).applymap(lambda x : tel_locations.loc[x].e_offset).values,
                             n_offset = summary.filter(['T1']).applymap(lambda x : tel_locations.loc[x].n_offset).values - 
                                           summary.filter(['T2']).applymap(lambda x : tel_locations.loc[x].n_offset).values,
                             z_offset = summary.filter(['T1']).applymap(lambda x : tel_locations.loc[x].z_offset).values - 
                                           summary.filter(['T2']).applymap(lambda x : tel_locations.loc[x].z_offset).values)
    summary = summary.assign(bx = -np.sin(lat)*summary.n_offset + np.cos(lat)*summary.z_offset,
                             by = summary.e_offset,
                             bz = np.cos(lat)*summary.n_offset + np.sin(lat)*summary.z_offset)
    summary = summary.assign(u = np.sin(summary.ha)*summary.bx + np.cos(summary.ha)*summary.by,
                             v = -np.sin(summary.dec)*np.cos(summary.ha)*summary.bx + np.sin(summary.dec)*np.sin(summary.ha)*summary.by 
                                    + np.cos(summary.dec)*summary.bz)
                             # w = np.cos(summary.dec)*np.cos(summary.ha)*summary.bx - np.cos(summary.dec)*np.sin(summary.ha)*by + np.sin(summary.dec)*summary.bz
    summary = summary.assign(bl = np.sqrt(summary.u**2 + summary.v**2),
                             pa = np.arctan2(summary.v,summary.u)*180/np.pi
                            )
    
    # return summary
    return summary.filter(['u','v','bl','pa'])

def get_diams(df, fractional_uncertainty):

    vk_diams = []

    for star in df.Star.unique():
        # Get V and K mags from Tycho-2 and 2MASS. Convert Tycho V to Johnson V
        vmag, kmag = get_vkmags(star)
        # Get Gaia coordinates and parallax
        coords = get_coords(star)
        # Get E(B-V) from Green et al. (2015) dust map
        ebv = get_extinction(coords)
        # De-redden V and K magntiudes
        vmag,kmag = deredden(vmag, kmag, ebv)
        # Calculate LD diamter from V-K relation of Boyajian et al. 2014
        thetaLD = estimate_theta_vk(vmag,kmag)

        vk_diams.append(thetaLD)

    uncertainty = fractional_uncertainty*np.asarray(vk_diams)

    return pd.DataFrame({'star': df.Star.unique(), 'diameter': vk_diams, 'uncertainty': uncertainty, 'sample_diameter': vk_diams}).set_index('star')

def get_ldcs(teff, logg, wavelengths):

    model_coeffs = pd.read_json(importlib_resources.files('pavpy') / 'stagger_4term_coeffs.json')

    coeffs = []

    for wavelength in wavelengths:
        coeff = griddata(model_coeffs.filter(['teff','logg']).values, 
            np.stack(np.concatenate(model_coeffs.filter([str(wavelength)]).values)), 
            np.array([teff,logg]), 
            method='cubic',
            rescale=True)
        coeffs.append(coeff[0])

    return pd.DataFrame(coeffs, index= wavelengths, columns = ['a1','a2','a3','a4'])


def wtmn(x, sigx):

    return (x*sigx**-2).agg('sum')/(sigx**-2).agg('sum')

def randomcorr(covmat):

    evals, evects = np.linalg.eigh(covmat)
    lin_comb = evects @ np.diag(np.sqrt(evals))
    random_vector = lin_comb @ np.random.normal(size = evals.shape)
    
    return random_vector
