import os
import operator
import sklearn, sklearn.tree
import numpy as np
import astropy as astr
import matplotlib.pyplot as plt
import pandas as pd
import lightkurve as lk
import astroquery as aq
import astropy.units as u
from astropy.io import fits
from sklearn import tree
import joblib as jl 
import pickle 

import warnings
from lightkurve import LightkurveWarning
warnings.filterwarnings("ignore", category=LightkurveWarning) 


def get_gaia_table(ra,dec,radius = 5*u.arcmin):
    from astroquery.gaia import Gaia
    Gaia.ROW_LIMIT = 5
    obj = astr.coordinates.SkyCoord(ra*u.deg,dec*u.deg)
    return Gaia.query_object_async(obj,width=radius,height=radius)[0]

def lomb(time, flux):
    from astropy.timeseries import lombscargle

    ls = lombscargle.LombScargle(time,flux,normalization='psd')
    freq = ls.autofrequency()
    power = ls.autopower()
    # freq = freq.value
    power = power[1]
    nyq_freq = 0.5/np.median(np.diff(time))

    w = freq < nyq_freq
    freq = freq[w]
    power = np.sqrt(power[w])
    
    return freq, power, ls

def find_peaks(freq, power, num=5, nstd=2):
    
    diff = np.diff(power)
    freq = freq[1:]
    power = power[1:]
    level = np.std(power)*nstd + np.median(power)
    
    peaks_freq = []
    peaks_power = []
    diff_last = diff[0]
    peak_found = False

    temp_freq = 0
    temp_power = 0
    
    for i in range(len(freq)):
        if power[i] < level:
            peak_found = False
            continue
            
        if diff[i] >= 0:
            temp_freq = freq[i]
            temp_power = power[i]
            peak_found = False
            diff_last = diff[i]
            continue 
            
        if (diff[i] < 0) and not peak_found:
            peaks_freq.append(temp_freq)
            peaks_power.append(temp_power)
            peak_found = True
    
    peaks_freq = np.asarray(peaks_freq)
    peaks_power = np.asarray(peaks_power)
    w = peaks_power.argsort()[::-1]
    
    peaks_freq = peaks_freq[w]
    peaks_power = peaks_power[w]

    return peaks_freq[:num], peaks_power[:num]


def get_TESS_gaia(TIC_ID):
    
    lc = lk.search_lightcurve('TIC {}'.format(TIC_ID), mission='TESS')
    lc.exptime.sort()

    lc = lc[0].download()
    lc = lc.remove_nans()
    
    ra = lc.hdu[0].header['RA_OBJ']
    dec = lc.hdu[0].header['DEC_OBJ']
    
    time = lc.time.value
    flux = lc.flux.value
    mag = -2.5*np.log10(flux)
    
    freq, power, ls = lomb(time, mag)    
    
    gaia = get_gaia_table(ra,dec,radius=2*u.arcmin)    

    return time, flux, mag, ls, gaia, freq, power


def get_star_properties(time, flux, ls, gaia, freq, power):
    
#     lc = lk.search_lightcurve('TIC {}'.format(TIC_ID), mission='TESS')
#     lc.exptime.sort()

#     lc = lc[0].download()
#     lc = lc.remove_nans()
    
#     ra = lc.hdu[0].header['RA_OBJ']
#     dec = lc.hdu[0].header['DEC_OBJ']
    
#     time = lc.time.value
#     flux = lc.flux.value
#     mag = -2.5*np.log10(flux)
    
#     freq, power, ls = lomb(time, mag)    

    peak_freq, peak_power = find_peaks(freq, power, num=3, nstd=1)
    peak_period = 1.0 / peak_freq
    
    peak_amp = []
    for i in range(3):
        y_mod = ls.model(t = time, frequency = peak_freq[i])
        peak_amp.append((y_mod.max() - y_mod.min())/2)

    B_V = gaia['bp_g']
    B_R = gaia['bp_rp']
    V_R = gaia['g_rp']
    Teff = gaia['teff_val']

    return B_V, B_R, V_R, Teff, peak_period, peak_freq, peak_power, peak_amp

def plot_lc(time, flux, ls, TIC_ID):
    
    freq = ls.autofrequency()
    power = ls.autopower(normalization='psd')[1]

    nyq_freq = 0.5/np.median(np.diff(time))

    w = freq < nyq_freq
    freq = freq[w]
    power = np.sqrt(power[w])    
    
    fig, ax = plt.subplots(nrows=2,ncols=1, figsize=(8,12))
    
    ax[0].plot(time, flux,'k-')
    ax[0].set_xlabel('BJD - 2457000',fontsize=16)
    ax[0].set_ylabel('Flux [counts]',fontsize=16)

    ax[1].plot(freq, power,'k-')
    ax[1].set_xlabel('Frequency [$day^{-1}$]',fontsize=16)
    ax[1].set_ylabel('Power [counts]',fontsize=16)

    plt.tight_layout()
    plt.savefig('lightcurve_plots/{}.png'.format(TIC_ID),facecolor='white')
    
    return


def get_feature_data(filename):
    all_obj = pd.read_csv(filename)
    
    feature_names = ['B-V', 'B-R', 'V-R', 'Teff', 
                     'period0', 'period1', 'period1',
                      'power0', 'power1', 'power2',
                      'amp0', 'amp1', 'amp2']
    
    label_names = ['ACV', 'BCEP', 'SPB']

    if os.path.exists('./features.npy') & os.path.exists('./classes.npy'):
        features = np.load('./features.npy', allow_pickle=True)
        classes = np.load('./classes.npy', allow_pickle=True)
        return features, classes


    features = np.zeros((len(all_obj), len(feature_names)))
    cls = []

    for i in all_obj.index:
        obj = all_obj.iloc[i]        

        TIC_ID = obj['TIC']
        
        print('####################################################')
        print('Getting data from TIC {}'.format(TIC_ID))        
        print('\n \n \n')
        
        if os.path.exists('./star_data/{}.dat'.format(TIC_ID)):
            time, flux, mag, ls, gaia, freq, power = astr.io.misc.fnunpickle('./star_data/{}.dat'.format(TIC_ID))
        else:
            time, flux, mag, ls, gaia, freq, power  = get_TESS_gaia(TIC_ID)

            # jl.dump([lc, ls, gaia, freq, power],'./star_data/{}.dat'.format(TIC_ID))        

            with open('./star_data/{}.dat'.format(TIC_ID),'wb') as f:
                astr.io.misc.fnpickle([time, flux, mag, ls, gaia, freq, power], f)

        print('Making a plot \n \n')
        if os.path.exists('lightcurve_plots/{}.png'.format(TIC_ID)):
            continue
        else:
            plot_lc(time, flux, ls, TIC_ID)
        
        print('Getting Stellar Parameters \n \n')
        if os.path.exists('./star_parm/{}.dat'.format(TIC_ID)):
            params = astr.io.misc.fnunpickle('./star_parm/{}.dat'.format(TIC_ID))
        else:
            params = get_star_properties(time, flux, ls, gaia, freq, power)
            # jl.dump(params,'./star_parm/{}.dat'.format(TIC_ID))  

            with open('./star_parm/{}.dat'.format(TIC_ID),'wb') as f:
                astr.io.misc.fnpickle(params, f)

        B_V, B_R, V_R, Teff, peak_period, peak_freq, peak_power, peak_amp = params
        
        features[i,0] = B_V
        features[i,1] = B_R
        features[i,2] = V_R
        features[i,3] = Teff
        features[i,4] = peak_period[0]
        features[i,5] = peak_period[1]
        features[i,6] = peak_period[2]
        features[i,7] = peak_power[0]
        features[i,8] = peak_power[1]
        features[i,9] = peak_power[2]
        features[i,10] = peak_amp[0]
        features[i,11] = peak_amp[1]
        features[i,12] = peak_amp[2]
        
        cls.append(obj['Type'])
    
    print('Saving features \n')
    np.save('features.npy', features, allow_pickle=True)
    np.save('classes.npy', cls, allow_pickle=True)
        
    astr.io.misc.fnpickle(features,'features.dat')  
    astr.io.misc.fnpickle(cls,'classes.dat')  
        
    return features, cls


print('start')
    
if True:
    
    feature_names = ['B-V', 'B-R', 'V-R', 'Teff', 
                     'period0', 'period1', 'period1',
                      'power0', 'power1', 'power2',
                      'amp0', 'amp1', 'amp2']
    
    label_names = ['ACV', 'BCEP', 'SPB']
    
    filename = 'obj_list_trim.csv'

    print('Starting Job \n \n \n ')


    print('Getting Features and Classes for all Stars')
    features, classes = get_feature_data(filename)
    targets = np.zeros(len(classes))
    
    for i, name in enumerate(label_names):
        w = classes == name
        targets[w] = i
    
    print('Making Decision Tree')

    decision_tree = tree.DecisionTreeClassifier(random_state=0, max_depth=5)
    decision_tree = decision_tree.fit(features, targets)
    r = export_text(decision_tree, feature_names=feature_names)
    
    print('Saving Decision Tree')

    dump(decision_tree, 'dt_model.dat')
