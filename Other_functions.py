# ----------------------------------------------------------------------------
#
#  Copyright (C) 2015,2022,2023 by David D Turner, Joshua Gebauer, and Tyler Bell 
#  All Rights Reserved
#
#  This file is part of the "TROPoe" retrieval system.
#
#  TROPoe is free software developed while the authors were at NOAA, and is
#  intended to be free software.  It is made available WITHOUT ANY WARRANTY.
#  For more information, contact the authors.
#
# ----------------------------------------------------------------------------

import sys
import numpy as np
import scipy.io
import scipy.interpolate
import scipy.integrate
from datetime import datetime
from netCDF4 import Dataset

import Calcs_Conversions
import Output_Functions

################################################################################
# This file contains the following functions:
# matchtimes()
# smooth()
# predict_co2_concentration()
# find_wnum_idx()
# add_sys_error()
# trace_gas_profile()
# inflate_prior_covariance()
# interpolate_prior_covariance()
# compute_pblh()
# compute_sbi()
# compute_lcl()
# get_a2_pblh()
# covariance_matrix_decorrelate_level()
# get_aeri_bb_emis()
# aeri_recal()
# find_cloud()
# get_scat_properties()
# od2numdens()
# numdens2od()
# get_ir_cld_ods()
# cloud_reflectivity()
# radxfer()
# apodize_norton_beer()
# apodize_kaiser_bessel()
# apodizer()
# convolve_to_aeri()
# convolve_to_refir()
# convolve_to_irs()
# write_arm_sonde_file()
# radxfer_microwave()
# compute_cgamma()
# lcurve()
# compute_vres_from_akern()
# aeri_zerofill()
# fix_aeri_vlaser_mod()
# compute_extra_layers()
# test_monotonic()
# make_monotonic()
# irs_ffovc()              -- TODO -- add this bad boy
# change_irs_ffovc()       -- TODO -- add this bad boy
# fix_nonphysical_wv()
# decompose_irs_band_noise_inflation()
# calc_derived_indices()
# inflate_in_tropoe_uncertainty()
# do_mlev_cbh()
# condition_matrix()
################################################################################


################################################################################
# This function takes two arrays of julian times and compares the two to find where
# array two is equal to array one within a certain tolerance.
################################################################################

def matchtimes(time1,time2,delta):
    flag = np.zeros(len(time2)).astype('int')
    indx = (np.zeros(len(time2))-1).astype('int')
    
    for i in range(len(flag)):
        foo = np.where(np.nanmin(np.abs(time1-time2[i])) == np.abs(time1-time2[i]))[0]
        if len(foo) == 0:
            print('Big problem in matchtime(). How did this happen???')
            return flag, 0
        if np.abs(time1[foo[0]] - time2[i]) <= delta:
            flag[i] = 1
            indx[i] = foo[0]
    
    # By using the min distance above, we have eliminated one-to-many scenario.
    # We now need to eliminate the many-to-one possiblility
    
    vals = np.unique(indx)
    foo = np.where(vals >= 0)[0]
    for i in range(len(foo)):
        bar = np.where(indx == vals[foo[i]])[0]
        if len(bar) > 1:
            duh = np.where(np.nanmin(np.abs(time1[vals[foo[i]]] - time2[bar])) == np.abs(time1[vals[foo[i]]] - time2[bar]))
            if len(duh) == 0:
                print('Big problem in matchtime(). This should not have happened')
                return flag, 0
            flag[bar] = 0         #reset to zero
            flag[bar[duh[0]]] = 1 #and set only one
    
    return flag, 1

################################################################################
# This function is a boxcar smoother with mirrored edges. Npts needs to be an
# odd number
################################################################################

def smooth(data,npts):
    
    if npts % 2 == 0:
        print('Error in boxcar smoother. Npts needs to be an odd number. Not sure how this happened')
        return data, 0
    
    extended = np.insert(data,0,np.flipud(data[0:int(npts/2)]))
    extended = np.append(extended,np.flipud(extended[len(extended)-(int(npts/2)):len(extended)]))
    
    cumsum = np.cumsum(np.insert(extended,0,0))
    
    smoothed = (cumsum[npts:]-cumsum[:-npts])/float(npts)
    
    return smoothed, 1

################################################################################   
# This is a simple model that DDT uses to predict CO2 concentration
# It is a simple fit to the Mauna Loa data, accounting for the seasonal
# variability and the yearly increase.  In tests, the error was within about 4 ppm
################################################################################

def predict_co2_concentration(yy,mm,dd):
    # Coefficients from "plot_predict_co2.pro" on nollij, which is
    # based upon Mauna Loa data from 1958-2010
    
    offset = [ 0.03, 0.67, 1.43, 2.56, 2.97, 2.31, 0.71, -1.40, -3.10, -3.21, -2.07, -0.90]
    coef2 = [0.0121792,-46.8970,45446.4]
    nyears = 100
    year0 = 1960
    x = np.arange(nyears*12)/12. + year0 + (0.5/12)
    y0 = np.array(offset).T
    for i in range(1,nyears):
        y0 = np.append(y0,offset)
    y = y0 + np.polyval(coef2, x)
    # Now get the current year + year_fraction
    jt = datetime(yy,mm,dd).timetuple().tm_yday
    yr = yy + jt/365.
    if (yr < np.nanmin(x)) | (yr > np.nanmax(x)):
        print('Error: the desired year in predict_co2_concentration is outside bounds')
        return -999.
    else:
        return np.interp(yr,x,y)

################################################################################
# This function finds the index of the spectral bands
################################################################################

def find_wnum_idx(wnum, bands):
    flg = np.zeros(len(wnum))
    sz = bands.shape
    if len(sz) == 1:
        if sz[0] != 2:
            print('Problem with band definition in find_wnum_idx')
            return -1, -1
        foo = np.where((bands[0] <= wnum) & (wnum <= bands[1]))[0]
    elif len(sz) == 2:
        if sz[0] == 2:
            for j in range(sz[1]):
                foo = np.where((bands[0,j] <= wnum) & (wnum <= bands[1,j]))[0]
                if len(foo) > 0:
                    flg[foo] = 1
    else:
        'Problem with band definition in find_wnum_idx'
        return -1, -1
    foo = np.where(flg == 1)[0]
    
    return foo, len(foo)
    
################################################################################
# This function computes a covariance matrix with off-diagonal elements; i.e.,
# this covariance includes both random error and systematic error. This
# function was written to simulate calibration error (in PWV) in the RLID WV data
################################################################################

def add_sys_error(x,sig_x,syserr):
    n = 25*len(x)
    prof = np.zeros((len(x),n))
    for i in range(n):
        calfactor = np.random.normal()*(syserr/100.) + 1
        noise = np.random.normal(size=len(x))*sig_x
        prof[:,i] = x*calfactor + noise
    
    cov = np.cov(prof)
    
    return cov
         
################################################################################
# This function computes the trace gas profile based upon the three input
# coefficients, and the assumed function form
################################################################################

def trace_gas_prof(types, z, coef):
    if types == 0:
        # The constant profile, given by the first element of the input vector
        tg_prof = np.ones(len(z))*coef[0]
    elif types == 1:
        # The stair step profile
        #       Coef[0] is the value in the free troposhere [ppm]
        #       Coef[1] is the value in boundary layer relative to Coef[0] [ppm]
        #       Coef[2] is the top of the boundary layer [km AGL]
        tg_prof = np.ones(len(z))*coef[0]
        foo = np.where(z <= coef[2])[0]
        if len(foo) > 0:
            tg_prof[foo] += coef[1]
    elif types == 2:
        # The exponential profile used by Kobra Khosravain.
        #       Coef[0] is the value in the free troposhere [ppm]
        #       Coef[1] is the value at the surface relative to Coef[0] [ppm]
        #       Coef[2] is the shape parameter. It should be between (approx) -20 to -1. [unitless]
        tg_prof = coef[0] + coef[1]*np.exp( coef[2]*z)
        print('Error in trace_gas_prof: profile shape =2 (exponential profile is disabled -- aborting')
        return -999.
    else:
        # This option has not been defined. Abort !!
        print('Error in trace_gas_prof: Undefined type specified.')
        return -999.
    return tg_prof

################################################################################    
#  This function is used to inflate the variance in the temperature and/or humidity
#  profile in the near-surface portion of the prior covariance matrix.  It can 
#  also decrease the covariance between temperature and humidity over the entire matrix
################################################################################

def inflate_prior_covariance(Sa, z, prior_t_ival, prior_t_iht, prior_q_ival,
                prior_q_iht, prior_tq_cov_val, prior_chimney_ht, verbose):
                
    status = 0           # Default is bad
    
    # First, apply some QC to the input inflation factors
    if prior_t_ival < 1:
        print('Error: Prior temperature inflation value must be >= 1 -- aborting')
        return -999, status
    if prior_q_ival < 1:
        print('Error: Prior wate vapor mixing ratio inflation value must be >= 1 -- aborting')
        return -999, status
    if ((prior_t_iht <= 0) | (prior_t_iht > np.nanmax(z))):
        print('Error: Prior temperature inflation height must be a valid height (0 to max(z)) -- aborting')
        return -999, status
    if ((prior_q_iht <= 0) | (prior_t_iht > np.nanmax(z))):
        print('Error: Prior water vapor mixing ratio inflation height must be a valid height (0 to max(z)) -- aborting')
        return -999, status
    if ((prior_tq_cov_val < 0) | (prior_tq_cov_val > 1)):
        print('Error: Prior T/q covariance scalar must be in the interval [0,1] -- aborting')
        return -999, status
        
    if ((prior_t_ival > 1) & (verbose > 1)):
        print('Inflating the prior temperature profile near the surface by a factor of ' + str(prior_t_ival))
    if ((prior_q_ival > 1) & (verbose > 1)):
        print('Inflating the prior WVMR profile near surface by a factor of ' + str(prior_q_ival))
    if ((prior_tq_cov_val < 1) & (verbose > 1)):
        print('Decreasing the covariance between T and q in the prior by a factor of ' + str(prior_tq_cov_val))
    
    # Extract ou the variance values
    var =np.copy(np.diag(Sa))
    
    # Compute the correlation matrix, and scale the T/q covariance values
    
    C = np.copy(Sa)
    newC = np.copy(C)
    for i in range(len(Sa[0,:])):
        for j in range(len(Sa[:,0])):
            C[j,i] = Sa[j,i]/(np.sqrt(var[i])*np.sqrt(var[j]))
            newC[j,i] = np.squeeze(np.copy(C[j,i]))
            if (((j < len(z)) & (i >=len(z))) | ((i < len(z)) & (j >= len(z)))):
                newC[j,i] = newC[j,i] * prior_tq_cov_val
    
    # Scale the temperature variances
    sval = np.array([prior_t_ival,1,1])
    sht = np.array([0,prior_t_iht,np.max(z)])
    var[0:len(z)] = var[0:len(z)] * np.interp(z,sht,sval)
    
    # Scale the WVMR variances
    sval = np.array([prior_q_ival,1,1])
    sht = np.array([0,prior_q_iht,np.max(z)])
    var[len(z):2*len(z)] = var[len(z):2*len(z)] * np.interp(z,sht,sval)
    
    # For all heights below the chimney height, set all covariance to zero. In
    # other words, we don't want there to exist any correlations with heights
    # below the top of the chimney and any other level. Note that I need to do the 
    # same action for the covariance (T,T), covariance(Q,Q) and covariance(T,Q),
    # but do not make the diagonal elements zero in (T,T) or (Q,Q) as then the 
    # matrix is not invertable
    
    foo = np.where(z < prior_chimney_ht)[0]
    if ((len(foo) > 0) & (verbose >= 1)):
        print(f'  Modifying Sa to account for a {prior_chimney_ht*1000.:.1f} m chimney effect')
    k = len(z)
    for i in range(len(foo)):
        for j in range(len(z)):
            if (i != j):
                
                # This handles the (T,T) (note that diagonal not affected)
                newC[i,j] = 0.
                newC[j,i] = 0.
                
                # This handles the (Q,Q) (note that diagonal not affected)
                newC[i+k,j+k] = 0.
                newC[j+k,i+k] = 0.
            
                # This handles the (T,Q) and (Q,T)
            newC[i,j+k] = 0.
            newC[i+k,j] = 0.
            newC[j,i+k] = 0.
            newC[j+k,i] = 0.
    
    # Rebuild the covariance matrix again
    newS = np.copy(newC)
    for i in range(len(Sa[0,:])):
        for j in range(len(Sa[:,0])):
            newS[j,i] = newC[j,i] * (np.sqrt(var[i])*np.sqrt(var[j]))
            
    status = 1
    return newS, status
    
################################################################################    
# This function interpolates the prior covariance to a different height grid.
# It first converts the covariance to correlation, and interpolates the correlation
# matrix to the new height grid.  The variance is interpolated the new height
# grid.  And then the new covariance matrix is computed.  This routine also 
# linearly interpolates the mean prior and pressure profile to the new vertical grid
################################################################################    
def interpolate_prior_covariance(z,Xa,Sa,Pa,zz,verbose=3,debug=False):
    k  = len(z)
    kk = len(zz)
    kidx  = np.arange(k)
    kkidx = np.arange(kk)
        # Pull out the mean T and Q profiles from Xa
    meanT = Xa[kidx]
    meanQ = Xa[k+kidx]
        # Compute the correlation matrix
    Ca = np.zeros_like(Sa)
    for i in range(2*k):
        for j in range(2*k):
            Ca[i,j] = Sa[i,j]/(np.sqrt(Sa[i,i])*np.sqrt(Sa[j,j]))
        # Interpolate this correlation matrix to the new height grid
    if(verbose >= 2):
        print('    Computing prior correlation matrix to the new grid')
    newCa = np.zeros((2*kk,2*kk))
    for i in range(kk):
        j = np.where(z >= zz[i])[0]
        if j[0] == 0:
            j = 1
        else:
            j = j[0]
        wgt = (zz[i]-z[j-1])/(z[j]-z[j-1])
        newCa[i,kkidx]       = np.interp(zz,z,Ca[j-1,kidx]*(1-wgt)     + wgt*Ca[j,kidx])
        newCa[kk+i,kk+kkidx] = np.interp(zz,z,Ca[k+j-1,k+kidx]*(1-wgt) + wgt*Ca[k+j,k+kidx])
        newCa[i,kk+kkidx]    = np.interp(zz,z,Ca[j-1,k+kidx]*(1-wgt)   + wgt*Ca[j,k+kidx])
        newCa[kk+i,kkidx]    = np.interp(zz,z,Ca[k+j-1,kidx]*(1-wgt)  + wgt*Ca[k+j,kidx])
            # Ensure that the diagonal of the new correlation matrix is unity
    for i in range(2*kk):
        newCa[i,i] = 1
            # compute the variance vector on the new height grid
            # and use the same logic to get the new mean vector on this grid
    varSaT    = np.zeros(len(z))
    for i in range(k):
        varSaT[i] = Sa[i,i]
    newvarSaT = np.interp(zz, z, varSaT)
    newmeanT  = np.interp(zz, z, meanT)
    varSaQ    = np.zeros(len(z))
    for i in range(k):
        varSaQ[i] = Sa[k+i,k+i]
    newvarSaQ = np.interp(zz, z, varSaQ)
    newmeanQ  = np.interp(zz, z, meanQ)
    newvarSa  = np.append(newvarSaT,newvarSaQ)
    newXa     = np.append(newmeanT,newmeanQ)
    newPa     = np.interp(zz, z, Pa)

            # Now rebuild the new covariance matrix
    if(verbose >= 2):
        print('    Rebuilding prior covariance matrix on the new grid')
    newSa = np.zeros_like(newCa)
    for i in range(2*kk):
        for j in range(2*kk):
            newSa[i,j] = newCa[i,j] * np.sqrt(newvarSa[i])*np.sqrt(newvarSa[j])

            # If the debug option is set, then write out the interpolated prior output
    if debug == True:
        print('    Writing the parts generated from interpolate_prior_covariance()')
        Output_Functions.write_variable(Sa,'/data/origSa.cdf')
        Output_Functions.write_variable(Ca,'/data/origCa.cdf')
        Output_Functions.write_variable(newCa,'/data/newCa.cdf')
        Output_Functions.write_variable(newSa,'/data/newSa.cdf')
        Output_Functions.write_variable(z,'/data/z.cdf')
        Output_Functions.write_variable(zz,'/data/newz.cdf')
        Output_Functions.write_variable(Xa,'/data/oXa.cdf')
        Output_Functions.write_variable(newXa,'/data/nXa.cdf')

    return newXa, newSa, newPa

################################################################################
# This function computes the height of the convective mixed layer (PBL) using
# a parcel theory approach (i.e., lift a parcel from the minht up and look for 
# when its potential temperature is above the surface value). I will slightly
# nudge the theta(minht) value to account for retrieval noise (this is commonly done
# even for radiosonde profiles)
################################################################################

def compute_pblh(ht, t, p, sigt, nudge=0.5, minht=0.300, maxht=6.0):
    st = np.zeros_like(sigt)
    st[0] = sigt[0]
    theta = Calcs_Conversions.t2theta(t+st, np.zeros(len(t)), p)
    sval  = theta[0]
    sval += nudge
    foo = np.where((sval < theta) & (ht >= minht))[0]
    if len(foo) == 0:
        pblh = minht
    elif foo[0] == 0:
        pblh = ht[0]
    else:
        tmpz = ht[foo[0]-1:foo[0]+1]
        tmpt = theta[foo[0]-1:foo[0]+1]

        if np.abs(tmpt[0]-tmpt[1]) < 0.001:
            pblh = ht[foo[0]]
        else:
            pblh = np.interp(sval, tmpt, tmpz)

    if pblh < minht:
        pblh = minht

    if pblh > maxht:
        pblh = maxht

    return pblh

################################################################################
# This function computes the depth of the surface based inversion (sbih, km AGL)
# intensity (sbim, degC)
################################################################################

def compute_sbi(z, t, start_height = -1):
    
    for ii in range(0,len(z)):
        if z[ii] > start_height:
            break
    
    if ii >= len(z)-1:
        return -999., -999.
    
    sbi = ii
    for i in range(ii+1,len(t)):
        if ((t[i] >= t[i-1]) & (sbi == i-1)):
            sbi = i
        else:
            if sbi > i-1:
                break
    
    if sbi > ii:
        sbih = z[sbi]
        sbim = t[sbi] - t[ii]
        return sbih, sbim
    else:
        return -999., -999.
        
################################################################################
# This function computes the lifting condensation level (LCL). The first three
# imputes are the surface temperature [C], water vapor [g/kg], and pressure [mb],
# while the next two are profiles of pressure [mb] and height [arbitrary]. The
# LCL value returned is the height of the LCL in the same units as the height
# field.
################################################################################

def compute_lcl(tsfc, wsfc, psfc, p, z):
    
    pp = psfc - np.arange(1000)
    foo = np.where(pp > 1)[0]
    pp = pp[foo]
    kappa = 0.286
    tt = (((pp / psfc)**kappa) *(tsfc+273.16)) - 273.16
    ws = np.zeros(len(pp))
    for i in range(len(pp)):
        ws[i] = Calcs_Conversions.wsat(tt[i], pp[i])
    foo = np.where(wsfc >= ws)[0]
    if len(foo) <= 0:
        return -999., -999.
    fct = scipy.interpolate.interp1d(p,z)
    zlcl = fct(pp[foo[0]])
    
    
    return zlcl, pp[foo[0]]
    
################################################################################
# This function, suggested by Kobra Khosravian, is used to get the value of the "a2"
# parameter from the boundary layer height.
################################################################################

def get_a2_pblh(pblh):
    scalar = 0.1             # We want to get the 20% value
    coef = np.log(scalar)/pblh
    return coef
    
################################################################################
# This function decorrelates the levels above the PBL from the levels below.
################################################################################
             
def covariance_matrix_decorrelate_level(Sa,z,pblh,prior_pblh_decorrelate_factor):
    
    # Compute the correlation matrix from the covariance matrix. Note I am
    # only working with the T/q part of the covariance matrix; the cloud and
    # trace gas parts will be ignored (and thus will remain the same
    
    Ca = np.copy(Sa)
    for i in range(2*len(z)):
        for j in range(2*len(z)):
            Ca[i,j] = Sa[i,j]/(np.sqrt(Sa[i,i])*np.sqrt(Sa[j,j]))
    
    # Find the decorrelation height
    lev = np.where(z >= pblh)[0][0]
    
    # Now multiply the correlation values at this level by this factor
    # (which is constrained to be between 1 and 100)
    factor = np.max([prior_pblh_decorrelate_factor, 1])
    factor = np.min([factor, 1e2])
    idx = np.arange(len(z)*2)
    f0 = np.ones(len(z)*2)*factor
    f1 = np.copy(f0)
    f1[lev] = 1
    f1[len(z)+lev] = np.sqrt(factor)
    Ca[idx,lev] = Ca[idx,lev]/f1
    Ca[lev,idx] = Ca[lev,idx]/f1
    f1 = np.copy(f0)
    f1[lev] = np.sqrt(factor)
    f1[len(z)+lev] = 1
    Ca[idx,len(z)+lev] = Ca[idx,len(z)+lev]/f1
    Ca[len(z)+lev,idx] = Ca[len(z)+lev,idx]/f1
    
    # Now convert the correlation matrix back into a covariance matrix
    newSa = np.copy(Sa)
    for i in range(2*len(z)):
        for j in range(2*len(z)):
            newSa[i,j] = Ca[i,j]*(np.sqrt(Sa[i,i])*np.sqrt(Sa[j,j]))
    
    return newSa

################################################################################
# Look at the VCEIL sample to determine if there is a cloud or not in the
# FOV. If a cloud is found at a nearby time, then capture its median value.
# First look in the innere window for a cloud, and if there is one, use that
# height (most accurate). But if no cloud is found in the inner window, then
# look in the outer window to find the approximate height where clouds could
# be located (a bit less accurate). If both fail, then use the default CBH
################################################################################

def find_cloud(irsch1, vceil, window_in, window_out, default_cbh):
    
    twin = window_in    # Full-width of time window for IRS observations [min]
    twout = window_out  # Full-width of time window for IRS observations [max]
       
        # The cloud flag. 1-> Inner window, 2-> Outer window, 3-> defaultCBH
    vcbh = np.zeros(len(irsch1['secs']))
    vflag = np.zeros(len(irsch1['secs']))
    
    for i in range(len(irsch1['secs'])):
        
        # Get any ceil cloud base height data that exist in the inner window
        
        foo = np.where((irsch1['secs'][i]-(twin/2.)*60. <= vceil['secs']) &
                       (vceil['secs'] <= irsch1['secs'][i]+(twin/2.)*60.) &
                       (vceil['cbh'] >= 0))[0]
        
        if len(foo) != 0:
            vcbh[i] = np.nanmedian(vceil['cbh'][foo])
            vflag[i] = 1
        else:
            # Get any ceil cloud base height data that exist in the outer window
            
            foo = np.where((irsch1['secs'][i]-(twout/2.)*60. <= vceil['secs']) &
                       (vceil['secs'] <= irsch1['secs'][i]+(twout/2.)*60.) &
                       (vceil['cbh'] >= 0))[0]
            
            if len(foo) != 0:
                vcbh[i] = np.nanmedian(vceil['cbh'][foo])
                vflag[i] = 2
            else:
                vcbh[i] = default_cbh
                vflag[i] = 3
    
    return ({'success':1, 'secs':irsch1['secs'], 'ymd':irsch1['ymd'], 'hour':irsch1['hour'],
            'yy':irsch1['yy'], 'mm':irsch1['mm'], 'dd':irsch1['dd'], 'cbh':vcbh, 'cbhflag':vflag,
            'hatchopen':irsch1['hatchopen'], 'avg_instant':irsch1['avg_instant'],
            'wnum':irsch1['wnum'], 'radmn':irsch1['radmn'], 'noise':irsch1['noise'], 
            'atmos_pres':irsch1['atmos_pres'], 'nearSfcTb':irsch1['nearSfcTb'], 
            'tsfc':irsch1['tsfc'], 'fv':irsch1['fv'], 'fa':irsch1['fa'], 'missingDataFlag':irsch1['missingDataFlag']})

################################################################################
# This routine returns the emissivity spectrum for the AERI blackbodies,
# interpolated onto the correct wavenumber grid.
#
# Original code from Dave Tobin, UW-Madison
# See notebook SSEC/CIMSS pg 55, or email from Tobin dated Aug 2, 2002
# Yao Te's calculations are from a Monte Carlo simulation he performed at SSEC
# during the fall of 2002. The results were passed to Dave Turner by BobK on
# 12/17/02 in an email
################################################################################

def get_aeri_bb_emis(wn,cavity_factor = 12.79, option=0, verbose=1):
    
    if ((cavity_factor == 12.79) & (option == 0) & (verbose > 1) & (verbose < 5)):
        print('    WARNING: Cavity factor set to default value of 12.79 (in get_aeri_bb_emis)')

    #The emissivity spectrum, from Dave Tobin's file
    v = [400.00000, 500.00000, 550.00000, 600.00000, 650.00000, 700.00000,
         740.00000, 765.00000, 800.00000, 850.00000, 900.00000, 950.00000,
         1000.0000, 1060.0000, 1100.0000, 1150.0000, 1200.0000, 1300.0000,
         1400.0000, 1510.0000, 1550.0000, 1650.0000, 1700.0000, 1732.0000,
         1746.0000, 1800.0000, 1850.0000, 1900.0000, 2030.0000, 2100.0000,
         2200.0000, 2300.0000, 2400.0000, 2500.0000, 2600.0000, 2700.0000,
         2800.0000, 2900.0000, 3000.0000, 3100.0000]
    e = [0.96230000, 0.96280000, 0.96200000, 0.95940000, 0.95600000, 0.95560000,
         0.95410000, 0.95480000, 0.95410000, 0.95560000, 0.95600000, 0.95560000,
         0.95030000, 0.94130000, 0.94950000, 0.96860000, 0.97320000, 0.97600000,
         0.97660000, 0.97700000, 0.97700000, 0.97700000, 0.97670000, 0.97510000,
         0.97440000, 0.96690000, 0.96440000, 0.96360000, 0.96400000, 0.96400000,
         0.96460000, 0.96540000, 0.96460000, 0.96460000, 0.96540000, 0.96540000,
         0.96610000, 0.96690000, 0.96760000, 0.96760000]
         
    #Interpolate this to our desired wavenumbers
    emis = np.interp(wn,v,e)
    
    #Using the curves provided by HankR on 5 Oct 2002
    #These were derived from Monte Carlo simulations of the BBs
    
    #Emissivity of the "wall" (i.e., paint) for each of the others
    paint_emis = [0.94,0.95,0.96,0.97,0.98]
    
    #The emissivity numbers below are based upon some
    #research into various models of the AERI BBs. This research was done
    #primarily in the late 1990's - early 2000's
    
    #Hemispherical FOV emissivity
    hemis_emis = [0.995032, 0.996407, 0.997149, 0.997867, 0.998587]
    
    #Normal FOV
    normal_emis = [0.998911, 0.999107, 0.999298, 0.999482, 0.999661]
    
    #FOV is 46 mrad full-angle about normal
    restricted_emis = [0.998895, 0.999097, 0.990291, 0.999475, 0.999658]
    
    #FOV is 46 mrad, full-angle, but reflectance is diffuse
    yaote_emis00 = [0.9993954, 0.9995041, 0.9996146, 0.9997120, 0.9998141]
    
    #FOV is 46 mrad, full-angle, but reflectance is specular
    yaote_emis99 = [0.9997791, 0.9998307, 0.9998798, 0.9999208, 0.9999522]
    
    #Apply the cavity factor and convert to blackbody emissivity
    #Note that the method chosen depends on the option selected
    
    if option == -1:
        print('Returning emissivity spectrum of the paint (no cavity factor applied)')
    elif option == 0:
        #The original way to use the wavelength-independent cavity factor
        rcf = 1.0/cavity_factor
        bb_emis = emis/(emis + rcf * (1 - emis))
    elif option == 9:
        #This spectrally-dependent cavity factor, along with the coefficients in
        #in "a", are the latest results from the Monte Carlo models performed by
        #UW-Madison as of Feb 2006.
        a = np.array([0.06,-3.9,93.])
        cavity_factor = np.polyval(a, 10000./wn)
        rcf = 1.0/cavity_factor
        bb_emis = emis/(emis + rcf * (1 - emis))
    else:
        if option == 1:
            fov_emis = hemis_emis
        elif option == 2:
            fov_emis = normal_emis
        elif option == 3:
            fov_emis = restricted_emis
        elif option == 4:
            fov_emis = yaote_emis00
        elif option == 5:
            fov_emis = yaote_emis99
        else:
            print('Error: Undefined value for option in get_aeri_bb_emis()')
            return 0
        
        bb_emis = emis * 0.
        for i in range(len(bb_emis)):
            bb_emis[i] = np.interp(emis[i],paint_emis,fov_emis)
    
    return bb_emis

################################################################################
# This routine recalibrates AERI spectrum based upon the temperature and
# emissivity of the blackbodies. It can also be used to estimate the uncertainty
# in the AERI radiance based upon uncertainties in the measured BB temperatures/
# emissivities. Note that it is designed to only process one radiance spectrum
# at a time.
################################################################################

def aeri_recal(wn,rad1,th1,th2,ta1,ta2,tr1,tr2,eh1,eh2,ea1,ea2):
    
    #Start with simple QC
    n = len(wn)
    if ((n != len(rad1)) | (n != len(eh1)) | (n != len(eh2)) | (n != len(ea1)) | (n != len(ea2))):
        print('Error: the wavenumber, radiance, and emissivity arrays must be the same length')
        return 0
        
    if not ((np.isscalar(th1)) | (np.isscalar(th2)) | (np.isscalar(ta1)) | (np.isscalar(ta2)) | (np.isscalar(tr1)) | (np.isscalar(tr2))):
        print('Error: the various temperatures must all be scalars')
        return 0
    
    if ((th1 <= 0) or (th2 <= 0) or (ta1 <= 0) or (ta2 <= 0) or (tr1 <= 0) or (tr2 <= 0)):
        print('Warning: the various IRS calibration temperatures must all be positive -- recal function is not being used')
        return rad1

    #Compute the radiances for the various temperatures
    
    b_th1 = Calcs_Conversions.planck(wn, th1)
    b_th2 = Calcs_Conversions.planck(wn, th2)
    b_ta1 = Calcs_Conversions.planck(wn, ta1)
    b_ta2 = Calcs_Conversions.planck(wn, ta2)
    b_tr1 = Calcs_Conversions.planck(wn, tr1)
    b_tr2 = Calcs_Conversions.planck(wn, tr2)
    
    #Compute pseudo radiances
    Bh1 = eh1 * b_th1 + (1-eh1) * b_tr1
    Ba1 = ea1 * b_ta1 + (1-ea1) * b_tr1
    Bh2 = eh2 * b_th2 + (1-eh2) * b_tr2
    Ba2 = ea2 * b_ta2 + (1-ea2) * b_tr2
    
    #and recalibrate
    
    rad2 = ((rad1 - Ba1) / (Bh1 - Ba1)) * (Bh2 - Ba2) + Ba2
    
    return rad2

################################################################################
# This routine extracts out the single scattering properties from the
# matrices that were previously read in. It then extracts out the slices
# associated with the wavenumbers just above and below the desired wavenumber,
# and then interpolates these to the desired wavenumber. Finally it
# interpolates the properties to the effective radius and returns.
# 
# The is some error checking in this routine, but if an error is found 
# the code just stops there. Probably the best answer if the code ever stops
# in one of these traps is to improve the scattering properties database(s) to
# include the wavelength/size desired
################################################################################

def get_scat_properties(field, wnum, reff, database):
    
    if field == 'ext_xsec':
        index = 3
    elif field == 'sca_xsec':
        index = 4
    elif field == 'abs_xsec':
        index = 5
    elif field == 'omega':
        index = 6
    elif field == 'w0':
        index = 6
    elif field == 'asym':
        index = 7
    elif field == 'g':
        index = 7
    elif field == 'Qe':
        index = 8
    elif field == 'Qa':
        index = 9
    elif field == 'Qs':
        index = 10
    elif field == 'vol':
        index = 11
    elif field == 'proj_area':
        index = 12
    elif field == 'pf':
        index = np.arange(database['nphase']) + 13
    else:
        print('Undefined field used in get_scat_properties()')
        return -999.
    
    scattdb = np.copy(database['data'])
    
    # Check to see if r or wnum is an array, but make sure that they are not
    # both arrays
    if ((type(wnum) == np.ndarray) & (type(reff) == np.ndarray)):
        print('Error: both Reff and wnum are arrays -- only one of these is allowed to be')
        return -999.
        
    elif type(reff) == np.ndarray:
        iidx = 2        # Interpolation index
        sidx = 1        # Selection index
        sval = wnum
        scom = 'wavenumber'
        icom = 'effective radius'
        desx = np.copy(reff)
        
    else:
        iidx = 1       # Interpolation index
        sidx = 2       # Selection index
        sval = reff
        scom = 'effective_radius'
        icom = 'wavenumber'
        if type(wnum) == np.ndarray:
            desx = np.copy(wnum)
        else:
            desx = wnum
    
    # Now find the surrounding data
    sdata = scattdb[sidx,:]
    sdata = np.sort(sdata)
    sdata_uniq = np.unique(sdata)
    bar = np.where(sdata_uniq >= sval)[0]
    if len(bar) == 0:
        print('Error: the entered ' + scom + ' is above the max value in the SSP database (' + str(np.max(sdata_uniq)) +  ')')
        return -999.
    
    if bar[0] == 0:
        print('Error: the entired ' + scom + ' is below the min value in the SSP database (' + str(np.min(sdata_uniq)) + ')')
        return -999.
    
    foo1 = np.where(scattdb[sidx,:] == sdata_uniq[bar[0]-1])[0]
    foo2 = np.where(scattdb[sidx,:] == sdata_uniq[bar[0]])[0]
    
    if len(foo1) != len(foo2):
        print('This should NEVER happen and these should be the same')
        return -999.
        
    wt = (sval-sdata_uniq[bar[0]-1])/(sdata_uniq[bar[0]]-sdata_uniq[bar[0]-1])
    
    ivalx = np.copy(scattdb[iidx,foo1])
    ivaly1 = np.copy(scattdb[index,foo1])
    ivaly2 = np.copy(scattdb[index,foo2])
    ivaly = ivaly1 * (1-wt) + wt * ivaly2
    
    if np.min(ivalx) > np.min(desx):
        print('Error: the entired ' + icom + ' is below the min value in the SSP database (' + str(np.min(ivalx)) + ')')
        return -999.
    
    if np.max(ivalx) < np.max(desx):
        print('Error: the entered ' + icom + ' is below the max value in the SSP database (' + str(np.max(ivalx)) + ')')
        return -999.
    
    desy = np.interp(desx,ivalx,ivaly)
    return desy

################################################################################
# This routine is used to convert cloud particle number density into
# optical depth.  One must be careful of the units used here.  The extinction
# cross section coefficient is assumed to have come from the single scattering
# databases, and hence have units [um^2].  All other units are in meters to
# some power.  Note that if the user wishes to use equivalent volume spheres,
# or equivalent area spheres, or equivalent volume/area spheres, the adjustment
# of the effective radius (which was used to get the extinction coefficient)
# and the number density needs to be done outside of this function.
################################################################################

def od2numdens(ext_coef, thickness, od):
    
    # Convert from [um^2] to [m^2]
    extcoefxsec = ext_coef/(1e6**2)
    
    # Get the volume extincition coef [m-1] from the optical depth
    extcoefvol = od/thickness
    
    # Back out the number density from the two extinction coefficients
    # Units are [m^-1] / [m^2] = [m^-3], which is correct
        
    numdens = extcoefvol / extcoefxsec
    
    return numdens
    
################################################################################
# This routine is used to convert cloud optical depth to cloud particle number
# density.  One must be careful of the units used here.  The extinction
# cross section coefficient is assumed to have come from the single scattering
# databases, and hence have units [um^2].  All other units are in meters to
# some power.  Note that if the user wishes to use equivalent volume spheres,
# or equivalent area spheres, or equivalent volume/area spheres, the adjustment
# of the effective radius (which was used to get the extinction coefficient)
# and the number density needs to be done outside of this function.
################################################################################

def numdens2od(ext_coef, thickness, numdens):
    
    # Convert from [um^2] to [m^2]
    extcoefxsec = ext_coef/(1e6**2)
    
    # Convert extinction cross section [m^2]
    # to volume extinction coefficient [m-1]
    extcoefvol = extcoefxsec * numdens
    
    # And assuming a constant extinction coef with thickness, compute optical depth
    od = extcoefvol * thickness
    
    return od
        
################################################################################
# This routine gets the spectral IR absorption optical depths. It uses the scattering
# properties in a SSP database (read into python with a previous call). The logic is the
# same as what is encoded in LBLDIS.
################################################################################

def get_ir_cld_ods(ssp, intau, wnum, reff, cldthick):
    
    # Quick QC
    if ((reff < np.nanmin(ssp['data'][2,:])) | (reff > np.nanmax(ssp['data'][2,:]))):
        print('Error: the desired Reff is outside the Reff range in the SSP ' + ssp['dbname'])
        return -999.
    
    # Get the extinction efficiency and extinction coefficient at the
    # reference wavenumber for the input (geometric-limit) optical depth
    # and cloud thickness...
    wnum_reff = 900.
    Qext = get_scat_properties('Qe',wnum_reff,reff,ssp)
    kext = get_scat_properties('ext_xsec',wnum_reff,reff,ssp)
    
    # Compute the number density for this cloud using the reference wavenumber
    numdens = od2numdens(kext,cldthick*1000.,intau*Qext/2.)
    
    # Get the absoprtion coefficient, absorption coefficient spectrum, and
    # then compute the spectral cloud absorption optical depth
    
    kabs = get_scat_properties('abs_xsec',wnum,reff,ssp)
    outtau = numdens2od(kabs, cldthick*1000.,numdens)
    
    return outtau
    
################################################################################
# This routine returns the IR spectral cloud reflectivity (an estimate of i,
# using the form shown in DDT's dissertation) as a function of cloud optical depth
################################################################################

def cloud_reflectivity(dwnum, tau, ice = False):
    
    if type(tau) is np.ndarray:
        print('Warning: the routine cloud_reflectivity needs tau as a signleton - returning zeros')
        return np.zeros(len(dwnum))
    
    # This spectrum is from DDT's dissertation, with water at Reff=7.5 and
    # optical depth 4.0
    
    wnum = np.array( [400.0, 478.500, 496.750, 530.700, 560.250, 772.850, 788.300, 810.950,
  	819.850, 831.450, 845.450, 862.050, 874.850, 893.850, 901.800, 934.650,
	962.100, 991.500, 1080.70, 1095.45, 1114.95, 1128.50, 1145.10, 1159.30] )
    wref40 = np.array( [0.0169606, 0.0147522, 0.0142854, 0.0130697, 0.0120277, 0.00572757,
  	0.00532051, 0.00481670, 0.00467556, 0.00448597, 0.00424655, 0.00434656,
	0.00451425, 0.00476012, 0.00492731, 0.00691303, 0.00935897, 0.0128049,
	0.0222350, 0.0237693, 0.0252363, 0.0261802, 0.0273126, 0.0281716] )
    iref40 = np.array( [0.0575502, 0.0325507, 0.0224449, 0.0170424, 0.0132832, 0.00555305,
  	0.00546641, 0.00514759, 0.00497298, 0.00474025, 0.00445012, 0.00397647, 
	0.00357329, 0.00301045, 0.00276519, 0.00185897, 0.00231630, 0.00420333, 
	0.00673380, 0.00687247, 0.00708907, 0.00725474, 0.00743006, 0.00753798] )
    
    if ice:
        wref40 = np.copy(iref40)
    
    # DDT then did some fooling around, and determined this extrapolation
    # techniue to go to at least 1400 cm-1. The better way would be to
    # recompute the cloud reflectivity spectrum in the first place to cover
    # the entire spectral range I desire.
    
    awnum = np.append(wnum,[1200,1250,1300,1350,1400])
    foo = np.where(wnum > 1050)[0]
    coef = np.polyfit(wnum[foo],wref40[foo],2)
    aref40 = np.append(wref40,[0,0,0,0,0])
    bar = np.where(awnum > 1050)[0]
    aref40[bar] = np.polyval(coef,awnum[bar])
    bar = np.where((aref40 == np.max(aref40)) & (awnum > 1100))[0]
    foo = np.where(awnum > awnum[bar[0]])[0]
    if len(foo) > 0:
        aref40[foo] = aref40[bar[0]]
    
    # This bit of logic scales the reflectivity spectrum to account for the
    # change in optical depth of the cloud. Watch for negative optical depth
    dtau = tau
    if(tau < 0):
        dtau = 0
    emis = 1.-np.exp(-dtau)
    reflect = aref40 * (emis**0.8)
    dreflect = np.interp(dwnum,awnum,reflect)
    
    return dreflect
    
################################################################################
# Clear-sky downwelling radiance using optical depths previously calculated.
# It uses a "linear in tau" approach. Note that the profiles *MUST* extend from
# the surface (index 0) to the TOA...
################################################################################

def radxfer(wnum, t, od, sfc_t = None, sfc_e = None, upwelling = False):
    
    # Some basic checks to ensure that the arrays are dimesioned correctly
    if type(wnum) != np.ndarray:
        opd = np.copy(od)
        a = opd.shape
        if len(a) != 1:
            print('Error: Optical depth array dimension does not match wnum dimension')
            return -1
    else:
        opd = np.copy(od)
        a = opd.shape
        if a[0] != len(wnum):
            opd = opd.T
            a = opd.shape
            if a[0] != len(wnum):
                print('Error: Optical depth array dimension does not match wnum dimension')
                return -3
    
    if a[0] != len(t)-1:
        if len(a) == 2:
            opd = opd.T
            a = opd.shape
        if a[0] == len(t)-1:
            looking_good = 1
        else:
            print('Error: Optical depth array dimension does not match temperature dimension')
            return -2
    foo = np.where(t < 0)[0]
    if len(foo) > 0:
        print('Error: temperature array has negative values; units should be Kelvin')
        return -4
    
    if upwelling:
        if sfc_e is None:
            if wnum != np.ndarray:
                sfce = 1.0
            else:   
                sfce = np.ones(len(wnum))
        else:
            sfce = np.squeeze(np.copy(sfc_e))
        foo = np.where((sfce < 0) | (sfce > 1))
        if len(foo) > 0:
            print('Error: surface emissivity is outside [0,1]')
            return -5
            
        # Make sure that the surface emissivity is the right size
        if type(wnum) != np.ndarray:
            if sfce == np.ndarray:
                print('Error: surface emissivity does not have the same dimension size as wnum')
                return -6
        else:
            if len(sfce) != len(wnum):
                print('Error: surface emissivity does not have the same dimension size as wnum')
            return -6
        
        # If no surface temperatue is given then use temp of lowest level
        if sfc_t is None:
            sfct = t[0]
        else:
            sfct = np.squeeze(np.copy(sfc_t))
        
        # set the indices for the loop
        k_start = 0
        k_end = len(opd[:,0])
        k_step = 1
    
    else:
        # No surface emission since this is downwelling
        if type(wnum) != np.ndarray:
            sfce = 0.0
        else:   
            sfce = np.zeros(len(wnum))
        sfct = 0. # Temperature of deep space...
        
        # Set the indices for the loop
        k_start = len(opd[:,0])-1
        k_end = -1
        k_step = -1
    
    # Get the radiance for a blackbody emitting at sfct. Note that
    # this isn't correct for downwelling radiation, but in that case
    # the emissivity is zero so it doesn't matter
    
    if upwelling:
        sfcb = Calcs_Conversions.planck(wnum,sfct)
    else:
        sfcb = 0.0
    
    # Now we are set to do the radiative transfer
    # Will use the linear in tau approach
    foo = np.where(opd < 1.0e-6)
    opd[foo] = 1.0e-6       # Add a small fudge factor to avoid singularity
    opd = opd.T
    rad = sfcb * sfce
    
    for k in range(k_start, k_end, k_step):
        # Get the transmission of the layer
        trans = np.exp(-1*opd[:,k])
        
        # Compute the blackbody radiance at the boundary closet to the observer
        if upwelling:
            b_boundary = Calcs_Conversions.planck(wnum,t[k+1])
        else:
            b_boundary = Calcs_Conversions.planck(wnum,t[k])
        
        # Compute the blackbody radiance at the middle of the layer (average temp)
        b_avg = Calcs_Conversions.planck(wnum, (t[k]+t[k+1])/2.)
        
        # Compute the radiance using Eq 13 in Clough et al 1992
        rad = rad*trans + (1.-trans)*(b_boundary+2*(b_avg-b_boundary)*(1./opd[:,k] - trans/(1.-trans)))
    return rad

################################################################################
# Norton-Beer apodization function
################################################################################

def apodize_norton_beer(n,md):
    n = int(n)
    beer = np.zeros(n)
    beer[0] = 1.
    for i in range(1,int(n/2)):
        if i <= md:
            beer[i] = (1-((i-1)/np.float(md))**2)**2
        else:
            beer[i] = 0.
    
    if n%2 == 0:
        beer[int(n/2):n] = np.fliplr([beer[0:int(n/2)]])[0]
    else:
        beer[int(n/2):n] = np.fliplr([beer[0:int(n/2)+1]])[0]

    return beer
    
################################################################################
# Kaiser-Bessel  apodization function
################################################################################

def apodize_kaiser_bessel(n,md,k=6):
    
    n = int(n)
    k = int(k)
    if k < 1:
        print('ERROR in apodize kaiser_bessel: Coefficient must be an integer > 6')
        sys.exit()
    
    apod = np.zeros(n)
    d = np.arange(n/2)
    d = d/np.float(np.max(d))
    d = d * (n/2)
    x = k * np.sqrt(1 - (d/np.float(md))**2)
    r = np.ones(n/2)
    f = 1
    for j in range(1,9):
        f = f * j
        r = r + ((x/2)**(2*j)) / (f**2)
    
    s = 1.
    f = 1
    for j in range(1,9):
        f = f * j
        s = s + ((k/2.)**(2*j)) / (f**2)
    
    foo = np.where(np.abs(d) <= md)[0]
    c = 1 * r[foo] / s
    apod[0:int(n/2)] = c
    
    if n%2 == 0:
        apod[int(n/2):n] = np.fliplr([apod[0:int(n/2)]])[0]
    else:
        apod[int(n/2):n] = np.fliplr([apod[0:int(n/2)+1]])[0]
    
    return apod
    
################################################################################
# Function used to apodize the input spectrum, using one of a few apodization
# functions
################################################################################

def apodizer(spectrum, aflag):
    n = len(spectrum)
    imd = n/2
    
    if aflag == 0:
        apod = apodize_norton_beer(n, imd)
    elif aflag == 1:
        apod = apodize_kaiser_bessel(n, imd)
    else:
        print('ERROR in apodizer: Undetermined apodization funcion - abort')
        sys.exit()
    
    new_spectrum = np.fft.fft(np.fft.ifft(spectrum)*apod)
    
    return new_spectrum

################################################################################
# This function convolves the spectrum to the AERI / ASSIST spectral resolution
def convolve_to_aeri(wnum, radiance):
    vlaser = 15799.
    tmp = convolve_to_irs(wnum, radiance, vlaser)
    return tmp

################################################################################
# This function convolves the spectrum to the REFIR-PAD spectral resolution
def convolve_to_refir(wnum, radiance, refir_del_wnum):
    #vlaser = 13107.2        # irs_type = 6, which is for del_wnum = 0.4000000 cm-1 (like in DomeC)
    #vlaser = 12754.8        # irs_type = 7, which is for del_wnum = 0.3892450 cm-1 (like in RHUBC-II)
    vlaser = refir_del_wnum * (2**15)
    rad1 = convolve_to_irs(wnum, radiance, vlaser)
    rad2 = convolve_to_irs(wnum, radiance, vlaser, do_sinc2=1)
    tmp = {'wnum':rad1['wnum'], 'spec':0.9*rad1['spec']+0.1*rad2['spec']}
    return tmp


################################################################################
# This routine takes the input spectrum, and convolves it with the IRS filter
# function to get data at the IRS resolution.  Actually, instead of performing
# a convolution, it does this by multiplying the interferogram by the boxcar
# that is associated with the IRS's Max OPD.  To do this, the input spectrum
# is tapered and zeropadded to create a smooth spectrum, which is then interpolated
# to a multiple of the AERI's spectral resolution, which is then transformed back
# into interferogram space for chopping. 
################################################################################
def convolve_to_irs(wnum, radiance, vlaser, do_sinc2=0):
    
        # Set this to 1 to write out temporary files to see if the
        # zeropadding is inducing ringing in the spectra
    write_debug_ringing = 0

    # These will be needed later
    minv = np.min(wnum)
    maxv = np.max(wnum)
    
    x = np.copy(wnum)
    y = np.copy(radiance)
    
    # Quick QC
    if len(y) != len(x):
        print('ERROR: wnum and radiance should have the same number of elements!')
        return 0
    
    # The AERI's delta nu
    aeri_dv = vlaser/(2**15)
    # And the AERI's maximum optical path delay
    aeri_opd = 1./(2*aeri_dv)
    
    # Apply tapers to the end of the radiance spectrum
    # This will also result in the spectrum having 2^n + 1 points
    
    # Find the mean wavenumber delta. I want to use the mean, rather than
    # the difference of the first two elements
    
    delx = np.mean(x[1:len(x)] - x[0:len(x)-1])
    
    # First step: taper the head (long wavelength side) of the spectrum
    npts = int((x[0] - 0)/delx  - 1)
    xx = np.arange(npts+1) * delx
    yy = np.zeros(npts+1)
    x = np.append(xx,x)
    y = np.append(yy,y)
    
    # Second step: insert taper at tail to get the proper number of
    # points in the spectrum (2^n + 1 points)
    for i in range(100+1):
        if 2**i >= len(x):
            break
    npts = 2**(i+1) - len(x) +1
    xx = np.arange(npts) * delx + delx + x[len(x)-1]
    yy = np.zeros(len(xx))
    x = np.append(x,xx)
    y = np.append(y,yy)

    if write_debug_ringing == 1:
        Output_Functions.write_variable(x,'/data/wnum_after_appending_tapers.cdf')
        Output_Functions.write_variable(y,'/data/spec_after_appending_tapers.cdf')
    
    # Determin the size of the rolloff to apply; it should be 
    # no more than 100 cm-1, but may need to be smaller...
    rolloffsize = np.min([np.max(x)-maxv, minv, 100.])
    tapersize = 20.            # The amount of space in the spectrum to taper [cm-1]
    tapersize =  5.            # The amount of space in the spectrum to taper [cm-1]
    
    # Find the spectral regions that require a roll-off
    v_rolloff1 = np.where((minv-rolloffsize <= x) & (x <= minv))[0]
    v_rolloff2 = np.where((maxv <= x) & (x <= maxv + rolloffsize))[0]
    #v_rolloff1 = np.where(x <= minv)[0]     # New approach to go to the end
    #v_rolloff2 = np.where(maxv <= x)[0]     # New approach to go to the end
    
    # Apply the roll-off and then make it smooth for a
    # small wavenumber region around the roll-off
    oldy = np.copy(y)               # Keep a copy for debugging, but not needed anymore
    bar = (np.cos((np.arange(len(v_rolloff1))/(len(v_rolloff1)-1.)) * np.pi - np.pi) + 1)/2.
    feh = np.where((minv <= x) & (x <= minv+tapersize))[0]
    y[v_rolloff1] = bar * np.mean(y[feh])
    weight = np.arange(len(feh))/(len(feh)-1.)
    y[feh] = y[feh]*weight + (1-weight)*np.mean(y[feh])

    bar = (np.cos((np.arange(len(v_rolloff2))/(len(v_rolloff2)-1.)) * np.pi) + 1)/2.
    feh = np.where((maxv-tapersize <= x) & (x <= maxv))[0]
    y[v_rolloff2] = bar * np.mean(y[feh])
    weight = np.arange(len(feh))/(len(feh)-1.)
    y[feh] = y[feh]*(1-weight) + weight*np.mean(y[feh])

    if write_debug_ringing == 1:
        print(f"DEBUG: writing out the spectrum before the zeropad happens")
        Output_Functions.write_variable(x,'/data/wnum_before_zeropad.cdf')
        Output_Functions.write_variable(y,'/data/spec_before_zeropad.cdf')

    # If the wavenumber resolution is "coarse", then we need to zeropad
    # the spectrum to allow us to interpolate to a multiple of the AERI
    # resolution...
    
    # This threshold is a bit arbitrary...
    if delx > 0.01:
        
        # Now fold the spectrum to get a new spectrum with 2^(n+1) points
        n = len(y)
        yfold = np.append(y, np.fliplr([y[1:n-1]])[0])
        # Compute the interferogram
        n = len(yfold)
        inter = np.fft.ifft(yfold)*len(yfold)
        yyi = np.real(np.roll(inter, int(-1*n/2)))

        # Now we want to zeropad the spectrum to have 2^14 points
        # so we need to figure out how many zeros to put at the ends of
        # the interferogram. And we need to keep track of the factor that
        # we are expanding the interferogram by so we can multiply the
        # spectrum by it in a later step to put the energy back into it.
        
        for i in range(101):
            if 2**i >= len(x):
                break
        if i < 18:
            npts = 2**18 - 2**i
            factor = 2**18/(2**i)
            yyi_pad = np.append(np.zeros(int(npts/2)), yyi)
            yyi_pad = np.append(yyi_pad, np.zeros(int(npts/2)))
        else:
            factor = 1
            yyi_pad = np.copy(yyi)
        
        # Now compute the spectrum from this zeropadded spectrum
        n_pad = len(yyi_pad)
        yyi_pad_shift = np.roll(yyi_pad, int(n_pad/2))
        new_spec = np.fft.fft(yyi_pad_shift)/len(yyi_pad_shift)
        new_dv = delx/np.float(factor)
        
        # Cut the spectrum down (i.e., throw away the folded part)
        # and put the energy back in
        
        new_x = np.arange(len(new_spec)/2) * new_dv
        new_y = factor * np.real(new_spec[0:int(len(new_spec)/2)])
        new_delx = np.mean(new_x[1:len(new_x)] - new_x[0:len(new_x)-1])
    
    else:
        # Just reassign the vectors, as the spectral resolution is high enough
        
        new_x = np.copy(x)
        new_y = np.copy(y)
        new_delx = delx

    if write_debug_ringing == 1:
        print(f"DEBUG: orig_delx is {delx:f} and new_delx is {new_delx:f}")
        print(f"DEBUG: writing out the spectrum before the zeropad happens")
        Output_Functions.write_variable(new_x,'/data/wnum_after_zeropad.cdf')
        Output_Functions.write_variable(new_y,'/data/spec_after_zeropad.cdf')
    
    # Now we need to interpolate this spectrum to a multiple of the AERI's
    # wavenumber grid (power of 2, actually).
    if (aeri_dv / new_delx) > 256:
        sfac = 256.
    elif (aeri_dv / new_delx) > 128:
        sfac = 128.
    elif (aeri_dv / new_delx) > 64:
        sfac = 64.
    elif (aeri_dv / new_delx) > 32:
        sfac = 32.
    elif (aeri_dv / new_delx) > 16:
        sfac = 16.
    else:
        # If this happens, warn the user and still use sfac=16, but
        # this will not be an optimal interpolation
        if(aeri_dv / new_delx) < 8:
            print('Warning in convolve_to_aeri: Unanticipated problem in computing new_x')
        sfac = 16.
    
    new_aeri_dv = aeri_dv / sfac
    max_v = np.max(new_x)
    new_aeri_wnum = np.arange(int(max_v / new_aeri_dv)) * new_aeri_dv
    func = scipy.interpolate.interp1d(new_x,new_y, fill_value = 'extrapolate')
    new_aeri_spec = func(new_aeri_wnum)
    
    # In our desire to have a spectrum with 2^n + 1 points, I may need
    # to throw away a few points (but these are probably in the taper)
    for i in range(101):
        if 2**i >= len(new_aeri_wnum):
            break
    npts = 2**(i-1)
    new_aeri_wnum = new_aeri_wnum[0:npts+1]
    new_aeri_spec = new_aeri_spec[0:npts+1]
    
    # Now fold this spectrum, and compute its interferogrm
    n_aeri = len(new_aeri_spec)
    new_aeri_spec_fold = np.append(new_aeri_spec, np.fliplr([new_aeri_spec[1:n_aeri-1]])[0])
    n_fold = len(new_aeri_spec_fold)
    new_aeri_inter = np.fft.ifft(new_aeri_spec_fold) * len(new_aeri_spec_fold)
    new_aeri_inter = np.real(new_aeri_inter)
    new_aeri_inter = np.roll(new_aeri_inter, -1*int(n_fold/2))
    new_aeri_opd = 1./(2*new_aeri_dv)
    new_aeri_xx = ((np.arange(n_fold)/np.float(n_fold)) * 2  - 1) * new_aeri_opd
    
    # This is where I insert the triangle function to get a sinc^2 instrument function
    if do_sinc2 == 1:
        triangle_multiplier = np.interp(new_aeri_xx, [-2*aeri_opd,-1*aeri_opd,0,aeri_opd,2*aeri_opd], [0.,0,1,0,0])
        new_aeri_inter *= triangle_multiplier
    
    # Now chop this at the desired optical path delay
    foo = np.where((-1*aeri_opd <= new_aeri_xx) & (new_aeri_xx < aeri_opd))[0]
    aeri_chop_inter = np.copy(new_aeri_inter[foo])
    
    # And transform back into the spectral domain
    n_chop = len(aeri_chop_inter)
    aeri_chop_inter_shift = np.roll(aeri_chop_inter, int(n_chop/2))
    final_aeri_spec = np.fft.fft(aeri_chop_inter_shift)/len(aeri_chop_inter_shift)
    final_aeri_spec = np.real(final_aeri_spec)
    
    # Compute the scale factor that will account for the energy redistribution
    factor = len(final_aeri_spec) / np.float(len(new_aeri_inter))
    final_aeri_spec = factor * final_aeri_spec
    
    # And compute the wavenumber grid for this data
    final_aeri_dv = 1./(2*aeri_opd)
    final_aeri_wnum = np.arange(int(len(final_aeri_spec)/2)) * final_aeri_dv
    final_aeri_spec = final_aeri_spec[0:len(final_aeri_wnum)]
    
    # And the last step: cut off data before and after the actual
    # minimum and maximum wavenumber intervals of the input data
    foo = np.where((minv+tapersize <= final_aeri_wnum) & (final_aeri_wnum <= maxv-tapersize))[0]
    if len(foo) == 0:
        print('ERROR: the spectral range is smaller than twice the tapersize -- aborting')
        return 0
    else:
        final_aeri_wnum = final_aeri_wnum[foo]
        final_aeri_spec = final_aeri_spec[foo]
    
    if write_debug_ringing == 1:
        Output_Functions.write_variable(final_aeri_wnum,'/data/final_wnum.cdf')
        Output_Functions.write_variable(final_aeri_spec,'/data/final_spec.cdf')
        sys.exit()

    return {'wnum':final_aeri_wnum, 'spec':final_aeri_spec}

################################################################################
# This routine writes a netCDF file that looks like an ARM radiosonde file.
################################################################################

def write_arm_sonde_file(z, p, t, u, name, glatt=None, base_time=None, time_offset=None, silent=False):
    
    if (len(z) != len(p)) | (len(z) != len(t)) | (len(z) != len(u)):
        print('Error: Profiles are not the same length')
        print('No radiosonde netCDF file was created')
        return
    
    if not silent:
        print('Creating the file ' + name)
    
    fid = Dataset(name, 'w', format='NETCDF3_CLASSIC')
    did = fid.createDimension('time',None)
    b_time = fid.createVariable('base_time','i4')
    b_time.long_name = 'Time since 1970-01-01 00:00:00'
    b_time.units = 'seconds'
    if base_time is None:
        b_time.comment = 'This is a dummy field and has no real meaning it is only included to match the ARM data format'
    t_offset = fid.createVariable('time_offset', 'f8', ('time',))
    t_offset.long_name = 'Time since base_time'
    t_offset.units = 'seconds'
    if time_offset is None:
        t_offset.comment = 'This is a dummy field and has no real meaning it is only included to match the ARM data format'
    pres = fid.createVariable('pres', 'f4', ('time',))
    pres.long_name = 'Pressure'
    pres.units = 'hPa'
    tdry = fid.createVariable('tdry', 'f4', ('time',))
    tdry.long_name = 'Dry bulb temperature (i.e., ambient temperature)'
    tdry.units = 'C'
    rh = fid.createVariable('rh', 'f4', ('time',))
    rh.long_name = 'Relative humidity'
    rh.units = '%'
    alt = fid.createVariable('alt', 'f4', ('time',))
    alt.long_name = 'Altitude'
    alt.units = 'm MSL'
    if glatt is None:
        fid.comment = 'ad-hoc created sonde'
    else:
        comment = ''
        for i in range(glatt):
            comment = comment + glatt[i] + '; '
        fid.comment = comment
    if base_time is not None:
        b_time = base_time
    else:
        b_time = 0
    if time_offset is not None:
        if len(time_offset) == len(z):
            t_offset[0:len(z)] = time_offset
        else:
            t_offset[0:len(z)] = np.arange(len(z))*1.0
    else:
        t_offset[0:len(z)] = np.arange(len(z))*1.0
    
    pres[0:len(p)] = p[:]
    tdry[0:len(t)]= t[:]
    rh[0:len(u)] = u[:]
    alt[0:len(z)] = z[:]
    fid.close()

################################################################################
# The function to compute radiative transfer in the microwave regime
################################################################################

def radxfer_microwave(freq, t, od, sfc_t=None, sfc_e=None, upwelling=None):
    
    # Some basic checks to ensure that the arrays are dimesioned correctly
    if type(freq) != np.ndarray:
        opd = np.copy(od)
        a = opd.shape
        if len(a) != 1:
            print('Error: Optical depth array dimension does not match wnum dimension')
            return -1
    else:
        opd = np.copy(od)
        a = opd.shape
        if a[0] != len(freq):
            opd = opd.T
            a = opd.shape
            if a[0] != len(freq):
                print('Error: Optical depth array dimension does not match wnum dimension')
                return -3
    
    if a[0] != len(t)-1:
        if len(a) == 2:
            opd = opd.T
            a = opd.shape
        if a[0] == len(t)-1:
            looking_good = 1
        else:
            print('Error: Optical depth array dimension does not match temperature dimension')
            return -2
    foo = np.where(t < 0)[0]
    if len(foo) > 0:
        print('Error: temperature array has negative values; units should be Kelvin')
        return -4
    
    if upwelling:
        if sfc_e is None:
            if freq != np.ndarray:
                sfce = 1.0
            else:   
                sfce = np.ones(len(freq))
        else:
            sfce = np.copy(sfc_e)
        foo = np.where((sfce < 0) | (sfce > 1))
        if len(foo) > 0:
            print('Error: surface emissivity is outside [0,1]')
            return -5
            
        # Make sure that the surface emissivity is the right size
        if type(freq) != np.ndarray:
            if sfce == np.ndarray:
                print('Error: surface emissivity does not have the same dimension size as wnum')
                return -6
        else:
            if len(sfce) != len(freq):
                print('Error: surface emissivity does not have the same dimension size as wnum')
            return -6
        
        # If no surface temperatue is given then use temp of lowest level
        if sfc_t is None:
            sfct = t[0]
        else:
            sfct = np.copy(sfc_t)
        
        # set the indices for the loop
        k_start = 0
        k_end = len(opd[:,0])
        k_step = 1
    
    else:
        # No surface emission since this is downwelling
        if freq != np.ndarray:
            sfce = 0.0
        else:   
            sfce = np.zeros(len(freq))
        sfct = 2.75 # K, Temperature of deep space...
        
        # Set the indices for the loop
        k_start = len(opd[:,0])-1
        k_end = -1
        k_step = -1
    
    # Get the radiance for a blackbody emitting at sfct. Note that
    # this isn't correct for downwelling radiation, but in that case
    # the emissivity is zero so it doesn't matter
    
    sfcb = Calcs_Conversions.planck_microwave(freq,sfct)
    
    # Now we are set to do the radiative transfer
    # Will use the linear in tau approach
    foo = np.where(opd < 1.0e-6)[0]
    opd[foo] = 1.0e-6       # Add a small fudge factor to avoid singularity
    opd = opd.T
    rad = sfcb * sfce
    
    for k in range(k_start, k_end, k_step):
        # Get the transmission of the layer
        trans = np.exp(-1*opd[:,k])
        
        # Compute the blackbody radiance at the boundary closet to the observer
        if upwelling:
            b_boundary = Calcs_Conversions.planck_microwave(freq,t[k+1])
        else:
            b_boundary = Calcs_Conversions.planck_microwave(freq,t[k])
        
        # Compute the blackbody radiance at the middle of the layer (average temp)
        b_avg = Calcs_Conversions.planck_microwave(freq, (t[k]+t[k+1])/2.)
        
        # Compute the radiance using Eq 13 in Clough et al 1992
        rad = rad*trans + (1.-trans)*(b_boundary+2*(b_avg-b_boundary)*(1./opd[:,k] - trans/(1.-trans)))
    
    tb = Calcs_Conversions.inv_planck_microwave(freq, rad)
    
    return tb

################################################################################
# This function is part of the L-curve logic. It computes a curvature
# value as a function of the gamma value assumed. It came straight from
# the paper by Carissimo et al. 2005.
################################################################################

def compute_cgamma(ggamma, Xa, V, w, G, z):
    
    cgamma = np.zeros(len(ggamma))
    for ig in range(ggamma):
        summ = np.diag(1./(ggamma[ig]+w))
        u0 = V.dot(summ).dot(V.T).dot(G.T).dot(z)
        u1 = -1 * (V.dot(summ).dot(V.T).dot(u0))
        u2 = -2 * (V.dot(summ).dot(V.T).dot(u1))
        a1 = u1.T.dot(u0) + u0.T.dot(u1)
        a2 = u2.T.dot(u0) + u0.T.dot(u2) + 2*(u1.T.dot(u1))
        b1 = (G.dot(u1)).T.dot(G.dot(u0)-z) + (G.dot(u0)-z).T.dot(G.dot(u1))
        b2 = (G.dot(u2)).T.dot(G.dot(u0)-z) + (G.dot(u0)-z).T.dot(G.dot(u2)) + 2*(G.dot(u1)).T.dot(G.dot(u1))
        cgamma[ig] = np.abs(a1*b2 - a2*b1) / ((a1**2 + b1**2)**(1.5))
    
    return cgamma
    
################################################################################
# This function is start of L-curve logic. It computes the components
# needed to find the optimal gamma value, following the logic in the paper
# by Carissimo et al. 2005.
################################################################################

def lcurve(ggamma, flagY, Y0, FXn0, Kij0, Xn0, Xa0, Sa0, Sm0, z0):
    
    # Keep on the state data associated with T and q profiless
    
    k = len(z0)
    Kij = np.copy(Kij0[:,0:2*k])
    Xn = np.copy(Xn0[0:2*k])
    Xa = np.copy(Xa0[0:2*k])
    Sa = np.copy(Sa0[0:2*k,0:2*k])
    
    # Keep only the obs data associated with IRS radiances
    foo = np.where(flagY == 1)
    Y = np.copy(Y0[foo])
    FXn = np.copy(FXn0[foo])
    Kij = np.copy(Kij[foo,:])
    Sm = np.copy(Sm0[foo,foo])
    
    # Do the Carissimo math
    Yn = (Y-FXn)[:,None] + np.dot(Kij,(Xn-Xa)[:,None])
    sqrt_Sm = scipy.linalg.sqrtm(Sm)
    sqrt_Sa = scipy.linalg.sqrtm(Sa)
    z = np.dot(scipy.linalg.inv(sqrt_Sm), Yn) 
    Ji = np.dot(scipy.linalg.inv(sqrt_Sm), Kij)
    G = np.dot(Ji,sqrt_Sa)
    U,w,V = scipy.linalg.svd(np.dot(G.T, G))
    cgamma = compute_cgamma(ggamma, Xa, V, w, G, z)
    idx = np.where(cgamma == np.max(cgamma))[0][0]
    gfac = ggamma[idx]         # this is the optimal gamma value
    print('Best gamma value using Carissimo logic is ' + str(gfac))
    cgamma /= cgamma[idx]
    
    # Compute the actual L-curve with the two norms
    invSa = scipy.linalg.inv(Sa)
    invSm = scipy.linalg.inv(Sm)
    norm1 = np.zeros(len(ggamma))
    norm2 = np.zeros(len(ggamma))
    for ig in range(len(ggamma)):
        Xnp1 = (scipy.linalg.inv(ggamma[ig]*invSa + Kij.T.dot(invSm).dot(Kij))).dot(Kij.T).dot(invSm).dot(Yn)
        norm1[ig] = Xnp1.T.dot(invSa).dot(Xnp1)
        #tmp = Kij.dot(Xnp1) - Yn                  As stated in Carissimo paper
        tmp = Kij.dot(Xnp1) - Y                # What I think it should be based on Hansen paper
        norm2[ig] = tmp.T.dot(invSm).dot(tmp)
    
    n = len(ggamma)
    norm1 = np.log10(norm1)
    norm1 = np.log10(norm2)
    
    # This is my (DDT's) way to determine the best gamma value by looking for the 
    # sharpest kink in the L-curve.  I do that by normalizing the length of
    # both legs of the L, and then looking for the point that is closest in
    # Euclidean distance to the origin (as was suggested by Hansen as one way
    # to do this).  Note I tried multiple ways to find the L-curve point including
    # looking at 2nd derivatives, rotating the L-curve spatially, etc. and this
    # method worked the best.  But I had to change the "norm2" calculation to be
    #  Kij # Xnp1 - Y (instead of subtracting the Yn) -- this seemed to be much
    # more in the spirit of how the method was supposed to work.
    #     This actually worked very well, much better than the Carissimo logic.
    #     It seems that the Carissimo logic always picks the point with the
    #     largest gamma value.  This works well, but as the iterations increase
    #     and thus Y and FXn get closer to each other, it seems that the gamma
    #     value found by the L-curve method does not decrease like I think it 
    #     should.  This makes the entire method kind of hokey.
    
    x1 = np.copy(norm1)                      # normalized
    y1 = np.copy(norm2)                      # shifted
    x11 = (x1*((np.max(y1)-np.min(y1))/(np.max(x1)-np.min(x1))))       # normalized
    y11 = y1 - np.median(y1) + np.median(x11)                          # shifted
    r1 = np.sqrt(x11**2 + y11**2)                                      # Euclidean distance
    ridx = np.where(r1 == np.min(r1))[0][0]
    print('Best gamma value using Turner logic is ' + ggamma[ridx])
    
    return ggamma[ridx]
    
################################################################################
# The logic here is to only compute vres for the T and Q profiles, not from 
# any of the other variables. I am taking this logic from Hewison TGRS 2007,
# which is simpler than FwHM logic I was previously using...
################################################################################

def compute_vres_from_akern(akern, z, do_cdfs = False, do_area = False):
    
    vres = np.zeros((2,len(z)))
    cdfs = np.zeros((2,len(z)))
    area = np.zeros((2,len(z)))
    
    # First get the vertical spacing between the retrieval levels
    k = len(z)
    zres = [(z[1]-z[0])/2.]
    for i in range(1,k-1):
        zres.append((z[i+1]-z[i-1])/2.)
    
    zres.append((z[i]-z[i-1])*2)
    zres = np.array(zres)
    
    # Now scale the diagonal of the averaging kernal by this vertical spacing
    mval = 0.0001	# This is the minimum value for the DFS for the first level
    if akern[0,0] > mval:
        tval = akern[0,0]
    else:
        tval = mval
    if akern[k,k] > mval:
        qval = akern[k,k]
    else:
        qval = mval
    for i in range(k):
        # Watch for zeros along the averaging kernal diagonal. If that
        # happens, then use the last good value for the vres calculation
        if akern[i,i] > 0:
           tval = akern[i,i]
        if akern[k+i,k+i] > 0:
           qval = akern[k+i,k+i]
        # capture the cumulative DFS profile
        if(i == 0):
            cdfs[0,i] = tval
            cdfs[1,i] = qval
        else:
            cdfs[0,i] = tval + cdfs[0,i-1]
            cdfs[1,i] = qval + cdfs[1,i-1]

        # This is the Hewison method
        vres[0,i] = zres[i] / tval            # temperature profile
        vres[1,i] = zres[i] / qval        # water vapor profile
    
    # Now compute the area of the averaging kernal (pg 56 in Rodgers)
    tmp = np.copy(akern[0:k,0:k])
    area[0,:] = np.squeeze(tmp.dot(np.ones(k)[:,None]))
    tmp = np.copy(akern[k:2*k,k:2*k])
    area[1,:] = np.squeeze(tmp.dot(np.ones(k)[:,None]))
    
    if do_cdfs and do_area:
        return vres, area, cdfs
    elif do_cdfs:
        return vres, cdfs
    elif do_area:
        return vres, area
    else:
        return vres

################################################################################
# Takes an AERI-like spectrum and performs Fourier interpolation from the nominal
# order 2,000 to 4,000 points to 2^14+1 points.  Copied from Paul van Delst's code
#  Inputs:
#      iwnum is in cm-1
#      irad  is in mW/(m2 sr cm-1)
#      channel is 1 or 2 (like the AERI)
#      epad is the number of wavenumbers to pad at the end of the input spectrum
#  Output:
#       2xn vector, where output(0,:) is the new wnum array
#                     and output(1,:) is the new radiance spectrum
################################################################################
def aeri_zerofill(iwnum,irad,channel,epad=0):
    eepad = np.abs(epad)
    band  = channel - 1
    v_laser = 15799.0
    rf      = [4.0, 2.0]
    v_nyquist = v_laser / (2. * rf[band])
    if channel == 1: n_pts = 4097
    elif channel == 2: n_pts = 8193
    else:
        print('ERROR in aeri_zerofill -- channel is not properly defined')
        sys.exit()
    
    v = v_nyquist * np.arange(n_pts) / (n_pts - 1)

    # Determine the temperature to use for the Planck function,
    # which is used to help pad the data
    v_teststart = [627.0, 2380.]
    v_testend   = [632.0, 2384.]
    loc = np.where((v_teststart[band] <= iwnum) & (iwnum <= v_testend[band]))
    tb = Calcs_Conversions.invplanck(iwnum[loc], irad[loc])
    avg_tb = np.mean(tb)

    # Quick trap to prevent a divide-by-zero error in the planck function
    if(v[0] < 0.00001):
        orig_v0 = v[0]
        v[0] = v[1]
        replace_v0 = True
    else:
        replace_v0 = False
    r = Calcs_Conversions.planck(v, avg_tb)
    if(replace_v0):
        v[0] = orig_v0
        r[0] = 0.

    # Determine the bounds of the AERI data
    n_apts = len(iwnum)
    thres = 0.1
    count = 0
    ntries = 0
    while (count != n_apts) & (ntries < 10):
        loc = np.where((iwnum[0]-thres <= v) & (v <= iwnum[len(iwnum)-1]+thres))[0]
        count = len(loc)
        thres += 0.05
        ntries += 1

        if ntries >= 10:
            print('Error: unable to find the right bounds for the AERI in aeri_zerofill')
            sys.exit()

    # Replace the AERI data in the planck radiance curve
    r[loc] = irad

    # Fourier transform the radiance data
    rr = np.append(r, np.flip(r[1:n_pts-1]))
    # Quick trap to make sure we have this size right
    if(len(rr) != 2*n_pts-2):
        print('Problem in aeri_zerofill: the length of the appended vector is not correct 1')
        sys.exit()
    ifg = np.fft.fft(rr, norm='forward')
    
    # Zerofill the inteferogram
    n_total = 2**18
    fill = np.zeros(n_total - len(ifg))
    ifg  = np.concatenate((ifg[0:n_pts], fill, ifg[n_pts:2*(n_pts-1)]))
    # Quick trap to make sure we have this size right
    if(len(ifg) != n_total):
        print('Problem in aeri_zerofill: the length of the appended vector is not correct 2')
        sys.exit()

    # Fourier transform the zerofilled interferogram
    spc = np.fft.ifft(ifg, norm='forward')
    spc = spc[0:int(n_total/2)]
    v2 = v_nyquist * np.arange(n_total/2) / (n_total/2)
    loc = np.where((iwnum[0]-eepad <= v2) & (v2 <= iwnum[n_apts-1]+eepad))
   
    return np.array([v2[loc],np.real(spc[loc])])

################################################################################
# This function spectrally calibrates the AERi.  It requires input from a routine
# like "check_aeri_vlaser".  It only works with one channel at a time, and one
# spectrum at a time
################################################################################
def fix_aeri_vlaser_mod(wnum,irad,multiplier):
    # determine which channels this is
    foo = np.where((wnum > 900) & (wnum < 901))
    if(len(foo) > 0):
        channel = 1
    else:
        channel = 2

    result = aeri_zerofill(wnum, irad, channel)
    newrad = np.interp(wnum, result[0,:]*multiplier, result[1,:])

    return newrad

################################################################################
# This routine creates a vertical grid that can be added to the top of our profile
################################################################################
def compute_extra_layers(maxz):
    max_step_size = 5    # This is the maximum step size, in km
    max_ht        = 40   # This is the maximum height to take the extra RT layers
    stepsize = np.round(maxz / 4. * 10) / 10.
    rt_extra_layers = maxz + stepsize
    while(np.max(rt_extra_layers) < max_ht):
        stepsize = np.round(np.max(rt_extra_layers) / 4. * 10) / 10.
        if(stepsize > max_step_size):
            stepsize = max_step_size
        rt_extra_layers = np.append(rt_extra_layers, np.max(rt_extra_layers) + stepsize)
    
    return rt_extra_layers

################################################################################
# This function removes the original finite field of view correction applied 
# to the AERI data, and replaces it with a new one.  The correction is determined
# by the half angle of the field of view of the instrument (assuming the detector
# is perfectly on axis -- following Knuteson et al. 2004)
################################################################################
def change_irs_ffovc(wnum,orad,orig_halfAngle,new_halfAngle):
    print('Stubbing this change_irs_ffovc function out -- it does nothing')
    return orad

###############################################################################
# This routine tests whether an array is monotonic (returns TRUE) or not (FALSE)
###############################################################################
def test_monotonic(x,strict=True):
    dx = np.diff(x)
    if strict == True:
        return np.all(dx <  0) or np.all(dx >  0)
    else:
        return np.all(dx <= 0) or np.all(dx >= 0)

###############################################################################
# This function checks to make sure sounding data altitude is monotonic and if
# no it will return the indices to make the altitudes monotonic by removing the
# bad points. It is a bit hackish but it works.
###############################################################################
def make_monotonic(heights):
    
    z = np.copy(heights)
    indices = np.arange(len(z))
        
    # It is possible we will have to do this mulitple times (hopefully not)
    # This fixes those cases.
        
    bad = True          # A flag we will use to quit the loop
    while bad:
        foo = np.where(np.diff(z) <= 0)[0]
        if len(foo) > 0:
            z[foo+1] = -999
            fah = np.where(z != -999)[0]
            z = z[fah]
            indices = indices[fah]
        else:
            bad = False
        
    return indices

###############################################################################
# This function interpolates over nonphysical values. If the nonphysical values
# is at the ends of the profile then the nearest good point is used at the new
# value.
###############################################################################

def fix_nonphysical_wv(wv, z, Xa, foo):
    
    old_wv = np.copy(wv)
    old_wv[foo] = np.nan

        # Create a scaled version of the prior, using the
        # good values in the current profile to do the scaling
    sf = np.nanmean(old_wv / Xa)
    if np.isnan(sf):
        print('DDT -- this should not happen within fix_nonphysical_wv -- aborting to allow for debugging')
        sys.exit()
    newXa = Xa * sf

    new_wv = []
    for i, value in enumerate(old_wv):
        if np.isnan(value):
            # Find first good point
            j = i
            while np.isnan(old_wv[j]) and j > 0:
                j -= 1
        
            # Find second good point
            k = i
            while np.isnan(old_wv[k]) and k < len(old_wv) - 1:
                k += 1
        
            if np.isnan(old_wv[j]):
                # The first value is nonphysical so set to nearest good value after
                new_wv.append(newXa[i])
            elif np.isnan(old_wv[k]):
                # The last value is nonphysical so set to nearest good value before
                new_wv.append(newXa[i])
            else:
                # Interpolate across the good values
                new_wv.append(old_wv[j] + (old_wv[k] - old_wv[j]) * (z[i]-z[j])/(z[k]-z[j]))
        else:
            new_wv.append(value)
    
    return np.array(new_wv)

###############################################################################
# This function determines the parcel temperature that comes from pseudoadiabatic lapse
# rates. Adapted from Metpy.
###############################################################################

def moist_lapse(pressure,temperature,lcl):
    Rd = 287.04749
    Lv = 2.50084e6
    Cp_d = 1004.6662
    epsilon = 0.6219571
    
    # This requires us to use an ODE solver so need to define the function here
    def dt(p,t):
        rs = 0.001*Calcs_Conversions.wsat(t-273.16,p)
        frac = (
            (Rd * t + Lv * rs)
            / (Cp_d + (
                Lv*Lv*rs*epsilon
                / (Rd*t**2)
            ))
        )
        
        return frac/p
    
    tt = np.atleast_1d(temperature+273.16)
    pp = np.atleast_1d(pressure)
    
    # Put pressure in increasing order for now
    pp = pp[::-1]
    
    # We are going to use a ode solver
    solver_arg = {'fun': dt, 'y0': tt, 'method':'RK45', 'atol': 1e-7,
                  'rtol':1.5-8}
    
    # Need to handle the initial pressure point
    ret = np.broadcast_to(tt[:, np.newaxis], (tt.size,1))
    
    # Put pressure back in decreasing form
    pp = pp[::-1]
    # Because we know how this is getting passed we only have points above
    # the reference pressure, so integrate upward.
    
    trace = scipy.integrate.solve_ivp(t_span=(pp[0],pp[-1]),
                                      t_eval=pp[1:], **solver_arg).y
    
    ret = np.concatenate((ret,trace), axis=-1)
    
    return ret.squeeze()

###############################################################################
# This function finds the level of free convectin and the equilibrium level
# from a parcel profile and environental temperature profile. Adapted from Metpy.
###############################################################################
def find_lfc_el(t, p, parcel, lcl):
    
    
    # We first need to find the intersections between the parcel profiles and
    # the environment, but exclude first point in the search
    nearest_idx, = np.nonzero(np.diff(np.sign(parcel[1:]-t[1:])))
    
    # Need to add one here since the first point wasn't in the array
    nearest_idx = nearest_idx+1
    
    next_idx = nearest_idx + 1
   
    sign_change = np.sign(parcel[next_idx]-t[next_idx])
   
    # Calculate the x-intersecion
    x0 = np.log(p[nearest_idx])
    x1 = np.log(p[next_idx])
   
    a0 = parcel[nearest_idx]
    a1 = parcel[next_idx]
   
    b0 = t[nearest_idx]
    b1 = t[next_idx]
   
    delta_y0 = a0 - b0
    delta_y1 = a1 - b1
    intersect_x = (delta_y1*x0 - delta_y0*x1)/(delta_y1 - delta_y0)
    
    if len(intersect_x) == 0:
        increasing = np.array([])
        decreasing = np.array([])
    else:
        # Return the coordinates back to linear
        intersect_x = np.exp(intersect_x)
        
        # There shouldn't be masked values but just double check here
        duplicates = (np.ediff1d(intersect_x, to_end=1) != 0)
    
        intersect_x = intersect_x[duplicates]
        sign_change = sign_change[duplicates]
    
        increasing = intersect_x[sign_change > 0]
        decreasing = intersect_x[sign_change < 0]
        
        
    # if there is no intersections see if the lfc is the lcl 
    # and set the el to a nan
    if len(increasing) == 0:
        # Is there any positive area above the LCL
        mask = p < lcl
        if np.all((parcel[mask]<t[mask]) | (np.isclose(parcel[mask],t[mask]))):
            # LFC doesn't exist
            lfc = np.nan
        else:
            lfc = np.copy(lcl)
    
    else:
        # The LFC will be the lowest (so highest pressure), positive sign change
        # First we have to make sure that the LFC is greater or equal the LCL
        
        idx = increasing <= lcl
        
        if not any(idx):
            if len(decreasing) > 0:
                if np.min(decreasing) > lcl:
                    lfc = np.nan
                else:
                    lfc = np.copy(lcl)
            else:
                lfc = np.nan
        else:
            lfc = np.nanmax(increasing[idx])
    
    # Now find the EL
    if len(decreasing) > 0 and decreasing[-1] < lcl:
        el = np.nanmin(decreasing)
    else:
        el = np.nan
    
    return lfc, el
                
    
###############################################################################
# This function calculates SBCAPE and CIN from the retrieved profiles. We assume
# that the LCL pressure has already been calculated. Portions of this function
# were adapted from Metpy.
###############################################################################

def cape_cin(t,p,lcl):
    
    kappa = 0.286
    Rd = 287.04749
    
    # First we need to construct the parcel profile
    parcel = []
    
    # if the lcl wasn't found then the parcel profile is just dry adiabatic
    if lcl == -999:
        parcel.extend(((t+273.16)*((p/p[0])**kappa))-273.16)
    else:
        
        # Start with the temperature profile from surface to lcl which is
        # dry adiabatic
        foo = np.where((p > lcl) & (~np.isclose(p,lcl)))[0]
        
        parcel.extend(((t[0]+273.16)*((p[foo]/p[0])**kappa))-273.16)
        
        # Add in the LCL
        
        parcel.append(((t[0]+273.16)*((lcl/p[0])**kappa))-273.16)
        
        foo = np.where((p < lcl) & (~np.isclose(p,lcl)))[0]
        
        # If there are pressures above the LCL add pseudo-adiabatic ascent
        if len(foo) != 0:
            parcel.extend(moist_lapse(np.append(np.atleast_1d(lcl),p[foo])
                                      ,parcel[-1], lcl)[1:]-273.16)
        
        parcel = np.array(parcel)
        
        
        # We need to make sure that the LCL is included in the environment
        # temperature and pressure
        if any(np.isclose(p,lcl)):
            new_p = np.copy(p)
            new_t = np.copy(t)
        
        else:
            # Need to add in the LCL. We have the index where it needs to be
            # inserted stored in foo
            
            if len(foo) != 0:
                new_p = np.insert(p,foo[0],lcl)
                new_t = np.insert(t,foo[0],np.interp(lcl,p[::-1],t[::-1]))
            else:
                # The lcl is above the input profile so we will not include
                # it. This shouldn't happen, but leaving this in just in case.
                new_p = np.copy(p)
                new_t = np.copy(t)
                
                
        # Now we need to find the LFC and EL
        lfc, el = find_lfc_el(new_t, new_p, parcel,lcl)
        
        # If ther is no LFC then there is no CAPE or CIN and can return zeros
        if np.isnan(lfc):
            return 0, 0
        
        # If there is no EL then use the top of the profile
        if np.isnan(el):
            el = new_p[-1]
        
        # Difference between the parcel path and measured temperature profiles
        y = parcel - new_t
        
        # We need to find the zero crossings.
        zeros = np.zeros_like(y)
        nearest_idx, = np.nonzero(np.diff(np.sign(y[1:]-zeros[1:])))
        nearest_idx += 1
        next_idx = nearest_idx +1
       
        # Calculate the x-intersecion
        x0 = np.log(new_p[nearest_idx])
        x1 = np.log(new_p[next_idx])
       
        a0 = y[nearest_idx]
        a1 = y[next_idx]
       
        b0 = zeros[nearest_idx]
        b1 = zeros[next_idx]
       
        delta_y0 = a0 - b0
        delta_y1 = a1 - b1
        intersect_x = (delta_y1*x0 - delta_y0*x1)/(delta_y1 - delta_y0)
        
        intersect_y = ((intersect_x - x0) / (x1 - x0)) * (a1 - a0) + a0
        
        # Return the coordinates back to linear
        intersect_x = np.exp(intersect_x)
        
        if len(intersect_x) != 0:
        # There shouldn't be duplicates but just double check here
            duplicates = (np.ediff1d(intersect_x, to_end=1) != 0)
            intersect_x = intersect_x[duplicates]
            intersect_y = intersect_y[duplicates]
        
        x = np.concatenate((p, intersect_x))
        y = np.concatenate((y, intersect_y))
        
        # Resort so that the data are in order
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        y = y[sort_idx]
        
        # Remove duplicates data points if there are any
        keep_idx = np.ediff1d(x, to_end=[1]) > 1e-6
        x = x[keep_idx]
        y = y[keep_idx]
        
        # CAPE
        # Only use data between the LFC and EL for calculation
        p_mask = np.where(((x < lfc) | (np.isclose(x,lfc))) & 
                          ((x>el) | (np.isclose(x,el))))
        x_clipped = x[p_mask]
        y_clipped = y[p_mask]
        cape = Rd*np.trapz(y_clipped,np.log(x_clipped))
        
        # CIN
        # Only use data between the surface and LFC for calculation
        p_mask = np.where(((x>lfc) | (np.isclose(x,lfc))))
        x_clipped = x[p_mask]
        y_clipped = y[p_mask]
        cin = Rd*np.trapz(y_clipped,np.log(x_clipped)) 
        
        if cin > 0:
            cin = 0
            
        if cape < 0:
            cape = 0
            cin = 0
            
        return cape, cin

###############################################################################
# This function gives the temperature and water vapor profile that is used to 
# calculate MLCAPE and CIN.
###############################################################################

def mixed_layer(t,p,wv, depth = 100):
    
    # Convert the temperature to potential temperature
    theta = Calcs_Conversions.t2theta(t, np.zeros(len(t)), p)
    
    p_top = p[0] - depth
    
    p_layer = p[p>=p_top]
    t_layer = theta[p>=p_top]
    wv_layer = wv[p>=p_top]
    
    # Check to make sure that the top pressure is in the the layer. If not 
    # interpolate to the top pressure.
    
    if not np.any(np.isclose(p_top,p_layer)):
        p_layer = np.append(p_layer, p_top)
        t_layer = np.append(t_layer, np.interp(np.log(p_top),np.log(p[::-1]),theta[::-1]))
        wv_layer = np.append(wv_layer, np.interp(np.log(p_top),np.log(p[::-1]),wv[::-1]))
      
    t_mix = -np.trapz(t_layer,p_layer)/np.abs(p_layer[0]-p_layer[-1])
    wv_mix = -np.trapz(wv_layer,p_layer)/np.abs(p_layer[0]-p_layer[-1])
    
    # Convert t_mix back to temperature
    t_mix = Calcs_Conversions.theta2t(t_mix, 0, p_layer[0])
    
    pressure_prof = p[p < (p[0]-depth)]
    temp_prof = t[p < (p[0]-depth)]
    wv_prof = wv[p < (p[0]-depth)]
    
    pressure_prof = np.append(np.atleast_1d(p_layer[0]),pressure_prof)
    temp_prof = np.append(np.atleast_1d(t_mix),temp_prof)
    wv_prof = np.append(np.atleast_1d(wv_mix), wv_prof)
    
    return pressure_prof, temp_prof, wv_prof

###############################################################################
# This function is the driver for calculating the derived indices that are 
# provided with the final output. It performs a simple monte carlo sampling to 
# derive the uncertanties in the indices. Note that even though the uncertainties
# might not be Gaussian distributed, I am going to report a 1-sigma standard
# deviation.
###############################################################################

def calc_derived_indices(xret, vip, derived, verbose, do_compute, num_mc=20):
    
    # These are the derived indices that I will compute later one. I need to
    # define them here in order to build the netcdf file correctly
    dindex_name = ['pwv', 'pblh', 'sbih', 'sbim', 'sbLCL', 'sbCAPE', 'sbCIN', 
                   'mlLCL','mlCAPE', 'mlCIN']
    dindex_units = ['cm', 'km', 'km', 'degC', 'km', 'J/kg', 'J/kg',
                    'km', 'J/kg', 'J/kg']
    
    indices = np.zeros(len(dindex_name))-999.
    sigma_indices = np.zeros(len(dindex_name))-999.

    # If we don't compute thense, then just return these missing values
    if(do_compute != 1):
        return {'indices':indices, 'sigma_indices':sigma_indices, 'name':dindex_name, 'units':dindex_units}
    
    # Get the number of height levels
    nht = len(xret['z'])
    
    # Extract out the temperature and water vapor profiles
    pp = np.copy(xret['p'])
    tt = np.copy(xret['Xn'][0:nht])
    ww = np.copy(xret['Xn'][nht:2*nht])
    zz = np.copy(xret['z'])

    # Extract out the posterior covariance matrix, and get the 1-sigma uncertainties
    Sop_tmp = np.copy(xret['Sop'][0:2*nht,0:2*nht])
    sig_t = np.sqrt(np.diag(Sop_tmp))[0:nht]
    
    # Now compute the indices
    
    # PWV
    indices[0] = Calcs_Conversions.w2pwv(ww,pp)
    
    # PBLH
    indices[1] = compute_pblh(zz, tt, pp, sig_t, minht=vip['min_PBL_height'],
                              maxht=vip['max_PBL_height'], nudge=vip['nudge_PBL_height'])
    
    # SBIH & SBIM
    indices[2], indices[3] = compute_sbi(zz,tt)
    
    # sbLCL
    indices[4], plcl = compute_lcl(tt[0],ww[0],pp[0],pp,zz)
    
    # sbCAPE, sbCIN
    # For now putting this in a try, except statment to make sure this
    # does cause TROPoe to quit if something goes wrong
    try:
        indices[5], indices[6] = cape_cin(tt, pp, plcl)
    except:
        print('Something went wrong in SBCAPE and CIN calculation.')
        indices[5] = -9999.
        indices[6] = -9999.
    
    try:
        # Find mixed layer profiles for the ML indices. 
        mlpp, mltt, mlww = mixed_layer(tt, pp, ww)
    
        # mlLCL
        indices[7], pmllcl = compute_lcl(mltt[0],mlww[0],mlpp[0],pp,zz)
    
        indices[8], indices[9] = cape_cin(mltt,mlpp, pmllcl)
    except:
        print('Something went wrong in MLCAPE and CIN calculation')
        indices[7] = -999.
        indices[8] = -9999.
        indices[9] = -9999.
    
    # Only compute the uncertainties in the indices if the number of Monte Carlo 
    # samples is strictly positive; otherwise, return -9999 as the uncertainties
    if num_mc <= 0:
        print('      Not computing the uncertainties in the derived indices')
        sigma_indices = np.zeros_like(indices) - 9999.
    else:
        # Perform SVD of posterior covariance matrix
        u, w, v, = scipy.linalg.svd(Sop_tmp.T,False)
    
        # Generate the Monte Carlo profiles
        b = np.random.default_rng().normal(size=(2*nht,num_mc))
        pert = u.dot(np.diag(np.sqrt(w))).dot(b)
        tprofs = tt[:,None] + pert[0:nht,:]
        wprofs = ww[:,None] + pert[nht:2*nht,:]
    
        # allocate room to compute the indices for the various profiles in the MC sampling
        tmp_pwv = np.zeros(num_mc)
        tmp_pblh = np.zeros(num_mc)
        tmp_sbih = np.zeros(num_mc)
        tmp_sbim = np.zeros(num_mc)
        tmp_lcl = np.zeros(num_mc)
        tmp_CAPE = np.zeros(num_mc)
        tmp_CIN = np.zeros(num_mc)
        tmp_mllcl = np.zeros(num_mc)
        tmp_MLCAPE = np.zeros(num_mc)
        tmp_MLCIN = np.zeros(num_mc)
                         
        for j in range(num_mc):
            tmp_pwv[j] = Calcs_Conversions.w2pwv(wprofs[:,j],pp)
            tmp_pblh[j] = compute_pblh(zz,tprofs[:,j],pp, sig_t, minht=vip['min_PBL_height'],
                                  maxht=vip['max_PBL_height'], nudge=vip['nudge_PBL_height'])
            tmp_sbih[j], tmp_sbim[j] = compute_sbi(zz,tprofs[:,j])
            tmp_lcl[j], tmp_plcl = compute_lcl(tprofs[0,j], wprofs[0,j], pp[0], pp, zz)
        
            try:
                tmp_CAPE[j], tmp_CIN[j] = cape_cin(tprofs[:,j], pp, tmp_plcl)
            except:
                if verbose > 1:
                    print('Something went wrong in SBCAPE and CIN error calculation.')
                tmp_CAPE[j] = -9999.
                tmp_CIN[j] = -9999.
        
            try:
                tmp_mlpp, tmp_mltt, tmp_mlww, = mixed_layer(tprofs[:,j], pp, wprofs[:,j])
                tmp_mllcl[j], tmp_pmllcl = compute_lcl(tmp_mltt[0], tmp_mlww[0], tmp_mlpp[0],pp,zz)
                tmp_MLCAPE[j], tmp_MLCIN[j] = cape_cin(tmp_mltt, tmp_mlpp, tmp_pmllcl)
            except:
                if verbose > 1:
                    print('Something went wrong in MLCAPE and CIN error calculation.')
                tmp_CAPE[j] = -9999.
                tmp_CIN[j] = -9999.
        
        # PWV
        sigma_indices[0] = np.nanstd(indices[0]-tmp_pwv)
    
        # PBLH
        foo = np.where(tmp_pblh > 0)[0]
        if ((len(foo) > 1) & (indices[1] > 0)):
            sigma_indices[1] = np.nanstd(indices[1]-tmp_pblh[foo])
        else:
            sigma_indices[1] = -999.
        if ((sigma_indices[1] < vip['min_PBL_height']) & (indices[1] <= vip['min_PBL_height'])):
            sigma_indices[1] = vip['min_PBL_height']
    
        # SBIH
        foo = np.where(tmp_sbih > 0)[0]
        if ((len(foo) > 1) & (indices[2] > 0)):
            sigma_indices[2] = np.nanstd(indices[2]-tmp_sbih[foo])
        else:
            sigma_indices[2] = -999.
    
        # SBIM
        foo = np.where(tmp_sbim > 0)[0]
        if ((len(foo) > 1) & (indices[3] > 0)):
            sigma_indices[3] = np.nanstd(indices[3]-tmp_sbim[foo])
        else:
            sigma_indices[3] = -999.
    
        # LCL
        foo = np.where(tmp_lcl > 0)[0]
        if ((len(foo) > 1) & (indices[4] > 0)):
            sigma_indices[4] = np.nanstd(indices[4]-tmp_lcl[foo])
        else:
            sigma_indices[4] = -999.
    
        # sbCAPE
        foo = np.where(tmp_CAPE >= 0)[0]
        if ((len(foo) > 1) & (indices[5] >= 0)):
            sigma_indices[5] = np.nanstd(indices[5]-tmp_CAPE[foo])
        else:
            sigma_indices[5] = -9999.
    
        # sbCIN
        foo = np.where(tmp_CIN >= -9000)[0]
        if ((len(foo) > 1) & (indices[6] >= -9000)):
            sigma_indices[6] = np.nanstd(indices[6]-tmp_CIN[foo])
        else:
            sigma_indices[6] = -9999.
    
        # mlLCL
        foo = np.where(tmp_mllcl > 0)[0]
        if ((len(foo) > 1) & (indices[7] > 0)):
            sigma_indices[7] = np.nanstd(indices[7]-tmp_mllcl[foo])
        else:
            sigma_indices[7] = -999.
        
        # MLCAPE
        foo = np.where(tmp_MLCAPE >= 0)[0]
        if ((len(foo) > 1) & (indices[8] >= 0)):
            sigma_indices[8] = np.nanstd(indices[8]-tmp_MLCAPE[foo])
        else:
            sigma_indices[8] = -9999.
    
        # MLCIN
        foo = np.where(tmp_CIN >= -9000)[0]
        if ((len(foo) > 1) & (indices[9] >= -9000)):
            sigma_indices[9] = np.nanstd(indices[9]-tmp_MLCIN[foo])
        else:
            sigma_indices[9] = -9999.
    
    return {'indices':indices, 'sigma_indices':sigma_indices, 'name':dindex_name,
            'units':dindex_units}
        
    
###############################################################################
# This function decomposes the keywords in the VIP file associated with the IRS
# band noise inflation, and returns a structure with these parameters
###############################################################################

def decompose_irs_band_noise_inflation(vip, verbose=2):
     err = {'success':0}
     onoff = vip['irs_band_noise_inflation']
     tmp   = vip['irs_band_noise_wnums']
     wnums = tmp.split(',')
     wnums = np.array(wnums).astype(np.float)
     if(len(wnums) != 2):
         print('Error: unable to find two wavenumbers in vip.irs_band_noise_wnums')
         return err
     if(wnums[0] >= wnums[1]):
         print('Error: vip.irs_band_noise_wnums[0] >= vip.irs_band_noise_wnums[1]')
         return err
     npts = vip['irs_band_noise_sfc_npts']
     tmp  = vip['irs_band_noise_sfc_wvmr_vals']
     wvmr = tmp.split(',')
     wvmr = np.array(wvmr).astype(np.float)
     if(len(wvmr) != npts):
         print('Error: number of elements in vip.irs_band_noise_sfc_wvmr_vals does not match vip.irs_band_noise_sfc_npts')
         return err
     tmp  = vip['irs_band_noise_sfc_multiplier']
     mult = tmp.split(',')
     mult = np.array(mult).astype(np.float)
     if(len(mult) != npts):
         print('Error: number of elements in vip.irs_band_noise_sfc_multiplier does not match vip.irs_band_noise_sfc_npts')
         return err
     out = {'success':1, 'onoff':onoff, 'wnum1':wnums[0], 'wnum2':wnums[1], 'npts':npts, 
         'wvmr':wvmr, 'multiplier':mult}
     if(verbose >= 2):
         print('  Inside decompose_irs_band_noise_inflation(), here are the values:')
         print('        onoff',out['onoff'])
         print('        wnum1',out['wnum1'])
         print('        wnum2',out['wnum2'])
         print('        npts',out['npts'])
         print('        wvmr',out['wvmr'])
         print('        multiplier',out['multiplier'])
     return out

###############################################################################
# This function inflates the noise from the "TROPoe_in" profile, using the logic
# from Adler et al. AMT 2024
###############################################################################

def inflate_in_tropoe_uncertainty(flag, sampleTime, in_tropoe, zSa, Sa, vip, verbose=2):
        # Quick check, but this should always be the case
    if len(zSa) != len(in_tropoe['height']):
        print('ERROR in inflate_in_tropoe_uncertainty: the height arrays from prior and retrieval are not the same -- aborting')
        sys.exit()

        # The temporal resolution of the retrieval
    tres = vip['tres']*60       # get the temporal resolution of the retrievals in seconds

        # Extract out the 1-sigma uncertainties from the prior
    Sa_sigT = np.sqrt(np.diag(Sa))[0:len(zSa)]
    Sa_sigq = np.sqrt(np.diag(Sa))[len(zSa):2*len(zSa)]
    Sa_sigc = np.sqrt(np.diag(Sa))[2*len(zSa)+4:2*len(zSa)+7]

        # Only apply logic if there is a valid profile in the in_tropoe structure (i.e., secs > 0)
    if in_tropoe['secs'] > 0:
                # Compute the time delta
        delt = np.abs(sampleTime - in_tropoe['secs'])
        if np.isclose(delt,0):
            delt = tres
                # Make sure this time delta makes sense
        if delt <= 0:
            print('ERROR in inflate_in_tropoe_uncertainty: the time delta must be strictly positive -- aborting')
            sys.exit()
                # Noise inflation factor
        inflateFactorTQ   = np.sqrt(1 + (delt - tres)/tres)
        inflateFactorCO2a = np.sqrt(1 + (delt - tres)/(vip['add_tropoe_co2_delay_value'][0]*tres))
        inflateFactorCO2b = np.sqrt(1 + (delt - tres)/(vip['add_tropoe_co2_delay_value'][1]*tres))
        if verbose > 1:
            print(f"      Inflating in_tropoe {flag:s} profile by a factor of {inflateFactorTQ:.2f} using delt = {delt:.1f} and tres = {tres:.1f}")
                # Determine what profile to scale
        if flag == 'T':
            eprofile = in_tropoe['sigma_temperature']
            scaleht  = vip['add_tropoe_T_noise_adder_hts']
            scaleval = vip['add_tropoe_T_noise_adder_val']
            minvalue = 0.0      # Will require that the adder always be positive
        elif flag == 'q':
            eprofile = in_tropoe['sigma_waterVapor']
            scaleht  = vip['add_tropoe_q_noise_mult_hts']
            scaleval = vip['add_tropoe_q_noise_mult_val']
            minvalue = 1.0      # Will require that the multiplier always be above 1
        elif flag == 'co2':
            eprofile = in_tropoe['sigma_co2']
            scaleval = vip['add_tropoe_co2_noise_adder']
            minvalue = 0.0      # Will require that the adder always be positive
        else:
            print('Error in inflate_in_tropoe_uncertainty: the flag selecting temp, WVMR, or co2 was not set properly -- aborting')
            sys.exit()
                # The scaleht profiles are all 3-pt arrays.  If the VIP.add_tropoe_use_pblh_flag is set, then replace the middle
                # then replace the center value with the max{in_tropoe.pblh,scaleht(1)}
        if((vip['add_tropoe_use_pblh_flag'] == 1) & (scaleht[1] < in_tropoe['pblh'])):
                scaleht[1] = in_tropoe['pblh']
        if verbose > 1:
            print(f"        Using {scaleht[1]:3f} km as the pivot point in the inflation of {flag:s} from previous TROPoe input profile")
                # Interpolate the scale factor profile to the current height grid,
                # but trap any points that go below the minimum value and reset them
        if((flag == 'T') | (flag == 'q')):
            sf = np.interp(in_tropoe['height'],scaleht,scaleval)
            foo = np.where(sf < minvalue) [0]
            if len(foo) > 0:
                sf[foo] = minvalue
                    # Now inflate this noise profile by the inflation factor, which is govered by the time difference
            sf = sf * inflateFactorTQ
        else:
            sf[0] = scaleval[0] * inflateFactorCO2a
            sf[1] = scaleval[1] * inflateFactorCO2b
            foo = np.where(sf < minvalue) [0]
            if len(foo) > 0:
                sf[foo] = minvalue

                # Add or multiply the error profile with this new scale factor profile
        if flag == 'T':
            eprofile  = eprofile + sf
            mxerrprof = Sa_sigT
        elif flag == 'q':
            eprofile = eprofile * sf
            mxerrprof = Sa_sigq
        elif flag == 'co2':
            eprofile = eprofile + sf
            mxerrprof = Sa_sigc
        else:
            print('Error in inflate_in_tropoe_uncertainty: the flag selecting temp or WVMR was not set properly -- aborting')
            sys.exit()

                # If this test is on, then the maximum uncertainty allowed is the prior uncertainty
        if vip['add_tropoe_use_prior_as_max'] == 1:
            for i in range(len(in_tropoe['height'])):
                if eprofile[i] > mxerrprof[i]:
                    eprofile[i] = mxerrprof[i]

        return eprofile
    else:
        return np.zeros(len(in_tropoe['height'])).astype('float')+999.

###############################################################################
# This function computes an estimate of the cloud base height using the minimum
# local emissivity variance (MLEV) approach
###############################################################################

def do_tcld_cbh(obswnum, obsrad, Xn, z, mlev_cbh, vip, verbose=1):
    cbh   = -999.
    if len(obswnum) != len(obsrad):
        print('Error in do_tcld_cbh: the lengths of the two wavenumber arrays are different')
        return cbh
    if vip['irs_tcld_wnum_range'][0] < 0:
        return cbh
    ix0 = np.where((vip['irs_tcld_wnum_range'][0] <  obswnum) &  (obswnum < vip['irs_tcld_wnum_range'][1]))[0]
    if len(ix0) < 2:
        print('      Error in do_tcld_cbh: there are not enough spectral elements in the band')
        return cbh

        # Calculate the blackbody temperature in this window channel [degC]
    winTb = Calcs_Conversions.invplanck(np.mean(obswnum[ix0]),np.mean(obsrad[ix0])) - 273.15
        # Extract out the temperature profile from the retrieval
    retT  = Xn[0:len(z)]
        # Find the heights where the retrieved temperature is smaller than the window Tb
    foo   = np.where(retT <= winTb)[0]
        # If the surface temperature is less than the window Tb, then there is likely
        # an inversion, so let's take the value that is closer to the MLEV-derived cbh
    if len(foo) == 0:
        cbh  = -888.
    elif z[foo[0]] < 0.001:
        delz = np.abs(z[foo] - mlev_cbh)
        bar  = np.where(delz == np.min(delz))[0]
        cbh  = z[foo[bar[0]]]
        # DDT print(f"In do_tcld_cbh, dual value so chose {cbh:.3f} using MLEV value of {mlev_cbh:.3f}")
    else:
        cbh  = z[foo[0]]
        # DDT print(f"In do_tcld_cbh, single value of {cbh:.3f}")

    return cbh

###############################################################################
# This function estimates the cloud base height using the minimum local emissivity variance method

def do_mlev_cbh(obswnum, obsrad, calcwnum, calcradclear, calcBrad, maxht, z, itern, vip, verbose=1):
    cbh   = -999.
    memis = -999.
    if vip['irs_mlev_cbh_wnum_range'][0] < 0:
        return cbh, memis
    if len(obswnum) != len(calcwnum):
        print('Error in do_mlev_cbh: the lengths of the two wavenumber arrays are different')
        return cbh, memis
    ix0 = np.where((vip['irs_mlev_cbh_wnum_range'][0] <  obswnum) &  (obswnum < vip['irs_mlev_cbh_wnum_range'][1]))[0]
    ix1 = np.where((vip['irs_mlev_cbh_wnum_range'][0] < calcwnum) & (calcwnum < vip['irs_mlev_cbh_wnum_range'][1]))[0]
    if len(ix0) != len(ix1):
        print('      Error in do_mlev_cbh: the obs and calc wnum arrays do not have the same number of elements in the band')
        return cbh, memis
    if len(ix0) < 10:
        print('      Error in do_mlev_cbh: there are not enough spectral elements in the band')
        return cbh, memis

    foo  = np.where(z <= maxht)[0]
    mlev = np.zeros(len(foo)-1)
    mean = np.zeros(len(foo)-1)
    for j in range(len(foo)-1):
        emis = (obsrad[ix0] - calcradclear[ix1]) / (calcBrad[j,ix1] - calcradclear[ix1])
        mlev[j] = np.nanstd(emis)
        mean[j] = np.nanmean(emis)
    foo   = np.where(mlev == np.min(mlev))[0]
    cbh   = z[foo[0]]
    memis = mean[foo[0]]
    return cbh, memis

###############################################################################
# This function regularizes the input covariance matrix using Tikhonov Regularization (ridge) approach
#     NewS = OrigS + alpha*Identity, for alpha small but greater than zero
# where OrigS is a square covariance matrix, and alpha is the regularization parameter

def condition_matrix(origS, alpha):

    error = ''
    dims = np.shape(origS)
    if len(dims) != 2:
        error = 'Error in function condition_matrix: input structure is not a 2-dimensional structure'
    if dims[0] != dims[1]:
        error = 'Error in function condition_matrix: input structure is not a square matrix'
    if alpha < 0:
        error = 'Error in function condition_matrix: alpha must be >= 0'

    if error != '':
        return origS, error
    iden = np.identity(dims[0])
    newS = origS + alpha*iden
    return newS, 'success'
