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

import os, re
import sys
import numpy as np
import scipy.io
from netCDF4 import Dataset, num2date
import calendar
from datetime import datetime, timedelta
import time

import Other_functions
import Output_Functions
import Calcs_Conversions
import VIP_Databases_functions


################################################################################
# This file contains the following functions:
# findfile()
# read_all_data()
# read_mwr()
# read_mwrscan()
# read_irs_eng()
# read_irs_ch()
# read_irs_sum()
# read_vceil()
# grid_irs()
# grid_mwr()
# grid_mwrscan()
# read_external_profile_data()
# read_external_timeseries()
# get_tropoe_version()
# recenter_prior()
# tape6_min_max_wnum()
################################################################################

################################################################################
# This function finds all of the files that match the inputted pattern
# Inputs:
#   path:       the directory (relative or absolute); must not have any wildcards
#   pattern:    the file pattern to find.  Can have wildcards or python options (t1|t2|t3)
################################################################################

def findfile(path,pattern,verbose=2):
    if(verbose >= 2):
        print('      Searching for files matching '+path+'/'+pattern)
    #print('Input path is: '+path)
    #print('Input pattern is: '+pattern)

        # We want to preserve periods (dots) in the pattern,
        # as many of our file patterns have periods in the name
    pattern = pattern.replace('.','\.')
        # Regex requires that we replace any asterisks with .*
    pattern = pattern.replace('*','.*')
        # Regex uses a period as a single character wildcard
    pattern = pattern.replace('?','.')

        # Trap the first and last character of the pattern
    first = pattern[0]
    lastl = pattern[-1]

        # If the first letter is an * or ?, then we prepend a dot
    if(first == '*'):
      pattern = '.'+pattern
    elif(first == '?'):
      pattern = '.'+pattern
    else:
      pattern = '^'+pattern

        # If the last letter is not a *, then we append a $
    if(lastl != '*'):
      pattern = pattern+'$'

    #print('Modified pattern is: '+pattern)
    # Good idea to write out the pattern to StdErr for future debugging

         # Compile the regex expression, and return the files that are found
    prog  = re.compile(pattern)

    # Check to see if the file directory exists
    if os.path.exists(path):
        files = [f for f in os.listdir(path) if re.search(prog, f)]

        # Now prepend the path to each files
        for i in range(len(files)):
            files[i] = path+'/'+files[i]
        files.sort()
        return files, 0
    else:
        print('The directory: ' + path)
        print('    does not exist!')
        return [], 1


################################################################################
# This function reads in the input file for an O-B calculation
# It is able to read netCDF files of two formats:
#    1) ARM radiosondes, with fields pres [mb], tdry [C], rh [%], and alt [m MSL]
#    2) Custom, with fields pressure [mb], temperature [C], rh [%], and height [km AGL]
# If the function finds the field "tdry", it assumes this is an ARM radiosonde
################################################################################
def read_omb_file(path, filename, hour, maxht, verbose=1):
    ombfile = path+'/'+filename

    if os.path.exists(ombfile) == False:
        errmess = (f"Unable to find the O-B input file {path:s}/{filename:s}")
        return ({'success':0, 'errstring':errmess})
    else:
        print('    Reading this file for the O-B calculation: '+ombfile)
        fid = Dataset(ombfile)
        if len(np.where(np.array(list(fid.variables.keys())) == 'tdry')[0]) > 0:
            print('        This is an ARM-like radiosonde file')  # Has a field named "tdry"
            z  = fid.variables['alt'][:]
            p  = fid.variables['pres'][:]
            rh = fid.variables['rh'][:]
            t  = fid.variables['tdry'][:]
            z  = (z - z[0])/1000.           # Convert m MSL to km AGL
        elif len(np.where(np.array(list(fid.variables.keys())) == 'gamma')[0]) > 0:
            foo = np.where(fid.variables['hour'][:] >= hour)[0]
            if len(foo) == 0:
                errmess = (f"Unable to find a matching time between input spectral and O-B input file {path:s}/{filename:s}")
                return ({'success':0, 'errstring':errmess})
            print(f"        This is a TROPoe file; extracting sample {foo[0]:d} at time {hour:.3f} UTC") # Has a field named "gamma"
            z  = fid.variables['height'][:]
            p  = fid.variables['pressure'][:]
            rh = fid.variables['rh'][:]
            t  = fid.variables['temperature'][:]
                # Now extract out the sample we desire
            p  =  p[foo[0],:]
            t  =  t[foo[0],:]
            rh = rh[foo[0],:]
        else:
            print('        This is the fixed-format OMB input file')   # The default
            z  = fid.variables['height'][:]
            p  = fid.variables['pressure'][:]
            rh = fid.variables['rh'][:]
            t  = fid.variables['temperature'][:]
        fid.close()
        q  = Calcs_Conversions.rh2w(t, rh/100., p)
        err = 0
        if len(z) < 3:
            err = 1
            errmess = 'Too few height elements -- needs to be at least 3 heights'
        elif np.max(z) < maxht:
            err = 1
            errmess = (f"Maximum height in the OMB input file ({np.max(z):.3f}) must be larger than the omb_required_minht in the VIP file ({maxht:.3f})")
        elif ((len(z) != len(t)) | (len(z) != len(p)) | (len(z) != len(rh))):
            err = 1
            errmess = 'All profiles in OMB input file must be of the same length'
        if err == 1:
            return ({'success':0, 'errstring':errmess})
    return ({'success':1, 'errstring':'None', 'z':z, 't':t, 'p':p, 'q':q, 'rh':rh, 'filename':path+'/'+filename})


################################################################################
# This function recenters the prior information
################################################################################

def recenter_prior(z0, p0, Xa, Sa, input_value, vip, sfc_or_pwv=0, changeTmethod=0, verbose=1):
    """
    This code recenters the mean of the prior dataset.
    The water vapor profile is recentered first, using a height-independent scale factor determined
    from either the surfaceWVMR value or the PWV (selected using the sfc_or_pwv flag).
    The temperature profile is then recentered, using either the "conserve-RH" or
    "conserve-covariance" methods.
    Note that the uncertainty of the water vapor is also recentered, but not the temperature.
    :z: The vertical grid of the prior
    :p: The mean pressure profile of the prior
    :Xa: The mean profiles of [temperature,waterVapor] (also called [T,q])
    :Sa: The covariance matrix of [[TT,Tq],[qT,qq]]
    :vip: The VIP key/values dictionary
    :param sfc_or_pwv: This keyword indicates the what the input_value represents:
                            0-> the default value, which forces the user to actually think!
                            1-> implies that the input_value is the surface WVMR [g/kg]
                            2-> implies that the input_value is the column PWV [cm]
    :param changeTmethod: This keyword indicates which method is used to recenter the temperature
                            0-> the default value, which forces the user to actually think!
                            1-> indicates that the conserve-RH method should be used
                            2-> indicates that the conserve-covariance method should be used
    :param verbose: This keyword indicates how noisy the routine should be
    :return: successFlag, newXa, newSa, comments
            SuccessFlag is 1 if the prior was successfully scaled, 0 if the function failed.
            newXa is the new mean prior
            newSa is the new prior covariance matrix
            comments is some strings
    """

    comments = {'Comment_on_recentering1': 'There was an error in the call of recenter_prior()', 
                'Comment_on_recentering2': 'So the code aborted'}

    if ((sfc_or_pwv < 1) | (sfc_or_pwv > 2)):
        print('Error: the sfc_or_pwv keyword has an undefined value (must be 1 or 2) -- see usage')
        return 0, Xa, Sa, comments
    if ((changeTmethod < 1) | (changeTmethod > 2)):
        print('Error: the changeTmethod keyword has an undefined value (must be 1 or 2) -- see usage')
        return 0, Xa, Sa, comments

    # Extract out the mean temperature and humidity profiles
    k    = len(z0)
    t0   = Xa[0:k]
    q0   = Xa[k:2*k]

    # Compute the correlation matrix from the prior's covariance
    sig   = np.sqrt(np.diag(Sa))
    sigT0 = sig[0:k]
    sigQ0 = sig[k:2*k]
    corM  = np.copy(Sa)
    for i in range(len(sig)):
        for j in range(len(sig)):
            corM[i,j] = Sa[i,j] / (sig[i] * sig[j])

    # Calculate the RH and PWV from this prior
    u0   = Calcs_Conversions.w2rh(q0, p0, t0, 0)
    pwv0 = Calcs_Conversions.w2pwv(q0, p0)

    # Get the min and max scale factor for the prior
    pmin = vip['recenter_covar_min_sfactor']
    if((pmin <= 0.001) | (pmin > 1)):
        print('Error: VIP.recenter_covariance_sfactor_min keyword must have a value in [0.001, 1] ')
        return 0, Xa, Sa, comments
    pmax = vip['recenter_covar_max_sfactor']
    if((pmax < 1) | (pmax > 5)):
        print('Error: VIP.recenter_covariance_sfactor_max keyword must have a value in [1, 5] ')
        return 0, Xa, Sa, comments

    # Scale the WV profile
    if sfc_or_pwv == 1:
        if(verbose >= 2):
            print('    Recenter_prior is using the scale-by-surface method')
        input_comment = f"surface WVMR value of {input_value:5.2f} g/kg"
        sf = input_value / q0[0]
        sfact1_comment = f'The WVMR mean profile was scaled by a factor of {sf:5.2f}'
          # Scale the new mean prior with the proper scale factor
        q1 = q0 * sf
          # Now adjust the scale factor somewhat, so that we don't overly scale the prior covariance
        sf = np.sqrt(sf)
        if(sf < pmin):
            sf = pmin
        elif(sf > pmax):
            sf = pmax
        sigQ1 = sigQ0 * sf
        sfact2_comment = f'The WVMR covariance was scaled by a factor of {sf:5.2f}'

    elif sfc_or_pwv == 2:
        if(verbose >= 2):
            print('    Recenter_prior is using the scale-by-PWV method')
        input_comment = f"column PWV value of {input_value:5.2f} cm"
        sf = input_value / pwv0
        sfact1_comment = f'The WVMR mean profile was scaled by a factor of {sf:5.2f}'
          # Scale the new mean prior with the proper scale factor
        q1 = q0 * sf
          # Now adjust the scale factor somewhat, so that we don't overly scale the prior covariance
        sf = np.sqrt(sf)
        if(sf < pmin):
            sf = pmin
        elif(sf > pmax):
            sf = pmax
        sigQ1 = sigQ0 * sf
        sfact2_comment = f'The WVMR covariance was scaled by a factor of {sf:5.2f}'

    else:
        print("Error with sfc_or_pwv: This should not happen within recenter_prior")
        return 0, Xa, Sa, comments

    if(verbose >= 2):
        print(f'    {sfact1_comment}')
        print(f'    {sfact2_comment}')

    # Adjust the temperature depending on the method selected
    if changeTmethod == 1:
            # Now iterate to find the best temperature, preserving the RH profile in the original prior
        tmethod_comment = 'converve-RH method'
        t1 = np.full_like(t0, -999.)     # Allocate an empty array
        off = np.arange(4001)/50. - 40.  # An array of temperature offsets

        for i in range(len(z0)):
            tmp = Calcs_Conversions.w2rh(q1[i], p0[i], t0[i] + off)
            foo = np.argmin(np.abs(tmp - u0[i]))
            t1[i] = t0[i] + off[foo]
    elif changeTmethod == 2:
        tmethod_comment = 'converve-covariance method'
        covTQ = np.zeros((len(z0),len(z0)))
        covQQ = np.zeros((len(z0),len(z0)))
        for i in range(len(z0)):
            for j in range(len(z0)):
                covTQ[i,j] = Sa[i,len(z0)+j]
                covQQ[i,j] = Sa[len(z0)+i,len(z0)+j]
        sf2qqInv = scipy.linalg.pinv(sf*sf*covQQ)
        slope = (sf*covTQ).dot(sf2qqInv)
        t1 = t0 + slope.dot(q1-q0)
    else:
        print("Error with changeTmethod: This should not happen within recenter_prior")
        return 0, Xa, Sa, comments

    # Now create the new mean prior and its covariance matrix
    newXa  = np.append(t1, q1)
    newSig = np.append(sigT0, sigQ1)
    newSa  = np.copy(Sa)
    for i in range(len(newSig)):
        for j in range(len(newSig)):
            newSa[i,j] = corM[i,j] * (newSig[i] * newSig[j])

    comments = {'Comment_on_recentering1': 'The WVMR profile prior recentered using a '+input_comment,
                'Comment_on_recentering2': sfact1_comment,
                'Comment_on_recentering3': sfact2_comment,
                'Comment_on_recentering4': 'The temperature prior was recentered using the '+tmethod_comment
    }

    # Echo some output, indicating how the prior was rescaled
    if(verbose >= 1):
        print('    The prior dataset was recentered')
        for i in range(len(list(comments.keys()))):
            print('      '+list(comments.keys())[i]+': '+comments[list(comments.keys())[i]])

    return 1, newXa, newSa, comments

################################################################################
# This function reads the IRS channel data file
################################################################################

def read_irs_ch(path,date,irs_type,fv,fa,irs_spec_cal_factor,
                engsecs,engtemp,bbcavfactor,get_irs_missingDataFlag,
                zenith_scene_mirror_angle,
                old_ffov_halfangle, new_ffov_halfangle, verbose):

    if verbose >= 1:
        print('  Reading irs_ch data in ' + path)
    err = {'success':0}
    if irs_type <= 0:
        print(('Error in read_irs_ch: This piece of code should not be exercised, as ' +
               'this option should be screened in read_irs_eng() earlier'))
        return err
    elif ((irs_type == 1) | (irs_type == 4)):
        filename = '*aeri*ch*' + str(date) + '*(cdf|nc)'
    elif irs_type == 2:
        filename = '*' + str(date % 2000000) + 'C1_rnc.(cdf|nc)'
    elif irs_type == 3:
        filename = '*' + str(date % 2000000) + 'C1.RNC.(cdf|nc)'
    elif irs_type == 5:
        filename = '*assist*(Ch|ch)*' + str(date) + '*(cdf|nc)'
    elif ((irs_type == 6) or (irs_type == 7)):
        filename = '*refir*(Ch|ch)*' + str(date) + '*(cdf|nc)'
    else:
        print('Error in read_irs_ch: unable to decode irs_type')
        return err

    if verbose >= 3:
        print('    Looking for IRS channel data as ' + filename)

    files,status = findfile(path,filename)
    if status == 1:
        return err

    if len(files) == 0:
        print('Error: Unable to find any IRS channel data -- aborting')
        return err

    for jj in range(len(files)):
        fid = Dataset(files[jj])
        fid.set_auto_mask(False)
        bt = fid['base_time'][:].astype('float')
            # If this is an ASSIST, then convert the base_time from milliseconds to seconds
        if(irs_type == 5):
            bt = bt / 1000.
            # Check for "time_offset"; if not there, assume it is "time"
        if len(np.where(np.array(list(fid.variables.keys())) == 'time_offset')[0]) > 0:
            to = fid['time_offset'][:].astype('float')
        else:
            to = fid['time'][:].astype('float')

        vid = np.where(np.array(list(fid.variables.keys())) == 'wnum')[0]
        if len(vid) > 0:
            wnum = fid.variables['wnum'][:]
        else:
            vid = np.where(np.array(list(fid.variables.keys())) == 'wnum1')[0]
            if len(vid) > 0:
                wnum = fid.variables['wnum1'][:]
            else:
                print('Error in read_irs_ch: unable to find either "wnum" or "wnum1 -- aborting')
                return err

        xmrad = fid.variables['mean_rad'][:]
        if len(np.where(np.array(list(fid.variables.keys())) == 'hatchOpen')[0]) > 0:
            xhatchOpen = fid.variables['hatchOpen'][:]
        elif  len(np.where(np.array(list(fid.variables.keys())) == 'hatchIndicator')[0]) > 0:
            #The AERI-01 has a hatchIndicator flag instead of hatchOpen,
            #but I need to change some of the values so that they are consistent

            xhatchOpen = fid.variables['hatchIndicator'][:]
            foo = np.where((xhatchOpen >= 1.5) & (xhatchOpen < 2.5))[0]
            if len(foo) > 0:
                xhatchOpen[foo] = 0
            foo = np.where((xhatchOpen > 2.5) | (xhatchOpen < -0.5))[0]
            if len(foo) > 0:
                xhatchOpen[foo] = 3
        elif  len(np.where(np.array(list(fid.variables.keys())) == 'hatch_open_indicator')[0]) > 0:
                    # The ASSIST has two flags that have to be merged together to mimic the AERI's hatchOpen flag
            xopen  = fid.variables['hatch_open_indicator'][:]
            xclose = fid.variables['hatch_closed_indicator'][:]
            xhatchOpen = np.full_like(xopen,-3)
            foo = np.where(xopen == 1 and xclose == 0)[0]
            if(len(foo) > 0):
                xhatchOpen[foo] = 1
            foo = np.where(xopen == 0 and xclose == 1)[0]
            if(len(foo) > 0):
                xhatchOpen[foo] = 0
        else:
            print(('Warning in read_irs_ch: Unable to find IRS hatchOpen or hatchIndicator field in ' +
                  'data file -- assuming hatch is always open'))
            xhatchOpen = np.ones(len(to))

        if len(np.where(np.array(list(fid.variables.keys())) == 'BBsupportStructureTemp')[0]) > 0:
            xbbsupport = fid.variables['BBsupportStructureTemp'][:]
            tmp = np.nanmean(xbbsupport)
            if(tmp < 150):
                if(verbose >= 2):
                    print('      Converting BBsupportStructureTemp from degC to degK')
                xbbsupport += 273.15
        elif len(np.where(np.array(list(fid.variables.keys())) == 'calibrationAmbientTemp')[0]) > 0:
            xbbsupport = fid.variables['calibrationAmbientTemp'][:]
            tmp = np.nanmean(xbbsupport)
            if(tmp < 150):
                if(verbose >= 2):
                    print('      Converting BBsupportStructureTemp from degC to degK')
                xbbsupport += 273.15
        elif len(np.where(np.array(list(fid.variables.keys())) == 'abb_thermistor_top')[0]) > 0:
            xbbsupport = fid.variables['abb_thermistor_top'][:]
            tmp = np.nanmean(xbbsupport)
            if(tmp < 150):
                if(verbose >= 2):
                    print('      Converting BBsupportStructureTemp from degC to degK')
                xbbsupport += 273.15
        else:
            xbbsupport = np.full_like(to,-999)

        if len(np.where(np.array(list(fid.variables.keys())) == 'calibrationAmbientTemp')[0]) > 0:
            xcalibambt = fid.variables['calibrationAmbientTemp'][:]
            tmp = np.nanmean(xcalibambt)
            if(tmp < 150):
                if(verbose >= 2):
                    print('      Converting calibrationAmbientTemp from degC to degK')
                xcalibambt += 273.15
        elif len(np.where(np.array(list(fid.variables.keys())) == 'abb_thermistor_top')[0]) > 0:
            xcalibambt = fid.variables['abb_thermistor_top'][:]
            tmp = np.nanmean(xcalibambt)
            if(tmp < 150):
                if(verbose >= 2):
                    print('      Converting calibrationAmbientTemp from degC to degK')
                xcalibambt += 273.15
        else:
            xcalibambt = np.full_like(to,-999)

        if len(np.where(np.array(list(fid.variables.keys())) == 'calibrationCBBtemp')[0]) > 0:
            xcalibcbbt = fid.variables['calibrationCBBtemp'][:]
            tmp = np.nanmean(xcalibcbbt)
            if(tmp < 150):
                if(verbose >= 2):
                    print('      Converting calibrationCBBtemp from degC to degK')
                xcalibcbbt += 273.15
        elif len(np.where(np.array(list(fid.variables.keys())) == 'abb_mean_temp')[0]) > 0:
            xcalibcbbt = fid.variables['abb_mean_temp'][:]
            tmp = np.nanmean(xcalibcbbt)
            if(tmp < 150):
                if(verbose >= 2):
                    print('      Converting calibrationCBBtemp from degC to degK')
                xcalibcbbt += 273.15
        else:
            xcalibcbbt = np.full_like(to,-999)

        if len(np.where(np.array(list(fid.variables.keys())) == 'calibrationHBBtemp')[0]) > 0:
            xcalibhbbt = fid.variables['calibrationHBBtemp'][:]
            tmp = np.nanmean(xcalibhbbt)
            if(tmp < 150):
                if(verbose >= 2):
                    print('      Converting calibrationHBBtemp from degC to degK')
                xcalibhbbt += 273.15
        elif len(np.where(np.array(list(fid.variables.keys())) == 'hbb_mean_temp')[0]) > 0:
            xcalibhbbt = fid.variables['hbb_mean_temp'][:]
            tmp = np.nanmean(xcalibhbbt)
            if(tmp < 150):
                if(verbose >= 2):
                    print('      Converting calibrationHBBtemp from degC to degK')
                xcalibhbbt += 273.15
        else:
            xcalibhbbt = np.full_like(to,-999)

        if len(np.where(np.array(list(fid.variables.keys())) == 'atmosphericPressure')[0]) > 0:
            xambPres = fid.variables['atmosphericPressure'][:]
            tmp = np.nanmean(xambPres)
            if((tmp > 0) & (tmp < 150)):
                if(verbose >= 2):
                    print('      Converting atmosphericPressure from kPa to hPa')
                xambPres *= 10.
        else:
            xambPres = np.full_like(to,-999)

        if (len(np.where(np.array(list(fid.variables.keys())) == 'sceneMirrorAngle')[0])> 0):
            xscenemirrorangle = fid.variables['sceneMirrorAngle'][:]
        else:
            print('WARNING: File does not contain variable "sceneMirrorAngle".')
            print('         Setting sceneMirrorAngle to ' + str(zenith_scene_mirror_angle))
            xscenemirrorangle = np.full_like(to, zenith_scene_mirror_angle)

        #Read in the field "missingDataFlag". If it does not exist, then abort
        if get_irs_missingDataFlag == 1:
            if len(np.where(np.array(list(fid.variables.keys())) == 'missingDataFlag')[0]) > 0:
                xmissingDataFlag = fid.variables['missingDataFlag'][:]
                xmissingDataFlag = xmissingDataFlag.astype('int')
            elif len(np.where(np.array(list(fid.variables.keys())) == 'missingDataFlag')[0]) == 0:
                xmissingDataFlag = np.zeros_like(to)
                xmissingDataFlag = xmissingDataFlag.astype('int')
            else:
                print('Error in read_irs_ch: unable to find the field missingDataFlag')
                return err
        else:
            xmissingDataFlag = np.zeros(len(to))
        fid.close()

        #Append the data into single arrays
        if jj == 0:
            secs = bt+to
            mrad = np.copy(xmrad)
            hatchOpen = np.copy(xhatchOpen)
            bbsupport = np.copy(xbbsupport)
            calibambt = np.copy(xcalibambt)
            calibcbbt = np.copy(xcalibcbbt)
            calibhbbt = np.copy(xcalibhbbt)
            ambPres = np.copy(xambPres)
            missingDataFlag = np.copy(xmissingDataFlag)
            sceneMirrorAngle = np.copy(xscenemirrorangle)
        else:
            secs = np.append(secs,bt+to)
            mrad = np.append(mrad,xmrad, axis = 0)
            hatchOpen = np.append(hatchOpen,xhatchOpen)
            bbsupport = np.append(bbsupport,xbbsupport)
            calibambt = np.append(calibambt,xcalibambt)
            calibcbbt = np.append(calibcbbt,xcalibcbbt)
            calibhbbt = np.append(calibhbbt,xcalibhbbt)
            ambPres = np.append(ambPres,xambPres)
            missingDataFlag = np.append(missingDataFlag,xmissingDataFlag)
            sceneMirrorAngle = np.append(sceneMirrorAngle,xscenemirrorangle)

    chsecs = np.copy(secs)
    mrad = mrad.T

    # Keep only the zenith pointing data, using the keyword specified in the VIP to control the behavior
    if((zenith_scene_mirror_angle < 3) | (zenith_scene_mirror_angle > 357)):
        foo = np.where((sceneMirrorAngle < 3) | (sceneMirrorAngle > 357))[0]
    else:
        foo = np.where((zenith_scene_mirror_angle-3 < sceneMirrorAngle) &
                       (sceneMirrorAngle < zenith_scene_mirror_angle+3))[0]

    if len(foo) == 0:
        # Check to see if all of the data is missing and if using AERI data then
        # set to default value. If not abort. This can happen in ARM data
        fah = np.where(sceneMirrorAngle > -100)[0]
        if((len(fah) == 0) & (irs_type < 5)):
            sceneMirrorAngle[:] = zenith_scene_mirror_angle
            foo = np.arange(len(sceneMirrorAngle))
        else:
            print('Error in read_irs_ch: Unable to find any zenith pointing IRS data')
            return err

    chsecs           = np.copy(chsecs[foo])
    mrad             = np.copy(mrad[:,foo])
    hatchOpen        = np.copy(hatchOpen[foo])
    bbsupport        = np.copy(bbsupport[foo])
    calibambt        = np.copy(calibambt[foo])
    calibcbbt        = np.copy(calibcbbt[foo])
    calibhbbt        = np.copy(calibhbbt[foo])
    ambPres          = np.copy(ambPres[foo])
    missingDataFlag  = np.copy(missingDataFlag[foo])
    sceneMirrorAngle = np.copy(sceneMirrorAngle[foo])

    # Apply the spectral recalibration, if desired
    if(np.abs(irs_spec_cal_factor - 1.0) > 0.0000001):
        if(verbose >= 1): print('      Adjusting the IRSs spectral calibration')
        tmp = mrad
        for jj in range(0,len(mrad[0,:])):
            tmp[:,jj] = Other_functions.fix_aeri_vlaser_mod(wnum,mrad[:,jj],irs_spec_cal_factor)
        mrad = tmp

    # Apply the new finite field-of-view correction, if desired
    if((10 <= new_ffov_halfangle) & (new_ffov_halfangle < 50)):
        print('DDT -- the new_ffov_halfangle function needs additional testing -- aborting')
        sys.exit()
        if(verbose >= 1):
            print(f'    Adjusting the IRSs FFOV correction half angle from {old_ffov_halfangle} to {new_ffov_halfangle} mrad')
        tmp = mrad
        for jj in range(0,len(chsecs)):
            tmp[:,jj] = Other_functions.change_irs_ffovc(wnum,mrad[:,jj],old_ffov_halfangle,new_ffov_halfangle)
        mrad = tmp
    else:
        if((verbose >= 1) & (new_ffov_halfangle > 0.0)):
            print('      The desired new FFOV correction half angle is outside realistic bounds -- not applied')

    #I need to match the times of the IRS channel data with that from
    #the engineering file (which is the summary file).

    flag, prob = Other_functions.matchtimes(engsecs, chsecs, 0.5)
    if prob == 0:
        print('Error in read_irs_ch: matchtimes() failed very badly')
        return err

    foo = np.where(flag == 1)[0]
    if len(foo) == 0:
        print('Error in read_irs_ch: None of the ch data match the eng data times')
        return err

    chsecs = np.copy(chsecs[foo])
    mrad = np.copy(mrad[:,foo])
    hatchOpen = np.copy(hatchOpen[foo])
    missingDataFlag = np.copy(missingDataFlag[foo])
    bbsupport = np.copy(bbsupport[foo])
    calibambt = np.copy(calibambt[foo])
    calibcbbt = np.copy(calibcbbt[foo])
    calibhbbt = np.copy(calibhbbt[foo])
    ambPres = np.copy(ambPres[foo])
    sceneMirrorAngle = np.copy(sceneMirrorAngle[foo])

    #And now the reverse
    flag, prob = Other_functions.matchtimes(chsecs,engsecs,0.5)
    if prob == 0:
        print('Error in read_irs_ch: matchtimes() failed very badly')
        return err

    foo = np.where(flag == 1)[0]
    if len(foo) == 0:
        print('Error in read_irs_ch: None of the eng data match the ch data times')
        return err
    engsecs = np.copy(engsecs[foo])
    engtemp = np.copy(engtemp[foo])
    bbcavfactor = np.copy(bbcavfactor[foo])

    #Convert the time to something useful
    yy = np.array([datetime.utcfromtimestamp(x).year for x in chsecs])
    mm = np.array([datetime.utcfromtimestamp(x).month for x in chsecs])
    dd = np.array([datetime.utcfromtimestamp(x).day for x in chsecs])
    ymd = yy*10000 + mm*100 + dd
    hour = np.array([((datetime.utcfromtimestamp(x)-datetime(yy[0],mm[0],dd[0])).total_seconds())/3600. for x in chsecs])

    #I can only apply one of the Fv or Fa corrections, so abort if they are
    #both turned on

    if ((fv > 0) & (fa > 0)):
        print('Error in irs_read_ch: both the IRS obscuration (Fv) and ' +
	          'aft-optics (Fa) corrections are turned on in the VIP -- turn one or both off')
        return err

    #Recalibrate the IRS data to ensure that the BB emissivity is correct.
    #Note that the typical (and now obsolete) value being used was 12.79
    #but we should be using something in excess of 35 (more like 39)

    if verbose == 3:
        print('    Correcting for the blackbody emissivity cavity factor')
    nrad = mrad * 0
    emisn = Other_functions.get_aeri_bb_emis(wnum, cavity_factor=39.0, verbose=verbose)
    for i in range(len(hour)):
        emiso = Other_functions.get_aeri_bb_emis(wnum, cavity_factor=bbcavfactor[i], verbose=verbose+i)
        nrad[:,i] = Other_functions.aeri_recal(wnum, mrad[:,i], calibhbbt[i], calibhbbt[i], calibcbbt[i],
                    calibcbbt[i], calibambt[i], calibambt[i], emiso, emisn, emiso, emisn)

    mrad = np.copy(nrad)

    # If the Fa value is greater than zero, remove the contribution from the
    # aft optics, but first see if the field "interferometerSecondPortTemp"
    # existed in the data, because if it did not then we can not apply this
    # correction and need to make some noise about it.

    foo = np.where(engtemp > 0)
    if ((len(foo) == 0) & (fa > 0)):
        print('Error: The algorithm wants to apply an aft-optics correction to the IRS')
        print('          data (i.e., fa > 0), but the field "interferometerSecondPortTemp"')
        print('          was not found in the input data. Thus we must abort this')
        print('          processing. Please either rerun with fa == 0 or modify the')
        print('          code so that a different field is used as the aft optic temperature')
        return err

    if fa > 0:
        if verbose >= 2:
            print('    Correcting IRS data with Fa = ' + str(fa))
        nrad = np.copy(mrad)
        for i in range(len(hour)):
            Bref = Calcs_Conversions.planck(wnum,calibambt[i])
            Bcbb = Calcs_Conversions.planck(wnum,calibcbbt[i])
            Rc = emisn*Bcbb + (1.-emisn)*Bref
            aft_temp = np.interp(bt+to[i],engsecs,engtemp)

            #Apply a bit of QC here!
            if ((calibcbbt[i] - 20 > aft_temp) | (aft_temp > calibhbbt[i]+2)):
                print('Error: the temperature used for the IRS aft optics (Fa correction) is out of bounds')
                return err
            Baft = Calcs_Conversions.planck(wnum,aft_temp)
            nrad[:,i] = (mrad[:,i] - fa*Baft) / (1. - fa)
        mrad = np.copy(nrad)

    if fv > 0:
        if verbose >= 2:
            print('      Correcting IRS data with Fv = ' + str(fv))
        nrad = np.copy(mrad)
        for i in range(len(hour)):
            Brad = Calcs_Conversions.planck(wnum, bbsupport[i])
            nrad[:,i] = (mrad[:,i] - fv*Brad) / (1. - fv)
        mrad = np.copy(nrad)

    # Compute the near-air surface temperature (i.e., brightness temperature in a
    # spectral band sensitive to near-surface air temperature) [degC].  I need to 
    # test to see if I am using ch1 or ch2 radiance (look for the former first)
    foo = np.where((675 < wnum) & (wnum < 680))[0]
    if len(foo) == 0:
        foo = np.where((2295 < wnum) & (wnum < 2300))[0]
    nearSfcTb = np.zeros(len(chsecs)) - 999.
    if len(foo) > 0:
        mnwnum = np.mean(wnum[foo])
        for i in range(len(hour)):
            nearSfcTb[i] = Calcs_Conversions.invplanck(mnwnum,np.mean(mrad[foo,i])) - 273.15

    # Sometimes the data are not in time-ascending order.  So sort the data
    sidx = np.argsort(chsecs)

    # Return the data structure
    return ({'success':1, 'secs':chsecs[sidx], 'ymd':ymd[sidx], 'yy':yy[sidx], 'mm':mm[sidx], 'dd':dd[sidx],
            'hour':hour[sidx], 'wnum':wnum, 'rad':mrad[:,sidx], 'nearSfcTb':nearSfcTb[sidx], 'hatchopen':hatchOpen[sidx],
            'atmos_pres':ambPres[sidx],'missingDataFlag':missingDataFlag[sidx], 'fv':fv, 'fa':fa})


################################################################################
# This function controls the reading of the IRS, MWR, and ceilometer data. The function
# calls out many other functions to do this.
################################################################################

def read_all_data(date, retz, tres, dostop, verbose, avg_instant, ch1_path,
              pca_nf, fv, fa, sum_path, eng_path, irs_type, cal_irs_pres,
              irs_smooth_noise, get_irs_missingDataFlag, irs_min_675_tb,
              irs_max_675_tb, irs_spec_cal_factor, irs_noise_inflation,
              mwr_path, mwr_rootname, mwr_type, mwr_elev_field,
              mwr_n_tb_fields, mwr_tb_replicate, mwr_tb_field_names, mwr_tb_freqs,
              mwr_tb_noise, mwr_tb_bias, mwr_tb_field1_tbmax, vceil_path, vceil_type,
              vceil_window_in, vceil_window_out, vceil_default_cbh,
              hatchOpenSwitch, missingDataFlagSwitch, vip):

    fail = 0

    # Check the flag to make sure it has reasonable values
    if ((avg_instant != -1) & (avg_instant != 0) & (avg_instant != 1)):
        print('Error: The "avg_instant" flag can only have a value of -1:avg with no sqrt(N), 0:avg with sqrt(N), or 1:instantaneous')
        fail = 1
        return fail, -999, -999, -999

    # Check the irs_type flag. If it is < 0, then we will not read any IRS data
    # in, and instead use tres to create data that acts as the master dataset. I will also simulate a
    # few channels on the IRS because it makes the rest of the code easier (fewer changes) to make

    if ((irs_type <= -1) & (mwr_type > 0)):

        #In this example, the "replicate" keyword should be unity. Since we aren't
        # using IRS data in this case, it should not have any other value

        if mwr_tb_replicate != 1:
            print('Error: The mwr_tb_replicate should be unity in this MWR-only retrieval (no IRS data)')
            fail = 1
            return fail, -999, -999, -999

        # I will read in the MWR data here and again later. I need to read it here
        # so that I can simulate IRS data (all as MISSING) so that the rest of the code
        # code can work.  Note that irs_type (which is a negative number) also
        # defines the "step" used to read in the MWR data, which is pretty useful if
        # we are reading in HATPRO data which can have >3000 samples per day (i.e., the
        # irs_type is used to subsample the MWR data)

        print('  Attempting to use MWR as the master instrument')

        # Check to make sure the directory exists and has files in it
        if os.path.isdir(mwr_path):
            if len(os.listdir(mwr_path)) > 0:
                mwr_data = read_mwr(mwr_path, mwr_rootname, date, mwr_type, abs(irs_type),
                           vip['mwr_freq_field'], mwr_elev_field, mwr_n_tb_fields,
                           mwr_tb_field_names, mwr_tb_freqs, mwr_tb_noise, mwr_tb_bias, mwr_tb_field1_tbmax,
                           verbose, single_date=True)
            else:
                print('  The directory for the master instrument: ' + mwr_path)
                print('      has no files in it!')
                return 1, -999, -999, -999
        else:
            print('  The directory for the master instrument: ' + mwr_path)
            print('      does not exist!')
            return 1, -999, -999, -999

        if (mwr_data['success'] != 1) or (mwr_data['type'] == 0):
            print('Problem reading MWR data -- unable to continue because the MWR is the master instrument')
            fail = 1
            return fail, -999, -999, -999
        else:
            #Quick check of the surface pressure field. If the values are negative,
            #then the code is not finding the pressure field and the retrieval won't
            #be able to run -- abort here.

            foo = np.where(mwr_data['psfc'] > 0)
            if len(foo) == 0:
                print('    Warning: no surface data found in the MWR files. Default station_pres values will be used')
                print('        unless an external surface pressure data source is provided.')

            wnum = np.arange(int((905-900)/0.5)+1)*0.5+900            #Simulated wavenumber array
            mrad = np.ones((len(wnum),len(mwr_data['secs'])))*-999.0   #Radiance is all missing
            noise = np.ones((len(wnum),len(mwr_data['secs'])))         #Set all noise values to 1
            yy = np.array([datetime.utcfromtimestamp(x).year for x in mwr_data['secs']])
            mm = np.array([datetime.utcfromtimestamp(x).month for x in mwr_data['secs']])
            dd = np.array([datetime.utcfromtimestamp(x).day for x in mwr_data['secs']])
            hour = np.array([((datetime.utcfromtimestamp(x)-datetime(yy[0],mm[0],dd[0])).total_seconds())/3600. for x in mwr_data['secs']])
            ymd = yy*10000 + mm*100 + dd

            irseng = ({'success':1, 'secs':mwr_data['secs'], 'ymd':ymd, 'hour':hour,
                       'bbcavityfactor': np.zeros(len(mwr_data['secs'])),
                       'interferometerSecondPortTemp': np.ones(len(mwr_data['secs']))*300.0})

            irsch1 = ({'success':1, 'secs':mwr_data['secs'], 'ymd':ymd, 'yy':yy, 'mm':mm, 'dd':dd, 'hour':hour,
                        'wnum':wnum, 'rad':mrad, 'hatchopen': np.ones(len(mwr_data['secs'])),
                        'atmos_pres':mwr_data['psfc'], 'nearSfcTb': np.zeros(len(mwr_data['secs']))-999.,
                        'missingDataFlag': np.zeros(len(mwr_data['secs'])),
                        'fv':0.0, 'fa': 0.0})

            irssum = ({'success':1, 'secs':mwr_data['secs'], 'ymd':ymd, 'hour':hour, 'wnum':wnum, 'noise':noise})

    elif (irs_type <= -1) & (mwr_type == 0):

        if tres <= 0:
            print("Error: tres must be greater than 0 if irs_type is < 0 and mwr_type are less than 0")
            fail = 1
            return fail, -999, -999, -999

        print('  No IRS or MWR data specified. Creating time grid based on date and tres from VIP...')

        # Create a fake time grid based on tres so we have use some fake IRS observations
        # to make the rest of the code work properly
        blah = (datetime.strptime(str(date), "%Y%m%d") - datetime(1970,1,1)).total_seconds()
        fake_secs = np.array([blah + i for i in np.arange(0, 60*60*24, tres * 60)])

        wnum = np.arange(int((905 - 900) / 0.5) + 1) * 0.5 + 900  # Simulated wavenumber array
        mrad = np.ones((len(wnum), len(fake_secs))) * -999.0  # Radiance is all missing
        noise = np.ones((len(wnum), len(fake_secs)))  # Set all noise values to 1
        yy = np.array([datetime.utcfromtimestamp(x).year for x in fake_secs])
        mm = np.array([datetime.utcfromtimestamp(x).month for x in fake_secs])
        dd = np.array([datetime.utcfromtimestamp(x).day for x in fake_secs])
        hour = np.array(
            [((datetime.utcfromtimestamp(x) - datetime(yy[0], mm[0], dd[0])).total_seconds()) / 3600. for x in fake_secs])
        ymd = yy * 10000 + mm * 100 + dd

        irseng = ({'success': 1, 'secs': fake_secs, 'ymd': ymd, 'hour': hour,
                   'bbcavityfactor': np.zeros(len(fake_secs)),
                   'interferometerSecondPortTemp': np.ones(len(fake_secs)) * 300.0})

        irsch1 = ({'success': 1, 'secs': fake_secs, 'ymd': ymd, 'yy': yy, 'mm': mm, 'dd': dd, 'hour': hour,
                   'wnum': wnum, 'rad': mrad, 'hatchopen': np.ones(len(fake_secs)),
                   'atmos_pres': np.ones(len(fake_secs)) * 1013, 'nearSfcTb': np.zeros(len(fake_secs))-999., 
                   'missingDataFlag': np.zeros(len(fake_secs)),
                   'fv': 0.0, 'fa': 0.0})

        irssum = ({'success': 1, 'secs': fake_secs, 'ymd': ymd, 'hour': hour, 'wnum': wnum, 'noise': noise})


    else:

        # Read in the IRS data. Befoe we do though we are going to check if directories
        # for the engineering, summmary, and ch1 files exist and have files

        if not os.path.isdir(eng_path):
            print('  The IRS engineering file directory: ' + eng_path)
            print('    does not exist!')
            return 1, -999, -999, -999
        else:
            if os.listdir(eng_path) == []:
                print('  The IRS engineering file directory: ' + eng_path)
                print('    has no files in it!')
                return 1, -999, -999, -999

        if not os.path.isdir(ch1_path):
            print('  The IRS channel file directory: ' + ch1_path)
            print('    does not exist!')
            return 1, -999, -999, -999
        else:
            if os.listdir(ch1_path) == []:
                print('  The IRS channel file directory: ' + ch1_path)
                print('    has no files in it!')
                return 1, -999, -999, -999

        if not os.path.isdir(sum_path):
            print('  The IRS summary file directory: ' + sum_path)
            print('    does not exist!')
            return 1, -999, -999, -999
        else:
            if os.listdir(sum_path) == []:
                print('  The IRS summary file directory: ' + sum_path)
                print('    has no files in it!')
                return 1, -999, -999, -999

        irseng = read_irs_eng(eng_path,date,irs_type,verbose)
        if irseng['success'] == 0:
            print('  Problem reading IRS eng data')
            if dostop:
                wait = input('Stopping inside routine for debugging. Press enter to continue')
            fail = 1
            return fail, -999, -999, -999

        irsch1 = read_irs_ch(ch1_path,date,irs_type,fv,fa,irs_spec_cal_factor,
                              irseng['secs'],
                              irseng['interferometerSecondPortTemp'],
                              irseng['bbcavityfactor'], get_irs_missingDataFlag,
                              vip['irs_zenith_scene_mirror_angle'],
                              vip['irs_old_ffov_halfangle'], vip['irs_new_ffov_halfangle'], verbose)

        irssum = read_irs_sum(sum_path,date,irs_type,irs_smooth_noise,verbose)

        if irsch1['success'] != 1:
            print('Problem reading IRS ch1 data')
            fail = 1
            return fail, -999, -999, -999
        if irssum['success'] != 1:
            print('Problem reading IRS sum data')
            fail = 1
            return fail, -999, -999, -999

        #Apply the additional IRS QC tests.
        foo = np.where((irsch1['wnum'] >= 675) & (irsch1['wnum'] < 680))[0]
        if len(foo) > 0:
            for i in range(len(irsch1['secs'])):
                tmp = np.nanmean(irsch1['rad'][foo,i])
                tb = Calcs_Conversions.invplanck(677.5,tmp)
                if ((tb < irs_min_675_tb) | (tb > irs_max_675_tb)):
                    irsch1['missingDataFlag'][i] = 10
            isclose = np.isclose(irsch1['missingDataFlag'][:],10)
            foo = np.where(isclose == True)[0]
            if(len(foo) >= 0.9*len(irsch1['secs'])):
                print('Error: more than 90% of the IRS data are outside the 675 cm-1 BT range - aborting')
                fail = 1
            if(len(foo) > 0.5*len(irsch1['secs'])):
                print('WARNING: more than 50% of the IRS data are outside the 675 cm-1 BT range')

        if ((fail == 1) & (dostop != 0)):
            wait = input('Stopping inside routine for debugging. Press enter to continue')
        elif fail == 1:
            return fail, -999, -999, -999

        #Calibrate the IRS pressure sensor using a linear function
        if len(cal_irs_pres) != 2:
            print('  Error: The calibration information for the IRS pressure sensor is ill-formed')
            fail = 1
        else:
            if ((cal_irs_pres[0] != 0) | (cal_irs_pres[1] != 1)):
                print(' Calibrating IRS pressure with intercept ' + str(cal_irs_pres[0]) + ' and  slope ' + str(cal_irs_pres[1]))
            irsch1['atmos_pres'] = cal_irs_pres[0] + irsch1['atmos_pres'] *cal_irs_pres[1]

        if ((fail == 1) & (dostop != 0)):
            wait = input('Stopping inside routine for debugging. Press enter to continue')
        elif fail == 1:
            return fail, -999, -999, -999

    #Specify the times of the retrieved output
    if tres <= 0:
        ret_secs = irsch1['secs']-0.001      # Essentially the same as IRS sample time
        ret_tavg = 10./60                    # 10-s resolution
    else:
        yy = np.copy(irsch1['yy'])
        mm = np.copy(irsch1['mm'])
        dd = np.copy(irsch1['dd'])
        hour = np.copy(irsch1['hour'])
        if np.nanmax(hour) <= 24:
            nmins = 1440.0
        elif np.nanmax(hour) <= 2*24:
            nmins = 2*1440.0
        elif np.nanmax(hour) <= 3*24:
            nmins = 3*1440.0
        else:
            print('Error -- the IRS data files span more than 3 days -- code needs an update')
            fail = 1
            return fail, -999, -999, -999
        d = datetime(yy[0],mm[0],dd[0],0,0,0)
        bt = calendar.timegm(d.timetuple())
        ret_secs = np.arange(int(nmins/tres + 0.5)) * tres * 60+bt+tres/2.
        ret_tavg = tres

    #Read in the MWR zenith data
    if mwr_type > 0:
        print('  Reading in MWR-zenith data')
    mwr_data = read_mwr(mwr_path, mwr_rootname, date, mwr_type, 1, vip['mwr_freq_field'], mwr_elev_field, mwr_n_tb_fields,
                        mwr_tb_field_names, mwr_tb_freqs, mwr_tb_noise, mwr_tb_bias, mwr_tb_field1_tbmax,
                        verbose)

    if mwr_data['success'] != 1:
        print('Problem reading in MWR-zenith data')
        fail = 1
        return fail, -999, -999, -999

    #Read in the MWR scan data
    if vip['mwrscan_type'] > 0:
        print('  Reading in MWR-scan data')
    mwrscan_data = read_mwrscan(vip['mwrscan_path'], vip['mwrscan_rootname'], date, vip['mwrscan_type'],
                   vip['mwrscan_freq_field'], vip['mwrscan_elev_field'], vip['mwrscan_n_tb_fields'], vip['mwrscan_tb_field_names'],
                   vip['mwrscan_tb_freqs'], vip['mwrscan_tb_noise'], vip['mwrscan_tb_bias'],
                   vip['mwrscan_tb_field1_tbmax'], vip['mwrscan_n_elevations'], vip['mwrscan_elevations'], verbose)

    if mwrscan_data['success'] == 0:
        print('Problem reading MWR-scan data')
        fail = 1
        return fail, -999, -999, -999

    #Read in the ceilometer data
    vceil = read_vceil(vceil_path, date, vceil_type, ret_secs, vip['cbh_mpd_abc_thres'], verbose)

    if vceil['success'] < 0:
        fail = 1
    elif vceil['success'] == 0:
        print('  No vceil data -- using default cbh for entire period')
        vceil = {'success':2, 'secs':irsch1['secs'], 'ymd':irsch1['ymd'], 'hour':irsch1['hour'], 'cbh':np.ones(len(irsch1['secs']))*-1}

    if ((fail == 1) & (dostop)):
        wait = input('Stopping for debugging. Press enter to continue')
    elif fail == 1:
        return fail, -999, -999, -999


    # Now put the IRS and MWR data on the same temporal grid.  Realize
    # that the RL sample time is for the start of the period, whereas the
    # IRS sample time is the middle of its period and the MWR's is the end.

    irs = grid_irs(irsch1, irssum, avg_instant, hatchOpenSwitch, missingDataFlagSwitch,
                    ret_secs, ret_tavg, irs_noise_inflation, vip, verbose)

    if irs['success'] == 0:
        fail = 1
        return fail, -999, -999, -999

    # Now apply a temporal screen to see if there are cloudy samples in the
    # IRS data by looking at the standard deviation of the 11 um data. If there
    # are cloudy samples, use the lidar to get an estimate of the cloud
    # base height for the subsequent retrieval

    cirs = Other_functions.find_cloud(irs, vceil, vceil_window_in, vceil_window_out, vceil_default_cbh)
    
    mwr = grid_mwr(mwr_data, avg_instant, ret_secs, ret_tavg, vip['mwr_time_delta'], verbose)

    if mwr['success'] == 0:
        fail = 1
        return fail, -999, -999, -999

    mwrscan = grid_mwrscan(mwrscan_data, ret_secs, vip['mwrscan_n_elevations'],
                        vip['mwrscan_elevations'], vip['mwrscan_time_delta'], verbose)

    if mwrscan['success'] == 0:
        fail = 1
        return fail, -999, -999, -999

    if dostop:
        wait = input('Stopping inside to debug this bad boy. Press enter to continue')

    return fail, cirs, mwr, mwrscan

################################################################################
# This function is the one that actually reads in the MWR data.
################################################################################

def read_mwr(path, rootname, date, mwr_type, step, mwr_freq_field, mwr_elev_field, mwr_n_tb_fields,
            mwr_tb_field_names, mwr_tb_freqs, mwr_tb_noise, mwr_tb_bias,
            mwr_tb_field1_tbmax, verbose, single_date=False):

    if verbose >= 2:
        print('  Reading MWR data in ' + path)
    err = {'success':0, 'type':-1, 'alt':-1}

    # Check to make sure "step" is an integer greater than 0
    if(type(step) != int):
        print("        Error: the step in read_mwr is not an integer -- aborting")
        return err
    elif(step < 0):
        print("        Error: the step in read_mwr is not a positive integer -- aborting")
        return err


    # If single date is True, only read in the single date. This is used for MWR only mode upon the first
    # read of the mwr data. This is mainly to get the times. If we read in data from before and after the
    # inputted date, it messes up the times.
    if single_date:
        udate = [str(date)]
    else:
        udate = [(datetime.strptime(str(date), '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d') ,
                  str(date) ,  (datetime.strptime(str(date), '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d') ]

    if mwr_type < 0:
        print(' ')
        print('-----------------------------------------------------------------')
        print('****** options for the VIP parameter "mwr_type" *******')
        print('   mwr_type=0 --->')
        print('             No MWR data specified')
        print('   mwr_type=1 --->')
        print('             MWR Tb data are in individual 1-d variables in the input file')
        print('   mwr_type=2 --->')
        print('             MWR tb data are in a single 2-d variable in the input file')
        print('   mwr_type=3 --->')
        print('             MWR tb data are in a single 2-d variable in the input file, using the "Univ Cologne" format')
        print('   mwr_type=4 --->')
        print('             MWR tb data are in a single 2-d variable in the input file, using the "E-PROFILE" format')
        print('-----------------------------------------------------------------')
        print(' ')
        err = {'success':-1, 'alt':-1}
        return err

    # Let's get the Tb frequencies, noise levels, and bias from the VIP file into arrays
    if mwr_n_tb_fields > 0:
        parts = mwr_tb_freqs.split(',')
        if len(parts) != mwr_n_tb_fields:
            print('Error: The number of entered MWR frequencies does not match desired MWR channels')
            return err
        else:
            freq = np.array(parts).astype(np.float)
        foo = np.where(((freq >= 15) & (freq < 240)))[0]
        if len(foo) != mwr_n_tb_fields:
            print('Error: The entered frequencies are not between 15 and 240 GHz -- probably an error')
            return err
        parts = mwr_tb_bias.split(',')
        if len(parts) != mwr_n_tb_fields:
            print('Error: The number of entered MWR Tb bias values does not match desired MWR channels')
            return err
        else:
            bias = np.array(parts).astype(np.float)
        parts = mwr_tb_noise.split(',')
        if len(parts) != mwr_n_tb_fields:
            print('Error: The number of entered MWR Tb noise values does not match desired MWR channels')
            return err
        else:
            noise = np.array(parts).astype(np.float)

        # Only read MWR data if mwr_type > 0

    files = []
    if mwr_type > 0:
        for i in range(len(udate)):
            tempfiles,status = (findfile(path,'*' + rootname + '*' + udate[i] + '*(cdf|nc)'))
            if status == 1:
                return err
            files = files + tempfiles
        if len(files) == 0:
            print('  Warning: No MWR files found for this date')
            mwr_type = 0

    # Only read MWR data if mwr_type > 0
    # Note that it is possible to read in MWR data but no Tbsky data
    nmwr_points = 0
    if mwr_type > 0:
        for i in range(len(files)):
            if verbose >= 2:
                print("    Reading: " + files[i])
            fid = Dataset(files[i],'r')
            fid.set_auto_mask(False)
            if(mwr_type <= 2):
                bt = fid.variables['base_time'][:].astype('float')
                to = fid.variables['time_offset'][:].astype('float')
            elif(mwr_type == 3):
                if verbose >= 3:
                    print("    using the Univ Cologne data format")
                bt = 0
                to = fid.variables['time'][:].astype('float')
            elif(mwr_type == 4):
                if verbose >= 3:
                    print("    using the E-PROFILE data format")
                bt = 0
                to = fid.variables['time'][:].astype('float')
            if len(to) <= 1:
                fid.close()
                continue

            # Get the surface pressure field, if it exists
            # It might be "p_sfc" or "sfc_p" or "sfc_pres". Units are mb in both cases
            foo = np.where(np.array(list(fid.variables.keys())) == 'sfc_pres')[0]
            if len(foo) > 0:
                psfcx = fid.variables['sfc_pres'][:]
            else:
                foo = np.where(np.array(list(fid.variables.keys())) == 'sfc_p')[0]
                if len(foo) > 0:
                    psfcx = fid.variables['sfc_p'][:]
                else:
                   foo = np.where(np.array(list(fid.variables.keys())) == 'p_sfc')[0]
                   if len(foo) > 0:
                       psfcx = fid.variables['p_sfc'][:]
                   else:
                       foo = np.where(np.array(list(fid.variables.keys())) == 'pres')[0]
                       if len(foo) > 0:
                           psfcx = fid.variables['pres'][:]
                       else:
                           foo = np.where(np.array(list(fid.variables.keys())) == 'air_pressure')[0]
                           if len(foo) > 0:
                               psfcx = fid.variables['air_pressure'][:]
                           else:
                               foo = np.where(np.array(list(fid.variables.keys())) == 'pa')[0]
                               if len(foo) > 0:
                                   psfcx  = fid.variables['pa'][:]
                                   psfcx /= 100.    # convert from Pa to hPa (note this is the Univ Cologne MWR format)
                                        # There is a lot of missing data in the pressure field in this format.  So
                                        # we want to interpolate across the gaps.  But we don't want to extrapolate,
                                        # so we need to catch the first and last good point, and use that for the two
                                        # ends of the day.
                                   foo    = np.where(psfcx > 0)[0]
                                   idx0 = foo[0]
                                   idx1 = foo[-1]
                                   psfcx  = np.interp(bt+to,bt+to[foo],psfcx[foo])
                                   psfcx[0:idx0]  = psfcx[idx0]
                                   psfcx[idx1:-1] = psfcx[idx1]
                               else:
                                   psfcx = np.ones(to.shape)*-999.0

            # See if the elevation variable exists. If so, read it in. If not then
            # assume all samples are zenith pointing and create the elev field as such
            foo = np.where(np.array(list(fid.variables.keys())) == mwr_elev_field)[0]
            if len(foo) > 0:
                elevx = fid.variables[mwr_elev_field][:]
            else:
                if((mwr_elev_field != 'none')): 
                    print('    Warning: Unable to find the elevation field ' + mwr_elev_field + ' in the MWR input file')
                elevx = np.ones(to.shape)*90.0

            # Now read in the Tb data, if desired
            if mwr_n_tb_fields > 0:
                if mwr_type == 1:

                    #If mwr_type is 1, then I expect to have mwr_n_tb_fields specified
                    fields = mwr_tb_field_names.split(',')
                    if len(fields) != mwr_n_tb_fields:
                        print('Error: The number of desired MWR Tb fields does not equal the number of input field names')
                        return err

                    for j in range(mwr_n_tb_fields):
                        foo = np.where(np.array(list(fid.variables.keys())) == fields[j])[0]
                        if len(foo) == 0:
                            print('Error: Unable to find the Tb field ' + fields[j] + ' in the MWR input file')
                            return err
                        tmp = fid.variables[fields[j]][:]
                        if j == 0:
                            tbskyx = np.copy(tmp)
                        else:
                            tbskyx = np.vstack((tbskyx,tmp))

                elif((mwr_type == 2) | (mwr_type == 3) | (mwr_type == 4)):

                    # if mwr_type is 2 or 3, then I expect there to only be a single mwr_tb_field_name
                    foo = np.where(np.array(list(fid.variables.keys())) == mwr_tb_field_names)[0]
                    if len(foo) == 0:
                        print('Error: Unable to find the Tb field ' + mwr_tb_field_names + ' in the MWR input file')
                        return err
                    tbskyx = fid.variables[mwr_tb_field_names][:].T

                    # Now get the frequency field that goes with this 2-d array, as we need
                    # to select the subset of channels that are desired (i.e. probably not all
                    # of the channels in the dataset are desired). Select the channels that are
                    # closest in frequency to the entered (desired) frequencies.

                    foo = np.where(np.array(list(fid.variables.keys())) == mwr_freq_field)[0]
                    if len(foo) == 0:
                        print('Error: Unable to find the VIP.mwr_freq_field ('+mwr_freq_field+') in the MWR input file')
                        return err
                    freqx = fid.variables[mwr_freq_field][:]

                    idx = np.ones(freq.shape)*-1
                    for j in range(len(freq)):
                        dell = abs(freq[j]-freqx)
                        foo = np.where(dell == np.nanmin(dell))[0]
                        idx[j] = foo
                    idx = idx.astype(int)
                    tbskyx = np.copy(tbskyx[idx,:])

                else:
                    print('Error: Undefined mwr_type in VIP file -- aborting')
                    return err
            fid.close()

            # Now append the data (i.e., merge data from multiple files)

            if nmwr_points == 0:
                secs = bt+to
                elev = np.copy(elevx)
                if mwr_n_tb_fields > 0:
                    tbsky = np.copy(tbskyx)
                psfc = np.copy(psfcx)
            else:
                secs = np.append(secs, bt+to)
                elev = np.append(elev, elevx)
                if mwr_n_tb_fields > 0:
                    tbsky = np.append(tbsky,tbskyx,axis =1)
                psfc = np.append(psfc, psfcx)
            nmwr_points = len(secs)

        if len(secs) == 0:
            mwr_type = 0

        if mwr_type > 0:

            # Sanity check

            if mwr_n_tb_fields > 0:
                if len(tbsky[:,0]) != mwr_n_tb_fields:
                    print('Error: Big problem in read_mwr')
                    return err
                tbsky0 = np.copy(tbsky)
                for i in range(mwr_n_tb_fields):
                    tbsky[i,:] = tbsky[i,:] + bias[i]

            # Select only the data that have an elevation of 90 degrees (zenith) and
            # where the bias-corrected brightness temperatures are above the cosmic background

            if mwr_n_tb_fields > 0:
                foo = np.where((((elev >= 89) & (elev < 91)) & ((tbsky[0,:] >= 2.7) & (tbsky[0,:] < mwr_tb_field1_tbmax))))[0]
            else:
                foo = np.where(((elev >= 89) & (elev < 91)))[0]

            if len(foo) == 0:
                print('  Warning: All MWR data are at elevations other than 90 degrees (zenith) or of bad quality')
                mwr_type = 0
            else:
                secs = secs[foo]
                if mwr_n_tb_fields > 0:
                    tbsky = np.copy(tbsky[:,foo])
                    tbsky0 = np.copy(tbsky0[:,foo])
                psfc = psfc[foo]

    # Build the output data structure
    if mwr_type == 0:
        return {'success':1, 'type':mwr_type, 'alt':-1}
    else:
        yy = np.array([datetime.utcfromtimestamp(x).year for x in secs])
        mm = np.array([datetime.utcfromtimestamp(x).month for x in secs])
        dd = np.array([datetime.utcfromtimestamp(x).day for x in secs])
        hour = np.array([((datetime.utcfromtimestamp(x)-datetime(yy[0],mm[0],dd[0])).total_seconds())/3600. for x in secs])
        ymd = yy*10000 + mm*100 + dd
        idx = np.arange(0, len(secs)/step, dtype=int)*step

        if mwr_n_tb_fields == 0:
           return ({'success':1, 'secs':secs[idx], 'ymd':ymd[idx], 'hour':hour[idx],
                 'psfc':psfc[idx], 'n_fields':mwr_n_tb_fields,
                 'type':mwr_type, 'rootname':rootname})
        else:
           # Compute the near-air surface temperature (i.e., brightness temperature in a
           # spectral channel sensitive to near-surface air temperature) [degC].  I want 
           # to find the channel closest to 60 GHz (without going over), but it must be 
           # above 56.5 GHz to be used
           foo = np.where((56.5 < freq) & (freq <= 60))[0]
           nearSfcTb = np.zeros(len(secs)) - 999.
           if len(foo) > 0:
               bar = np.where(freq[foo] == np.max(freq[foo]))[0]
               foo = foo[bar[0]]
               nearSfcTb = np.copy(tbsky[foo,:]) - 273.15  # Convert to degC

           return ({'success':1, 'secs':secs[idx], 'ymd':ymd[idx], 'hour':hour[idx],
                 'psfc':psfc[idx], 'n_fields':mwr_n_tb_fields,
                 'type':mwr_type, 'rootname':rootname, 'tbsky_orig':tbsky0[:,idx], 'tbsky_corr':tbsky[:,idx],
                 'freq':freq, 'noise':noise, 'bias':bias, 'nearSfcTb':nearSfcTb[idx]})

################################################################################
# This function reads in the mwr scan data.
################################################################################

def read_mwrscan(path, rootname, date, mwrscan_type, mwrscan_freq_field, mwrscan_elev_field, mwrscan_n_tb_fields,
            mwrscan_tb_field_names, mwrscan_tb_freqs, mwrscan_tb_noise, mwrscan_tb_bias,
            mwrscan_tb_field1_tbmax, mwrscan_n_elevations, mwrscan_elevations, verbose):

    if verbose >= 2:
        print('  Reading MWR-scan data in ' + path)
    err = {'success':0, 'type':-1}

    # Read in the data from yesterday, today, and tomorrow
    udate = [(datetime.strptime(str(date), '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d') ,
              str(date) ,  (datetime.strptime(str(date), '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d') ]

    if mwrscan_type < 0:
        print(' ')
        print('-----------------------------------------------------------------')
        print('****** options for the VIP parameter "mwrscan_type" *******')
        print('   mwrscan_type=0 --->')
        print('             No MWR-scan data specified')
        print('   mwrscan_type=1 --->')
        print('             MWR-scan Tb data are in individual 1-d variables in the input file')
        print('   mwrscan_type=2 --->')
        print('             MWR-scan tb data are in a single 2-d variable in the input file')
        print('   mwrscan_type=3 --->')
        print('             MWR-scan tb data are in a single 2-d variable in the input file, using the "Univ Cologne" format')
        print('   mwrscan_type=4 --->')
        print('             MWR-scan tb data are in a single 2-d variable in the input file, using the "E-PROFILE" format')
        print('-----------------------------------------------------------------')
        print(' ')
        err = {'success':-1}
        return err

    # Let's get the Tb frequencies, noise levels, and bias from the VIP file into arrays
    if mwrscan_n_tb_fields > 0:
        parts = mwrscan_tb_freqs.split(',')
        if len(parts) != mwrscan_n_tb_fields:
            print('Error: The number of entered MWR-scan frequencies does not match desired number of channels')
            return err
        else:
            freq = np.array(parts).astype(np.float)
        foo = np.where(((freq >= 15) & (freq < 240)))[0]
        if len(foo) != mwrscan_n_tb_fields:
            print('Error: The entered MWR-scan frequencies are not between 15 and 240 GHz -- probably an error')
            return err
        parts = mwrscan_tb_bias.split(',')
        if len(parts) != mwrscan_n_tb_fields:
            print('Error: The number of entered MWR-scan Tb bias values does not match desired number of channels')
            return err
        else:
            bias = np.array(parts).astype(np.float)
        parts = mwrscan_tb_noise.split(',')
        if len(parts) != mwrscan_n_tb_fields:
            print('Error: The number of entered MWR-scan Tb noise values does not match desired number of channels')
            return err
        else:
            noise = np.array(parts).astype(np.float)
    if mwrscan_n_elevations > 0:
        parts = mwrscan_elevations.split(',')
        if len(parts) != mwrscan_n_elevations:
            print('Error: The number of entered MWR-scan elevations does not match desired number of elevations')
            return err
        else:
            delev = np.array(parts).astype(np.float)
    else:
        print('Error: The number of entered MWR-scan elevations must be larger than 0')
        return err



    # Only read MWR-scan data if mwrscan_type > 0
    files = []
    if mwrscan_type > 0:
        for i in range(len(udate)):
            tempfiles,status = (findfile(path,'*' + rootname + '*' + udate[i] + '*(cdf|nc)'))
            if status == 1:
                return err
            files = files + tempfiles
        if len(files) == 0:
            print('  Warning: No MWR files found for this date')
            mwrscan_type = 0

    # Only read MWR-scan data if mwrscan_type > 0
    nmwrscan_points = 0
    if mwrscan_type > 0:
        for i in range(len(files)):
            fid = Dataset(files[i],'r')
            fid.set_auto_mask(False)
            if(mwrscan_type <= 2):
                bt = fid.variables['base_time'][:].astype('float')
                to = fid.variables['time_offset'][:].astype('float')
            elif(mwrscan_type == 3):
                if verbose >= 3:
                    print("    using the Univ Cologne data format")
                bt = 0
                to = fid.variables['time'][:].astype('float')
            elif(mwrscan_type == 4):
                if verbose >= 3:
                    print("    using the E-PROFILE data format")
                bt = 0
                to = fid.variables['time'][:].astype('float')
            if len(to) <= 1:
                fid.close()
                continue

            # See if the elevation variable exists. If so, read it in. If not, then
            # assume all samples are zenith pointing and create the elev field as such
            vid = np.where(np.array(list(fid.variables.keys())) == mwrscan_elev_field)[0]
            if len(vid) > 0:
                elevx = fid.variables[mwrscan_elev_field][:]
            else:
                print('  Warning: Unable to find the field ' + mwrscan_elev_field + ' in the MWR-scan input file')

            # Now read in the Tb data, if desired
            if mwrscan_n_tb_fields > 0:
                if mwrscan_type == 1:

                    #If mwrscam_type is 1, then I expect to have mwrscan_n_tb_fields specified
                    fields = mwrscan_tb_field_names.split(',')
                    if len(fields) != mwrscan_n_tb_fields:
                        print('Error: The number of desired MWR-scan Tb fields does not equal the number of input field names')
                        return err

                    for j in range(mwrscan_n_tb_fields):
                        vid = np.where(np.array(list(fid.variables.keys())) == fields[j])[0]
                        if len(vid) == 0:
                            print('Error: Unable to find the Tb field ' + fields[j] + ' in the MWR-scan input file')
                        tmp = fid.variables[fields[j]][:]
                        if j == 0:
                            tbskyx = np.copy(tmp)
                        else:
                            tbskyx = np.vstack((tbskyx,tmp))

                elif ((mwrscan_type == 2) | (mwrscan_type == 4)):

                    # if mwrscan_type is 2, then I expect there to only be a single mwrscan_tb_field_name
                    vid = np.where(np.array(list(fid.variables.keys())) == mwrscan_tb_field_names)[0]
                    if len(vid) == 0:
                        print('Error: Unable to find the Tb field ' + mwrscan_tb_field_names + ' in the MWR-scan input file')
                        return err
                    tbskyx = fid.variables[mwrscan_tb_field_names][:].T

                    # Now get the frequency field that goes with this 2-d array, as we need
                    # to select the subset of channels that are desired (i.e. probably not all
                    # of the channels in the dataset are desired). Select the channels that are
                    # closest in frequency to the entered (desired) frequencies.

                    vid = np.where(np.array(list(fid.variables.keys())) == 'freq')[0]
                    if len(vid) == 0:
                        vid = np.where(np.array(list(fid.variables.keys())) == 'frequency')[0]

                        if len(vid) == 0:
                            print('Error: Unable to find the field "freq" or "frequency" in the MWR-scan input file')
                            return err
                        else:
                            freqx = fid.variables['frequency'][:]
                    else:
                        freqx = fid.variables['freq'][:]

                    idx = np.ones(freq.shape, dtype=int)*-1
                    for j in range(len(freq)):
                        dell = abs(freq[j]-freqx)
                        foo = np.where(dell == np.nanmin(dell))[0]
                        idx[j] = foo
                    tbskyx = np.copy(tbskyx[idx,:])

                elif mwrscan_type == 3:

                    # if mwrscan_type is 3, then I expect there to only be a single mwrscan_tb_field_name in the Univ Cologne format
                    vid = np.where(np.array(list(fid.variables.keys())) == mwrscan_tb_field_names)[0]
                    if len(vid) == 0:
                        print('Error: Unable to find the Tb field ' + mwrscan_tb_field_names + ' in the MWR-scan input file')
                        return err
                    tbskyx = fid.variables[mwrscan_tb_field_names][:].T
                    sz = tbskyx.shape
                    if(len(sz) != 3):
                        print('Error: For mwrscan_type == 3, the Tbsky field ('+mwrscan_tb_field_names+') needs to be 3-dimensional')
                        sys.exit()

                    # Now get the frequency field that goes with this 2-d array, as we need
                    # to select the subset of channels that are desired (i.e. probably not all
                    # of the channels in the dataset are desired). Select the channels that are
                    # closest in frequency to the entered (desired) frequencies.

                    foo = np.where(np.array(list(fid.variables.keys())) == mwrscan_freq_field)[0]
                    if len(foo) == 0:
                        print('Error: Unable to find the VIP.mwrscan_freq_field ('+mwrscan_freq_field+') in the MWR input file')
                        return err
                    freqx = fid.variables[mwrscan_freq_field][:]

                    # I need to unroll the 3-d Tbsky data into a 2-d array, with elevation scans having different times
                    newtb = np.zeros((len(freqx),len(elevx)*len(to)))
                    newto = np.zeros(len(elevx)*len(to))
                    newel = np.zeros(len(elevx)*len(to))
                    for ii in range(len(to)):
                        for jj in range(len(elevx)):
                            newto[ii*len(elevx)+jj]   = to[ii]+jj
                            newel[ii*len(elevx)+jj]   = elevx[jj]
                            newtb[:,ii*len(elevx)+jj] = tbskyx[:,jj,ii]
                    tbskyx = np.copy(newtb)
                    to     = np.copy(newto)
                    elevx  = np.copy(newel)

                    idx = np.ones(freq.shape, dtype=int)*-1
                    for j in range(len(freq)):
                        dell = abs(freq[j]-freqx)
                        foo = np.where(dell == np.nanmin(dell))[0]
                        idx[j] = foo
                    tbskyx = np.copy(tbskyx[idx,:])

                else:
                    print('Error: Undefined mwrscan_type in VIP file -- aborting')
                    return err
            fid.close()

            # Now append the data (i.e., merge data from multiple files)

            if nmwrscan_points == 0:
                secs = bt+to
                elev = np.copy(elevx)
                if mwrscan_n_tb_fields > 0:
                    tbsky = np.copy(tbskyx)
            else:
                secs = np.append(secs, bt+to)
                elev = np.append(elev, elevx)
                if mwrscan_n_tb_fields > 0:
                    tbsky = np.append(tbsky,tbskyx,axis =1)
            nmwrscan_points = len(secs)

        if len(secs) == 0:
            mwrscan_type = 0

        if mwrscan_type > 0:

            # Sanity check
            if mwrscan_n_tb_fields > 0:
                if len(tbsky[:,0]) != mwrscan_n_tb_fields:
                    print('Error: Big problem in read_mwrscan')
                    return err
                tbsky0 = np.copy(tbsky)
                for i in range(mwrscan_n_tb_fields):
                    tbsky[i,:] = tbsky[i,:] + bias[i]

            # This should never happen. If it does assume an error and abort
            if mwrscan_n_tb_fields <= 0:
                print('Error: The number of Tb fields in MWR-scan was less than 1 -- aborting')
                return err

            # Select only the data that have elevations that match the desired elevations (within 1 deg)
            # and where the bias-corrected brightness temperature are above the cosmic background (2.7 K)

            keep_samples = np.zeros(len(secs))    # all zeros right now
            for i in range(mwrscan_n_elevations):
                foo = np.where((((elev >= delev[i]-1) & (elev <= delev[i]+1)) & ((tbsky[0,:] >= 2.7) & (tbsky[0,:] < mwrscan_tb_field1_tbmax))))[0]
                if len(foo) > 0:
                    keep_samples[foo] = 1
            foo = np.where(keep_samples == 1)[0]
            if len(foo) == 0:
                print('  Warning: Did not find any MWR-scan data at the desired elevations')
                mwrscan_type = 0
            else:
                secs = np.copy(secs[foo])
                elev = np.copy(elev[foo])
                if mwrscan_n_tb_fields > 0:
                    tbsky = np.copy(tbsky[:,foo])
                    tbsky0 = np.copy(tbsky0[:,foo])

    if mwrscan_type == 0:
        return {'success':1, 'type':mwrscan_type}
    else:
        yy = np.array([datetime.utcfromtimestamp(x).year for x in secs])
        mm = np.array([datetime.utcfromtimestamp(x).month for x in secs])
        dd = np.array([datetime.utcfromtimestamp(x).day for x in secs])
        hour = np.array([((datetime.utcfromtimestamp(x)-datetime(yy[0],mm[0],dd[0])).total_seconds())/3600. for x in secs])
        ymd = yy*10000 + mm*100 + dd

        return ({'success':1, 'secs':secs, 'ymd':ymd, 'hour':hour, 'elev':elev,
                'n_fields':mwrscan_n_tb_fields,
                'type':mwrscan_type, 'rootname':rootname, 'tbsky_orig':tbsky0, 'tbsky_corr':tbsky,
                'freq':freq, 'noise':noise, 'bias':bias})



################################################################################
# This function reads in the irs_eng file.
################################################################################

def read_irs_eng(path, date, irs_type, verbose):

    if verbose >= 1:
        print('  Reading irs_eng data in ' + path)
    err = {'success':0}
    if irs_type <= 0:
        print(' ')
        print('-----------------------------------------------------------------')
        print('****** options for the VIP parameter "irs_type" ******')
        print('   irs_type=1 --->')
        print('             ARM-style ch1/ch2/sum/eng AERI input')
        print('   irs_type=2 --->')
        print('             SSEC dmv-ncdf converted netcdf files, with')
        print('                     _sum.cdf and C1_rnc.cdf endings')
        print('   irs_type=3 --->')
        print('             SSEC dmv-ncdf converted netcdf files, with ')
        print('                     .SUM.cdf and C1.RNC,cdf endings')
        print('   irs_type=4 --->')
        print('             ARM-style ch1/ch2/sum AERI datastream names, but all of the')
        print('                     engineering data is found in the summary file (dmv2ncdf)')
        print('   irs_type=5 --->')
        print('             ARM-style ch1/ch2/sum ASSIST datastream names, but all of the')
        print('                     engineering data is found in the summary file')
        print('   irs_type=6 --->')
        print('             ARM-style ch1/ch2/sum REFIR-PAD datastream names, but all of the')
        print('                     engineering data is found in the summary file and the ')
        print('                     spectral resolution of the radiance data is 0.4000000 cm-1')
        print('   irs_type=7 --->')
        print('             ARM-style ch1/ch2/sum REFIR-PAD datastream names, but all of the')
        print('                     engineering data is found in the summary file and the ')
        print('                     spectral resolution of the radiance data is 0.3892450 cm-1')
        print('-----------------------------------------------------------------')
        print(' ')
        err = {'success':0}
        return err

    elif irs_type == 1:
        filename = '*aeri*eng*' + str(date) + '*(cdf|nc)'
    elif irs_type == 2:
        filename = '*' + str(date % 2000000) + '_sum.(cdf|nc)'
    elif irs_type == 3:
        filename = '*' + str(date % 2000000) + '.SUM.(cdf|nc)'
    elif irs_type == 4:
        filename = '*aeri*sum*' + str(date) + '*(cdf|nc)'
    elif irs_type == 5:
        filename = '*assist*sum*' + str(date) + '*(cdf|nc)'
    elif ((irs_type == 6) or (irs_type == 7)):
        filename = '*refir*sum*' + str(date) + '*(cdf|nc)'
    else:
        print('Error in read_irs_eng: unable to decode irs_type')
        return err

    if verbose == 3:
        print('    Looking for IRS engineering data as ' + filename)

    files,status = findfile(path,filename)
    if status == 1:
        return err

    if len(files) == 0:
        print('Error: Unable to find any IRS engineering data - aborting')
        print('Set irs_type=0 and rerun to compare format of IRS data to irs_type options')
        return err

    for jj in range(len(files)):
        fid = Dataset(files[jj],'r')
        fid.set_auto_mask(False)
        bt = fid['base_time'][:].astype('float')
            # If this is an ASSIST, then convert the base_time from milliseconds to seconds
        if(irs_type == 5):
            bt = bt / 1000.
            # Check for "time_offset"; if not there, assume it is "time"
        if len(np.where(np.array(list(fid.variables.keys())) == 'time_offset')[0]) > 0:
            to = fid['time_offset'][:].astype('float')
        else:
            to = fid['time'][:].astype('float')

        # I want to get the temperature of the plug in the 2nd input port of the
        # interfermoeter. The name of this field was different for the v2 and v4
        # IRS systems, so I have to look for either field

        if len(np.where(np.array(list(fid.variables.keys())) == 'interferometerSecondPortTemp')[0]) > 0:
            xinterferometerSecondPortTemp = fid.variables['interferometerSecondPortTemp'][:]
        elif len(np.where(np.array(list(fid.variables.keys())) == 'airNearInterferometerTemp')[0]) > 0:
            xinterferometerSecondPortTemp = fid.variables['airNearInterferometerTemp'][:]
        elif len(np.where(np.array(list(fid.variables.keys())) == 'port2_bb_temp')[0]) > 0:
            xinterferometerSecondPortTemp  = fid.variables['airNearInterferometerTemp'][:]
            xinterferometerSecondPortTemp += 273.15
        else:
            xinterferometerSecondPortTemp = np.ones(len(to))*-999.0

        #If the airNearInterferometerTemp exists, I will get this too. It is not used.

        if len(np.where(np.array(list(fid.variables.keys())) == 'airNearInterferometerTemp')[0]) > 0:
            xairNearInterferometerTemp = fid.variables['airNearInterferometerTemp'][:]
        else:
            xairNearInterferometerTemp = np.ones(len(to))*-999.0

        if len(np.where(np.array(list(fid.variables.keys())) == 'BBcavityFactor')[0]) > 0:
            xbbcavityfactor = fid.variables['BBcavityFactor'][:]
        else:
            xbbcavityfactor = np.ones(len(to))*12.79

        fid.close()

        #Append the data from the different files into single arrays
        if jj == 0:
            secs = bt+to
            interferometerSecondPortTemp = np.copy(xinterferometerSecondPortTemp)
            airNearInterferometerTemp = np.copy(xairNearInterferometerTemp)
            bbcavityfactor = np.copy(xbbcavityfactor)
        else:
            secs = np.append(secs,bt+to)
            interferometerSecondPortTemp = np.append(interferometerSecondPortTemp, xinterferometerSecondPortTemp)
            airNearInterferometerTemp = np.append(airNearInterferometerTemp, xairNearInterferometerTemp)
            bbcavityfactor = np.append(bbcavityfactor, xbbcavityfactor)

    #If the interferometerSecondPortTemp are all negative (i.e., missing)
    #then replace these values with the airNearInterferometerTemp

    foo = np.where(interferometerSecondPortTemp > 50)
    if len(foo) == 0:
        interferometerSecondPortTemp = np.copy(airNearInterferometerTemp)

    yy = np.array([datetime.utcfromtimestamp(x).year for x in secs])
    mm = np.array([datetime.utcfromtimestamp(x).month for x in secs])
    dd = np.array([datetime.utcfromtimestamp(x).day for x in secs])
    hour = np.array([((datetime.utcfromtimestamp(x)-datetime(yy[0],mm[0],dd[0])).total_seconds())/3600. for x in secs])

    ymd = yy*10000 + mm*100 + dd

    # Sometimes the data are not in time-ascending order.  So sort the data
    sidx = np.argsort(secs)

    return ({'success':1, 'secs':secs[sidx], 'ymd':ymd[sidx], 'hour':hour[sidx],
            'bbcavityfactor':bbcavityfactor[sidx],
            'interferometerSecondPortTemp':interferometerSecondPortTemp[sidx]})



################################################################################
# This function reads the irs_sum file
################################################################################

def read_irs_sum(path,date,irs_type,smooth_noise,verbose):
    if verbose >= 1:
        print('  Reading irs_sum data in ' + path)
    err = {'success':0}
    if irs_type <= 0:
        print(('Error in read_irs_sum: This piece of code should not be exercised, as this option should ' +
               'be screened in read_irs_eng() earlier'))
        return err
    elif ((irs_type == 1) | (irs_type == 4)):
        filename = '*aeri*sum*' + str(date) + '*(cdf|nc)'
    elif irs_type == 2:
        filename = '*' + str(date % 2000000) + '_sum.(cdf|nc)'
    elif irs_type == 3:
        filename = '*' + str(date % 2000000) + '.SUM.(cdf|nc)'
    elif irs_type == 5:
        filename = '*assist*sum*' + str(date) + '*(cdf|nc)'
    elif ((irs_type == 6) or (irs_type == 7)):
        filename = '*refir*sum*' + str(date) + '*(cdf|nc)'
    else:
        print('Error in read_irs_sum: unable to decode irs_type')
        return err

    if verbose >= 3:
        print('  Looking for IRS summary data as ' + filename)

    files, status = findfile(path,filename)
    if status == 1:
        return err

    if len(files) == 0:
        print('Error: Unable to find any IRS summary data - aborting')

    for jj in range(len(files)):
        fid = Dataset(files[jj],'r')
        fid.set_auto_mask(False)
        bt = fid['base_time'][:].astype('float')
            # If this is an ASSIST, then convert the base_time from milliseconds to seconds
        if(irs_type == 5):
            bt = bt / 1000.
            # Check for "time_offset"; if not there, assume it is "time"
        if len(np.where(np.array(list(fid.variables.keys())) == 'time_offset')[0]) > 0:
            to = fid['time_offset'][:].astype('float')
        else:
            to = fid['time'][:].astype('float')

        if len(np.where(np.array(list(fid.variables.keys())) == 'wnumsum5')[0]) > 0:
            wnum1 = fid.variables['wnumsum5'][:]
            wnum2 = fid.variables['wnumsum6'][:]
        elif len(np.where(np.array(list(fid.variables.keys())) == 'wnumsum1')[0]) > 0:
            wnum1 = fid.variables['wnumsum1'][:]
            wnum2 = fid.variables['wnumsum2'][:]
        elif len(np.where(np.array(list(fid.variables.keys())) == 'wnum1')[0]) > 0:
            wnum1 = fid.variables['wnum1'][:]
            wnum2 = fid.variables['wnum2'][:]
        else:
            print('Error in read_irs_sum: unable to find the wnumsum fields')
            return err

        if len(np.where(np.array(list(fid.variables.keys())) == 'SkyNENCh1')[0]) > 0:
            xnoise1 = fid.variables['SkyNENCh1'][:]
        elif len(np.where(np.array(list(fid.variables.keys())) == 'SkyNENch1')[0]) > 0:
            xnoise1 = fid.variables['SkyNENch1'][:]
        elif len(np.where(np.array(list(fid.variables.keys())) == 'ch1_sky_nen')[0]) > 0:
            xnoise1 = fid.variables['SkyNENch1'][:]
        else:
            print('Error in read_irs_sum: unable to find the SkyNENCh1 field')
            return err

        if len(np.where(np.array(list(fid.variables.keys())) == 'SkyNENCh2')[0]) > 0:
            xnoise2 = fid.variables['SkyNENCh2'][:]
        elif len(np.where(np.array(list(fid.variables.keys())) == 'SkyNENch2')[0]) > 0:
            xnoise2 = fid.variables['SkyNENch2'][:]
        elif len(np.where(np.array(list(fid.variables.keys())) == 'ch2_sky_nen')[0]) > 0:
            xnoise2 = fid.variables['SkyNENch2'][:]
        else:
            print('Error in read_irs_sum: unable to find the SkyNENCh2 field')
            return err

        fid.close()

        #Append the data together into single arrays
        if jj == 0:
            secs = bt + to
            noise1 = np.copy(xnoise1)
            noise2 = np.copy(xnoise2)
        else:
            secs = np.append(secs,bt+to)
            noise1 = np.append(noise1,xnoise1,axis = 0)
            noise2 = np.append(noise2,xnoise2,axis = 0)

    noise1 = noise1.T
    noise2 = noise2.T

    # Only keep the noise from the MCT detector below 1800 cm-1
    foo    = np.where(wnum1 < 1800)[0]
    wnum1  = wnum1[foo]
    noise1 = noise1[foo,:]

    # Now append the noise from the InSB detector above that limit
    foo = np.where(wnum2 >= np.nanmax(wnum1)+0.1)[0]
    wnum = np.append(wnum1,wnum2[foo])
    noise = np.append(noise1,noise2[foo,:], axis = 0)

    # If desired, smooth the random error in the IRS observations
    # over at temporal window. This is needed because the IRS derives
    # an independent noise estimate for each sample, but sometimes the
    # temporal variation in the noise can impact the retrieval in
    # negative ways

    if smooth_noise > 0:
        if verbose >= 2:
            print('  Smoothing the IRS noise temporally')

            #Derive the number of points to use in the smoother. The
            #variable "smooth_noise" is the temporal window in minutes,
            #so I will select the median time offset between samples
            #(which is in seconds) to determine this value

        delt = np.nanmedian(secs[1:len(secs)] - secs[0:len(secs)-1])
        npts = int(60*smooth_noise/delt)
        if npts % 2 == 0:
            npts = npts + 1
        snoise = np.copy(noise)

        #Apply the temporal noise smoother
        for j in range(len(secs)):
            vals = j + (np.arange(npts)-npts/2)
            foo = np.where((vals >= 0) & (vals <= len(secs)))[0]
            vals = vals[foo]
            for i in range(wnum):
                snoise[i,j] = np.nanmedian(noise[i,vals])
        npts = npts/4 + 1
        for i in range(wnum):
            snoise[i,:], flag = Other_functions.smooth(snoise[i,:],npts)
            if flag == 0:
                print('Error in IRS smoothing function. Aborting.')
                return err
        noise = np.copy(snoise)

    # Get some times from the data
    yy = np.array([datetime.utcfromtimestamp(x).year for x in secs])
    mm = np.array([datetime.utcfromtimestamp(x).month for x in secs])
    dd = np.array([datetime.utcfromtimestamp(x).day for x in secs])
    ymd = yy*10000 + mm*100 + dd
    hour = np.array([((datetime.utcfromtimestamp(x)-datetime(yy[0],mm[0],dd[0])).total_seconds())/3600. for x in secs])

    # Sometimes the data are not in time-ascending order.  So sort the data
    sidx = np.argsort(secs)

    return ({'success':1, 'secs':secs[sidx],'ymd':ymd[sidx], 'hour':hour[sidx], 
             'wnum':wnum, 'noise':noise[:,sidx]})

################################################################################
# This function read in the ceilometer/vertically pointing lidar data.
################################################################################

def read_vceil(path, date, vceil_type, ret_secs, mpd_abc_thres, verbose):
    if verbose >= 2:
        print('  Reading ceilometer data in ' + path)

    # Read in the cloud base height data from yesterday, today, and tomorrow.
    udate = [(datetime.strptime(str(date), '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d') ,
            str(date) ,  (datetime.strptime(str(date), '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d') ]

    vdate = [str(int((datetime.strptime(str(date), '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d')) % 1000000) ,
            str(date % 1000000), str(int((datetime.strptime(str(date), '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')) % 1000000) ]

    err = {'success':0}

    files = []
    if vceil_type < 0:
        print(' ')
        print('------------------------------------------------------')
        print('****** options for the VIP parameter "cbh_type" ******')
        print('   cbh_type=0 --->')
        print('             No cloud base height data used')
        print('             Default value used for all times')
        print('   cbh_type=1 --->')
        print('             ARM-style Vaisala ceilometer input')
        print('                     Files named "*ceil*cdf" ')
        print('                     Reads field "first_cbh", which has units m AGL')
        print('   cbh_type=2 --->')
        print('             Greg Blumbergs simple tranlation of ASOS CBH dataset')
        print('                     Files named "*ceil*cdf" or "cbh*cdf" ')
        print('                     Reads field "cloudHeight", which has units km AGL')
        print('   cbh_type=3 --->')
        print('             CLAMPS Doppler lidar fixed point (zenith) data')
        print('                     Files named "*dlfp*cdf"')
        print('                     Reads field "cbh", which has units km AGL')
        print('   cbh_type=4 --->')
        print('             ARM Doppler lidar "dlprofwstats" data')
        print('             Files named "*dlprofwstats*cdf" ')
        print('             Reads field "dl_cbh", which has units m AGL')
        print('   cbh_type=5 --->')
        print('             JOYCE-style Vaisala ceilometer input')
        print('                     Files named "*ct25k*nc" ')
        print('                     Reads field "first_cbh", which has units m AGL')
        print('   cbh_type=6 --->')
        print('             JOYCE-style Jenoptic ceilometer input')
        print('                     Files named "*CHM*nc" ')
        print('                     Reads field "cbh", which has units m AGL')
        print('                     Time field is weird (starts at 1904)')
        print('   cbh_type=7 --->')
        print('             E-PROFILE style ceilometer input')
        print('                     Files named "{alc,L2}*nc" ')
        print('                     Reads field "cbh", which has units m AGL')
        print('   cbh_type=8 --->')
        print('             DWD CHM15k style ceilometer input')
        print('                     Files named "*ceil*nc" ')
        print('                     Reads field "cbh(3,time)", which has units m AGL')
        print('   cbh_type=9 --->')
        print('             Vaisala CF51 BL-View stlye ceilometer input')
        print('                     Files named "L2*nc" ')
        print('                     Reads field "cloud_data[time,0]", which has units m AGL')
        print('   cbh_type=10 -->')
        print('             CBH derived from NCAR MPD aerosol_backscatter_coefficient')
        print('                     Files named "mpd04*nc" ')
        print('                     CBH comes from range coordinate variable, units m AGL')
        print('-------------------------------------------------------')
        print(' ')
        err = {'success':-1}
        return err
    elif vceil_type == 0:
        if verbose >= 1:
            print('  User selected the option indicating to use the default cloud cloud base height set in the VIP')
        return err
    elif ((vceil_type == 1) | (vceil_type == 5)):
        if verbose >= 1:
            print('  Reading in CEIL data')
        for i in range(len(udate)):
            tempfiles,status = (findfile(path,'*(ceil|ct25)*' + udate[i] + '*.(cdf|nc)'))
            if status == 1:
                return err
            files = files + tempfiles
        if len(files) == 0:
            print('    No CBH files found for this date')
            return err

        for i in range(len(files)):
            fid = Dataset(files[i],'r')
            fid.set_auto_mask(False)
            bt = fid.variables['base_time'][:].astype('float')
            if len(np.where(np.array(list(fid.variables.keys())) == 'time_offset')[0]) > 0:
                to = fid.variables['time_offset'][:].astype('float')
            elif len(np.where(np.array(list(fid.variables.keys())) == 'time')[0]) > 0:
                to = fid.variables['time'][:].astype('float')
            else:
                fid.close()
                print('  Error reading the time fields -- aborting read_vceil')
                return err
            cbhx = fid.variables['first_cbh'][:]
            fid.close()
            cbhx = cbhx / 1000.

            if i == 0:
                secs = bt+to
                cbh = np.copy(cbhx)
            else:
                secs = np.append(secs,bt+to)
                cbh = np.append(cbh,cbhx)

    elif vceil_type == 2:
        if verbose >= 1:
            print('  Reading in ASOS/AWOS ceilometer')

        for i in range(len(udate)):
            tempfiles, status = (findfile(path, '*(ceil|cbh)*' + udate[i] + '*.(cdf|nc)'))
            if status == 1:
                return err
            files = files + tempfiles
        if len(files) == 0:
            print('  No CBH files found for this data')
            return err

        for i in range(len(files)):
            if verbose == 3:
                print('    Reading in file ' + files[i])

            fid = Dataset(files[i],'r')
            fid.set_auto_mask(False)
            bt = fid.variables['base_time'][:].astype('float')
            to = fid.variables['time_offset'][:].astype('float')
            cbhx = fid.variables['cloudHeight'][:]
            fid.close()

            if i == 0:
                secs = bt + to
                cbh = np.copy(cbhx)
            else:
                secs = np.append(secs,bt+to)
                cbh = np.append(cbh,cbhx)

    elif vceil_type == 3:
        if verbose >= 1:
            print('  Reading in CLAMPS dlfp data')
        for i in range(len(udate)):
            tempfiles, status = (findfile(path,'*dlfp*' + udate[i] + '*.(cdf|nc)'))
            if status == 1:
                return err
            files = files + tempfiles
        if len(files) == 0:
            print('    No CBH files found for this date')
            return err
        for i in range(len(files)):
            if verbose == 3:
                print('    Reading the file ' + files[i])

            fid = Dataset(files[i],'r')
            fid.set_auto_mask(False)
            bt = fid.variables['base_time'][:].astype('float')
            to = fid.variables['time_offset'][:].astype('float')
            cbhx = fid.variables['cbh'][:]
            fid.close()

            if i == 0:
                secs = bt + to
                cbh = np.copy(cbhx)
            else:
                secs = np.append(secs,bt+to)
                cbh = np.append(cbh,cbhx)

    elif vceil_type == 4:
        if verbose >= 1:
            print('  Reading in ARM dlprofwstats data')
        for i in range(len(udate)):
            tempfiles, status = (findfile(path,'*dlprofwstats*' + udate[i] + '*.(cdf|nc)'))
            if status == 1:
                return err
            files = files + tempfiles
        if len(files) == 0:
            print('    No CBH files found for this date')
            return err

        for i in range(len(files)):
            if verbose == 3:
                print('    Reading the file ' + files[i])
            fid = Dataset(files[i],'r')
            fid.set_auto_mask(False)
            bt = fid.variables['base_time'][:].astype('float')
            to = fid.variables['time_offset'][:].astype('float')
            cbhx = fid.variables['dl_cbh'][:]
            fid.close()

            if i == 0:
                secs = bt + to
                cbh = np.copy(cbhx)
            else:
                secs = np.append(secs,bt+to)
                cbh = np.append(cbh,cbhx)
        cbh = cbh/1000.              # Convert m AGL to km AGL

    elif vceil_type == 6:
        if verbose >= 1:
            print('  Reading in JOYCE Jenoptic data')
        for i in range(len(udate)):
            tempfiles, status = (findfile(path,'*CHM*' + udate[i] + '*.(cdf|nc)'))
            if status == 1:
                return err
            files = files + tempfiles
        if len(files) == 0:
            print('    No CBH files found for this date')
            return err

        for i in range(len(files)):
            if verbose == 3:
                print('    Reading the file ' + files[i])
            fid = Dataset(files[i],'r')
            fid.set_auto_mask(False)
            to  = fid.variables['time'][:]
            cbhx = fid.variables['cbh'][:]
            fid.close()

            if i == 0:
                secs = to
                cbh = np.copy(cbhx)
            else:
                secs = np.append(secs,to)
                cbh = np.append(cbh,cbhx)
        secs -= 2.0828448e+09   # This is the appropriate offset to apply to get to unix time
        cbh = cbh/1000.              # Convert m AGL to km AGL

    elif vceil_type == 7:
        if verbose >= 1:
            print('  Reading in E-PROFILE ceilometer data')
        for i in range(len(udate)):
            tempfiles, status = (findfile(path,'*(alc|ceil|L2)*' + udate[i] + '*.nc'))
            if status == 1:
                return err
            files = files + tempfiles
        if len(files) == 0:
            print('    No CBH files found for this date')
            return err

        for i in range(len(files)):
            if verbose == 3:
                print('    Reading the file ' + files[i])
            fid = Dataset(files[i],'r')
            fid.set_auto_mask(False)
            to  = fid.variables['time'][:].astype('float')
                # Because the E-PROFILE has this field as a 2-d field
            cbhx = fid.variables['cloud_base_height'][:,0]
            visx = fid.variables['vertical_visibility'][:]
            fid.close()

            if i == 0:
                secs = to
                cbh  = np.copy(cbhx)
                vis  = np.copy(visx)
            else:
                secs = np.append(secs,to)
                cbh  = np.append(cbh,cbhx)
                vis  = np.append(vis,visx)
        secs *= (24.*60*60)	     # Convert unix days into unix time
        cbh = cbh/1000.              # Convert m AGL to km AGL
        vis = vis/1000.              # Convert m AGL to km AGL
        bt  = secs[0]

        # If the CBH <= and vis > 0, the replace the CBH with visibility
        foo = np.where( (cbh <= 0) & (vis > 0) )[0]
        if len(foo) > 1:
            if verbose >= 1:
                print('      Replacing some non-positive CBH values with visibility ' +
                              'in the E-PROFILE ceilometer reader')
            cbh[foo] = vis[foo]

    elif vceil_type == 8:
        if verbose >= 1:
            print('  Reading in DWD CHM15k ceilometer data')
        for i in range(len(udate)):
            tempfiles, status = (findfile(path,'*ceil*' + udate[i] + '*.(cdf|nc)'))
            if status == 1:
                return err
            files = files + tempfiles
        if len(files) == 0:
            print('    No CBH files found for this date')
            return err

        for i in range(len(files)):
            if verbose == 3:
                print('    Reading the file ' + files[i])
            fid = Dataset(files[i],'r')
            fid.set_auto_mask(False)
            to  = fid.variables['time'][:]
            cbhx = fid.variables['cbh'][:]
            fid.close()
            cbhx = cbhx[:,0]        # Keep only the first dimension (i.e., the lowest cloud base)

            if i == 0:
                secs = to
                cbh = np.copy(cbhx)
            else:
                secs = np.append(secs,to)
                cbh = np.append(cbh,cbhx)
        cbh = cbh/1000.              # Convert m AGL to km AGL
        bt  = secs[0]

    elif vceil_type == 9:
        if verbose >= 1:
            print('  Reading in Viasala CL51 ceilometer data (L2 BL-View NetCDF files)')
        for i in range(len(udate)):
            tempfiles, status = (findfile(path, '*(alc|ceil|L2)*' + udate[i] + '*.nc'))
            if status == 1:
                return err
            files = files + tempfiles
        if len(files) == 0:
            print('    No CBH files found for this date')
            return err

        for i in range(len(files)):
            if verbose == 3:
                print('    Reading the file ' + files[i])
            fid = Dataset(files[i], 'r')
            fid.set_auto_mask(False)
            to = fid.variables['time'][:].astype('float')
            # CL51 supports upto 3 cloud layers; we want the first (lowest); hence the [:, 0]
            cbhx = fid.variables['cloud_data'][:, 0]
            fid.close()

            if i == 0:
                secs = to
                cbh = np.copy(cbhx)
            else:
                secs = np.append(secs, to)
                cbh = np.append(cbh, cbhx)

        cbh = cbh / 1000.  # Convert m AGL to km AGL
        bt = secs[0]

        nonpos = np.where(cbh <= 0)
        if len(nonpos) > 1:
            print('     Some CBH values are non-positive!! ')

    elif vceil_type == 10:
        if verbose >= 1:
            print('  Reading in CBH data from NCAR MPD datasets')
        for i in range(len(udate)):
            tempfiles, status = (findfile(path, 'mpd04.*' + udate[i] + '*.nc'))
            if status == 1:
                return err
            files = files + tempfiles
        if len(files) == 0:
            print('    No CBH files found for this date')
            return err

        for i in range(len(files)):
            if verbose == 3:
                print('    Reading the file ' + files[i])
            fid = Dataset(files[i], 'r')
            fid.set_auto_mask(False)
            tvar = fid.variables['time']
            tsecs = tvar[:].astype('float')
            aerosolBackscatter = fid.variables['Aerosol_Backscatter_Coefficient']
            lidarrange = fid.variables['range']

            # The premise here is that aerosol-backscatter (ABC) values above a specific
            # threshold can be considered cloud. The algorithm then for finding CBH is
            # for each time step, search ABC along the range dimension for >= the threshold,
            # and if found, the index along the range dimension is used to look up the
            # height in the 'range' coordinate variable.
            cbhx = np.zeros(len(tsecs), float)
            for j in range(len(tsecs)):
                for k in range(len(lidarrange)):
                    if aerosolBackscatter[j, k] > mpd_abc_thres:
                        cbhx[j] = lidarrange[k]
                        break

            cal = 'standard'
            print(tvar)
            if 'calendar' in tvar.ncattrs():
                cal = tvar.calendar
            tsecs = num2date(tvar, tvar.units, calendar=cal,
                                only_use_cftime_datetimes=False, only_use_python_datetimes=True)
            tsecs = [x.timestamp() for x in tsecs]

            fid.close()

            # convert from "secs since start of day" to unix time
            #y = int(udate[i][0:4])
            #m = int(udate[i][4:6])
            #d = int(udate[i][6:8])
            #dt = datetime(y, m, d, 0, 0)
            #basetime = time.mktime(dt.timetuple())
            #print(y, m, d, dt, basetime)
            #tsecs = tsecs + basetime
            print(files[i], ":")
            print(tsecs)
            print(cbhx)

            if i == 0:
                secs = tsecs
                cbh = np.copy(cbhx)
            else:
                secs = np.append(secs, tsecs)
                cbh = np.append(cbh, cbhx)

        cbh = cbh / 1000.  # Convert m AGL to km AGL
        bt = secs[0]

        foo = np.where(cbh <= 0)
        if len(foo) > 1:
            print('     Some CBH values are non-positive!! ')

    else:
        print('Error in read_vceil: Undefined ceilometer type')
        return err

    # Put in a trap for the GDL version of the code, as occasionally some
    # of Greg's mimicked ceilometer data files (generated from ASOS/AWOS data
    # have the base_time as type int64, which GDL did not handle well...

    if (bt == 0):
        print('Error in read_vceil: Unable to read in base_time -- likely the GDL/int64 issue')
        return err

    yy = np.array([datetime.utcfromtimestamp(x).year for x in secs])
    mm = np.array([datetime.utcfromtimestamp(x).month for x in secs])
    dd = np.array([datetime.utcfromtimestamp(x).day for x in secs])
    ymd = yy*10000 + mm*100 + dd
    hour = np.array([((datetime.utcfromtimestamp(x)-datetime(yy[0],mm[0],dd[0])).total_seconds())/3600. for x in secs])

    return ({'success':1, 'secs':secs, 'ymd':ymd, 'hour':hour, 'cbh':cbh})

################################################################################
# This function puts the IRS data on the same temporal grid as the MWR data.
################################################################################

def grid_irs(ch1, irssum, avg_instant, hatchOpenSwitch, missingDataFlagSwitch,
              secs, tavg, irs_noise_inflation, vip, verbose):

    if verbose >= 3:
        print('    Temporally gridding the IRS data')
    if verbose >= 2:
        print(f'  In grid_irs: hatchOpenSwitch is {hatchOpenSwitch:d} and missingDataFlagSwitch is {missingDataFlagSwitch:d}')

    err = {'success':0}

    rrad = np.zeros((len(ch1['wnum']),len(secs)))
    nrad = np.zeros((len(irssum['wnum']),len(secs)))
    hatflag = np.zeros(len(secs))
    mssflag = np.zeros(len(secs))
    atmos_pres = np.zeros(len(secs))
    nsfc_temp  = np.zeros(len(secs))

    for i in range(len(secs)):
        # Get the channel 1 data on this grid

        if ((hatchOpenSwitch == 1) & (missingDataFlagSwitch == 0)):
            if ((i == 0) & (verbose >= 2)):
                print('      Only averaging IRS data where hatchOpen is 1 (missingDataFlag is anything)')
            if ((avg_instant == 0) | (avg_instant == -1)):
                foo = np.where((secs[i]-tavg*60./2. <= ch1['secs']) & (ch1['secs'] < secs[i]+tavg*60./2.) &
                               ((ch1['hatchopen'] >= 0.8) & (ch1['hatchopen'] < 1.2)))[0]
            else:
                dell = np.abs(secs[i] - ch1['secs'])
                foo = np.where((secs[i]-tavg*60./2. <= ch1['secs']) & (ch1['secs'] < secs[i]+tavg*60./2.) &
                               (dell == np.nanmin(dell)) & ((ch1['hatchopen'] >= 0.8) & (ch1['hatchopen'] < 1.2)))[0]

                if len(foo) > 1:
                    foo = np.array([foo[0]])
        elif ((hatchOpenSwitch == 0) & (missingDataFlagSwitch == 0)):
            if ((i == 0) & (verbose >= 2)):
                print('      Averaging all IRS data regardless of hatchOpen or missingDataFlag')
            if ((avg_instant == 0) | (avg_instant == -1)):
                foo = np.where((secs[i]-tavg*60./2. <= ch1['secs']) & (ch1['secs'] < secs[i]+tavg*60./2.))[0]
            else:
                dell = np.abs(secs[i] - ch1['secs'])
                foo = np.where((secs[i]-tavg*60./2. <= ch1['secs']) & (ch1['secs'] < secs[i]+tavg*60./2.) &
                               (dell == np.nanmin(dell)))[0]

                if len(foo) > 1:
                    foo = np.array([foo[0]])
        elif ((hatchOpenSwitch == 1) & (missingDataFlagSwitch == 1)):
            if ((i == 0) & (verbose >= 2)):
                print('      Only averaging IRS data where hatchOpen is 1 and missingDataFlag is not 1')
            if ((avg_instant == 0) | (avg_instant == -1)):
               foo = np.where((secs[i]-tavg*60./2. <= ch1['secs']) & (ch1['secs'] < secs[i]+tavg*60./2.) &
                               (ch1['missingDataFlag'] == 0) & ((ch1['hatchopen'] >= 0.8) & (ch1['hatchopen'] < 1.2)))[0]
            else:
               dell = np.abs(secs[i] - ch1['secs'])
               foo = np.where((secs[i]-tavg*60./2. <= ch1['secs']) & (ch1['secs'] < secs[i]+tavg*60./2.) &
                               (np.isclose(ch1['missingDataFlag'],0)) & (dell == np.nanmin(dell)) & ((ch1['hatchopen'] >= 0.8) & (ch1['hatchopen'] < 1.2)))[0]

               if len(foo) > 1:
                    foo = np.array([foo[0]])

        elif ((hatchOpenSwitch == 0) & (missingDataFlagSwitch == 1)):
            if ((i == 0) & (verbose >= 2)):
                print('      Averaging all IRS data where missingDataFlag is 0 (hatchOpen can be anything)')

            if ((avg_instant == 0) | (avg_instant == -1)):
               foo = np.where((secs[i]-tavg*60./2. <= ch1['secs']) & (ch1['secs'] < secs[i]+tavg*60./2.) &
                               (ch1['missingDataFlag'] == 0))[0]
            else:
               dell = np.abs(secs[i] - ch1['secs'])
               foo = np.where((secs[i]-tavg*60./2. <= ch1['secs']) & (ch1['secs'] < secs[i]+tavg*60./2.) &
                               (ch1['missingDataFlag'] == 0) & (dell == np.nanmin(dell)))[0]

               if len(foo) > 1:
                    foo = np.array([foo[0]])

        else:
            print('Error: This piece of code should never be executed -- logic trap in grid_irs()')
            return err

        if len(foo) == 0:
            rrad[:,i] = -9999.
            hatflag[i] = -9999
            mssflag[i] = 1    #The sample is missing
            atmos_pres[i] = -9999.
            nsfc_temp[i]  = -9999.
        else:
            if len(foo) == 1:
                rrad[:,i] = np.squeeze(ch1['rad'][:,foo])
                hatflag[i] = ch1['hatchopen'][foo]
                mssflag[i] = ch1['missingDataFlag'][foo]
                atmos_pres[i] = ch1['atmos_pres'][foo]
                nsfc_temp[i]  = ch1['nearSfcTb'][foo]
            else:
                rrad[:,i] = np.nansum(ch1['rad'][:,foo], axis = 1)/np.float(len(foo))
                mssflag[i] = np.nanmax(ch1['missingDataFlag'][foo])


                # Determine the appropriate value for the hatch, given there
                # are several IRS samples in this window.  If the hatch is open
                # for all of these samples, then call it open.  If the hatch is
                # closed for all of the samples, then call it closed.  Otherwise
                # call it "indeterminant"

                bar = np.where(ch1['hatchopen'][foo] == 1)[0]
                if len(bar) == len(foo):
                    hatflag[i] = 1                             # Hatch was always open
                else:
                    bar = np.where(ch1['hatchopen'][foo] == 0)[0]
                    if len(bar) == len(foo):
                      hatflag[i] = 0                           # Hatch was always closed
                    else:
                      hatflag[i] = 3                           # If we are here it is neither
            atmos_pres[i] = np.nanmean(ch1['atmos_pres'][foo])
            nsfc_temp[i]  = np.nanmean(ch1['nearSfcTb'][foo])

        # Get the summary data on this grid
        if ((avg_instant == 0) | (avg_instant == -1)):
            foo = np.where((secs[i]-tavg*60./2. <= irssum['secs']) & (irssum['secs'] < secs[i]+tavg*60./2.))[0]
        else:
            dell = np.abs(secs[i]-irssum['secs'])
            foo = np.where((secs[i]-tavg*60./2. <= irssum['secs']) & (irssum['secs'] < secs[i]+tavg*60./2.) &
                               (dell == np.nanmin(dell)))[0]

        if len(foo) == 0:
            nrad[:,i] = -9999.
        elif len(foo) == 1:
            nrad[:,i] = np.squeeze(irssum['noise'][:,foo])
        else:
            if(avg_instant ==0):
                if(verbose >= 2):
                    print('      Computing the average IRS instrument noise, dividing by sqrt(N)')
                nrad[:,i] = (np.nansum(irssum['noise'][:,foo],axis = 1)/np.float(len(foo))) / np.sqrt(len(foo))
            else:
                if(verbose >= 2):
                    print('      Computing the average IRS instrument noise, with NO division by sqrt(N)')
                nrad[:,i] = (np.nansum(irssum['noise'][:,foo],axis = 1)/np.float(len(foo)))

    # Inflate the IRS noise, as specified by the user
    if(verbose >= 2):
        print(f'    Inflating the IRS spectral noise by a factor of {irs_noise_inflation:.2f}')
    nrad = nrad * irs_noise_inflation

    # Put all IRS data on same spectral grid
    wnum = np.copy(ch1['wnum'])
    noise = rrad*0.
    for i in range(len(secs)):
        noise[:,i] = np.interp(wnum,irssum['wnum'],nrad[:,i])

    # Test to ensure that the noise is above the IRS noise floor
    if vip['irs_min_noise_flag'] != 0: 
        nmessage = 0

        # Convert the input noise floor data from strings to floating point arrays
        parts = vip['irs_min_noise_wnum'].split(',')
        fwnum = np.array(parts).astype(np.float)
        parts = vip['irs_min_noise_spec'].split(',')
        if len(parts) != len(fwnum):
            print('Error: The number of entered VIP.irs_min_noise_wnum does not match number of VIP.irs_min_noise_spec')
            return err
        else:
            fnoise = np.array(parts).astype(np.float)

        # Interpolate the input noise floor array to the current spectral grid (no extrapolation)
        floor = np.interp(wnum,fwnum,fnoise)

        for j in range(len(wnum)):
            foo = np.where(noise[j,:] < floor[j])[0]
            if len(foo) > 0:
                noise[j,foo] = floor[j]
                if((nmessage == 0) & (verbose >= 1)):
                    print('    Resetting some of IRS noise spectrum, which was below the noise floor')
                    nmessage = 1

    # Get the surface temperature from the IRS radiance observations
    # Use the actual IRS radiances, not the subset that was extracted
    bar = np.where((wnum >= 670) & (wnum <= 675))[0]
    if len(bar) >= 8:
        Tsfc = Calcs_Conversions.invplanck(np.nansum(wnum[bar])/np.float(len(bar)),np.nansum(rrad[bar,:],axis = 0)/np.float(len(bar)))
        Tsfc = Tsfc - 273.16

        bar = np.where(np.isnan(Tsfc))

        if len(bar) > 0:
            Tsfc[bar] = -999.
    else:
        Tsfc = np.ones(len(secs))*-999.

    yy = np.array([datetime.utcfromtimestamp(x).year for x in secs])
    mm = np.array([datetime.utcfromtimestamp(x).month for x in secs])
    dd = np.array([datetime.utcfromtimestamp(x).day for x in secs])
    ymd = yy*10000 + mm*100 + dd
    hour = np.array([((datetime.utcfromtimestamp(x)-datetime(yy[0],mm[0],dd[0])).total_seconds())/3600. for x in secs])

    return ({'success':1, 'secs':secs, 'ymd':ymd, 'hour':hour, 'yy':yy, 'mm':mm, 'dd':dd,
            'hatchopen':hatflag, 'avg_instant':avg_instant,
            'wnum':wnum, 'radmn':rrad, 'noise':noise, 'atmos_pres':atmos_pres, 'nearSfcTb':nsfc_temp, 
            'tsfc':Tsfc, 'fv':ch1['fv'], 'fa':ch1['fa'], 'missingDataFlag':mssflag})

################################################################################
# This function puts the MWR data onto a common temporal grid.
################################################################################

def grid_mwr(mwr, avg_instant, secs, tavg, time_delta, verbose):

    if verbose == 3:
        print('  Temporally gridding the MWR data')
    err = {'success':0}

    yy = np.array([datetime.utcfromtimestamp(x).year for x in secs])
    mm = np.array([datetime.utcfromtimestamp(x).month for x in secs])
    dd = np.array([datetime.utcfromtimestamp(x).day for x in secs])
    ymd = yy*10000 + mm*100 + dd
    hour = np.array([((datetime.utcfromtimestamp(x)-datetime(yy[0],mm[0],dd[0])).total_seconds())/3600. for x in secs])

    if mwr['type'] <= 0:
        missing = np.ones(len(secs))*-999.
        return {'success':1, 'secs':secs, 'ymd':ymd, 'hour':hour, 'n_fields':0,
                'type':mwr['type'], 'rootname':'none found', 'alt':-1}

    # If the time_delta [hours] is lower than tavg [min], then warn the user
    if time_delta/60. < tavg:
        print('    WARNING: mwr_time_delta is smaller than Tavg; consider resetting it in the VIP file to a larger value')

    # Now directly specifying the time for the full-width averaging window
    twin = time_delta*60*2  # Need twin to have units of minutes

    nsfc_temp = np.zeros(len(secs)) - 999.0
    if mwr['n_fields'] > 0:
        tbsky = np.ones((mwr['n_fields'],len(secs)))*-999.0

        # Grid the data
    if ((avg_instant == 0) | (avg_instant == -1)):
        # We are averaging the MWR data over the averaging interval

        for i in range(len(secs)):
            foo = np.where((mwr['secs'] >= secs[i]-twin*60./2.) & (mwr['secs'] < secs[i]+twin*60./2.))[0]
            if mwr['n_fields'] > 0:
                foo = np.where((mwr['secs'] >= secs[i]-twin*60./2.) & (mwr['secs'] < secs[i]+twin*60./2.))[0]

                if len(foo) > 0:
                    nsfc_temp[i] = np.nanmean(mwr['nearSfcTb'][foo])
                    for j in range(mwr['n_fields']):
                        tbsky[j,i] = np.nanmean(mwr['tbsky_corr'][j,foo])

    elif avg_instant == 1:
        # We are taking the closest point to the center of the averaging interval,
        # but the point taken must be within the averaging interval

        for i in range(len(secs)):
            foo = np.where((mwr['secs'] >= secs[i]-twin*60./2.) & (mwr['secs'] < secs[i]+twin*60./2.))[0]
            if len(foo) > 0:
                dell = np.abs(secs[i] - mwr['secs'][foo])
                bar = np.where(dell == np.nanmin(dell))[0][0]
                nsfc_temp[i] = mwr['nearSfcTb'][foo[bar]]

            if mwr['n_fields'] > 0:
                foo = np.where((mwr['secs'] >= secs[i]-twin*60./2.) & (mwr['secs'] < secs[i]+twin*60./2.))[0]

                if len(foo) > 0:
                    dell = np.abs(secs[i] - mwr['secs'][foo])
                    bar = np.where(dell == np.nanmin(dell))[0][0]
                    for j in range(mwr['n_fields']):
                        tbsky[j,i] = mwr['tbsky_corr'][j,foo[bar]]
    else:
        print('Error: the avg_instant flag has an unknown value in grid mwr()')
        return err

    # The structure being returned depends on the number of Tb fields desired
    if mwr['n_fields'] == 0:
        return ({'success':1, 'secs':secs, 'ymd':ymd, 'hour':hour,
                'n_fields':0, 'type': mwr['type'], 'rootname':mwr['rootname']})
    else:
        return ({'success':1, 'secs':secs, 'ymd':ymd, 'hour':hour,
                'n_fields':mwr['n_fields'], 'tbsky':tbsky, 'freq':mwr['freq'], 'nearSfcTb':nsfc_temp, 
                'noise':mwr['noise'], 'bias':mwr['bias'], 'type': mwr['type'], 'rootname':mwr['rootname']})

################################################################################
# This function puts the mwrscan data onto a common temporal grid.
################################################################################

def grid_mwrscan(mwrscan, secs, n_elevations, elevations, timewindow, verbose):

    if verbose == 3:
        print('  Temporally gridding the MWR-scan data')
    err = {'success':0}

        # An observation has to be within +/- this threshold to be considered at this angle
        # This is needed because there is a bit of a jitter in HATPRO elevation angles
    angle_threshold = 1.0       # degrees

    yy = np.array([datetime.utcfromtimestamp(x).year for x in secs])
    mm = np.array([datetime.utcfromtimestamp(x).month for x in secs])
    dd = np.array([datetime.utcfromtimestamp(x).day for x in secs])
    ymd = yy*10000 + mm*100 + dd
    hour = np.array([((datetime.utcfromtimestamp(x)-datetime(yy[0],mm[0],dd[0])).total_seconds())/3600. for x in secs])

    if mwrscan['type'] <= 0:
        missing = np.ones(len(secs))*-999.
        return ({'success':1, 'secs':secs, 'ymd':ymd, 'hour':hour, 'n_fields':0,
                'type':mwrscan['type'], 'rootname':'none found'})

    parts = elevations.split(',')

    if len(parts) != n_elevations:
        print('Error: Problem with elevations in grid_mwrscan (this should not happen)')
        return err
    else:
        delev = np.array(parts).astype(np.float)

    # Compute the time window to use in seconds, and it should be at least 2 minutes

    if ((timewindow*3600.) < (2.*60)):
        print('    Original time window for averaging mwr scan data was too small. Changing to 2 minutes.')
        twin = 2*60.
    else:
        twin = timewindow*3600.

    # Allocate the needed space

    tbsky = np.ones((n_elevations,mwrscan['n_fields'],len(secs)))*-999.

    # Regardless of the averaging type (averaging or taking the closest one)
    # I am going to take the closest one at each elevation scan (if it is within
    # the averaging window) and assign it to the sample time.

    for i in range(len(secs)):
        for j in range(n_elevations):
            foo = np.where((mwrscan['secs'] >= secs[i]-twin) & (mwrscan['secs'] < secs[i]+twin) &
                            (np.abs(mwrscan['elev'] - delev[j]) < angle_threshold))[0]

            if len(foo) > 0:
                dell = np.abs(secs[i] - mwrscan['secs'][foo])
                bar = np.where(dell == np.nanmin(dell))[0][0]
                tbsky[j,:,i] = mwrscan['tbsky_corr'][:,foo[bar]]
            else:
                tbsky[j,:,i] = -999.

    # Now I have the tbksy data in a 3-d structure, and I need to reshape it into
    # a 2-d structure (mixing frequency and elevation angle).

    ntbsky = np.ones((n_elevations*mwrscan['n_fields'],len(secs)))*-999.
    dim = np.ones(n_elevations*mwrscan['n_fields'])*-999.
    noise = np.ones(n_elevations*mwrscan['n_fields'])*-999.
    idx = np.arange(mwrscan['n_fields'])
    for i in range(n_elevations):
        ntbsky[i*mwrscan['n_fields']+idx,:] = tbsky[i,:,:]
        dim[i*mwrscan['n_fields']+idx] = mwrscan['freq']*1000. + delev[i]/1000.
        noise[i*mwrscan['n_fields']+idx] = mwrscan['noise']

    return ({'success': 1, 'secs': secs, 'ymd': ymd, 'hour': hour, 'n_elevations': n_elevations,
            'elevations': delev, 'n_fields': mwrscan['n_fields'], 'tbsky': ntbsky, 'freq': mwrscan['freq'],
            'noise': noise, 'bias': mwrscan['bias'], 'dim': dim, 'type': mwrscan['type'],
            'rootname': mwrscan['rootname'], 'timewindow': timewindow})

################################################################################
# This function reads in the thermodynamic profiles from the external source
# (sonde, lidar, etc)
################################################################################

def read_external_profile_data(date, ht, secs, tres, avg_instant,
            wv_prof_type, wv_prof_path, wv_noise_multiplier_hts, wv_noise_multiplier_val,
            wv_prof_minht, wv_prof_maxht, wv_time_delta, temp_prof_type, temp_prof_path,
            temp_noise_adder_hts, temp_noise_adder_val, temp_prof_minht, temp_prof_maxht,
            temp_time_delta, temp_ht_offset, wv_ht_offset, dostop, verbose):

    external = {'success':0, 'nTprof':-1, 'nQprof':-1, 'attrs': {}}

    # Dict to save attributes from the input data. Currently only used for generic_grid obs
    # TODO - Implement this for the other obs types
    saved_attributes = {}

    model_type = 'None'
    model_lat = -999.
    model_lon = -999.

    #### Read the external water vapor data first

    # No external WV source specified....

    qunit = ' '
    wvmultiplier = 1.0              # I will need this below for the DIAL data

    dates = [(datetime.strptime(str(date), '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d') ,
            str(date) ,  (datetime.strptime(str(date), '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d') ]

    if wv_prof_type == 0:
        qtype = 'None'
        external['nQprof'] = 0.0

    # Read in  the radiosondes as a constraint on the water vapor profile
    # over some range

    elif wv_prof_type == 1:
        if verbose >= 1:
            print('  Reading in radiosonde data to constrain the WV profile')

        files = []
        for i in range(len(dates)):
            tempfiles, status = findfile(wv_prof_path,'*sonde*' + dates[i] + '*.(cdf|nc)')
            if status == 1:
                return external
            files = files + tempfiles

        external['nQprof'] = 0
        if len(files) == 0:
            if verbose >= 1:
                print('      No radiosondes found in this directory for this date')
        else:
            maxht = int(wv_prof_maxht+0.1)
            if maxht < wv_prof_maxht:
                maxht += 1
            zzq = np.arange(maxht*100+1)*0.01  #Define a default 10-m grid for these sondes [km AGL]
            if verbose >= 1:
                print(f'      Attempting to read in {len(files):d} radiosonde files')

            for i in range(len(files)):
                fid = Dataset(files[i],'r')
                fid.set_auto_mask(False)
                bt = fid.variables['base_time'][0].astype('float')
                to = fid.variables['time_offset'][:].astype('float')
                p = fid.variables['pres'][:]
                t = fid.variables['tdry'][:]
                u = fid.variables['rh'][:]
                z = fid.variables['alt'][:]

                fid.close()
                z = (z-z[0])/1000.
                foo = np.where((p > 0) & (p < 1050) & (t > -150) & (t < 60) & (u >= 0) & (u < 103) & (z >= 0))[0]

                if len(foo) < 2:
                    if verbose >= 1:
                        print(f'        Radiosonde {files[i]:s} did not pass the QC for WVMR')
                    continue
                z = z[foo]
                p = p[foo]
                t = t[foo]
                u = u[foo]

                # Make sure sonde is monotonically increasing, not a simple sort
                # we will remove heights that decrease since they are most likely
                # bad data
                foo = Other_functions.make_monotonic(z)
                if len(foo) < 2:
                    continue

                z = z[foo]
                p = p[foo]
                t = t[foo]
                u = u[foo]

                if np.nanmax(z) < maxht:
                    if verbose >= 1:
                        print(f'        Sonde {files[i]:s} did not go above the required height ({np.nanmax(z):.3f} < {maxht:.3f}) for WVMR')
                    continue            # The sonde must be above this altitude to be used here

                # Compute WVMR and its uncertainty (using RMS and typical sonde uncertainties
                # of 3% in Rh and 0.5 degC in temperature)

                w = Calcs_Conversions.rh2w(t, u/100., p)
                w1 = Calcs_Conversions.rh2w(t, u/100.-0.03, p)
                w2 = Calcs_Conversions.rh2w(t+0.5, u/100., p)
                we = np.sqrt((w - w1)**2. + (w-w2)**2)            # Sum of squared errors

                qunit = 'g/kg'
                qtype = 'radiosonde'

                # Append the data to the growing structure

                if external['nQprof'] <= 0:
                    qsecs = np.array([bt + to[0]])
                    wv = np.array([np.interp(zzq,z,w,left=-999,right=-999)])
                    swv = np.array([np.interp(zzq,z,we,left=-999,right=-999)])
                else:
                    qsecs = np.append(qsecs,bt+to[0])
                    wv = np.vstack((wv, np.interp(zzq,z,w,left=-999,right=-999)))
                    swv = np.vstack((swv, np.interp(zzq,z,we,left=-999,right=-999)))
                external['nQprof'] += 1

            if external['nQprof'] > 0:
                wv = wv.T
                swv = swv.T

    # Read in the ARM Raman lidar data (rlprofmr)

    elif wv_prof_type == 2:
        if verbose >= 1:
            print('    Reading in ARM Raman lidar (rlprofmr) data to constrain the WV profile')

        qunit = 'g/kg'
        files = []
        for i in range(len(dates)):
            tempfiles, status = (findfile(wv_prof_path,'*rlprofmr*' + dates[i] + '*.(cdf|nc)'))
            if status == 1:
                return external
            files = files + tempfiles
        if len(files) == 0:
            if verbose >= 1:
                print('      No ARM RLID WV found in this directory for this date')
            external['nQprof'] = 0.
        else:
            if verbose >= 2:
                print('      Reading ' + str(len(files)) + ' ARM RLID WV data files')
            nprof = 0.
            for i in range(len(files)):
                fid = Dataset(files[i],'r')
                fid.set_auto_mask(False)
                bt = fid.variables['base_time'][0].astype('float')
                to = fid.variables['time_offset'][:].astype('float')

                # There are two types of RLPROF_MR data.
                # This handles the rlprofmr1turn dataset.

                if len(np.where(np.array(list(fid.variables.keys())) == 'mixing_ratio_3')[0]) > 0:
                    qtype = 'ARM Raman lidar (rlprofmr1turn)'
                    htx = fid.variables['height'][:]
                    wvx = fid.variables['mixing_ratio_3'][:]
                    swvx = fid.variables['mixing_ratio_3_error'][:]
                    qcflag = fid.variables['qc_flag'][:]
                    fid.close()

                    # Only keep data above 70 m to ensure the 60-m tower isn't included

                    foo = np.where(htx > 0.070)[0]
                    if len(foo) == 0:
                        print('Error: This should not happen when processing the RLID WV data')
                        return external
                    htx = htx[foo]
                    wvx = wvx[:,foo]
                    swvx = swvx[:,foo]

                    # And only keep samples where the qc_flag is 0 (i.e. good quality)
                    foo = np.where(qcflag == 0)[0]
                    if len(foo) == 0:
                        print('      Warning: no good samples found for the RLID WV on this day')
                        continue
                    else:
                        to = to[foo]
                        wvx = wvx[foo,:]
                        swvx = swvx[foo,:]

                else:
                    # This handles the rlprofmr1news dataset
                    qtype = 'ARM Raman lidar (rlprofmr1news)'
                    htx = fid.variables['height'][:]
                    wvx = fid.variables['mr_merged'][:]
                    swvx = fid.variables['mr_merged_err'][:]
                    fid.close()

                    #Only keep data above 70 m to ensure the 60-m tower isn't included
                    foo = np.where(htx > 0.070)[0]
                    if len(foo) == 0:
                        print('Error: This should not happen when processing the RLID WV data')
                        return external
                    htx = htx[foo]
                    wvx = wvx[:,foo]
                    swvx = swvx[:,foo]

                # Now append the data to the growing structure
                if nprof == 0:
                    qsecs = bt+to
                    wv = np.copy(wvx)
                    swv = np.copy(swvx)
                else:
                    qsecs = np.append(qsecs, bt+to)
                    wv = np.append(wv,wvx, axis = 0)
                    swv = np.append(swv,swvx, axis = 0)
                nprof = len(qsecs)

            external['nQprof'] = len(qsecs)

        if external['nQprof'] > 0:
            zzq = np.copy(htx)
            wvmultiplier = 1.      # To scale the WV profiles to be reasonable order of magnitude
            wv = wv/wvmultiplier
            swv = swv/wvmultiplier
            wv = wv.T
            swv = swv.T

    # Read in the NCAR WV DIAL data (for 2014-2017 time period)

    elif wv_prof_type == 3:
        if verbose >= 1:
            print('  Reading in NCAR WV DIAL data to constrain the WV profile')

        files = []
        for i in range(len(dates)):
            tempfiles, status = (findfile(wv_prof_path,'*' + str(int(dates[i])%1000000) + '*.(cdf|nc)'))
            if status == 1:
                return external
            files = files + tempfiles

        if len(files) == 0:
            if verbose >= 1:
                print('    No NCAR WV DIAL found in this directory for this date')
            external['nQprof'] = 0.0
        else:
            if verbose >= 2:
                print('    Reading ' + str(len(files)) + ' NCAR WV DIAL data files')
            for i in range(len(files)):
                fid = Dataset(files[i],'r')
                fid.set_auto_mask(False)
                secsx = fid.variables['time_unix'][:]
                zzq = fid.variables['range'][:]
                wvx = fid.variables['N_avg'][:]
                swvx = fid.variables['N_error'][:]
                fid.close()

                if i == 0:
                    qsecs = np.copy(secsx)
                    wv = np.copy(wvx)
                    swv = np.copy(swvx)
                else:
                    qsecs = np.append(qsecs,secsx)
                    wv = np.append(wv, wvx, axis = 0)
                    swv = np.append(swv, swvx, axis = 0)

                external['nQprof'] = len(qsecs)
            zzq = zzq / 1000.           # Convert m AGL to km AGL

        if external['nQprof'] > 0:
            wvmultiplier = 1e16           # To scale the WV profiles to be reasonable order of magnitude
            wv = wv/wvmultiplier
            swv = swv/wvmultiplier
            qunit = 'molecules/cm3 (sccaled by 1e16)'
            qtype = 'NCAR water vapor DIAL'

            wv = wv.T
            swv = swv.T

    # Read in the numerical weather model soundings (Greg Blumberg's format)

    elif wv_prof_type == 4:
        if verbose >= 1:
            print('  Reading in NWP model output to constrain the WV profile')

        files = []
        for i in range(len(dates)):
            tempfiles, status = (findfile(wv_prof_path,'*model*' + dates[i] + '*.(cdf|nc)'))
            if status == 1:
                return external
            files = files + tempfiles
        if len(files) == 0:
            if verbose >= 1:
                print('    No NWP model output data ound in this directory for this date')
            external['nQprof'] = 0
        else:
            if verbose >= 2:
                print('    Reading ' + str(len(files))  + ' NWP output WV files')
            for i in range(len(files)):
                fid = Dataset(files[i], 'r')
                fid.set_auto_mask(False)
                bt = fid.variables['base_time'][0].astype('float')
                to = fid.variables['time_offset'][:].astype('float')
                zzq = fid.variables['height'][:]
                wwx = fid.variables['waterVapor'][:]
                ssx = fid.variables['sigma_waterVapor'][:]
                if len(np.where(np.array(list(fid.ncattrs())) == 'model')[0]) > 0:
                    model_type = fid.model
                else:
                    model_type = "unknown"
                    print("    Warning: The mod dataset does not have the global attribute 'model' defined")
                if len(np.where(np.array(list(fid.ncattrs())) == 'gridpoint_lat')[0]) > 0:
                    model_lat = np.float(fid.gridpoint_lat)
                else:
                    model_lat = -999.
                if len(np.where(np.array(list(fid.ncattrs())) == 'gridpoint_lon')[0]) > 0:
                    model_lon = np.float(fid.gridpoint_lon)
                else:
                    model_lon = -999.
                fid.close()
                    # Sometimes, the model data are not ascending with height, so sort them
                sidx = np.argsort(zzq)
                zzq = zzq[sidx]
                wwx = wwx[:,sidx]
                ssx = ssx[:,sidx]
                    # Now append the data to a growing structure
                if i == 0:
                    qsecs = bt+to
                    wv = np.copy(wwx)
                    swv = np.copy(ssx)
                else:
                    qsecs = np.append(qsecs,bt+to)
                    wv = np.vstack((wv, wwx))
                    swv = np.vstack((swv,ssx))
            wv = wv.T
            swv = swv.T
            external['nQprof'] = len(qsecs)

            # Perform a quick check to make sure all profiles exist. I am
            # assuming that (since this is a model) I only have to look at the
            # first level. If there are no valid profiles (ie, all of them are
            # NaN, then reset the counter to zero

            foo = np.where(((~np.isnan(wv[0,:])) & (np.abs(wv[0,:] < 500))))[0]
            if len(foo) > 0:
                wv = wv[:,foo]
                swv = swv[:,foo]
                qsecs = qsecs[foo]
                external['nQprof'] = len(qsecs)
            else:
                external['nQprof'] = 0

        qunit = 'g/kg'
        qtype = 'NWP model output from ' + model_type + ' at ' + str(model_lat) + ' degN, ' + str(model_lon) + ' degE'

    # Read in the Vaisala water vapor DIAL profiles
    elif wv_prof_type == 5:
        if verbose >= 1:
            print('  Reading in Vaisala WV DIAL to constrain the WV profile')

        files = []
        for i in range(len(dates)):
            tempfiles, status = (findfile(wv_prof_path,'*dial*' + dates[i] + '*.(cdf|nc)'))
            if status == 1:
                return external
            files = files + tempfiles

        if len(files) == 0:
            if verbose >= 1:
                print('    No Vaisala WV DIAL data found in this directory for this date')
            external['nQprof'] = 0
        else:
            if verbose >= 2:
                print('    Reading ' + str(len(files)) + ' Vaisala WV DIAL files')

            for i in range(len(files)):
                fid = Dataset(files[i], 'r')
                fid.set_auto_mask(False)
                bt = fid.variables['base_time'][0].astype('float')
                to = fid.variables['time_offset'][:].astype('float')
                zzq = fid.variables['height'][:]
                wwx = fid.variables['waterVapor'][:]
                ssx = fid.variables['sigma_waterVapor'][:]
                mhtx = fid.variables['maxht'][:]
                fid.close()

                if i == 0:
                    qsecs = bt+to
                    wv = np.copy(wwx)
                    swv = np.copy(ssx)
                    maxht = np.copy(mhtx)
                else:
                    qsecs = np.append(qsecs,bt+to)
                    wv = np.append(wv,wwx, axis = 0)
                    swv = np.append(swv,ssx, axis = 0)

            external['nQprof'] = len(qsecs)

            wv = wv.T
            swv = swv.T

            # Set all of the points above the maximum height to -999.

            for i in range(len(qsecs)):
                foo = np.where(zzq >= maxht[i])[0]
                if len(foo) > 0:
                    wv[foo,i] = -999.
                    swv[foo,i] = -999.

        qunit = 'g/kg'
        qtype = 'Vaisala WV DIAL data'

    # Read in the NCAR water vapor DIAL profiles (from 2019 and beyond)

    elif wv_prof_type == 6:
        if verbose >= 1:
            print('  Reading in NCAR MPD data to constrain the WV profile')

        files = []
        for i in range(len(dates)):
            tempfiles, status = (findfile(wv_prof_path,'*mpd*' + str(int(dates[i])) + '*.(cdf|nc)'))
            if status == 1:
                return external
            files = files + tempfiles

        if len(files) == 0:
            if verbose >= 1:
                print('    No NCAR MPD data found in this directory for this date')
            external['nQprof'] = 0.0
        else:
            if verbose >= 2:
                print('    Reading ' + str(len(files)) + ' NCAR MPD data files')
            for i in range(len(files)):
                fid = Dataset(files[i],'r')
                fid.set_auto_mask(False)
                bt = (datetime.strptime(files[i][-18:-10],'%Y%m%d') - datetime(1970,1,1)).total_seconds()
                to = fid.variables['time'][:].astype('float')
                zzq = fid.variables['range'][:]
                wwx = fid.variables['Absolute_Humidity'][:]
                ssx = np.sqrt(fid.variables['Absolute_Humidity_variance'][:])
                maskx = fid.variables['Absolute_Humidity_mask'][:]
                fid.close()

                if i == 0:
                    qsecs = bt+to
                    wv = np.copy(wwx)
                    swv = np.copy(ssx)
                    mask = np.copy(maskx)
                else:
                    qsecs = np.append(qsecs,bt+to)
                    wv = np.append(wv,wwx, axis = 0)
                    swv = np.append(swv,ssx, axis = 0)
                    maxht = np.append(mask,maskx,axis=0)

                external['nQprof'] = len(qsecs)

            zzq = zzq / 1000.           # Convert m AGL to km AGL

            wv = wv.T
            swv = swv.T
            mask = mask.T

            # Apply the mask to the data.
            foo = np.where(mask == 1)
            wv[foo] = np.nan
            swv[foo] = np.nan

        if external['nQprof'] > 0:
            qunit = 'g/m3'
            qtype = 'NCAR MPD data'


    # Read in a generic observation grid
    elif wv_prof_type == 7:
        if verbose >= 1:
            print('  Reading in genric observation grid to constrain the WV profile')

        files = []
        for i in range(len(dates)):
            tempfiles, status = (findfile(wv_prof_path,'*grid*' + dates[i] + '*.(cdf|nc)'))
            if status == 1:
                return external
            files = files + tempfiles
        if len(files) == 0:
            if verbose >= 1:
                print('    No gridded data found in this directory for this date')
            external['nQprof'] = 0
        else:
            if verbose >= 2:
                print('    Reading ' + str(len(files))  + ' gridded WV files')
            for i in range(len(files)):
                fid = Dataset(files[i], 'r')
                fid.set_auto_mask(False)
                bt = fid.variables['base_time'][0].astype('float')
                to = fid.variables['time_offset'][:].astype('float')
                zzq = fid.variables['height'][:]
                wwx = fid.variables['waterVapor'][:]
                ssx = fid.variables['sigma_waterVapor'][:]

                # Keep the attributes from this so we can write them to the output netCDF
                for key in fid.ncattrs():
                    saved_attributes[f'extprof_{key}']= fid.getncattr(key)

                fid.close()

                if i == 0:
                    qsecs = bt+to
                    wv = np.copy(wwx)
                    swv = np.copy(ssx)
                else:
                    qsecs = np.append(qsecs,bt+to)
                    wv = np.vstack((wv, wwx))
                    swv = np.vstack((swv,ssx))

            wv = wv.T
            swv = swv.T
            external['nQprof'] = len(qsecs)

            qunit = 'g/kg'
            qtype = 'Generic observation'



    elif wv_prof_type == 99:
        if verbose >= 1:
            print('  Reading in RHUBC-2 AER GVRP-retrieval radiosonde data to constrain the WV profile')

        files = []
        for i in range(len(dates)):
            tempfiles, status = (findfile(wv_prof_path,'*sonde*' + dates[i] + '*.(cdf|nc)'))
            if status == 1:
                return external
            files = files + tempfiles

        if len(files) == 0:
            if verbose >=1:
                print('    No AER GVRP radiosondes found in this directory for this date')
            external['nQprof'] = 0
        else:
            maxht = int(wv_prof_maxht + 0.1)
            if maxht < wv_prof_maxht:
                maxht = maxht + 1
            zzq = np.arange(maxht*100+1)*0.01  #Define a default 10-m grid for these sondes [km AGL]

            external['nQprof'] = 0
            for i in range(len(files)):
                fid = Dataset(files[i], 'r')
                fid.set_auto_mask(False)
                bt = fid.variables['base_time'][0].astype('float')
                to = fid.variables['time_offset'][:].astype('float')
                p = fid.variables['pres'][:]
                t = fid.variables['tdry'][:]
                u = fid.variables['rh'][:]
                z = fid.variables['alt'][:]
                fid.close()

                z = (z-z[0])/1000.

                foo = np.where((p > 0) & (p < 1050) & (t > -150) & (t < 60) & (u >= 0) & (u < 103) & (z >= 0))[0]

                if len(foo) < 2:
                    continue
                z = z[foo]
                p = p[foo]
                t = t[foo]
                u = u[foo]

                # Make sure sonde is monotonically increasing, not a simple sort
                # we will remove heights that decrease since they are most likely
                # bad data
                foo = Other_functions.make_monotonic(z)
                if len(foo) < 2:
                    continue

                z = z[foo]
                p = p[foo]
                t = t[foo]
                u = u[foo]

                if np.nanmax(z) < maxht:
                    continue # The sonde must be above this altitude to be used here

                w = Calcs_Conversions.rh2w(t, u/100., p)
                we = np.ones(len(w))*0.00001             # Assuming no error in GVRP retrieval

                qunit = 'g/kg'
                qtype = 'AER profile for RHUBC-2'

                # Append the data to the growing structure

                if external['nQprof'] <= 0:
                    qsecs = np.array([bt+to[0]])
                    wv = np.array([np.interp(zzq,z,w,left=-999,right=-999)])
                    swv = np.array([np.interp(zzq,z,we,left=-999,right=-999)])
                else:
                    qsecs = np.append(qsecs,bt+to[0])
                    wv = np.vstack((wv, np.interp(zzq,z,w,left=-999,right=-999)))
                    swv = np.vstack((swv, np.interp(zzq,z,we,left=-999,right=-999)))

                external['nQprof'] = len(qsecs)

            if external['nQprof'] > 0:
                wv = wv.T
                swv = swv.T

    # An undefined external WV data source was specified...
    else:
        print('Error in read_external_profile_data: Undefined external water vapor profile source specified')
        return external

    # Read the external temperature data next
    # No external temperature profile source specified.....
    tunit = ' '
    if temp_prof_type == 0:
        ttype = 'none'
        external['nTprof'] = 0

    # Read in the radiosondes as a constraint on the temperature profile over some range

    elif temp_prof_type == 1:
        if verbose >= 1:
            print('  Reading in radiosonde data to constrain the temp profile')

        files = []
        for i in range(len(dates)):
            tempfiles, status = (findfile(temp_prof_path,'*sonde*' + dates[i] + '*.(cdf|nc)'))
            if status == 1:
                return external
            files = files + tempfiles

        external['nTprof'] = 0.
        if len(files) == 0:
            if verbose >= 1:
                print('    No radiosondes found in this directory for this date')

        else:
            if verbose >= 2:
                print('    Reading ' + str(len(files)) + ' radiosonde files')
            maxht = int(temp_prof_maxht+0.1)
            if (maxht < temp_prof_maxht):
                maxht += 1
            zzt = np.arange(maxht*100+1)*0.01  #Define a default 10-m grid for these sondes [km AGL]

            if verbose >= 1:
                print(f'      Attempting to read in {len(files):d} radiosonde files')
            external['nTprof'] = 0
            for i in range(len(files)):
                fid = Dataset(files[i], 'r')
                fid.set_auto_mask(False)
                bt = fid.variables['base_time'][0].astype('float')
                to = fid.variables['time_offset'][:].astype('float')
                p = fid.variables['pres'][:]
                t = fid.variables['tdry'][:]
                u = fid.variables['rh'][:]
                z = fid.variables['alt'][:]
                fid.close()

                z = (z-z[0])/1000.

                foo = np.where((p > 0) & (p < 1050) & (t > -150) & (t < 60) & (u >= 0) & (u < 103) & (z >= 0))[0]

                if len(foo) < 2:
                    if verbose >= 1:
                        print(f'        Radiosonde {files[i]:s} did not pass the QC for temp')
                    continue
                z = z[foo]
                p = p[foo]
                t = t[foo]
                u = u[foo]

                # Make sure sonde is monotonically increasing, not a simple sort
                # we will remove heights that decrease since they are most likely
                # bad data
                foo = Other_functions.make_monotonic(z)
                if len(foo) < 2:
                    continue

                z = z[foo]
                p = p[foo]
                t = t[foo]
                u = u[foo]

                if np.nanmax(z) < maxht:
                    if verbose >= 1:
                        print(f'        Sonde {files[i]:s} did not go above the required height ({np.nanmax(z):.3f} < {maxht:.3f}) for temp')
                    continue # The sonde must be above this altitude to be used here

                sigma_t = 0.5     # The assumed uncertainty of the radiosonde temperature measurement

                tunit = 'degC'
                ttype = 'radiosonde'

                # Append the data to the growing structure
                if external['nTprof'] <= 0:
                    tsecs = np.array([bt+to[0]])
                    temp = np.array([np.interp(zzt,z,t,left=-999,right=-999)])
                    stemp = np.array([np.ones(len(zzt))*sigma_t])
                else:
                    tsecs = np.append(tsecs,bt+to[0])
                    temp = np.vstack((temp, np.interp(zzt,z,t,left=-999,right=-999)))
                    stemp = np.vstack((stemp, np.ones(len(zzt))*sigma_t))
                external['nTprof'] += 1

            if external['nTprof'] > 0:
                temp = temp.T
                stemp = stemp.T
    # Read in the ARM Raman lidar data (rlproftemp)

    elif temp_prof_type == 2:
        if verbose >= 1:
            print('  Reading in ARM Raman lidar (rlproftemp) data to constrain the temp profile')

        ttype = 'ARM Raman lidar (rlproftemp)'
        tunit = 'degC'

        files = []
        for i in range(len(dates)):
            tempfiles, status = (findfile(temp_prof_path,'*rlproftemp*' + dates[i] + '*.(cdf|nc)'))
            if status == 1:
                return external
            files = files + tempfiles

        if len(files) == 0:
            if verbose >= 1:
                print('    No ARM RLID TEMP found in this directory for this date')
            external['nTprof'] = 0
        else:
            if verbose >= 2:
                print('    Reading ' + str(len(files)) + ' ARM RLID TEMP data files')

            nprof = 0
            for i in range(len(files)):
                fid = Dataset(files[i], 'r')
                fid.set_auto_mask(False)
                bt = fid.variables['base_time'][0].astype('float')
                to = fid.variables['time_offset'][:].astype('float')

                #There are two types of RLPROF_TEMP data.
                #This handles the rlprofmr1news dataset

                if len(np.where(np.array(list(fid.variables.keys())) == 'rot_raman_temperature')[0]) > 0:
                    ttype = 'ARM Raman lidar (rlprofmr1news)'
                    htx = fid.variables['height'][:]
                    tempx = fid.variables['rot_raman_temperature'][:]
                    stempx = fid.variables['rot_raman_temperature_error'][:]

                    # Only keep data above 70 m to ensure the 60-m tower isn't included

                    foo = np.where(htx > 0.070)[0]
                    if len(foo) == 0:
                        print('Error: This should not happen when processing the RLID WV data')
                        return external
                    htx = htx[foo]
                    tempx = tempx[:,foo]
                    stempx = stempx[:,foo]

                else:

                    # This handles the rlprofmr2news dataset

                    ttype = 'ARM Raman lidar (rlprofmr2news)'
                    htx   = fid.variables['height'][:]
                    tempx = fid.variables['temperature'][:]
                    stempx = fid.variables['temperature_error'][:]

                    fid.close()

                    # Only keep data above 70 m to ensure the 60-m tower isn't included

                    foo = np.where(htx > 0.070)[0]
                    if len(foo) == 0:
                        print('Error: This should not happen when processing the RLID WV data')
                        return external
                    htx = htx[foo]
                    tempx = tempx[:,foo]
                    stempx = stempx[:,foo]


                # Now append the data to the growing structure

                if nprof == 0:
                    tsecs = bt+to
                    temp  = np.copy(tempx)
                    stemp = np.copy(stempx)
                else:
                    tsecs = np.append(tsecs, bt+to)
                    temp  = np.append(temp,tempx, axis = 0)
                    stemp = np.append(stemp,stempx, axis = 0)
                nprof = len(tsecs)

            temp = temp - 273.16
            external['nTprof'] = len(tsecs)
            zzt = np.copy(htx)
            temp = temp.T
            stemp = stemp.T

    # Read in the numerical weather model soundings (Greg Blumberg's format)
    elif temp_prof_type == 4:
        if verbose >= 1:
            print('  Reading in NWP model output to constrain the temperature profile')

        files = []
        for i in range(len(dates)):
            tempfiles, status = (findfile(temp_prof_path,'*model*' + dates[i] + '*.(cdf|nc)'))
            if status == 1:
                return external
            files = files + tempfiles

        if len(files) == 0:
            if verbose >= 1:
                print('    No NWP model output data found in this directory for this date')
            external['nTprof'] = 0
        else:
            if verbose >= 2:
                print('    Reading ' + str(len(files)) + ' NWP output temp files')
            for i in range(len(files)):
                fid = Dataset(files[i], 'r')
                fid.set_auto_mask(False)
                bt = fid.variables['base_time'][0].astype('float')
                to = fid.variables['time_offset'][:].astype('float')
                zzt = fid.variables['height'][:]
                ttx = fid.variables['temperature'][:]
                ssx = fid.variables['sigma_temperature'][:]
                if len(np.where(np.array(list(fid.ncattrs())) == 'model')[0]) > 0:
                    model_type = fid.model
                else:
                    model_type = "unknown"
                    print("    Warning: The mod dataset does not have the global attribute 'model' defined")
                if len(np.where(np.array(list(fid.ncattrs())) == 'gridpoint_lat')[0]) > 0:
                    model_lat = np.float(fid.gridpoint_lat)
                else:
                    model_lat = -999.
                if len(np.where(np.array(list(fid.ncattrs())) == 'gridpoint_lon')[0]) > 0:
                    model_lon = np.float(fid.gridpoint_lon)
                else:
                    model_lon = -999.
                fid.close()
                    # Sometimes, the model data are not ascending with height, so sort them
                sidx = np.argsort(zzt)
                zzt = zzt[sidx]
                ttx = ttx[:,sidx]
                ssx = ssx[:,sidx]
                    # Now append the data to a growing structure
                if i == 0:
                    tsecs = bt+to
                    temp = np.copy(ttx)
                    stemp = np.copy(ssx)
                else:
                    tsecs = np.append(tsecs,bt+to)
                    temp = np.vstack((temp, ttx))
                    stemp = np.vstack((stemp,ssx))

            temp = temp.T
            stemp = stemp.T
            external['nTprof'] = len(tsecs)

            # Perform a quick check to make sure all profiles exist. I am
            # assuming that (since this is a model) I only have to look at the
            # first level. If there are no valid profiles (ie, all of them are
            # NaN, then reset the counter to zero

            foo = np.where(((~np.isnan(temp[0,:])) & (np.abs(temp[0,:] < 500))))[0]
            if len(foo) > 0:
                temp = temp[:,foo]
                stemp = stemp[:,foo]
                tsecs = tsecs[foo]
                external['nTprof'] = len(tsecs)
            else:
                external['nQprof'] = 0

        tunit = 'degC'
        ttype = 'NWP model output from ' + model_type + ' at ' + str(model_lat) + ' degN, ' + str(model_lon) + ' degE'

            # Read in the RASS virtual temperature data
    elif temp_prof_type == 5:
        if verbose >= 1:
            print('  Reading in RASS virtual temperature data')

        ttype = 'RASS Tv data'
        tunit = 'degC'

        files = []
        for i in range(len(dates)):
            tempfiles, status = (findfile(temp_prof_path,'*rass*' + dates[i] + '*.(cdf|nc)'))
            if status == 1:
                return external
            files = files + tempfiles

        if len(files) == 0:
            if verbose >= 1:
                print('    No RASS data found in this directory for this date')
            external['nTprof'] = 0
        else:
            if verbose >= 2:
                print('    Reading ' + str(len(files)) + ' RASS data files')

            nprof = 0
            for i in range(len(files)):
                fid = Dataset(files[i], 'r')
                fid.set_auto_mask(False)
                bt = fid.variables['base_time'][0].astype('float')
                to = fid.variables['time_offset'][:].astype('float')
                htx    = fid.variables['height'][:]
                tempx  = fid.variables['temp'][:]
                stempx = fid.variables['sigma_temp'][:]

                # Now append the data to the growing structure
                if nprof == 0:
                    tsecs = bt+to
                    temp  = np.copy(tempx)
                    stemp = np.copy(stempx)
                else:
                    tsecs = np.append(tsecs, bt+to)
                    temp  = np.append(temp,tempx, axis = 0)
                    stemp = np.append(stemp,stempx, axis = 0)
                nprof = len(tsecs)

                # There can be missgin data in the middle of the RASS profile,
                # and I don't want to interpolate over these gaps (or above
                # or below  the good data).  I will replace all of the bad
                # data with NaNs, and then trap these after the vertical
                # interpolation below.
            foo = np.where((temp < -100) | (temp > 100))
            if(len(foo) > 0):
                temp[foo] = np.nan

            external['nTprof'] = len(tsecs)
            zzt = np.copy(htx)
            temp = temp.T
            stemp = stemp.T


    elif temp_prof_type == 7:
        if verbose >= 1:
            print('  Reading in genric observation grid to constrain the temperature profile')

        files = []
        for i in range(len(dates)):
            tempfiles, status = (findfile(temp_prof_path,'*grid*' + dates[i] + '*.(cdf|nc)'))
            if status == 1:
                return external
            files = files + tempfiles

        if len(files) == 0:
            if verbose >= 1:
                print('    No gridded data found in this directory for this date')
            external['nTprof'] = 0
        else:
            if verbose >= 2:
                print('    Reading ' + str(len(files)) + ' gridded temp files')
            for i in range(len(files)):
                fid = Dataset(files[i], 'r')
                fid.set_auto_mask(False)
                bt = fid.variables['base_time'][0].astype('float')
                to = fid.variables['time_offset'][:].astype('float')
                zzt = fid.variables['height'][:]
                ttx = fid.variables['tdry'][:]
                ssx = fid.variables['sigma_tdry'][:]

                # Keep the attributes from this so we can write them to the output netCDF
                for key in fid.ncattrs():
                    saved_attributes[f'extprof_{key}'] = fid.getncattr(key)

                fid.close()

                if i == 0:
                    tsecs = bt+to
                    temp = np.copy(ttx)
                    stemp = np.copy(ssx)
                else:
                    tsecs = np.append(tsecs,bt+to)
                    temp = np.vstack((temp, ttx))
                    stemp = np.vstack((stemp,ssx))

            temp = temp.T
            stemp = stemp.T
            external['nTprof'] = len(tsecs)

            # Perform a quick chec

            tunit = 'degC'
            ttype = 'Generic observation output'


    # Read in the RHUBC-2 radiosonde data from AER's files
    elif temp_prof_type == 99:
        if verbose >= 1:
            print('  Reading in RHUBC-2 AER radiosonde data to constrain the temperature profile')

        files = []
        for i in range(len(dates)):
            tempfiles, status = (findfile(temp_prof_path,'*sonde*' + dates[i] + '*.(cdf|nc)'))
            if status == 1:
                return external
            files = files + tempfiles

        if len(files) == 0:
            if verbose >= 1:
                print('    No radiosondes found in this directory for this date')
            external['nTprof'] = 0
        else:
            maxht = int(temp_prof_maxht + 0.1)
            if maxht < temp_prof_maxht:
                maxht = maxht + 1
            zzt = np.arange(maxht*100+1)*0.01  #Define a default 10-m grid for these sondes [km AGL]

            external['nTprof'] = 0
            for i in range(len(files)):
                fid = Dataset(files[i], 'r')
                fid.set_auto_mask(False)
                bt = fid.variables['base_time'][0].astype('float')
                to = fid.variables['time_offset'][:].astype('float')
                p = fid.variables['pres'][:]
                t = fid.variables['tdry'][:]
                u = fid.variables['rh'][:]
                z = fid.variables['alt'][:]
                fid.close()

                z = (z-z[0])/1000.

                foo = np.where((p > 0) & (p < 1050) & (t > -150) & (t < 60) & (u >= 0) & (u < 103) & (z >= 0))[0]

                if len(foo) < 2:
                    continue
                z = z[foo]
                p = p[foo]
                t = t[foo]
                u = u[foo]

                # Make sure sonde is monotonically increasing, not a simple sort
                # we will remove heights that decrease since they are most likely
                # bad data
                foo = Other_functions.make_monotonic(z)
                if len(foo) < 2:
                    continue

                z = z[foo]
                p = p[foo]
                t = t[foo]
                u = u[foo]

                if np.nanmax(z) < maxht:
                    continue # The sonde must be above this altitude to be used here

                # The assumed uncertainty of the radiosonde temperature measurement that
                # Dave hard coded

                sigma_t = np.interp(z,[0,2.5,3.5,20],[1.5,1.5,0.5,0.5])
                tunit = 'degC'
                ttype = 'AER profile for RHUBC-2'

                # Append the data to the growing structure

                if external['nTprof'] <= 0:
                    tsecs = np.array([bt+to[0]])
                    temp = np.array([np.interp(zzt,z,t,left=-999,right=-999)])
                    stemp = np.array([np.interp(zzt,z,sigma_t,left=-999,right=-999)])
                else:
                    tsecs = np.append(tsecs,bt+to[0])
                    temp = np.vstack((temp, np.interp(zzt,z,t,left=-999,right=-999)))
                    stemp = np.vstack((stemp, np.interp(zzt,z,sigma_t,left=-999,right=-999)))
                external['nTprof'] += len(tsecs)

            if external['nTprof'] > 0:
                temp = temp.T
                stemp = stemp.T

    else:
        print('Error in read_external_profile_data: Undefined external temperature source specified')
        return external

    #If there were neither input external temperature or humidity data then return gracefully


    if ((external['nQprof'] == 0) & (external['nTprof'] == 0)):
        external['success'] = 1
        return external
    elif ((external['nQprof'] < 0) | (external['nTprof'] < 0)):
        if verbose >= 2:
            print('Error associated with read_external_profile_data -- inconsistent processing somehow')
        return external

    # Interpolate the data to the time/height grid
    # Put the external data on the same vertical grid as the TROPoe retrievals.
    # There are many ways this could be done, but since sometimes the IRS has
    # better vertical resolution than the external source (near the surface) or
    # vice versa (aloft), we are just going to do something simple and linearly
    # interpolate and then put the data on the same temporal grid as the IRS retrievels.
    # Again, there are multiple ways to do this. I will interpolate linearly with time, but then
    # flag the samples that are within the temporal resolution window of the TROPoe
    # sample time, so tht this information can be passed to the output file.

    timeflag = np.zeros(len(secs))  # this captures the time flags
    if tres == 0:
        timeres = 30         # since the IRS's rapid-sample data is order of 30 seconds
    else:
        timeres = tres * 60       # because the units here should be seconds

    # Humidity first
    if external['nQprof'] > 0:
        # First interpolate to the correct TROPoe vertical grid. But also
        # look for bad data in the raw profile.


        tmp_water = np.zeros((len(ht),len(qsecs)))
        tmp_swater = np.zeros((len(ht), len(qsecs)))
        new_water = np.zeros((len(ht), len(secs)))
        new_swater = np.zeros((len(ht), len(secs)))

        foo = np.where(wv < 0)
        wv[foo] = np.nan
        swv[foo] = np.nan
        for i in range(external['nQprof']):
            tmp_water[:,i] = np.interp(ht,zzq+wv_ht_offset,wv[:,i])
            tmp_swater[:,i] = np.interp(ht,zzq+wv_ht_offset,swv[:,i])

        # Set the data below or above the instrument's min/max heights to missing value
        foo = np.where((ht < min(zzq)) | (max(zzq) < ht))[0]
        if(len(foo) > 0):
            tmp_water[foo,:]  = np.nan
            tmp_swater[foo,:] = np.nan

        # But set the data below or above the min/max values to a missing value
        foo = np.where((ht < wv_prof_minht) | (wv_prof_maxht < ht))[0]
        if len(foo) > 0:
            tmp_water[foo,:]  = np.nan
            tmp_swater[foo,:] = np.nan


        # Now interpolate to the TROPoe temporal grid.
        for j in range(len(ht)):
            new_water[j,:]  = np.interp(secs,qsecs,tmp_water[j,:])
            new_swater[j,:] = np.interp(secs,qsecs,tmp_swater[j,:])

        # Set the interpolated data before and after the end external times to missing
        # but we will allow the interpolated data to be used over some delta_time
        # around the end times

        foo = np.where(((np.nanmin(qsecs)-wv_time_delta*3600) > secs) |
                       ((np.nanmax(qsecs)+wv_time_delta*3600) < secs))[0]

        if len(foo) > 0:
            new_water[:,foo] = np.nan
            new_swater[:,foo] = np.nan

        # Now loop over the samples, and see if there is an external sample
        # that is within tres of the TROPoe time. If so, then flag this

        for i in range(len(secs)):
            foo = np.where((secs[i]-timeres <= qsecs) & (secs[i]+2*timeres >= qsecs))[0]
            if len(foo) > 0:
                timeflag[i] = 2                   # Use 2 for water vapor flag ("1" for temp)

        # This checks to make sure there is no bad data that gets through. Mostly
        # for the old NCAR DIAL data. Water vapor must be 0 or positive and the
        # uncertainty must be greater than zero (We are never certain of anything).

        foo = np.where((new_water < 0) | (new_swater <= 0))
        new_water[foo] = np.nan
        new_swater[foo] = np.nan

        # Replace any NaN data with missing values
        new_water[np.isnan(new_water)] = -999
        new_swater[np.isnan(new_swater)] = -999

        # Now temperature....
    if external['nTprof'] > 0:
        # First interpolate to the correct TROPoe vertical grid. But also
        # look for bad data in the raw profile.

        tmp_temp = np.zeros((len(ht),len(tsecs)))
        tmp_stemp = np.zeros((len(ht), len(tsecs)))
        new_temp = np.zeros((len(ht), len(secs)))
        new_stemp = np.zeros((len(ht), len(secs)))

        foo = np.where(temp < -900)
        temp[foo] = np.nan
        stemp[foo] = np.nan
        for i in range(external['nTprof']):
            tmp_temp[:,i] = np.interp(ht,zzt+temp_ht_offset,temp[:,i])
            tmp_stemp[:,i] = np.interp(ht,zzt+temp_ht_offset,stemp[:,i])

        # Set the data below or above the instrument's min/max heights to missing value
        foo = np.where((ht < min(zzt)) | (max(zzt) < ht))[0]
        if(len(foo) > 0):
            tmp_temp[foo,:]  = np.nan
            tmp_stemp[foo,:] = np.nan

        # But set the data below or above the min/max values to a missing value
        foo = np.where((ht < temp_prof_minht) | (temp_prof_maxht < ht))[0]
        if len(foo) > 0:
            tmp_temp[foo,:]  = np.nan
            tmp_stemp[foo,:] = np.nan

        # Now interpolate to the TROPoe temporal grid
        for j in range(len(ht)):
            new_temp[j,:] = np.interp(secs,tsecs,tmp_temp[j,:])
            new_stemp[j,:] = np.interp(secs,tsecs,tmp_stemp[j,:])

        # Set the interpolated data before and after the end external times to missing
        # but we will allow the interpolated data to be used over some delta_time
        # around the end times

        foo = np.where(((np.nanmin(tsecs)-temp_time_delta*3600) > secs) |
                    ((np.nanmax(tsecs)+temp_time_delta*3600) < secs))[0]

        if len(foo) > 0:
            new_temp[:,foo] = np.nan
            new_stemp[:,foo] = np.nan

        # Now loop over the samples, and see if there is an external sample
        # that is within tres of the TROPoe time. If so, then flag this

        for i in range(len(secs)):
            foo = np.where((secs[i]-timeres <= tsecs) & (secs[i]+2*timeres >= tsecs))[0]
            if len(foo) > 0:
                timeflag[i] = 1                   # Use 1 for temperature flag ("2" for WV)

        # Replace any NaN data with missing values
        new_temp[np.isnan(new_temp)] = -999
        new_stemp[np.isnan(new_stemp)] = -999

    # Apply the multiplier to the WV noise profile, if specified and if
    # the current value is not a missing value
    if external['nQprof'] > 0:
        wv_noise_multiplier = np.interp(ht, wv_noise_multiplier_hts, wv_noise_multiplier_val)
        foo = np.where(wv_noise_multiplier < 1)[0]
        if len(foo) > 0:
            print('Error in read_external_profile_data: wv_noise_multiplier_val must be >= 1')
            return external
        if verbose >= 2:
            print('    Applying external_wv_noise_multiplier')
        for j in range(len(secs)):
            feh = np.where(new_swater[:,j] > -900)[0]
            if len(feh) > 0:
                new_swater[feh,j] *= wv_noise_multiplier[feh]

    # Apply the additive factor to the temperature noise profile, if specified
    # the current value is not a missing value

    if external['nTprof'] > 0:
        temp_noise_adder = np.interp(ht, temp_noise_adder_hts, temp_noise_adder_val)
        foo = np.where(temp_noise_adder < 0)[0]
        if len(foo) > 0:
            print('Error in read_external_profile_data: temp_noise_adder should not be negative')
            return external
        if verbose >= 2:
            print('    Applying externl_temp_noise_adder')
        for j in range(len(secs)):
            feh = np.where(new_stemp[:,j] > -900)[0]
            if len(feh) > 0:
                new_stemp[feh,j] += temp_noise_adder[feh]

    # Build the output structure properly
    if ((external['nQprof'] > 0) & (external['nTprof'] > 0)):
        external = ({'success':1, 'nTprof':external['nTprof'], 'nQprof':external['nQprof'], 'secs':secs, 'ht':ht,
                    'wvmultiplier':wvmultiplier, 'wv':new_water, 'sig_wv':new_swater, 'temp':new_temp, 'sig_temp':new_stemp,
                    'wvminht':wv_prof_minht, 'wvmaxht':wv_prof_maxht, 'tempminht':temp_prof_minht, 'tempmaxht':temp_prof_maxht,
                    'timeflag':timeflag, 'wv_type':wv_prof_type, 'temp_type':temp_prof_type, 'tunit':tunit, 'qunit':qunit,
                    'ttype':ttype, 'qtype':qtype, 'attrs': saved_attributes})

    elif external['nQprof'] > 0:
        external = ({'success':1, 'nTprof':external['nTprof'], 'nQprof':external['nQprof'], 'secs':secs, 'ht':ht,
                    'wvmultiplier':wvmultiplier, 'wv':new_water, 'sig_wv':new_swater,
                    'wvminht':wv_prof_minht, 'wvmaxht':wv_prof_maxht,
                    'timeflag':timeflag, 'wv_type':wv_prof_type, 'qunit':qunit,
                    'qtype':qtype, 'attrs': saved_attributes})

    elif external['nTprof'] > 0:
         external = ({'success':1, 'nTprof':external['nTprof'], 'nQprof':external['nQprof'], 'secs':secs, 'ht':ht,
                    'temp':new_temp, 'sig_temp':new_stemp,
                    'tempminht':temp_prof_minht, 'tempmaxht':temp_prof_maxht,
                    'timeflag':timeflag, 'temp_type':temp_prof_type, 'tunit':tunit,
                    'ttype':ttype, 'attrs': saved_attributes})

    else:
         external = ({'success':1, 'nTprof':external['nTprof'], 'nQprof':external['nQprof'],
                    'wv_type':wv_prof_type, 'temp_type':temp_prof_type, 'tunit':tunit, 'qunit':qunit,
                    'ttype':ttype, 'qtype':qtype, 'attrs': saved_attributes})

    return external

################################################################################
# This function reads in time-series data that will be used to help constrain the
# retrieval (e.g., surface met data)
################################################################################

def read_external_timeseries(date, secs, vip, dostop, verbose):

    external = {'success':0, 'nTsfc':-1, 'nQsfc':-1, 'nPsfc':-1, 'nCO2sfc':-1}
    ttype = 'None'
    qtype = 'None'
    ptype = 'None'
    co2type = 'None'

    # Some quick QC of the input entries in the VIP file
    estring = ' '
    maxpts = 1000                         # Maximum number of surface replication points allowed
    if vip['ext_sfc_temp_npts'] > 0:
        if ((vip['ext_sfc_temp_npts'] < 1) | (maxpts < vip['ext_sfc_temp_npts'])):
            estring = 'VIP input error: when sfc_temp_type > 0, then 1 <= ext_sfc_temp_npts < ' + str(maxpts)
    if vip['ext_sfc_wv_npts'] > 0:
        if ((vip['ext_sfc_wv_npts'] < 1) | (maxpts < vip['ext_sfc_wv_npts'])):
            estring = 'VIP input error; when sfc_wv_type > 0, then 1 <= ext_sfc_wv_npts < ' + str(maxpts)
    if vip['co2_sfc_npts'] > 0:
        if ((vip['co2_sfc_npts'] < 1) | (maxpts < vip['co2_sfc_npts'])):
            estring = 'VIP input error; when co2_sfc_npts > 0, then 1 <= co2_sfc_npts < ' + str(maxpts)

    if estring != ' ':
        if verbose >= 1:
            print(estring)
        return external
    
    if verbose >= 1:
            print('  Reading in external surface data')

    # Read the external surface met data all at once.  Then we will fill out the 
    # vectors that need data, as specified by the VIP file

    dates = [(datetime.strptime(str(date), '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d') ,
            str(date) ,  (datetime.strptime(str(date), '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d') ]

    files = []
    if (vip['ext_sfc_pres_type'] == 0) & (vip['ext_sfc_temp_type'] == 0) & (vip['ext_sfc_wv_type'] == 0):
        pass  # No need to try and read in files 
    
    else:
        for i in range(len(dates)):
            tempfiles, status = (findfile(vip['ext_sfc_path'],'*'+vip['ext_sfc_rootname']+'*' + dates[i] + '*.(cdf|nc)'))
            if status == 1:
                return external
            files = files + tempfiles

        # Print out a warning if no files are found, but don't need to fail since things are handled later
        if len(files) == 0:
            if verbose >= 1:
                print('    No external met found in the input directory for this date')

    # if len(files) == 0, this is not run   
    for i in range(len(files)):
        fid = Dataset(files[i], 'r')
        fid.set_auto_mask(False)

        # Read in the time field(s)
        if vip['ext_sfc_time_format'] == 0:
            bt = fid.variables['base_time'][:].astype('float')
        elif vip['ext_sfc_time_format'] == 1:
            bt = fid.variables['base_time'][:].astype('float')
            bt /= 1000.     # Convert milliseconds to seconds (epoch time)
        else:
            bt = 0
        if len(np.where(np.array(list(fid.variables.keys())) == 'time_offset')[0]) > 0:
            to = fid.variables['time_offset'][:].astype('float')
        elif len(np.where(np.array(list(fid.variables.keys())) == 'time')[0]) > 0:
            to = fid.variables['time'][:].astype('float')
        else:
            fid.close()
            print('    Problem reading the external surface met time data -- returning missing data')
            return external

        # Now read in the pressure field, as specified by the VIP file
        if len(np.where(np.array(list(fid.variables.keys())) == vip['ext_sfc_pres_fieldname'])[0]) > 0:
            p = fid.variables[vip['ext_sfc_pres_fieldname']][:].astype('float')
            if vip['ext_sfc_pres_units'] == 1:
                p *= 10.                        # Convert kPa to hPa
            elif vip['ext_sfc_pres_units'] == 2:
                p *=  1.                        # Do nothing (units are correct)
            elif vip['ext_sfc_pres_units'] == 3:
                p /= 100.                       # Convert Pa to hPa
            else:
                fid.close()
                print(f"    Problem converting pressure units in the external met data -- ",
                                f"undefined option {vip['ext_sfc_pres_units']:d} but needs to be one of (1,2,3)")
                sys.exit()
        else:
            print('    Problem reading the external met pressure data -- the field ',
                            vip['ext_sfc_pres_fieldname'],' does not exist')
            p = np.ones(len(to))*(-999.)

        # Now read in the temperature field, as specified by the VIP file
        if len(np.where(np.array(list(fid.variables.keys())) == vip['ext_sfc_temp_fieldname'])[0]) > 0:
            t = fid.variables[vip['ext_sfc_temp_fieldname']][:].astype('float')
            if vip['ext_sfc_temp_units'] == 1:
                t +=  0.                        # Do nothing (units are correct in degC)
            elif vip['ext_sfc_temp_units'] == 2:
                t -=  273.15                    # Convert degK to degC
            else:
                fid.close()
                print(f"    Problem converting temperature units in the external met data -- ",
                                f"undefined option {vip['ext_sfc_temp_units']:d} but needs to be one of (1,2)")
                sys.exit()
        else:
            print('    Problem reading the external met temperature data -- the field ',
                            vip['ext_sfc_temp_fieldname'],' does not exist')
            t = np.ones(len(to))*(-999.)

        # Now read in the water vapor (relative humidity) field, as specified by the VIP file
        if len(np.where(np.array(list(fid.variables.keys())) == vip['ext_sfc_wv_fieldname'])[0]) > 0:
            u = fid.variables[vip['ext_sfc_wv_fieldname']][:].astype('float')
            if vip['ext_sfc_wv_units'] == 1:
                u +=  0.                        # Do nothing (units are correct in %rh)
            elif vip['ext_sfc_wv_units'] == 2:
                u *=  100.                      # Convert fraction_rh to %rh
            else:
                fid.close()
                print(f"    Problem converting water vapor (RH) units in the external met data -- ",
                                f"undefined option {vip['ext_sfc_wv_units']:d} but needs to be one of (1,2)")
                sys.exit()
        else:
            print('    Problem reading the external met water vapor (RH) data -- the field ',
                            vip['ext_sfc_wv_fieldname'],' does not exist')
            u = np.ones(len(to))*(-999.)

        # Now close the file being read
        fid.close()

        if verbose >= 2: 
            print('    Here are the first 10 surface met data that was just read in:')
            print('           sfcP  sfcT   sfcRH')
            for j in range(1,10):
                print(f"        {p[j]:.3f}, {t[j]:.2f}, {u[j]:.2f}")

        # Check if any variable is completely missing
        is_missing = lambda x: set(x) == set(np.array([-999.]))
        if is_missing(p) | is_missing(t) | is_missing(u):
            print("    Error: If using any surface data, then ext_sfc_pres_fieldname, ext_sfc_temp_fieldname,  \n" +
                  "      and ext_sfc_wv_fieldname must point to real variables with non-missing data within \n" + 
                  "      the file (even if they are not being used in the retrieval)")
            return external
        
        # Do some checks to see if the units are anywhere close
        foo = np.where((p < 300) | (p > 1050))[0]   # these are generous thresholds for pres in the trop [mb]
        if len(foo) == len(p):
            print("    Possible Units Error: all external surface pressure values are outside (300,1050) mb",
                            "-- check units in VIP file")
            sys.exit(1)
        foo = np.where((t < -100) | (t > 200))[0]   # these are generous thresholds for temp in the trop [degC]
        if len(foo) == len(t):
            print("    Possible Units Error: all external surface temperature values are outside (-100,200) degC",
                            "-- check units in VIP file")
            sys.exit(1)
        foo = np.where((u < 1) | (u > 105))[0]   # these are generous thresholds for RH in the trop [%]
        if len(foo) == len(u):
            print("    Possible Units Error: all external surface RH values are outside (1,105) %RH",
                            "-- check units in VIP file")
            sys.exit(1)

            # Some simple QC for pressure (find any bad points)
        foo = np.where((p > 0) & (p < 1050))[0]
        if len(foo) < 1:
            print("    External Pressure QC: all values fall outside (0, 1050)",
                                f"-- Skipping {files[i]}")
            continue
        to = to[foo]
        p  = p[foo]
        t  = t[foo]
        u  = u[foo]

            # Append the pressure data to the growning structure
        if external['nPsfc'] <= 0:
            psecs = bt+to
            press = np.copy(p)
        else:
            psecs = np.append(psecs,bt+to)
            press = np.append(press, p)
        external['nPsfc'] = len(psecs)

            # Some simple QC for temperature (find any bad points; this adds to previous QC on pres)
        foo = np.where((t < 70) & (-100 < t))[0]
        if len(foo) < 1:
            print("    External Temp QC: all values fall outside (-100, 70)",
                                f"-- Skipping {files[i]}")
            continue
        to = to[foo]
        p  = p[foo]
        t  = t[foo]
        u  = u[foo]

            # Append the temperature data to the growing structure
        sigma_t = vip['ext_sfc_temp_random_error']
        if external['nTsfc'] <= 0:
            tsecs = bt+to
            temp  = np.copy(t)
            stemp = np.ones(len(t))*sigma_t
        else:
            tsecs = np.append(tsecs,bt+to)
            temp  = np.append(temp,t)
            stemp = np.append(stemp,np.ones(len(t))*sigma_t)
        external['nTsfc'] = len(tsecs)

            # Some simple QC for water vapor (find any bad points; this adds to previous QC on pres and temp)
        foo = np.where((u >= 0) & (u < 103))[0]
        if len(foo) < 1:
            print("    External RH QC: all values fall outside (0, 103)",
                                f"-- Skipping {files[i]}")
            continue
        to = to[foo]
        p  = p[foo]
        t  = t[foo]
        u  = u[foo]

            # Convert surface RH to surface WVMR, and compute its random uncertainty
        sigma_u = vip['ext_sfc_rh_random_error']       # %RH
        w0 = Calcs_Conversions.rh2w(t, u/100., p)
        w1 = Calcs_Conversions.rh2w(t+sigma_t, u/100., p)
        w2 = Calcs_Conversions.rh2w(t-sigma_t, u/100., p)
        u_plus  = (u+sigma_u)/100.
        u_minus = (u-sigma_u)/100.
        u_plus[u_plus > 1] = 1
        u_minus[u_minus < 0] = 0
        w3 = Calcs_Conversions.rh2w(t, u_plus, p)
        w4 = Calcs_Conversions.rh2w(t, u_minus, p)
            # Sum of squared errors, but take two-side average for T and RH uncerts
        sigma_w = np.sqrt( ((w1-w0)**2 + (w2-w0)**2)/2. + ((w3-w0)**2 + (w4-w0)**2)/2. )

            # Append the water vapor data to the growning structure
        if external['nQsfc'] <= 0:
            qsecs = bt+to
            wv = np.copy(w0)
            swv = np.copy(sigma_w)
        else:
            qsecs = np.append(qsecs,bt+to)
            wv = np.append(wv,w0)
            swv = np.append(swv,sigma_w)
        external['nQsfc'] = len(qsecs)

        if(verbose >= 2):
            print(f"    After reading in met file {files[i]:s}, the number of p, t, u points ",
                            f"are {external['nPsfc']:d}, {external['nTsfc']:d}, and {external['nQsfc']:d}")
    # Done with the loop over reading the files

        # Now add the logic to set the data fields properly
    if vip['ext_sfc_pres_type'] == 0:
        ptype = 'none'
        punit = ''
        external['nPsfc'] = 0
    else:
        punit = 'mb'
        ptype = 'External surface met station'
    
    if vip['ext_sfc_temp_type'] == 0:
        ttype = 'none'
        tunit = ''
        external['nTsfc'] = 0
    else:
        tunit = 'degC'
        ttype = 'External surface met station'
    
    if vip['ext_sfc_wv_type'] == 0:
        qtype = 'none'
        qunit = ''
        external['nQsfc'] = 0
    else:
        qunit = 'g/kg'
        qtype = 'External surface met station'

    # Add on the representativeness errors that were specified
    if ((external['nTsfc'] > 0) & (vip['ext_sfc_temp_rep_error'] > 0)):
        stemp += vip['ext_sfc_temp_rep_error']
    if ((external['nQsfc'] > 0) & (vip['ext_sfc_wv_mult_error'] > 0)):
        swv *= vip['ext_sfc_wv_mult_error']
    if ((external['nQsfc'] > 0) & (vip['ext_sfc_wv_rep_error'] > 0)):
        swv += vip['ext_sfc_wv_rep_error']

    # Now I need to bin/interpolate the data appropriately
    if external['nTsfc'] > 0:
        # Compute the median time interval between Tsfc measurements [minutes]
        tdel = np.nanmedian(tsecs[1:len(tsecs)] - tsecs[0:len(tsecs)-1]) / 60.   # in minutes

        # If the median time interval is much smaller than tavg, then we will
        # bin up the data. Otherwise, we will just interpolate linearly

        if (tdel*4 < vip['tres']):
            #Bin the data
            tt0 = np.zeros(len(secs))
            st0 = np.zeros(len(secs))
            for i in range(len(secs)):
                foo = np.where((secs[i]-vip['tres']*60/2. <= tsecs) & (tsecs <= secs[i] + vip['tres']*60/2.))[0]
                if len(foo) > 0:
                    tt0[i] = np.nanmean(temp[foo])
                    st0[i] = np.nanmean(stemp[foo])
                else:
                    tt0[i] = -999.
                    st0[i] = -999.

        else:
            tt0 = np.interp(secs,tsecs,temp)
            st0 = np.interp(secs,tsecs,stemp)

            # Make sure we did not interpolate out of bounds here.
            foo = np.where(secs < tsecs[0]-vip['ext_sfc_time_delta']*3600)[0]
            if len(foo) > 0:
                tt0[foo] = -999.
                st0[foo] = -999.
            foo = np.where((tsecs[0]-vip['ext_sfc_time_delta']*3600 <= secs) & (secs < tsecs[0]))[0]
            if len(foo) > 0:
                tt0[foo] = temp[0]
                st0[foo] = stemp[0]
            n = len(tsecs) - 1
            foo = np.where(tsecs[n]+vip['ext_sfc_time_delta']*3600 < secs)[0]
            if len(foo) > 0:
                tt0[foo] = -999.
                st0[foo] = -999.
            foo = np.where((tsecs[n] < secs) & (secs <= tsecs[n]+vip['ext_sfc_time_delta']*3600))[0]
            if len(foo) > 0:
                tt0[foo] = temp[n]
                st0[foo] = stemp[n]
    else:
        tt0 = -999.
        st0 = -999

    if external['nQsfc'] > 0:
        # Compute the median time interval between Tsfc measurements [minutes]
        tdel = np.nanmedian(qsecs[1:len(qsecs)] - qsecs[0:len(qsecs)-1]) / 60.

        # If the median time interval is much smaller than tavg, then we will
        # bin up the data. Otherwise, we will just interpolate linearly

        if (tdel*4 < vip['tres']):
            #Bin the data
            qq0 = np.zeros(len(secs))
            sq0 = np.zeros(len(secs))
            for i in range(len(secs)):
                foo = np.where((secs[i]-vip['tres']*60/2. <= qsecs) & (qsecs <= secs[i] + vip['tres']*60/2.))[0]
                if len(foo) > 0:
                    qq0[i] = np.nanmean(wv[foo])
                    sq0[i] = np.nanmean(swv[foo])
                else:
                    qq0[i] = -999.
                    sq0[i] = -999.
        else:
            qq0 = np.interp(secs,qsecs,wv)
            sq0 = np.interp(secs,qsecs,swv)

            # Make sure we did not interpolate out of bounds here.
            foo = np.where(secs < qsecs[0]-vip['ext_sfc_time_delta']*3600)[0]
            if len(foo) > 0:
                qq0[foo] = -999.
                sq0[foo] = -999.
            foo = np.where((qsecs[0]-vip['ext_sfc_time_delta']*3600 <= secs) & (secs < qsecs[0]))[0]
            if len(foo) > 0:
                qq0[foo] = wv[0]
                sq0[foo] = swv[0]
            n = len(qsecs) - 1
            foo = np.where(qsecs[n]+vip['ext_sfc_time_delta']*3600 < secs)[0]
            if len(foo) > 0:
                qq0[foo] = -999.
                sq0[foo] = -999.
            foo = np.where((qsecs[n] < secs) & (secs <= qsecs[n]+vip['ext_sfc_time_delta']*3600))[0]
            if len(foo) > 0:
                qq0[foo] = wv[n]
                sq0[foo] = swv[n]
    else:
        qq0 = -999.
        sq0 = -999.

    if external['nPsfc'] > 0:
        # Compute the median time interval between Psfc measurements [minutes]
        tdel = np.nanmedian(psecs[1:len(psecs)] - psecs[0:len(psecs)-1]) / 60.

        # If the median time interval is much smaller than tavg, then we will
        # bin up the data. Otherwise, we will just interpolate linearly

        if (tdel*4 < vip['tres']):
            #Bin the data
            p0 = np.zeros(len(secs))
            for i in range(len(secs)):
                foo = np.where((secs[i]-vip['tres']*60/2. <= psecs) & (psecs <= secs[i] + vip['tres']*60/2.))[0]
                if len(foo) > 0:
                    p0[i] = np.nanmean(press[foo])
                else:
                    p0[i] = -999.
        else:
            p0 = np.interp(secs,psecs,press)

            # Make sure we did not interpolate out of bounds here.
            foo = np.where(secs < psecs[0]-vip['ext_sfc_time_delta']*3600)[0]
            if len(foo) > 0:
                p0[foo] = -999.
            foo = np.where((psecs[0]-vip['ext_sfc_time_delta']*3600 <= secs) & (secs < psecs[0]))[0]
            if len(foo) > 0:
                p0[foo] = press[0]
            n = len(psecs) - 1
            foo = np.where(psecs[n]+vip['ext_sfc_time_delta']*3600 < secs)[0]
            if len(foo) > 0:
                p0[foo] = -999.
            foo = np.where((psecs[n] < secs) & (secs <= psecs[n]+vip['ext_sfc_time_delta']*3600))[0]
            if len(foo) > 0:
                p0[foo] = press[n]
    else:
        p0 = -999.

    # This section is for the CO2 obs
    # Read in the surface in-situ CO2 data, if desired
    # No external surface CO2 source specified....
    co2unit = ' '
    if vip['co2_sfc_type'] == 0:
        co2type = 'none'
        external['nCo2sfc'] = 0

    # Read in the surface in-situ CO2 data (assuming DDT's PGS qc1turn datastream)
    elif vip['co2_sfc_type'] == 1:
        if verbose >= 1:
            print('  Reading in ARM PGS qc1turn datastream')

        files = []
        for i in range(len(dates)):
            tempfiles, status = (findfile(vip['co2_sfc_path'],'*pgs*' + dates[i] + '*.(cdf|nc)'))
            if status == 1:
                return external
            files = files + tempfiles

        if len(files) == 0:
            if verbose >= 1:
                print('    No ARM CO2 found in this directory for this date')
        else:
            for i in range(len(files)):
                fid = Dataset(files[i],'r')
                fid.set_auto_mask(False)
                bt = fid.variables['base_time'][:].astype('float')
                to = fid.variables['time_offset'][:].astype('float')
                xco2 = fid.variables['c02_02m'][:]                  # ppm
                xsco2 = fid.variables['sigma_co2_02m'][:]           # ppm
                fid.close()
                co2unit = 'ppm'
                co2type = 'ARM PGS qc1turn'

                # Append the data to the growing structure
                if external['nCo2sfc <= 0']:
                    co2secs = bt+to
                    co2 = np.copy(xco2)
                    sco2 = np.copy(xsco2)
                else:
                    co2secs = np.append(co2secs,bt+to)
                    co2 = np.append(co2,xco2)
                    sco2 = np.append(sco2,xsco2)
                external['nCO2sfc'] = len(co2secs)

        # An undefined surface in-situ CO2 source was specified...
    else:
        print('Error in read_external_tseries: Undefined in-situ CO2 surface obs source specified')
        return external

    # Strip out any missing values before the next step
    if external['nCo2sfc'] > 0:
        foo = np.where(co2 > 0)[0]
        if len(foo) > 0:
            external['nCo2sfc'] = len(foo)
            co2secs = co2secs[foo]
            co2 = co2[foo]
            sco2 = sco2[foo]
        else:
            external['nCo2sfc'] = 0

    # Add on the representativeness errors that were specified
    if ((external['nCo2sfc'] > 0) & (vip['co2_sfc_rep_error'] > 0)):
        sco2 += vip['co2_sfc_rep_error']

    # Now I need to bin/interpolate the data appropriately
    if external['nCo2sfc'] > 0:
        # Compute the median time interval between CO2 measurements [minutes]
        tdel = np.nanmedian(co2secs[1:len(co2secs)] - co2secs[0:len(co2secs)-1]) / 60.

        # If the median time interval is much smaller than tavg, then
        # we will bin up the data. Otherwise, we will just interpolate linearly

        if (tdel*4 < vip['tres']):
            # Bin the data
            cco2 = np.zeros(len(secs))
            scco2 = np.zeros(len(secs))
            for i in range(len(secs)):
                foo = np.where((secs[i]-vip['tres']*60/2. <= co2secs) & (co2secs <= secs[i] + vip['tres']*60/2.))[0]
                if len(foo) > 0:
                    cco2[i] = np.nanmean(co2[foo])
                    scco2[i] = np.nanmean(sco2[foo])
                else:
                    cco2[i] = -999.
                    scco2[i] = -999.
        else:
            cco2 = np.interp(secs,co2secs,co2)
            scco2 = np.interp(secs,co2secs,sco2)

            # Make sure we did not interpolate out of bounds here.
            foo = np.where(secs < co2secs[0]-vip['co2_sfc_time_delta']*3600)[0]
            if len(foo) > 0:
                cco2[foo] = -999.
                scco2[foo] = -999.
            foo = np.where((co2secs[0]-vip['co2_sfc_time_delta']*3600 <= secs) & (secs < co2secs[0]))[0]
            if len(foo) > 0:
                cco2[foo] = co2[0]
                scco2[foo] = sco2[0]
            n = len(co2secs) - 1
            foo = np.where(co2secs[n]+vip['co2_sfc_time_delta']*3600 < secs)[0]
            if len(foo) > 0:
                cco2[foo] = -999.
                scco2[foo] = -999.
            foo = np.where((co2secs[n] < secs) & (secs <= co2secs[n]+vip['co2_sfc_time_delta']*3600))[0]
            if len(foo) > 0:
                cco2[foo] = co2[n]
                scco2[foo] = sco2[n]
    else:
        cco2 = -999.
        scco2 = -999.

    # Now I need to build arrays for each suface sample, so that I can
    # replicate a point multiple times if desired. I don't want to make each
    # of the replicates the same as the original, so I will insert 1/10th of
    # the random error into the time-series...

    if external['nTsfc'] > 0:
        tt1 = np.zeros((vip['ext_sfc_temp_npts'],len(secs)))
        st1 = np.zeros((vip['ext_sfc_temp_npts'],len(secs)))
        tt1[0,:] = tt0
        st1[0,:] = st0
        for j in range(1,vip['ext_sfc_temp_npts']):
            tt1[j,:] = tt0 + np.random.normal(size = len(secs))*st0/10.
            st1[j,:] = np.copy(st0)
    else:
        tt1 = np.copy(tt0)
        st1 = np.copy(st0)

    if external['nQsfc'] > 0:
        qq1 = np.zeros((vip['ext_sfc_wv_npts'],len(secs)))
        sq1 = np.zeros((vip['ext_sfc_wv_npts'],len(secs)))
        qq1[0,:] = np.copy(qq0)
        sq1[0,:] = np.copy(sq0)
        for j in range(1,vip['ext_sfc_wv_npts']):
            qq1[j,:] = qq0 + np.random.normal(size = len(secs))*sq0/10.
            sq1[j,:] = np.copy(sq0)
    else:
        qq1 = np.copy(qq0)
        sq1 = np.copy(sq0)

    if external['nCo2sfc'] > 0:
        cco2a = np.zeros((vip['co2_sfc_npts'],len(secs)))
        scco2a = np.zeros((vip['co2_sfc_npts'],len(secs)))
        cco2a[0,:] = np.copy(cco2)
        scco2a[0,:] = np.copy(scco2)
        for j in range(1,vip['co2_sfc_npts']):
            cco2a[j,:] = cco2 + np.random.normal(size = len(secs))*scco2/10.
            scco2a[j,:] = np.copy(scco2)
    else:
        cco2a = np.copy(cco2)
        scco2a = np.copy(scco2)

    # Build the output structure and return it

    external = {'success':1, 'sfc_relative_height':vip['ext_sfc_relative_height'],
          'nTsfc':external['nTsfc'],'nptsT':vip['ext_sfc_temp_npts'], 'tunit':tunit, 'ttype':ttype,
          'temp':tt1, 'stemp':st1, 'nQsfc':external['nQsfc'], 'nptsQ':vip['ext_sfc_wv_npts'], 'qunit':qunit,
          'qtype':qtype, 'wv':qq1, 'swv':sq1, 'sfc_temp_rep_error':vip['ext_sfc_temp_rep_error'],
          'sfc_wv_rep_error':vip['ext_sfc_wv_rep_error'], 'sfc_wv_mult_error':vip['ext_sfc_wv_mult_error'],
          'nCO2sfc':external['nCO2sfc'], 'co2unit':co2unit, 'nptsCO2':vip['co2_sfc_npts'],
          'co2type':co2type, 'co2':cco2a, 'sco2':scco2a, 'co2_sfc_rep_error':vip['co2_sfc_rep_error'],
          'co2_sfc_relative_height':vip['co2_sfc_relative_height'], 'nPsfc':external['nPsfc'],
          'ptype':ptype, 'psfc': p0}
    return external

################################################################################
# This function looks at the name of the TROPoe tar file to get the version of the code
################################################################################

def get_tropoe_version():
    tropoe_tar_file = findfile(".","TROPoe*tar.gz",verbose=1)[0]
    if(len(tropoe_tar_file) > 0):
        tropoe_tar_file = tropoe_tar_file[0]
        version = tropoe_tar_file[2:-7]
    else:
        tropoe_tar_file = findfile(".","TROPoe*tar",verbose=1)[0]
        if(len(tropoe_tar_file) > 0):
            tropoe_tar_file = tropoe_tar_file[0]
            version = tropoe_tar_file[2:-4]
        else:
            version = 'UNKNOWN'

    return version

################################################################################
# This function is used to read the TAPE6 generated by the LBLRTM (which is like
# reading the stdout from the model), and parsing out the minimum and maximum
# wavenumber that were in the TAPE3 file (i.e., the spectroscopic dataset)
################################################################################

def tape6_min_max_wnum(lblrtm_path,verbose=2):

    err = {'success':0}
    filename = lblrtm_path + '/TAPE6'
    if os.path.exists(filename):
        with open(filename) as file:
            data = file.readlines()
    else:
        print('  The LBLRTM TAPE6 file ' + filename + ' does not exist')
        return err

    if len(data) == 0:
        print('  There were no valid lines found in the LBLRTM TAPE6 file')
        return err

    i = 0
    maxi = 100
    for item in data:
        if item[2:13] == 'LOWEST LINE':
            break
        i = i + 1
        if i > maxi:
            break

    if i >= maxi:
        if verbose > 1:
            print('  Error: Unable to determine the minimum and maximum wavenumbers used in the TAPE3')
        return err

    minw = float(item[17:31])
    maxw = float(item[49:62])
    nlin = int(item[88:100])
    if verbose > 1:
        print(f'      The wavenumber limits in the TAPE3 file: {minw:.3f} to {maxw:.3f} cm-1')

    return {'success':1, 'minw':minw, 'maxw':maxw, 'nlines':nlin}
