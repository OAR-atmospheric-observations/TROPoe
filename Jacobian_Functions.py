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

import numpy as np
import os
import glob
import scipy
import errno
from datetime import datetime
from subprocess import Popen, PIPE
from netCDF4 import Dataset


import Data_reads
import Other_functions
import Output_Functions
import Calcs_Conversions
import LBLRTM_Functions
import sys

################################################################################
# This file contains the following functions:
# compute_jacobian_irs_interpol()
# compute_jacobian_microwave_finitediff()
# compute_jacobian_microwave_3method()
# compute_jacobian_microwave_lwp_only()
# compute_jacobian_external_temp_profiler()
# compute_jacobian_external_wv_profiler()
# compute_jacobian_external_sfc_met()
# compute_jacobian_external_sfc_co2()
# compute_jacobian_microwavescan_3method()
# compute_jacobian_pobs()
# make_lblrtm_calc()
# mixcra_forward_model()
################################################################################


################################################################################
# This routine computes the radiance jacobian for change in the state vector
# (which includes temperature and humidity profiles, trace gas amounts, and
# cloud properties).  It interpolates the monochromatic optical depths to a
# fixed spectral grid, which is coarser than monochromatic but finer than
# the IRS's resolution, which allows the routine to be pretty fast.
################################################################################

def compute_jacobian_irs_interpol(X, p, zz, lblhome, lbldir, lblroot, lbl_std_atmos, tp5, tp3,
                    cbh, sspl, sspi, lblwnum1, lblwnum2, fixt, fixwv, doco2, doch4, don2o,
                    fixlcld, fixicld, fix_co2_shape, fix_ch4_shape, fix_n2o_shape,
                    jac_maxht, awnum, forward_threshold, sfc_alt, extra_layers,
                    stdatmos, npts_per_wnum, irs_type, tape3_info, vip, 
                    verbose, debug, doapodize):

    success = 0
    quiet   = 1              # this is for the lbl_read() function
    append_devnull = True    # Set this to have the LBLRTM output piped to /dev/null, otherwise it will be output

    if sfc_alt is None:
        sfcz = 0.
    else:
        sfcz = sfc_alt / 1000.

    if npts_per_wnum is None:
        npts_per_wnum = 10

    doapoJac = False         # The flag to apodize (or not) when computing the jacobian
    doapoFor = doapodize     # The flag to apodize (or not) when computing the final forward calculation

    stime = datetime.now()

    k = len(zz)
    t = np.copy(X[0:k])            # degC
    w = np.copy(X[k:2*k])          # g/kg
    lwp = X[2*k]                   # g/m2
    reffl = X[2*k+1]               # um
    taui  = X[2*k+2]               # (ice optical depth) unitless
    reffi = X[2*k+3]               # um
    co2 = np.copy(X[2*k+4:2*k+7])           # [ppmv,ppmv,unitless], but depends on the model used
    ch4 = np.copy(X[2*k+7:2*k+10])          # [ppmv,ppmv,unitless], but depends on the model used
    n2o = np.copy(X[2*k+10:2*k+13])         # [ppmv,ppmv,unitless], but depends on the model used

    t += 273.16          # Convert degC to degK

    # Path to the "lblrun" script, as I need to assume it is not in the users path
    lblrun = lblhome + '/bin/lblrun'

    # define the wavenumber delta for the two different types of REFIR-PAD instruments 
    if(irs_type == 6):
        refir_delwnum = 0.4000
    if(irs_type == 7):
        refir_delwnum = 0.3892450

    # Get the trace gas profiles
    co2prof = Other_functions.trace_gas_prof(doco2, zz, co2)
    ch4prof = Other_functions.trace_gas_prof(doch4, zz, ch4)
    n2oprof = Other_functions.trace_gas_prof(don2o, zz, n2o)
    if((doco2 >= 1) & (verbose >= 2)):
        print(f'      Inside IRS jacobian routine, CO2 vector is {co2[0]:.2f}, {co2[1]:.2f}, {co2[2]:.3f}')
    if((doch4 >= 1) & (verbose >= 2)):
        print(f'      Inside IRS jacobian routine, CH4 vector is {ch4[0]:.4f}, {ch4[1]:.4f}, {ch4[2]:.3f}')
    if((don2o >= 1) & (verbose >= 2)):
        print(f'      Inside IRS jacobian routine, N2O vector is {n2o[0]:.4f}, {n2o[1]:.4f}, {n2o[2]:.3f}')

    # Define the model layers
    if len(extra_layers) > 0:
        mlayerz = sfcz + np.append(zz, extra_layers)
        mlayert = np.append(t, np.interp(sfcz+extra_layers, stdatmos['z'], stdatmos['t']))
    else:
        mlayerz = sfcz + zz
        mlayert = t

    # Make the baseline run
    if verbose >= 3:
        print('Making the LBLRTM runs for the Jacobian')

    LBLRTM_Functions.rundecker(3, lbl_std_atmos, zz+sfcz, p, t, w,
             co2_profile=co2prof, ch4_profile=ch4prof, n2o_profile=n2oprof,
             od_only=1, mlayers=mlayerz, wnum1=lblwnum1, wnum2=lblwnum2, tape5=tp5+'.1',
             v10=True, silent=True)

    command1 = ('setenv LBL_HOME ' +lblhome + ' ; '+
                    'rm -rf ' + lblroot + '.1 ; '+
                    'mkdir ' + lblroot + '.1 ; ' +
                    'setenv LBL_RUN_ROOT ' + lblroot + '.1 ; '+
                    'rm -rf ' + lbldir + '.1 ; '+
                    '(' + lblrun + ' ' + tp5 + '.1 ' + lbldir + '.1 ' + tp3 + ')')
    if append_devnull:
         command1 = (command1 + ' >& /dev/null')

    if fixt != 1:
        tpert = 1.0            # Additive perturbation of 1 K

        LBLRTM_Functions.rundecker(3, lbl_std_atmos, zz+sfcz, p, t+tpert, w,
             co2_profile=co2prof, ch4_profile=ch4prof, n2o_profile=n2oprof,
             od_only=1, mlayers=mlayerz, wnum1=lblwnum1, wnum2=lblwnum2, tape5=tp5+'.2',
             v10=True, silent=True)

        command2 = ('setenv LBL_HOME ' +lblhome + ' ; '+
                    'rm -rf ' + lblroot + '.2 ; '+
                    'mkdir ' + lblroot + '.2 ; ' +
                    'setenv LBL_RUN_ROOT ' + lblroot + '.2 ; '+
                    'rm -rf ' + lbldir + '.2 ; '+
                    '(' + lblrun + ' ' + tp5 + '.2 ' + lbldir + '.2 ' + tp3 + ')')
        if append_devnull:
            command2 = (command2 + ' >& /dev/null')

    else:
        command2 = 'sleep 1'

    if fixwv != 1:
	    # Set the size of the perturbation as a crude function of the WVMR at the surface
        h2opert = np.interp(w[0],[0.01,0.1,2],[0.75,0.90,0.99])

        LBLRTM_Functions.rundecker(3, lbl_std_atmos, zz+sfcz, p, t, w*h2opert,
             co2_profile=co2prof, ch4_profile=ch4prof, n2o_profile=n2oprof,
             od_only=1, mlayers=mlayerz, wnum1=lblwnum1, wnum2=lblwnum2, tape5=tp5+'.3',
             v10=True, silent=True)

        command3 = ('setenv LBL_HOME ' +lblhome + ' ; '+
                    'rm -rf ' + lblroot + '.3 ; '+
                    'mkdir ' + lblroot + '.3 ; ' +
                    'setenv LBL_RUN_ROOT ' + lblroot + '.3 ; '+
                    'rm -rf ' + lbldir + '.3 ; '+
                    '(' + lblrun + ' ' + tp5 + '.3 ' + lbldir + '.3 ' + tp3 + ')')
        if append_devnull:
            command3 = (command3 + ' >& /dev/null')
    else:
        command3 = 'sleep 1'

    if doco2 >= 1:
        co2pert = 4.
        c0 = np.copy(co2)
        c0[0] += co2pert        #An additive perturbation [ppm]
        co2prof2 = Other_functions.trace_gas_prof(doco2, zz, c0)

        LBLRTM_Functions.rundecker(3, lbl_std_atmos, zz+sfcz, p, t, w,
             co2_profile=co2prof2, ch4_profile=ch4prof, n2o_profile=n2oprof,
             od_only=1, mlayers=mlayerz, wnum1=lblwnum1, wnum2=lblwnum2, tape5=tp5+'.4',
             v10=True, silent=True)

        command4 = ('setenv LBL_HOME ' +lblhome + ' ; '+
                    'rm -rf ' + lblroot + '.4 ; '+
                    'mkdir ' + lblroot + '.4 ; ' +
                    'setenv LBL_RUN_ROOT ' + lblroot + '.4 ; '+
                    'rm -rf ' + lbldir + '.4 ; '+
                    '(' + lblrun + ' ' + tp5 + '.4 ' + lbldir + '.4 ' + tp3 + ')')
        if append_devnull:
            command4 = (command4 + ' >& /dev/null')
    else:
        command4 = 'sleep 1'

    if doch4 >= 1:
        ch4pert = 0.020
        c0 = np.copy(ch4)
        c0[0] += ch4pert        #An additive perturbation [ppm]
        ch4prof2 = Other_functions.trace_gas_prof(doch4, zz, c0)

        LBLRTM_Functions.rundecker(3, lbl_std_atmos, zz+sfcz, p, t, w,
             co2_profile=co2prof, ch4_profile=ch4prof2, n2o_profile=n2oprof,
             od_only = 1, mlayers=mlayerz, wnum1=lblwnum1, wnum2=lblwnum2, tape5=tp5+'.5',
             v10=True, silent=True)

        command5 = ('setenv LBL_HOME ' +lblhome + ' ; '+
                    'rm -rf ' + lblroot + '.5 ; '+
                    'mkdir ' + lblroot + '.5 ; ' +
                    'setenv LBL_RUN_ROOT ' + lblroot + '.5 ; '+
                    'rm -rf ' + lbldir + '.5 ; '+
                    '(' + lblrun + ' ' + tp5 + '.5 ' + lbldir + '.5 ' + tp3 + ')')
        if append_devnull:
            command5 = (command5 + ' >& /dev/null')
    else:
        command5 = 'sleep 1'

    if don2o >= 1:
        n2opert = 0.0031
        c0 = np.copy(n2o)
        c0[0] += n2opert      #An additive perturbation [ppm]
        n2oprof2 = Other_functions.trace_gas_prof(don2o, zz, c0)

        LBLRTM_Functions.rundecker(3, lbl_std_atmos, zz+sfcz, p, t, w,
             co2_profile=co2prof, ch4_profile=ch4prof, n2o_profile=n2oprof2,
             od_only=1, mlayers=mlayerz, wnum1=lblwnum1, wnum2=lblwnum2, tape5=tp5+'.6',
             v10=True, silent=True)

        command6 = ('setenv LBL_HOME ' +lblhome + ' ; '+
                    'rm -rf ' + lblroot + '.6 ; '+
                    'mkdir ' + lblroot + '.6 ; ' +
                    'setenv LBL_RUN_ROOT ' + lblroot + '.6 ; '+
                    'rm -rf ' + lbldir + '.6 ; '+
                    '(' + lblrun + ' ' + tp5 + '.6 ' + lbldir + '.6 ' + tp3 + ')')
        if append_devnull:
            command6 = (command6 + ' >& /dev/null')
    else:
        command6 = 'sleep 1'

    # String all of the commands together and make a single Popen call
    command = ('('+command1+')& ; ('+command2+')& ; ('+command3+')& ; ('+command4 +
  		')& ; ('+command5+')& ; ('+command6+')& ; wait ')

    if append_devnull:
        command = (command + ' >& /dev/null')

    # Logic to catch various errors that could occur when we spawn off the LBLRTM runs
    try:
        with Popen(command, stdout = PIPE, stderr = PIPE, shell=True, executable = '/bin/csh') as process:
            stdout, stderr = process.communicate()
                # Catch cases where the process launched, but LBLRTM exited with an error
            if process.returncode != 0:
                print(f"LBLRTM execution (1) failed with exit code {process.returncode}")
                print(f"Stderr output: {stderr.decode('utf-8', errors='replace')}")
                return success, -999., -999., -999., -999., -999., tape3_info
        # Catches failures occurring during process creation
    except OSError as e:
        if e.errno == errno.EAGAIN:
            print("Popen error with LBLRTM: Out of PIDs or process limit reached (EAGAIN).")
        elif e.errno == errno.ENOMEM:
            print("Popen error with LBLRTM: Out of system memory (ENOMEM).")
        elif isinstance(e, FileNotFoundError):
            print("Popen error with LBLRTM: C-shell executable '/bin/csh' was not found.")
        else:
            print(f"Popen OS error with LBLRTM ({e.errno}): {e.strerror}")
        return success, -999., -999., -999., -999., -999., tape3_info
        # Catches non-OS exceptions (e.g., KeyboardInterrupt or I/O errors during communication)
    except Exception as e:
        print(f"Unexpected error running LBLRTM: {e}")
        return success, -999., -999., -999., -999., -999., tape3_info

    # Now read in the baseline optical depths
    files1 = []
    files1 = files1 + sorted(glob.glob(lbldir+'.1/OD*'))
    if len(files1) != len(mlayerz)-1:
        print('This should not happen (.1/OD*) in compute_jacobian_interpol')
        print(f'DDT -- the number of lbloutput files is {len(files1):d} and the number of layers is {len(mlayerz)-1:d}')
        if verbose >= 3:
            print('The working LBLRTM directory is ' +lbldir+ '.1')
            print('    Here is LBLRTM command issued: '+command)
        if debug:
            wait = input('Stopping inside compute_jacobian_interpol to debug. Press enter to continue')
        else:
            return success, -999., -999., -999., -999., -999., tape3_info

    # Read in the spectral limits used in the TAPE3 file
    if(tape3_info['success'] == -2):
        tape3_info = Data_reads.tape6_min_max_wnum(lbldir+'.1',verbose=verbose)

    # Use the spectral spacing at x km AGL for the spectral spacing of the
    # layer optical depths.  Note that I did experiment with this value,
    # and using values higher in the atmosphere (a) cause the code to go
    # much slower, and (b) after applying the convolve_to_aeri, this value
    # at the maximum height where the jacobian is computed is more than sufficient.
    spec_resolution_ht = jac_maxht              # km AGL
    foo = np.where(zz >= spec_resolution_ht)[0]
    if len(foo) == 0:
        foo = np.array([len(files1)-1])
    if foo[0] >= len(files1):
        foo[0] = len(files1)-1
    if verbose >= 3:
        print('Using the spectral resolution at height ' + str(zz[foo[0]]) + ' km AGL')

    s0, v0 = LBLRTM_Functions.lbl_read(files1[foo[0]], do_load_data=True)
    v = np.copy(v0)
    od00 = np.zeros((len(files1),len(v)))

    # Compute the "interpolated" wavenumber grid
    npts = (np.max(v)-np.min(v))*npts_per_wnum
    iv = np.arange(npts+1)/npts_per_wnum+np.min(v)
    iod00 = np.zeros((len(files1), len(iv)))

    # Loop to read in the level optical depths
    for i in range(len(files1)):
        s0, v0 = LBLRTM_Functions.lbl_read(files1[i], do_load_data=True)
        od00[i, :] = np.interp(v, v0, s0)
        iod00[i, :] = np.interp(iv, v, od00[i, :])
        if fixt != 1:
            if i == 0:
                files2 = []
                files2 = files2 + sorted(glob.glob(lbldir+'.2/OD*'))
                if len(files2) != len(files1):
                    print('This should not happen (.2/OD*) in compute_jacobian_interpol')
                    if verbose >= 3:
                        print('The working LBLRTM directory is ' +lbldir+ '.2')
                    if debug:
                        wait = input('Stopping inside compute_jacobian_interpol to debug. Press enter to continue')
                    else:
                        return success, -999., -999., -999., -999., -999., tape3_info
                od11 = np.zeros((len(files1), len(v)))
                iod11 = np.zeros((len(files1), len(iv)))
            s0, v0 = LBLRTM_Functions.lbl_read(files2[i], do_load_data=True)
            od11[i, :] = np.interp(v, v0, s0)
            iod11[i, :] = np.interp(iv, v, od11[i, :])

        if fixwv != 1:
            if i == 0:
                files3 = []
                files3 = files3 + sorted(glob.glob(lbldir+'.3/OD*'))
                if len(files3) != len(files1):
                    print('This should not happen (.3/OD*) in compute_jacobian_interpol')
                    if verbose >= 3:
                        print('The working LBLRTM directory is ' +lbldir+ '.3')
                    if debug:
                        wait = input('Stopping inside compute_jacobian_interpol to debug. Press enter to continue')
                    else:
                        return success, -999., -999., -999., -999., -999., tape3_info
                od22 = np.zeros((len(files1),len(v)))
                iod22 = np.zeros((len(files1),len(iv)))
            s0, v0 = LBLRTM_Functions.lbl_read(files3[i], do_load_data=True)
            od22[i, :] = np.interp(v, v0, s0)
            iod22[i, :] = np.interp(iv, v, od22[i, :])

        if doco2 >= 1:
            if i == 0:
                files4 = []
                files4 = files4 + sorted(glob.glob(lbldir+'.4/OD*'))
                if len(files4) != len(files1):
                    print('This should not happen (.4/OD*) in compute_jacobian_interpol')
                    if verbose >= 3:
                        print('The working LBLRTM directory is ' +lbldir+ '.4')
                    if debug:
                        wait = input('Stopping inside compute_jacobian_interpol to debug. Press enter to continue')
                    else:
                        return success, -999., -999., -999., -999., -999., tape3_info
                od33 = np.zeros((len(files1),len(v)))
                iod33 = np.zeros((len(files1),len(iv)))
            s0, v0 = LBLRTM_Functions.lbl_read(files4[i], do_load_data=True)
            od33[i, :] = np.interp(v, v0, s0)
            iod33[i, :] = np.interp(iv, v, od33[i, :])

        if doch4 >= 1:
            if i == 0:
                files5 = []
                files5 = files5 + sorted(glob.glob(lbldir+'.5/OD*'))
                if len(files5) != len(files1):
                    print('This should not happen (.5/OD*) in compute_jacobian_interpol')
                    if verbose >= 3:
                        print('The working LBLRTM directory is ' +lbldir+ '.5')
                    if debug:
                        wait = input('Stopping inside compute_jacobian_interpol to debug. Press enter to continue')
                    else:
                        return success, -999., -999., -999., -999., -999., tape3_info
                od44 = np.zeros((len(files1),len(v)))
                iod44 = np.zeros((len(files1),len(iv)))
            s0, v0 = LBLRTM_Functions.lbl_read(files5[i], do_load_data=True)
            od44[i, :] = np.interp(v,v0,s0)
            iod44[i, :] = np.interp(iv, v, od44[i, :])

        if don2o >= 1:
            if i == 0:
                files6 = []
                files6 = files6 + sorted(glob.glob(lbldir+'.6/OD*'))
                if len(files6) != len(files1):
                    print('This should not happen (.6/OD*) in compute_jacobian_interpol')
                    if verbose >= 3:
                        print('The working LBLRTM directory is ' +lbldir+ '.6')
                    if debug:
                        wait = input('Stopping inside compute_jacobian_interpol to debug. Press enter to continue')
                    else:
                        return success, -999., -999., -999., -999., -999., tape3_info
                od55 = np.zeros((len(files1),len(v)))
                iod55 = np.zeros((len(files1),len(iv)))
            s0, v0 = LBLRTM_Functions.lbl_read(files6[i], do_load_data=True)
            od55[i, :] = np.interp(v, v0, s0)
            iod55[i, :] = np.interp(iv, v, od55[i, :])

    if verbose >= 3:
        print('    Computing the baseline radiance spectrum')

    wnum = np.copy(iv)
    gasod = np.copy(iod00)

    # Get the desired cloud absorption optical depth spectrum
    cldodvis = lwp * (3/2.) / reffl
    lcldodir = Other_functions.get_ir_cld_ods(sspl, cldodvis, wnum, reffl, 0.3)
    icldodir = Other_functions.get_ir_cld_ods(sspi, taui, wnum, reffi, 0.3)

    # Add the absorption cloud optical depth to the right altitude
    cldidx = np.where(zz >= cbh)[0]
    if len(cldidx) == 0:
        cldidx = len(zz)-2
    else:
        cldidx = np.max([cldidx[0]-1, 0])

    gasod0 = np.copy(gasod)     # Keep a copy of this for later
    gasod[cldidx, :] += lcldodir + icldodir

    # Compute the surface to layer transmission
    trans1 = np.copy(gasod)
    trans1[0, :] = 1
    for i in range(1, len(t)-1):
        trans1[i, :] = trans1[i-1, :] * np.exp(-gasod[i-1, :])

        # Compute the reflected radiance from the cloud.
        # I am using DDT's simple approximation for cloud reflectivity
        # that varies as a function of wavenumber and cloud optical depth.
        # I am assuming the surface is black and has the same temperature
        # as the lowest atmospheric layer. I need to account for the 2-way
        # attenuation by the atmosphere. Note that I am also assuming that
        # that the ammount of radiation emitted by the atmosphere and reflected
        # by the cloud is negligible

    reflection = Other_functions.cloud_reflectivity(wnum, cldodvis+taui)
    sfcrad = Calcs_Conversions.planck(wnum,t[0])
    cldrefrad = sfcrad*reflection*trans1[cldidx,:]*trans1[cldidx,:]

    # Compute the baseline radiance and optical depth of the lowest 1-m layer
    radc0 = Other_functions.radxfer(wnum, mlayert, gasod)
    radc0 += cldrefrad              # Add the cloud reflected radiance to this value
    o1mc0 = gasod[0,:] / ((mlayerz[1]-mlayerz[0])*1000) # The spectral optical depth for a 1-m layer
    t1mc0 = np.exp(-1*o1mc0)    # Convert this 1m optical depth to transmission
    bar   = np.where(np.isnan(t1mc0))[0]    # Really large ODs may have NaN transmission; set these to zero
    if len(bar) > 0:
        t1mc0[bar] = 0
    if((irs_type == 6) or (irs_type == 7)):
        tmp = Other_functions.convolve_to_refir(wnum, radc0, refir_delwnum)
        feh = Other_functions.convolve_to_refir(wnum, t1mc0, refir_delwnum)
    else:
        tmp = Other_functions.convolve_to_aeri(wnum, radc0)
        feh = Other_functions.convolve_to_aeri(wnum, t1mc0)
    Kwnum = np.copy(tmp['wnum'])
    radc0 = np.copy(tmp['spec'])
    t1mc0 = np.copy(feh['spec'])    # This is the convolved transmission
    bar   = np.where(t1mc0 <= 0)[0]
    if len(bar) > 0:
        t1mc0[bar] = 1e-20
    o1mc0 = -1*np.log(t1mc0)        # This is the convolved optical depth
    if doapoJac:
        radc0 = np.real(Other_functions.apodizer(radc0,0))
        t1mc0 = np.real(Other_functions.apodizer(t1mc0,0))

    if verbose >= 2:
        print('Computing the Jacobian using the interpol method')

    Kij = np.zeros((len(radc0),2*k+4+3*3))

    # Compute the temperature perturbation
    # Note I'm changing both the optical depth spectrum for the layer
    # (which has the impact on the temperature dependence of the strength/
    # width of the lines and on the continuum strength) as well as the
    # emission temperature of the layer
    if fixt != 1:
        if verbose >= 3:
            print('Computing Jacobian for temperature')
        for kk in range(k):
            if zz[kk] > jac_maxht:
                Kij[:,kk] = 0.
            else:
                gasod = np.copy(iod00)              # Take baseline monochromatic ODs
                gasod[kk,:] = np.copy(iod11[kk,:])          # Insert in the OD from perturbed temp run

                # Add the absorption cloud optical depth to the right altitude
                gasod[cldidx,:] += lcldodir + icldodir

                # Compute the perturbed radiance
                # Remember to perturb the temperature profile to get
                # the emission temeprature correct

                t0 = np.copy(mlayert)
                t0[kk] += tpert
                radc1 = Other_functions.radxfer(wnum, t0, gasod)
                radc1 += cldrefrad

                if((irs_type == 6) or (irs_type == 7)):
                    tmp = Other_functions.convolve_to_refir(wnum, radc1, refir_delwnum)
                else:
                    tmp = Other_functions.convolve_to_aeri(wnum, radc1)
                radc1 = np.copy(tmp['spec'])

                if doapoJac:
                    radc1 = np.real(Other_functions.apodizer(radc1,0))

                if kk == 0:
                    mult = 0.5
                else:
                    mult = 1.0

                mult = 1.0         # DDT -- I will keep the multiplier at 1 until I test it
                Kij[:,kk] = mult * (radc1-radc0)/tpert

    else:
        if verbose >= 3:
            print('Temperatue jacobian set to zero (fixed T profile)')
        Kij[:,0:k] = 0.


    # Compute the water vapor perturbation
    if fixwv != 1:
        if verbose >= 3:
            print('Computing Jacobian for water vapor')
        for kk in range(k):
            if zz[kk] > jac_maxht:
                Kij[:,kk+k] = 0.
            else:
                gasod = np.copy(iod00)           # Take baseline ODs
                gasod[kk,:] = np.copy(iod22[kk,:])        # Insert in the OD from perturbed H2O run

                # Add the absorption cloud optical depth to the right altitude
                gasod[cldidx,:] += lcldodir + icldodir

                # Compute the perturbed radiance
                radc1 = Other_functions.radxfer(wnum, mlayert,  gasod)
                radc1 += cldrefrad

                if((irs_type == 6) or (irs_type == 7)):
                    tmp = Other_functions.convolve_to_refir(wnum, radc1, refir_delwnum)
                else:
                    tmp = Other_functions.convolve_to_aeri(wnum, radc1)
                radc1 = np.copy(tmp['spec'])

                if doapoJac:
                    radc1 = np.real(Other_functions.apodizer(radc1,0))
                if kk == 0:
                    mult = 0.5
                else:
                    mult = 1.0
                Kij[:,kk+k] = mult * ( (radc1-radc0) / (w[kk]*h2opert - w[kk]) )
    else:
        if verbose >= 3:
            print('Water vapor jacobian set to zero (fixed WV profile)')
        Kij[:,k:2*k] = 0.

    # Compute the carbon dioxide perturbation
    if doco2 >= 1:
        if verbose >= 3:
            print('Computing Jacobian for carbon dioxide')

        # Compute the sensitivity to the first coefficient
        gasod = np.copy(iod33)                  # Will use the entire perturbed CO2 dataset
        gasod[cldidx,:] += lcldodir + icldodir

        # Compute the perturbed radiance
        radc1 = Other_functions.radxfer(wnum, mlayert, gasod)
        radc1 += cldrefrad
        if((irs_type == 6) or (irs_type == 7)):
            tmp = Other_functions.convolve_to_refir(wnum, radc1, refir_delwnum)
        else:
            tmp = Other_functions.convolve_to_aeri(wnum, radc1)
        radc1 = np.copy(tmp['spec'])
        if doapoJac:
            radc1 = np.real(Other_functions.apodizer(radc1,0))
        Kij[:,2*k+4] = (radc1-radc0) / co2pert

        # Compute the sensitivity to the 2nd coefficient. Do this by
        # weighting the optical depth profile as a function of height

        c0 = np.copy(co2)
        c0[1] += co2pert
        co2prof3 = Other_functions.trace_gas_prof(doco2, zz, c0)
        weight = (co2prof3-co2prof)/(co2prof2-co2prof)
        gasod = np.copy(iod33)
        for j in range(len(weight)-1):
            gasod[j,:] = iod33[j,:]*weight[j] + (1-weight[j])*iod00[j,:]
        gasod[cldidx,:] += lcldodir + icldodir

        # Compute the perturbed radiance
        radc1 = Other_functions.radxfer(wnum, mlayert, gasod)
        radc1 += cldrefrad
        if((irs_type == 6) or (irs_type == 7)):
            tmp = Other_functions.convolve_to_refir(wnum, radc1, refir_delwnum)
        else:
            tmp = Other_functions.convolve_to_aeri(wnum, radc1)
        radc1 = np.copy(tmp['spec'])

        if doapoJac:
            radc1 = np.real(Other_functions.apodizer(radc1,0))
        Kij[:,2*k+5] = (radc1-radc0) / co2pert

        # Compute the sensitivity to the 3rd coefficient. Do this by
        # weighting the optical depth profile as a function of height

        if (doco2 == 2) & (fix_co2_shape == 1):
            Kij[:,2*k+6] = 0             # By making the Jacobian zero, then retrieval will not change this value
        elif doco2 == 2:
	    # This bit of code should only be executed if we are using the expoential trace gas profile shape; otherwise, this jacobian is zero
            print("Logic error within compute_jacobian_irs_interpol() -- this code section was supposed to be disabled -- aborting")
            sys.exit()
            c0 = np.copy(co2)
            c0[2] -= 2.
            c0[2] = -np.exp(np.log(-co2[2])-0.7)    # This makes the change in the coefficient more reasonable

            # If the 2nd coefficient is zero, the Jacobian will not show any sensitivity
            # to the shape of the profile, and thus it can not change the shape. This is a
            # problem, so I will allow a bit of sensitivity to occur.
            if np.abs(c0[1]) < co2pert/10.:
                if c0[1] < 0:
                    c0[1] = -co2pert/10.
                else:
                    c0[1] = co2pert/10.

            co2prof3 = Other_functions.trace_gas_prof(doco2, zz, c0)
            weight = (co2prof3-co2prof)/(co2prof2-co2prof)
            gasod = np.copy(iod33)
            for j in range(len(weight)-1):
                gasod[j,:] = iod33[j,:]*weight[j] + (1-weight[j])*iod00[j,:]

            gasod[cldidx,:] += lcldodir + icldodir

            # Compute the perturbed radiance
            radc1 = Other_functions.radxfer(wnum, mlayert, gasod)
            radc1 += cldrefrad
            if((irs_type == 6) or (irs_type == 7)):
                tmp = Other_functions.convolve_to_refir(wnum, radc1, refir_delwnum)
            else:
                tmp = Other_functions.convolve_to_aeri(wnum, radc1)
            radc1 = np.copy(tmp['spec'])

            if doapoJac:
                radc1 = np.real(Other_functions.apodizer(radc1,0))
            Kij[:,2*k+6] = (radc1-radc0) / (c0[2] - co2[2])
        else:
            Kij[:,2*k+6] = 0
    else:
        if verbose >= 3:
            print('Carbon dioxide jacobian set to zero (fixed CO2 profile)')
        Kij[:,2*k+4:2*k+7] = 0.

    # Compute the methane perturbation
    if doch4 >= 1:
        if verbose >= 3:
            print('Computing Jacobian for methane')

        # Compute the sensitivity to the first coefficient
        gasod = np.copy(iod44)                 # Will use the entire perturbed CH4 data
        gasod[cldidx,:] += lcldodir +icldodir

        # Compute the perturbed radiance
        radc1 = Other_functions.radxfer(wnum, mlayert, gasod)
        radc1 += cldrefrad
        if((irs_type == 6) or (irs_type == 7)):
            tmp = Other_functions.convolve_to_refir(wnum, radc1, refir_delwnum)
        else:
            tmp = Other_functions.convolve_to_aeri(wnum, radc1)
        radc1 = np.copy(tmp['spec'])

        if doapoJac:
            radc1 = np.real(Other_functions.apodizer(radc1,0))
        Kij[:,2*k+7] = (radc1-radc0) / ch4pert

        # Compute the sensitivity to the 2nd coefficient. Do this by
        # weighting the optical depth profile as a function of height
        c0 = np.copy(ch4)
        c0[1] += ch4pert
        ch4prof3 = Other_functions.trace_gas_prof(doch4, zz, c0)
        weight = (ch4prof3-ch4prof)/(ch4prof2-ch4prof)
        gasod = np.copy(iod44)
        for j in range(len(weight)-1):
            gasod[j,:] = iod44[j,:]*weight[j] + (1-weight[j])*iod00[j,:]

        gasod[cldidx,:] += lcldodir + icldodir

        # Compute the perturbed radiance
        radc1 = Other_functions.radxfer(wnum, mlayert, gasod)
        radc1 += cldrefrad
        if((irs_type == 6) or (irs_type == 7)):
            tmp = Other_functions.convolve_to_refir(wnum, radc1, refir_delwnum)
        else:
            tmp = Other_functions.convolve_to_aeri(wnum, radc1)
        radc1 = np.copy(tmp['spec'])

        if doapoJac:
            radc1 = np.real(Other_functions.apodizer(radc1,0))
        Kij[:,2*k+8] = (radc1-radc0) / ch4pert

        # Compute the sensitivity to the 3rd coefficient. Do this by
        # weighting the optical depth profile as a function of height
        if ((doch4 == 2) & (fix_ch4_shape == 1)):
            Kij[:,2*k+9] = 0         # By making the Jacobian zero, then the retrieval will not change this value
        elif doch4 == 2:
            c0 = np.copy(ch4)
            c0[2] -= 2.
            # If the 2nd coefficient is zero, then the Jacobian will not show any sensitivity
            # to the shape of the profile, and thus it cannont change the shape. This is a
            # problem, so I will allow a bit of sensitivity to occur
            if (np.abs(c0[1]) < ch4pert/10.):
                if c0[1] < 0:
                    c0[1] = -ch4pert/10.
                else:
                    c0[1] = ch4pert/10.

            ch4prof3 = Other_functions.trace_gas_prof(doch4,zz,c0)
            weight = (ch4prof3-ch4prof)/(ch4prof2-ch4prof)
            gasod = np.copy(iod44)
            for j in range(len(weight)-1):
                gasod[j,:] = iod44[j,:]*weight[j] + (1-weight[j])*iod00[j,:]

            gasod[cldidx,:] += lcldodir + icldodir

            # Compute the perturbed radiance
            radc1 = Other_functions.radxfer(wnum, mlayert, gasod)
            radc1 += cldrefrad
            if((irs_type == 6) or (irs_type == 7)):
                tmp = Other_functions.convolve_to_refir(wnum, radc1, refir_delwnum)
            else:
                tmp = Other_functions.convolve_to_aeri(wnum, radc1)
            radc1 = np.copy(tmp['spec'])

            if doapoJac:
                radc1 = np.real(Other_functions.apodizer(radc1,0))
            Kij[:,2*k+9] = (radc1-radc0) / (c0[2] - ch4[2])
        else:
            Kij[:,2*k+9] = 0
    else:
        if verbose >= 3:
            print('Methane jacobian set to zero (fixed CH4 value)')
        Kij[:,2*k+7:2*k+10] = 0.

    # Compute the nitrous oxide perturbation
    if don2o >= 1:
        if verbose >= 3:
            print('Computing Jacobian for nitrous oxide')

        # Compute the sensitivity to the first coefficient
        gasod = np.copy(iod55)
        gasod[cldidx,:] += lcldodir + icldodir

        # Compute the perturbed radiance
        radc1 = Other_functions.radxfer(wnum, mlayert, gasod)
        radc1 += cldrefrad
        if((irs_type == 6) or (irs_type == 7)):
            tmp = Other_functions.convolve_to_refir(wnum, radc1, refir_delwnum)
        else:
            tmp = Other_functions.convolve_to_aeri(wnum, radc1)
        radc1 = np.copy(tmp['spec'])

        if doapoJac:
            radc1 = np.real(Other_functions.apodizer(radc1,0))
        Kij[:,2*k+10] = (radc1-radc0) / n2opert

        # Compute the sensitivity to the 2nd coefficient. Do this by
        # weighting the optical depth profile as a function of height
        c0 = np.copy(n2o)
        c0[1] += n2opert
        n2oprof3 = Other_functions.trace_gas_prof(don2o, zz, c0)
        weight = (n2oprof3-n2oprof)/(n2oprof2-n2oprof)
        gasod = np.copy(iod55)
        for j in range(len(weight)-1):
            gasod[j,:] = iod55[j,:]*weight[j] + (1-weight[j])*iod00[j,:]

        gasod[cldidx,:] += lcldodir + icldodir

        # Compute the perturbed radiance
        radc1 = Other_functions.radxfer(wnum, mlayert, gasod)
        radc1 += cldrefrad
        if((irs_type == 6) or (irs_type == 7)):
            tmp = Other_functions.convolve_to_refir(wnum, radc1, refir_delwnum)
        else:
            tmp = Other_functions.convolve_to_aeri(wnum, radc1)
        radc1 = np.copy(tmp['spec'])

        if doapoJac:
            radc1 = np.real(Other_functions.apodizer(radc1,0))
        Kij[:,2*k+11] = (radc1-radc0) / n2opert

        # Compute the sensitivity to the 3rd coefficient. Do this by weighting
        # the optical depth profile as a function of height
        if ((don2o == 2) & (fix_n2o_shape == 1)):
            Kij[:,2*k+12] = 0                 # By making the Jacobian zero, then retrieval will not change this value
        elif don2o == 2:
            c0 = np.copy(n2o)
            c0[2] -= 2.
            # If the 2nd coefficient is zero, then the Jacobian will not show any sensitivity
            # to the shape of the profile, and thus it cannont change the shape. This is a
            # problem, so I will allow a bit of sensitivity to occur
            if np.abs(c0[1]) < n2opert/10.:
                if c0[1] < 0:
                    c0[1] = -n2opert/10.
                else:
                    c0[1] = n2opert/10.

            n2oprof3 = Other_functions.trace_gas_prof(don2o, zz, c0)
            weight = (n2oprof3-n2oprof)/(n2oprof2-n2oprof)
            gasod = np.copy(iod55)
            for j in range(len(weight)-1):
                gasod[j,:] = iod55[j,:]*weight[j] + (1-weight[j])*iod00[j,:]

            gasod[cldidx,:] += lcldodir + icldodir

            # Compute the perturbed radiance
            radc1 = Other_functions.radxfer(wnum, mlayert, gasod)
            radc1 += cldrefrad
            if((irs_type == 6) or (irs_type == 7)):
                tmp = Other_functions.convolve_to_refir(wnum, radc1, refir_delwnum)
            else:
                tmp = Other_functions.convolve_to_aeri(wnum, radc1)
            radc1 = np.copy(tmp['spec'])

            if doapoJac:
                radc1 = np.real(Other_functions.apodizer(radc1,0))
            Kij[:,2*k+12] = (radc1-radc0) / (c0[2] - n2o[2])
        else:
            Kij[:,2*k+12] = 0
    else:
        if verbose >= 3:
            print('Nitrous oxide jacobian set to zero (fixed N2O value)')
        Kij[:,2*k+10:2*k+13] = 0.

    # Compute the liquid cloud property perturbation
    if fixlcld != 1:
        if verbose >= 3:
            print('Computing Jacobian for the liquid cloud properties (LWP and ReffL)')

        # Get the desired cloud absorption optical depth spectrum
        lwppert = 2.          # The additive LWP perturbation [g/m2]
        reffpert = 0.5        # The additive ReffL perturbation [um]

        # Compute the perturbed radiance for LWP
        cldodvis = (lwp+lwppert)*(3/2.) / reffl
        cldodir = Other_functions.get_ir_cld_ods(sspl,cldodvis,wnum,reffl,0.3)
        gasod = np.copy(gasod0)        # Using the original profile gaseous optical depth data
        gasod[cldidx,:] += cldodir +icldodir

        radc1 = Other_functions.radxfer(wnum, mlayert, gasod)
        radc1 += cldrefrad             # Not changing cloud reflectivity component here
        if((irs_type == 6) or (irs_type == 7)):
            tmp = Other_functions.convolve_to_refir(wnum, radc1, refir_delwnum)
        else:
            tmp = Other_functions.convolve_to_aeri(wnum, radc1)
        radc1 = np.copy(tmp['spec'])

        if doapoJac:
            radc1 = np.real(Other_functions.apodizer(radc1,0))

        # Compute the perturbed radiance for ReffL
        cldodvis = lwp*(3/2.)/(reffl+reffpert)
        cldodir = Other_functions.get_ir_cld_ods(sspl,cldodvis,wnum,reffl+reffpert,0.3)
        gasod = np.copy(gasod0)        # Using the original profile gaseous optical depth data
        gasod[cldidx,:] += cldodir + icldodir

        radc2 = Other_functions.radxfer(wnum, mlayert, gasod)
        radc2 += cldrefrad             # Not changing cloud reflectivity component here
        if((irs_type == 6) or (irs_type == 7)):
            tmp = Other_functions.convolve_to_refir(wnum, radc2, refir_delwnum)
        else:
            tmp = Other_functions.convolve_to_aeri(wnum, radc2)
        radc2 = np.copy(tmp['spec'])

        if doapoJac:
            radc2 = np.real(Other_functions.apodizer(radc2,0))

        Kij[:,2*k] = (radc1-radc0) / lwppert
        Kij[:,2*k+1] = (radc2-radc0) / reffpert
    else:
        if verbose >= 3:
            print('Cloud jacobian set to zero (fixed LWP and ReffL values)')
        Kij[:,2*k] = 0.
        Kij[:,2*k+1] = 0.

    # Compute the ice cloud property perturbation
    if fixicld != 1:
        if verbose >= 3:
            print('Computing Jacobian for the ice cloud properties (TauI and ReffI)')

        # Get the desired cloud absorption optical depth spectrum
        taupert = 0.5            # The additive perturbation for ice optical depth
        reffpert = 1.0           # The additive ReffI perturbation [um]

        # Compute the perturbed radiance for tau ice (TauI)
        cldodvis = taui+taupert
        cldodir = Other_functions.get_ir_cld_ods(sspi, cldodvis, wnum, reffi, 0.3)
        gasod = np.copy(gasod0)                 # Using the original profile gaseous optical depth
        gasod[cldidx,:] += cldodir + lcldodir

        radc1 = Other_functions.radxfer(wnum, mlayert, gasod)
        radc1 += cldrefrad             # Not changing cloud reflectivity component here
        if((irs_type == 6) or (irs_type == 7)):
            tmp = Other_functions.convolve_to_refir(wnum, radc1, refir_delwnum)
        else:
            tmp = Other_functions.convolve_to_aeri(wnum, radc1)
        radc1 = np.copy(tmp['spec'])

        if doapoJac:
            radc1 = np.real(Other_functions.apodizer(radc1,0))

        # Compute the perturbed radiance for ReffI
        cldodvis = taui
        cldodir = Other_functions.get_ir_cld_ods(sspi, cldodvis, wnum, reffi+reffpert, 0.3)
        gasod = np.copy(gasod0)                 # Using the original profile gaseous optical depth
        gasod[cldidx,:] += cldodir + lcldodir

        radc2 = Other_functions.radxfer(wnum, mlayert, gasod)
        radc2 += cldrefrad                     # Not changing cloud reflectivity component here
        if((irs_type == 6) or (irs_type == 7)):
            tmp = Other_functions.convolve_to_refir(wnum, radc2, refir_delwnum)
        else:
            tmp = Other_functions.convolve_to_aeri(wnum, radc2)
        radc2 = np.copy(tmp['spec'])

        if doapoJac:
            radc2 = np.real(Other_functions.apodizer(radc2,0))

        Kij[:,2*k+2] = (radc1-radc0) / taupert
        Kij[:,2*k+3] = (radc2-radc0) / reffpert
    else:
        if verbose >= 3:
            print('Cloud jacobian set to zero (fixed TauI and ReffI values)')
        Kij[:,2*k+2] = 0.
        Kij[:,2*k+3] = 0.

    # Cut down the Jacobian to match this spectral interval
    wpad = 5  # TODO - is this supposed to be 5?
    foo = np.where((np.min(awnum)+wpad <= Kwnum) & (Kwnum <= np.max(awnum)-wpad))[0]
    Kij = Kij[foo,:]
    radc0 = radc0[foo]
    wnumc = Kwnum[foo]

    # The forward calculation above is not as accurate as it could be, which
    # will hammer the retrieval. Improve on its accuracy here by using the original
    # monochromatic optical depths (whereas the above calculations used the 
    # reduced spectral resolution optical depths)

    if verbose >= 3:
        print('Forward model F(Xn) using radxfer and assuming clouds')

    gasod = np.copy(od00)    # this is only the gaseous optical depths
    totod = np.copy(od00)    # this will be the total (gas + cloud) optical depths
    lcldodir = np.interp(v,wnum,lcldodir)
    icldodir = np.interp(v,wnum,icldodir)
    totod[cldidx,:] += lcldodir + icldodir

        # Compute the surface to layer transmission
    trans1 = np.copy(totod)
    trans1[0,:] = 1
    for i in range(1,len(t)-1):
        trans1[i,:] = trans1[i-1,:]*np.exp(-totod[i-1,:])

        # Compute the reflected radiance from the cloud.
        # I am using my simple approximation for cloud reflectivity
        # that varies as a function of wavenumber and cloud optical
        # depth.  I am assuming the surface is black and has the same
        # temperature as the lowest atmospheric layer.  I need to
        # account for the 2-way attenution by the atmosphere.  Note
        # that I am also assuming that the amount of radiation emitted
        # by the atmosphere and reflected by the cloud is negligible.
    reflection = Other_functions.cloud_reflectivity(v, cldodvis+taui)
    sfcrad = Calcs_Conversions.planck(v,t[0])
    cldrefrad = sfcrad * reflection * trans1[cldidx,:] * trans1[cldidx,:]

        # Compute the baseline radiance
    radv = Other_functions.radxfer(v, mlayert, totod)
    radv += cldrefrad
    if((irs_type == 6) or (irs_type == 7)):
        bar = Other_functions.convolve_to_refir(v, radv, refir_delwnum)
    else:
        bar = Other_functions.convolve_to_aeri(v, radv)
    bwnum = np.copy(bar['wnum'])
    brad = np.copy(bar['spec'])
    if doapoFor:
        brad = np.real(Other_functions.apodizer(brad,0))

    # Compute the downwelling clear sky radiance, and the downwelling radiance for each 
    # height, assuming that the "cloud" is opaque at each height (for the MLEV later)
    foo = np.where(mlayerz <= jac_maxht+sfcz)[0]
    radclear = Other_functions.radxfer(v, mlayert, gasod)
    radBcld  = np.zeros((len(foo),len(v)))
    for k in range(len(foo)):
        tmpod = np.copy(gasod)
        tmpod[k,:] = 1e6        # Make this layer have super high OD
        tmprad = Other_functions.radxfer(v, mlayert, tmpod)
        radBcld[k,:] = tmprad

        # Now convolve the radiances from the MLEV step to the IRS instrument function
    if((irs_type == 6) or (irs_type == 7)):
        bar = Other_functions.convolve_to_refir(v, radclear, refir_delwnum)
    else:
        bar = Other_functions.convolve_to_aeri(v, radclear)
    bradclear = np.copy(bar['spec'])
    if doapoFor:
        bradclear = np.real(Other_functions.apodizer(bradclear,0))
    bradBcld = np.zeros((len(foo),len(bradclear)))
    for k in range(len(foo)):
        if((irs_type == 6) or (irs_type == 7)):
            bar = Other_functions.convolve_to_refir(v, radBcld[k,:], refir_delwnum)
        else:
            bar = Other_functions.convolve_to_aeri(v, radBcld[k,:])
        bradBcld[k,:] = np.copy(bar['spec'])
        if doapoFor:
            bradBcld[k,:] = np.real(Other_functions.apodizer(bradBcld[k,:],0))

        # Now cut the radiance down; this is the forward calculation
    foo = np.where((np.min(wnumc)-0.1 <= bwnum) & (bwnum <= np.max(wnumc)+0.1))[0]
    if ((len(foo) != len(wnumc)) | (np.abs(np.min(wnumc)-np.min(bwnum[foo])) > 0.1)):
        print('PROBLEM inside compute_jacobian_interpol -- wavenumber do not match')
        return success, -999., -999., -999., -999., -999., tape3_info
    FXn   = np.copy(brad[foo])
    o1mc0 = o1mc0[foo]
    bradclear = bradclear[foo]
    bradBcld  = bradBcld[:,foo]
    mlev = {'wnum':wnumc, 'radclear':bradclear, 'radBcld':bradBcld, 'maxht':jac_maxht}

    # We don't want spectral elements in very opaque channels have any influence
    # on the retrieval, as these may be representing absorption within the instrument
    # Find these channels, and set the Jacobian for them to zero
    od1mthres = vip['irs_1m_od_thres']
    foo = np.where(o1mc0 > od1mthres)[0]
    if verbose >= 2:
        print(f"      Within IRS forward calc: {len(t1mc0):d} total spectral points of which {len(foo):d} were flagged with having high 1m optical depth above {od1mthres:f}")
    if len(foo) > 0:
        Kij[foo,:] = 0
        FXn[foo] = -777     # Just a random flag value for the output netCDF file

    # Capture the total time and return
    etime = datetime.now()
    totaltime = (etime-stime).total_seconds()
    if verbose >= 3:
        print(' It took ' + str(totaltime) + ' s to compute Jacobian (interpol)')
    success = 1

    return success, Kij, FXn, wnumc, totaltime, mlev, tape3_info
################################################################################
# This function performs the forward model calculation and computes the jacobian
# for the microwave radiometer. It is designed very similarly to
# compute_jacobian_finitediff()
################################################################################

def compute_jacobian_microwave_finitediff(Xn, p, z, freq, cbh, vip, workdir,
                monortm_tfile, monortm_exec, fixt, fixwv, fixlcld, jac_maxht,
                stdatmos, sfc_alt, verbose):

    monortm_outputfile = workdir+'/monortm_output.txt'      # A temporary output file
    flag = 0               # Failure
    k = len(z)
    t = np.copy(Xn[0:k])          # degC
    w = np.copy(Xn[k:2*k])        # g/kg
    lwp = Xn[2*k]                 # g/m2
    cth = cbh + 0.300             # km; define the cloud top at x m above the could base

    if(sfc_alt == None):
        sfcz=0
    else:
        sfcz = sfc_alt / 1000.

    # Allocate space for the Jacobian and forward calculation
    Kij = np.zeros((len(freq),len(Xn)))
    FXn = np.zeros(len(freq))

    if verbose >= 2:
        print('Computing the Jacobian using the finite diff method with MonoRTM')
    stime = datetime.now()

    # Perform the baseline calculation
    u = Calcs_Conversions.w2rh(w, p, t, 0) * 100
    Other_functions.write_arm_sonde_file((z+sfcz)*1000, p, t, u, workdir+'/'+monortm_tfile, silent = True)
    command = ('rm -f '+monortm_outputfile+' ; '+monortm_exec + ' ' + monortm_tfile + ' {:3.1f} {:8.2f} {:6.3f} {:6.3f}'.format(1.0, lwp, cbh, cth)+' > '+monortm_outputfile)
    a = LBLRTM_Functions.run_monortm(command, freq, z, stdatmos, monortm_outputfile)
    if a['status'] == 0:
        print('Problem with MonoRTM calc 0')
        return flag, -999., -999., -999.
    FXn = np.copy(a['tb'])

    # Compute the temperature perturbation
    if fixt != 1:
        delta = 1.0        # Additive perturbation of 1 K
        if verbose >= 3:
            print('Computing Jacobian for temperature')
        for kk in  range(k):
            if z[kk] > jac_maxht:
                Kij[:,kk] = 0.
            else:
                t0 = np.copy(t)
                t0[kk] += delta
                u = Calcs_Conversions.w2rh(w, p, t0, 0)
                Other_functions.write_arm_sonde_file((z+sfcz)*1000, p, t0, u, workdir+'/'+monortm_tfile, silent=True)
                command = ('rm -f '+monortm_outputfile+' ; '+monortm_exec + ' ' + monortm_tfile + ' {:3.1f} {:8.2f} {:6.3f} {:6.3f}'.format(1.0, lwp, cbh, cth)+' > '+monortm_outputfile)
                b = LBLRTM_Functions.run_monortm(command, freq, z, stdatmos, monortm_outputfile)
                if b['status'] == 0:
                    print('Problem with MonoRTM calc 1')
                    return flag, -999., -999., -999.
                FXp = np.copy(b['tb'])
                Kij[:,kk] = (FXp-FXn) / (t0[kk] - t[kk])
    else:
        if verbose >= 3:
            print('Temperature jacobian set to zero (fixed T profile)')
        Kij[:,0:k] = 0.

    # Compute the water vapor perturbation
    if fixwv != 1:
        delta = 0.99           # Multiplicative perturbation of 1%
        if verbose >= 3:
            print('Computing Jacobian for water vapor')
        for kk in range(k):
            if z[kk] > jac_maxht:
                Kij[:,kk+k] = 0.
            else:
                w0 = np.copy(w)
                w0[kk] += delta
                u = Calcs_Conversions.w2rh[w0, p, t, 0] * 100
                Other_functions.write_arm_sonde_file((z+sfcz)*1000, p, t, u, workdir+'/'+monortm_tfile, silent = True)
                command = ('rm -f '+monortm_outputfile+' ; '+monortm_exec + ' ' + monortm_tfile + ' {:3.1f} {:8.2f} {:6.3f} {:6.3f}'.format(1.0, lwp, cbh, cth)+' > '+monortm_outputfile)
                b = LBLRTM_Functions.run_monortm(command, freq, z, stdatmos, monortm_outputfile)
                if b['status'] == 0:
                    print('Problem with MonoRTM calc 2')
                    return flag, -999., -999., -999.
                FXp = np.copy(b['tb'])
                Kij[:,kk+k] = (FXp-FXn) / (w0[kk] - w[kk])

    else:
        if verbose >= 3:
            print('Water vapor jacobian set to zero (fixed WV profile)')
        Kij[:,k:2*k] = 0.

    # Compute the Jacobian forthe perturbation of LWP
    u = Calcs_Conversions.w2rh(w, p, t, 0) * 100
    Other_functions.write_arm_sonde_file((z+sfcz)*1000, p, t, u, workdir+'/'+monortm_tfile, silent = True)
    lwpp = lwp + 25.
    command = ('rm -f '+monortm_outputfile+' ; '+monortm_exec + ' ' + monortm_tfile + ' {:3.1f} {:8.2f} {:6.3f} {:6.3f}'.format(1.0, lwpp, cbh, cth)+' > '+monortm_outputfile)
    b = LBLRTM_Functions.run_monortm(command, freq, z, stdatmos, monortm_outputfile)
    if b['status'] == 0:
        print('Problem with MonoRTM calc 3')
    FXp = np.copy(b['tb'])
    Kij[:,2*k] = (FXp - FXn) / (lwpp - lwp)

    # Capture the execution time
    etime = datetime.now()
    totaltime = (etime-stime).total_seconds()
    if verbose >= 3:
        print(' It took ' + str(totaltime) + ' s to compute Jacobian (finite diffs)')

    flag = 1

    return flag, Kij, FXn, totaltime

################################################################################
# This function performs the forward model calculation and computes the jacobian
# for the microwave radiometer. It is designed very similarly to
# compute_jacobian_3method()
################################################################################
def compute_jacobian_microwave_3method(Xn, p, z, freq, cbh, vip, workdir,
                monortm_tfile, monortm_exec, fixt, fixwv, fixlcld, jac_maxht,
                stdatmos, sfc_alt, verbose):

    monortm_outputfile = workdir+'/monortm_output.txt'      # A temporary output file
    flag = 0               # Failure
    k = len(z)
    t = np.copy(Xn[0:k])          # degC
    w = np.copy(Xn[k:2*k])        # g/kg
    lwp = Xn[2*k]                 # g/m2
    cth = cbh + 0.300             # km; define the cloud top at x m above the could base

    if(sfc_alt == None):
        sfcz=0
    else:
        sfcz = sfc_alt / 1000.

    # Allocate space for the Jacobian and forward calculation
    Kij = np.zeros((len(freq),len(Xn)))
    FXn = np.zeros(len(freq))

    if verbose >= 2:
        print('Computing the MWR-zenith Jacobian using the 3method with MonoRTM')

    stime = datetime.now()
    # Perform the baseline calculation
    u = Calcs_Conversions.w2rh(w, p, t, 0) * 100
    Other_functions.write_arm_sonde_file((z+sfcz)*1000, p, t, u, workdir+'/'+monortm_tfile, silent = True)
    command = ('rm -f '+monortm_outputfile+' ; '+monortm_exec + ' ' + monortm_tfile + ' {:3.1f} {:8.2f} {:6.3f} {:6.3f}'.format(1.0, lwp, cbh, cth)+' > '+monortm_outputfile)
    a = LBLRTM_Functions.run_monortm(command, freq, z, stdatmos, monortm_outputfile)
    if a['status'] == 0:
        print('Problem with MonoRTM calc 0')
        return flag, -999., -999., -999.
    FXn = np.copy(a['tb'])

    if fixt != 1:
        tpert = 1.0
        t0 = t + tpert
        u = Calcs_Conversions.w2rh(w, p, t0, 0) * 100
        Other_functions.write_arm_sonde_file((z+sfcz)*1000, p, t0, u, workdir+'/'+monortm_tfile, silent = True)
        command = ('rm -f '+monortm_outputfile+' ; '+monortm_exec + ' ' + monortm_tfile + ' {:3.1f} {:8.2f} {:6.3f} {:6.3f}'.format(1.0, lwp, cbh, cth)+' > '+monortm_outputfile)
        b = LBLRTM_Functions.run_monortm(command, freq, z, stdatmos, monortm_outputfile)
        if b['status'] == 0:
            print('Problem with MonoRTM calc 1')
            return flag, -999., -999., -999.
    else:
        command = 'sleep 1'

    if fixwv != 1:
	# Set the size of the perturbation as a crude function of the WVMR at the surface
        h2opert = np.interp(w[0],[0.01,0.1,2],[0.75,0.90,0.99])

        w0 = w*h2opert
        u = Calcs_Conversions.w2rh(w0, p, t, 0) * 100
        Other_functions.write_arm_sonde_file((z+sfcz)*1000, p, t, u, workdir+'/'+monortm_tfile, silent = True)
        command = ('rm -f '+monortm_outputfile+' ; '+monortm_exec + ' ' + monortm_tfile + ' {:3.1f} {:8.2f} {:6.3f} {:6.3f}'.format(1.0, lwp, cbh, cth)+' > '+monortm_outputfile)
        c = LBLRTM_Functions.run_monortm(command, freq, z, stdatmos, monortm_outputfile)
        if c['status'] == 0:
            print('Problem with MonoRTM calc 2')
            return flag, -999., -999., -999.
    else:
        command = 'sleep 1'

    if fixlcld != 1:
        lwpp = lwp + 25.
        u = Calcs_Conversions.w2rh(w, p, t, 0) * 100
        Other_functions.write_arm_sonde_file((z+sfcz)*1000, p, t, u, workdir+'/'+monortm_tfile, silent = True)
        command = ('rm -f '+monortm_outputfile+' ; '+monortm_exec + ' ' + monortm_tfile + ' {:3.1f} {:8.2f} {:6.3f} {:6.3f}'.format(1.0, lwpp, cbh, cth)+' > '+monortm_outputfile)
        d = LBLRTM_Functions.run_monortm(command, freq, z, stdatmos, monortm_outputfile)
        if d['status'] == 0:
            print('Problem with MonoRTM calc 3')
            return flag, -999., -999., -999.
    else:
        command = 'sleep 1'

    # Captue the different optical depths into simple matrices
    od0 = np.copy(a['od'].T)                  # Baseline run
    if fixt != 1:
        od1 = np.copy(b['od'].T)              # Run with entire temperature profile perturbed
    if fixwv != 1:
        od2 = np.copy(c['od'].T)              # Run with entire water vapor profile perturbed

    wnum = freq / 30.
    if verbose >= 3:
        print('Computing the baseline microwave brightness temperature spectrum')
    gasod = np.copy(od0)
    t = t + 273.16         # Convert degC to degK

    # Get the appropriate temperature profile for the calculations. Start by
    # interpolating the Standard Atmosphere profile to our vertical microwave
    # RT grid, and the lower part of the profile from what comes from the retrieval.

    tt = np.interp(a['z'], stdatmos['z'], stdatmos['t'])
    foo = np.where(a['z'] <= np.max(z+sfcz+0.1))[0]    #tolerance is needed due to floating point numbers
    if len(foo) != len(t):
        print('Problem here -- this should not happen')
        return flag, -999., -999., -999.
    tt[foo] = np.copy(t)
    t = np.copy(tt)

    # Compute the baseline radiance
    tb0 = Other_functions.radxfer_microwave(freq, t, gasod)

    # Compute the temperature perturbation
    # Note I'm changing both the optical depth spectrum for the layer
    # (which has the impact on the temperature dependence of the strength/
    # width of lines and on the continuum strength) as well as the
    # emission temperature of the layer
    if fixt != 1:
        if verbose >= 3:
            print(' Computing Jacobian for temperature')
        for kk in range(k):
            if z[kk] > jac_maxht:
                Kij[:,kk] = 0.
            else:
                gasod = np.copy(od0)          # Take baseline monochromatic ODs
                gasod[kk,:] = np.copy(od1[kk,:])     # Insert in the mono OD from perturbed temp run

                t0 =np.copy(t)
                t0[kk] += tpert
                tb1 = Other_functions.radxfer_microwave(freq, t0, gasod)
                if kk == 0:
                    mult = 0.5
                else:
                    mult = 1.0
                mult = 1.          # DDT -- I will keep the multiplier at 1 until I test it
                Kij[:,kk] = mult*(tb1-tb0)/tpert
    else:
        if verbose >= 3:
            print('Temperature jacobian set to zero (fixed T profile)')
        Kij[:,0:k] = 0.

    # Compute the water vapor perturbation
    if fixwv != 1:
        if verbose >= 3:
            print('Computing Jacobian for water vapor')
        for kk in range(k):
            if z[kk] > jac_maxht:
                Kij[:,kk+k] = 0.
            else:
                gasod = np.copy(od0)                 # Take baseline monochromatic ODs
                gasod[kk,:] = np.copy(od2[kk,:])     # Insert in the mono OD from perturbed H20 run

                # compute the baseline radiance
                tb1 = Other_functions.radxfer_microwave(freq, t, gasod)
                if kk == 0:
                    mult = 0.5
                else:
                    mult = 1.0
                mult = 1.          # DDT -- I will keep the multiplier at 1 until I test it
                Kij[:,kk+k] = mult * ( (tb1-tb0) / (w[kk]*h2opert - w[kk]))
    else:
        if verbose >= 3:
            print('Water vapor jacobian set to zero (fixed WV profile)')
        Kij[:,k:2*k] = 0.

    # Compute the liquid cloud property perturbation
    if fixlcld != 1:
        if verbose >= 3:
            print('Computing Jacobian for the liuid cloud properties')
        Kij[:,2*k] = (d['tb']-a['tb'])/(lwpp-lwp)
    else:
        if verbose >= 3:
            print('Cloud jacobian set to zero (fixed LWP)')
        Kij[:,2*k] = 0.

    # Capture the most accurate forward calculation
    FXn = np.copy(a['tb'])

    # Capture the total time and return
    etime = datetime.now()
    totaltime = (etime-stime).total_seconds()
    if verbose >= 3:
        print(' It took ' + str(totaltime) + ' s to compute Jacobian (3method)')

    flag = 1

    return flag, Kij, FXn, totaltime

################################################################################
# This function performs the forward model calculation and computes the jacobian
# for the pseudo-obs of RH at cloud base height
################################################################################
def compute_jacobian_pobs(Xn, p, z, cbh_pobs, vip, Y, sigY, fixt, fixwv, verbose):

        # Allocate space
    Kij = np.zeros((1,len(Xn)))

        # Compute the RH at the CBH.  First extract out the T and q vectors
    k   = len(z)
    t   = Xn[0:k]
    q   = Xn[k:2*k]
    lwp = Xn[2*k]
    rh  = Calcs_Conversions.w2rh(q, p, t) * 100.  # Convert to %rh
    FXn = np.interp(cbh_pobs['cbh'], z, rh)

        # Now compute the Jacobian
    delt = 0.5
    foo1 = np.where(z <  cbh_pobs['cbh'])[0]
    foo2 = np.where(z >= cbh_pobs['cbh'])[0]
    if((len(foo1) > 0) & (len(foo2) > 0)):
            # Only compute the sensitivity wrt temperature if the temperature is not fixed
        if(fixt != 1):    
                # Temperature ones first
                #    The layer below the cloud
            tt            = np.copy(t)
            qq            = np.copy(q)
            tt[foo1[-1]]  = t[foo1[-1]] + delt
            rh0 = Calcs_Conversions.w2rh(qq, p, tt) * 100.  # Convert to %rh
            rh1 = np.interp(cbh_pobs['cbh'], z, rh0)
            Kij[0,foo1[-1]] = (rh1-FXn) / delt
                #    The layer at or above the cloud
            tt           = np.copy(t)
            tt[foo2[0]]  = t[foo2[0]] + delt
            rh0 = Calcs_Conversions.w2rh(qq, p, tt) * 100.  # Convert to %rh
            rh1 = np.interp(cbh_pobs['cbh'], z, rh0)
            Kij[0,foo2[0]] = (rh1-FXn) / delt

            # Only compute the sensitivity wrt water vapor if the WVMR is not fixed
        if(fixwv != 1):    
                # Water vapor next
                #    The layer below the cloud
            tt            = np.copy(t)
            qq            = np.copy(q)
            qq[foo1[-1]]  = q[foo1[-1]]*0.99
            rh0 = Calcs_Conversions.w2rh(qq, p, tt) * 100.  # Convert to %rh
            rh1 = np.interp(cbh_pobs['cbh'], z, rh0)
            Kij[0,k+foo1[-1]] = (rh1-FXn) / (qq[foo1[-1]] - q[foo1[-1]])
                #    The layer at or above the cloud
            qq           = np.copy(q)
            qq[foo2[0]]  = q[foo2[0]]*0.99
            rh0 = Calcs_Conversions.w2rh(qq, p, tt) * 100.  # Convert to %rh
            rh1 = np.interp(cbh_pobs['cbh'], z, rh0)
            Kij[0,k+foo2[0]] = (rh1-FXn) / (qq[foo2[0]] - q[foo2[0]])

        # Now evaluate the current state of the state vector to see apply the proper 
        # value and uncertainty to the pseudo-ob (what is currently in place depends
        # on the cloud temperature and the LWP)
            # Look at the cloud temperature first
    valm0 = vip['cbh_pobs_out_rh_val']
    vals0 = vip['cbh_pobs_out_rh_uncert']
    valm1 = cbh_pobs['mn']
    vals1 = cbh_pobs['sg']
    tcld = np.interp(cbh_pobs['cbh'], z, t)
    if(tcld < vip['cbh_pobs_cld_temp_thres']):
        pobsmn1 = valm0
        pobssg1 = vals0
    elif(tcld > vip['cbh_pobs_cld_temp_thres']+vip['cbh_pobs_cld_temp_buffer']):
        pobsmn1 = valm1
        pobssg1 = vals1
    else:
        pobsmn1 = np.interp(tcld,[vip['cbh_pobs_cld_temp_thres'],vip['cbh_pobs_cld_temp_thres']+vip['cbh_pobs_cld_temp_buffer']],[valm0,valm1])
        pobssg1 = np.interp(tcld,[vip['cbh_pobs_cld_temp_thres'],vip['cbh_pobs_cld_temp_thres']+vip['cbh_pobs_cld_temp_buffer']],[vals0,vals1])
            # Look at the LWP next
    if(lwp < vip['cbh_pobs_lwp_thres']):
        pobsmn2 = valm0
        pobssg2 = vals0
    elif(tcld > vip['cbh_pobs_lwp_thres']+vip['cbh_pobs_lwp_buffer']):
        pobsmn2 = valm1
        pobssg2 = vals1
    else:
        pobsmn2 = np.interp(lwp,[vip['cbh_pobs_lwp_thres'],vip['cbh_pobs_lwp_thres']+vip['cbh_pobs_lwp_buffer']],[valm0,valm1])
        pobssg2 = np.interp(lwp,[vip['cbh_pobs_lwp_thres'],vip['cbh_pobs_lwp_thres']+vip['cbh_pobs_lwp_buffer']],[vals0,vals1])

            # Now take the most conservative option (the one with the largest uncertainty) to use as the current pseudo-ob
    if(pobssg2 > pobssg1):
        mn_pob = pobsmn2
        sg_pob = pobssg2
    else:
        mn_pob = pobsmn1
        sg_pob = pobssg1

    return Kij, [[FXn]], mn_pob, sg_pob

################################################################################
# This function performs the forward model calculation and computes the jacobian
# for the microwave radiometer where only the LWP is perturbed
################################################################################
def compute_jacobian_microwave_lwp_only(z, p, t, w, freq, cbh, Xn, Qext_ratio, vip, workdir,
                monortm_tfile, monortm_exec, stdatmos, sfc_alt, stored_jacobian, verbose):

    monortm_outputfile = workdir+'/monortm_output.txt'      # A temporary output file
    flag = 0                      # Failure
    k = len(z)
    cth = cbh + 0.300             # km; define the cloud top at x m above the could base

    if(sfc_alt == None):
        sfcz=0
    else:
        sfcz = sfc_alt / 1000.

    # Allocate space for the Jacobian and forward calculation
    Kij = np.zeros((len(freq),len(Xn)))
    FXn = np.zeros(len(freq))

    if verbose >= 2:
        print('Computing the MWR-lwp-only Jacobian using the MonoRTM')

    # Perform the baseline calculation
    lwp = Other_functions.compute_lwp(Xn[0]*Qext_ratio,Xn[1])
    u   = Calcs_Conversions.w2rh(w, p, t, 0) * 100
    Other_functions.write_arm_sonde_file((z+sfcz)*1000, p, t, u, workdir+'/'+monortm_tfile, silent = True)
    command = ('rm -f '+monortm_outputfile+' ; '+monortm_exec + ' ' + monortm_tfile + ' {:3.1f} {:8.2f} {:6.3f} {:6.3f}'.format(1.0, lwp, cbh, cth)+' > '+monortm_outputfile)
    a = LBLRTM_Functions.run_monortm(command, freq, z, stdatmos, monortm_outputfile)
    if a['status'] == 0:
        print('Problem with MonoRTM-lwp-only calc 0')
        return flag, -999., -999.
    FXn = np.copy(a['tb'])

    # Perform the pertubation calculation.  If I already have precomputed it, and the computed Tbs are
    # less than 200 K for all channels, then I can use the precomputed one.  Otherwise, recompute it.
    foo = np.where(FXn < 200)[0]
    if((stored_jacobian['computed'] != 1) or (len(foo) != len(FXn))):
        if verbose >= 3:
            print('Computing Jacobian for the liquid cloud properties')
        lwpp = lwp + 25.
        command = ('rm -f '+monortm_outputfile+' ; '+monortm_exec + ' ' + 
            monortm_tfile + ' {:3.1f} {:8.2f} {:6.3f} {:6.3f}'.format(1.0, lwpp, cbh, cth)+' > '+monortm_outputfile)
        d = LBLRTM_Functions.run_monortm(command, freq, z, stdatmos, monortm_outputfile)
        if d['status'] == 0:
            print('Problem with MonoRTM-lwp-only calc 1')
            return flag, -999., -999.
        dTb_dLWP  = (d['tb']-a['tb'])/(lwpp-lwp)
            # Store the Jacobian to reuse next time
        if(stored_jacobian['computed'] != 1):
            if(verbose >= 2):
                print('      Storing the computed MWR jacobian')
            stored_jacobian = {'computed':1, 'dTb_dLWP':dTb_dLWP}
    else:
        if(verbose >= 2):
            print('      Using the precomputed MWR jacobian')
        dTb_dLWP  = stored_jacobian['dTb_dLWP']

    # Adjust units from perturbations in LWP to perturbations in liquid water optical depth
    ltau_pert = 1.
    dLWP_dtau = (lwp - Other_functions.compute_lwp((Xn[0]+ltau_pert)*Qext_ratio,Xn[1])) / (Qext_ratio*(Xn[0] - (Xn[0]+ltau_pert)))
    Kij[:,0]  = dTb_dLWP * dLWP_dtau

    flag = 1

    return flag, Kij, FXn, stored_jacobian

################################################################################
# This function performs the forward model calculation and computes the Jacobian
# for the external temperature profiler. Note that for now the code is the same
# for all types of data. This is repeatitive, but I am leaving it that way so these
# can be changed individually if needed.
################################################################################

def compute_jacobian_external_temp_profiler(Xn, p, z, minht, maxht, temp_type):

    flag = 0            # Failure
    k = len(z)
    t = np.copy(Xn[0:k])

    # The action depends on the type of the observation
    # ARM radiosonde
    if temp_type == 1:
        # The radiosonde data were read in and are in degC
        foo = np.where((minht <= z) & (z <= maxht))[0]
        if len(foo) == 0:
            print('Error in external temperature profiler forward model -- no vertical levels')
            return flag, -999., -999.

        # We have a perfect forward model and the Jacobian has perfect sensitivity
        # to changes in the temperature (over the range where we have observations)
        # and no sensitivity anywhere else

        FXn = np.copy(t[foo])
        Kij = np.zeros((len(foo),len(Xn)))
        for j in range(len(foo)):
            Kij[j,foo[j]] = 1.

    # ARM Raman lidar
    elif temp_type == 2:
        # The RLID temperature data are in degC
        foo = np.where((minht <= z) & (z <= maxht))[0]
        if len(foo) == 0:
            print('Error in external temperature profiler forward model -- no vertical levels found')
            return flag, -999., -999

        # We have a perfect forward model and the Jacobian has perfect sensitivity
        # to changes in the temperature (over the range where we have observations)
        # and no sensitivity anywhere else

        FXn = np.copy(t[foo])
        Kij = np.zeros((len(foo),len(Xn)))
        for j in range(len(foo)):
            Kij[j,foo[j]] = 1.

    # RAP/RUC model input
    elif temp_type == 4:
        foo = np.where((minht <= z) & (z <= maxht))[0]
        if len(foo) == 0:
            print('Error in external temperature profiler forward model -- no vertical levels found')
            return flag, -999., -999.

        # We have a perfect forward model and the Jacobian has perfect sensitivity
        # to changes in the temperature (over the range where we have observations)
        # and no sensitivity anywhere else

        FXn = np.copy(t[foo])
        Kij = np.zeros((len(foo),len(Xn)))
        for j in range(len(foo)):
            Kij[j,foo[j]] = 1.

    # RASS virtual temperature
    elif temp_type == 5:
        foo = np.where((minht <= z) & (z <= maxht))[0]
        if len(foo) == 0:
            print('Error in external temperature profiler forward model -- no vertical levels found')
            return flag, -999., -999
        # The RASS's temperature data is in virtual temperature [C]
        t0 = np.copy(Xn[0:k])
        q0 = np.copy(Xn[k:2*k])
        p0 = np.copy(p) * 100  # convert pressure from mb to pascals (only used in Tv calc)
        rh = Calcs_Conversions.w2rh(q0, p, t0, 0)   # RH is unitless between 0 and 1
        ttv = Calcs_Conversions.tvirt(t0, rh, p0)   # virtual temperature in [C]

        # Compute the jacobian over the appropriate height range
        Kij = np.zeros((len(foo),len(Xn)))
        tpert = 1.0    # Additive
        qpert = 0.95   # multiplicative
        for i in range(len(foo)):
            # Compute sensitivity to a perturbation in temperature
            t1 = t0.copy()
            t1[foo[i]] += tpert
            rh = Calcs_Conversions.w2rh(q0, p, t1, 0)   # RH is unitless
            tmp = Calcs_Conversions.tvirt(t1, rh, p0)   # virtual temperature in [C]
            Kij[i,foo[i]] = (tmp[foo[i]]-ttv[foo[i]]) / (t1[foo[i]] - t0[foo[i]])

            # Compute sensitivity to a perturbation in water vapor mixing ratio
            q1 = q0.copy()
            if(q1[foo[i]] <= 0):
                q1[foo[i]] = 0.05
            else:
                q1[foo[i]] *= qpert
            rh = Calcs_Conversions.w2rh(q1, p, t0, 0)   # RH is unitless
            tmp = Calcs_Conversions.tvirt(t0, rh, p0)   # virtual temperature in [C]
            Kij[i,foo[i]+k] = (tmp[foo[i]]-ttv[foo[i]]) / (q1[foo[i]] - q0[foo[i]])
        FXn = ttv[foo]

    elif temp_type == 7:
        foo = np.where((minht <= z) & (z <= maxht))[0]
        if len(foo) == 0:
            print('Error in general temperature profiler forward model -- no vertical levels found')
            return flag, -999., -999.

        # We have a perfect forward model and the Jacobian has perfect sensitivity
        # to changes in the temperature (over the range where we have observations)
        # and no sensitivity anywhere else

        FXn = np.copy(t[foo])
        Kij = np.zeros((len(foo),len(Xn)))
        for j in range(len(foo)):
            Kij[j,foo[j]] = 1.

    # AER's GVRP temperature retrievals from RHUBC-2
    elif temp_type == 99:
        # The radiosonde data were read in and are in degC
        foo = np.where((minht <= z) & (z <= maxht))[0]
        if len(foo) == 0:
            print('Error in external temperature profiler forward model -- no vertical levels')
            return flag, -999., -999.

        # We have a perfect forward model and the Jacobian has perfect sensitivity
        # to changes in the temperature (over the range where we have observations)
        # and no sensitivity anywhere else

        FXn = np.copy(t[foo])
        Kij = np.zeros((len(foo),len(Xn)))
        for j in range(len(foo)):
            Kij[j,foo[j]] = 1.

    # Undefined external temperature profiler source
    else:
        print('Undefined external temperature profiler used in its forward model -- aborting')
        return flag, -999., -999.

    flag = 1
    return flag, Kij, FXn

################################################################################
# This function performs the forward model calculation and computes the Jacobian
# for the external water vapor profiler
################################################################################

def compute_jacobian_external_wv_profiler(Xn, p, z, minht, maxht, wv_type, wv_multiplier):

    flag = 0            # Failure
    k = len(z)
    t = np.copy(Xn[0:k])
    w = np.copy(Xn[k:2*k])

    # The action depends on the type of observation
    # ARM radiosonde
    if wv_type == 1:
        # The radiosonde data were read in and converted to wv mixing ratio
        foo = np.where((minht <= z) & (z <= maxht))[0]
        if len(foo) == 0:
            print('Error in external water vapor profiler forward model -- no vertical levels')
            return flag, -999., -999.

        # We have a perfect forward model and the Jacobian has perfect sensitivity
        # to changes in the WVMR (over the range where we have observations)
        # and no sensitivity anywhere else

        FXn = np.copy(w[foo])
        Kij = np.zeros((len(foo),len(Xn)))
        for j in range(len(foo)):
            Kij[j,k+foo[j]] = 1.

    # ARM Raman lidar
    elif wv_type == 2:
        # The RLID water vapor data are in mixing ratio.
        # Since the retrieval works in the same unit, this forward model
        # is trivial, as is the Jacobian
        foo = np.where((minht <= z) & (z <= maxht))[0]
        if len(foo) == 0:
            print('Error in external water vapor profiler forward model -- no vertical levels')
            return flag, -999., -999.

        # We have a perfect forward model and the Jacobian has perfect sensitivity
        # to changes in the WVMR (over the range where we have observations)
        # and no sensitivity anywhere else

        FXn = np.copy(w[foo])
        Kij = np.zeros((len(foo),len(Xn)))
        for j in range(len(foo)):
            Kij[j,k+foo[j]] = 1.

    # NCAR WV DIAL (from the 2014-2017 time period (FRAPPE, PECAN, Perdigao, LAFE))
    elif wv_type == 3:
        # The DIAL's WV data is density [mol cm-3]
        wvdens = Calcs_Conversions.tq2wvdens(t,w,p)/wv_multiplier

        # Compute the Jocobian over the appropriate height
        foo = np.where((minht <= z) & (z <= maxht))[0]
        if len(foo) == 0:
            print('Error in external water vapor profiler forward model -- no vertical levels')
            return flag, -999., -999.

        Kij = np.zeros((len(foo),len(Xn)))

        # We have a perfect forward model, and the Jacobian has perfect sensitivity

        tpert = 1.0    # additive
        qpert = 0.95   # Multiplicative
        for i in range(len(foo)):
            # Compute the sensitivity to a perturbation in temperature
            t1 = np.copy(t)
            t1[foo[i]] += tpert
            tmp = Calcs_Conversions.tq2wvdens(t1,w,p)/wv_multiplier
            Kij[i,foo[i]] = (tmp[foo[i]]-wvdens[foo[i]]) / (t1[foo[i]] - t[foo[i]])

            # Compute the snsitivity to a perturbation in water vapor
            q1 = np.copy(w)
            if q1[foo[i]] <= 0:
                q1[foo[i]] = 0.05
            else:
                q1[foo[i]] *= qpert
            tmp = Calcs_Conversions.tq2wvdens(t,q1,p)/wv_multiplier
            Kij[i,foo[i]+k] = (tmp[foo[i]]-wvdens[foo[i]]) / (q1[foo[i]]-w[foo[i]])

        FXn = wvdens[foo]

    #RAP/RUC Model
    elif wv_type == 4:
        foo = np.where((minht <= z) & (z <= maxht))[0]
        if len(foo) == 0:
            print('Error in external water vapor profiler forward model -- no vertical levels')
            return flag, -999., -999.

        # We have a perfect forward model and the Jacobian has perfect sensitivity
        # to changes in the WVMR (over the range where we have observations)
        # and no sensitivity anywhere else

        FXn = np.copy(w[foo])
        Kij = np.zeros((len(foo),len(Xn)))
        for j in range(len(foo)):
            Kij[j,k+foo[j]] = 1.

    #Vaisala WV DIAL
    elif wv_type == 5:
        foo = np.where((minht <= z) & (z <= maxht))[0]
        if len(foo) == 0:
            print('Error in external water vapor profiler forward model -- no vertical levels')
            return flag, -999., -999.

        # We have a perfect forward model and the Jacobian has perfect sensitivity
        # to changes in the WVMR (over the range where we have observations)
        # and no sensitivity anywhere else

        FXn = np.copy(w[foo])
        Kij = np.zeros((len(foo),len(Xn)))
        for j in range(len(foo)):
            Kij[j,k+foo[j]] = 1

    # NCAR DIAL (from 2019 time frame sugh as SGP MPL demonstration IOP)
    elif wv_type == 6:
        # Compute the Jacobian over the appropriate height
        foo = np.where((minht <= z) & (z <= maxht))[0]
        if len(foo) == 0:
            print('Error in external water vapor profiler forward model -- no vertical levels')
            return flag, -999., -999.

        Kij = np.zeros((len(foo),len(Xn)))
        rho = Calcs_Conversions.w2rho(w, t, p)

        tpert = 1.0    # additive
        qpert = 0.95   # Multiplicative
        for i in range(len(foo)):
            # Compute the sensitivity to a perturbation in temperature
            t1 = np.copy(t)
            t1[foo[i]] += tpert
            tmp = Calcs_Conversions.w2rho(w, t1, p)
            Kij[i,foo[i]] = (tmp[foo[i]]-rho[foo[i]]) / (t1[foo[i]] - t[foo[i]])

            # Compute the snsitivity to a perturbation in water vapor
            q1 = np.copy(w)
            if q1[foo[i]] <= 0:
                q1[foo[i]] = 0.05
            else:
                q1[foo[i]] *= qpert
            tmp = Calcs_Conversions.w2rho(q1, t, p)
            Kij[i,foo[i]+k] = (tmp[foo[i]]-rho[foo[i]]) / (q1[foo[i]]-w[foo[i]])

        FXn = rho[foo]

    elif wv_type == 7:
        foo = np.where((minht <= z) & (z <= maxht))[0]
        if len(foo) == 0:
            print('Error in general water vapor profiler forward model -- no vertical levels found')
            return flag, -999., -999.

        # We have a perfect forward model and the Jacobian has perfect sensitivity
        # to changes in the temperature (over the range where we have observations)
        # and no sensitivity anywhere else

        FXn = np.copy(w[foo])
        Kij = np.zeros((len(foo),len(Xn)))
        for j in range(len(foo)):
            Kij[j,k+foo[j]] = 1.

    # AER's GVRP water vapor retrievals from RHUBC-2
    elif wv_type == 99:
        # The radiosonde data were read in and converted to wv mixing ratio
        foo = np.where((minht <= z) & (z <= maxht))[0]
        if len(foo) == 0:
            print('Error in external water vapor profiler forward model -- no vertical levels')
            return flag, -999., -999.

        # We have a perfect forward model and the Jacobian has perfect sensitivity
        # to changes in the WVMR (over the range where we have observations)
        # and no sensitivity anywhere else

        FXn = np.copy(w[foo])
        Kij = np.zeros((len(foo),len(Xn)))
        for j in range(len(foo)):
            Kij[j,k+foo[j]] = 1.

    else:
        print('Undefned external water vapor profiler used in its forward model -- aborting')
        return flag, -999., -999.

    flag = 1
    return flag, Kij, FXn

################################################################################
# This function performs the forward model calculation and computes the jacobian
# for the surface met data
################################################################################

def compute_jacobian_external_sfc_met(Xn, p, z, sfc_relative_height,
                            units, chimney_height):

    flag = 0                      # Faliure
    k = len(z)
    t = np.copy(Xn[0:k])
    w = np.copy(Xn[k:2*k])

    # Convert the surface relative height from m to km
    sfc_rel_height = sfc_relative_height /1000.

    # The action depends on the type of observation. Note that we are just interpolating
    # the data linearly to the sfc_relative_height, and will do this in either temperatuer
    # or water vapor mixing ratio space.
    Kij = np.zeros((len(units), len(Xn)))
    FXn = np.zeros(len(units))
    foo = np.where(z >= chimney_height)[0]
    if len(foo) < 2:
        print('Warning: The chimney is too high in compute_jacobian_external_sfc_met')
        return flag, -999., -999.

    for i in range(len(units)):
        if units[i] == 'degC':
            func = scipy.interpolate.interp1d(z[foo], t[foo], fill_value = 'extrapolate')
            FXn[i] = func(sfc_rel_height)
            dell = 0.1
            for j in range(k):
                tt = np.copy(t)
                tt[j] += dell
                func = scipy.interpolate.interp1d(z[foo], tt[foo], fill_value = 'extrapolate')
                Kij[i,j] = (func(sfc_rel_height) - FXn[i]) / (tt[j] - t[j])

        elif units[i] == 'g/kg':
            func = scipy.interpolate.interp1d(z[foo], w[foo], fill_value = 'extrapolate')
            FXn[i] = func(sfc_rel_height)
            dell = 0.1
            for j in range(k):
                ww = np.copy(w)
                ww[j] += dell
                func = scipy.interpolate.interp1d(z[foo], ww[foo], fill_value = 'extrapolate')
                Kij[i,j+k] = (func(sfc_rel_height) - FXn[i]) / (ww[j] - w[j])
        else:
            print('Undefined external surface met unit used in the forward model -- aborting')
            return flag, -999., -999

    flag = 1
    return flag, Kij, FXn

################################################################################
# This function performs the forward model calculation and computes the jacobian
# for the in-situ surface co2 data
################################################################################

def compute_jacobian_external_sfc_co2(Xn, p, z, sfc_relative_height, retrieve_co2,
                                  fix_co2_shape, nptsCO2):

    flag = 0                      # Failure
    k = len(z)
    coef = np.copy(Xn[2*k+4:2*k+7])        # Extract out the co2 coefficients

    # Convert the surface relative height from m to km
    sfc_rel_height = sfc_relative_height / 1000.

    # Build the CO2 profile from our three coefficients
    co2_prof = Other_functions.trace_gas_prof(retrieve_co2, z, coef)

    # Ensure that the profile makes some sense
    foo = np.where(co2_prof <= 0)[0]
    if len(foo) > 0:
        print('Error inside compute_jacobian_external_sfc_co2: CO2 is less than zero')
        return flag, -999., -999.

    # We are just interpolating the data linearly to the sfc_relative_height
    Kij = np.zeros((nptsCO2,len(Xn)))
    func = scipy.interpolate.interp1d(z, co2_prof, fill_value = 'extrapolate')
    FXn = func(sfc_rel_height)    # I will get the dimensionality of this correct at end of routine

    # Perturb each of the CO2 coefficients here to get the jacobian
    # but use the same approach as in the compute_jacobian_deltaod function
    co2_pert = 4.0        # will be used for coefficients 0 and 1

    #Perturb first coefficient
    pcoef = np.copy(coef)
    pcoef[0] += co2_pert
    pco2_prof = Other_functions.trace_gas_prof(retrieve_co2,z,pcoef)
    func = scipy.interpolate.interp1d(z, pco2_prof, fill_value = 'extrapolate')
    tmp = func(sfc_rel_height)
    Kij[:,2*k+4] = (tmp - FXn) / (pcoef[0] - coef[0])

    # Perturb second coefficient
    pcoef = np.copy(coef)
    pcoef[1] += co2_pert
    pco2_prof = Other_functions.trace_gas_prof(retrieve_co2, z, pcoef)
    func = scipy.interpolate.interp1d(z, pco2_prof, fill_value = 'extrapolate')
    tmp = func(sfc_rel_height)
    Kij[:,2*k+5] = (tmp - FXn) / (pcoef[1] - coef[1])

    # Perturb third coefficient
    if ((retrieve_co2 == 1) | (fix_co2_shape == 1)):
        Kij[:,2*k+6] = 0.            # Don't let the Jacobian have any sensitivity to the shape parameter here.
    else:
        pcoef = np.copy(coef)
        pcoef[2] += co2_pert

        # If the 2nd coefficient is zero, then the Jacobian will not show any sensitivity
        # to the shape of the profile, and thus it can not change the shape. This is a
        # problem , so I will allow a bit of sensitivity to occur. (This is identical
        # logic to what is in compute_jacobian_deltaod)
        if np.abs(pcoef[1]) <  co2_pert/10.:
            if pcoef[1] < 0:
                pcoef[1] = -co2_pert/10.
            else:
                pcoef[1] = co2_pert/10.

        pco2_prof = Other_functions.trace_gas_prof(retrieve_co2, z, pcoef)
        func = scipy.interpolate.interp1d(z, pco2_prof, fill_value = 'extrapolate')
        tmp = func(sfc_rel_height)
        Kij[:,2*k+6] = (tmp - FXn) / (pcoef[2] - coef[2])

    FXn = np.ones(nptsCO2)*FXn         # Get the dimensionality of forward calculation correct

    flag = 1
    return flag, Kij, FXn

################################################################################
# This function performs the forward model calculation and computes the jacobian
# for the microwave radiometer in scan mode. It is designed very similarly to
# compute_jacobian_3method()
################################################################################

def compute_jacobian_microwavescan_3method(Xn, p, z, mwrscan, cbh, vip, workdir,
                monortm_tfile, monortm_exec, fixt, fixwv, fixlcld, jac_maxht,
                stdatmos, sfc_alt, verbose):
    
    monortm_outputfile = workdir+'/monortm_output.txt'      # A temporary output file
    flag = 0               # Failure

    # Allocate space for the Jacobian and forward calculation
    Kij = np.zeros((len(mwrscan['dim']),len(Xn)))
    FXn = np.zeros(len(mwrscan['dim']))
    missing = -999.

    if(sfc_alt == None):
        sfcz=0
    else:
        sfcz = sfc_alt / 1000.

    if verbose >= 2:
        print('Computing the MWR-scan Jacobian using the 3method with MonoRTM')

    stime = datetime.now()

    # Extract out the really unique angles
    elev = np.copy(mwrscan['elevations'])
    foo = np.where(elev > 90)[0]
    if len(foo) > 0:
        elev[foo] = 180-elev[foo]
    uelev = np.array([elev[0]])
    for ii in range(len(elev)):
        unq = 1
        for jj in range(len(uelev)):
            if np.abs(elev[ii]-uelev[jj]) < 0.1: unq = 0
        if unq == 1: uelev = np.append(uelev, elev[ii])

    # Loop overthe elevation angles, running the forward model
    # and computing the Jacobian

    for ii in range(len(uelev)):
            # Initialize these after every elevation height
        KKij = np.zeros((mwrscan['n_fields'],len(Xn)))
        FFXn = np.zeros((mwrscan['n_fields']))*0 + missing

        # Extract out the information needed from the state vector
        k = len(z)
        t = np.copy(Xn[0:k])          # degC
        w = np.copy(Xn[k:2*k])        # g/kg
        lwp = Xn[2*k]                 # g/m2
        cth = cbh + 0.300             # km; define the cloud top at x m above the could base

        # Perform the baseline calculation
        u = Calcs_Conversions.w2rh(w, p, t, 0) * 100
        Other_functions.write_arm_sonde_file((z+sfcz)*1000, p, t, u, workdir +'/' + monortm_tfile, silent = True)
        elevOff = 0.1   # this is in degrees elevation
        didfail = 0     # If I am unable to get an accurate computation along the slant angle, then didfail = 1
        cnt = 0
        command = ('rm -f '+monortm_outputfile+' ; '+monortm_exec + ' ' + monortm_tfile + ' {:3.1f} {:8.2f} {:6.3f} {:6.3f} {:6.3f}'.format(1.0, lwp, cbh, cth, 90-uelev[ii]+cnt*elevOff)+' > '+monortm_outputfile)
        a = LBLRTM_Functions.run_monortm(command, mwrscan['freq'], z, stdatmos, monortm_outputfile)
        while((a['status'] != 1) & (cnt < 2)):
            cnt += 1
            command = ('rm -f '+monortm_outputfile+' ; '+monortm_exec + ' ' + monortm_tfile + ' {:3.1f} {:8.2f} {:6.3f} {:6.3f} {:6.3f}'.format(1.0, lwp, cbh, cth, 90-uelev[ii]+cnt*elevOff)+' > '+monortm_outputfile)
            a = LBLRTM_Functions.run_monortm(command, mwrscan['freq'], z, stdatmos, monortm_outputfile)

        if a['status'] == 0:
            print('    Bending angle problem with MonoRTM in mwrScan0')
            didfail = 1
        else: FFXn = np.copy(a['tb'])

        if fixt != 1:
            tpert = 1.0           # Additive perturbation of 1 K
            t0 = t + tpert
            u = Calcs_Conversions.w2rh(w, p, t0, 0) * 100
            Other_functions.write_arm_sonde_file((z+sfcz)*1000, p, t0, u, workdir +'/' + monortm_tfile, silent = True)
            cnt = 0
            command = ('rm -f '+monortm_outputfile+' ; '+monortm_exec + ' ' + monortm_tfile + ' {:3.1f} {:8.2f} {:6.3f} {:6.3f} {:6.3f}'.format(1.0, lwp, cbh, cth, 90-uelev[ii]+cnt*elevOff)+' > '+monortm_outputfile)
            b = LBLRTM_Functions.run_monortm(command, mwrscan['freq'], z, stdatmos, monortm_outputfile)
            while((b['status'] != 1) & (cnt < 2)):
                cnt += 1
                command = ('rm -f '+monortm_outputfile+' ; '+monortm_exec + ' ' + monortm_tfile + ' {:3.1f} {:8.2f} {:6.3f} {:6.3f} {:6.3f}'.format(1.0, lwp, cbh, cth, 90-uelev[ii]+cnt*elevOff)+' > '+monortm_outputfile)
                b = LBLRTM_Functions.run_monortm(command, mwrscan['freq'], z, stdatmos, monortm_outputfile)
            if b['status'] == 0:
                print('    Bending angle problem with MonoRTM in mwrScan1')
                didfail = 1
        else:
            command = 'sleep 1'

        if fixwv != 1:
	    # Set the size of the perturbation as a crude function of the WVMR at the surface
            h2opert = np.interp(w[0],[0.01,0.1,2],[0.75,0.90,0.99])

            w0 = w*h2opert
            u = Calcs_Conversions.w2rh(w0, p, t, 0) * 100
            Other_functions.write_arm_sonde_file((z+sfcz)*1000, p, t, u, workdir +'/' + monortm_tfile, silent = True)
            cnt = 0
            command = ('rm -f '+monortm_outputfile+' ; '+monortm_exec + ' ' + monortm_tfile + ' {:3.1f} {:8.2f} {:6.3f} {:6.3f} {:6.3f}'.format(1.0, lwp, cbh, cth, 90-uelev[ii]+cnt*elevOff)+' > '+monortm_outputfile)
            c = LBLRTM_Functions.run_monortm(command, mwrscan['freq'], z, stdatmos, monortm_outputfile)
            while((c['status'] != 1) & (cnt < 2)):
                cnt += 1
                command = ('rm -f '+monortm_outputfile+' ; '+monortm_exec + ' ' + monortm_tfile + ' {:3.1f} {:8.2f} {:6.3f} {:6.3f} {:6.3f}'.format(1.0, lwp, cbh, cth, 90-uelev[ii]+cnt*elevOff)+' > '+monortm_outputfile)
                c = LBLRTM_Functions.run_monortm(command, mwrscan['freq'], z, stdatmos, monortm_outputfile)
            if c['status'] == 0:
                print('    Bending angle problem with MonoRTM in mwrScan2')
                didfail = 1
        else:
            command = 'sleep 1'

        if fixlcld != 1:
            lwpp = lwp + 25.
            u = Calcs_Conversions.w2rh(w, p, t, 0) * 100
            Other_functions.write_arm_sonde_file((z+sfcz)*1000, p, t, u, workdir +'/' + monortm_tfile, silent = True)
            cnt = 0
            command = ('rm -f '+monortm_outputfile+' ; '+monortm_exec + ' ' + monortm_tfile + ' {:3.1f} {:8.2f} {:6.3f} {:6.3f} {:6.3f}'.format(1.0, lwp, cbh, cth, 90-uelev[ii]+cnt*elevOff)+' > '+monortm_outputfile)
            d = LBLRTM_Functions.run_monortm(command, mwrscan['freq'], z, stdatmos, monortm_outputfile)
            while((d['status'] != 1) & (cnt < 2)):
                cnt += 1
                command = ('rm -f '+monortm_outputfile+' ; '+monortm_exec + ' ' + monortm_tfile + ' {:3.1f} {:8.2f} {:6.3f} {:6.3f} {:6.3f}'.format(1.0, lwp, cbh, cth, 90-uelev[ii]+cnt*elevOff)+' > '+monortm_outputfile)
                d = LBLRTM_Functions.run_monortm(command, mwrscan['freq'], z, stdatmos, monortm_outputfile)
            if d['status'] == 0:
                print('    Bending angle problem with MonoRTM in mwrScan3')
                didfail = 1
        else:
            command = 'sleep 1'

        # Capture the different optical depths into simple matrices
        if(didfail == 1):
            if(verbose >= 1):
                print('    Bending angle problem at {:6.3f} degree elevation angle in mwrScan'.format(uelev[ii]))
        else:   # code did not fail
            od0 = np.copy(a['od'].T)
            if fixt != 1:
                od1 = np.copy(b['od'].T)
            if fixwv != 1:
                od2 = np.copy(c['od'].T)

            wnum = mwrscan['freq']/30.
            if verbose >= 3:
                print('Computing the baseline microwave brightness temperature spectrum MWR-scan')

            gasod = np.copy(od0)
            t = t + 273.16  # Convert degC to degK

            # Get the appropriate temperature profile for the calculation. Start by
            # interpolating the Standard Atmosphere profile to our vertical microwave
            # RT grid, and then replace the lower part of the profile from what comes
            # from the retrieval.

            tt = np.interp(a['z'], stdatmos['z'], stdatmos['t'])

            foo = np.where(a['z'] <= np.max(z+sfcz+0.1))[0]
            if len(foo) != len(t):
                print('Problem here -- this should not be happen MWR-scan')
            tt[foo] = np.copy(t)

            # Compute the baseline radiance
            tb0 = Other_functions.radxfer_microwave(mwrscan['freq'], tt, gasod)

            # Compute the temperature perturbation
            # Note I'm changing both the optical depth spectrum for the layer
            # (which has the impact on the temeprature dependence of the strength/
            # width of the lines and on the continuum strength) as well as the
            # emission temperature of the layer

            if fixt != 1:
                if verbose >= 3:
                    print('Computing Jacobian for temperature MWR-scan')
                for kk in range(k):
                    if z[kk] > jac_maxht:
                        KKij[:,kk] = 0.
                    else:
                        gasod = np.copy(od0)       # Take baseline monochromatic ODs
                        gasod[kk,:] = od1[kk,:]      # Insert in the mono OD from perturbed temp run

                        t0 = np.copy(tt)
                        t0[kk] += tpert
                        tb1 = Other_functions.radxfer_microwave(mwrscan['freq'], t0, gasod)
                        if kk == 0:
                            mult = 0.5
                        else:
                            mult = 1.0
                        mult = 1.0           # TODO: DDT -- I will keep the multiplier at 1 until I test it
                        KKij[:,kk] = mult * (tb1-tb0) / tpert
            else:
                if verbose >= 3:
                    print('Temperature jacobian set to zero (fixed T profile) MWR-scan')
                KKij[:,0:k] = 0.

            # Compute the water vapor perturbation

            if fixwv != 1:
                if verbose >= 3:
                    print('Computing Jacobian for water vapor MWR-scan')
                for kk in range(k):
                    if z[kk] > jac_maxht:
                        KKij[:,kk+k] = 0.
                    else:
                        gasod = np.copy(od0)       # Take baseline monochromatic ODs
                        gasod[kk,:] = od2[kk,:]      # Insert in the mono OD from perturbed H20 run

                        # Compute the baseline radiance
                        tb1 = Other_functions.radxfer_microwave(mwrscan['freq'], tt, gasod)
                        if kk == 0:
                            mult = 0.5
                        else:
                            mult = 1.0
                        mult = 1.          # DDT -- I will keep the multiplier at 1 until I test it
                        KKij[:,kk+k] = mult * ( (tb1-tb0) / (w[kk]*h2opert - w[kk]) )
            else:
                if verbose >= 3:
                    print('Water vapor jacobian set to zero (fixed WV profile) MWR-scan')
                KKij[:,k:2*k] = 0.

            # Compute the liquid cloud property perturbation
            if fixlcld != 1:
                if verbose >= 3:
                    print(' Computing Jacobian for the liquid cloud properties (LWP) MWR-scan')

                KKij[:,2*k] = (d['tb']-a['tb']) / (lwpp-lwp)
            else:
                if verbose >= 3:
                    print('Cloud jacobian set to zero (fixed LWP) MWR-scan')
                KKij[:,2*k] = 0.

                # Capture the most accurate forward calculation and Jacobian into the
                # appropriate structures. Since it is possible that we may have symmetric
                # angles around zenith (i.e., elevation angles 20 and 160 degrees), there
                # was no need to make the same calculation twice -- I just have to replicate
                # the entry here.
            FFXn = np.copy(a['tb'])
                # now I am at the end of "didfail != 1

        # I want to do this step every time, regardless if the MonoRTM calcs
        # succeeded (then KKij would be non-zero) or failed (KKij = 0)
        #   This allows me to only perform a single calc for (say) 10 deg elevation
        #   and then use the same calculation for the 170 deg "elevation"
        foo = np.where(elev == uelev[ii])[0]    # No longer used, kept for consistency with IDL code for now
        foo = np.where(np.abs(elev - uelev[ii]) < 0.1)[0]
        idx = np.arange(mwrscan['n_fields'])
        for kk in range(len(foo)):
            FXn[foo[kk]*mwrscan['n_fields']+idx] = FFXn
            Kij[foo[kk]*mwrscan['n_fields']+idx,:] = KKij

    # Capture the total time and return
    etime = datetime.now()
    totaltime = (etime-stime).total_seconds()
    if verbose >= 3:
        print('It took ' + str(totaltime) + ' s to compute Jacobian (3method) MWR-scan')

    flag = 1

    return flag, Kij, FXn, totaltime

##################################################################################
# This script makes the LBLRTM runs using the input profiles and the VIP input 
# variables, reads in the gaseous optical depths, and returns the structure
##################################################################################

def make_lblrtm_calc(vip, ymd, hour, co2, z, p, t, w, pwv, wnum1, wnum2, delt, verbose):

    err = {'status':0}
    if verbose >-2:
        quiet=0
    else:
        quiet=1
    
    # Get the name for the output files
    hhmm = int(hour)*100 + int((hour-int(hour))*60)

    # Check that the wavenumbers make sense
    if wnum2 <= wnum1:
        print('The wavenumber limits passed into make_lblrtm_cal are incorrect -- aborting')
        return err
    
    if wnum2-wnum1 > 2000:
        print('The wavenumber limits passed into make_lblrtm_calc span more than 2000 cm^-1 -- aborting')
        return err
    
    # Make the TAPES and the LBLRTM run
    tp5 = vip['workdir'] + '/tp5.' + str(ymd).zfill(8) + '.' + str(hhmm).zfill(4)
    out = vip['workdir'] + '/' + vip['lblout'] + '.' + str(ymd).zfill(8) + '.' + str(hhmm).zfill(4)

    # Check to see if I need to remake the LBLRTM runs or not
    if (delt == 0) and (os.path.exists(tp5)) and (os.path.exists(out)):
        if verbose >= 1:
            print(f'  Reading previous lblrtm runs for {ymd:08d}.{hhmm:04d} UTC')
    else:
        print(f'  Making the lblrtm runs for {ymd:08d}.{hhmm:04d} UTC')
    
        # Make sure that the output paths are ready
        if not os.path.exists(vip['workdir']):
            os.makedirs(vip['workdir'])
        
        # Make the tape5 file and run the model
        LBLRTM_Functions.rundecker(3, vip['lbl_std_atmos'], z, p, t+273.16, w,
             od_only=1, mlayers=z, wnum1=wnum1, wnum2=wnum2, tape5=tp5, co2_sfactor=np.array([co2,1]),
             v10=True, silent=True)
        
        command = ('setenv LBL_RUN_ROOT ' + vip['workdir'] + ' ; '+
                    'setenv LBL_HOME ' + vip['lblrtm_home'] + ' ; '+
                    '$LBL_HOME/bin/lblrun ' + tp5 + ' ' + out + ' ' + vip['lbl_tape3'])
        
        # Logic to catch various errors that could occur when we spawn off the LBLRTM runs
        try:
            with Popen(command, stdout = PIPE, stderr = PIPE, shell=True, executable = '/bin/csh') as process:
                stdout, stderr = process.communicate()
                    # Catch cases where the process launched, but LBLRTM exited with an error
                if process.returncode != 0:
                    print(f"LBLRTM execution (2) failed with exit code {process.returncode}")
                    print(f"Stderr output: {stderr.decode('utf-8', errors='replace')}")
                    return err
            # Catches failures occurring during process creation
        except OSError as e:
            if e.errno == errno.EAGAIN:
                print("Popen error with LBLRTM: Out of PIDs or process limit reached (EAGAIN).")
            elif e.errno == errno.ENOMEM:
                print("Popen error with LBLRTM: Out of system memory (ENOMEM).")
            elif isinstance(e, FileNotFoundError):
                print("Popen error with LBLRTM: C-shell executable '/bin/csh' was not found.")
            else:
                print(f"Popen OS error with LBLRTM ({e.errno}): {e.strerror}")
            return err
            # Catches non-OS exceptions (e.g., KeyboardInterrupt or I/O errors during communication)
        except Exception as e:
            print(f"Unexpected error running LBLRTM: {e}")
            return err

    # Confirm tha the LBLRTM ran properly
    odfiles = []
    odfiles = odfiles + sorted(glob.glob(out+'/OD*'))
    if (len(odfiles) != len(z)-1):
        print('Error within make_lblrtm_calc -- number of levels is incorrect')
        if verbose >= 1:
            print('Here is the output from the LBLRTM')
            print(stdout)
        return err
    
    return {'status':1, 'ymd':ymd, 'hour':hour, 'filename':out, 'pwv':pwv}

############################################################################################
# This routine runs the LBLDIS as the forward model.
############################################################################################

def mixcra_forward_model(Xn, z, lblout, lwc, vip, jday, sza, sfc_emissivity, ref_wnum,
                         nobs, microwin_file, retrieve_lcloud, retrieve_icloud, verbose):

    ifile = vip['workdir'] + '/lbldis.parameters'
    ofile = vip['workdir'] + '/lbldis.output'
    dfile = vip['workdir'] + '/lbldis.'

    err = 0

    # The command that will be used to run the LBLDIS
    command = 'rm -f ' + ofile + '* ; '+vip['lbldis_exec']+' '+ifile+' 0 '+ofile
    if verbose < 2:
        command = '('+command+') >& /dev/null'
    
    # Define the size of the perturbations
    pLtau = 0.1  # optical depth
    pItau = 0.1  # optical depth
    plRe = 0.2   # microns
    piRe = 0.2   # microns

    # Count the number of layers that actually have cloud
    cldlay = np.where((lwc > 0))[0]

    if len(cldlay) == 0:
        print('This should not happen -- should always have some cloud')
        return err, -999, -999
    
    # Compute forward model and jacobian for liquid cloud
    if verbose >= 2:
        print('    Computing baseline F(Xn) in forward_model')
    
    # Allocate space for the Jacobian
    Kij = np.ones((nobs,len(Xn)))*0.

    # Generate normalized liquid cloud optical depth profile
    ltau  = np.ones(len(cldlay))
    ltau *= Xn[0]/np.sum(ltau)

    # Generate normalized ice cloud optical depth profile
    itau  = np.ones(len(cldlay))
    itau *= Xn[2]/np.sum(itau)

    # Write the parameter file
    LBLRTM_Functions.write_lbldis_parmfile(ifile, sza, microwin_file, z, cldlay, Xn,
                                           ltau, itau, vip, lblout, sfc_emissivity, ref_wnum)
    
    # Now run the RT model
    # Logic to catch various errors that could occur when we spawn off the LBLRTM runs
    try:
        with Popen(command, stdout = PIPE, stderr = PIPE, shell=True, executable = '/bin/csh') as process:
            stdout, stderr = process.communicate()
                # Catch cases where the process launched, but LBLRTM exited with an error
            if process.returncode != 0:
                print(f"LBLRTM execution (3) failed with exit code {process.returncode}")
                print(f"Stderr output: {stderr.decode('utf-8', errors='replace')}")
                return err, -999, -999
        # Catches failures occurring during process creation
    except OSError as e:
        if e.errno == errno.EAGAIN:
            print("Popen error with LBLRTM: Out of PIDs or process limit reached (EAGAIN).")
        elif e.errno == errno.ENOMEM:
            print("Popen error with LBLRTM: Out of system memory (ENOMEM).")
        elif isinstance(e, FileNotFoundError):
            print("Popen error with LBLRTM: C-shell executable '/bin/csh' was not found.")
        else:
            print(f"Popen OS error with LBLRTM ({e.errno}): {e.strerror}")
        return err, -999, -999
        # Catches non-OS exceptions (e.g., KeyboardInterrupt or I/O errors during communication)
    except Exception as e:
        print(f"Unexpected error running LBLRTM: {e}")
        return err, -999, -999

    # Now read in the output data
    fid = Dataset(ofile + '.cdf')

    dim_name = list(fid.dimensions.keys())[0]
    dim_size = len(fid.dimensions[dim_name])

    if (dim_name == 'n_wnum') and (dim_size == 0):
        fid.close()
        print('Error: no radiance data found in LBLDIS output file. Most likely, need to rerun LBLRTM over new range')
        return err, -999., -999.
    
    orad = np.squeeze(fid.variables['radiance'][:])
    fid.close()

    if len(orad) != nobs:
        print('Error within forward model -- dimensions do not match up')
        print(f'   len(orad) is {len(orad):d} and nobs is {nobs:d}')
        return err, -999., -999.
    
    # Now make the perturbations needed for liquid clouds
    if retrieve_lcloud == 1:
        if verbose >= 3:
            print('      Making the perturbed liquid calculation for Jacobian')
        # Make the run with the perturbed LWP

        pXn     = np.copy(Xn)
        pLtau   = np.max([Xn[0] * 0.1, pLtau]) # Perturb OD by 10% or 0.1 which ever is larger
        pXn[0] += pLtau
        plod    = np.ones(len(cldlay))
        plod   *= pXn[0]/np.sum(plod)

        LBLRTM_Functions.write_lbldis_parmfile(ifile, sza, microwin_file, z, cldlay, Xn,
                                           plod, itau, vip, lblout, sfc_emissivity, ref_wnum)

        # Now run the RT model
        # Logic to catch various errors that could occur when we spawn off the LBLRTM runs
        try:
            with Popen(command, stdout = PIPE, stderr = PIPE, shell=True, executable = '/bin/csh') as process:
                stdout, stderr = process.communicate()
                    # Catch cases where the process launched, but LBLRTM exited with an error
                if process.returncode != 0:
                    print(f"LBLRTM execution (4) failed with exit code {process.returncode}")
                    print(f"Stderr output: {stderr.decode('utf-8', errors='replace')}")
                    return err, -999, -999
            # Catches failures occurring during process creation
        except OSError as e:
            if e.errno == errno.EAGAIN:
                print("Popen error with LBLRTM: Out of PIDs or process limit reached (EAGAIN).")
            elif e.errno == errno.ENOMEM:
                print("Popen error with LBLRTM: Out of system memory (ENOMEM).")
            elif isinstance(e, FileNotFoundError):
                print("Popen error with LBLRTM: C-shell executable '/bin/csh' was not found.")
            else:
                print(f"Popen OS error with LBLRTM ({e.errno}): {e.strerror}")
            return err, -999, -999
            # Catches non-OS exceptions (e.g., KeyboardInterrupt or I/O errors during communication)
        except Exception as e:
            print(f"Unexpected error running LBLRTM: {e}")
            return err, -999, -999

        fid = Dataset(ofile + '.cdf')
        prad = np.squeeze(fid.variables['radiance'][:])
        fid.close()

        Kij[:,0] = (prad-orad)/(pXn[0] - Xn[0])

        # Make the run with the perturbed lReff
        #      but only if the liquid OD is greater than zero
        if Xn[0] > 0:
            pXn = np.copy(pXn)            # Note this line was not in the IDL code but I added it because I think it was a bug
            pXn[1] += plRe 

            LBLRTM_Functions.write_lbldis_parmfile(ifile, sza, microwin_file, z, cldlay, pXn,
                                           ltau, itau, vip, lblout, sfc_emissivity, ref_wnum)
        
            # Now run the RT model
            # Logic to catch various errors that could occur when we spawn off the LBLRTM runs
            try:
                with Popen(command, stdout = PIPE, stderr = PIPE, shell=True, executable = '/bin/csh') as process:
                    stdout, stderr = process.communicate()
                        # Catch cases where the process launched, but LBLRTM exited with an error
                    if process.returncode != 0:
                        print(f"LBLRTM execution (5) failed with exit code {process.returncode}")
                        print(f"Stderr output: {stderr.decode('utf-8', errors='replace')}")
                        return err, -999, -999
                # Catches failures occurring during process creation
            except OSError as e:
                if e.errno == errno.EAGAIN:
                    print("Popen error with LBLRTM: Out of PIDs or process limit reached (EAGAIN).")
                elif e.errno == errno.ENOMEM:
                    print("Popen error with LBLRTM: Out of system memory (ENOMEM).")
                elif isinstance(e, FileNotFoundError):
                    print("Popen error with LBLRTM: C-shell executable '/bin/csh' was not found.")
                else:
                    print(f"Popen OS error with LBLRTM ({e.errno}): {e.strerror}")
                return err, -999, -999
                # Catches non-OS exceptions (e.g., KeyboardInterrupt or I/O errors during communication)
            except Exception as e:
                print(f"Unexpected error running LBLRTM: {e}")
                return err, -999, -999

            fid = Dataset(ofile + '.cdf')
            prad = np.squeeze(fid.variables['radiance'][:])
            fid.close()

            Kij[:,1] = (prad-orad)/(pXn[1] - Xn[1])
        else:
            Kij[:,1] = 0

     # Now make the perturbations needed for ice clouds
    if retrieve_icloud == 1:
        if verbose > 3:
            print('      Making the perturbed ice calculations for Jacobian')

         # Make the run with the perturbed ice optical depth
        pXn     = np.copy(Xn)
        pItau   = np.max([Xn[2] * 0.1, pItau]) # Perturb OD by 10% or 0.1 which ever is larger
        pXn[2] += pItau
        piod    = np.ones(len(cldlay))
        piod   *= pXn[2]/np.sum(piod)

        LBLRTM_Functions.write_lbldis_parmfile(ifile, sza, microwin_file, z, cldlay, Xn,
                                           ltau, piod, vip, lblout, sfc_emissivity, ref_wnum)

        # Now run the RT model
        # Logic to catch various errors that could occur when we spawn off the LBLRTM runs
        try:
            with Popen(command, stdout = PIPE, stderr = PIPE, shell=True, executable = '/bin/csh') as process:
                stdout, stderr = process.communicate()
                    # Catch cases where the process launched, but LBLRTM exited with an error
                if process.returncode != 0:
                    print(f"LBLRTM execution (6) failed with exit code {process.returncode}")
                    print(f"Stderr output: {stderr.decode('utf-8', errors='replace')}")
                    return err, -999, -999
            # Catches failures occurring during process creation
        except OSError as e:
            if e.errno == errno.EAGAIN:
                print("Popen error with LBLRTM: Out of PIDs or process limit reached (EAGAIN).")
            elif e.errno == errno.ENOMEM:
                print("Popen error with LBLRTM: Out of system memory (ENOMEM).")
            elif isinstance(e, FileNotFoundError):
                print("Popen error with LBLRTM: C-shell executable '/bin/csh' was not found.")
            else:
                print(f"Popen OS error with LBLRTM ({e.errno}): {e.strerror}")
            return err, -999, -999
            # Catches non-OS exceptions (e.g., KeyboardInterrupt or I/O errors during communication)
        except Exception as e:
            print(f"Unexpected error running LBLRTM: {e}")
            return err, -999, -999

        fid = Dataset(ofile + '.cdf')
        prad = np.squeeze(fid.variables['radiance'][:])
        fid.close()

        Kij[:,2] = (prad-orad)/(pXn[2] - Xn[2])

        # Make the run with the perturbed iReff
        #      but only if the liquid OD is greater than zero
        if Xn[2] > 0:
            pXn = np.copy(Xn)
            pXn[3] += piRe

            LBLRTM_Functions.write_lbldis_parmfile(ifile, sza, microwin_file, z, cldlay, pXn,
                                           ltau, itau, vip, lblout, sfc_emissivity, ref_wnum)
        
            # Now run the RT model
            # Logic to catch various errors that could occur when we spawn off the LBLRTM runs
            try:
                with Popen(command, stdout = PIPE, stderr = PIPE, shell=True, executable = '/bin/csh') as process:
                    stdout, stderr = process.communicate()
                        # Catch cases where the process launched, but LBLRTM exited with an error
                    if process.returncode != 0:
                        print(f"LBLRTM execution (7) failed with exit code {process.returncode}")
                        print(f"Stderr output: {stderr.decode('utf-8', errors='replace')}")
                        return err, -999, -999
                # Catches failures occurring during process creation
            except OSError as e:
                if e.errno == errno.EAGAIN:
                    print("Popen error with LBLRTM: Out of PIDs or process limit reached (EAGAIN).")
                elif e.errno == errno.ENOMEM:
                    print("Popen error with LBLRTM: Out of system memory (ENOMEM).")
                elif isinstance(e, FileNotFoundError):
                    print("Popen error with LBLRTM: C-shell executable '/bin/csh' was not found.")
                else:
                    print(f"Popen OS error with LBLRTM ({e.errno}): {e.strerror}")
                return err, -999, -999
                # Catches non-OS exceptions (e.g., KeyboardInterrupt or I/O errors during communication)
            except Exception as e:
                print(f"Unexpected error running LBLRTM: {e}")
                return err, -999, -999

            fid = Dataset(ofile + '.cdf')
            prad = np.squeeze(fid.variables['radiance'][:])
            fid.close()

            Kij[:,3] = (prad-orad)/(pXn[3] - Xn[3])
        else:
            Kij[:,3] = 0
    
    FXn = orad

    return 1, FXn, Kij
