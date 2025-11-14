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

import os
import numpy as np
import scipy.io
import glob
import sys
from datetime import datetime
from netCDF4 import Dataset

import Other_functions
import Calcs_Conversions
import VIP_Databases_functions

################################################################################
# This file contains the following functions:
# write_example_vip_file()
# add_vip_to_global_atts()
# write_variable()
# write_output()
# create_xret()
################################################################################

################################################################################
# This function will write out an example VIP file, and then stop
################################################################################
def write_example_vip_file(experimental=False, console=False):

    # Grab the full vip from VIP_Databases_functions
    vip = VIP_Databases_functions.full_vip.copy()

    # build a list of vip entries
    # output_lines = [f"{key}: {data['value']} # {data['comment']}" for key, data in vip.items() if data['default'] is True]
    output_lines = []
    for key, data in vip.items():
        # See if the value is a list. If so change to a string to keep things consistent
        if type(data['value']) is list:
            data['value'] = str(data['value'])[1:-1]  # Convert to string and remove the brackets

        if experimental:
            output_lines.append(f"{key} = {data['value']} # {data['comment']}")
        elif not experimental and data['default']:
            output_lines.append(f"{key} = {data['value']} # {data['comment']}")
        else:
            continue

    # Write out the default vip to a file
    with open('default_pyVIP.txt', 'w') as fh:
        for line in output_lines:
            fh.write(line + '\n')

    # If desired, write to console
    if console:
        for l in output_lines:
            print(l)

    sys.exit()

################################################################################
# This function takes all of the entries in the VIP structure, and makes
# them global attributes in the netCDF file that is currently open
################################################################################
def add_vip_to_global_atts(nc, full_vip):
    """
    Function to write out the vip vile into the global attributes
    :param nc: NetCDF4 Dataset object to write the VIP to
    :param vip: Dict containing the vip information
    :return:
    """

    vip = full_vip.copy()  # Copy the VIP so we don't overwrite anything in the next iterations

    for key in sorted(vip.keys()):
        # Grab the data
        data = vip[key]

        # Convert the spectral bands back to a string since we can't save arrays to the attributes
        if key == 'spectral_bands':
            data = ','.join([f"{data[0, i]}-{data[1, i]}" for i in range(len(data[0])) if data[0, i] > 0])

        # If the attribute is an integer, cast it as a 32-bit one (otherwise it defaults to a long int)
        #     This was a request of the ARM program (since none of the global attributes need this precision)
        if isinstance(data,int):
            data = np.int32(data)

        # Set the netcdf attribute
        nc.setncattr(f"VIP_{key}", data)

    return

################################################################################
# This function writes a variable into a netCDF files for debugging purposes
################################################################################

def write_variable(variable, filename, verbose=3):
    print(f'  DEBUG -- Creating the file {filename:s}')
    dims = np.shape(variable)
    if len(dims) == 0:
        if verbose > 1:
            print(f'     The variable is a single variable')
    elif len(dims) == 1:
        if verbose > 1:
            print(f'     The variable is a vector')
            print(f'        with a dimension of {dims[0]:d}')
        fid = Dataset(filename, 'w')
        did = fid.createDimension('dim', dims[0])
        var = fid.createVariable('var','f8',('dim',))
        var[:] = variable
        fid.close()
    elif len(dims) == 2:
        if verbose > 1:
            print(f'     The variable is a matrix')
            print(f'        with dimensions of {dims[0]:d} and {dims[1]:d}')
        fid = Dataset(filename, 'w')
        did = fid.createDimension('dim1', dims[0])
        did = fid.createDimension('dim2', dims[1])
        var = fid.createVariable('var','f8',('dim1','dim2',))
        var[:] = variable
        fid.close()
    else:
        print('    This is a 3+ dimension variable ??  This code needs to be modified to handle this...')
    return

################################################################################
# This function writes out netCDF files with the output
################################################################################

def write_output(vip, ext_prof, mod_prof, ext_tseries, globatt, xret, prior,
                fsample, derived, dindex, version, exectime, modeflag, nfilename, location,
                cbh_string, shour, verbose):

    success = 0
    # I will replace all temp/WVMR data below the chimney height with this
    # flag
    #nochim = -888.

    # These are the derived indices that I will compute later one. I need to
    # define them here in order to build the netcdf file correctly
    nht = len(xret[0]['z'])
    
    # The strings that will be added to the various fields, depending on what their use is
    varType_observation    = 'Observation used within retrieval'
    varType_retrievalOn    = 'Retrieved'
    varType_retrievalOff   = 'Retrieved but disabled -- will default to the a priori'
    varType_uncertainty    = 'Uncertainty of associated variable'
    varType_derived        = 'Derived from retrieved variables'
    varType_qualityControl = 'Quality control'
    varType_diagnostic     = 'Diagnostic variable'

    # If fsample is zero, then we will create the netCDF file
    if fsample == 0:
        dt = datetime.utcfromtimestamp(xret[0]['secs'])
        hh = datetime.utcfromtimestamp(xret[0]['secs']).hour
        nn = datetime.utcfromtimestamp(xret[0]['secs']).minute
        ss = datetime.utcfromtimestamp(xret[0]['secs']).second
        hms = hh*10000 + nn*100 + ss

        nfilename = vip['output_path'] + '/' + vip['output_rootname'] + '.' + dt.strftime('%Y%m%d.%H%M%S') + '.nc'
        
        
        if ((os.path.exists(nfilename)) & (vip['output_clobber'] == 0)):
            print('Error: output file exists -- aborting (' + nfilename + ')')
            return success, nfilename
        elif (os.path.exists(nfilename)):
            print('Warning: clobbering existing output file (' +nfilename + ')')

        fid = Dataset(nfilename, 'w')
        tdim = fid.createDimension('time', None)
        hdim = fid.createDimension('height', nht)
        vdim = fid.createDimension('obs_dim', len(xret[0]['dimY']))
        gdim = fid.createDimension('gas_dim', 3)
        ddim = fid.createDimension('dfs_dim', len(xret[0]['dfs']))
        adim1 = fid.createDimension('arb_dim1', len(xret[0]['Xn']))
        adim2 = fid.createDimension('arb_dim2', len(xret[0]['Xn']))

        base_time = fid.createVariable('base_time','i4')
        base_time.long_name = 'Epoch time'
        base_time.units = 'seconds since 1970-01-01 00:00:00 0:00'

        time_offset = fid.createVariable('time_offset', 'f8', ('time',))
        time_offset.long_name = 'Time offset from base_time'
        time_offset.units = 'Seconds'

        time = fid.createVariable('time', 'f8', ('time',))
        time.long_name = 'Time'
        time.units = 'seconds since ' + dt.strftime('%Y-%m-%d') +' 00:00:00 0:00'
        time.calendar = 'standard'

        hour = fid.createVariable('hour', 'f8', ('time',))
        hour.long_name = 'Time'
        hour.units = 'hours since ' + dt.strftime('%Y-%m-%d') +' 00:00:00 0:00'

        qc_flag = fid.createVariable('qc_flag', 'i2', ('time',))
        qc_flag.long_name = 'Manual QC flag'
        qc_flag.variable_type = varType_qualityControl
        qc_flag.comment1 = 'unitless'
        qc_flag.comment2 = 'Value of 0 implies quality of the retrieved fields is ok; non-zero values indicate that the sample has suspect quality'
        qc_flag.comment3 = 'See the global attribute Retrieval_option_flags to determine which variables were being retrieved for this data file'
        qc_flag.value_0 = 'Quality of retrieval is ok'
        qc_flag.value_2 = 'Implies retrieval did not converge'
        qc_flag.value_3 = 'Implies retrieval converged but RMS between the observed_vector and forward_calc is too large'
        qc_flag.value_4 = 'Implies the gamma value of the retrieval was too large'
        qc_flag.RMSa_threshold_used_for_QC = str(vip['qc_rms_value']) + ' [unitless]'
        qc_flag.gamma_threshold_used_for_QC = str(vip['qc_gamma_value']) + ' [unitless]'

        height = fid.createVariable('height', 'f4', ('height',))
        height.long_name = 'Height above ground level'
        height.units = 'km'

        temperature = fid.createVariable('temperature', 'f4', ('time','height',))
        temperature.long_name = 'Temperature'
        temperature.units = 'degC'
        if(modeflag[0] == 1):
            temperature.variable_type = varType_retrievalOn
        else:
            temperature.variable_type = varType_retrievalOff

        waterVapor = fid.createVariable('waterVapor', 'f4', ('time', 'height',))
        waterVapor.long_name = 'Water vapor mixing ratio'
        waterVapor.units = 'g/kg'
        if(modeflag[1] == 1):
            waterVapor.variable_type = varType_retrievalOn
        else:
            waterVapor.variable_type = varType_retrievalOff

        lwp = fid.createVariable('lwp', 'f4', ('time',))
        lwp.long_name = 'Liquid water path'
        lwp.units = 'g/m2'
        if(modeflag[2] == 1):
            lwp.variable_type = varType_retrievalOn
        else:
            lwp.variable_type = varType_retrievalOff

        lReff = fid.createVariable('lReff', 'f4', ('time',))
        lReff.long_name = 'Liquid water effective radius'
        lReff.units = 'microns'
        if(modeflag[2] == 1):
            lReff.variable_type = varType_retrievalOn
        else:
            lReff.variable_type = varType_retrievalOff

        iTau = fid.createVariable('iTau', 'f4', ('time',))
        iTau.long_name = 'Ice cloud optical depth (geometric limit)'
        if(modeflag[3] == 1):
            iTau.variable_type = varType_retrievalOn
        else:
            iTau.variable_type = varType_retrievalOff
        iTau.comment1 = 'unitless'

        iReff = fid.createVariable('iReff', 'f4', ('time',))
        iReff.long_name = 'Ice effective radius'
        iReff.units = 'microns'
        if(modeflag[3] == 1):
            iReff.variable_type = varType_retrievalOn
        else:
            iReff.variable_type = varType_retrievalOff

        co2 = fid.createVariable('co2', 'f4', ('time','gas_dim',))
        co2.long_name = 'Carbon dioxide concentration'
        co2.units = 'ppm'
        if(modeflag[4] == 1):
            co2.variable_type = varType_retrievalOn
        else:
            co2.variable_type = varType_retrievalOff
        co2.comment = 'Parameterized profile information; see users guide'

        ch4 = fid.createVariable('ch4', 'f4', ('time', 'gas_dim',))
        ch4.long_name = 'Methane concentration'
        ch4.units = 'ppm'
        if(modeflag[5] == 1):
            ch4.variable_type = varType_retrievalOn
        else:
            ch4.variable_type = varType_retrievalOff
        ch4.comment = 'Parameterized profile information; see users guide'

        n2o = fid.createVariable('n2o', 'f4', ('time', 'gas_dim',))
        n2o.long_name = 'Nitrous oxide concentration'
        n2o.units = 'ppm'
        if(modeflag[6] == 1):
            n2o.variable_type = varType_retrievalOn
        else:
            n2o.variable_type = varType_retrievalOff
        n2o.comment = 'Parameterized profile information; see users guide'

        sigmaT = fid.createVariable('sigma_temperature', 'f4', ('time','height',))
        sigmaT.long_name = '1-sigma uncertainty in temperature'
        sigmaT.units = 'degC'
        sigmaT.variable_type = varType_uncertainty

        sigmaWV = fid.createVariable('sigma_waterVapor', 'f4', ('time','height',))
        sigmaWV.long_name = '1-sigma uncertainty in water vapor mixing vapor'
        sigmaWV.units = 'g/kg'
        sigmaWV.variable_type = varType_uncertainty

        sigma_lwp = fid.createVariable('sigma_lwp', 'f4', ('time',))
        sigma_lwp.long_name = '1-sigma uncertainty in liquid water path'
        sigma_lwp.units = 'g/m2'
        sigma_lwp.variable_type = varType_uncertainty

        sigma_lReff = fid.createVariable('sigma_lReff', 'f4', ('time',))
        sigma_lReff.long_name = '1-sigma uncertainty in liquid water effective radius'
        sigma_lReff.units = 'microns'
        sigma_lReff.variable_type = varType_uncertainty

        sigma_iTau = fid.createVariable('sigma_iTau', 'f4', ('time',))
        sigma_iTau.long_name = '1-sigma uncertainty in ice cloud optical depth (geometric limit)'
        sigma_iTau.variable_type = varType_uncertainty
        sigma_iTau.comment1 = 'unitless'

        sigma_iReff = fid.createVariable('sigma_iReff', 'f4', ('time',))
        sigma_iReff.long_name = '1-sigma uncertainty in ice effective radius'
        sigma_iReff.units = 'microns'
        sigma_iReff.variable_type = varType_uncertainty

        sigma_co2 = fid.createVariable('sigma_co2', 'f4', ('time','gas_dim',))
        sigma_co2.long_name = '1-sigma uncertainty in carbon dioxide concentration'
        sigma_co2.units = 'ppm'
        sigma_co2.variable_type = varType_uncertainty
        sigma_co2.comment = 'Parameterized profile information; see users guide'

        sigma_ch4 = fid.createVariable('sigma_ch4', 'f4', ('time','gas_dim',))
        sigma_ch4.long_name = '1-sigma uncertainty in methane concentration'
        sigma_ch4.units = 'ppm'
        sigma_ch4.variable_type = varType_uncertainty
        sigma_ch4.comment = 'Parameterized profile information; see users guide'

        sigma_n2o = fid.createVariable('sigma_n2o', 'f4', ('time','gas_dim',))
        sigma_n2o.long_name = '1-sigma uncertaintiy in nitrous oxide concentration'
        sigma_n2o.units = 'ppm'
        sigma_n2o.variable_type = varType_uncertainty
        sigma_n2o.comment = 'Parameterized profile information; see users guide'

        converged_flag = fid.createVariable('converged_flag', 'i2', ('time',))
        converged_flag.long_name = 'Convergence flag'
        converged_flag.variable_type = varType_qualityControl
        converged_flag.comment1 = 'unitless'
        converged_flag.value_0 = '0 indicates no convergence'
        converged_flag.value_1 = '1 indicates convergence in Rodgers sense (i.e., di2m << dimY)'
        converged_flag.value_2 = '2 indicates convergence (best rms after rms increased drastically'
        converged_flag.value_3 = '3 indicates convergence (best rms after max_iter)'
        converged_flag.value_9 = '9 indicates found NaN in Xnp1'

        gamma = fid.createVariable('gamma', 'f4', ('time',))
        gamma.long_name = 'Gamma parameter'
        gamma.variable_type = varType_diagnostic
        gamma.comment1 = 'unitless'
        gamma.comment2 = 'See Turner and Loehnert JAMC 2014 for details'

        n_iter = fid.createVariable('n_iter', 'i2', ('time',))
        n_iter.long_name = 'Number of iterations performed'
        n_iter.variable_type = varType_diagnostic
        n_iter.comment1 = 'unitless'

        rmsr = fid.createVariable('rmsr', 'f4', ('time',))
        rmsr.long_name = 'Root mean square error between IRS and MWR obs in the observation vector and the forward calculation'
        rmsr.variable_type = varType_qualityControl
        rmsr.comment1 = 'unitless'
        rmsr.comment2 = 'Computed as sqrt( sum_over_i[ ((Y_i - F(Xn_i)) / sigma_Y_i)^2 ] / sizeY)'
        rmsr.comment3 = 'Only IRS radiance observations in the observation vector are used'
        
        rmsa = fid.createVariable('rmsa', 'f4', ('time',))
        rmsa.long_name = 'Root mean square error between observation vector and the forward calculation'
        rmsa.variable_type = varType_qualityControl
        rmsa.comment1 = 'unitless'
        rmsa.comment2 = 'Computed as sqrt( sum_over_i[ ((Y_i - F(Xn_i)) / sigma_Y_i)^2 ] / sizeY)'
        rmsa.comment3 = 'Entire observation vector used in this calculation'
        
        rmsp = fid.createVariable('rmsp', 'f4', ('time',))
        rmsp.long_name = 'Root mean square error between prior T/q profile and the retrieved T/q profile'
        rmsp.variable_type = varType_diagnostic
        rmsp.comment1 = 'unitless'
        rmsp.comment2 = 'Computed as sqrt( mean[ ((Xa - Xn) / sigma_Xa)^2 ] )'
        
        chi2 = fid.createVariable('chi2', 'f4', ('time',))
        chi2.long_name = 'Chi-square statistic of Y vs. F(Xn)'
        chi2.variable_type = varType_diagnostic
        chi2.comment1 = 'unitless'
        chi2.comment2 = 'Computed as sqrt( sum_over_i[ ((Y_i - F(Xn_i)) / Y_i)^2 ] / sizeY)'
        
        convergence_criteria = fid.createVariable('convergence_criteria', 'f4', ('time',))
        convergence_criteria.long_name = 'Convergence criteria di^2'
        convergence_criteria.variable_type = varType_diagnostic
        convergence_criteria.comment1 = 'unitless'

        dfs = fid.createVariable('dfs', 'f4', ('time','dfs_dim',))
        dfs.long_name = 'Degrees of freedom of signal'
        dfs.variable_type = varType_diagnostic
        dfs.comment1 = 'unitless'
        dfs.comment2 = 'total DFS, then DFS for each of temperature, waterVapor, LWP, L_Reff, I_tau, I_Reff, carbonDioxide, methane, nitrousOxide'
        
        dfs_nm = fid.createVariable('dfs_no_model', 'f4', ('time', 'dfs_dim',))
        dfs_nm.long_name = 'Degrees of freedom of signal excluding model data'
        dfs_nm.variable_type = varType_diagnostic
        dfs_nm.comment1 = 'unitless'
        dfs_nm.comment2 = 'total DFS, then DFS for each of temperature, waterVapor, LWP, L_Reff, I_tau, I_Reff, carbonDioxide, methane, nitrousOxide'
        dfs_nm.comment3 = 'If no model data is used in the obs vector this field will be the same as dfs'
        
        sic = fid.createVariable('sic', 'f4', ('time',))
        sic.long_name = 'Shannon information content'
        sic.variable_type = varType_diagnostic
        sic.comment1 = 'unitless'

        vres_temp = fid.createVariable('vres_temperature', 'f4', ('time','height',))
        vres_temp.long_name = 'Vertical resolution of the temperature profile'
        vres_temp.units = 'km'
        vres_temp.variable_type = varType_diagnostic
        
        vres_temp_nm = fid.createVariable('vres_temperature_no_model', 'f4', ('time','height',))
        vres_temp_nm.long_name = 'Vertical resolution of the temperature profile exluding model data'
        vres_temp_nm.units = 'km'
        vres_temp_nm.variable_type = varType_diagnostic
        vres_temp_nm.comment1 = 'If no model data is used in the obs vector this field will be the same as vres_temperature'

        vres_wv = fid.createVariable('vres_waterVapor', 'f4', ('time','height',))
        vres_wv.long_name = 'Vertical resolution of the water vapor profile'
        vres_wv.units = 'km'
        vres_wv.variable_type = varType_diagnostic
        
        vres_wv_nm = fid.createVariable('vres_waterVapor_no_model', 'f4', ('time','height',))
        vres_wv_nm.long_name = 'Vertical resolution of the water vapor profile exluding model data'
        vres_wv_nm.units = 'km'
        vres_wv_nm.variable_type = varType_diagnostic
        vres_wv_nm.comment1 = 'If no model data is used in the obs vector this field will be the same as vres_waterVapor'
        
        cdfs_temp = fid.createVariable('cdfs_temperature', 'f4', ('time','height',))
        cdfs_temp.long_name = 'Vertical profile of the cumulative degrees of freedom of signal for temperature'
        cdfs_temp.variable_type = varType_diagnostic
        cdfs_temp.comment1 = 'unitless'
        
        cdfs_temp_nm = fid.createVariable('cdfs_temperature_no_model', 'f4', ('time','height',))
        cdfs_temp_nm.long_name = 'Vertical profile of the cumulative degrees of freedom of signal for temperature excluding model data'
        cdfs_temp_nm.variable_type = varType_diagnostic
        cdfs_temp_nm.comment1 = 'unitless'
        cdfs_temp_nm.comment2 = ' If no model data is used in the obs vector this field will be the same as cdfs_temperature'
        
        cdfs_wv = fid.createVariable('cdfs_waterVapor', 'f4', ('time','height',))
        cdfs_wv.long_name = 'Vertical profile of the cumulative degrees of freedom of signal for water vapor'
        cdfs_wv.variable_type = varType_diagnostic
        cdfs_wv.comment1 = 'unitless'
        
        cdfs_wv_nm = fid.createVariable('cdfs_waterVapor_no_model', 'f4', ('time','height',))
        cdfs_wv_nm.long_name = 'Vertical profile of the cumulative degrees of freedom of signal for water vapor excluding model data'
        cdfs_wv_nm.variable_type = varType_diagnostic
        cdfs_wv_nm.comment1 = 'unitless'
        cdfs_wv_nm.comment2 = 'If no model data is used in the obs vector this field will be the same as cdfs_waterVapor'

        cbh = fid.createVariable('cbh', 'f4', ('time',))
        cbh.long_name = 'Cloud base height above ground level'
        cbh.units = 'km'
        cbh.variable_type = varType_observation

        cbh_flag = fid.createVariable('cbh_flag', 'i2', ('time',))
        cbh_flag.long_name = 'Flag indicating the source of the cbh'
        cbh_flag.variable_type = varType_diagnostic
        cbh_flag.comment1 = 'unitless'
        cbh_flag.comment2 = 'See users guide for explanation of the values'
        cbh_flag.value0 = 'Value 0 implies Clear Sky'
        cbh_flag.value1 = 'Value 1 implies Inner Window from ceilometer'
        cbh_flag.value2 = 'Value 2 implies Outer Window from ceilometer'
        cbh_flag.value3 = 'Value 3 implies Default CBH'

        cbh_tcld = fid.createVariable('cbh_tcld', 'f4', ('time',))
        cbh_tcld.long_name = 'Cloud base height above ground level using Tcld method'
        cbh_tcld.units = 'km'
        cbh_tcld.variable_type = varType_derived
        cbh_tcld.spectral_range = (f"Spectral window used is {vip['irs_tcld_wnum_range'][0]:.2f} to {vip['irs_tcld_wnum_range'][1]:.2f} cm-1")

        cbh_mlev = fid.createVariable('cbh_mlev', 'f4', ('time',))
        cbh_mlev.long_name = 'Cloud base height above ground level using MLEV method'
        cbh_mlev.units = 'km'
        cbh_mlev.variable_type = varType_derived
        cbh_mlev.spectral_range = (f"Spectral window used is {vip['irs_mlev_cbh_wnum_range'][0]:.2f} to {vip['irs_mlev_cbh_wnum_range'][1]:.2f} cm-1")

        cld_emis = fid.createVariable('cld_emis', 'f4', ('time',))
        cld_emis.long_name = 'Cloud emissivity derived using the MLEV method'
        cld_emis.comment1 = 'unitless'
        cld_emis.variable_type = varType_derived
        cld_emis.spectral_range = (f"Spectral window used is {vip['irs_mlev_cbh_wnum_range'][0]:.2f} to {vip['irs_mlev_cbh_wnum_range'][1]:.2f} cm-1")

        pressure = fid.createVariable('pressure', 'f4', ('time','height',))
        pressure.long_name = 'Derived pressure'
        pressure.units = 'mb'
        pressure.variable_type = varType_derived
        pressure.comment = 'derived from surface pressure observations and the hyposmetric calculation using the thermodynamic profiles'

        theta = fid.createVariable('theta', 'f4', ('time','height',))
        theta.long_name = 'Potential temperature'
        theta.units = 'degK'
        theta.variable_type = varType_derived
        theta.comment = 'This field is derived from the retrieved fields'

        thetae = fid.createVariable('thetae', 'f4', ('time','height',))
        thetae.long_name = 'Equivalent potential temperature'
        thetae.units = 'degK'
        thetae.variable_type = varType_derived
        thetae.comment = 'This field is derived from the retrieved fields'

        rh = fid.createVariable('rh', 'f4', ('time','height',))
        rh.long_name = 'Relative humidity'
        rh.units = '%'
        rh.variable_type = varType_derived
        rh.comment = 'This field is derived from the retrieved field'

        dewpt = fid.createVariable('dewpt', 'f4', ('time','height',))
        dewpt.long_name = 'Dew point temperature'
        dewpt.units = 'degC'
        dewpt.variable_type = varType_derived
        dewpt.comment = 'This field is derived from the retrieved fields'

        co2_profile = fid.createVariable('co2_profile', 'f4', ('time','height',))
        co2_profile.long_name = 'CO2 profile'
        co2_profile.units = 'ppm'
        co2_profile.variable_type = varType_derived
        co2_profile.comment = 'This field is derived from the retrieved fields'

        ch4_profile = fid.createVariable('ch4_profile', 'f4', ('time','height',))
        ch4_profile.long_name = 'CH4 profile'
        ch4_profile.units = 'ppm'
        ch4_profile.variable_type = varType_derived
        ch4_profile.comment = 'This field is derived from the retrieved fields'

        n2o_profile = fid.createVariable('n2o_profile', 'f4', ('time','height',))
        n2o_profile.long_name = 'N2O profile'
        n2o_profile.units = 'ppm'
        n2o_profile.variable_type = varType_derived
        n2o_profile.comment = 'This field is derived from the retrieved fields'

        pwv = fid.createVariable('pwv', 'f4', ('time',))
        pwv.long_name = 'Precipitable water vapor'
        pwv.units = dindex['units'][0]
        pwv.variable_type = varType_derived
        pwv.comment1 = 'This field is derived from the retrieved fields'
        pwv.comment2 = 'A value of -999 indicates that this field could not be computed (typically because the value was aphysical)'
        
        pblh = fid.createVariable('pblh', 'f4', ('time',))
        pblh.long_name = 'Planetary boundary layer height'
        pblh.units = dindex['units'][1]
        pblh.variable_type = varType_derived
        pblh.comment1 = 'This field is derived from the retrieved fields'
        pblh.comment2 = 'A value of -999 indicates that this field could not be computed (typically because the value was aphysical)'
        
        sbih = fid.createVariable('sbih', 'f4', ('time',))
        sbih.long_name = 'Surface-based inversion height'
        sbih.units = dindex['units'][2]
        sbih.variable_type = varType_derived
        sbih.comment1 = 'This field is derived from the retrieved fields'
        sbih.comment2 = 'A value of -999 indicates that this field could not be computed (typically because the value was aphysical)'
        
        sbim = fid.createVariable('sbim', 'f4', ('time',))
        sbim.long_name = 'Surface-based inversion magnitude'
        sbim.units = dindex['units'][3]
        sbim.variable_type = varType_derived
        sbim.comment1 = 'This field is derived from the retrieved fields'
        sbim.comment2 = 'A value of -999 indicates that this field could not be computed (typically because the value was aphysical)'
        
        sblcl = fid.createVariable('sbLCL', 'f4', ('time',))
        sblcl.long_name = 'Lifted condesation level for a surface-based parcel'
        sblcl.units = dindex['units'][4]
        sblcl.variable_type = varType_derived
        sblcl.comment1 = 'This field is derived from the retrieved fields'
        sblcl.comment2 = 'A value of -999 indicates that this field could not be computed (typically because the value was aphysical)'
        
        sbcape = fid.createVariable('sbCAPE', 'f4', ('time',))
        sbcape.long_name = 'Convective available potential energy for a surface-based parcel'
        sbcape.units = dindex['units'][5]
        sbcape.variable_type = varType_derived
        sbcape.comment1 = 'This field is derived from the retrieved fields'
        sbcape.comment2 = 'A value of -9999 indicates that this field could not be computed (typically because the value was aphysical)'
        
        sbcin = fid.createVariable('sbCIN', 'f4', ('time',))
        sbcin.long_name = 'Convective inhibition for a surface-based parcel'
        sbcin.units = dindex['units'][6]
        sbcin.variable_type = varType_derived
        sbcin.comment1 = 'This field is derived from the retrieved fields'
        sbcin.comment2 = 'A value of -9999 indicates that this field could not be computed (typically because the value was aphysical)'
        
        mllcl = fid.createVariable('mlLCL', 'f4', ('time',))
        mllcl.long_name = 'Lifted condesation level for a mixed-layer parcel'
        mllcl.units = dindex['units'][7]
        mllcl.variable_type = varType_derived
        mllcl.comment1 = 'This field is derived from the retrieved fields'
        mllcl.comment2 = 'A value of -999 indicates that this field could not be computed (typically because the value was aphysical)'
        
        mlcape = fid.createVariable('mlCAPE', 'f4', ('time',))
        mlcape.long_name = 'Convective available potential energy for a mixed-layer parcel'
        mlcape.units = dindex['units'][8]
        mlcape.variable_type = varType_derived
        mlcape.comment1 = 'This field is derived from the retrieved fields'
        mlcape.comment2 = 'A value of -9999 indicates that this field could not be computed (typically because the value was aphysical)'
        
        mlcin = fid.createVariable('mlCIN', 'f4', ('time',))
        mlcin.long_name = 'Convective inhibition for a mixed-layer parcel'
        mlcin.units = dindex['units'][9]
        mlcin.variable_type = varType_derived
        mlcin.comment1 = 'This field is derived from the retrieved fields'
        mlcin.comment2 = 'A value of -9999 indicates that this field could not be computed (typically because the value was aphysical)'

        sigma_pwv = fid.createVariable('sigma_pwv', 'f4', ('time',))
        sigma_pwv.long_name = '1-sigma uncertainties in precipitable water vapor'
        sigma_pwv.units = dindex['units'][0]
        sigma_pwv.variable_type = varType_uncertainty
        sigma_pwv.comment1 = 'This field is derived from the retrieved fields'
        sigma_pwv.comment2 = 'The uncertainties were determined using a monte carlo sampling of the posterior covariance matrix'
        sigma_pwv.comment3 = 'A value of -999 indicates that the uncertainty in this inded could not be computed (typically because the values were all unphysical)'

        sigma_pblh = fid.createVariable('sigma_pblh', 'f4', ('time',))
        sigma_pblh.long_name = '1-sigma uncertainties in the PBL height'
        sigma_pblh.units = dindex['units'][1]
        sigma_pblh.variable_type = varType_uncertainty
        sigma_pblh.comment1 = 'This field is derived from the retrieved fields'
        sigma_pblh.comment2 = 'The uncertainties were determined using a monte carlo sampling of the posterior covariance matrix'
        sigma_pblh.comment3 = 'A value of -999 indicates that the uncertainty in this inded could not be computed (typically because the values were all unphysical)'
        
        sigma_sbih = fid.createVariable('sigma_sbih', 'f4', ('time',))
        sigma_sbih.long_name = '1-sigma uncertainties in the surface-based inversion height'
        sigma_sbih.units = dindex['units'][2]
        sigma_sbih.variable_type = varType_uncertainty
        sigma_sbih.comment1 = 'This field is derived from the retrieved fields'
        sigma_sbih.comment2 = 'The uncertainties were determined using a monte carlo sampling of the posterior covariance matrix'
        sigma_sbih.comment3 = 'A value of -999 indicates that the uncertainty in this inded could not be computed (typically because the values were all unphysical)'
        
        sigma_sbim = fid.createVariable('sigma_sbim', 'f4', ('time',))
        sigma_sbim.long_name = '1-sigma uncertainties in the surface-based inversion magnitude'
        sigma_sbim.units = dindex['units'][3]
        sigma_sbim.variable_type = varType_uncertainty
        sigma_sbim.comment1 = 'This field is derived from the retrieved fields'
        sigma_sbim.comment2 = 'The uncertainties were determined using a monte carlo sampling of the posterior covariance matrix'
        sigma_sbim.comment3 = 'A value of -999 indicates that the uncertainty in this inded could not be computed (typically because the values were all unphysical)'
        
        sigma_sblcl = fid.createVariable('sigma_sbLCL', 'f4', ('time',))
        sigma_sblcl.long_name = '1-sigma uncertainties in the LCL for a surface-based parcel'
        sigma_sblcl.units = dindex['units'][4]
        sigma_sblcl.variable_type = varType_uncertainty
        sigma_sblcl.comment1 = 'This field is derived from the retrieved fields'
        sigma_sblcl.comment2 = 'The uncertainties were determined using a monte carlo sampling of the posterior covariance matrix'
        sigma_sblcl.comment3 = 'A value of -999 indicates that the uncertainty in this inded could not be computed (typically because the values were all unphysical)'
        
        sigma_sbcape = fid.createVariable('sigma_sbCAPE', 'f4', ('time',))
        sigma_sbcape.long_name = '1-sigma uncertainties in surface-based CAPE'
        sigma_sbcape.units = dindex['units'][5]
        sigma_sbcape.variable_type = varType_uncertainty
        sigma_sbcape.comment1 = 'This field is derived from the retrieved fields'
        sigma_sbcape.comment2 = 'The uncertainties were determined using a monte carlo sampling of the posterior covariance matrix'
        sigma_sbcape.comment3 = 'A value of -999 indicates that the uncertainty in this inded could not be computed (typically because the values were all unphysical)'
        
        sigma_sbcin = fid.createVariable('sigma_sbCIN', 'f4', ('time',))
        sigma_sbcin.long_name = '1-sigma uncertainties in surface-based CIN'
        sigma_sbcin.units = dindex['units'][6]
        sigma_sbcin.variable_type = varType_uncertainty
        sigma_sbcin.comment1 = 'This field is derived from the retrieved fields'
        sigma_sbcin.comment2 = 'The uncertainties were determined using a monte carlo sampling of the posterior covariance matrix'
        sigma_sbcin.comment3 = 'A value of -999 indicates that the uncertainty in this inded could not be computed (typically because the values were all unphysical)'
        
        sigma_mllcl = fid.createVariable('sigma_mlLCL', 'f4', ('time',))
        sigma_mllcl.long_name = '1-sigma uncertainties in the LCL for a mixed-layer parcel'
        sigma_mllcl.units = dindex['units'][7]
        sigma_mllcl.variable_type = varType_uncertainty
        sigma_mllcl.comment1 = 'This field is derived from the retrieved fields'
        sigma_mllcl.comment2 = 'The uncertainties were determined using a monte carlo sampling of the posterior covariance matrix'
        sigma_mllcl.comment3 = 'A value of -999 indicates that the uncertainty in this inded could not be computed (typically because the values were all unphysical)'
        
        sigma_mlcape = fid.createVariable('sigma_mlCAPE', 'f4', ('time',))
        sigma_mlcape.long_name = '1-sigma uncertainties in mixed-layer CAPE'
        sigma_mlcape.units = dindex['units'][8]
        sigma_mlcape.variable_type = varType_uncertainty
        sigma_mlcape.comment1 = 'This field is derived from the retrieved fields'
        sigma_mlcape.comment2 = 'The uncertainties were determined using a monte carlo sampling of the posterior covariance matrix'
        sigma_mlcape.comment3 = 'A value of -999 indicates that the uncertainty in this inded could not be computed (typically because the values were all unphysical)'
        
        sigma_mlcin = fid.createVariable('sigma_mlCIN', 'f4', ('time',))
        sigma_mlcin.long_name = '1-sigma uncertainties in mixed-layer CIN'
        sigma_mlcin.units = dindex['units'][9]
        sigma_mlcin.variable_type = varType_uncertainty
        sigma_mlcin.comment1 = 'This field is derived from the retrieved fields'
        sigma_mlcin.comment2 = 'The uncertainties were determined using a monte carlo sampling of the posterior covariance matrix'
        sigma_mlcin.comment3 = 'A value of -999 indicates that the uncertainty in this inded could not be computed (typically because the values were all unphysical)'
        
        obs_flag = fid.createVariable('obs_flag', 'i2', ('obs_dim',))
        obs_flag.long_name = 'Flag indicating type of observation for each vector element'
        obs_flag.variable_type = varType_observation
        obs_flag.comment1 = 'unitless'

        # This will make sure that I capture all of the units right in
        # the metadata, but "blotting them out" as I add the comments

        marker = np.copy(xret[0]['flagY'])
        foo = np.where(xret[0]['flagY'] == 1)[0]
        if len(foo) > 0:
            obs_flag.value_01 = 'IRS spectral radiance in wavenumber -- i.e., cm^(-1)'
            marker[foo] = -1

        foo = np.where(xret[0]['flagY'] == 2)[0]
        if len(foo) > 0:
            obs_flag.value_02 = 'MWR spectral brightness temperature [degK] from a zenith-pointing microwave radiometer'
            marker[foo] = -1

        foo = np.where(xret[0]['flagY'] == 3)[0]
        if len(foo) > 0:
            obs_flag.value_03 = 'Temperature in ' + ext_prof['tunit'] + ' from ' + ext_prof['ttype']
            marker[foo] = -1

        foo = np.where(xret[0]['flagY'] == 4)[0]
        if len(foo) > 0:
            obs_flag.value_04 = 'Water vapor in ' + ext_prof['qunit'] + ' from ' + ext_prof['qtype']
            marker[foo] = -1

        foo = np.where(xret[0]['flagY'] == 5)[0]
        if len(foo) > 0:
            obs_flag.value_05 = 'Surface met temperature in ' + ext_tseries['tunit'] + ' from ' + ext_tseries['ttype']
            obs_flag.value_05_comment1 = 'Surface met station is ' + str(ext_tseries['sfc_relative_height']) + ' m above height=0 level'
            if ext_tseries['sfc_temp_rep_error'] > 0:
                obs_flag.value_05_comment2 = 'Adding ' + str(ext_tseries['sfc_temp_rep_error']) + ' C to uncertainty to account for representativeness error'
            marker[foo] = -1

        foo = np.where(xret[0]['flagY'] == 6)[0]
        if len(foo) > 0:
            obs_flag.value_06 = 'Surface met water vapor in ' + ext_tseries['qunit'] + ' from ' + ext_tseries['qtype']
            obs_flag.value_06_comment1 = 'Surface met station is ' + str(ext_tseries['sfc_relative_height']) + ' m above height=0 level'
            if np.abs(ext_tseries['sfc_wv_mult_error']-1) > 0.01:
                obs_flag.value_06_comment2 = 'Multiplying by ' + str(ext_tseries['sfc_wv_mult_error'])
            if ext_tseries['sfc_wv_rep_error'] > 0:
                obs_flag.value_06_comment2 = 'Adding ' + str(ext_tseries['sfc_wv_rep_error']) + ' g/kg to uncertainty to account for representativeness error'
            marker[foo] = -1

        foo = np.where(xret[0]['flagY'] == 7)[0]
        if len(foo) > 0:
            obs_flag.value_07 = 'Temperature in ' + mod_prof['tunit'] + ' from ' + mod_prof['ttype']
            marker[foo] = -1

        foo = np.where(xret[0]['flagY'] == 8)[0]
        if len(foo) > 0:
            obs_flag.value_08 = 'Water vapor in ' + mod_prof['qunit'] + ' from '  + mod_prof['qtype']
            marker[foo] = -1

        foo = np.where(xret[0]['flagY'] == 9)[0]
        if len(foo) > 0:
            obs_flag.value_09 = 'CO2 in-situ obs in ' + ext_tseries['co2unit'] + ' from ' + ext_tseries['co2type']
            if ext_tseries['co2_sfc_rep_error'] > 0:
                obs_flag.value_09_comment1 = 'Adding ' + ext_tseries['co2_sfc_rep_error'] + ' ppm to uncertainty to account for representativeness error'
            marker[foo] = -1

        foo = np.where(xret[0]['flagY'] == 10)[0]
        if len(foo) > 0:
            obs_flag.value_10 = 'MWR spectral brightness temperature [degK] from an elevation scanning microwave radiometer'
            obs_flag.value_10_comment1 = 'Dimension is coded to be (frequency[GHz]*100)+(elevation_angle[deg]/1000)'
            marker[foo] = -1

        foo = np.where(xret[0]['flagY'] == 11)[0]
        if len(foo) > 0:
            obs_flag.value_11 = 'Retrieved temperature [degC] from a previous good TROPoe retrieval on this day'
            marker[foo] = -1

        foo = np.where(xret[0]['flagY'] == 12)[0]
        if len(foo) > 0:
            obs_flag.value_12 = 'Retrieved water vapor [g/kg] from a previous good TROPoe retrieval on this day'
            marker[foo] = -1

        # If there were any values for marker that were not treated above,
        # then the code must assume that I've screwed up and should abort.

        foo = np.where(marker >= 0)[0]
        if len(foo) > 0:
            print('Error in write_output: there seems to be a unit that is not handled here properly')
            return success, nfilename

        obs_dimension = fid.createVariable('obs_dimension', 'f8', ('obs_dim',))
        obs_dimension.long_name = 'Dimension of the observation vector'
        obs_dimension.comment1 = 'mixed units -- see obs_flag field above'
        obs_dimension.variable_type = varType_observation

        obs_vector = fid.createVariable('obs_vector', 'f4', ('time','obs_dim',))
        obs_vector.long_name = 'Observation vector Y'
        obs_vector.comment1 = 'mixed units -- see obs_flag field above'
        obs_vector.variable_type = varType_observation

        obs_vector_uncertainty = fid.createVariable('obs_vector_uncertainty', 'f4', ('time','obs_dim',))
        obs_vector_uncertainty.long_name = '1-sigma uncertainty in the observation vector (sigY)'
        obs_vector_uncertainty.comment1 = 'mixed units -- see obs_flag field above'
        obs_vector_uncertainty.variable_type = varType_uncertainty

        forward_calc = fid.createVariable('forward_calc', 'f4', ('time','obs_dim',))
        forward_calc.long_name = 'Forward calculation from state vector (i.e., F(Xn))'
        forward_calc.comment1 = 'mixed units -- see obs_flag field above'
        forward_calc.variable_type = varType_diagnostic

        arb1 = fid.createVariable('arb1', 'i2', ('arb_dim1',))
        arb1.long_name = 'Arbitrary dimension'
        arb1.comment1 = 'mixed units'
        arb1.comment2 = ('contains (1) temperature profile, (2) water vapor profile'
                       + ' (3) liquid water path, (4) liquid water Reff, '
                       + '(5) ice cloud optical depth, (6) ice cloud Reff, (7) carbon dioxide'
                       + ' (8) methane, (9) nitrous oxide')
        arb1.variable_type = varType_diagnostic

        arb2 = fid.createVariable('arb2', 'i2', ('arb_dim2',))
        arb2.long_name = 'Arbitrary dimension'
        arb2.comment1 = 'mixed units'
        arb2.comment2 = ('contains (1) temperature profile, (2) water vapor profile'
                       + ' (3) liquid water path , (4) liquid water Reff, '
                       + '(5) ice cloud optical depth, (6) ice cloud Reff, (7) carbon dioxide' 
                       + ' (8) methane, (9) nitrous oxide')
        arb2.variable_type = varType_diagnostic

        Xa = fid.createVariable('Xa', 'f8', ('arb_dim1',))
        Xa.long_name = 'Prior mean state'
        Xa.comment1 = 'mixed units -- see field arb above'
        Xa.variable_type = varType_diagnostic

        Sa = fid.createVariable('Sa', 'f8', ('arb_dim1','arb_dim2',))
        Sa.long_name = 'Prior covariance'
        Sa.comment1 = 'mixed units -- see field arb above'
        Sa.variable_type = varType_diagnostic
        
        # If we are trying to keep the output file small, then do not include
        # these fields in the output file
        if vip['output_akernal'] >= 1:
            
            Xop = fid.createVariable('Xop', 'f4', ('time','arb_dim1',))
            Xop.long_name = 'Optimal solution'
            Xop.comment1 = 'mixed units -- see field arb above'
            Xop.variable_type = varType_diagnostic

            Sop = fid.createVariable('Sop', 'f4', ('time','arb_dim1','arb_dim2',))
            Sop.long_name = 'Covariance matrix of the solution'
            Sop.comment1 = 'mixed units -- see field arb above'
            Sop.variable_type = varType_diagnostic

            Akernal = fid.createVariable('Akernal', 'f4', ('time','arb_dim1','arb_dim2',))
            Akernal.long_name = 'Averaging kernal'
            Akernal.comment1 = 'mixed units -- see field arb above'
            Akernal.variable_type = varType_diagnostic
        
        # Output the no model data averaging kernal
        if vip['output_akernal'] == 2:
            
            Akernal_nm = fid.createVariable('Akernal_no_model', 'f4', ('time', 'arb_dim1', 'arb_dim2',))
            Akernal_nm.long_name = 'Averaging kernal with no model data'
            Akernal_nm.comment1 = 'mixed units -- see field arb above'
            Akernal_nm.variable_type = varType_diagnostic
            

        # These should be the last three variables in the file
        lat = fid.createVariable('lat', 'f4')
        lat.long_name = 'Latitude'
        lat.units = 'degrees_north'

        lon = fid.createVariable('lon', 'f4')
        lon.long_name = 'Longitude'
        lon.units = 'degrees_east'

        alt = fid.createVariable('alt', 'f4')
        alt.long_name = 'station height above mean sea level'
        alt.units = 'm'

        # Add some global attributes
        for i in range(len(list(globatt.keys()))):
            fid.setncattr(list(globatt.keys())[i], globatt[list(globatt.keys())[i]])
        fid.Prior_dataset_comment = prior['comment']
        fid.Prior_dataset_filename = prior['filename']
        fid.Prior_dataset_number_profiles = prior['nsonde']
        fid.Prior_dataset_T_inflation_factor = str(vip['prior_t_ival']) + ' at the surface to 1.0 at ' + str(vip['prior_t_iht']) + ' km AGL'
        fid.Prior_dataset_Q_inflation_factor = str(vip['prior_q_ival']) + ' at the surface to 1.0 at ' + str(vip['prior_q_iht']) + ' km AGL'
        fid.Prior_dataset_TQ_correlation_reduction_factor = vip['prior_tq_cov_val']
        fid.Total_clock_execution_time_in_s = exectime
        fid.Retrieval_option_flags = '{:0d}, {:0d}, {:0d}, {:0d}, {:0d}, {:0d}, {:0d}'.format(modeflag[0], modeflag[1], modeflag[2], modeflag[3], modeflag[4], modeflag[5], modeflag[6])
        fid.Retrieval_option_flags_variables = 'doTemp, doWVMR, doLiqCloud, doIceCloud, doCO2, doCH4, doN2O'
        fid.vip_tres = (str(vip['tres']) + ' minutes. Note that the sample time corresponds to the '
                      + 'center of the averaging interval. A value of 0 implies that no averaging was performed')
        fid.Retrieval_start_hour = shour
        fid.Conventions = 'CF-1.10'
        add_vip_to_global_atts(fid, vip)

        # Add some of the static (non-time-dependent) data
        base_time[:] = xret[0]['secs']
        height[:] = xret[0]['z']
        obs_flag[:] = xret[0]['flagY']
        obs_dimension[:] = xret[0]['dimY']

        ones = np.ones(nht)
        twos = np.ones(nht)*2
        tmp = np.append(ones,twos)
        tmp = np.append(tmp, np.array([3,4,5,6,7,7,7,8,8,8,9,9,9]))

        arb1[:] = tmp
        arb2[:] = tmp
        Xa[:] = prior['Xa']
        Sa[:,:] = prior['Sa']

        # Fill these in, if available
        if type(location) is dict:
            if ((type(location['lat']) is int) or (type(location['lat'] is float))):
                try:
                    lat[:] = np.float(location['lat'])
                except TypeError:
                    # TODO - Allow changing lat/lon/alt. I.e. let the lat/lon/alt have a time dimension
                    lat[:] = np.float(np.mean(location['lat']))
            else:
                lat[:] = -999.0

            if ((type(location['lon']) is int) or (type(location['lon'] is float))):
                try:
                    lon[:] = np.float(location['lon'])
                except TypeError:
                    lon[:] = np.float(np.mean(location['lon']))
            else:
                lon[:] = -999.0

            if ((type(location['alt']) is int) or (type(location['alt'] is float))):
                try:
                    alt[:] = np.float(location['alt'])
                except TypeError:
                    alt[:] = np.float(np.mean(location['alt']))
            else:
                alt[:] = -999.0

        else:
            lat[:] = -999.
            lon[:] = -999.
            alt[:] = -999.

        fid.close()
        
    # Commenting this out but leaving it here in case we need to come back to it
    
    # foo = np.where(xret[0]['z'] < vip['prior_chimney_ht'])[0]
    # if len(foo) > 0:
    #     temp_tmp[foo,:] = nochim
    #     wvmr_tmp[foo,:] = nochim
    #     stemp_tmp[foo,:] = nochim
    #     swvmr_tmp[foo,:] = nochim
    #     theta_tmp[foo,:] = nochim
    #     thetae_tmp[foo,:] = nochim
    #     dewpt_tmp[foo,:] = nochim
    #     rh[foo,:] = nochim

    # Now append all of the samples from fsample onward into the file
    if verbose >= 3:
        print('Appending data to ' + nfilename)
    fid = Dataset(nfilename, 'a')
    fid.Total_clock_execution_time_in_s = str(exectime)

    time_offset = fid.variables['time_offset']
    time = fid.variables['time']
    hour = fid.variables['hour']
    qc_flag = fid.variables['qc_flag']

    temperature = fid.variables['temperature']
    waterVapor = fid.variables['waterVapor']
    lwp = fid.variables['lwp']
    lReff = fid.variables['lReff']
    iTau = fid.variables['iTau']
    iReff = fid.variables['iReff']
    co2 = fid.variables['co2']
    ch4 = fid.variables['ch4']
    n2o = fid.variables['n2o']

    sigmaT = fid.variables['sigma_temperature']
    sigmaWV = fid.variables['sigma_waterVapor']
    sigma_lwp = fid.variables['sigma_lwp']
    sigma_lReff = fid.variables['sigma_lReff']
    sigma_iTau = fid.variables['sigma_iTau']
    sigma_iReff = fid.variables['sigma_iReff']
    sigma_co2 = fid.variables['sigma_co2']
    sigma_ch4 = fid.variables['sigma_ch4']
    sigma_n2o = fid.variables['sigma_n2o']

    converged_flag = fid.variables['converged_flag']
    gamma = fid.variables['gamma']
    n_iter = fid.variables['n_iter']
    rmsr = fid.variables['rmsr']
    rmsa = fid.variables['rmsa']
    rmsp = fid.variables['rmsp']
    chi2 = fid.variables['chi2']
    convergence_criteria = fid.variables['convergence_criteria']
    dfs = fid.variables['dfs']
    dfs_nm = fid.variables['dfs_no_model']
    sic = fid.variables['sic']
    vres_temp = fid.variables['vres_temperature']
    vres_temp_nm = fid.variables['vres_temperature_no_model']
    vres_wv = fid.variables['vres_waterVapor']
    vres_wv_nm = fid.variables['vres_waterVapor_no_model']
    cdfs_temp = fid.variables['cdfs_temperature']
    cdfs_temp_nm = fid.variables['cdfs_temperature_no_model']
    cdfs_wv = fid.variables['cdfs_waterVapor']
    cdfs_wv_nm = fid.variables['cdfs_waterVapor_no_model']

    cbh = fid.variables['cbh']
    cbh_flag = fid.variables['cbh_flag']
    cbh_tcld = fid.variables['cbh_tcld']
    cbh_mlev = fid.variables['cbh_mlev']
    cld_emis = fid.variables['cld_emis']
    pressure = fid.variables['pressure']

    theta = fid.variables['theta']
    thetae = fid.variables['thetae']
    rh = fid.variables['rh']
    dewpt = fid.variables['dewpt']
    co2_profile = fid.variables['co2_profile']
    ch4_profile = fid.variables['ch4_profile']
    n2o_profile = fid.variables['n2o_profile']
    pwv = fid.variables['pwv']
    pblh = fid.variables['pblh']
    sbih = fid.variables['sbih']
    sbim = fid.variables['sbim']
    sblcl = fid.variables['sbLCL']
    sbcape = fid.variables['sbCAPE']
    sbcin = fid.variables['sbCIN']
    mllcl = fid.variables['mlLCL']
    mlcape = fid.variables['mlCAPE']
    mlcin = fid.variables['mlCIN']
    
    sigma_pwv = fid.variables['sigma_pwv']
    sigma_pblh = fid.variables['sigma_pblh']
    sigma_sbih = fid.variables['sigma_sbih']
    sigma_sbim = fid.variables['sigma_sbim']
    sigma_sblcl = fid.variables['sigma_sbLCL']
    sigma_sbcape = fid.variables['sigma_sbCAPE']
    sigma_sbcin = fid.variables['sigma_sbCIN']
    sigma_mllcl = fid.variables['sigma_mlLCL']
    sigma_mlcape = fid.variables['sigma_mlCAPE']
    sigma_mlcin = fid.variables['sigma_mlCIN']

    obs_vector = fid.variables['obs_vector']
    obs_vector_uncertainty = fid.variables['obs_vector_uncertainty']
    forward_calc = fid.variables['forward_calc']

    if vip['output_akernal'] >= 1:
        Xop = fid.variables['Xop']
        Sop = fid.variables['Sop']
        Akernal = fid.variables['Akernal']
    
    if vip['output_akernal'] == 2:
        Akernal_nm = fid.variables['Akernal_no_model']

    basetime = fid.variables['base_time'][:]
    
    time_offset[fsample] = xret[fsample]['secs'] - basetime
    time[fsample] = xret[fsample]['hour']*60*60     # compute the seconds since midnight
    hour[fsample] = xret[fsample]['hour']
    qc_flag[fsample] = xret[fsample]['qcflag']

    did = np.where(np.array(list(fid.dimensions.keys())) == 'height')[0]
    if len(did) == 0:
        print('Whoaa -- this should not happen -- aborting')
        return success, nfilename
    
    if fid.dimensions['height'].size != len(xret[0]['z']):
        print('Whoaa -- this should not happen size -- aborting')
        return success, nfilename

    temperature[fsample,:] = xret[fsample]['Xn'][0:nht]
    waterVapor[fsample,:] = xret[fsample]['Xn'][nht:2*nht]
    lwp[fsample] = xret[fsample]['Xn'][2*nht]
    lReff[fsample] = xret[fsample]['Xn'][2*nht+1]
    iTau[fsample] = xret[fsample]['Xn'][2*nht+2]
    iReff[fsample] = xret[fsample]['Xn'][2*nht+3]
    co2[fsample,:] = xret[fsample]['Xn'][2*nht+4:2*nht+7]
    ch4[fsample,:] = xret[fsample]['Xn'][2*nht+7:2*nht+10]
    n2o[fsample,:] = xret[fsample]['Xn'][2*nht+10:2*nht+13]

    sig = np.sqrt(np.diag(xret[fsample]['Sop']))
    sigmaT[fsample,:] = sig[:nht]
    sigmaWV[fsample,:] = sig[nht:2*nht]
    sigma_lwp[fsample] = sig[2*nht]
    sigma_lReff[fsample] = sig[2*nht+1]
    sigma_iTau[fsample] = sig[2*nht+2]
    sigma_iReff[fsample] = sig[2*nht+3]
    sigma_co2[fsample,:] = sig[2*nht+4:2*nht+7]
    sigma_ch4[fsample,:] = sig[2*nht+7:2*nht+10]
    sigma_n2o[fsample,:] = sig[2*nht+10:2*nht+13]

    converged_flag[fsample] = xret[fsample]['converged']
    gamma[fsample] = xret[fsample]['gamma']
    n_iter[fsample] = xret[fsample]['niter']
    rmsr[fsample] = xret[fsample]['rmsr']
    rmsa[fsample] = xret[fsample]['rmsa']
    rmsp[fsample] = xret[fsample]['rmsp']
    chi2[fsample] = xret[fsample]['chi2']
    convergence_criteria[fsample] = xret[fsample]['di2m']
    dfs[fsample,:] = xret[fsample]['dfs']
    dfs_nm[fsample,:] = xret[fsample]['dfs_nm']
    sic[fsample] = xret[fsample]['sic']
    vres_temp[fsample,:] = xret[fsample]['vres'][0,:]
    vres_temp_nm[fsample,:] = xret[fsample]['vres_nm'][0,:]
    vres_wv[fsample,:] = xret[fsample]['vres'][1,:]
    vres_wv_nm[fsample,:] = xret[fsample]['vres_nm'][1,:]
    cdfs_temp[fsample,:] = xret[fsample]['cdfs'][0,:]
    cdfs_temp_nm[fsample,:] = xret[fsample]['cdfs_nm'][0,:]
    cdfs_wv[fsample,:] = xret[fsample]['cdfs'][1,:]
    cdfs_wv_nm[fsample,:] = xret[fsample]['cdfs_nm'][1,:]
    
    cbh[fsample] = xret[fsample]['cbh']
    cbh_flag[fsample] = xret[fsample]['cbhflag']
    cbh_tcld[fsample] = xret[fsample]['cbh_tcld']
    cbh_mlev[fsample] = xret[fsample]['cbh_mlev']
    cld_emis[fsample] = xret[fsample]['cld_emis']
    pressure[fsample,:] = xret[fsample]['p'][0:nht]

    theta[fsample,:] = derived['theta'][:]
    thetae[fsample,:] = derived['thetae'][:]
    rh[fsample,:] = derived['rh'][:]
    dewpt[fsample,:] = derived['dewpt'][:]
    co2_profile[fsample,:] = derived['co2_profile'][:]
    ch4_profile[fsample,:] = derived['ch4_profile'][:]
    n2o_profile[fsample,:] = derived['n2o_profile'][:]
    
    pwv[fsample] = dindex['indices'][0]
    pblh[fsample] = dindex['indices'][1]
    sbih[fsample] = dindex['indices'][2]
    sbim[fsample] = dindex['indices'][3]
    sblcl[fsample] = dindex['indices'][4]
    sbcape[fsample] = dindex['indices'][5]
    sbcin[fsample] = dindex['indices'][6]
    mllcl[fsample] = dindex['indices'][7]
    mlcape[fsample] = dindex['indices'][8]
    mlcin[fsample] = dindex['indices'][9]
    
    sigma_pwv[fsample] = dindex['sigma_indices'][0]
    sigma_pblh[fsample] = dindex['sigma_indices'][1]
    sigma_sbih[fsample] = dindex['sigma_indices'][2]
    sigma_sbim[fsample] = dindex['sigma_indices'][3]
    sigma_sblcl[fsample] = dindex['sigma_indices'][4]
    sigma_sbcape[fsample] = dindex['sigma_indices'][5]
    sigma_sbcin[fsample] = dindex['sigma_indices'][6]
    sigma_mllcl[fsample] = dindex['sigma_indices'][7]
    sigma_mlcape[fsample] = dindex['sigma_indices'][8]
    sigma_mlcin[fsample] = dindex['sigma_indices'][9]

    obs_vector[fsample,:] = xret[fsample]['Y']
    obs_vector_uncertainty[fsample,:] = xret[fsample]['sigY']
    forward_calc[fsample,:] = xret[fsample]['FXn']

    if vip['output_akernal'] >= 1:
        Xop[fsample,:] = xret[fsample]['Xn']
        Sop[fsample,:,:] = xret[fsample]['Sop']
        Akernal[fsample,:,:] = xret[fsample]['Akern']
    
    if vip['output_akernal'] == 2:
        Akernal_nm[fsample,:,:] = xret[fsample]['Akern_nm']

    fid.close()
    success = 1

    return success, nfilename

################################################################################
# This function creates a "simulated" output structure needed for running the
# code in the "output_clobber eq 2" (append) mode. The number of fields, the
# dimensions and types of each of the fields, needs to be correct. Only the
# "secs" field needs to have the proper values though...
################################################################################

def create_xret(xret, fsample, vip, irs, Xa, Sa, z, bands, obsdim, obsflag, shour, in_tropoe):

    # Set a default value
    opres = -8888.  # Special flag, to help with the debugging if the pressure is negative

    # Find all of the output files with this date
    yy = np.array([datetime.utcfromtimestamp(x).year for x in irs['secs']])
    mm = np.array([datetime.utcfromtimestamp(x).month for x in irs['secs']])
    dd = np.array([datetime.utcfromtimestamp(x).day for x in irs['secs']])
    ymd = yy*10000 + mm*100 + dd

    files = []
    filename = vip['output_path'] + '/' + vip['output_rootname'] + '.' + str(ymd[0]) + '.*.nc'
    files = files + (glob.glob(filename))

    # If none are found, then just run code as normal. Note that xret and fsample
    if len(files) == 0:
        if vip['output_clobber'] == 2:
            print('The flag output_clobber was set to 2 for append, but no prior file was found')
            print('    so code will run as normal')
            nfilename = ' '
            return xret, fsample, nfilename, Xa, Sa, opres, in_tropoe
        else:
            nfilename = ' '
            return xret, fsample, nfilename, Xa, Sa, opres, in_tropoe
    
    # Check to see if a file with the same shour exists
    found = False
    for i in range(len(files)):
        fid = Dataset(files[i],'r')
        previous_shour = fid.Retrieval_start_hour
        fid.close()
        if np.isclose(shour,previous_shour):
            found = True
            break

    if found and vip['output_clobber'] == 0:
        print('A file with the same rootname and shour as the current run was found. Aborting retrieval')
        print('     to prevent clobbering.')
        print('     Existing file name: ' + files[i])
        nfilename = files[i]
        return xret, -1, nfilename, Xa, Sa, opres, in_tropoe
    
    # There was a file for this day, but there were different shours
    if not found and vip['output_clobber'] == 0:
        nfilename = ' '
        return xret, fsample, nfilename, Xa, Sa, opres, in_tropoe
    
    if not found and vip['output_clobber'] == 2:
        print('The flag output_clobber was set to 2 for append, but no prior file')
        print('      with the same shour was found so code will run as normal')
        print('      and create a new file.')
        nfilename = ' '
        return xret, fsample, nfilename, Xa, Sa, opres, in_tropoe
        
    if not found:
        print('Error: There seems to be some condition that was unanticipated -- aborting')
        sys.exit()

    # Otherwise, let's initialize from the last file
    nfilename = files[i]
    fid = Dataset(nfilename, 'r')
    bt = fid.variables['base_time'][:]
    to = fid.variables['time_offset'][:]
    xhour = fid.variables['hour'][:]
    xobsdim = fid.variables['obs_dimension'][:]
    xobsflag = fid.variables['obs_flag'][:]
    pressure = fid.variables['pressure'][:]
    xz = fid.variables['height'][:]
    xt = fid.variables['temperature'][:]
    xq = fid.variables['waterVapor'][:]
    xst = fid.variables['sigma_temperature'][:]
    xsq = fid.variables['sigma_waterVapor'][:]
    gfac = fid.variables['gamma'][:]
    pblh = fid.variables['pblh'][:]
    cbh = fid.variables['cbh'][:]
    lwp = fid.variables['lwp'][:]
    xXa = fid.variables['Xa'][:]
    xSa = fid.variables['Sa'][:]
    co2 = fid_variables['co2'][:]
    sco2 = fid_variables['sigma_co2'][:]
    fid.close()

    secs = bt+to

    # Load the in_tropoe structure with the last good profile (if one exists)
    # Be sure to keep the logic to identify samples that go into in_tropoe the same as in TROPoe.py
    if((vip['add_tropoe_T_input_flag'] == 1) | (vip['add_tropoe_q_input_flag'] == 1) | (vip['add_tropoe_co2_input_flag'] == 1)):
        idx = -1
        for i in range(len(secs)):
            if((gfac[i] < vip['add_tropoe_gamma_threshold']) & ((cbh[i] > vip['add_tropoe_input_cbh_thres']) | (lwp[i] < vip['add_tropoe_input_lwp_thres']))):
                idx = i
        if idx > 0:
            print(f'  Extracted retrieved profile at {xhour[idx]:.4f} UTC to the "in_tropoe" structure, to be used as input later')
            in_tropoe = {'secs':secs[idx], 'height':xz, 'pblh':pblh[idx], 'lwp':lwp[idx],
                           'temperature':np.copy(xt[idx,:]), 'waterVapor':np.copy(xq[idx,:]),
                           'sigma_temperature':np.copy(xst[idx,:]), 'sigma_waterVapor':np.copy(xsq[idx,:]), 
                           'co2':np.copy(co2[idx,:]), 'sigma_co2':np.copy(sco2[idx,:])}

    # Convert the masked arrays to ndarrays
    #   Note that setting the mask values to zero should not do anything, as there shouldn't
    # be any NaNs in the prior....
    fooXa = np.where(xXa.mask != 0)[0]
    fooSa = np.where(xSa.mask != 0)[0]
    if((len(fooXa) > 0) or (len(fooSa) > 0)):
        print('The mean prior or its covariance, which is coming from a preexisting output file, has NaNs. Aborting retrieval')
        nfilename = files[i]
        return xret, -1, nfilename, Xa, Sa, opres, in_tropoe
    oXa = xXa.filled(fill_value=np.double(-9999))
    oSa = xSa.filled(fill_value=np.double(-9999))

    # We want to extract out the surface pressure for the first sample also, as this
    # is needed to build the pressure profile if VIP.first_guess=1 in append mode
    opres = pressure[0,0]

    # Set up some default values
    w0idx, nY = Other_functions.find_wnum_idx(irs['wnum'],bands)
    wnum = irs['wnum'][w0idx]
    nY = len(xobsflag)
    Y = np.zeros(nY)
    Sop = np.diag(np.ones(len(Xa)))
    Kij = np.zeros((nY,len(Xa)))
    Gain = np.zeros((len(Xa),nY))
    Akern = np.copy(Sop)
    vres = np.zeros((2,len(z)))
    cdfs = np.zeros((2,len(z)))
    dfs = np.zeros(16)
    sic = 0.0

    # A few very basic checks to make sure that some of the variables in the
    # output file match the current ones (e.g., height, wavenumbers, etc)
    diff = np.abs(z-xz)
    foo = np.where(diff > 0.001)[0]
    if ((len(xz) != len(z)) | (len(foo) > 0)):
        print('Error: output_clobber is set to 2 (append), but there is a mismatch in heights')
        fsample = -1
        return xret, fsample, nfilename, Xa, Sa, opres, in_tropoe

    diff = np.abs(obsdim - xobsdim)
    foo = np.where(diff > 0.001)[0]
    if ((len(obsdim) != len(xobsdim)) | (len(foo) > 0)):
        print('Error: output_clobber is set to 2 (append), but there is a mismatch in obs_dim')
        fsample = -1
        return xret, fsample, nfilename, Xa, Sa, opres, in_tropoe

    diff = np.abs(obsflag - xobsflag)
    foo = np.where(diff > 0.001)[0]
    if ((len(obsflag) != len(xobsflag)) | (len(foo) > 0)):
        print('Error: output_clobber is set to 2 (append), but there is a mismatch in obs_flag')
        fsample = -1
        return xret, fsample, nfilename, Xa, Sa, opres, in_tropoe

    # This structure must match that at the end of the main iteration loop
    # where the retrieval is performed. The values can be anything, with
    # the exception of the "secs" field

    xtmp = {'idx':0, 'secs':0.0, 'ymd':ymd[0], 'hour':0.0,
             'nX':len(Xa), 'nY':len(obsflag),
             'dimY':np.zeros(len(obsflag)),
             'Y':np.zeros(len(obsflag)),
             'sigY':np.zeros(len(obsflag)),
             'flagY':np.zeros(len(obsflag)), 'nitr':0,
             'z':z, 'p':np.zeros(len(z)), 'hatchopen':0,
             'cbh':0., 'cbhflag':0,
             'x0':Xa*0, 'Xn':Xa*0, 'Fxn':Y*0, 'Sop':Sop, 'K':Kij, 'Gain':Gain, 'Akern':Akern,
             'vres':vres, 'cdfs':cdfs, 'gamma':0., 'qcflag':0, 'sic':sic, 'dfs':dfs, 'di2m':0.,
             'rmsa':0., 'rmsr':0., 'rmsp':0., 'chi2':0., 'converged':0}

    xret = []
    xret.append(xtmp)
    xret[0]['secs'] = secs[0]
    for i in range(1,len(secs)):
        xret.append(xtmp)
        xret[i]['secs'] = secs[i]
    fsample = len(secs)

    return xret, fsample, nfilename, oXa, oSa, opres, in_tropoe

################################################################################
# This function writes out the prior data into a temporary netCDF file (used for debugging)
################################################################################

def write_XaSa(filename, nht, X0, Xa, Sa):
    print(f'  DEBUG -- Creating the file {filename:s}')

    fid = Dataset(filename, 'w')

    adim1 = fid.createDimension('arb_dim1', len(X0))
    adim2 = fid.createDimension('arb_dim2', len(X0))

    arb1 = fid.createVariable('arb1', 'i2', ('arb_dim1',))
    arb1.long_name = 'Arbitrary dimension'
    arb1.comment1 = 'mixed units'
    arb1.comment2 = ('contains (1) temperature profile, (2) water vapor profile'
                       + ' (3) liquid water path, (4) liquid water Reff, '
                       + '(5) ice cloud optical depth, (6) ice cloud Reff, (7) carbon dioxide'
                       + ' (8) methane, (9) nitrous oxide')

    arb2 = fid.createVariable('arb2', 'i2', ('arb_dim2',))
    arb2.long_name = 'Arbitrary dimension'
    arb2.comment1 = 'mixed units'
    arb2.comment2 = ('contains (1) temperature profile, (2) water vapor profile,'
                       + ' (3) liquid water path, (4) liquid water Reff, '
                       + '(5) ice cloud optical depth, (6) ice cloud Reff, (7) carbon dioxide,'
                       + ' (8) methane, (9) nitrous oxide')

    xXa = fid.createVariable('Xa', 'f8', ('arb_dim1',))
    xXa.long_name = 'Prior mean state'
    xXa.comment1 = 'mixed units -- see field arb above'

    xSa = fid.createVariable('Sa', 'f8', ('arb_dim1','arb_dim2',))
    xSa.long_name = 'Prior covariance'
    xSa.comment1 = 'mixed units -- see field arb above'
        
    fid.dataset_comment = 'File containing the prior information'

    ones = np.ones(nht)
    twos = np.ones(nht)*2
    tmp  = np.append(ones,twos)
    tmp  = np.append(tmp, np.array([3,4,5,6,7,7,7,8,8,8,9,9,9]))

    arb1[:]  = tmp
    arb2[:]  = tmp
    xXa[:]   = Xa
    xSa[:,:] = Sa

    fid.close()
    return

################################################################################
# This function reads in the prior data into a temporary netCDF file (used for debugging)
################################################################################

def read_XaSa(filename):
    print(f'  DEBUG -- Reading the file {filename:s}')

    fid = Dataset(filename, 'r')
    xXa = fid.variables['Xa'][:]
    xSa = fid.variables['Sa'][:]
    fid.close()

    xXa.mask = xXa.mask * 0
    xSa.mask = xSa.mask * 0
    oXa = xXa.filled(fill_value=np.double(-9999)).astype('float64')
    oSa = xSa.filled(fill_value=np.double(-9999)).astype('float64')
    oXa = xXa.filled(fill_value=np.double(-9999))
    oSa = xSa.filled(fill_value=np.double(-9999))

    return oXa, oSa
