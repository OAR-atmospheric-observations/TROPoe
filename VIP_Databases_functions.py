# ----------------------------------------------------------------------------
#
#  Copyright (C) 2015,2022, 2023 by David D Turner, Joshua Gebauer, and Tyler Bell 
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
import shutil
import numpy as np
import scipy.io

################################################################################
# This file contains the following functions:
# read_tropoe_vip_file()
# check_vip_file()
# abort()
# read_scat_databases()
# read_stdatmos()
# read_mixcra_vip_file()
################################################################################


# This is the stucture with all of the input that we need.
# The code below will be searching the VIP input file for the same keys
# that are listed in this structure. If a line is found with the same
# name , then the code will determine the type for the value
# (e.g., is the value for that key a float, string, integer, etc) and then
# will cast the value provided in the VIP file as that.
# There are three exceptions to this
#   1) "Success" has the status of this routine -- it is not set by VIP file
#   2) "irs_calib_pres" is a 2-elements floating point array
#   3) "vip_filename" captures the name of the input VIP file itself
# The code will ouput how many of the keys in this structure were found.
# Note that not all of them have to be in the VIP file; if a key in this
# structure is not found in the VIP file then it maintains it default value

maxbands = 200  # The maximum number of bands to enable for the retrieval

full_tropoe_vip = ({
    'success': {'value': 0, 'comment': 'Interal success flag. Not for outside use', 'default': False},
    'tres': {'value': 0, 'comment': 'Temporal resolution [min], 0 implies maximum (native) temporal resolution', 'default': True},
    'avg_instant': {'value': 1, 'comment': 'A flag to specify -1:avg with no sqrt(N), 0:avg with sqrt(N), or 1:instantaneous', 'default': True},
    'tag': {'value': 'tropoe', 'comment': 'String for temporary files / directories\n', 'default': False},

    'zgrid': {'value': '0.000, 0.010, 0.021, 0.033, 0.046, 0.061, 0.077, 0.095, 0.114, 0.136, 0.159, 0.185, 0.214, 0.245, 0.280, 0.318, 0.359, 0.405, 0.456, 0.512, 0.573, 0.640, 0.714, 0.795, 0.885, 0.983, 1.092, 1.211, 1.342, 1.486, 1.645, 1.819, 2.011, 2.223, 2.455, 2.710, 2.991, 3.300, 3.640, 4.014, 4.426, 4.879, 5.376, 5.924, 6.526, 7.189, 7.918, 8.720, 9.602,10.572,11.639,12.813,14.104,15.525,17.087','comment': 'Comma delimited list of heights for the z-grid to use in the retrieval [km]\n', 'default': True},

    'station_lat': {'value': -999., 'comment': 'Station latitude [degN]', 'default': True},
    'station_lon': {'value': -999., 'comment': 'Station longitude [degE]', 'default': True},
    'station_alt': {'value': -999., 'comment': 'Station altitude [m MSL]', 'default': True},
    'station_pres': {'value': 1000., 'comment': 'Station pressure [mb]; will be only be used if there is no other surface pressure input', 'default': True},
    'station_psfc_min': {'value': 800., 'comment': 'Default minimum surface pressure [mb], used for QC', 'default': True},
    'station_psfc_max': {'value': 1040., 'comment': 'Default maximum surface pressure [mb], used for QC\n', 'default': True},

    'omb_flag': {'value': 0, 'comment': 'Flag to perform single forward calculation using input file; useful for o-minus-b (O-B) calculations (0 - off, 1 - on)', 'default': True},
    'omb_path': {'value': "/data", 'comment': 'Path to the input NetCDF file to use for the O-B calculation', 'default': True},
    'omb_file': {'value': "omb_input.nc", 'comment': 'NetCDF file with profiles of temperature [C], rh [%], pressure [mb], and height [km AGL] to use in the forward O-B calculation', 'default': True},
    'omb_required_minht': {'value': 10.0, 'comment': 'Minimum height [km AGL] required for the input profiles used in the forward O-B calculation\n', 'default': True},

    'irs_type': {'value': 0, 'comment': '0- output, options, and stop, 1 - ARM AERI data, 2 - dmv2cdf AERI data (C1_rnc.cdf and _sum.cdf), 3 - dmv2ncdf (C1.RNC.cdf and .SUM.cdf), 5- ASSIST, -1 - MWR data is to be used as MASTER dataset (no IRS data being read in)', 'default': True},
    'irsch1_path': {'value': "/data/aerich1", 'comment': 'Path to the IRS ch1 radiance files', 'default': True},
    'irsch2_path': {'value': "/data/aerich2", 'comment': 'Path to the IRS ch2 radiance files', 'default': True},
    'irssum_path': {'value': "/data/aerisum", 'comment': 'Path to the IRS summary files', 'default': True},
    'irseng_path': {'value': "/data/aerieng", 'comment': 'Path to the IRS engineering files', 'default': True},
    'irs_zenith_scene_mirror_angle': {'value': 180, 'comment': 'SceneMirrorAngle [deg] to use for zenith views (default is 180)', 'default': True},
    'irs_min_noise_flag':{'value': 1, 'comment': 'If non-zero, then the irs_min_noise_spectrum will be used as a floor; otherwise, will use input IRS noise spectrum from instrument', 'default': True},
    'irs_min_noise_wnum':{'value': '500,522,546,575,600,631,747,1439,1770,1884,2217,3000', 'comment': 'Wavenumber array [cm-1] for the minimum noise spectrum', 'default': True},
    'irs_min_noise_spec':{'value': '65.309,14.056,3.283,1.333,0.813,0.557,0.304,0.581,0.822,0.025,0.023,0.044', 'comment': 'Noise array [RU] for the minimum noise spectrum', 'default': True},
    'irs_noise_inflation':{'value': 1.0, 'comment': 'Value to increase the assumed random error in the IRS data', 'default': False},
    'irs_smooth_noise': {'value': 0, 'comment': 'The temporal window [minutes] used to smooth the IRS noise with time', 'default': False},
    'irs_band_noise_inflation': {'value': 1, 'comment': '0 -- off (no noise inflation), 1 -- on (will inflate the noise in the spectral band below; requires surface WVMR input from met station)', 'default': False}, 
    'irs_band_noise_wnums': {'value': '775.0,810.0', 'comment': 'A comma separated list of two wavenumbers; the noise spectrum between these will be inflated', 'default': False}, 
    'irs_band_noise_sfc_npts': {'value': 4, 'comment': 'The number of points in the x- (surface water vapor) and y- (noise inflation) axes', 'default': False}, 
    'irs_band_noise_sfc_wvmr_vals': {'value': '0.,7,12,18', 'comment': 'A comma separated list of the surface water vapor mixing ratio values [g/kg] for the x-axis', 'default': False}, 
    'irs_band_noise_sfc_multiplier': {'value': '20.0,20,1,1', 'comment': 'A comma separated list of multipliers [unitless] for the y-axis', 'default': False}, 
    'irs_calib_pres': {'value': [0.0, 1.0], 'comment': 'Intercept [mb] and slope [mb/mb to calib (newP = int = slope*obsP) (need comma between them)', 'default': False},
    'irs_use_missingDataFlag': {'value': 1, 'comment': 'Set this to 1 to use the field \'missingDataFlag\' (from the ch1 file) to remove bad IRS data from analysis. If zero, then all IRS data will be processed,', 'default': False},
    'irs_hatch_switch': {'value': 1, 'comment': '1 - only include hatchOpen=1 when averaging, 0 - include all IRS samples when averaging', 'default': False},
    'irs_ignore_status_hatch': {'value': 0, 'comment': '0 -- use the hatchOpen flag; 1 -- ignore the hatchOpen flag', 'default': False}, 
    'irs_ignore_status_missingDataFlag': {'value': 0, 'comment': '0 -- use the missingDataFlag flag; 1 -- ignore the missingDataFlag flag', 'default': False}, 
    'irs_fv': {'value': 0.0, 'comment': 'Apply a foreoptics obscuration correction', 'default': False},
    'irs_fa': {'value': 0.0, 'comment': 'Apply an aftoptics obscuration correction', 'default': False},
    'irs_old_ffov_halfangle': {'value': 23.0, 'comment': 'Original Half angle [millirad] of the finite field of view of the instrument', 'default': False},
    'irs_new_ffov_halfangle': {'value': 0.0, 'comment': 'New half angle [millirad] of the finite field of view of the instrument (values} <= 0 will result in no correction being applied)', 'default': False},
    'irs_min_675_bt': {'value': 233., 'comment': 'Minimum brightness temp [K] in the 675-680 cm-1 window -- this is QC screen', 'default': True},
    'irs_max_675_bt': {'value': 313., 'comment': 'Maximum brightness temp [K] in the 675-680 cm-1 window -- this is QC screen\n', 'default': True},
    'irs_spec_cal_factor_ch1': {'value': 1.0, 'comment': 'The multiplicative stretch factor to change the spectral calibration of IRS ch1 data', 'default': False},
    'irs_spec_cal_factor_ch2': {'value': 1.0, 'comment': 'The multiplicative stretch factor to change the spectral calibration of IRS ch2 data', 'default': False},
    'irs_mlev_cbh_wnum_range': {'value': [790.,810], 'comment':'The wavenumber range [cm-1] that should be used for the MLEV CBH calculation', 'default':False},
    'irs_tcld_wnum_range': {'value': [898.,904], 'comment':'The wavenumber range [cm-1] that should be used for the Tcld CBH calculation', 'default':False},
    'irs_1m_od_thres': {'value': 1.0, 'comment': 'IRS channels where the 1m optical depth is above this threshold are not used', 'default': False},

    'mwr_type': {'value': 0, 'comment': '0 - none, 1 - Tb fields are individual time series, 2 - Tb field is 2-d array', 'default': True},
    'mwr_path': {'value': '/data/mwr', 'comment': 'Path to the MWR data', 'default': True},
    'mwr_rootname': {'value': 'mwr', 'comment': 'Rootname of the MWR data files', 'default': True},
    'mwr_elev_field': {'value': 'elev', 'comment': 'Name of the scene mirror elevation field; use \'none\' if all data are zenith', 'default': True},
    'mwr_freq_field': {'value': 'freq', 'comment': 'Name of the frequency field needed if mwr_type >= 2', 'default': True},
    'mwr_n_tb_fields': {'value': 0, 'comment': 'Number of fields to read in', 'default': True},
    'mwr_tb_field_names': {'value': 'tbsky23,tbsky31', 'comment': 'Comma separated list of field names for the Tb fields', 'default': True},
    'mwr_tb_field1_tbmax': {'value': 100., 'comment': 'Maximum value [K] in the first Tb field, used for QC', 'default': True},
    'mwr_tb_freqs': {'value': '23.8,31.4', 'comment': 'Comma separated list of frequency [GHz] of MWR Tb fields', 'default': True},
    'mwr_tb_noise': {'value': '0.3,0.3', 'comment': 'Comma separated list of noise levels [K] in the MWR Tb fields', 'default': True},
    'mwr_tb_bias': {'value': '0.0,0.0', 'comment': 'Comma separated list of bias [K] in the MWR Tb fields; this value is ADDED to the MWR observations', 'default': True},
    'mwr_tb_replicate': {'value': 1, 'comment': 'Number of times to replicate the obs -- this allows us to change the weight of the MWR data in the retrieval (useful for LWP)', 'default': True},
    'mwr_time_delta': {'value': 0.083, 'comment': 'The maximum amount of time [hours] that the MWR zenith obs must be to the sampling time to be used \n', 'default': True},

    'mwrscan_type': {'value': 0, 'comment': '0 - none, 1 - Tb fields are individual time series, 2 - Tb field is 2-d array', 'default': True},
    'mwrscan_path': {'value': 'None', 'comment': 'Path to the MWRscan data', 'default': True},
    'mwrscan_rootname': {'value': 'mwr', 'comment': 'Rootname of the MWRscan data files', 'default': True},
    'mwrscan_freq_field': {'value': 'freq', 'comment': 'Name of the frequency field needed if mwrscan_type >= 2', 'default': True},
    'mwrscan_elev_field': {'value': 'elev', 'comment': 'Name of the scene mirror elevation field; this field must exist', 'default': True},
    'mwrscan_n_tb_fields': {'value': 0, 'comment': 'Number of fields to read in', 'default': True},
    'mwrscan_n_elevations': {'value': 2, 'comment': 'The number of elevations to use in retrieval (put zenith obs in \'mwr_type\')', 'default': True},
    'mwrscan_elevations': {'value': '20,160', 'comment': 'The elevation angles to use in deg, where 90 is zenith.  The code will look for these obs within the averaging interval', 'default': True},
    'mwrscan_tb_field1_tbmax': {'value': 330., 'comment': 'Maximum value [K] in the first Tb field, used for QC', 'default': True},
    'mwrscan_tb_field_names': {'value': 'tbsky23,tbsky31', 'comment': 'Comma separated list of field names for the Tb fields', 'default': True},
    'mwrscan_tb_freqs': {'value': '23.8,31.4', 'comment': 'Comma separated list of frequency [GHz] of MWR', 'default': True},
    'mwrscan_tb_noise': {'value': '0.3,0.3', 'comment': 'Comma separated list of noise levels [K] in the MWR Tb fields', 'default': True},
    'mwrscan_tb_bias': {'value': '0.0,0.0', 'comment': 'Comma separated list of bias [K] in the MWR Tb fields; this value is ADDED to the MWR observations', 'default': True},
    'mwrscan_time_delta': {'value': 0.25, 'comment': 'The maximum amount of time [hours] that the elevation scan must be to the sampling time to be used.\n', 'default': True},

    'ext_sfc_temp_type': {'value': 0, 'comment': 'External surface temperature met data type: 0-none, 1-external met data', 'default': True},
    'ext_sfc_temp_fieldname': {'value': 'temp_mean', 'comment': 'Name of surface temperature met field', 'default': True},
    'ext_sfc_temp_units': {'value': 1, 'comment': 'Units of surface temperature met field: 1-degC, 2-degK', 'default': True},
    'ext_sfc_temp_npts': {'value': 1, 'comment': 'Number of surface temperature met points to use in the retrieval.  Minimum=1, maximum=1000.  Larger number increases the weight of the observation', 'default': True},
    'ext_sfc_temp_random_error': {'value': 0.5, 'comment': 'Random error for the surface temperature measurement [degC]', 'default': True},
    'ext_sfc_temp_rep_error': {'value': 0.0, 'comment': 'Representativeness error for the surface temperature measurement [degC], which is added to the random error', 'default': True},

    'ext_sfc_wv_type': {'value': 0, 'comment': 'External surface water vapor met data type: 0-none, 1-external met data', 'default': True},
    'ext_sfc_wv_fieldname': {'value': 'rh_mean', 'comment': 'Name of surface water vapor met field', 'default': True},
    'ext_sfc_wv_units': {'value': 1, 'comment': 'Units of surface water vapor met field: 1-percent_rh, 2-fraction_rh', 'default': True},
    'ext_sfc_wv_npts': {'value': 1, 'comment': 'Number of surface water vapor met points to use in the retrieval.  Minimum=1, maximum=1000.  Larger number increases the weight of the observation', 'default': True},
    'ext_sfc_rh_random_error': {'value': 3.0, 'comment': 'Random error for the surface relative humidity measurement [%], which is used with the ext_sfc_random_temp_error to get the uncertainty in WVMR', 'default': True},
    'ext_sfc_wv_mult_error': {'value': 1.0, 'comment': 'Multiplier for the error in the surface water vapor mixing ratio measurement.  This is applied BEFORE the \'rep_error\' value', 'default': True},
    'ext_sfc_wv_rep_error': {'value': 0.0, 'comment': 'Representativeness error for the surface water vapor measurement [g/kg], which is added to the assumed random uncertainty of 0.5 degC and 3%RH', 'default': True},

    'ext_sfc_pres_type': {'value': 0, 'comment': '0 - Use the internal {IRS,MWR} pressure sensor for psfc; 1-external surface met data', 'default': True},
    'ext_sfc_pres_fieldname': {'value': 'atmos_pressure', 'comment': 'Name of surface pressure met field', 'default': True},
    'ext_sfc_pres_units': {'value': 1, 'comment': 'Units of surface pressure field: 1-kPa, 2-hPa, 3-Pa', 'default': True},

    'ext_sfc_path': {'value': '/data/met', 'comment': 'Path to the external surface met data', 'default': True},
    'ext_sfc_rootname': {'value': 'met', 'comment': 'Rootname of the external surface met datastream', 'default': True},
    'ext_sfc_time_format': {'value': 0, 'comment': 'Time format for external surface met datastream: 0-base_time/{time_offset,time} [epoch seconds], 1-same as option 0 but base_time is in milliseconds (ASSIST), 2-time [epoch seconds]', 'default': True},
    'ext_sfc_time_delta': {'value': 0.2, 'comment': 'Maximum amount of time from endpoints of external surface met dataset to extrapolate [hours]', 'default': True},
    'ext_sfc_relative_height': {'value': 0, 'comment': 'Relative height of the met station to the IRS zenith port [m]; note if met station is below IRS port then the value should be negative\n', 'default': True},

    'ext_wv_prof_type': {'value': 0, 'comment': 'External WV profile source: 0-none; 1-sonde; 2-ARM Raman lidar (rlprofmr), 3-NCAR WV DIAL, 4-Model sounding', 'default': False},
    'ext_wv_prof_path': {'value': 'None', 'comment': 'Path to the external profile of WV data', 'default': False},
    'ext_wv_prof_minht': {'value': 0.0, 'comment': 'Minimum height to use the data from the external WV profiler [km AGL]', 'default': False},
    'ext_wv_prof_maxht': {'value': 10.0, 'comment': 'Maximum height to use the data from the external WV profiler [km AGL]', 'default': False},
    'ext_wv_prof_min_error': {'value': 0.01, 'comment': 'Minimum random error allowed in the profile at any height [g/kg]', 'default': False},
    'ext_wv_noise_mult_val': {'value': [1.0, 1, 1.0], 'comment': '3-element comma delimited list with the multipliers to apply the noise profile of the external water vapor profile (must be > 1)', 'default': False},
    'ext_wv_noise_mult_hts': {'value': [0.0, 3, 20], 'comment': '3-element comma delimited list with the corresponding heights for the noise multipliers [km AGL]', 'default': False},
    'ext_wv_add_rel_error': {'value': 0.0, 'comment': 'When using the RLID, I may want to include a relative error contribution to the uncertainty to account for calibration. This is a correlated error component, and thus effects the off-diagonal elements of the observation covariance matrix. Units are [%]', 'default': False},
    'ext_wv_ht_offset': {'value': 0.0, 'comment': 'Height offset relative to the master instrument [km]; a negative value indicates that this profile starts below the master instrument', 'default': False},
    'ext_wv_time_delta': {'value': 1.0, 'comment': 'Maximum amount of time from endpoints of external WV dataset to extrapolate [hours]\n', 'default': False},

    'ext_temp_prof_type': {'value': 0, 'comment': 'External temperature profile source: 0-none; 1-sonde; 2-ARM Raman lidar (rlproftemp); 4-Model sounding, 5-RASS', 'default': False},
    'ext_temp_prof_path': {'value': 'None', 'comment': 'Path to external profile of temp data', 'default': False},
    'ext_temp_prof_minht': {'value': 0.0, 'comment': 'Minimum height to use the data from the external temp profiler [km AGL]', 'default': False},
    'ext_temp_prof_maxht': {'value': 10.0, 'comment': 'Maximum height to use the data from the external temp profiler [km AGL]', 'default': False},
    'ext_temp_noise_adder_val': {'value': [0.0, 0, 0], 'comment': '3-element comma delimited list of additive values to apply the noise profile of the external temperature profile (must be >= 0)', 'default': False},
    'ext_temp_noise_adder_hts': {'value': [0.0, 3, 20], 'comment': '3-element comma delimited list with the corresponding heights for the additive value [km AGL]', 'default': False},
    'ext_temp_ht_offset': {'value': 0.0, 'comment': 'Height offset relative to the master instrument [km]; a negative value indicates that this profile starts below the master instrument', 'default': False},
    'ext_temp_time_delta': {'value': 1.0, 'comment': 'Maximum amount of time from endpoints of external temp dataset to extrapolate [hours]\n', 'default': False},

    'mod_wv_prof_type': {'value': 0, 'comment': 'NWP model WV profile source: 0-none; 4-Model sounding', 'default': False},
    'mod_wv_prof_path': {'value': 'None', 'comment': 'Path to the model profile of WV data', 'default': False},
    'mod_wv_prof_minht': {'value': 0.0, 'comment': 'Minimum height to use the data from the model WV profiler [km AGL]', 'default': False},
    'mod_wv_prof_maxht': {'value': 10.0, 'comment': 'Maximum height to use the data from the model WV profiler [km AGL]', 'default': False},
    'mod_wv_prof_min_error': {'value': 0.01, 'comment': 'Minimum random error allowed in the profile at any height [g/kg]', 'default': False},
    'mod_wv_noise_mult_val': {'value': [1.0, 1, 1.0], 'comment': '3-element comma delimited list with the multipliers to apply the noise profile of the model water vapor profile (must be > 1)', 'default': False},
    'mod_wv_noise_mult_hts': {'value': [0.0, 3, 20], 'comment': '3-element comma delimited list with the corresponding heights for the noise multipliers [km AGL]', 'default': False},
    'mod_wv_ht_offset': {'value': 0.0, 'comment': 'Height offset relative to the master instrument [km]; a negative value indicates that this profile starts below the master instrument', 'default': False},
    'mod_wv_time_delta': {'value': 1.0, 'comment': 'Maximum amount of time from endpoints of model WV dataset to extrapolate [hours]\n', 'default': False},

    'mod_temp_prof_type': {'value': 0, 'comment': 'NWP model temperature profile source: 0-none; 4-Model sounding', 'default': False},
    'mod_temp_prof_path': {'value': 'None', 'comment': 'Path to the model profile of temp data', 'default': False},
    'mod_temp_prof_minht': {'value': 0.0, 'comment': 'Minimum height to use the data from the model temp profiler [km AGL]', 'default': False},
    'mod_temp_prof_maxht': {'value': 10.0, 'comment': 'Maximum height to use the data from the model temp profiler [km AGL]', 'default': False},
    'mod_temp_noise_adder_val': {'value': [0.0, 0, 0], 'comment': '3-element comma delimited list of additive values to apply the noise profile of the model temperature profile (must be >= 0)', 'default': False},
    'mod_temp_noise_adder_hts': {'value': [0.0, 3, 20], 'comment': '3-element comma delimited list with the corresponding heights for the additive value [km AGL]', 'default': False},
    'mod_temp_ht_offset': {'value': 0.0, 'comment': 'Height offset relative to the master instrument [km]; a negative value indicates that this profile starts below the master instrument', 'default': False},
    'mod_temp_time_delta': {'value': 1.0, 'comment': 'Maximum amount of time from endpoints of model WV dataset to extrapolate [hours]\n', 'default': False},

    'add_tropoe_T_input_flag':      {'value': 0, 'comment': 'Flag to add previous good TROPoe temperature retrieval as input to current retrieval (1-yes, 0-no)', 'default': True},
    'add_tropoe_q_input_flag':      {'value': 0, 'comment': 'Flag to add previous good TROPoe water vapor retrieval as input to current retrieval (1-yes, 0-no)', 'default': True},
    'add_tropoe_input_lwp_thres':   {'value': 10., 'comment': 'Only will consider using previous TROPoe profile if LWP is below this threshold [g/m2] or CBH is above threshold below', 'default': True},
    'add_tropoe_input_cbh_thres':   {'value': 5.0, 'comment': 'Only will consider using previous TROPoe profile if CBH is above this threshold [km AGL] or LWP is below threshold above', 'default': True},
    'add_tropoe_gamma_threshold':   {'value': 2, 'comment': 'Only samples with gamma values less than this threshold will be considered for use in future retrieval', 'default': False},
    'add_tropoe_T_noise_adder_val': {'value': [3.0, 1, 1], 'comment': '3-element comma delimited list of additive values to apply the noise profile of the previous TROPoe input temperature profile (must be >= 0)', 'default': False},
    'add_tropoe_T_noise_adder_hts': {'value': [0.0, 1, 20], 'comment': '3-element comma delimited list with the corresponding heights for the additive value [km AGL]', 'default': False},
    'add_tropoe_q_noise_mult_val':  {'value': [5.0, 2, 2], 'comment': '3-element comma delimited list with the multipliers to apply the noise profile of the previous TROPoe input water vapor profile (must be > 1)', 'default': False},
    'add_tropoe_q_noise_mult_hts':  {'value': [0.0, 1, 20], 'comment': '3-element comma delimited list with the corresponding heights for the noise multipliers [km AGL]', 'default': False},
    'add_tropoe_use_prior_as_max':  {'value': 1, 'comment': 'If set, then the maximum amount of uncertainty is the uncertainty of the prior', 'default': False},
    'add_tropoe_use_pblh_flag':     {'value': 1, 'comment': 'If set, then the middle height point in the adder and multiplier profiles becomes maximum of that value or the PBLH [km AGL]\n', 'default': True},

    'co2_sfc_type': {'value': 0, 'comment': 'External CO2 surface data type: 0-none, 1-DDT QC PGS data', 'default': False},
    'co2_sfc_npts': {'value': 1, 'comment': 'Number of surface CO2 in-situ points to use in the retrieval.  Minimum=1, maximum=1000.  Larger number increases the weight of the observation', 'default': False},
    'co2_sfc_rep_error': {'value': 0.0, 'comment': 'Representativeness error for the CO2 surface measurement [ppm], which is added to the uncertainty of the obs in the input file', 'default': False},
    'co2_sfc_path': {'value': 'None', 'comment': 'Path to the external surface CO2 data', 'default': False},
    'co2_sfc_relative_height': {'value': 0, 'comment': 'Relative height of the CO2 surface measurement to the IRS zenith port [m]; note if in-situ obs is below IRS port then the value should be negative', 'default': False},
    'co2_sfc_time_delta': {'value': 1.5, 'comment': 'Maximum amount of time from endpoints of external CO2 in-situ dataset to extrapolate [hours] \n', 'default': False},

    'cbh_type': {'value': 0, 'comment': '0 - output options and stop, 1 - VCEIL, 2 - Gregs ASOS CBH file, 3 - CLAMPS DLfp data, 4 - ARM dlprofwstats data', 'default': True},
    'cbh_path': {'value': '/data/ceil', 'comment': 'Path to the CBH data', 'default': True},
    'cbh_window_in': {'value': 20, 'comment': 'Inner temporal window (full-size) centered upon IRS time to look for cloud', 'default': True},
    'cbh_window_out': {'value': 180, 'comment': 'Outer temporal window (full-size) centered upon IRS time to look for cloud}', 'default': True},
    'cbh_default_ht': {'value': 2.0, 'comment': 'Default CBH height [km AGL], if no CBH data found', 'default': True},
    'cbh_mpd_abc_thres': {'value':  10 ** -5.5 , 'comment': 'aerosol backscatter threshold for cloud\n', 'default': True},

    'output_rootname': {'value': 'None', 'comment': 'String with the rootname of the output file', 'default': True},
    'output_path': {'value': '/data/tropoe', 'comment': 'Path where the output file will be placed', 'default': True},
    'output_akernal': {'value': 1, 'comment': '0 - do not output Sop and Akernal; 1 - output normal Sop and normal Akernal; 2 - output normal Sop, normal Akernal, and "no-model" Akernal', 'default': False},
    'output_clobber': {'value': 0, 'comment': '0 - do not clobber preexisting output files, 1 - clobber them, 2 - append to the last file of this day\n', 'default': True},

    'lbl_home': {'value': '/home/tropoe/vip/src/lblrtm_v12.17/lblrtm', 'comment': 'String with the LBL_HOME path', 'default': False},
    'lbl_temp_dir': {'value': '/tmp', 'comment': 'Temporary working directory for the retrieval', 'default': False},
    'lbl_std_atmos': {'value': 6, 'comment': 'Standard atmosphere to use in LBLRTM and MonoRTM calcs', 'default': False},
    'path_std_atmos': {'value': '/home/tropoe/vip/src/input/idl_code/std_atmosphere.idl', 'comment': 'The path to the IDL save file with the standard atmosphere info in it', 'default': False},
    'lbl_tape3': {'value': 'TAPE3.350-1600cm-1.first_7_molecules', 'comment': 'The TAPE3 file to use in the lblrtm calculation.  Needs to be in the directory lbl_home/hitran/', 'default': False},
    'monortm_version': {'value': 'v5.0', 'comment': 'String with the version information on MonoRTM', 'default': False},
    'monortm_wrapper': {'value': '/home/tropoe/vip/src/monortm_v5.0/wrapper/monortm_v5', 'comment': 'Turner wrapper to run MonoRTM', 'default': False},
    'monortm_exec': {'value': '/home/tropoe/vip/src/monortm_v5.0/monortm/monortm_v5.0_linux_intel_sgl', 'comment': 'AERs MonoRTM executable', 'default': False},
    'monortm_spec': {'value': '/home/tropoe/vip/src/monortm_v5.0/monolnfl_v1.0/TAPE3.spectral_lines.dat.0_55.v5.0_veryfast', 'comment': 'MonoRTM spectral database\n', 'default': False},

    'lblrtm_jac_option': {'value': 4, 'comment': '1 - LBLRTM Finite Diffs, 2 - 3calc method (deprecated), 3 - deltaOD method (deprecated), 4 - interpol method', 'default': False},
    'lblrtm_forward_threshold': {'value': 0., 'comment': 'The upper LWP threshold [g/m2] to use LBLRTM vs. radxfer in forward calculation', 'default': False},
    'lblrtm_jac_interpol_npts_wnum': {'value': 10, 'comment': 'The number of points per wnum to use in the compute_jacobian_irs_interpol() function', 'default': False},
    'monortm_jac_option': {'value': 2, 'comment': '1 - MonoRTM Finite Diffs, 2 - 3calc method', 'default': False},
    'jac_max_ht': {'value': 8.0, 'comment': 'Maximum height to compute the Jacobian [km AGL]', 'default': False},
    'max_iterations': {'value': 10, 'comment': 'The maximum number of iterations to use', 'default': False},
    'cvgmult': {'value': 0.25, 'comment':'The multiplier used for the convergence criteria (e.g., di2m < cvgmult*dim[Y])', 'default': False}, 
    'first_guess': {'value': 1, 'comment': '1 - use prior as FG, 2 - use lapse rate and 60% RH profile as FG, 3 - use previous sample as FG', 'default': False},
    'superadiabatic_maxht': {'value': 0.300, 'comment': 'The maximum height a superadiabatic layer at the surface can have [km AGL]', 'default': False},
    'spectral_bands': {'value': 'None', 'comment': 'An array of spectral bands to use (e.g. 612-618,624-660,674-713,713-722,538-588,793-804,860.1-864.0,872.2-877.5,898.2-905.4)', 'default': False},
    'retrieve_temp': {'value': 1, 'comment': '0 - do not retrieve temp, 1 - do retrieve temp (default)', 'default': True},
    'retrieve_wvmr': {'value': 1, 'comment': '0 - do not retrieve wvmr, 1 - do retrieve wvmr (default)', 'default': True},
    'retrieve_co2': {'value': 0, 'comment': '0 - do not retrieve co2, 1 - do retrieve co2 (step model), 2 - do retrieve co2 (exponential model -- disabled)', 'default': False},
    'fix_co2_shape': {'value': 0, 'comment': '(This option only works with retrieve_co2=1): 0 - retrieve all three coefs, 1 - shape coef is f(PBLH) and fixed', 'default': False},
    'retrieve_ch4': {'value': 0, 'comment': '0 - do not retrieve ch4, 1 - do retrieve ch4 (step model), 2 - do retrieve ch4 (exponential model -- disabled)', 'default': False},
    'fix_ch4_shape': {'value': 0, 'comment': '(This option only works with retrieve_ch4=1): 0 - retrieve all three coefs, 1 - shape coef is f(PBLH) and fixed', 'default': False},
    'retrieve_n2o': {'value': 0, 'comment': '0 - do not retrieve n2o, 1 - do retrieve n2o (step model), 2 - do retrieve n2o (exponential model -- disabled)', 'default': False},
    'fix_n2o_shape': {'value': 0, 'comment': '(This option only works with retrieve_n2o=1): 0 - retrieve all three coefs, 1 - shape coef is f(PBLH) and fixed', 'default': False},
    'retrieve_lcloud': {'value': 1, 'comment': '0 - do not retrieve liquid clouds, 1 - retrieve liquid cloud properties', 'default': True},
    'retrieve_icloud': {'value': 0, 'comment': '0 - do not retrieve   ice  clouds, 1 - retrieve   ice  cloud properties', 'default': True},
    'lcloud_ssp': {'value': '/home/tropoe/vip/src/input/ssp_db_files/ssp_db.mie_wat.gamma_sigma_0p100', 'comment': 'SSP file for liquid cloud properties', 'default': False},
    'icloud_ssp': {'value': '/home/tropoe/vip/src/input/ssp_db_files/ssp_db.mie_ice.gamma_sigma_0p100', 'comment': 'SSP file for ice cloud properties', 'default': False},
    'qc_rms_value': {'value': 10.0, 'comment': 'The RMSa value between ((obs minus calc)/obs_uncert), with values less than this being \'good\' ', 'default': True},
    'qc_gamma_value': {'value': 5.0, 'comment': 'The gamma value, with values less than this being \'good\' \n', 'default': True},

    'recenter_prior': {'value': 1, 'comment': '0 - do not recenter, 1 - Recenter WVMR based on sfc wv field and using conserve-RH for temp, 2 - Recenter WVMR based on PWV and conserve-RH for temp, 3 - Recenter WVMR based on sfc wv field and using conserve-covariance for temp, 4 - Recenter WVMR based on PWV and conserve-covariance for temp, 5 - Recenter based on near-sfc radiometric air temperature', 'default': True},
    'recenter_input': {'value': 0.0, 'comment': 'Sfc WVMR or PWV value to use in the recentering process. Set to zero for the value to be determined from other input data (i.e. sfc met)', 'default': False},
    'recenter_covar_min_sfactor': {'value': 0.6, 'comment': 'Minimum scale factor that will be used to rescale the prior covariance matrix', 'default': False},
    'recenter_covar_max_sfactor': {'value': 1.4, 'comment': 'Maximum scale factor that will be used to rescale the prior covariance matrix', 'default': False},
    'prior_tikhonov_factor': {'value': 1.5e-5, 'comment': 'The Tikhonov factor (>= 0) to apply at the prior TQ covariance matrix to improve its conditioning (set to 0 to turn off)', 'default': False},
    'prior_t_ival': {'value': 1.0, 'comment': 'The prior inflation factor (>= 1) to apply at the surface for temperature', 'default': False},
    'prior_t_iht': {'value': 1.0, 'comment': 'The height [km AGL] where the inflation factor goes to 1 (linear) for temperature', 'default': False},
    'prior_q_ival': {'value': 1.0, 'comment': 'The prior inflation factor (>= 1) to apply at the surface for water vapor mixing ratio', 'default': False},
    'prior_q_iht': {'value': 1.0, 'comment': 'The height [km AGL] where the inflation factor goes to 1 (linear) for water vapor mixing ratio', 'default': False},
    'prior_tq_cov_val': {'value': 1.0, 'comment': 'A multiplicative value (0 < val <= 1) that is used to decrease the covariance in the prior between temperature and WVMR at all heights', 'default': False},
    'prior_chimney_ht': {'value': 0.0, 'comment': 'The height of any \'chimney\' [km AGL]; prior data below this height are totally decorrelated', 'default': False},
    'prior_co2_mn': {'value': [-1.0, 5, -5], 'comment': 'Mean co2 concentration [ppm] (see \'retrieve_co2\' above)', 'default': False},
    'prior_co2_sd': {'value': [2.0, 15, 3], 'comment': '1-sigma uncertainty in co2 [ppm]', 'default': False},
    'prior_ch4_mn': {'value': [1.793, 0, -5], 'comment': 'Mean ch4 concentration [ppm] (see \'retrieve_ch4\' above)', 'default': False},
    'prior_ch4_sd': {'value': [0.0538, 0.0015, 3], 'comment': '1-sigma uncertainty in ch4 [ppm]', 'default': False},
    'prior_n2o_mn': {'value': [0.310, 0, -5], 'comment': 'Mean n2o concentration [ppm] (see \'retrieve_n2o\' above)', 'default': False},
    'prior_n2o_sd': {'value': [0.0093, 0.0002, 3], 'comment': '1-sigma uncertainty in n2o [ppm]', 'default': False},
    'prior_lwp_mn': {'value':  0.0, 'comment': 'Mean LWP [g/m2]', 'default': True},
    'prior_lwp_sd': {'value': 200.0, 'comment': '1-sigma uncertainty in LWP [g/m2]', 'default': True},
    'prior_lReff_mn': {'value': 8.0, 'comment': 'Mean liquid Reff [microns]', 'default': True},
    'prior_lReff_sd': {'value': 4.0, 'comment': '1-sigma uncertainty in liquid Reff [microns]', 'default': True},
    'prior_itau_mn': {'value': 0.0, 'comment': 'Mean ice cloud optical depth (geometric limit)', 'default': True},
    'prior_itau_sd': {'value': 50.0, 'comment': '1-sigma uncertainty in ice cloud optical depth', 'default': True},
    'prior_iReff_mn': {'value': 25.0, 'comment': 'Mean ice cloud Reff [microns]', 'default': True},
    'prior_iReff_sd': {'value': 8.0, 'comment': '1-sigma uncertainty in ice cloud} Reff [microns] \n', 'default': True},
    'min_PBL_height': {'value': 0.050, 'comment': 'The minimum height of the planetary boundary layer (used for trace gases) [km AGL]', 'default': False},
    'max_PBL_height': {'value': 5.0, 'comment': 'The maximum height of the planetary boundary layer (used for trace gases) [km AGL]', 'default': False},
    'nudge_PBL_height': {'value': 0.5, 'comment': 'The temperature offset (nudge) added to the surface temp to find PBL height [C] \n', 'default': False},

    'plot_output': {'value': 0, 'comment': '0 - do not generate quicklooks of the output; 1 - generate quicklooks of the output after every iteration; 2 - plot from pre-existing file and abort', 'default': True},
    'plot_path'  : {'value': 'None', 'comment': 'Path to store outputted quicklooks. Defaults to output path', 'default': True},
    'plot_rootname': {'value': 'None', 'comment': 'Sets a prefix to the quicklooks. Defaults to the output_rootname vip entry', 'default': True},
    'plot_xlim'  : {'value': [0., 24.], 'comment': 'Hours of day', 'default': True},
    'plot_ylim'  : {'value': [0., 2.], 'comment': 'Height AGL in km', 'default': True},
    'plot_temp_lim'    : {'value': [-10., 20.], 'comment': 'Temperature limit (C)', 'default': True},
    'plot_wvmr_lim'    : {'value': [0., 20.], 'comment': 'WVMR limit (in g/kg)', 'default': True},
    'plot_tuncert_lim' : {'value': [0., 3.], 'comment': 'Temperature uncertainty limit (C)', 'default': True},
    'plot_wvuncert_lim': {'value': [0., 3.], 'comment': 'WVMR uncertainty limit (g/kg)', 'default': True},
    'plot_theta_lim'   : {'value': [290., 320.], 'comment': 'Potential Temperature limit (K)', 'default': True},
    'plot_rh_lim'      : {'value': [0., 100.], 'comment': "Relative humidity limit (%)", 'default': True},
    'plot_thetae_lim'  : {'value': [290., 320.], 'comment': 'equivalent potential temperature limit (K)', 'default': True},
    'plot_dewpt_lim'   : {'value': [-15., 25.], 'comment': 'Dewpoint limit (C)', 'default': True},
    'plot_comment'     : {'value': 'None', 'comment': 'String to put in the lower left of the quicklook figures', 'default': True},
    'plot_tres_min_gap': {'value': 'None', 'comment': 'In minutes. Defaults to 3 x tres', 'default': True},
    'plot_lwp_cbh_threshold': {'value': 8.0, 'comment': 'Min LWP to show CBH dots', 'default': True}

}
)

full_mixcra_vip = ({
    'success': {'value': 0, 'comment': 'Interal success flag. Not for outside use', 'default': False},
    'tres_lblrtm': {'value': 3, 'comment': 'Temporal resoltuion of the LBLRTM runs [h]', 'default':True},
    'workdir': {'value':'/data/tag', 'comment': 'The working directory for the iterating retrieval','default':True},
    'avg_instant': {'value': 1, 'comment': 'A flag to specify -1:avg with no sqrt(N), 0:avg with sqrt(N), or 1:instantaneous', 'default': False},

    'delete_temporary': {'value':1, 'comment': 'Delete the temporary output after MIXCRA completes (1) or keep it (0)', 'default':False},

    'irs_type': {'value': 0, 'comment': '0- output, options, and stop, 1 - ARM AERI data, 2 - dmv2cdf AERI data (C1_rnc.cdf and _sum.cdf), 3 - dmv2ncdf (C1.RNC.cdf and .SUM.cdf), 5- ASSIST, -1 - MWR data is to be used as MASTER dataset (no IRS data being read in)', 'default': True},
    'irsch1_path': {'value': "/data/aerich1", 'comment': 'Path to the IRS ch1 radiance files', 'default': True},
    'irsch2_path': {'value': "/data/aerich2", 'comment': 'Path to the IRS ch1 radiance files', 'default': True},
    'irssum_path': {'value': "/data/aerisum", 'comment': 'Path to the IRS summary files', 'default': True},
    'irseng_path': {'value': "/data/aerieng", 'comment': 'Path to the IRS engineering files', 'default': True},
    'irs_sampling': {'value': 1, 'comment': 'Process every (irs_sampling)th spectrum', 'default':True},
    'irs_min_noise_flag':{'value': 0, 'comment': 'If non-zero, then the irs_min_noise_spectrum will be used as a floor; otherwise, will use input IRS noise spectrum from instrument', 'default': False},
    'irs_min_noise_wnum':{'value': '500,522,546,575,600,631,747,1439,1770,1884,2217,3000', 'comment': 'Wavenumber array [cm-1] for the minimum noise spectrum', 'default': False},
    'irs_min_noise_spec':{'value': '65.309,14.056,3.283,1.333,0.813,0.557,0.304,0.581,0.822,0.025,0.023,0.044', 'comment': 'Noise array [RU] for the minimum noise spectrum', 'default': False},
    'irs_noise_inflation':{'value': 1.0, 'comment': 'Value to increase the assumed random error in the IRS data', 'default': True},
    'irs_smooth_noise': {'value': 0, 'comment': 'The temporal window [minutes] used to smooth the IRS noise with time', 'default': False},
    'irs_band_noise_inflation': {'value': 0, 'comment': '0 -- off (no noise inflation), 1 -- on (will inflate the noise in the spectral band below; requires surface WVMR input from met station)', 'default': False}, 
    'irs_band_noise_wnums': {'value': '775.0,810.0', 'comment': 'A comma separated list of two wavenumbers; the noise spectrum between these will be inflated', 'default': False}, 
    'irs_band_noise_sfc_npts': {'value': 4, 'comment': 'The number of points in the x- (surface water vapor) and y- (noise inflation) axes', 'default': False}, 
    'irs_band_noise_sfc_wvmr_vals': {'value': '0.,7,12,18', 'comment': 'A comma separated list of the surface water vapor mixing ratio values [g/kg] for the x-axis', 'default': False}, 
    'irs_band_noise_sfc_multiplier': {'value': '20.0,20,1,1', 'comment': 'A comma separated list of multipliers [unitless] for the y-axis', 'default': False}, 
    'irs_zenith_scene_mirror_angle': {'value':180, 'comment': 'The angle [degrees] of the IRS scene mirror that corresponds to zenith', 'default':True},
    'irs_min_675_bt': {'value': 233., 'comment': 'Minimum brightness temp [K] in the 675-680 cm-1 window -- this is QC screen', 'default': True},
    'irs_max_675_bt': {'value': 313., 'comment': 'Maximum brightness temp [K] in the 675-680 cm-1 window -- this is QC screen\n', 'default': True},
    'irs_use_missingDataFlag': {'value': 1, 'comment': 'Set this to 1 to use the field \'missingDataFlag\' (from the ch1 file) to remove bad IRS data from analysis. If zero, then all IRS data will be processed,', 'default': False},
    'irs_calib_pres': {'value': [0.0, 1.0], 'comment': 'Intercept [mb] and slope [mb/mb to calib (newP = int = slope*obsP) (need comma between them)', 'default': False},
    'irs_use_missingDataFlag': {'value': 1, 'comment': 'Set this to 1 to use the field \'missingDataFlag\' (from the ch1 file) to remove bad IRS data from analysis. If zero, then all IRS data will be processed,', 'default': False},
    'irs_hatch_switch': {'value': 1, 'comment': '1 - only include hatchOpen=1 when averaging, 0 - include all IRS samples when averaging', 'default': False},

    'mwr_type': {'value': 0, 'comment': '0 - none, 1 - Tb fields are individual time series, 2 - Tb field is 2-d array', 'default': True},
    'mwr_path': {'value': '/data/mwr', 'comment': 'Path to the MWR data', 'default': True},
    'mwr_rootname': {'value': 'mwr', 'comment': 'Rootname of the MWR data files', 'default': True},
    'mwr_elev_field': {'value': 'elev', 'comment': 'Name of the scene mirror elevation field; use \'none\' if all data are zenith', 'default': True},
    'mwr_freq_field': {'value': 'freq', 'comment': 'Name of the frequency field needed if mwr_type >= 2', 'default': True},
    'mwr_n_tb_fields': {'value': 0, 'comment': 'Number of fields to read in', 'default': True},
    'mwr_tb_field_names': {'value': 'tbsky23,tbsky31', 'comment': 'Comma separated list of field names for the Tb fields', 'default': True},
    'mwr_tb_field1_tbmax': {'value': 100., 'comment': 'Maximum value [K] in the first Tb field, used for QC', 'default': True},
    'mwr_tb_freqs': {'value': '23.8,31.4', 'comment': 'Comma separated list of frequency [GHz] of MWR Tb fields', 'default': True},
    'mwr_tb_noise': {'value': '0.3,0.3', 'comment': 'Comma separated list of noise levels [K] in the MWR Tb fields', 'default': True},
    'mwr_tb_bias': {'value': '0.0,0.0', 'comment': 'Comma separated list of bias [K] in the MWR Tb fields; this value is ADDED to the MWR observations', 'default': True},
    'mwrscan_type': {'value': 0, 'comment': '0 - none, 1 - Tb fields are individual time series, 2 - Tb field is 2-d array', 'default': False},
    'mwr_time_delta': {'value': 0.083, 'comment': 'The maximum amount of time [hours] that the MWR zenith obs must be to the sampling time to be used \n', 'default': True},

    'tropoe_path': {'value': "/data/tropoe", 'comment': 'Path to TROPoe output files', 'default': True},
    'tropoe_rmsr_thres': {'value': 5.0, 'comment': 'TROPoe input profiles need RMSr values below this threshold to be used \n', 'default': True},

    'station_lat': {'value': -999., 'comment': 'Station latitude [degN]; if negative get value from IRS/MWR data file', 'default': True},
    'station_lon': {'value': -999., 'comment': 'Station longitude [degE]; if negative get value from IRS/MWR data file', 'default': True},
    'station_alt': {'value': -999., 'comment': 'Station altitude [m MSL]; if negative get value from IRS/MWR data file \n', 'default': True},

    'cbh_type': {'value': 0, 'comment': '0 - output options and stop, 1 - VCEIL, 2 - Gregs ASOS CBH file, 3 - CLAMPS DLfp data, 4 - ARM dlprofwstats data', 'default': True},
    'cbh_path': {'value': '/data/ceil', 'comment': 'Path to the CBH data', 'default': True},
    'cbh_window_in': {'value': 20, 'comment': 'Inner temporal window (full-size) centered upon IRS time to look for cloud', 'default': True},
    'cbh_window_out': {'value': 180, 'comment': 'Outer temporal window (full-size) centered upon IRS time to look for cloud}', 'default': True},
    'cbh_default_ht': {'value': 2.0, 'comment': 'Default CBH height [km AGL], if no CBH data found \n', 'default': True},

    'lwc': {'value': 0.3, 'comment': 'The maximum liquid water content in a layer; used to help find cloud top [g/m^3]', 'default': True},
    'iwc': {'value': 0.1, 'comment': 'The maximum ice water content in a layer; used to find cloud top [g/m^3]', 'default':True},
    'fix_cbh': {'value': -1.0, 'comment': 'If greater than zero, then use this fixed value as the CBH (not the TROPoe value); the absolute value of the fix_cbh field will be used as the CBH if the observed CBH < 0 \n', 'default':True},

    'output_rootname': {'value': 'mixcra', 'comment': 'The rootname of the output file', 'default': True},
    'output_path': {'value': '/data/mixcra', 'comment': 'Path where the output file will be placed \n', 'default': True},

    'lblrtm_home': {'value': '/home/tropoe/vip/src/lblrtm_v12.17/lblrtm', 'comment': 'String with the LBL_HOME path', 'default': False},
    'lbldis_exec': {'value': '/home/tropoe/vip/src/lbldis.Release_3_0/lbldis', 'comment': 'The LBLDIS executable (with path)', 'default': False},
    'lbl_std_atmos': {'value': 6, 'comment': 'Standard atmosphere to use in LBLRTM and MonoRTM calcs', 'default': False},
    'lbl_tape3': {'value': 'TAPE3.10-3500cm-1.first_7_molecules', 'comment': 'The TAPE3 file to use in the lblrtm calculation.  Needs to be in the directory lbl_home/hitran/', 'default': False},
    'ssf': {'value': '/home/tropoe/vip/src/input/ssp_db_files/solar.kurucz.rad.1cm-1binned.full_disk.asc', 'comment':'The solar source fuction', 'default': False},
    'lblout': {'value': 'lblout', 'comment': 'The output path to the gaseous optical depth from the LBLRTM run','default': False},
    'monortm_version': {'value': 'v5.0', 'comment': 'String with the version information on MonoRTM', 'default': False},
    'monortm_wrapper': {'value': '/home/tropoe/vip/src/monortm_v5.0/wrapper/monortm_v5', 'comment': 'Turner wrapper to run MonoRTM', 'default': False},
    'monortm_exec': {'value': '/home/tropoe/vip/src/monortm_v5.0/monortm/monortm_v5.0_linux_intel_sgl', 'comment': 'AERs MonoRTM executable', 'default': False},
    'monortm_spec': {'value': '/home/tropoe/vip/src/monortm_v5.0/monolnfl_v1.0/TAPE3.spectral_lines.dat.0_55.v5.0_veryfast', 'comment': 'MonoRTM spectral database\n', 'default': False},

    'spectral_bands': {'value': 'None', 'comment': 'An 2-m matrix of spectral bands to use', 'default': False},
    'sfc_emissivity': {'value': 'None', 'comment': 'An 2-m matrix of spectral bands to use', 'default': False},
    'include_full_calc': {'value': -1, 'comment': 'If greater than zero, include obs and calc (using final retrieved values) for full spectral band at this spectral resolution [cm-1]', 'default': True},

    'nstreams': {'value': 16, 'comment': 'The number of streams to use in the LBLDIS calculation', 'default': False},
    'solver': {'value': 0, 'comment': 'The RT solver to use in LBLDIS: 0-DISORT, 1-AbsorptionOnly', 'default':False},
    'math_choice': {'value': 0, 'comment': 'The choice of Rodgers Equations to use 0->Eq 5.9, 1->Eq 5.8, 2->Eq 5.8 with annealing','default':False},

    'maxiter': {'value': 10, 'comment': 'The maximum number of iterations to use', 'default': False},
    'converge_factor': {'value': 0.2, 'comment': 'To be considered converged, di2m < converge_factor*dimY; must be (0,1]', 'default':False},

    'retrieve_lcloud': {'value': 1, 'comment': '0 - do not retrieve liquid clouds, 1 - retrieve liquid cloud properties', 'default': True},
    'retrieve_icloud': {'value': 0, 'comment': '0 - do not retrieve   ice  clouds, 1 - retrieve   ice  cloud properties', 'default': True},
    'lcloud_ssp': {'value': '/home/tropoe/vip/src/input/ssp_db_files/ssp_db.mie_wat.gamma_sigma_0p100', 'comment': 'SSP file for liquid cloud properties', 'default': True},
    'icloud_ssp': {'value': '/home/tropoe/vip/src/input/ssp_db_files/ssp_db.mie_ice.gamma_sigma_0p100', 'comment': 'SSP file for   ice  cloud properties', 'default': True},
    'compute_lwp': {'value': 1, 'comment': 'Set this to 1 if liquid is being retrieved with the lcloud variables', 'default':True},
    'apply_tcloud_constraints': {'value': 1, 'comment': 'If set and algorithm is in dual-phase, then do not let liquid water exist below -40 or ice exist above 0 degC \n', 'default':True},

    'ref_wnum': {'value': -1., 'comment': 'The reference wavenumber for the retrieved optical depths (-1 implies visible)', 'default':True},
    'min_ltau': {'value': 0.001, 'comment': 'The minimum optical depth [unitless] at the ref_wnum for the liquid', 'default':False},
    'min_itau': {'value': 0.001, 'comment': 'The minimum optical depth [unitless] at the ref_wnum for the ice', 'default':False},

    'prior_ltau_mn': {'value':  10.0, 'comment': 'Mean liquid cloud optical depth (geometric limit)', 'default': True},
    'prior_ltau_sd': {'value': 50.0, 'comment': '1-sigma uncertainty in liquid cloud optical depth', 'default': True},
    'prior_lReff_mn': {'value': 8.0, 'comment': 'Mean liquid Reff [microns]', 'default': True},
    'prior_lReff_sd': {'value': 4.0, 'comment': '1-sigma uncertainty in liquid Reff [microns]', 'default': True},
    'prior_itau_mn': {'value': 4.0, 'comment': 'Mean ice cloud optical depth (geometric limit)', 'default': True},
    'prior_itau_sd': {'value': 5.0, 'comment': '1-sigma uncertainty in ice cloud optical depth', 'default': True},
    'prior_iReff_mn': {'value': 25.0, 'comment': 'Mean ice cloud Reff [microns]', 'default': True},
    'prior_iReff_sd': {'value': 8.0, 'comment': '1-sigma uncertainty in ice cloud} Reff [Reff]', 'default': True},
    'prior_corr_ltau_lreff': {'value': 0.0, 'comment': 'The correlation to assume between ltau and lreff in the prior', 'default':True},
    'prior_corr_ltau_itau': {'value': 0.0, 'comment': 'The correlation to assume between ltau and itau in the prior', 'default':True},
    'prior_corr_ltau_ireff': {'value': 0.0, 'comment': 'The correlation to assume between ltau and ireff in the prior', 'default':True},
    'prior_corr_lreff_itau': {'value': 0.0, 'comment': 'The correlation to assume between lreff and itau in the prior', 'default':True},
    'prior_corr_lreff_ireff': {'value': 0.0, 'comment': 'The correlation to assume between lreff and ireff in the prior', 'default':True},
    'prior_corr_itau_ireff': {'value': 0.0, 'comment': 'The correlation to assume between itau and ireff in the prior', 'default':True},
    'vip_filename': {'value': 'none', 'comment': 'Just for tracability', 'default':True}
}
)


################################################################################
#This routine reads in the controlling parameters from the VIP file.
################################################################################

def read_tropoe_vip_file(filename,globatt,verbose,debug,dostop):

    # Read in the all the values from the full vip dict and create a smaller vip file
    vip = {}
    for key in full_tropoe_vip.keys():
        vip[key] = full_tropoe_vip[key]['value']

    if os.path.exists(filename):

        if verbose >= 1:
            print('  Reading the VIP file: ' + filename)

        try:
            # inputt = np.genfromtxt(filename, dtype=str,comments='#', usecols = (0,1,2))
            inputt = np.genfromtxt(filename, dtype=str, comments='#', delimiter='=', autostrip=True)
        except Exception as e:
            print(e)
            print('  There was an problem reading the VIP file. Check formatting.')
            return vip
    else:
        print('  The VIP file ' + filename + ' does not exist')
        return vip

    if len(inputt) == 0:
        print('  There were no valid lines found in the VIP file')
        return vip

    # Look for obsolete tags, and abort if they are found (forcing user to update their VIP file)
    obsolete_tags = ['AERI_LAT','AERI_LON','AERI_ALT','PSFC_MIN','PSFC_MAX','AERI_TYPE', 'output_file_keep_small']
    obsolete_idx  = np.zeros_like(obsolete_tags, dtype=int)
    vip_keys = np.array([k.upper() for k in inputt[:,0]])
    for i in range(len(obsolete_tags)):
        foo = np.where(obsolete_tags[i].upper() == vip_keys)[0]
        if(len(foo) > 0):
            obsolete_idx[i] = 1
    foo = np.where(obsolete_idx > 0)[0]
    if(len(foo) > 0):
        print('Error: there were obsolete tags in the VIP file:')
        for i in range(len(foo)):
            print('     '+obsolete_tags[foo[i]])
            return vip

    # Create a tracking array to make sure all keys in the user provided
    # vip file are used
    track = np.ones(inputt.shape[0])
    # Look for these tags
    nfound = 1
    for key in vip.keys():
        if key != 'success':
            nfound += 1
            if key == 'vip_filename':
                vip['vip_filename'] = filename
            else:
                foo = np.where(key == inputt[:,0])[0]
                if len(foo) > 1:
                    print('Error: There were multiple lines with the same key in VIP file: ' + key)
                    return vip

                elif len(foo) == 1:
                    track[foo] = 0
                    if verbose == 3:
                        print('Loading the key ' + key)
                    if key == 'spectral_bands':
                        bands = np.zeros((2, maxbands))-1
                        tmp = inputt[foo,1][0].split(',')

                        if len(tmp) >= maxbands:
                            print('Error: There were more spectral bands defined than maximum allowed (maxbands = ' + str(maxbands) + ')')
                            return vip

                        for j in range(len(tmp)):
                            feh = tmp[j].split('-')
                            if len(feh) != 2:
                                print('Error: Unable to properly decompose the spectral_bands key')
                                if dostop:
                                    wait = input('Stopping inside to debug this bad boy. Press enter to continue')
                                return vip
                            bands[0,j] = float(feh[0])
                            bands[1,j] = float(feh[1])
                        vip['spectral_bands'] = bands

                    elif key == 'irs_calib_pres':
                        feh = inputt[foo,1][0].split(',')
                        if len(feh) != 2:
                            print('Error: The key irs_calib_pres in VIP file must be intercept, slope')
                            if dostop:
                                wait = input('Stopping inside to debug this bad boy. Press enter to continue')
                            return vip
                        vip['irs_calib_pres'][0] = float(feh[0])
                        vip['irs_calib_pres'][1] = float(feh[1])
                        if vip['irs_calib_pres'][1] <= 0.0001:
                            print('Error: The key irs_calib_pres in VIP file must have positive slope')
                            if dostop:
                                wait = input('Stopping inside to debug this bad boy. Press enter to continue')
                            return vip

                    elif key in [
                        'ext_wv_noise_mult_val', 'ext_wv_noise_mult_hts', 'ext_temp_noise_adder_val',
                        'ext_temp_noise_adder_hts', 'mod_wv_noise_mult_val', 'mod_wv_noise_mult_hts',
                        'mod_temp_noise_adder_val', 'mod_temp_noise_adder_hts', 'prior_co2_mn', 'prior_co2_sd',
                        'prior_ch4_mn', 'prior_ch4_sd', 'prior_n2o_mn', 'prior_n2o_s','plot_xlim', 'plot_ylim',
                        'plot_temp_lim', 'plot_wvmr_lim', 'plot_tuncert_lim', 'plot_wvuncert_lim',
                        'plot_theta_lim', 'plot_rh_lim', 'plot_thetae_lim', 'plot_dewpt_lim',
                        'add_tropoe_T_noise_adder_val', 'add_tropoe_T_noise_adder_hts',
                        'add_tropoe_q_noise_mult_val', 'add_tropoe_q_noise_mult_hts',
                        'irs_mlev_cbh_wnum_range', 'irs_tcld_wnum_range'
                    ]:

                        feh = inputt[foo,1][0].split(',')
                        needed_length = len(vip[key])
                        if len(feh) != needed_length:
                            print('Error: The key ' + key + ' in VIP file must be a ' + str(needed_length) + ' element array')
                            if dostop:
                                wait = input('Stopping inside to debug this bad boy. Press enter to continue')
                            return vip

                        for m in range(needed_length):
                            vip[key][m] = float(feh[m])

                    else:
                        vip[key] = type(vip[key])(inputt[foo,1][0])
                else:
                    if verbose == 3:
                        print('UNABLE to find the key ' + key)
                    nfound -= 1

    if verbose == 3:
        print(vip)
    if verbose >= 2:
        print('    There were ' + str(nfound) + ' entries found out of ' + str(len(list(vip.keys()))))

    # Now look for any global attributes that might have been entered in the file

    matching = [s for s in inputt[:,0] if "globatt" in s]

    if verbose >= 2:
        print('    There were ' + str(len(matching)) + ' global attributes found')

    for i in range(len(matching)):
        foo = np.where(matching[i] == inputt[:,0])[0]
        track[foo] = 0
        globatt[matching[i][8:]] = inputt[foo,1][0]

    # Need to trap condition where spectral_bands was not set (and thus is the default string "None")
    # and then reset it to a standard set of bands used for IR sounding
    if(type(vip['spectral_bands']) == str):
        blo = [612., 624, 674, 713, 538, 793, 860.1, 872.2, 898.2]
        bhi = [618., 660, 713, 722, 588, 804, 864.0, 877.5, 905.4]
        vip['spectral_bands'] = np.array([blo,bhi])

    # Now that we've read everything in, go back and reset anything that could be dependent
    # on another VIP entry
    if vip['plot_path'] == 'None':
        vip['plot_path'] = vip['output_path']

    if vip['plot_rootname'] == 'None':
        vip['plot_rootname'] = vip['output_rootname']

    if vip['plot_tres_min_gap'] == 'None':
        if vip['tres'] != 0:
            vip['plot_tres_min_gap'] = vip['tres'] * 3

    if vip['plot_comment'] == 'None':
        vip['plot_comment'] = ''

    foo = np.nonzero(track)[0]
    if len(foo) > 0:
        print('  There were undefined entries in the VIP file:')
        for i in range(len(foo)):
            print('    ' + inputt[foo,0][i])
        return vip
    else:
        vip['success'] = 1

    if dostop:
        wait = input('Stopping inside to debug this bad boy. Press enter to continue')
    return vip

################################################################################
# This code performs some QC on the entries in the VIP structure to ensure
# that they are within a valid range.  Not every entry is checked...
################################################################################

def check_vip(vip):
    flag = 0           # Default is that everything is ok
    
    if ((vip['avg_instant'] < -1) or (vip['avg_instant'] > 1)):
        print('Error: avg_instant can only be set to -1, 0, or 1')
        flag = 1

    if ((vip['output_clobber'] < 0) or (vip['output_clobber'] > 2)):
        print('Error: The output_clobber flag can only be set to 0, 1, or 2')
        flag = 1
    
    if ((vip['output_akernal'] < 0) or (vip['output_akernal'] > 2)):
        print('Error: The output_akernal flag can only be set to 0, 1, or 2')
        flag = 1
        
    if ((vip['irs_min_noise_flag'] < 0) or (vip['irs_min_noise_flag'] > 1)):
        print('Error: irs_min_noise_flag must be 0 or 1')
        flag = 1
    
    if vip['irs_noise_inflation'] < 1:
        print('Error: irs_noise_inflation must be >= 1')
        flag = 1
        
    if ((vip['irs_smooth_noise'] < 0) or (vip['irs_smooth_noise'] > 1)):
        print('Error: irs_smooth_noise must be 0 or 1')
        flag = 1
    
    if ((vip['irs_band_noise_inflation'] < 0) or (vip['irs_band_noise_inflation'] > 1)):
        print('Error: irs_band_noise_inflation must be 0 or 1')
        flag = 1
    
    if ((vip['irs_use_missingDataFlag'] < 0) or (vip['irs_use_missingDataFlag'] > 1)):
        print('Error: The irs_use_missingDataFlag must be either 0 or 1')
        flag = 1

    if ((vip['irs_hatch_switch'] < 0) or (vip['irs_hatch_switch'] > 1)):
        print('Error: The irs_hatch_switch must be either 0 or 1')
        flag = 1
    
    if ((vip['irs_ignore_status_missingDataFlag'] < 0) or (vip['irs_ignore_status_missingDataFlag'] > 1)):
        print('Error: irs_ignore_status_missingDataFlag must be either 0 or 1')
        flag = 1

    if ((vip['irs_ignore_status_hatch'] < 0) or (vip['irs_ignore_status_hatch'] > 1)):
        print('Error: irs_ignore_status_hatch must be either 0 or 1')
        flag = 1
        
    if ((vip['irs_fv'] < 0.0) or (vip['irs_fv'] > 0.03)):
        print('Error: The IRS fv is too small or too large')
        flag = 1

    if ((vip['irs_fa'] < 0.0) or (vip['irs_fa'] > 0.03)):
        print('Error: The IRS fa is too small or too large')
        flag = 1

    if vip['mwr_time_delta'] <= 0:
        print('Error: mwr_time_delta must be positive')
        flag = 1
    
    if vip['mwrscan_time_delta'] <= 0:
        print('Error: mwr_time_delta must be positive')
        flag = 1
    
    if vip['ext_sfc_temp_random_error'] <= 0:
        print('Error: ext_sfc_temp_random_error must be positive')
        flag = 1
    
    if vip['ext_sfc_temp_rep_error'] < 0:
        print('Error: ext_sfc_temp_rep_error must be >= 0')
        flag = 1
    
    if vip['ext_sfc_rh_random_error'] <= 0:
        print('Error: ext_sfc_rh_random_error must be positive')
        flag = 1
    
    if vip['ext_sfc_wv_mult_error'] <= 0:
        print('Error: ext_sfc_wv_mult_error must be positive')
        flag = 1
        
    if vip['ext_sfc_wv_rep_error'] < 0:
        print('Error: ext_sfc_wv_rep_error must be >= 0')
        flag = 1
    
    if vip['ext_sfc_time_delta'] <= 0:
        print('Error: ext_sfc_time_delta must be positive')
        flag = 1
    
    if vip['ext_wv_add_rel_error'] < 0:
        print('Error: ext_wv_add_rel_error must be >= 0')
        flag = 1
    
    if vip['ext_wv_time_delta'] <= 0:
        print('Error: ext_wv_time_delta must be positive')
        flag = 1
    
    if vip['ext_temp_time_delta'] <= 0:
        print('Error: ext_temp_time_delta must be positive')
        flag = 1
    
    if vip['mod_wv_time_delta'] <= 0:
        print('Error: ext_wv_time_delta must be positive')
        flag = 1
    
    if vip['mod_temp_time_delta'] <= 0:
        print('Error: ext_temp_time_delta must be positive')
        flag = 1
    
    if vip['co2_sfc_rep_error'] < 0:
        print('Error: co2_sfc_rep_error must be >= 0')
        flag = 1
    
    if vip['co2_sfc_time_delta'] <= 0:
        print('Error: co2_sfc_time_delta must be positive')
        flag = 1
    
    if vip['cbh_window_in'] <= 0:
        print('Error: cbh_window_in must be positive')
        flag = 1
    
    if vip['cbh_window_out'] <= 0:
        print('Error: cbh_window_out must be positive')
        flag = 1
    
    if vip['cbh_window_in'] > vip['cbh_window_out']:
        print('Error: cbh_window_in must be less than cbh_window_out')
        flag = 1
    
    if vip['cbh_default_ht'] <= 0:
        print('Error: cbh_default_ht must be positive')
        flag = 1
    
    if vip['lblrtm_jac_interpol_npts_wnum'] <= 0:
        print('Error: lblrtm_jac_interpol_npts_wnum must be positive')
        flag = 1
    
    if vip['jac_max_ht'] <= 1.0:
        print('Error: The maximum height to compute the Jacobian is too small; please increase')
        flag = 1

    if ((vip['max_iterations'] < 0) or (vip['max_iterations'] > 25)):
        print('Error: The maximum number of iterations must be between 0 and 25')
        flag = 1
    
    if vip['superadiabatic_maxht'] < 0:
        print('Error: superadiabatic_maxht must be >= 0')
        flag = 1
    
    if (vip['retrieve_temp'] < 0) or (vip['retrieve_temp'] > 1):
        print('Error: retrieve_temp must be 0 or 1')
        flag = 1
    
    if (vip['retrieve_wvmr'] < 0) or (vip['retrieve_wvmr'] > 1):
        print('Error: retrieve_wvmr must be 0 or 1')
        flag = 1
    
    if (vip['retrieve_co2'] < 0) or (vip['retrieve_co2'] > 2):
        print('Error: retrieve_co2 must be 0, 1, or 2')
        flag = 1
    
    if (vip['retrieve_ch4'] < 0) or (vip['retrieve_ch4'] > 2):
        print('Error: retrieve_ch4 must be 0, 1, or 2')
        flag = 1
    
    if (vip['retrieve_n2o'] < 0) or (vip['retrieve_n2o'] > 2):
        print('Error: retrieve_n2o must be 0, 1, or 2')
        flag = 1
    
    if (vip['retrieve_lcloud'] < 0) or (vip['retrieve_lcloud'] > 1):
        print('Error: retrieve_lcloud must be 0 or 1')
        flag = 1
    
    if (vip['retrieve_icloud'] < 0) or (vip['retrieve_icloud'] > 2):
        print('Error: retrieve_icloud must be 0 or 1')
        flag = 1
    
    if vip['qc_rms_value'] <= 0:
        print('Error: qc_rms_value must be positive')
        flag = 1
    
    if ((vip['recenter_prior'] < 0) or (vip['recenter_prior'] > 5)):
        print('Error: recenter_prior must be 0, 1, 2, 3, 4, or 5')
        flag = 1
    
    if vip['recenter_input'] < 0:
        print('Error recenter_input must be >= 0')
        flag = 1
    
    if vip['prior_chimney_ht'] < 0:
        print('Error prior_chimney_ht must be >= 0')
        flag = 1
    
    if vip['prior_co2_sd'][0] <= 0 or vip['prior_co2_sd'][1] <= 0 or vip['prior_co2_sd'][2] <= 0:
        print('Error: the values of prior_co2_sd must be positive')
        flag = 1
    
    if vip['prior_ch4_sd'][0] <= 0 or vip['prior_ch4_sd'][1] <= 0 or vip['prior_ch4_sd'][2] <= 0:
        print('Error: the values of prior_ch4_sd must be positive')
        flag = 1
    
    if vip['prior_n2o_sd'][0] <= 0 or vip['prior_n2o_sd'][1] <= 0 or vip['prior_n2o_sd'][2] <= 0:
        print('Error: the values of prior_n2o_sd must be positive')
        flag = 1
        
    if ((vip['lbl_std_atmos'] < 1) or (vip['lbl_std_atmos'] > 6)):
        print('Error: The LBLRTM standard atmosphere must be an integer between 1 and 6')
        flag = 1
    
    if vip['prior_lwp_mn'] < 0:
        print('Error: The prior_lwp_mn must be >= 0')
        flag = 1
    
    if vip['prior_lwp_sd'] <= 0:
        print('Error: The prior_lwp_sd must be positive')
        flag = 1
    
    if vip['prior_lReff_mn'] <= 0:
        print('Error: The prior_lReff_mn must be positive')
        flag = 1
    
    if vip['prior_lReff_sd'] <= 0:
        print('Error: The prior_lReff_sd must be positive')
        flag = 1
    
    if vip['prior_itau_mn'] < 0:
        print('Error: The prior_itau_mn must be >= 0')
        flag = 1
    
    if vip['prior_lwp_sd'] <= 0:
        print('Error: The prior_itau_sd must be positive')
        flag = 1
    
    if vip['prior_iReff_mn'] <= 0:
        print('Error: The prior_iReff_mn must be positive')
        flag = 1
    
    if vip['prior_iReff_sd'] <= 0:
        print('Error: The prior_iReff_sd must be positive')
        flag = 1
    
    if vip['min_PBL_height'] <= 0:
        print('Error: min_PBL_height must be positive')
        flag = 1
    
    if vip['max_PBL_height'] <= 0:
        print('Error: max_PBL_height must be positive')
        flag = 1
    
    if vip['min_PBL_height'] > vip['max_PBL_height']:
        print('Error: min_PBL_height must be less than max_PBL_height')
        flag = 1
    
    if vip['nudge_PBL_height'] <= 0:
        print('Error: nudge_PBL_height must be positive')
        flag =1
        
    if ((vip['retrieve_lcloud'] == 0) and (vip['prior_lwp_mn'] > 0)):
        print('WARNING: retrieve_lcloud set to 0, but prior_lwp_mn is non-zero!')

    if ((vip['retrieve_icloud'] == 0) and (vip['prior_itau_mn'] > 0)):
        print('WARNING: retrieve_icloud set to 0, but prior_itau_mn is non-zero!')
    
    return flag

################################################################################
# This routine is called when TROPoe aborts
################################################################################

def abort(lbltmpdir, date):

    if os.path.exists(lbltmpdir):
        shutil.rmtree(lbltmpdir)

    print('>>> IRS retrieval on ' + str(date) + ' FAILED and ABORTED <<<')
    print('--------------------------------------------------------------------')
    print(' ')

#################################################################################
# This routine is called when Mixcra aborts
#################################################################################

def mixcra_abort(lbltmpdir, date):

    if os.path.exists(lbltmpdir):
        shutil.rmtree(lbltmpdir)

    print('>>> Mixcra retrieval on ' + str(date) + ' FAILED and ABORTED <<<')
    print('--------------------------------------------------------------------')
    print(' ')

################################################################################
#   This routine reads in the scattering properties from databases that were
#   created by Mie code (newmie_iteration) and by Ping Yang's computations (that
#   database was compiled by write_database.pro in IceScatProperties), and other
#   similar scattering property databases.  Rectangular matrices are returned,
#   which can then be processed by the routine get_scat_properties.pro to extract
#   out the properties depending on wavelength and effective radius.  Note that
#   there are two Mie databases: one for ice particles and one for water particles.
################################################################################

def read_scat_databases(dbname):

    #See if the database actually exists

    if not os.path.exists(dbname):
        print('ERROR: Unable to find the scattering database')
        return [], 1

    #Number of header lines (before the phase function angles)
    nheader = 5

    #Number of columns of data (not including the phase function)
    ncols = 13

    #Open and read the single scattering property database
    print('  Reading: ' + dbname)
    f = open(dbname, 'r')
    f.readline()
    f.readline()
    nlines = int(f.readline().split()[0])
    nphase = int(f.readline().split()[0])

    if nlines <= 0:
        print('ERROR: There were no datalines found in this database -- this should not occur')
        f.close()
        return [], 1

    if nphase <= 0:
        print('ERROR: The scattering phase function was not defined in this database')
        f.close()
        return [], 1
    f.close()

    pangle = np.genfromtxt(dbname ,skip_header = nheader, max_rows = 1)

    data = np.genfromtxt(dbname ,skip_header = nheader+2).T

    fields = (['wavelength [um]','wavenumber [cm-1]','effective radius',
              'extinction cross section', 'scattering cross section',
              'absorption cross section', 'single scatter albedo',
              'asymmetry parameter', 'Extinction efficiency',
              'Absorption efficiency', 'Scattering efficiency', 'Volume',
              'Projected area','Rest of the elements are phase function'])

    database = ({'dbname':dbname, 'ncols':ncols, 'nphase':nphase, 'pangle':pangle,
                'data':data, 'columns':fields})

    return database, 0

################################################################################
# This function reads in the standard atmosphere profiles
################################################################################

def read_stdatmos(filename, stdatmos, verbose):

    if not os.path.exists(filename):
        print('Error: Unable to find the IDL save file with the standard atmosphere information in it')
        return {'status':0}

    temp = scipy.io.readsav(filename, python_dict = True)
    idx = stdatmos-1
    if ((idx < 0) | (idx >= len(temp['name']))):
        print('Error: the standard atmosphere specified is out-of-range')
        return {'status':0}

    if verbose >= 1:
        print('  Using standard atmosphere: ' + temp['name'][idx].decode())

    return {'status':1, 'z':temp['z'][idx,:], 'p':temp['p'][idx,:], 't':temp['t'][idx,:], 'w':temp['w'][idx,:], 'pwv':temp['pwv'][idx], 'name':temp['name'][idx]}


################################################################################
#This routine reads in the controlling parameters from the MIXCRA VIP file.
################################################################################

def read_mixcra_vip_file(filename,globatt,verbose,dostop):

    vip = {}
    for key in full_mixcra_vip.keys():
        vip[key] = full_mixcra_vip[key]['value']

    if os.path.exists(filename):

        if verbose >= 1:
            print('  Reading the VIP file: ' + filename)

        try:
            # inputt = np.genfromtxt(filename, dtype=str,comments='#', usecols = (0,1,2))
            inputt = np.genfromtxt(filename, dtype=str, comments='#', delimiter='=', autostrip=True)
        except Exception as e:
            print(e)
            print('  There was an problem reading the VIP file. Check formatting.')
            return vip
    else:
        print('  The VIP file ' + filename + ' does not exist')
        return vip

    if len(inputt) == 0:
        print('  There were no valid lines found in the VIP file')
        return vip
    
    # Look for obsolete tags, and abort if they are found (forcing user to update their VIP file)
    obsolete_tags = ['AERI_LAT','AERI_LON','AERI_ALT','PSFC_MIN','PSFC_MAX']
    obsolete_idx  = np.zeros_like(obsolete_tags, dtype=int)
    vip_keys = np.array([k.upper() for k in inputt[:,0]])
    for i in range(len(obsolete_tags)):
        foo = np.where(obsolete_tags[i].upper() == vip_keys)[0]
        if(len(foo) > 0):
            obsolete_idx[i] = 1
    foo = np.where(obsolete_idx > 0)[0]
    if(len(foo) > 0):
        print('Error: there were obsolete tags in the VIP file:')
        for i in range(len(foo)):
            print('     '+obsolete_tags[foo[i]])
            return vip
    
    # Create a tracking array to make sure all keys in the user provided
    # vip file are used
    track = np.ones(inputt.shape[0])
    # Look for these tags
    nfound = 1
    for key in vip.keys():
        if key != 'success':
            nfound += 1
            if key == 'vip_filename':
                vip['vip_filename'] = filename
            else:
                foo = np.where(key == inputt[:,0])[0]
                if len(foo) > 1:
                    print('Error: There were multiple lines with the same key in VIP file: ' + key)
                    return vip

                elif len(foo) == 1:
                    track[foo] = 0
                    if verbose == 3:
                        print('Loading the key ' + key)
                    if key == 'spectral_bands':
                        bands = np.zeros((2, maxbands))-1
                        tmp = inputt[foo,1][0].split(',')

                        if len(tmp) >= maxbands:
                            print('Error: There were more spectral bands defined than maximum allowed (maxbands = ' + str(maxbands) + ')')
                            return vip

                        for j in range(len(tmp)):
                            feh = tmp[j].split('-')
                            if len(feh) != 2:
                                print('Error: Unable to properly decompose the spectral_bands key')
                                if dostop:
                                    wait = input('Stopping inside to debug this bad boy. Press enter to continue')
                                return vip
                            bands[0,j] = float(feh[0])
                            bands[1,j] = float(feh[1])
                        vip['spectral_bands'] = bands

                    elif key == 'sfc_emissivity':
                        sfce = np.zeros((2, maxbands))-1
                        tmp = inputt[foo,1][0].split(',')

                        if len(tmp) >= maxbands:
                            print('Error: There were more surface emissivity elements defined than maximum allowed (maxbands = ' + str(maxbands) + ')')
                            return vip

                        for j in range(len(tmp)):
                            feh = tmp[j].split('-')
                            if len(feh) != 2:
                                print('Error: Unable to properly decompose the sfc_emissivity key')
                                if dostop:
                                    wait = input('Stopping inside to debug this bad boy. Press enter to continue')
                                return vip
                            sfce[0,j] = float(feh[0])
                            sfce[1,j] = float(feh[1])
                        vip['sfc_emissivity'] = sfce

                    else:
                        vip[key] = type(vip[key])(inputt[foo,1][0])
                else:
                    if verbose == 3:
                        print('UNABLE to find the key ' + key)
                    nfound -= 1

    # Now look for any global attributes that might have been entered in the file

    matching = [s for s in inputt[:,0] if "globatt" in s]

    if verbose >= 2:
        print('    There were ' + str(len(matching)) + ' global attributes found')

    for i in range(len(matching)):
        foo = np.where(matching[i] == inputt[:,0])[0]
        track[foo] = 0
        globatt[matching[i][8:]] = inputt[foo,1][0]
    
    # Need to trap condition where spectral_bands was not set (and thus is the default string "None")
    # and then reset it to a standard set of bands used for IR sounding
    if(type(vip['spectral_bands']) == str):
        blo = [529.9, 558.5, 809.0, 815.3, 828.3, 842.8, 860.1, 872.2, 891.9, 898.2, 929.6, 959.9, 985.0, 1076.6, 1092.1, 1113.3, 1124.4, 1142.2, 1155.2]
        bhi = [531.5, 562.0, 812.9, 824.4, 834.6, 848.1, 864.0, 877.5, 895.8, 905.4, 939.7, 964.3, 998.0, 1084.8, 1098.8, 1116.6, 1132.6, 1148.0, 1163.4]
        vip['spectral_bands'] = np.array([blo,bhi])

    # Need to trap condition where sfc_emissivity was not set (and thus is the default string "None")
    # and then reset it to a standard emissivity spectrum
    if(type(vip['sfc_emissivity']) == str):
        bwn = [100.0,  500.0,  700.0, 1000.0, 1100.0, 1200.0, 1300.0, 2000.0, 2200.0, 2450.0, 2800.0, 3000.0]
        bem = [0.990,  0.990,  0.990,  0.980,  0.950,  0.950,  0.995,  0.990,  0.940,  0.920,  0.950,  0.990]
        vip['sfc_emissivity'] = np.array([bwn,bem])

    foo = np.nonzero(track)[0]
    if len(foo) > 0:
        print('  There were undefined entries in the VIP file:')
        for i in range(len(foo)):
            print('    ' + inputt[foo,0][i])
        return vip
    else:
        vip['success'] = 1

    if dostop:
        wait = input('Stopping inside to debug this bad boy. Press enter to continue')

    return vip
