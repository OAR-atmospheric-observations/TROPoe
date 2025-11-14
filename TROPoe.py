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

__version__ = '0.19.6'

import os
import sys
import numpy as np
import shutil
import scipy.io
import copy
import warnings
from netCDF4 import Dataset
from datetime import datetime, timezone
from glob import glob
from time import gmtime, strftime
from subprocess import Popen, PIPE
from argparse import ArgumentParser

import Other_functions
import VIP_Databases_functions
import Calcs_Conversions
import Data_reads
import Jacobian_Functions
import Output_Functions
import plot_tropoe

# Check to see if we are just writing out a blank vip
if '--vip' in sys.argv:
    # Write out a default vip file
    print("Writing default vip file to console")

    if '--experimental' in sys.argv:
        Output_Functions.write_example_vip_file(console=True, experimental=True)
    else:
        Output_Functions.write_example_vip_file(console=True)

    sys.exit()

# Create parser for command line arguments
parser = ArgumentParser()

parser.add_argument("date", type=int, help="Date to run the code [YYYYMMDD]")
parser.add_argument("vip_filename", help="Name if the VIP file (string)")
parser.add_argument("prior_filename", help="Name of the prior input dataset (string)")
parser.add_argument("--shour", type=float, help="Start hour (decimal, 0-24)")
parser.add_argument("--ehour", type=float, help="End hour (decimal, 0-24) [If ehour<0 process up to last IRS sample]")
parser.add_argument("--verbose",type=int, choices=[0,1,2,3], help="The verbosity of the output (0-very quiet, 3-noisy)")
parser.add_argument("--doplot", action="store_true", help="If set, then create real-time display of retrievals")
parser.add_argument("--debug", action="store_true", help="Set this to turn on the debug mode")
parser.add_argument("--dostop",action="store_true", help="Set this to stop at the end before exiting")

args = parser.parse_args()

date = args.date
vip_filename = args.vip_filename
prior_filename = args.prior_filename
shour = args.shour
ehour = args.ehour
verbose = args.verbose
doplot = args.doplot
debug = args.debug
dostop = args.dostop

#Check to see if any of these are set; if not, fall back to default values

if shour is None:
    shour = 0.
if ehour is None:
    ehour = -1.
if verbose is None:
    verbose = 1
if debug is None:
    debug = False
if dostop is None:
    dostop = False

# If doplot is True, need to import matplotlib functions
if doplot:
    import matplotlib.pyplot as plt

# Initialize
success = True

# We need the background shell to be the C-shell, as we will be spawning out
# a variety of commands that make this assumption. So we will do a
# quick check to find out if the C-shell exists on this system, and if so,
# set the SHELL to this.

if verbose == 3:
    print(' ')
    print(('The current shell is', os.getenv('SHELL')))
else:
    warnings.filterwarnings("ignore", category=UserWarning)

process = Popen('which csh', stdout = PIPE, stderr = PIPE, shell=True)
stdout, stderr = process.communicate()

if stdout.decode() == '':
    print('Error: Unable to find the C-shell command on this system')
    print(('>>> TROPoe retrieval on ' + str(date) + ' FAILED and ABORTED <<<'))
    sys.exit()
else:
    SHELL = stdout[:-1].decode()

if verbose == 3:
    print(('The shell for all commands is', SHELL))

# Get the version of the TROPoe package that was installed within the container
tropoe_version = Data_reads.get_tropoe_version()

#Capture the version of this file
globatt = {'algorithm_code': 'TROPoe Retrieval Code (formerly AERIoe)',
           'algorithm_authors': 'Dave Turner, NOAA Global Systems Laboratory (dave.turner@noaa.gov), ' +
                                'Josh Gebauer, NOAA National Severe Storms Laboratory / CIWRO (joshua.gebauer@noaa.gov), ' +
                                'Tyler Bell, NOAA National Severe Storms Laboratory / CIWRO (tyler.bell@noaa.gov)',
           'algorithm_comment1': 'TROPoe is a physical-iterative algorithm that retrieves thermodynamic profiles from ' +
                                 'a wide range of ground-based remote sensors.  It was primarily designed to use either ' +
                                 'infrared spectrometers or microwave radiometers as the primary instrument, and include ' +
                                 'observations from other sources to improve the quality of the retrieved profiles',
           'algorithm_comment2': 'Original code was written in IDL and is described by the "AERIoe" papers listed below',
           'algorithm_comment3': 'Code was ported to python, and packaged into a container with the ' +
                                 'needed radiative transfer models and other required inputs',
           'algorithm_disclaimer': 'TROPoe was developed by NOAA and is provided on an as-is basis, with no warranty',
           'algorithm_code_version': __version__,
           'algorithm_package_version': tropoe_version,
           'algorithm_reference1': 'DD Turner and U Loehnert, 2014: Information Content and ' +
                    'Uncertanties in Thermodynamic Profiles and Liquid Cloud Properties ' +
                    'Retrieved from the Ground-Based Atmospheric Emitted Radiance ' +
                    'Interferometer (AERI), J Appl Met Clim, vol 53, pp 752-771, ' +
                    'doi:10.1175/JAMC-D-13-0126.1',
           'algorithm_reference2': 'DD Turner and WG Blumberg, 2019: Improvements to the AERIoe ' +
                    'thermodynamic profile retrieval algorithm. IEEE Selected Topics ' +
                    'Appl. Earth Obs. Remote Sens., 12, 1339-1354, doi:10.1109/JSTARS.2018.2874968',
           'algorithm_reference3': 'DD Turner and U Loehnert, 2021: Ground-based temperature and humidity profiling: ' +
                    'Combining active and passive remote sensors, Atmos. Meas. Tech., vol 14, pp 3033-3048, ' +
                    'doi:10.5194/amt-14-3033-2021',
           'forward_model_reference1': 'The forward radiative transfer models are from Atmospheric and Environmental ' +
                    'Research Inc (AER); an overview is provided by Clough et al., 2005: Atmospheric radiative transfer ' +
                    'modeling: A summary of the AER codes, JQSRT, vol 91, pp 233-244, doi:10.1016/j.jqsrt.2004.05.058',
           'forward_model_reference2': 'The infrared model is LBLRTM; papers describing this model include ' +
                    'doi:10.1029/2018JD029508, doi:10.1175/amsmonographs-d-15-0041.1, and doi:10.1098/rsta.2011.0295',
           'forward_model_reference3': 'The microwave model is MonoRTM; papers describing this model include ' +
                    'doi:10.1109/TGRS.2010.2091416 and doi:10.1109/TGRS.2008.2002435',
           'datafile_created_on_date': strftime("%Y-%m-%d %H:%M:%S", gmtime()),
           'datafile_created_on_machine': os.uname()[-1]}


# Start the retrieval
print(' ')
print('-------------------------------------------------------------------------')
print('---- TROPoe is a thermodynamic retrieval algorithm developed at NOAA ----')
print('---- Contacts are dave.turner, joshua.gebauer, tyler.bell (@noaa.gov) ---')
print('------- The code is provided on an "as-is" basis, with no warranty ------')
print(' ')
print(('>>> Starting TROPoe retrieval for ' + str(date) + ' (from ' + str(shour) + ' to ' + str(ehour) + ' UTC) <<<'))
print('  TROPoe version: '+tropoe_version)

#Find the VIP file and read it

vip = VIP_Databases_functions.read_vip_file(vip_filename, globatt = globatt, debug = debug, verbose = verbose, dostop = dostop)

if vip['success'] != 1:
    print(('>>> TROPoe retrieval on ' + str(date) + ' FAILED and ABORTED <<<'))
    print('---------------------------------------------------------------------')
    print(' ')
    sys.exit()

if vip['plot_output'] == 2:
    print("plot_output = 2, so plotting previously retrieved output and aborting")

    files = glob(os.path.join(vip['output_path'], f"{vip['output_rootname']}*{date}*.nc"))

    if len(files) == 0:
        print("     No files found to plot. Aborting...")
        sys.exit()

    for fn in files:
        print(f"    Plotting {fn}")
        plot_tropoe.doplot(fn, vip['plot_xlim'], vip['plot_ylim'], vip['plot_temp_lim'], vip['plot_wvmr_lim'],
                           vip['plot_tuncert_lim'], vip['plot_wvuncert_lim'], vip['plot_theta_lim'], vip['plot_rh_lim'],
                           vip['plot_thetae_lim'], vip['plot_dewpt_lim'], vip['plot_lwp_cbh_threshold'],
                           vip['plot_tres_min_gap'], vip['plot_comment'], vip['plot_rootname'], vip['plot_path'])

    sys.exit()

uniquekey = vip['tag'] + '.' + str(np.random.randint(0,999999))

if debug:
    print('Saving the VIP and globatt structure into "vip.npy" -- for debugging')
    np.save('vip.npy', vip)

# Make sure that Paul van Delst's script "lblrun" is in the $LBL_HOME/bin
# directory, as this is used often. The assumption is that if it is there,
# that the rest of the LBLRTM distribution is set up to use it properly.

if not os.path.exists(vip['lbl_home'] + '/bin/lblrun'):
    print('Error: Unable to find the script "lblrun" in the "lbl_home"/bin directory')
    print('This is a critical component of the LBLRTM configuration - aborting')
    print(('>>> TROPoe retrieval on ' + str(date) + ' FAILED and ABORTED <<<'))
    print('---------------------------------------------------------------------')
    print(' ')
    sys.exit()

# Check if the prior data exists
if not os.path.exists(prior_filename):
    print(('Error: Unable to find the prior data file: ' + prior_filename))
    print(('>>> TROPoe retrieval on ' + str(date) + ' FAILED and ABORTED <<<'))
    print('---------------------------------------------------------------------')
    print(' ')
    sys.exit()

print(('  Using the prior file: ' + prior_filename))


# Make sure that the output directory exists
if not os.path.exists(vip['output_path']):
    print('Error: The output directory does not exist')
    print(('>>> TROPoe retrieval on ' + str(date) + ' FAILED and ABORTED <<<'))
    print('---------------------------------------------------------------------')
    print(' ')
    sys.exit()

# Look at the name of the LBLtmpDir; if it starts with a "$"
# then assume that first part is an environment variable and
# decode the path accordingly

if vip['lbl_temp_dir'][0] == '$':
    envpath = vip['lbl_temp_dir'].split('/')
    tmpdir = os.getenv(envpath[0])
    if not tmpdir:
        print('Error: The LBLRTM temporary directory is being set to an environment variable that does not exist')
        print(('>>> TROPoe retrieval on ' + str(date) + ' FAILED and ABORTED <<<'))
        print('---------------------------------------------------------------------')
        print(' ')
        sys.exit()
    for i in range(1,len(envpath)):
        tmpdir = tmpdir + '/' + envpath[i]
    vip['lbl_temp_dir'] = tmpdir

# Get the IRS band inflation information into a structure
irs_band_noise_inflation = Other_functions.decompose_irs_band_noise_inflation(vip, verbose=verbose)

# Create the temporary working directory
lbltmpdir = vip['lbl_temp_dir'] + '/' + uniquekey
print(('  Setting the temporary directory for RT model runs to: ' + lbltmpdir))

#Address this in Python 3 version
try:
    os.makedirs(lbltmpdir)
except:
    print('Error making the temporary directory')
    print(('>>> TROPoe retrieval on ' + str(date) + ' FAILED and ABORTED <<<'))
    print('---------------------------------------------------------------------')
    print(' ')

    
success = 0

# House keeping stuff
starttime = datetime.now()
endtime = starttime
print(' ')

# Read in the SSP databases
sspl, flag = VIP_Databases_functions.read_scat_databases(vip['lcloud_ssp'])
if flag == 1:
    print('Error: Problem reading SSP file for liquid cloud properties')
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()


sspi, flag = VIP_Databases_functions.read_scat_databases(vip['icloud_ssp'])
if flag == 1:
    print('Error: Problem reading SSP file for ice cloud properties')
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()

# Determine the minimum and maximum Reff values in these, but I need to have the
# minimum value be just a touch larger than the actual minimum value in the database
# hence the 1.01 multipliers

minLReff = np.nanmin(sspl['data'][2,:])*1.01
maxLReff = np.nanmax(sspl['data'][2,:])
miniReff = np.nanmin(sspi['data'][2,:])*1.01
maxiReff = np.nanmax(sspi['data'][2,:])

# Perform some more baseline checking of the keywords
if ((vip['cvgmult'] < 0.1) | (vip['cvgmult'] > 1)):
    print('Error: The error criteria vip.cvgmult is too small or too large (must be between 0.1 and 1.0)')
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()

# Perform some baseline checking of parameters in the VIP structure
# to make sure that the values are within the valid range
if VIP_Databases_functions.check_vip(vip) == 1:
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()

# Set switches associated with retrieving various variables and computing Jacobians
# This is a mess. Original was terribly inefficient. Maybe look to handle this better in future


if ((vip['retrieve_co2'] >= 1) & (vip['retrieve_co2'] <= 2)):
    doco2 = vip['retrieve_co2']
else:
    doco2 = 0
if ((vip['retrieve_ch4'] >= 1) & (vip['retrieve_ch4'] <= 2)):
    doch4 = vip['retrieve_ch4']
else:
    doch4 = 0
if ((vip['retrieve_n2o'] >= 1) & (vip['retrieve_n2o'] <= 2)):
    don2o = vip['retrieve_n2o']
else:
    don2o = 0
if vip['retrieve_lcloud'] >= 1:
    dolcloud = 1
    fixlcloud = 0           # Jacobian flag (for some reason 0 is on)
else:
    dolcloud = 0
    fixlcloud = 1           # Jacobian flag (for some reason 1 is off)
if vip['retrieve_icloud'] >= 1:
    doicloud = 1
    fixicloud = 0          # Jacobian flag (for some reason 0 is on)
else:
    doicloud = 0
    fixicloud = 1          # Jacobian flag (for some reason 1 is off)
if vip['retrieve_temp'] >= 1:
    dotemp = 1
    fixtemp = 0
else:
    dotemp = 0
    fixtemp = 1
if vip['retrieve_wvmr'] >= 1:
    dowvmr = 1
    fixwvmr = 0
else:
    dowvmr = 0
    fixwvmr = 1

modeflag = [dotemp, dowvmr, dolcloud, doicloud, doco2, doch4, don2o]

# Select the LBLRTM version to use
print(' ')
print(('  Working with the LBLRTM version ' + vip['lbl_version']))
print(('    in the directory ' + vip['lbl_home']))
print(('    and the TAPE3 file ' + vip['lbl_tape3']))
print(' ')

# Quick check: make sure the LBLRTM path is properly set
if not os.path.exists(vip['lbl_home'] + '/bin/lblrtm'):
    print('Error: lblhome is not properly set')
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()

# Make sure that the specified TAPE3 file exists
if not os.path.exists(vip['lbl_home'] + '/hitran/' + vip['lbl_tape3']):
    print('Error: unable to find the specified TAPE3 file in the LBL_HOME hitran directory')
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()

# Make sure that path to the MonoRTM files are set properly. We will only
# stop the code if MWR data is used and the files don't exist
if not os.path.exists(vip['monortm_wrapper']) and ((vip['mwr_type'] > 0) or (vip['mwrscan_type'] > 0)):
    print('Error: unable to find the the specified MonoRTM wrapper')
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()
    
if not os.path.exists(vip['monortm_exec']) and ((vip['mwr_type'] > 0) or (vip['mwrscan_type'] > 0)):
    print('Error: unable to find the the specified MonoRTM executable')
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()

if not os.path.exists(vip['monortm_spec']) and ((vip['mwr_type'] > 0) or (vip['mwrscan_type'] > 0)):
    print('Error: unable to find the the specified MonoRTM spectral line file')
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()
    
# Define some paths and constants
lbldir = lbltmpdir + '/lblout'       # Name of the lbl output directory
lbltp5 = lbltmpdir + '/lbltp5'       # Name of the tape5 file
lbltp3 = vip['lbl_tape3']           # Name of the tape3 file
lbllog = lbltmpdir + '/lbllog'       # Name of the lbl log file
lbltmp = lbltmpdir + '/lbltmp'       # Temporary directory for LBLRUN (will be LBL_RUN_ROOT)
monortm_config = 'monortm_config.txt'
monortm_zfreq = 'monortm_zfreqs.txt'    # For MWR-zenith calculations
monortm_sfreq = 'monortm_sfreqs.txt'    # For MWR-scan calculations
monortm_tfile = 'monortm_sonde.cdf'

# Make two commands: one for MWR-zenith and one for MWR-scan
monortm_zexec = ('cd ' + lbltmpdir + ' ; setenv monortm_config ' + monortm_config +
                ' ; setenv monortm_freqs ' + monortm_zfreq + ' ; ' + vip['monortm_wrapper'])

monortm_sexec = ('cd ' + lbltmpdir + ' ; setenv monortm_config ' + monortm_config +
                ' ; setenv monortm_freqs ' + monortm_sfreq + ' ; ' + vip['monortm_wrapper'])

# This should be included in the VIP file. Right now it is always set.
create_monortm_config = 1           # Set this flag to create a custom config file for MonoRTM
create_monortm_zfreq = 1            # Set this flag to create a custom freq-zenith file for MonoRTM
create_monortm_sfreq = 1            # Set this flag to create a custom freq-scan file for MonoRTM

#Load the standard atmosphere

stdatmos = VIP_Databases_functions.read_stdatmos(vip['path_std_atmos'], vip['lbl_std_atmos'], verbose)
if stdatmos['status'] == 0:
    print('Error: Unable to find/read the standard atmosphere file')
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()

# Define the spectral regions(s) to use in the retrieval
bands = vip['spectral_bands']
foo = np.where(bands[0,:] >= 0)[0]
if len(foo) <= 0:
    print('Error: the spectral bands do not have any properly defined values')
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()
bands = bands[:,foo]

# Make sure that the MWR flags make sense
if vip['mwr_type'] > 0:
    if vip['mwr_tb_replicate'] <= 0:
        print('Error: when using MWR data the "replicate" flag must be >= 1')
        VIP_Databases_functions.abort(lbltmpdir,date)
        sys.exit()

# Echo to the user the type of retrieval being performed
print(' ')
tmp = '  Retrieving: '
if dotemp == 1:
    tmp = tmp + 'T '
if dowvmr == 1:
    tmp = tmp + 'Q '
if dolcloud == 1:
    tmp = tmp + 'Liq_Cloud '
if doicloud == 1:
    tmp = tmp + 'Ice_Cloud '
if doco2 >= 1:
    tmp = tmp + 'CO2 '
if doch4 >= 1:
    tmp = tmp + 'CH4 '
if don2o >= 1:
    tmp = tmp + 'N2O '
print(tmp)
print(' ')

# Read in the a priori covariance matrix of T/Q for this study
nsonde_prior = -1
try:
    fid = Dataset(prior_filename,'r')
except:
    print('Error: Unable to open the XaSa file')
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()
z  = fid.variables['height'][:]
Pa = fid.variables['mean_pressure'][:]
Xa = fid.variables['mean_prior'][:]
Sa = fid.variables['covariance_prior'][:]
try:
    nsonde_prior = int(fid.Nsonde.split()[0])
except AttributeError:
    nsonde_prior = int(fid.Nprofiles.split()[0])

comment_prior = str(fid.Comment)
minT = float(fid.QC_limits_T.split()[5])
maxT = float(fid.QC_limits_T.split()[7])
if verbose == 3:
    print(('QC limits for T are ' + str(minT) + ' and ' + str(maxT)))
minQ = float(fid.QC_limits_q.split()[7])
maxQ = float( fid.QC_limits_q.split()[9])
if verbose == 3:
    print(('QC limits for Q are ' + str(minQ) + ' and ' + str(maxQ)))
fid.close()

# Interpolate the input prior to the new specified height grid
zz = vip['zgrid'].split(',')
zz = np.array(zz).astype(float)
if Other_functions.test_monotonic(zz) == False:
    print('Error: The input vip.zgrid is not strictly monotonically ascending -- aborting')
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()
if zz[0] != 0:
    print('Error: The first level of the input vip.zgrid is not zero (a requirement) -- aborting')
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()
if np.max(zz) >= np.max(z):
    print(f'Error: The maximum height in the input vip.grid must be less than {np.max(z):.3f} km')
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()
if(verbose >= 2):
    print('  Adjusting prior to the defined vertical grid')
newXa,newSa,newPa = Other_functions.interpolate_prior_covariance(z,Xa,Sa,Pa,zz,verbose=verbose)
        # Replace the prior values from the input file with the values on the input zgrid
z  = zz
Pa = newPa
Xa = newXa
Sa = newSa

if verbose >= 1:
    print(('  Retrieved profiles will have ' + str(len(z)) + ' levels (from VIP.zgrid)'))
if verbose >= 2:
    print(('    There were ' + str(nsonde_prior) + ' radiosondes used in the calculation of the prior'))

# Inflate the lowest levels of the prior covariance matrix, if desired
Sa, status = Other_functions.inflate_prior_covariance(Sa, z, vip['prior_t_ival'], vip['prior_t_iht'],
             vip['prior_q_ival'], vip['prior_q_iht'], vip['prior_tq_cov_val'],
             vip['prior_chimney_ht'], verbose)
if status == 0:
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()

# Apply the Tikhonov Regularization to improve the condition of the prior covariance
if ((vip['prior_tikhonov_factor'] > 0) & (verbose >= 1)):
    print(f"  Applying regularization factor to Tq prior covariance matrix (alpha={vip['prior_tikhonov_factor']:.3e})")
Sa, status = Other_functions.condition_matrix(Sa, vip['prior_tikhonov_factor'])
if status != 'success':
    print(status)
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()

# Read in the data
fail, irs, mwr, mwrscan = Data_reads.read_all_data(date, z, vip['tres'], dostop, verbose, vip['avg_instant'],
    vip['irsch1_path'], vip['irs_pca_nf'], vip['irs_fv'], vip['irs_fa'],
    vip['irssum_path'], vip['irseng_path'], vip['irs_type'], vip['irs_calib_pres'],
    vip['irs_smooth_noise'], vip['irs_use_missingDataFlag'],
    vip['irs_min_675_bt'], vip['irs_max_675_bt'], vip['irs_spec_cal_factor'], vip['irs_noise_inflation'], 
    vip['mwr_path'], vip['mwr_rootname'], vip['mwr_type'], vip['mwr_elev_field'], vip['mwr_n_tb_fields'],
    vip['mwr_tb_replicate'], vip['mwr_tb_field_names'], vip['mwr_tb_freqs'], vip['mwr_tb_noise'],
    vip['mwr_tb_bias'], vip['mwr_tb_field1_tbmax'], vip['cbh_path'], vip['cbh_type'], vip['cbh_window_in'],
    vip['cbh_window_out'], vip['cbh_default_ht'], vip['irs_hatch_switch'],
    vip['irs_use_missingDataFlag'], vip)

if fail == 1:
    print('Error reading in data: aborting')
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()

# Read in any external sources of WV and temperature profiles
ext_prof = Data_reads.read_external_profile_data(date, z, irs['secs'], vip['tres'], vip['avg_instant'],
              vip['ext_wv_prof_type'], vip['ext_wv_prof_path'], vip['ext_wv_noise_mult_hts'],
              vip['ext_wv_noise_mult_val'], vip['ext_wv_prof_minht'], vip['ext_wv_prof_maxht'],
              vip['ext_wv_time_delta'], vip['ext_temp_prof_type'], vip['ext_temp_prof_path'],
              vip['ext_temp_noise_adder_hts'], vip['ext_temp_noise_adder_val'],
              vip['ext_temp_prof_minht'], vip['ext_temp_prof_maxht'], vip['ext_temp_time_delta'],
              vip['ext_temp_ht_offset'], vip['ext_wv_ht_offset'], 
              dostop, verbose)

if ext_prof['success'] != 1:
    print('Error: there is some problem in the external profile WV or temperature specification')
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()

if len(ext_prof['attrs'].keys()) > 0:
    for key in ext_prof['attrs'].keys():
        globatt[key] = ext_prof['attrs'][key]

# Read in any model sources of WV and temperature profiles
mod_prof = Data_reads.read_external_profile_data(date, z, irs['secs'], vip['tres'], vip['avg_instant'],
              vip['mod_wv_prof_type'], vip['mod_wv_prof_path'], vip['mod_wv_noise_mult_hts'],
              vip['mod_wv_noise_mult_val'], vip['mod_wv_prof_minht'], vip['mod_wv_prof_maxht'],
              vip['mod_wv_time_delta'], vip['mod_temp_prof_type'], vip['mod_temp_prof_path'],
              vip['mod_temp_noise_adder_hts'], vip['mod_temp_noise_adder_val'],
              vip['mod_temp_prof_minht'], vip['mod_temp_prof_maxht'], vip['mod_temp_time_delta'],
              vip['mod_temp_ht_offset'], vip['mod_wv_ht_offset'], 
              dostop, verbose)

if mod_prof['success'] != 1:
    print('Error: there is some problem in the model profile WV or temperature specification')
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()

# Read in any external time series data that would be used
ext_tseries = Data_reads.read_external_timeseries(date, irs['secs'], vip, dostop, verbose)

if ext_tseries['success'] != 1:
    print('Error: there is some problem in the external time series data specification')
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()

# If the surface pressure field is all negative, then use the default station_pres from the VIP file
foo = np.where((irs['atmos_pres'][:] < 200) | (irs['atmos_pres'][:] > 1200))[0]
if(len(foo) > 0):
    print('    Warning: changing the surface pressure of some samples to the default "station_pres" in the VIP file')
    if(vip['station_pres'] < 0):
        print('    Error: the keyword "station_pres" must be set to a positive value (in mb) in the VIP file')
        sys.exit()
    irs['atmos_pres'][foo] = vip['station_pres']

# If ehour < 0, then set it to the time of the last IRS sample. (This was needed
# for those cases when the IRS did not automatically reboot at 0 Z.)
if ehour < 0:
    ehour = np.nanmax(irs['hour'])
    if verbose >= 2:
        print(('Resetting the processing end hour to ' + str(ehour) + ' UTC'))

# Capture the lat/lon/alt data in a structure.  Use the VIP supplied data
location = {'lat':vip['station_lat'], 'lon':vip['station_lon'], 'alt':int(vip['station_alt'])}

# Very simple check to make sure station altitude makes sense [m MSL]
if(location['alt'] <= 0):
    print('    Error: the station altitude must be > 0 [m MSL]')
    sys.exit()

# Do I use a hardcoded value for CO2, or use my simple model to predict
# the concentration? Unit is ppm
if vip['prior_co2_mn'][0] < 0:
   vip['prior_co2_mn'][0] = Other_functions.predict_co2_concentration(irs['yy'][0], irs['mm'][0], irs['dd'][0])

# Quick test to make sure that the trace gas models make sense. If not, abort
tmpco2 = Other_functions.trace_gas_prof(vip['retrieve_co2'], z, vip['prior_co2_mn'])
nfooco2 = len(np.where(tmpco2 < 0)[0])
tmpch4 = Other_functions.trace_gas_prof(vip['retrieve_ch4'], z, vip['prior_ch4_mn'])
nfooch4 = len(np.where(tmpch4 < 0)[0])
tmpn2o = Other_functions.trace_gas_prof(vip['retrieve_n2o'], z, vip['prior_n2o_mn'])
nfoon2o = len(np.where(tmpn2o < 0)[0])
if ((nfooco2 > 0) | (nfooch4 > 0) | (nfoon2o > 0)):
    print('Error: The CO2, CH4, and/or N2O parameters are incorrect - aborting')
    VIP_Databases_functions.abort(lbltmpdir,date)
    sys.exit()

# Recenter the prior if desired
recenter_input_value = -1
if vip['recenter_prior'] > 0:
    # If the vip.recenter_input is set, this is the override value
    if vip['recenter_input'] > 0:   # Rescale based on the value inputted into the vip
        recenter_input_value = vip['recenter_input']
    elif ((vip['recenter_prior'] == 1) | (vip['recenter_prior'] == 3) | (vip['recenter_prior'] == 5)):
        if ((vip['ext_sfc_wv_type'] > 0) & (ext_tseries['nQsfc'] > 0) & (vip['recenter_prior'] != 5)):  # Rescale based on the ext_sfc data
            foo = np.where(ext_tseries['wv'][0,:] > 0)[0]  # Need to take out any -999s
            if len(foo) > 0:
                recenter_input_value = np.mean(ext_tseries['wv'][0,foo])
        else:
            if vip['recenter_prior'] != 5:
                print('    Warning: Trying to recenter the prior using the surface met, but there are no valid WV surface obs')
            foo = np.where(irs['nearSfcTb'] > -990)[0]
            if len(foo) > 0:
                meanNearSfcTemp = np.nanmedian(irs['nearSfcTb'][foo])
                print('            Using the IRS radiometric near-surface air temperature to attempt to recenter')
            else:
                foo = np.where(mwr['nearSfcTb'] > -990)[0]
                if len(foo) > 0:
                    meanNearSfcTemp = np.nanmedian(mwr['nearSfcTb'][foo])
                    print('            So using the MWR radiometric near-surface air temperature to attempt to recenter')
                else:
                    meanNearSfcTemp = -999.
            if(meanNearSfcTemp > -990):
                priorSfcT = Xa[0]
                priorSfcQ = Xa[len(z)]
                priorSfcP = Pa[0]
                priorSfcRH = (Calcs_Conversions.w2rh(priorSfcQ, priorSfcP, priorSfcT))[0]
                estSfcQ    = (Calcs_Conversions.rh2w(meanNearSfcTemp, priorSfcRH, priorSfcP))[0]
                recenter_input_value = estSfcQ

    elif ((vip['recenter_prior'] == 2) | (vip['recenter_prior'] == 4)):
        print('Warning: Trying to recenter the prior using an external PWV obs, but one does not exist yet -- aborting')
        VIP_Databases_functions.abort(lbltmpdir, date)
        sys.exit()
    else:
        print('Error: the flag set in vip.recenter_prior is illegal; it must be one of {0,1,2,3,4} -- aborting')
        VIP_Databases_functions.abort(lbltmpdir, date)
        sys.exit()

    # Determine which method to scale the temperature
    if ((vip['recenter_prior'] == 1) | (vip['recenter_prior'] == 3) | (vip['recenter_prior'] == 5)):
        changeTmethod = 0
        if ((vip['recenter_prior'] == 1) | (vip['recenter_prior'] == 5)):
            changeTmethod = 1
        elif vip['recenter_prior'] == 3:
            changeTmethod = 2

    # Quick check to make sure the recenter value is ok before trying to recenter the prior
    if recenter_input_value > 0:

        # Recenter the prior, using the inputs determined above
        successflag, newXa, newSa, comments = Data_reads.recenter_prior(z, Pa, Xa, Sa, 
                        recenter_input_value, vip, sfc_or_pwv=1, changeTmethod=changeTmethod)

        # Now replace the variables, if successful
        #   and update the global attributes to note that prior recentering was performed
        if successflag == 0:
            print('Error: Unable to recenter the prior -- aborting')
            VIP_Databases_functions.abort(lbltmpdir, date)
            sys.exit()
        elif successflag == 1:
            Xa = newXa
            Sa = newSa
            globatt.update(comments)

# Splice these trace gases and clouds into the Xa and Sa matrices.
# I am assuming no correlations between the TGs and clouds and the T/Q profiles
# The state vector is going to be defined to be:
# X = [T(z), q(z), LWP, ReL, tauI, ReI, CO2(3), CH4(3), N2O(3)
minsd = 1e-5        # Smallest value that doesn't result in problems in the matrix inversions below
diag = np.diag([np.max([vip['prior_lwp_sd'],minsd])**2,           # LWP : index 0
               np.max([vip['prior_lReff_sd'],minsd])**2,         # liquid Reff : index 1
               np.max([vip['prior_itau_sd'],minsd])**2,          # ice optical depth : index 2
               np.max([vip['prior_iReff_sd'],minsd])**2,         # ice Reff : index 3
               np.max([vip['prior_co2_sd'][0],minsd])**2,        #
               np.max([vip['prior_co2_sd'][1],minsd])**2,        # CO2 : indices 4,5,6
               np.max([vip['prior_co2_sd'][2],minsd])**2,        #
               np.max([vip['prior_ch4_sd'][0],minsd])**2,        #
               np.max([vip['prior_ch4_sd'][1],minsd])**2,        # CH4 : indices 7,8,9
               np.max([vip['prior_ch4_sd'][2],minsd])**2,        #
               np.max([vip['prior_n2o_sd'][0],minsd])**2,        #
               np.max([vip['prior_n2o_sd'][1],minsd])**2,        # N2O : indices 10,11,12
               np.max([vip['prior_n2o_sd'][2],minsd])**2])       #

zero = np.zeros((len(diag[0,:]),len(Xa)))
Sa = np.append(np.append(Sa,zero,axis=0),np.append(zero.T,diag,axis=0),axis=1)
Xa = np.append(Xa, [vip['prior_lwp_mn'], vip['prior_lReff_mn'],
               vip['prior_itau_mn'], vip['prior_iReff_mn'],
               vip['prior_co2_mn'][0], vip['prior_co2_mn'][1], vip['prior_co2_mn'][2],
               vip['prior_ch4_mn'][0], vip['prior_ch4_mn'][1], vip['prior_ch4_mn'][2],
               vip['prior_n2o_mn'][0], vip['prior_n2o_mn'][1], vip['prior_n2o_mn'][2]])
sig_Xa = np.sqrt(np.diag(Sa))

# Put all of the prior information into a structure for later
prior = {'comment':comment_prior, 'filename':prior_filename, 'nsonde':nsonde_prior,
         'Xa':np.copy(Xa), 'Sa':np.copy(Sa)}

# Make a directory for LBL_RUN_ROOT
if os.path.exists(lbltmp):
    shutil.rmtree(lbltmp)

os.mkdir(lbltmp)

# I want the code to try at least one retrieval, so I may need to
# modify the end time accordingly
foo = np.where((shour <= irs['hour']) & (irs['hour'] <= ehour))[0]
if len(foo) <= 0:
    foo = np.where(shour <= irs['hour'])[0]
    if len(foo) <= 0:
        print('No samples were found after this start time. Quitting')
        VIP_Databases_functions.abort(lbltmpdir,date)
        sys.exit()
    print('  Resetting the end hour to process at least 1 sample')
    if len(foo) < 2:
        ehour = irs['hour'][foo] + 1/3600.     # Added 1 second to this IRS sample to make sure to get it
    else:
        ehour = irs['hour'][foo[0]] + 1/3600.

# If neither liquid or ice clouds are enabled, then indicate that
# all retrievals are done as clear sky
if ((vip['retrieve_lcloud'] == 0) & (vip['retrieve_icloud'] == 0)):
    if verbose >= 2:
        print('All cloud retrievals disabled -- assuming clear sky')
    Xa[2*len(z)] = 0  # Zero LWP
    Xa[2*len(z)+2] = 0 # Zero ice optical depth
    irs['cbhflag'][:] = 0        # Reset all flags to clear sky
    # Note that I am leaving the CBH values untouched...

# Now loop over the observations and perform the retrievals
xret = []                  #Initialize. Will overwrite this if a succesful ret
already_saved = 0          # Flag to say if saved already or not...
fsample = 0                # Counter for number of spectra processed
precompute_prior_jacobian = {'status':0}    # This will allow us to store the forward calculations from the prior, makes code faster
cbh_string = ['Clear Sky', 'Inner Window', 'Outer Window', 'Default CBH']

# Quick check to make sure that the spectral bands being selected are actually
# valid for this interferometer (this ensures spectral range matches calculation below)
if(vip['irs_type'] >= 1):
    minv = np.min(irs['wnum'])
    maxv = np.max(irs['wnum'])
    foo = np.where(bands < minv)
    if(len(foo) > 0):
        bands[foo] = minv+0.1
    foo = np.where(bands > maxv)
    if(len(foo) > 0):
        bands[foo] = maxv-0.1

# Build the name of the output file
foo = np.where(irs['hour'] >= shour)[0]
if(len(foo) == 0):
    print('Error: there are no samples after the desired shour value -- aborting')
    sys.exit()

# If clobber == 2, then we will try to append. But this requires that
# I perform a check to make sure that we are appending to a file that was
# created by a version of the code that makes sense. I only need to make this
# test once, hence this flag
if vip['output_clobber'] == 2 or vip['output_clobber'] == 0:
    check_clobber = 1      
else:
    check_clobber = 0

# Quick check to make sure that the stdout messages make sense
if(vip['irs_type'] <= -1):
    itype = 'MWR'
else:
    itype = 'IRS'

# Define the structure for the "TROPoe_in" -- which will use the logic from Adler et al. AMT 2024
# It is entirely empty right now, but the space needs to be allocated
missing  = np.zeros(len(z))-999.
missing3 = np.zeros(3)-999.
in_tropoe = {'secs':0, 'height':z, 'pblh':-1., 'lwp':-999., 
             'temperature':missing, 'waterVapor':missing, 'sigma_temperature':np.abs(missing), 'sigma_waterVapor':np.abs(missing), 
             'co2':np.abs(missing3), 'sigma_co2':np.abs(missing3)}
if((vip['add_tropoe_T_input_flag'] == 1) | (vip['add_tropoe_q_input_flag'] == 1)):
    print('  The option to use previous good TROPoe profiles as input is turned on')
if((vip['add_tropoe_co2_input_flag'] == 1)):
    print('  The option to use previous good TROPoe co2 retrieval as input is turned on')

# This defines the extra vertical layers that will be added to both
# the infrared and microwave radiative transfer calculations
rt_extra_layers = Other_functions.compute_extra_layers(np.max(z))

# Set these few variables; they will be reset below
noutfilename = ''
version = ''
tape3_info = {'success':-2}

################################################################################
# This is the main loop for the retrieval!
################################################################################
for i in range(len(irs['secs'])):                        # { loop_i
    if ((shour >= irs['hour'][i]) | (irs['hour'][i] > ehour)):
        continue

    # Make sure that the IRS data aren't missing or considered bad
    adderr = ''
    if(itype == 'IRS'):
        if(vip['irs_ignore_status_missingDataFlag'] != 1):
            if(irs['missingDataFlag'][i] != 0):
                if(irs['missingDataFlag'][i] == 10):
                    adderr = adderr+'(IRS radiance at 675 cm-1 is outside QC range -- check VIP file)'
                else:
                    adderr = adderr+'(missingDataFlag is set)'
        if(vip['irs_ignore_status_hatch'] != 1):
            if(irs['hatchopen'][i] != 1):
                adderr = adderr+'(hatch not open)'

        if(adderr != ''):
            print(f"  Sample {i:2d} at {irs['hour'][i]:.4f} UTC -- no valid {itype:s} data found ",adderr)
            print(f"                     missingDataFlag is {irs['missingDataFlag'][i]:.1f} and hatchOpen is {irs['hatchopen'][i]:.1f}")
            continue

    print(f"  Sample {i:2d} at {irs['hour'][i]:.4f} UTC is being processed (cbh is {irs['cbh'][i]:.3f})")

    # See if we want to use the external sfc pressure instead of irs pressure
    # and check to make sure external data read went okay
    if ((vip['ext_sfc_pres_type'] > 0) & (ext_tseries['nPsfc'] >= 0)):
        print("    Replacing pressure with " +  ext_tseries['ptype'] + " pressure")
        irs['atmos_pres'][i] = ext_tseries['psfc'][i]

    # Make sure the IRS's surface pressure is a valid value, as
    # this is needed to construct a pressure profile from the current X
    if ((vip['station_psfc_min'] > irs['atmos_pres'][i]) | (irs['atmos_pres'][i] > vip['station_psfc_max'])):
        print('Error: Surface pressure is not within range set in VIP -- skipping sample')
        print('     and the values are ',vip['station_psfc_min'], irs['atmos_pres'][i], vip['station_psfc_max'])
        continue
        
    # Select the spectral range to use for the retrieval
    # Define the observation vector and its covariance matrix
    # I'm going to try unapodized spectra, so the cov matrix is diagonal

    w0idx, nY = Other_functions.find_wnum_idx(irs['wnum'],bands)
    if nY < 0:
        VIP_Databases_functions.abort(lbltmpdir,date)
        sys.exit()
    wnum = np.copy(irs['wnum'][w0idx])
    Y = np.copy(irs['radmn'][w0idx,i])              # This is the actual observation vector
    sigY = np.copy(irs['noise'][w0idx,i])           # This is the assumed 1-sigma uncertainty of the obs

    # I need a flag for the observations, so I can select the proper forward model.
    # The values are:
    #                1 -- IRS (AERI/ASSIST)
    #                2 -- MWR zenith data
    #                3 -- external temperature profiler (sonde, lidar, NWP, etc) (*)
    #                4 -- external water vapor profiler (sonde, lidar, NWP, etc) (*)
    #                5 -- surface temperature measurement
    #                6 -- surface water vapor measurement
    #                7 -- external NWP model profile (*)
    #                8 -- external NWP model water vapor profile (*)
    #                9 -- in-situ surface CO2 obs
    #               10 -- MWR non-zenith data
    #               11 -- TROPoe T input following Adler et al. AMT
    #               12 -- TROPoe q input following Adler et al. AMT
    #               13, ... future obs, such as IASI, CrIS, S-HIS, others...
    #     (*) Note that the NWP input could come in two ways, but I wanted a way
    #         to bring in lidar and NWP data, while retaining backwards compatibility

    flagY = np.ones(wnum.shape)               # This will capture the source of input
    dimY = np.copy(wnum)                      # This will capture the dimension properly

    # Start building the observational covariance matrix. I am assuming that
    # the uncertainties in the IRS radiance are uncorrelated (channel-to-channel)
    # If I want to add off-diagonal elements (due to application of the noise
    # filter) this is the right place to do that....
    #   Note that I cannot place the irs_band_noise_inflation logic here, 
    #   as I don't have the sfc WVMR value yet
    Sy = np.diag(sigY**2) # DDT-delete

    # Append on the other observations, making sur to capture their "type"
    # using the flag values above and their dimensions. Note that I am assuming
    # that the covariance matrices between the different types of observations
    # are independent of each other, but in some cases I do allow off-diagonal
    # elements in the covariance matrix for that instrument

    if mwr['n_fields'] > 0:
        tbsky = np.copy(mwr['tbsky'][:,i])
        noise = np.copy(mwr['noise'])
        freq = np.copy(mwr['freq'])
        for gg in range(1,vip['mwr_tb_replicate']):
            # If I am replicating the measurement (to increase it's weight, then
            # I don't want to perfectly replicate it. I will perturb the observation
            # slightly (i.e., one-tenth of the noise level) -- this is what I am doing for surface met too.
            tbsky = np.append(tbsky, mwr['tbsky'][:,i] + np.random.normal(size = len(mwr['tbsky'][:,i]))*mwr['noise']/10.)
            noise = np.append(noise,mwr['noise'])
            freq = np.append(freq, mwr['freq'])

        foo = np.where(tbsky < 2.7)[0]                # Must be larger than the cosmic background
        if len(foo) > 0:
            tbsky[foo] = -999.
        Y = np.append(Y,tbsky)
        nSy = np.diag(noise**2)
        zero = np.zeros((len(noise),len(sigY)))
        Sy = np.append(np.append(Sy,zero,axis=0),np.append(zero.T,nSy,axis=0),axis=1) # DDT-delete
        sigY = np.append(sigY,noise)
        flagY = np.append(flagY, np.ones(vip['mwr_tb_replicate']*mwr['n_fields'])*2)
        dimY = np.append(dimY,freq)

    if ext_prof['nTprof'] > 0:
        foo = np.where((ext_prof['tempminht'] <= ext_prof['ht']) & (ext_prof['ht'] <= ext_prof['tempmaxht']))[0]
        if len(foo) <= 0:
            print('Major error when adding external temp profile to observation vector. This should not happen!')
            VIP_Databases_functions.abort(lbltmpdir,date)
            sys.exit()
        Y = np.append(Y,ext_prof['temp'][foo,i])
        nSy = np.diag(ext_prof['sig_temp'][foo,i]**2)
        zero = np.zeros((len(foo),len(sigY)))
        Sy = np.append(np.append(Sy,zero,axis=0),np.append(zero.T,nSy,axis=0),axis=1) # DDT-delete
        sigY = np.append(sigY, ext_prof['sig_temp'][foo,i])
        flagY = np.append(flagY, np.ones(len(foo))*3)
        dimY = np.append(dimY, z[foo])

    if ext_prof['nQprof'] > 0:
        foo = np.where((ext_prof['wvminht'] <= ext_prof['ht']) & (ext_prof['ht'] <= ext_prof['wvmaxht']))[0]
        if len(foo) <= 0:
            print('Major error when adding external wv profile to observation vector. This should not happen!')
            VIP_Databases_functions.abort(lbltmpdir,date)
            sys.exit()
        Y = np.append(Y,ext_prof['wv'][foo,i])
                # Make sure that the noise values are all above the floor
        uncertY = ext_prof['sig_wv'][foo,i]
        bar = np.where(uncertY < vip['ext_wv_prof_min_error'])[0]
        if len(bar) > 0:
            uncertY[bar] = vip['ext_wv_prof_min_error']
        if vip['ext_wv_add_rel_error'] > 0:
            if verbose >= 2:
                print('Adding systematic error to the external WV profile')
            nSy = Other_functions.add_sys_error(ext_prof['wv'][foo,i], uncertY, vip['ext_wv_add_rel_error'])
        else:
            nSy = np.diag(uncertY**2)
        zero = np.zeros((len(foo),len(sigY)))
        Sy = np.append(np.append(Sy,zero,axis=0),np.append(zero.T,nSy,axis=0),axis=1) # DDT-delete
        sigY = np.append(sigY, uncertY)
        flagY = np.append(flagY, np.ones(len(foo))*4)
        dimY = np.append(dimY, z[foo])

    if ext_tseries['nTsfc'] > 0:
        Y = np.append(Y, ext_tseries['temp'][:,i])
        nSy = np.diag(ext_tseries['stemp'][:,i]**2)
        zero = np.zeros((ext_tseries['nptsT'],len(sigY)))
        Sy = np.append(np.append(Sy,zero,axis=0),np.append(zero.T,nSy,axis=0),axis=1) # DDT-delete
        sigY = np.append(sigY, ext_tseries['stemp'][:,i])
        flagY = np.append(flagY, np.ones(ext_tseries['nptsT'])*5)
        dimY = np.append(dimY, np.ones(ext_tseries['nptsT'])*ext_tseries['sfc_relative_height'])

    if ext_tseries['nQsfc'] > 0:
        Y = np.append(Y, ext_tseries['wv'][:,i])
        nSy = np.diag(ext_tseries['swv'][:,i]**2)
        zero = np.zeros((ext_tseries['nptsQ'],len(sigY)))
        Sy = np.append(np.append(Sy,zero,axis=0),np.append(zero.T,nSy,axis=0),axis=1) # DDT-delete
        sigY = np.append(sigY, ext_tseries['swv'][:,i])
        flagY = np.append(flagY, np.ones(ext_tseries['nptsQ'])*6)
        dimY = np.append(dimY, np.ones(ext_tseries['nptsQ'])*ext_tseries['sfc_relative_height'])

    if mod_prof['nTprof'] > 0:
        foo = np.where((mod_prof['tempminht'] <= mod_prof['ht']) & (mod_prof['ht'] <= mod_prof['tempmaxht']))[0]
        if len(foo) <= 0:
            print('Major error when adding model temp profile to observation vector. This should not happen!')
            VIP_Databases_functions.abort(lbltmpdir,date)
            sys.exit()
        Y = np.append(Y,mod_prof['temp'][foo,i])
        nSy = np.diag(mod_prof['sig_temp'][foo,i]**2)
        zero = np.zeros((len(foo),len(sigY)))
        Sy = np.append(np.append(Sy,zero,axis=0),np.append(zero.T,nSy,axis=0),axis=1) # DDT-delete
        sigY = np.append(sigY, mod_prof['sig_temp'][foo,i])
        flagY = np.append(flagY, np.ones(len(foo))*7)
        dimY = np.append(dimY, z[foo])

    if mod_prof['nQprof'] > 0:
        foo = np.where((mod_prof['wvminht'] <= mod_prof['ht']) & (mod_prof['ht'] <= mod_prof['wvmaxht']))[0]
        if len(foo) <= 0:
            print('Major error when adding model wv profile to observation vector. This should not happen!')
            VIP_Databases_functions.abort(lbltmpdir,date)
            sys.exit()
        Y = np.append(Y,mod_prof['wv'][foo,i])
                # Make sure that the noise values are all above the floor
        uncertY = mod_prof['sig_wv'][foo,i]
        bar = np.where(uncertY < vip['mod_wv_prof_min_error'])[0]
        if len(bar) > 0:
            uncertY[bar] = vip['mod_wv_prof_min_error']
        nSy = np.diag(uncertY**2)
        zero = np.zeros((len(foo),len(sigY)))
        Sy = np.append(np.append(Sy,zero,axis=0),np.append(zero.T,nSy,axis=0),axis=1) # DDT-delete
        sigY = np.append(sigY, uncertY)
        flagY = np.append(flagY, np.ones(len(foo))*8)
        dimY = np.append(dimY, z[foo])

    if ext_tseries['nCO2sfc'] > 0:
        Y = np.append(Y,ext_tseries['co2'][:,i])
        nSy = np.diag(ext_tseries['sco2'][:,i]**2)
        zero = np.zeros((ext_tseries['nptsCO2'],len(sigY)))
        Sy = np.append(np.append(Sy,zero,axis=0),np.append(zero.T,nSy,axis=0),axis=1) # DDT-delete
        sigY = np.append(sigY, ext_tseries['sco2'][:,i])
        flagY = np.append(flagY, np.ones(ext_tseries['nptsCO2'])*9)
        dimY = np.append(dimY,np.ones(ext_tseries['nptsCO2'])*ext_tseries['co2_sfc_relative_height'])

    if mwrscan['n_fields'] > 0:
        tbsky = np.copy(mwrscan['tbsky'][:,i])
        noise = np.copy(mwrscan['noise'])
        foo = np.where(tbsky < 2.7)[0]
        if len(foo) > 0:
            tbsky[foo] = -999.
        Y = np.append(Y, tbsky)
        nSy = np.diag(noise**2)
        zero = np.zeros((len(noise),len(sigY)))
        Sy = np.append(np.append(Sy,zero,axis=0),np.append(zero.T,nSy,axis=0),axis=1) # DDT-delete
        sigY = np.append(sigY, noise)
        flagY = np.append(flagY, np.ones(len(mwrscan['dim']))*10)
        dimY = np.append(dimY, mwrscan['dim'])

    if vip['add_tropoe_T_input_flag'] == 1:
        if(dotemp < 1):
            print('Error: temperature retrieval must be enabled if the add_tropoe_t_input_flag is turned on - aborting')
            VIP_Databases_functions.abort(lbltmpdir,date)
        Y = np.append(Y, in_tropoe['temperature'])
        inflated_sigma = Other_functions.inflate_in_tropoe_uncertainty('T',irs['secs'][i],in_tropoe,z,Sa,vip,verbose=verbose)
        nSy = np.diag(inflated_sigma**2)
        zero = np.zeros((len(inflated_sigma),len(sigY)))
        Sy = np.append(np.append(Sy,zero,axis=0),np.append(zero.T,nSy,axis=0),axis=1) # DDT-delete
        sigY = np.append(sigY, inflated_sigma)
        flagY = np.append(flagY, np.ones(len(inflated_sigma))*11)
        dimY = np.append(dimY, in_tropoe['height'])

    if vip['add_tropoe_q_input_flag'] == 1:
        if(dowvmr < 1):
            print('Error: WVMR retrieval must be enabled if the add_tropoe_q_input_flag is turned on - aborting')
            VIP_Databases_functions.abort(lbltmpdir,date)
        Y = np.append(Y, in_tropoe['waterVapor'])
        inflated_sigma = Other_functions.inflate_in_tropoe_uncertainty('q',irs['secs'][i],in_tropoe,z,Sa,vip,verbose=verbose)
        nSy = np.diag(inflated_sigma**2)
        zero = np.zeros((len(inflated_sigma),len(sigY)))
        Sy = np.append(np.append(Sy,zero,axis=0),np.append(zero.T,nSy,axis=0),axis=1) # DDT-delete
        sigY = np.append(sigY, inflated_sigma)
        flagY = np.append(flagY, np.ones(len(inflated_sigma))*12)
        dimY = np.append(dimY, in_tropoe['height'])

    if vip['add_tropoe_co2_input_flag'] == 1:
        if(doco2 < 1):
            print('Error: CO2 retrieval must be enabled if the add_tropoe_co2_input_flag is turned on - aborting')
            VIP_Databases_functions.abort(lbltmpdir,date)
            sys.exit()
        Y = np.append(Y, in_tropoe['co2'])
        inflated_sigma = Other_functions.inflate_in_tropoe_uncertainty('co2',irs['secs'][i],in_tropoe,z,Sa,vip,verbose=verbose)
        print(inflated_sigma.shape)
        nSy = np.diag(inflated_sigma**2)
        zero = np.zeros((len(inflated_sigma),len(sigY)))
        Sy = np.append(np.append(Sy,zero,axis=0),np.append(zero.T,nSy,axis=0),axis=1) # DDT-delete
        sigY = np.append(sigY, inflated_sigma)
        flagY = np.append(flagY, np.ones(len(inflated_sigma))*13)
        dimY = np.append(dimY, np.arange(3))
        print("DDT -- shape of variables")
        print(Y.shape, nSy.shape, sigY.shape, flagY.shape, dimY.shape)

    nY = len(Y)

    # Now, inflate the noise in the IRS spectral band, if it is set and there is a valid surface WVMR measurement
    if((irs_band_noise_inflation['onoff'] > 0) & (vip['irs_type'] > 0)):
        if verbose >= 2:
            print('    Inflating the noise in the IRS spectral band between ',
                      irs_band_noise_inflation['wnum1'], ' and ',irs_band_noise_inflation['wnum2'], ' cm-1')
        feh6 = np.where(flagY == 6)[0]
        feh1 = np.where(flagY == 1)[0]
        if len(feh1) <= 0:
            if verbose >= 2:
                print('Error: Unable to find any IRS wavenumbers in this test (inflate noise band)')
            VIP_Databases_functions.abort(lbltmpdir,date)
            sys.exit()

            # See if the spectral band to inflate the noise is actually included in the desired observations
            # but if there are no spectral elements, then obviously we won't be inflating the noise in the band
        bar = np.where((irs_band_noise_inflation['wnum1'] <= dimY[feh1]) &
                       (dimY[feh1] <= irs_band_noise_inflation['wnum2']))[0]
        if len(bar) > 0:
                # If there is no surface met (or sfc met has negative WVMR) and the 
                #     irs_band_noise_inflation is on, then inflate the noise to its maximum by default
            tmpyval = max(irs_band_noise_inflation['multiplier'])
            if len(feh6) > 0:
                if np.mean(Y[feh6]) <= 0:
                    if verbose >= 2:
                        print('      Warning: the surface WVMR value was not positive; will use max(irs_band_noise_inflation)')
                else:
                    tmpxval = np.mean(Y[feh6])
                    if(tmpxval < np.min(irs_band_noise_inflation['wvmr'])):
                        tmpxval = np.min(irs_band_noise_inflation['wvmr'])
                    if(tmpxval > np.max(irs_band_noise_inflation['wvmr'])):
                        tmpxval = np.max(irs_band_noise_inflation['wvmr'])
                    tmpyval = np.interp(tmpxval, 
                              irs_band_noise_inflation['wvmr'],
                              irs_band_noise_inflation['multiplier'])
            else:
                if verbose >= 2:
                    print('      Warning: no surface WVMR value found; will use max(irs_band_noise_inflation)')
                # Now scale the noise in this spectral band
            if(verbose >= 2):
                print(f'      Scaling the IRS band noise by a factor of {tmpyval:.2f}')
                print(f'      {len(bar):d} spectral elements had their noise changed')
            sigY[feh1[bar]] = sigY[feh1[bar]] * tmpyval


    # Quick check: All of the 1-sigma uncertainties from the observations
    # should have been positive. If not then abort as extra logic needs
    # to be added above...
    foo = np.where((sigY <= 0) & (Y < -900))[0]
    if len(foo) > 0:
        tmp = np.copy(flagY[foo])
        feh = np.unique(tmp)
        if len(feh) <= 0:
            print('This should not happen. Major error in quick check of 1-sigma uncertainties')
            VIP_Databases_functions.abort(lbltmpdir,date)
            sys.exit()
        else:
            print(('Warning: There were missing values in these obs: ' + str(feh)))
        sigY[foo] *= -1           # Presumably, the missing values had -999 for their uncertainties

    foo = np.where((sigY <= 0) & (Y > -900))[0]
    if len(foo) > 0:
        tmp = np.copy(flagY[foo])
        feh = np.unique(tmp)
        if len(feh) <= 0:
            print('This should not happen. Major error in quick check of 1-sigma uncertainties')
            VIP_Databases_functions.abort(lbltmpdir,date)
            sys.exit()
        else:
            print(('Error: There were negative 1-sigma uncertainties in obs: ' + str(feh) + ' Skipping sample'))
            continue
    
    # Check if there is NaN's in the data
    foo = np.where(np.isnan(sigY) | (np.isnan(Y)))[0]
    if len(foo) >0:
       tmp = np.copy(flagY[foo])
       feh = np.unique(tmp)
       if len(feh) <= 0:
          print('This should not happen. Major error in quick check of 1-sigma uncertainties')
          VIP_Databases_functions.abort(lbltmpdir,date)
          sys.exit()
       else:
           print('There are NaNs in the obs_vector or 1-sigma uncertainties for obs: ' + str(feh) + ' Skipping sample')
           continue

    # Check if the Y vector is completely -999, if so, skip this sample
    foo = np.where(Y > -900)[0]
    if len(foo) == 0:
        print('There are no observations in the observation vector. Skipping sample...')
        continue

    # Compute Sy, assuming this is a diagonal matrix.  
    # Note that I have some logic above that needs to be deleted (marked with DDT-delete, if we are sure this matrix is always diagonal)
    Sy = np.diag(sigY**2)

    # Compute the estimate of the forward model uncertainty (Sf).
    # This is computed at Sf = kb # Sb # transpose(Kb), where
    # B is the vector of model parameters and Sb is its covariance matrix
    # Note: right now, I will assume a perfect forward model

    Sf = np.diag(np.zeros(len(np.diag(Sy))))          # This is a matrix of zeros

    # Build the observational covariance matrix, which is a combination of
    # the uncertainties in the forward model and the observations

    Sm = Sy + Sf
    # print(Sy)
    SmInv = scipy.linalg.pinv(Sm)

    # Get the other input variables that the forward model will need
    nX = len(z)*2                 # For both T and Q

    # Start building the first guess vector
    #    T & Q, LWP, ReL, TauI, ReI, co2(3), ch4(3), n2o(3)
    X0 = np.copy(Xa)     # Start with the prior, and overwrite portions of it if desired
    first_guess = 'prior'
    if vip['first_guess'] == 1:
        # Use the prior as the first guess
        if verbose >= 3:
            print('Using prior as first guess')
    elif vip['first_guess'] == 2:
        # Build a first guess from the IRS-estimated surface temperture,
        # an assumed lapse rate, and a 60% RH as first guess
        if verbose >= 3:
            print('Using Tsfc with lapse rate and 60& RH as first guess')
        first_guess = 'Tsfc with lapse rate and 60% RH'
        lapserate = -7.0        # C / km
        constRH = 60.           # percent RH
        t = irs['tsfc'][i] + z*lapserate
        p = Calcs_Conversions.inv_hypsometric(z, t+273.16, irs['atmos_pres'][i])  # [mb]
        q = Calcs_Conversions.rh2w(t, np.ones(len(z))*constRH/100., p)
        X0 = np.concatenate([t, q, Xa[nX:nX+13]])    # T, Q, LWP, ReL, TauI, ReI, co2(3), ch4(3), n2o(3)
    elif vip['first_guess'] == 3:
        # Get first guess from the previous retrieval, if there is one
        # If there isn't a valid prior retrieval, use prior
        if verbose >= 3:
            print('Using previous good retrieval as first guess')
        if type(xret) == list:
            for j in range(len(xret)-1, -1, -1):            # We want to use the last good retrieval, so loop from back to front
                if (xret[j]['converged'] == 1) & (irs['hour'][i] - xret[j]['hour'] < 1.01):
                    first_guess = 'lastSample'
                    X0 = np.copy(xret[j]['Xn'])
                    break
        if first_guess == 'prior':
            if verbose >= 3:
                print('but there were no good retrievals yet')
    else:
        print('Error: Undefined first guess option')
        VIP_Databases_functions.abort(lbltmpdir,date)
        sys.exit()

    # If this is an observation-minus-background calculation, then override multiple VIP entries and
    # populate the first-guess vector with the input profile
    if vip['omb_flag'] == 1:
        print('  Performing an observation-minus-background (O-B) calculation')
        omb_input = Data_reads.read_omb_file(vip['omb_path'],vip['omb_file'],irs['hour'][i], vip['omb_required_minht'])
        if omb_input['success'] == 0:
            print(omb_input['errstring'])
            VIP_Databases_functions.abort(lbltmpdir,date)
            sys.exit()
        vip['max_iterations'] = 0
        X0t = np.interp(z, omb_input['z'], omb_input['t'])
        X0q = np.interp(z, omb_input['z'], omb_input['q'])
        X0  = np.concatenate([X0t, X0q, Xa[nX:nX+13]])    # T, Q, LWP, ReL, TauI, ReI, co2(3), ch4(3), n2o(3)

    # Build the first guess vector
    itern = 0
    converged = 0
    Xn = np.copy(X0)
    Fxnm1 = np.array([-999.])
    
    # If we are to append to the file, then I need to find the last valid
    # sample in the file, so I only process after that point...
    if ((vip['output_clobber'] == 2) & (check_clobber == 1)):
        xret, fsample, noutfilename, origXa, origSa, opres, in_tropoe = Output_Functions.create_xret(xret,
                            fsample, vip, irs, Xa, Sa, z, bands, dimY, flagY, shour, in_tropoe)
        check_clobber = 0
        if fsample < 0:
            VIP_Databases_functions.abort(lbltmpdir,date)
            sys.exit()
        if fsample == 0:
            vip['output_clobber'] = 0
            xret = []
        if ((verbose >= 1) & (fsample > 0)):
            print(('  Will append output to the file ' + noutfilename))
            print(f'    Starting with sample {fsample:4d} and using the original prior data')
            Xa = origXa
            Sa = origSa

    # If we are not in append mode, but do not have clobber set to 1 then
    # check to see if a conflicting file exists and if so abort
    elif ((vip['output_clobber'] == 0) & (check_clobber == 1)):
        xret, fsample, noutfilename, origXa, origSa, opres, in_tropoe = Output_Functions.create_xret(xret,
                            fsample, vip, irs, Xa, Sa, z, bands, dimY, flagY, shour, in_tropoe)
        check_clobber = 0
        if fsample < 0:
           VIP_Databases_functions.abort(lbltmpdir,date)
           sys.exit()
        else:
            xret = []
    elif vip['output_clobber'] == 1:
        opres = -999

        
    # If we are in 'append' mode, then skip any IRS samples that are
    # before the last time in the xret structure. Generally, the current
    # IRS sample will always be before the last one in the xret structure,
    # except in the cases where we just started the code in append mode. If
    # that happens, then the xret structure will be (partially) populated
    # by the create_xret routine above, which gets the time from the
    # existing netCDF file. And the way we are writing the data that is all
    # that needed to be retrieved...

    if vip['output_clobber'] == 2:
        if irs['secs'][i] <= xret[fsample-1]['secs']:
            print('  ....but was already processed (append mode)')
            continue

    cbh = irs['cbh'][i]
    cbhflag = irs['cbhflag'][i]

    # Define the gamma factors needed to keep the retrieval sane
    # MWR-only retrievals are more linear and thus the gfactor can be more agressive

    if vip['irs_type'] <= -1:
        gfactor = np.array([100.,10.,3.,1.])
    else:
        if(first_guess == 'lastSample'):
            gfactor = np.array([100.,10.,3.,1.])
        else:
            gfactor = np.array([1000.,300.,100.,30.,10.,3.,1.])
    if len(gfactor) <= vip['max_iterations']:
        gfactor = np.append(gfactor, np.ones(vip['max_iterations']-len(gfactor)+3))

    # Select nice round numbers to use as the wavenumber limits for
    # the LBLRTM calc, but remember that I need to pad by 50 cm-1 for FSCAN
    if vip['irs_type'] <= -1:
        lblwnum1 =  100
        lblwnum2 = 1000
    else:
        lblwnum1 = int((np.min(wnum)-60)/100) * 100
        lblwnum2 = (int((np.max(wnum)+60)/100)+1)*100
    continue_next_sample = 0          # A flag used to watch for bad jacobian calcs

    while ((itern <= vip['max_iterations']) & (converged == 0)):        # { While loop over iter

        if verbose >= 3:
            print((' Making the forward calculation for iteration ' + str(itern)))

        if os.path.exists(lbltp5):
            shutil.rmtree(lbltp5)

        if os.path.exists(lbldir):
            shutil.rmtree(lbldir)
            
        if os.path.exists(lbllog):
            shutil.rmtree(lbllog)

        # Update the pressure profile using the current estimate of temperature.  However, if this is
        # the first iteration, we need to know what is being used for the first guess; if we are using 
        # the prior, then we are going to use a canned Jacobian for that first iteration, which means
        # we have to use the same pressure profile (which means the same surface pressure value)
        if((itern == 0) and (vip['first_guess'] == 1) & (opres > 0)):
            sfcp = opres
        else:
            sfcp = irs['atmos_pres'][i]
        if sfcp <= 0:
            print(f'  Error: the surface pressure is not positive (sfcP = {sfcp:f}); aborting')
            VIP_Databases_functions.abort(lbltmpdir,date)
            sys.exit()
        p = Calcs_Conversions.inv_hypsometric(z, Xn[0:int(nX/2)]+273.16, sfcp)

        # If the trace gas profile shape is mandated to be a function of the PBL height,
        # then set that here. First, compute the current estimate of the PBL height,
        # then the coefficient and overwrite any current shape coefficient

        if itern == 0:
            pblh = Other_functions.compute_pblh(z, Xn[0:int(nX/2)], p, np.sqrt(np.diag(Sa[0:int(nX/2), 0:int(nX/2)])),
                                   minht=vip['min_PBL_height'], maxht=vip['max_PBL_height'], nudge=vip['nudge_PBL_height'])
        else:
            pblh = Other_functions.compute_pblh(z, Xn[0:int(nX / 2)], p, np.sqrt(np.diag(Sop[0:int(nX/2), 0:int(nX/2)])),
                                   minht=vip['min_PBL_height'], maxht=vip['max_PBL_height'], nudge=vip['nudge_PBL_height'])

        coef = Other_functions.get_a2_pblh(pblh)           # Get the shape coef for this PBL height
        if (vip['retrieve_co2'] == 1):
            Xn[nX+4+2] = pblh
        elif ((vip['retrieve_co2'] == 2) & (vip['fix_co2_shape'] == 1)):
            Xn[nX+4+2] = coef

        if (vip['retrieve_ch4'] == 1):
            Xn[nX+4+3+2] = pblh
        elif ((vip['retrieve_ch4'] == 2) & (vip['fix_ch4_shape'] == 1)):
            Xn[nX+4+3+2] = coef

        if (vip['retrieve_n2o'] == 1):
            Xn[nX+4+6+2] = pblh
        elif ((vip['retrieve_n2o'] == 2) & (vip['fix_n2o_shape'] == 1)):
            Xn[nX+4+6+2] = coef

        # Decorrelate the levels above the PBLH from the levels below in the prior.
        # The decorrelation factor should be between 1 and 100, I think, and read from VIP.
        # This really should only be done in the daytime though, so will need to
        # add new logic to determin local time-of-day from lat/lon position sometime.
        # prior_pblh_decorrelate_factor = 3.

        # Sa = Other_functions.covariance_matrix_decorrelate_level(prior['Sa'], z, pblh, prior_pblh_decorrelate_factor)

        # Compute its inverse of the prior
        SaInv = scipy.linalg.pinv(Sa)

        # This function makes the forward calculation and computes the Jacobian
        # for the IRS component of the forward model
        if vip['lblrtm_jac_option'] == 1:
            if vip['irs_type'] <= -1:
                    # If this type, then IRS data aren't being used in the retrieval
                    # so the forward calc should be missing and the Jacobian is 0
                flag = 1
            else:
                print('Need to port the compute_jacobian_finitediff function. Have to abort... Sorry!!')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()
                flag, Kij, FXn, wnumc, totaltime  = \
                           Jacobian_Functions.compute_jacobian_finitediff(Xn, p, z,
                           vip['lbl_home'], lbldir, lbltmp, vip['lbl_std_atmos'], lbltp5, lbltp3,
                           cbh, sspl, sspi, lblwnum1, lblwnum2,
                           fixtemp, fixwvmr, doco2, doch4, don2o, fixlcloud, fixicloud,
                           vip['fix_co2_shape'], vip['fix_ch4_shape'], vip['fix_n2o_shape'],
                           vip['jac_max_ht'], vip['lblrtm_forward_threshold'],
                           location['alt'], rt_extra_layers, stdatmos,
                           verbose, debug, doapidize=True)

        elif vip['lblrtm_jac_option'] == 2:
            print('Error: the lblrtm_jac_option == 2 (3method) method is no longer available.  Use option == 4 instead')
            VIP_Databases_functions.abort(lbltmpdir,date)
            sys.exit()

        elif vip['lblrtm_jac_option'] == 3:
            print('Error: the lblrtm_jac_option == 3 (deltaOD) method is no longer available.  Use option == 4 instead')
            VIP_Databases_functions.abort(lbltmpdir,date)
            sys.exit()
            
        elif vip['lblrtm_jac_option'] == 4:
                # Will use the jacobian_interpol method
            if vip['irs_type'] <= -1:
                # If this type, then IRS data aren't being used in the retrieval
                # so the forward calc should be missing and the Jacobian is 0
                wnumc = np.copy(irs['wnum'])
                FXn = np.ones(len(wnumc))*-999.
                Kij = np.zeros((len(wnumc),len(Xn)))
                mlev = {}
                flag = 1
            else:
                if((precompute_prior_jacobian['status'] == 1) & (itern == 0) & (vip['omb_flag'] != 1)):
                        # Load the forward calculation stuff from the precompute prior data
                    if(verbose >= 1):
                        print('    Preloading forward calculation and jacobian from prior structure')
                    FXn   = np.copy(precompute_prior_jacobian['FX0'])
                    Kij   = np.copy(precompute_prior_jacobian['Kij0'])
                    flag  = np.copy(precompute_prior_jacobian['flag0'])
                    wnumc = np.copy(precompute_prior_jacobian['wnumc0'])
                    mlev  = copy.deepcopy(precompute_prior_mlev)
                else:
                        # Otherwise, run the forward model and compute the Jacobian
                    flag, Kij, FXn, wnumc, totaltime, mlev, tape3_info = \
                           Jacobian_Functions.compute_jacobian_irs_interpol(Xn, p, z,
                           vip['lbl_home'], lbldir, lbltmp, vip['lbl_std_atmos'], lbltp5, lbltp3,
                           cbh, sspl, sspi, lblwnum1, lblwnum2,
                           fixtemp, fixwvmr, doco2, doch4, don2o, fixlcloud, fixicloud,
                           vip['fix_co2_shape'], vip['fix_ch4_shape'], vip['fix_n2o_shape'],
                           vip['jac_max_ht'], irs['wnum'], vip['lblrtm_forward_threshold'],
                           location['alt'], rt_extra_layers, stdatmos, vip['lblrtm_jac_interpol_npts_wnum'],
                           vip['irs_type'], tape3_info, vip, verbose, debug, doapodize=False) 
                        # Confirm that the TAPE3 is appropriate for this calculation
                    if((flag != 0) & (tape3_info['success'] > 0)):
                        if((lblwnum1 < tape3_info['minw']) or (tape3_info['maxw'] < lblwnum2)):
                            print('Error: The tape3 selected in the VIP does not span the desired IRS calculation range')
                            print(f"            TAPE3 spans from {tape3_info['minw']:.3f} to {tape3_info['maxw']:.3f} cm-1")
                            print(f"            Needed LBLRTM range is {lblwnum1:.3f} to {lblwnum2:.3f} cm-1")
                            VIP_Databases_functions.abort(lbltmpdir,date)
                            sys.exit()

                    # If we are using the prior for the first guess (FG=1), and we have not already loaded
                    # this structure, then capture the forward calc and jacobian for the first guess
            if((precompute_prior_jacobian['status'] == 0) & (vip['first_guess'] == 1) & (flag != 0)):
                precompute_prior_mlev     = copy.deepcopy(mlev)
                precompute_prior_jacobian = {'status':1, 'X0':np.copy(Xn), 'FX0':np.copy(FXn), 'Kij0':np.copy(Kij),
                    'flag0':np.copy(flag), 'wnumc0':np.copy(wnumc)}
        else:
            print('Error: Undefined jacobian option selected')
            VIP_Databases_functions.abort(lbltmpdir,date)
            sys.exit()

        # If the Jacobian did not compute properly (i.e., an error occurred)
        # then we need to abort
        if flag == 0:
            print(' -- Skipping this sample due to issue with LBLRTM Jacobian (likely bad input profile)')
            continue_next_sample = 1
            break

        # Select the wavenumber indices to use
        w1idx, junk = Other_functions.find_wnum_idx(wnumc, bands)
        if len(w1idx) != len(w0idx):
            print('Problem with wnum indices1')
            VIP_Databases_functions.abort(lbltmpdir,date)
            sys.exit()
        wnumc = wnumc[w1idx]
        FXn = FXn[w1idx]
        Kij = Kij[w1idx,:]

        # Derive the CBH estimate from the spectral data
        if vip['irs_type'] > 0:
            cbh_mlev, cld_emis = Other_functions.do_mlev_cbh(irs['wnum'][w0idx],irs['radmn'][w0idx,i],
                            mlev['wnum'][w1idx],mlev['radclear'][w1idx],mlev['radBcld'][:,w1idx],mlev['maxht'],z,itern, vip)
            cbh_tcld = Other_functions.do_tcld_cbh(irs['wnum'][w0idx],irs['radmn'][w0idx,i], Xn, z, cbh_mlev, vip)
        else:
            cbh_mlev = -999.
            cld_emis = -999.
            cbh_tcld = -999.


        # Are there missing values from the IRS? If so,then we want to make the
        # forward model calculation have the same value and put no sensitivity
        # in the Jacobian there so that the retrieval is unaffected (this is really for
        # irs_type = -1)

        foo = np.where(flagY == 1)[0]
        bar = np.where(Y[foo] < -900)[0]
        if len(bar) > 0:
            FXn[bar] = np.copy(Y[foo[bar]])
            for gg in range(len(bar)):
                Kij[bar[gg],:] = 0.

        # Now start processing the other observation types that might be in the obs vector

        # Perform the forward model calculation and compute the Jacobian for the
        # MWR-zenith portion of the observation vector
        foo = np.where(flagY == 2)[0]
        if len(foo) > 0:
            if create_monortm_config == 1:
                # Create the MonoRTM configuration file
                lun = open(lbltmpdir + '/' + monortm_config, 'w')
                lun.write(vip['monortm_exec'] + '\n')
                lun.write(vip['monortm_spec'] + '\n')
                lun.write('0\n')          # The verbose flag
                lun.write('{:0d}\n'.format(vip['lbl_std_atmos']))
                lun.write('1\n')          # The 'output layer optical depths' flag
                for gg in range(6):       # The 6 continuum multipliers
                    lun.write('1.0\n')
                lun.write('{:7.3f}\n'.format(np.max(z)-0.01))
                lun.write('{:0d}\n'.format(len(z)+len(rt_extra_layers)))
                for gg in range(len(z)):
                    lun.write('{:7.3f}\n'.format(z[gg]))
                for gg in range(len(rt_extra_layers)):
                    lun.write('{:7.3f}\n'.format(rt_extra_layers[gg]))
                lun.close()

                # Turn the flag off, as we only need to create these files once
                create_monortm_config = 0



            if create_monortm_zfreq == 1:
                # Create the MonoRTM freuency file
                lun = open(lbltmpdir + '/' + monortm_zfreq, 'w')
                lun.write('\n')
                lun.write('{:0d}\n'.format(len(mwr['freq'])))
                for gg in range(len(mwr['freq'])):
                    lun.write('{:7.3f}\n'.format(mwr['freq'][gg]))
                lun.close()

                # Turn the flag off, as we only need to create these files once
                create_monortm_zfreq = 0

                # Run the forward model and compute the Jacobian
            if vip['monortm_jac_option'] == 1:
                flag, KK, FF, m_comp_time = Jacobian_Functions.compute_jacobian_microwave_finitediff(Xn, p, z,
                            mwr['freq'], cbh, vip, lbltmpdir, monortm_tfile, monortm_zexec,
                            fixtemp, fixwvmr, fixlcloud, vip['jac_max_ht'], stdatmos, location['alt'], verbose)

            elif vip['monortm_jac_option'] == 2:
                    flag, KK, FF, m_comp_time = Jacobian_Functions.compute_jacobian_microwave_3method(Xn, p, z,
                            mwr['freq'], cbh, vip, lbltmpdir, monortm_tfile, monortm_zexec,
                            fixtemp, fixwvmr, fixlcloud, vip['jac_max_ht'], stdatmos, location['alt'], verbose)

            else:
                print('Error: Undefined option for monortm_jac_option')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()

            # If the Jacobian did not compute properly (i.e., an error ocurred),
            # then we need to abort
            if flag == 0:
                print('-- Skipping this sample due to issue with MonoRTM Jacobian (likely bad input profile)')
                continue_next_sample = 1
                break

            # Check to see if I am replicating the observation; if so, then adjust the
            # forward calc and Jacobian appropriately (they are the same value repeated)
            # TODO - There's an issue here when the TB obs are replicated
            tmpFF = np.copy(FF)
            tmpKK = np.copy(KK)
            for gg in range(1,vip['mwr_tb_replicate']):
                tmpFF = np.append(tmpFF, FF)
                tmpKK = np.append(tmpKK, KK, axis = 0)
            FF = np.copy(tmpFF)
            KK = np.copy(tmpKK)

            # Now the size of the forward calculation should be the correct size to match
            # the number of MWR observations in the Y vector
            if len(foo) != len(FF):
                print('Problem computing the Jacobian for the microwave radiometer')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()

            # Are there misssing values from the MWR? If so, then we want to make
            # the forward model calculation have the same value and put no sensitivity
            # in the Jacobian there so that the retrieval is unaffected

            bar = np.where(Y[foo] < -900)[0]
            if len(bar) > 0:
                FF[bar] = Y[foo[bar]]
                KK[bar,:] = 0.
            Kij = np.append(Kij, KK, axis = 0)
            FXn = np.append(FXn,FF)

        # Perform the forward model calculation and compute the Jacobian for the
        # external temperature profiler portion of the observation vector
        foo = np.where(flagY == 3)[0]
        if len(foo) > 0:
            flag, KK, FF = Jacobian_Functions.compute_jacobian_external_temp_profiler(Xn, p, z,
                            ext_prof['tempminht'], ext_prof['tempmaxht'], ext_prof['temp_type'])

            if flag == 0:
                print('Problem computing the Jacobian for the external temp profiler. Have to abort')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()
            if len(foo) != len(FF):
                print('Problem computing the Jacobian for the external temp profiler')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()

            # Are there misssing values from the external profilers? If so, then we want to make
            # the forward model calculation have the same value and put no sensitivity
            # in the Jacobian there so that the retrieval is unaffected

            bar = np.where(Y[foo] < -900)[0]
            if len(bar) > 0:
                FF[bar] = Y[foo[bar]]
                KK[bar,:] = 0.
            Kij = np.append(Kij, KK, axis = 0)
            FXn = np.append(FXn,FF)

        # Perform the forward model calculation and compute the Jacobian for the
        # external water vapor profiler portion of the observation vector
        foo = np.where(flagY == 4)[0]
        if len(foo) > 0:
            flag, KK, FF = Jacobian_Functions.compute_jacobian_external_wv_profiler(Xn, p, z,
                            ext_prof['wvminht'], ext_prof['wvmaxht'], ext_prof['wv_type'], ext_prof['wvmultiplier'])


            if flag == 0:
                print('Problem computing the Jacobian for the external wv profiler. Have to abort')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()
            if len(foo) != len(FF):
                print('Problem computing the Jacobian for the external wv profiler')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()

            # Are there misssing values from the external profilers? If so, then we want to make
            # the forward model calculation have the same value and put no sensitivity
            # in the Jacobian there so that the retrieval is unaffected

            bar = np.where(Y[foo] < -900)[0]
            if len(bar) > 0:
                FF[bar] = Y[foo[bar]]
                KK[bar,:] = 0.
            Kij = np.append(Kij, KK, axis = 0)
            FXn = np.append(FXn,FF)

        # Perform the forward model calculation and compute the Jacobian for the
        # external water vapor profiler portion ofthe observation vector
        foo = np.where((flagY == 5) | (flagY == 6))[0]
        if len(foo) > 0:
            units = ''
            barT = np.where(flagY == 5)[0]
            barQ = np.where(flagY == 6)[0]

            if ((len(barT) > 0) & (len(barQ) > 0)):
                units = np.append(np.array([ext_tseries['tunit']]*len(barT)), np.array([ext_tseries['qunit']]*len(barQ)))
            elif ((len(barT) > 0) & (len(barQ) == 0)):
                units = np.array([ext_tseries['tunit']]*len(barT))
            elif ((len(barT) == 0) & (len(barQ) > 0)):
                units = np.array([ext_tseries['qunit']]*len(barQ))
            else:
                print('This absolutely should not happen, as I should be finding some units')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()

            flag, KK, FF = Jacobian_Functions.compute_jacobian_external_sfc_met(Xn, p, z,
                             ext_tseries['sfc_relative_height'], units, vip['prior_chimney_ht'])

            if flag == 0:
                print('Problem computing the Jacobian for the external surface met data. Have to abort')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()

            if len(foo) != len(FF):
                print('Problem computing the Jacobian for the external surface met')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()

            # Are there misssing values in the surface met data? If so, then we want to make
            # the forward model calculation have the same value and put no sensitivity
            # in the Jacobian there so that the retrieval is unaffected

            bar = np.where(Y[foo] < -900)[0]
            if len(bar) > 0:
                FF[bar] = Y[foo[bar]]
                KK[bar,:] = 0.
            Kij = np.append(Kij, KK, axis = 0)
            FXn = np.append(FXn,FF)

        # Perform the forward model calculation and compute the Jacobian for the
        # NWP model temperature profiler portion of the observation vector
        foo = np.where(flagY == 7)[0]
        if len(foo) > 0:
            flag, KK, FF = Jacobian_Functions.compute_jacobian_external_temp_profiler(Xn, p, z,
                            mod_prof['tempminht'], mod_prof['tempmaxht'], mod_prof['temp_type'])

            if flag == 0:
                print('Problem computing the Jacobian for the NWP temp profiler. Have to abort')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()
            if len(foo) != len(FF):
                print('Problem computing the Jacobian for the NWP temp profiler')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()

            # Are there misssing values from the NWP profile? If so, then we want to make
            # the forward model calculation have the same value and put no sensitivity
            # in the Jacobian there so that the retrieval is unaffected

            bar = np.where(Y[foo] < -900)[0]
            if len(bar) > 0:
                FF[bar] = Y[foo[bar]]
                KK[bar,:] = 0.
            Kij = np.append(Kij, KK, axis = 0)
            FXn = np.append(FXn,FF)

        # Perform the forward model calculation and compute the Jacobian for the
        # NWP model water vapor profiler portion of the observation vector
        foo = np.where(flagY == 8)[0]
        if len(foo) > 0:
            flag, KK, FF = Jacobian_Functions.compute_jacobian_external_wv_profiler(Xn, p, z,
                            mod_prof['wvminht'], mod_prof['wvmaxht'], mod_prof['wv_type'], mod_prof['wvmultiplier'])

            if flag == 0:
                print('Problem computing the Jacobian for the NWP wv profiler. Have to abort')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()
            if len(foo) != len(FF):
                print('Problem computing the Jacobian for the NWP wv profiler')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()

            # Are there misssing values from the NWP profile? If so, then we want to make
            # the forward model calculation have the same value and put no sensitivity
            # in the Jacobian there so that the retrieval is unaffected

            bar = np.where(Y[foo] < -900)[0]
            if len(bar) > 0:
                FF[bar] = Y[foo[bar]]
                KK[bar,:] = 0.
            Kij = np.append(Kij, KK, axis = 0)
            FXn = np.append(FXn,FF)

        # Perform the forward model calculation and compute the Jacobian for the
        # in-situ surface co2 measurements of the observation vector
        foo = np.where(flagY == 9)[0]
        if len(foo) > 0:
            flag, KK, FF = Jacobian_Functions.compute_jacobian_external_sfc_co2(Xn, p, z,
                        ext_tseries['co2_sfc_relative_height'], vip['retrieve_co2'],
                        vip['fix_co2_shape'], ext_tseries['nptsCO2'])

            if flag == 0:
                print('Problem computing the Jacobian for the in-situ surface co2 data. Have to abort')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()
            if len(foo) != len(FF):
                print('Problem computing the Jacobian for the in-situ surfac co2 data.')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()

            # Are there misssing values from the in-situ co2 data? If so, then we want to make
            # the forward model calculation have the same value and put no sensitivity
            # in the Jacobian there so that the retrieval is unaffected

            bar = np.where(Y[foo] < -900)[0]
            if len(bar) > 0:
                FF[bar] = Y[foo[bar]]
                KK[bar,:] = 0.
            Kij = np.append(Kij, KK, axis = 0)
            FXn = np.append(FXn,FF)

        # Perform the forward model calculation and compute the Jacobian for the
        # MWR-scan portion of the observation vector
        foo = np.where(flagY == 10)[0]
        if len(foo) > 0:
            if create_monortm_config == 1:
                # Note this is the same config file used in the MWR-zenith, so it
                # may have already been created. (The config is the same in both
                # -zenith and -scan)

                # Create the MonoRTM configuration file
                lun = open(lbltmpdir + '/' + monortm_config, 'w')
                lun.write(vip['monortm_exec'] + '\n')
                lun.write(vip['monortm_spec'] + '\n')
                lun.write('0\n')          # The verbose flag
                lun.write('{:0d}\n'.format(vip['lbl_std_atmos']))
                lun.write('1\n')          # The 'output layer optical depths' flag
                for gg in range(6):       # The 6 continuum multipliers
                    lun.write('1,0\n')
                lun.write('{:7.3f}\n'.format(np.max(z)-0.01))
                lun.write('{:0d}\n'.format(len(z)+len(rt_extra_layers)))
                for gg in range(len(z)):
                    lun.write('{:7.3f}\n'.format(z[gg]))
                for gg in range(len(rt_extra_layers)):
                    lun.write('{:7.3f}\n'.format(rt_extra_layers[gg]))
                lun.close()

                # Turn the flag off, as we only need to create these files once
                create_monortm_config = 0

            if create_monortm_sfreq == 1:
                # Create the MonoRTM frequency file
                lun = open(lbltmpdir + '/' + monortm_sfreq, 'w')
                lun.write('\n')
                lun.write('{:0d}\n'.format(len(mwrscan['freq'])))
                for gg in range(len(mwrscan['freq'])):
                    lun.write('{:7.3f}\n'.format(mwrscan['freq'][gg]))
                lun.close()

                # Turn the flag off, as we only need to create thee files once
                create_monortm_sfreq = 0

            # Run the forward model and compute the Jacobian
            if vip['monortm_jac_option'] == 1:
                print('Error: DDT has not enabled finite differencing in the MWR-scan jacobian yet')
            elif vip['monortm_jac_option'] == 2:
                flag, KK, FF, m_comp_time = Jacobian_Functions.compute_jacobian_microwavescan_3method(Xn, p, z,
                                        mwrscan, cbh, vip, lbltmpdir, monortm_tfile, monortm_sexec,
                                        fixtemp, fixwvmr, fixlcloud, vip['jac_max_ht'], stdatmos, location['alt'],
                                        verbose)
            else:
                print('Error: Undefined option for monortm_jac_option')

            # If the Jacobian did not compute properly (i.e., an error occurred)
            # then we need to abort

            if flag == 0:
                print(' -- Skipping this sample due to issure with MonoRTM Jacobian (likely bad input profile)')
                continue_next_sample = 1
                break

            # Now the size fo the forward calculation should be the correct size to match the
            # the number of MWR-scan observation in the Y vector
            if len(foo) != len(FF):
                print('Problem computing the Jacobian for the microwave radiometer scan')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()

            # Are there misssing values from the MWR? If so, then we want to make
            # the forward model calculation have the same value and put no sensitivity
            # in the Jacobian there so that the retrieval is unaffected

            bar = np.where(Y[foo] < -900)[0]
            if len(bar) > 0:
                FF[bar] = Y[foo[bar]]
                KK[bar,:] = 0.
            Kij = np.append(Kij, KK, axis = 0)
            FXn = np.append(FXn,FF)

        # Perform the forward model calculation and compute the Jacobian for the temp "in_tropoe" portion of the observation vector
        # The forward model is trivial, as the data are on the same height grid
        foo = np.where(flagY == 11)[0]
        if len(foo) > 0:
            FF  = Xn[0:int(nX/2)]
            KK  = np.zeros((len(in_tropoe['temperature']),len(Xn)))
            foo = np.where(in_tropoe['temperature'] > -900)[0]
            for ii in range(len(foo)):
                KK[foo[ii],foo[ii]] = 1
            Kij = np.append(Kij, KK, axis = 0)
            FXn = np.append(FXn,FF)

        # Perform the forward model calculation and compute the Jacobian for the WVMR "in_tropoe" portion of the observation vector
        # The forward model is trivial, as the data are on the same height grid
        foo = np.where(flagY == 12)[0]
        if len(foo) > 0:
            FF  = Xn[int(nX/2):nX]
            KK  = np.zeros((len(in_tropoe['waterVapor']),len(Xn)))
            foo = np.where(in_tropoe['waterVapor'] > -900)[0]
            for ii in range(len(foo)):
                KK[foo[ii],foo[ii]+int(nX/2)] = 1
            Kij = np.append(Kij, KK, axis = 0)
            FXn = np.append(FXn,FF)

        # Perform the forward model calculation and compute the Jacobian for the co2 "in_tropoe" portion of the observation vector
        # The forward model is non-trivial, as radiative transfer will be needed.  However, we will assume that the jacobian
        # that was computed in the RT for this section (regular data) is the same as for the in_tropoe section
        foo = np.where(flagY == 13)[0]
        if len(foo) > 0:
            FF  = Xn[nX+4+np.arange(3)]
            print(f"DDT -- here are the co2 values used for the in_tropoe part: {FF:.2f}")
            KK  = np.zeros((len(in_tropoe['co2']),len(Xn)))
            print(KK.shape)
            print(Kij.shape)
            foo = np.where(in_tropoe['co2'] > -900)[0]
            for ii in range(len(foo)):
                KK[foo[ii],:] = Kij[foo[ii]+nX+4,:]
            Kij = np.append(Kij, KK, axis = 0)
            FXn = np.append(FXn,FF)
            print(f"DDT -- the code needs some serious tests right here to make sure all is good!")
            sys.exit()


        ########
        # Done computing forward calculation and Jacobians. Now the retrieval math
        ########

        # Compute the L-curve values to determine an appropriate point for gamma
        use_L_curve = 0
        if use_L_curve > 0:
            # Starting here, I am following the Carissimo et al. logic hete
            if (itern == 0) & verbose >= 2:
                print('Using the L-curve method to optimize gamma')
                ggamma = np.arange(101)*0.10 - 5     # Values from -5 to +5
                ggamma = 10.**ggamma        # this is the range of values I want: 10^(-5) to 10^(+5)

                gfac = Other_functions.lcurve(ggamma, flagY, Y, FXn, Kij, Xn, Xa, Sa, Sm, z)
        else:
            gfac = gfactor[itern]

        # Retrieval Calculations
        B      = (gfac * SaInv) + Kij.T.dot(SmInv).dot(Kij)
        Binv   = scipy.linalg.pinv(B)
        Gain   = Binv.dot(Kij.T).dot(SmInv)
        Xnp1   = Xa[:,None] + Gain.dot(Y[:,None] - FXn[:,None] + Kij.dot((Xn-Xa)[:,None]))
        Sop    = Binv.dot(gfac*gfac*SaInv + Kij.T.dot(SmInv).dot(Kij)).dot(Binv)
        SopInv = scipy.linalg.pinv(Sop)
        Akern  = (Binv.dot(Kij.T).dot(SmInv).dot(Kij)).T
        
        # Calculate the Akern without model data included
        foo = np.where((flagY<7) | (flagY>8))[0]
        if len(foo) > 0:
            tmp_Kij = np.copy(Kij[foo,:])
            tmp_Sm = np.copy(Sm[foo,:])
            tmp_Sm = tmp_Sm[:,foo]
            tmp_SmInv = scipy.linalg.pinv(tmp_Sm)
            tmp_B = (gfac * SaInv) + tmp_Kij.T.dot(tmp_SmInv).dot(tmp_Kij)
            tmp_Binv = scipy.linalg.pinv(tmp_B)
            Akern_nm = (tmp_Binv.dot(tmp_Kij.T).dot(tmp_SmInv).dot(tmp_Kij)).T
        else:
            Akern_nm = np.copy(Akern) * 0 - 999.
        
        if debug:
            print('Special debug mode -- writing variables from retrieval calcs in the directory '+vip['lbl_temp_dir'])
                    # Only write out some variables once (i.e., not needed for every iteration as they don't change)
            if itern == 0:
                Output_Functions.write_variable(    Y,vip['lbl_temp_dir']+'/debug_output.Y.iter_'+str(itern)+'.cdf',verbose=verbose)
                Output_Functions.write_variable(   Sa,vip['lbl_temp_dir']+'/debug_output.Sa.iter_'+str(itern)+'.cdf',verbose=verbose)
                Output_Functions.write_variable(SaInv,vip['lbl_temp_dir']+'/debug_output.SaInv.iter_'+str(itern)+'.cdf',verbose=verbose)
                Output_Functions.write_variable(   Xa,vip['lbl_temp_dir']+'/debug_output.Xa.iter_'+str(itern)+'.cdf',verbose=verbose)
                Output_Functions.write_variable(SmInv,vip['lbl_temp_dir']+'/debug_output.SmInv.iter_'+str(itern)+'.cdf',verbose=verbose)
                     # Write out these variables for each iteration
            Output_Functions.write_variable(   Xn,vip['lbl_temp_dir']+'/debug_output.Xn.iter_'+str(itern)+'.cdf',verbose=verbose)
            Output_Functions.write_variable( Xnp1,vip['lbl_temp_dir']+'/debug_output.Xnp1.iter_'+str(itern)+'.cdf',verbose=verbose)
            Output_Functions.write_variable(    B,vip['lbl_temp_dir']+'/debug_output.B.iter_'+str(itern)+'.cdf',verbose=verbose)
            Output_Functions.write_variable( gfac,vip['lbl_temp_dir']+'/debug_output.gfac.iter_'+str(itern)+'.cdf',verbose=verbose)
            Output_Functions.write_variable( Gain,vip['lbl_temp_dir']+'/debug_output.Gain.iter_'+str(itern)+'.cdf',verbose=verbose)
            Output_Functions.write_variable(  Sop,vip['lbl_temp_dir']+'/debug_output.Sop.iter_'+str(itern)+'.cdf',verbose=verbose)
            Output_Functions.write_variable(Akern,vip['lbl_temp_dir']+'/debug_output.Akern.iter_'+str(itern)+'.cdf',verbose=verbose)
            Output_Functions.write_variable(  Kij,vip['lbl_temp_dir']+'/debug_output.Kij.iter_'+str(itern)+'.cdf',verbose=verbose)
            Output_Functions.write_variable(  FXn,vip['lbl_temp_dir']+'/debug_output.FXn.iter_'+str(itern)+'.cdf',verbose=verbose)

        # If we are trying to fix the shape of the TG profiles as a function of the
        # PBLH, then we need to make a special tweak here. The gain matrix for the
        # factor(s) will be zero, which would make the next iteration have the shape
        # factor in the prior. But I don't want to be changing the prior with each iteration,
        # as that will impact the "append" option (if we are using that). So we need this
        # stub of code to do the same thing

        if ((vip['retrieve_co2'] == 2) & (vip['fix_co2_shape'] == 1)):
            Xnp1[nX+4+2] = np.copy(Xn[nX+4+2])
        if ((vip['retrieve_ch4'] == 2) & (vip['fix_ch4_shape'] == 1)):
            Xnp1[nX+4+5] = np.copy(Xn[nX+4+5])
        if ((vip['retrieve_n2o'] == 2) & (vip['fix_n2o_shape'] == 1)):
            Xnp1[nX+4+8] = np.copy(Xn[nX+4+8])

        # Look for NaN values in this updated state vector. They should not
        # exist, but if they do, then let's stop the code here to allow
        # me to look at it. Not optimal solution for operation code
        # though, as it really should output a flagged result or abort/
        foo = np.where(np.isnan(Xnp1))[0]          # DDT
        if len(foo) > 0:
            print('Stopping for NaN issue 1')
            VIP_Databases_functions.abort(lbltmpdir,date)
            sys.exit()

        # Compute some information content numbers. The DFS will be computed
        # as the [total, temp, WVMR, LWP, ReffL, TauI, ReffI, co2, ch4, n2o]
        tmp = np.diag(Akern)
        dfs = np.array([np.sum(tmp), np.sum(tmp[0:int(nX/2)]), np.sum(tmp[int(nX/2):nX]), tmp[nX],
                    tmp[nX+1], tmp[nX+2], tmp[nX+3], tmp[nX+4], tmp[nX+5], tmp[nX+6],
                    tmp[nX+7], tmp[nX+8], tmp[nX+9], tmp[nX+10], tmp[nX+11], tmp[nX+12]])

        # Now compute the DFS assuming that there is no model data
        tmp = np.diag(Akern_nm)
        dfs_nm = np.array([np.sum(tmp), np.sum(tmp[0:int(nX/2)]), np.sum(tmp[int(nX/2):nX]), tmp[nX],
                    tmp[nX+1], tmp[nX+2], tmp[nX+3], tmp[nX+4], tmp[nX+5], tmp[nX+6],
                    tmp[nX+7], tmp[nX+8], tmp[nX+9], tmp[nX+10], tmp[nX+11], tmp[nX+12]])
        
        # Compute Shannon information content, but trap for non-positive numbers (which occassionally happens)
        dotproduct = scipy.linalg.det(Sa.dot(SopInv))
        if(dotproduct > 0):
            sic = 0.5 * np.log(dotproduct)
        else:
            sic = 0
        
        # Compute vertical resolution and cumulative degrees of freedom
        vres,cdfs = Other_functions.compute_vres_from_akern(Akern, z, do_cdfs=True)
        
        # Now do the same thing for profiles assuming there was no model data
        vres_nm, cdfs_nm = Other_functions.compute_vres_from_akern(Akern_nm, z, do_cdfs=True)
        
        # Compute the N-form and M-form convergence criteria (X and Y spaces, resp)
        if itern == 0:
        # Set the initial RMS and di2 values to large numbers

            old_rmsa = 1e20          # RMS for all observations
            old_rmsr = 1e20          # RMS for only the IRS and MWR radiance obs
            old_di2m = 1e20          # di-squared number

        di2n = ((Xn[:,None]-Xnp1).T.dot(SopInv).dot(Xn[:,None]-Xnp1))[0,0]
        if len(Fxnm1) == nY:
            di2m = ((FXn[:,None] - Fxnm1[:,None]).T.dot(
                scipy.linalg.pinv(Kij.dot(Sop).dot(Kij.T)+Sm)).dot(
                FXn[:,None] - Fxnm1[:,None]))[0,0]
        else:
            di2m = 9.0e9

        # Perform the RH_limit test (i.e., make sure thew WVMR Is not too large
        # such that RH > 100%)
        if ((itern == 0) & (verbose >= 3)):
            print('Testing for RH > 100%')
        rh = Calcs_Conversions.w2rh(np.squeeze(Xnp1[int(nX/2):nX]), p, np.squeeze(Xnp1[0:int(nX/2)]),0) * 100   # units are %RH
        feh = np.where(rh > 100)[0]
        if len(feh) > 0:
            if verbose >= 3:
                print('RH is above 100% somewhere in this profile -- setting it to 100%')
            rh[feh] = 100.
            Xnp1[int(nX/2):nX,0] = Calcs_Conversions.rh2w(np.squeeze(Xnp1[0:int(nX/2)]), rh/100., p)

        # Perform the monotonically ascending potential temperature test (i.e
        # make sure that theta never decreases with height)
        if ((itern == 0) & (verbose >= 3)):
            print('Testing for decreasing theta with height')

        # Multiply WVMR by zero to get theta, not theta-v
        theta = Calcs_Conversions.t2theta(np.squeeze(Xnp1[0:int(nX/2)]), 0*np.squeeze(Xnp1[int(nX/2):nX]), p)

        # This creates the maximum theta
        for ii in range(len(theta))[1:]:
            if ((theta[ii] <= theta[ii-1]) & (z[ii] > vip['superadiabatic_maxht'])):
                theta[ii] = theta[ii-1]

        # Multiply WVMR by zero to work with theta, not theta-v
        Xnp1[0:int(nX/2),0] = Calcs_Conversions.theta2t(theta, 0*np.squeeze(Xnp1[int(nX/2):nX]), p)

        # Make sure we don't get any nonphysical values here that would
        # make the next iteration of the LBLRTM croak
        multiplier = 5.

        # First check the water vapor profile
        feh = np.arange(int(nX/2)) + int(nX/2)
        
        # Check for values that are too low
        foo = np.where((Xnp1[feh,0] < minQ) | (Xnp1[feh,0] < Xa[feh] - multiplier*np.sqrt((np.diag(Sa)[feh]))))[0]
        if len(foo) > 0:
            # First check to make sure the entire profile isn't nonphysical
            if len(foo) == len(z):
                print('The entire water vapor profile is non-physical. Major error, must abort')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()
            # A nonphysical water vapor value exists so we are going interpolate across those values
            # by calling this function
            Xnp1[feh,0] = Other_functions.fix_nonphysical_wv(Xnp1[feh,0],z,Xa[feh],foo)
        
        # Check for values that are too high
        foo = np.where((Xnp1[feh,0] > maxQ) | (Xnp1[feh,0] > Xa[feh] + multiplier*np.sqrt((np.diag(Sa)[feh]))))[0]
        if len(foo) > 0:
            # First check to make sure the entire profile isn't nonphysical
            if len(foo) == len(z):
                print('The entire water vapor profile is non-physical. Major error, must abort')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()
            # A nonphysical water vapor value exists so we are going interpolate across those values
            # by calling this function
            Xnp1[feh,0] = Other_functions.fix_nonphysical_wv(Xnp1[feh,0],z,Xa[feh],foo)

        # Now test other variables in the retrieved state vector
        if dolcloud == 1:
            Xnp1[nX,0] = np.nanmax([Xnp1[nX],0])
            Xnp1[nX+1,0] = np.nanmax([Xnp1[nX+1], vip['prior_lReff_mn']-multiplier*vip['prior_lReff_sd'], minLReff])
            Xnp1[nX+1,0] = np.nanmin([Xnp1[nX+1], vip['prior_lReff_mn']+multiplier*vip['prior_lReff_sd'], maxLReff-2])

        if doicloud == 1:
            Xnp1[nX+2,0] = np.nanmax([Xnp1[nX+2],0])
            Xnp1[nX+3,0] = np.nanmax([Xnp1[nX+3], vip['prior_iReff_mn']-multiplier*vip['prior_iReff_sd'], miniReff])
            Xnp1[nX+3,0] = np.nanmin([Xnp1[nX+3], vip['prior_iReff_mn']+multiplier*vip['prior_iReff_sd'], maxiReff-2])

        if doco2 > 0:
            Xnp1[nX+4,0] = np.nanmax([Xnp1[nX+4], vip['prior_co2_mn'][0]-multiplier*vip['prior_co2_sd'][0], 0])
            Xnp1[nX+4,0] = np.nanmin([Xnp1[nX+4], vip['prior_co2_mn'][0]+multiplier*vip['prior_co2_sd'][0]])

            Xnp1[nX+5,0] = np.nanmax([Xnp1[nX+5], vip['prior_co2_mn'][1]-multiplier*vip['prior_co2_sd'][1]])
            Xnp1[nX+5,0] = np.nanmin([Xnp1[nX+5], vip['prior_co2_mn'][1]+multiplier*vip['prior_co2_sd'][1]])

            if Xnp1[nX+4] + Xnp1[nX+5]  < 0:
                Xnp1[nX+5] = -Xnp1[nX+4]

            if doco2 == 2:
                Xnp1[nX+6,0] = np.nanmax([Xnp1[nX+6], vip['prior_co2_mn'][2] - multiplier*vip['prior_co2_sd'][2], -20])
                Xnp1[nX+6,0] = np.nanmin([Xnp1[nX+6], vip['prior_co2_mn'][2] + multiplier*vip['prior_co2_sd'][2], -1])
            else:
                if Xnp1[nX+6] < vip['min_PBL_height']:
                    Xnp1[nX+6,0] = vip['min_PBL_height']
                if Xnp1[nX+6] > vip['max_PBL_height']:
                    Xnp1[nX+6,0] = vip['max_PBL_height']

        if doch4 > 0:
            Xnp1[nX+7,0] = np.nanmax([Xnp1[nX+7], vip['prior_ch4_mn'][0] - multiplier*vip['prior_ch4_sd'][0], 0])
            Xnp1[nX+7,0] = np.nanmin([Xnp1[nX+7], vip['prior_ch4_mn'][0] + multiplier*vip['prior_ch4_sd'][0]])

            Xnp1[nX+8,0] = np.nanmax([Xnp1[nX+8], vip['prior_ch4_mn'][1]-multiplier*vip['prior_ch4_sd'][1]])
            Xnp1[nX+8,0] = np.nanmin([Xnp1[nX+8], vip['prior_ch4_mn'][1]+multiplier*vip['prior_ch4_sd'][1]])

            if Xnp1[nX+7] + Xnp1[nX+8]  < 0:
                Xnp1[nX+8,0] = -Xnp1[nX+7]

            if doch4 == 2:
                Xnp1[nX+9,0] = np.nanmax([Xnp1[nX+9], vip['prior_ch4_mn'][2] - multiplier*vip['prior_ch4_sd'][2], -20])
                Xnp1[nX+9,0] = np.nanmin([Xnp1[nX+9], vip['prior_ch4_mn'][2] + multiplier*vip['prior_ch4_sd'][2], -1])

            else:
                if Xnp1[nX+9] < vip['min_PBL_height']:
                    Xnp1[nX+9,0] = vip['min_PBL_height']
                if Xnp1[nX+9] > vip['max_PBL_height']:
                    Xnp1[nX+9,0] = vip['max_PBL_height']

        if don2o > 0:
            Xnp1[nX+10,0] = np.nanmax([Xnp1[nX+10], vip['prior_n2o_mn'][0] - multiplier*vip['prior_n2o_sd'][0], 0])
            Xnp1[nX+10,0] = np.nanmin([Xnp1[nX+10], vip['prior_n2o_mn'][0] + multiplier*vip['prior_n2o_sd'][0]])

            Xnp1[nX+11,0] = np.nanmax([Xnp1[nX+11], vip['prior_n2o_mn'][1]-multiplier*vip['prior_n2o_sd'][1]])
            Xnp1[nX+11,0] = np.nanmin([Xnp1[nX+11], vip['prior_n2o_mn'][1]+multiplier*vip['prior_n2o_sd'][1]])

            if Xnp1[nX+10] + Xnp1[nX+11]  < 0:
                Xnp1[nX+11,0] = -Xnp1[nX+10]

            if don2o == 2:
                Xnp1[nX+12,0] = np.nanmax([Xnp1[nX+12], vip['prior_n2o_mn'][2] - multiplier*vip['prior_n2o_sd'][2], -20])
                Xnp1[nX+12,0] = np.nanmin([Xnp1[nX+12], vip['prior_n2o_mn'][2] + multiplier*vip['prior_n2o_sd'][2], -1])

            else:
                if Xnp1[nX+12] < vip['min_PBL_height']:
                    Xnp1[nX+12] = vip['min_PBL_height']
                if Xnp1[nX+12,0] > vip['max_PBL_height']:
                    Xnp1[nX+12,0] = vip['max_PBL_height']

        # Compute the RMS difference between the observation and the
        # forward calculation. However, this will be the relative RMS
        # difference (normalizing by the observation error here), because I
        # am mixing units from all of the different types of observation
        # But I will also compute the chi-square value of the obs vs. F(Xn)

        chi2 = np.sqrt(np.sum(((Y - FXn)/ Y)**2) / float(nY))
        rmsa = np.sqrt(np.sum(((Y - FXn)/sigY)**2) / float(nY))
        feh = np.where((flagY == 1) | (flagY == 2) & (Y > -900))[0]
        if len(feh) > 0:
            rmsr = np.sqrt(np.sum(((Y[feh] - FXn[feh])/sigY[feh])**2) / float(len(feh)))
        else:
            rmsr = -999.

        # I decided to just change the metric to look at the normalized
        # distance to the climatological prior, but I will let it have either
        # positive or negative values. ONly compute this for the Tq part though

        feh = np.arange(nX)
        rmsp = np.mean( (Xa[feh] - Xn[feh])/sig_Xa[feh] )

        ########################
        # Add doplot code here #
        ########################

        if doplot:
            if itern == 0:
                xx = np.array([X0.copy()])
                fxx = np.array([Y.copy()])
                frms = np.array([999.])


            xx = np.append(xx, np.array([Xn.copy()]), axis=0)
            fxx = np.append(fxx, np.array([FXn.copy()]), axis=0)
            frms = np.append(frms, rmsp)

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
            fig.set_figheight(10)
            fig.set_figwidth(10)
            ylim=(0, 3)
            # Temperature plot
            ax1.plot(xx[0, 0:int(nX/2)], z, label='prior')
            for ii in range(1, xx.shape[0]):
                ax1.scatter(xx[ii, 0:int(nX/2)], z, color=f"C{ii}", marker='x', label=f'iter{ii}')
            ax1.plot(Xnp1[0:int(nX/2)], z, linewidth=2, color='k')
            ax1.set_ylim(ylim)
            ax1.set_xlim(-30, 40)
            ax1.set_xlabel("Temperature")
            ax1.set_ylabel("Altitude (km AGL)")
            ax1.legend()

            ax2.plot(xx[0, int(nX/2):nX], z)
            for ii in range(1, xx.shape[0]):
                ax2.scatter(xx[ii, int(nX/2):nX], z, color=f"C{ii}", marker='x')
            ax2.plot(Xnp1[int(nX/2):nX], z, linewidth=2, color='k')
            ax2.set_ylim(ylim)
            ax2.set_xlim(0, maxQ)
            ax2.set_xlabel("WVMR")
            ax2.set_ylabel("Altitude (km AGL)")

            ax3.hlines(0, 0, len(Y), color='k')
            for ii in range(1, xx.shape[0]):
                ax3.plot(Y-fxx[ii], color=f"C{ii}")
            ax3.set_ylim(-10, 10)
            ax3.set_xlabel('Radiance Index')
            ax3.set_ylabel('Radiance Diff (RU)')

            plt.tight_layout()
            #ax1.plot(xx[])
            plt.savefig(f'{vip["output_path"]}/sample_{fsample}_temp_output_{itern}.png')
            plt.close()
        # Capture the iteration with the best RMS value
        if rmsa <= old_rmsa:
            old_rmsa = rmsa
            old_rmsr = rmsr
            old_iter = itern

        # Check for NaNs in the next iteration. If they exist, then
        # use the last valid sample as the solution and exit.
        foo = np.where(np.isnan(Xnp1))[0]
        if len(foo) > 0:
            print('Warning: Found NaNs in the next iteration -- using last iter')
            if itern == 0:
                print('Wow -- I never thought this could happen')
                VIP_Databases_functions.abort(lbltmpdir,date)
                sys.exit()

            converged = 9                             # Converged in "bad NaN sense
            Xn = np.copy(xsamp[itern-1]['Xn'])
            FXn = np.copy(xsamp[itern-1]['FXn'])
            Sop = np.copy(xsamp[itern-1]['Sop'])
            K = np.copy(xsamp[itern-1]['K'])
            Gain = np.copy(xsamp[itern-1]['Gain'])
            Akern = np.copy(xsamp[itern-1]['Akern'])
            Akern_nm = np.copy(xsamp[itern-1]['Akern_nm'])
            vres = np.copy(xsamp[itern-1]['vres'])
            vres_nm = np.copy(xsamp[itern-1]['vres_nm'])
            gfac = xsamp[itern-1]['gamma']
            sic = xsamp[itern-1]['sic']
            dfs = np.copy(xsamp[itern-1]['dfs'])
            dfs_nm = np.copy(xsamp[itern-1]['dfs_nm'])
            cdfs = np.copy(xsamp[itern-1]['cdfs'])
            cdfs_nm = np.copy(xsamp[itern-1]['cdfs_nm'])
            di2m = xsamp[itern-1]['di2m']
            rmsa = xsamp[itern-1]['rmsa']
            rmsr = xsamp[itern-1]['rmsr']
            rmsp = xsamp[itern-1]['rmsp']
            chi2 = xsamp[itern-1]['chi2']
            cbh_mlev = xsamp[itern-1]['cbh_mlev']
            cbh_tcld = xsamp[itern-1]['cbh_tcld']
            cld_emis = xsamp[itern-1]['cld_emis']
            itern = -1

        elif itern > 1:
            # Test for "convergence by looking at the best RMS value
            if ((rmsa > np.sqrt(gfactor[old_iter])*old_rmsa) & (old_iter > 0)):
                converged = 2                   # Converged in "rms increased drastically" sense

                Xn = np.copy(xsamp[old_iter]['Xn'])
                FXn = np.copy(xsamp[old_iter]['FXn'])
                Sop = np.copy(xsamp[old_iter]['Sop'])
                K = np.copy(xsamp[old_iter]['K'])
                Gain = np.copy(xsamp[old_iter]['Gain'])
                Akern = np.copy(xsamp[old_iter]['Akern'])
                Akern_nm = np.copy(xsamp[old_iter]['Akern_nm'])
                vres = np.copy(xsamp[old_iter]['vres'])
                vres_nm = np.copy(xsamp[old_iter]['vres_nm'])
                gfac = xsamp[old_iter]['gamma']
                sic = xsamp[old_iter]['sic']
                dfs = np.copy(xsamp[old_iter]['dfs'])
                cdfs = np.copy(xsamp[old_iter]['cdfs'])
                dfs_nm = np.copy(xsamp[old_iter]['dfs_nm'])
                cdfs_nm = np.copy(xsamp[old_iter]['cdfs_nm'])
                di2m = xsamp[old_iter]['di2m']
                rmsa = xsamp[old_iter]['rmsa']
                rmsr = xsamp[old_iter]['rmsr']
                rmsp = xsamp[old_iter]['rmsp']
                chi2 = xsamp[old_iter]['chi2']
                cbh_mlev = xsamp[old_iter]['cbh_mlev']
                cbh_tcld = xsamp[old_iter]['cbh_tcld']
                cld_emis = xsamp[old_iter]['cld_emis']
                itern = old_iter

            # But also check for convergence in the normal manner
            if ((gfactor[itern-1] <= 1) & (gfactor[itern] == 1)):
                if di2m < vip['cvgmult'] * nY:
                    converged = 1                    # Converged in "classical sense"

        prev_di2m = di2m

        # Place the data into a structure (before we do the update)
        xtmp = {'idx':i, 'secs':irs['secs'][i], 'ymd':irs['ymd'][i], 'hour':irs['hour'][i],
                'nX':nX, 'nY':nY, 'dimY':np.copy(dimY), 'Y':np.copy(Y), 'sigY':np.copy(sigY), 'flagY':np.copy(flagY),
                'niter':itern, 'z':np.copy(z), 'p':np.copy(p), 'hatchopen':irs['hatchopen'][i],
                'cbh':cbh, 'cbhflag':cbhflag, 'cbh_mlev':cbh_mlev, 'cbh_tcld':cbh_tcld, 'cld_emis':cld_emis,
                'X0':np.copy(X0), 'Xn':np.copy(Xn), 'FXn':np.copy(FXn), 'Sop':np.copy(Sop),
                'vres_nm':np.copy(vres_nm), 'K':np.copy(Kij), 'Gain':np.copy(Gain), 'Akern':np.copy(Akern), 'Akern_nm':np.copy(Akern_nm), 'vres':np.copy(vres),
                'gamma':gfac, 'qcflag':0, 'sic':sic, 'dfs':np.copy(dfs), 'dfs_nm':np.copy(dfs_nm),
                'cdfs':np.copy(cdfs), 'cdfs_nm':np.copy(cdfs_nm), 'di2m':di2m, 'rmsa':rmsa, 'rmsr':rmsr, 'rmsp':rmsp,
                'chi2':chi2, 'converged':converged}

        # Update the state vector, if we need to do another iteration
        if converged == 0:
            if verbose >= 1:
                print(f"    iter is {itern:2d}, di2m is {di2m:.3e}, and RMS is {rmsa:.3e}")
            Xn = np.copy(Xnp1[:,0])
            Fxnm1 = np.copy(FXn)
            itern += 1

        # And store each iteration in case I would like to investigate how
        # the retrieval functioned in a sample-by-sample way

        if itern == 1:
            xsamp = [copy.deepcopy(xtmp)]
        else:
            xsamp.append(copy.deepcopy(xtmp))

    if continue_next_sample == 1:
        continue         # This was set if the Jacobian could not be computed

    # If the retrieval converged, then let's store the various
    # pieces of information for later. Otherwise, let's just move on...
    if converged == 1:
        print('    Converged! (di2m << nY)')
    elif converged == 2:
        print('    Converged (best RMS as RMS drastically increased)')
    elif converged == 9:
        print('    Did not converge (found NaN in Xnp1 so abort sample)')
    else:

        # If the retrieval did not converged but performed max_iter iterations
        # means that the RMS didn't really increase drastically at any one step.
        # Let's select the sample that has the best RMS but weight the value
        # so that we are picking it towards the end ofthe iterations (use gamma
        # to do so), and save it

        vval = []
        for samp in xsamp:
            vval.append(np.sqrt(samp['gamma']) * samp['rmsa'])
        vval = np.array(vval)
        foo = np.where(vval < np.min(vval)*1.00001)[0]
        converged = 3           # Converged in "best rms after max_iter" sense
        itern = int(foo[0])
        Xn = np.copy(xsamp[itern]['Xn'])
        FXn = np.copy(xsamp[itern]['FXn'])
        Sop = np.copy(xsamp[itern]['Sop'])
        K = np.copy(xsamp[itern]['K'])
        Gain = np.copy(xsamp[itern]['Gain'])
        Akern = np.copy(xsamp[itern]['Akern'])
        Akern_nm = np.copy(xsamp[itern]['Akern_nm'])
        vres = np.copy(xsamp[itern]['vres'])
        vres_nm = np.copy(xsamp[itern]['vres_nm'])
        gfac = xsamp[itern]['gamma']
        sic = xsamp[itern]['sic']
        dfs = np.copy(xsamp[itern]['dfs'])
        dfs_nm = np.copy(xsamp[itern]['dfs_nm'])
        cdfs = np.copy(xsamp[itern]['cdfs'])
        cdfs_nm = np.copy(xsamp[itern]['cdfs_nm'])
        di2m = xsamp[itern]['di2m']
        rmsa = xsamp[itern]['rmsa']
        rmsr = xsamp[itern]['rmsr']
        rmsp = xsamp[itern]['rmsp']
        chi2 = xsamp[itern]['chi2']
        cbh_mlev = xsamp[itern]['cbh_mlev']
        cbh_tcld = xsamp[itern]['cbh_tcld']
        cld_emis = xsamp[itern]['cld_emis']
        xtmp = {'idx':i, 'secs':irs['secs'][i], 'ymd':irs['ymd'][i], 'hour':irs['hour'][i],
                'nX':nX, 'nY':nY, 'dimY':np.copy(dimY), 'Y':np.copy(Y), 'sigY':np.copy(sigY), 'flagY':np.copy(flagY),
                'niter':itern, 'z':np.copy(z), 'p':np.copy(p), 'hatchopen':irs['hatchopen'][i],
                'cbh':cbh, 'cbhflag':cbhflag, 'cbh_mlev':cbh_mlev, 'cbh_tcld':cbh_tcld, 'cld_emis':cld_emis,
                'X0':np.copy(X0), 'Xn':np.copy(Xn), 'FXn':np.copy(FXn), 'Sop':np.copy(Sop),
                'vres_nm':np.copy(vres_nm), 'K':np.copy(Kij), 'Gain':np.copy(Gain), 'Akern':np.copy(Akern), 'Akern_nm':np.copy(Akern_nm), 'vres':np.copy(vres),
                'gamma':gfac, 'qcflag':0, 'sic':sic, 'dfs':np.copy(dfs), 'dfs_nm':np.copy(dfs_nm),
                'cdfs':np.copy(cdfs), 'cdfs_nm':np.copy(cdfs_nm), 'di2m':di2m, 'rmsa':rmsa, 'rmsr':rmsr, 'rmsp':rmsp,
                'chi2':chi2, 'converged':converged}

        xsamp.append(xtmp)
        print('    Converged! (best RMS after max_iter)')

    # Store the data, regardless whether it converges or not
    if xret == []:
        xret = [copy.deepcopy(xsamp[len(xsamp)-1])]
    else:
        xret.append(copy.deepcopy(xsamp[len(xsamp)-1]))

    # Update the in_tropoe structure, if conditions warrant
    # Be sure to keep the logic to identify samples that go into in_tropoe the same as in Output_Functions.py
    if((gfac < vip['add_tropoe_gamma_threshold']) &
            ((cbh > vip['add_tropoe_input_cbh_thres']) | (Xn[nX] < vip['add_tropoe_input_lwp_thres']))):
        if((vip['add_tropoe_T_input_flag'] == 1) | (vip['add_tropoe_q_input_flag'] == 1) | (vip['add_tropoe_co2_input_flag'] == 1)):
            if verbose >= 1:
                print('    Adding retrieved profile to the "in_tropoe" structure, to be used as input later')
            sig_ret   = np.zeros(nX+13)
            for ii in range(nX+13):
                sig_ret[ii] = np.sqrt(Sop[ii,ii])
            in_tropoe = {'secs':irs['secs'][i], 'height':z, 'pblh':pblh, 'lwp':Xn[nX], 
                           'temperature':np.copy(Xn[0:int(nX/2)]), 'waterVapor':np.copy(Xn[int(nX/2):nX]), 
                           'sigma_temperature':np.copy(sig_ret[0:int(nX/2)]), 'sigma_waterVapor':np.copy(sig_ret[int(nX/2):nX]), 
                           'co2':np.copy(Xn[nX+4+np.arange(3)]), 'sigma_co2':np.copy(sig_ret[nX+4+np.arange(3)])}

    # Note that I will also save all of the iterations for the
    # last time sample that was processed (i.e., "xsamp")
    if version == '':
        try:
            version = globatt['algorithm_code_version']
        except:
            print('Did not find globatt version')
            version = 'Unknown Version'

    endtime = datetime.now()

    # Now store all of the data out
    ################################
    # IDL SAVE FILE CODE GOES HERE #
    ################################
    if debug == 1:
        import pickle
        # Create the .IDL file only if DEBUG is set
        # (Main reason is to get the xsamp data that show each iteration)
        dt = datetime.utcfromtimestamp(irs['secs'][i])
        hr = int(shour)*100+int(((shour-int(shour))*60+.5))
        savename = dt.strftime(f"{vip['output_path']}/{vip['output_rootname']}.%Y%m%d.{hr}.%H%M.pkl")
        out = {'xret': xret, 'xsamp': xsamp, 'bands': bands, 'vip':vip,
               'doco2':doco2, 'doch4': doch4, 'don2o':don2o,
               'dolcloud':dolcloud,'Sa':Sa, 'Xa':Xa, 'nsonde_prior':nsonde_prior,
               'comment_prior':comment_prior, 'irs':irs, 'shour':shour,'ehour':ehour,
               'starttime':starttime, 'endtime':endtime, 'version':__version__}
        with open(savename, 'wb') as fh:
            pickle.dump(out, fh)


    # Determine the QC of the sample
    # First look for a hatch that is isn't fully open
    # if ((0.8 > xret[fsample]['hatchopen']) & (xret[fsample]['hatchopen'] > 1.2)):
    #     xret[fsample]['qcflag'] = 1
    # Then look for a retrieval that didn't converge
    if (xret[fsample]['converged'] != 1) & (xret[fsample]['converged'] != 2):
        xret[fsample]['qcflag'] = 2
    # Then look for a retrieval where the RMS is too large
    if xret[fsample]['rmsa'] > vip['qc_rms_value']:
        xret[fsample]['qcflag'] = 3
    # Then look for a retrieval where gamma is too large
    if xret[fsample]['gamma'] > vip['qc_gamma_value']:
        xret[fsample]['qcflag'] = 4

    # Compute the various convective indices and other useful data.
    derived = {}
    if vip['omb_flag'] != 1:
        derived['theta'] = Calcs_Conversions.t2theta(xret[-1]['Xn'][0:int(nX/2)], 0*xret[-1]['Xn'][int(nX/2):nX], xret[-1]['p'])
        derived['thetae'] = Calcs_Conversions.t2thetae(xret[-1]['Xn'][0:int(nX/2)], xret[-1]['Xn'][int(nX/2):nX], xret[-1]['p'])
        derived['rh'] = Calcs_Conversions.w2rh(xret[-1]['Xn'][int(nX/2):nX], xret[-1]['p'], xret[-1]['Xn'][0:int(nX/2)],0) * 100
        derived['dewpt'] = Calcs_Conversions.rh2dpt(xret[-1]['Xn'][0:int(nX/2)], derived['rh']/100.)
        derived['co2_profile'] = Other_functions.trace_gas_prof(doco2, z, Xn[range(nX+4,nX+7)])
        derived['ch4_profile'] = Other_functions.trace_gas_prof(doch4, z, Xn[range(nX+7,nX+10)])
        derived['n2o_profile'] = Other_functions.trace_gas_prof(don2o, z, Xn[range(nX+10,nX+13)])
    
        dindices = Other_functions.calc_derived_indices(xret[-1],vip,derived,verbose,1)
    else:
        print("     Skipping the computation of derived fields/indices because code is in OMB mode -- filling with missing values")
        missing = np.zeros(len(z))-999.
        derived['theta']       = missing
        derived['thetae']      = missing
        derived['rh']          = missing
        derived['dewpt']       = missing
        derived['co2_profile'] = missing
        derived['ch4_profile'] = missing
        derived['n2o_profile'] = missing
        dindices = Other_functions.calc_derived_indices(xret[-1],vip,derived,verbose,0)

    # Write the data into the netCDF file
    success, noutfilename = Output_Functions.write_output(vip, ext_prof, mod_prof, ext_tseries,
              globatt, xret, prior, fsample, derived, dindices, version, (endtime-starttime).total_seconds(),
              modeflag, noutfilename, location, cbh_string, shour, verbose)

    if success == 0:
        VIP_Databases_functions.abort(lbltmpdir,date)
        sys.exit()

    if vip['plot_output'] == 1:
        print("     Plotting sample...")
        plot_tropoe.doplot(noutfilename, vip['plot_xlim'], vip['plot_ylim'], vip['plot_temp_lim'], vip['plot_wvmr_lim'],
                           vip['plot_tuncert_lim'], vip['plot_wvuncert_lim'], vip['plot_theta_lim'], vip['plot_rh_lim'],
                           vip['plot_thetae_lim'], vip['plot_dewpt_lim'], vip['plot_lwp_cbh_threshold'],
                           vip['plot_tres_min_gap'], vip['plot_comment'], vip['plot_rootname'], vip['plot_path'])

    already_saved = 1
    fsample += 1




shutil.rmtree(lbltmp)

totaltime = (endtime - starttime).total_seconds()

print(('  Processing took ' + str(totaltime) + ' seconds'))

shutil.rmtree(lbltmpdir)

# Successful exit
print(('>>> TROPoe retrieval on ' + str(date) + ' ended properly <<<'))
print('--------------------------------------------------------------------')
print(' ')
