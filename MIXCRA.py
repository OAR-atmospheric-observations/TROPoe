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
import sys
import numpy as np
import shutil
import scipy.io
import warnings
import glob
from netCDF4 import Dataset
from datetime import datetime, timezone
from time import gmtime, strftime
from subprocess import Popen, PIPE
from argparse import ArgumentParser
from pysolar import solar

import Other_functions
import VIP_Databases_functions
import Calcs_Conversions
import Data_reads
import Jacobian_Functions
import Output_Functions
import LBLRTM_Functions
import version

# Check to see if we are just writing out a blank vip
if '--vip' in sys.argv:
    # Write out a default vip file
    print("Writing default vip file to console")
    if '--experimental' in sys.argv:
        Output_Functions.write_example_vip_file('MIXCRA',console=True, experimental=True)
    else:
        Output_Functions.write_example_vip_file('MIXCRA',console=True)
    sys.exit()

# Create parser for command line arguments
parser = ArgumentParser()

parser.add_argument("date", type=int, help="Date to run the code [YYYYMMDD]")
parser.add_argument("vip_filename", help="Name of the VIP file (string)")
parser.add_argument("--step", type=int, help="The step size for MIXCRA")
parser.add_argument("--shour", type=float, help="Start hour (decimal, 0-24)")
parser.add_argument("--ehour", type=float, help="End hour (decimal, 0-24) [If ehour<0 process up to last IRS sample]")
parser.add_argument("--verbose", type=int, choices=[0,1,2,3], help="The verbosity of the output (0-very quiet, 3-noisy)")
parser.add_argument("--dostop", action="store_true", help="Set this to stop at the end before exiting")
parser.add_argument("--keep", action="store_true", help="Overrides vip[delete_temporary] to keep the LBLRTM runs for another go")

args = parser.parse_args()

date = args.date
vipfile = args.vip_filename
step = args.step
shour = args.shour
ehour = args.ehour
verbose = args.verbose
dostop = args.dostop
keep = args.keep

if shour is None:
    shour = 0.
if ehour is None:
    ehour = 24.
if verbose is None:
    verbose = 1
if keep is None:
    keep = 0
if step is None:
    step = 1

# Get the version of the TROPoe package and the software
tropoe_version   = Data_reads.get_tropoe_version()
software_version = version.get_software_version()

# Start the retrieval
print(' ')
print('-------------------------------------------------------------------------------------')
print('-----           MIXCRA is the mixed-phase cloud retrieval algorithm             -----')
print('-----It was developed by Dave.Turner, and ported to python by Josh.Gebauer      -----')
print('---- It is part of the TROPoe thermodynamic retrieval package developed at NOAA -----')
print('----       Contacts are dave.turner, joshua.gebauer, tyler.bell (@noaa.gov)     -----')
print('-------       The code is provided on an "as-is" basis, with no warranty       ------')
print(' ')
print('  TROPoe package version: '+tropoe_version)
print('  Software code version : '+software_version)

# We need the background shell to be the C-shell, as we will be spawning out
# a variety of commands that make this assumption. So we will do a
# quick check to find out if the C-shell exists on this system, and if so,
# set the SHELL to this.

if verbose == 3:
    print(' ')
    print(('The current shell is', os.getenv('SHELL')))
else:
    warnings.filterwarnings("ignore", category=UserWarning)

with Popen('which csh', stdout = PIPE, stderr = PIPE, shell=True) as process:
    try:
        stdout, stderr = process.communicate()
    except Exception as e:
        print('Error -- process did not return shell properly')
        sys.exit()

if stdout.decode() == '':
    print('Error: Unable to find the C-shell command on this system')
    print(('>>> TROPoe retrieval on ' + str(date) + ' FAILED and ABORTED <<<'))
    sys.exit()
else:
    SHELL = stdout[:-1].decode()

if verbose == 3:
    print(('The shell for all commands is', SHELL))

# Specify some global parameters
globatt = {'algorithm_code': 'MIXCRAv3 retrieval code, which is part of the TROPoe software package',
            'algorithm_author': 'Dave Turner, NOAA Global System Laboratory (dave.turner@noaa.gov), ' +
                     'Ported to Python by Josh Gebauer NOAA National Severe Storms Laboratory / CIWRO (joshua.gebauer@noaa.gov)',
            'algorithm_version': software_version,
            'algorithm_package_version': tropoe_version,
            'algorithm_reference1': 'DD Turner, 2005: Arctic mixed-phase cloud '+
                     'properties from AERI-lidar observations: Algorithm and results from SHEBA. '+
                     'J. Appl. Meteor., 44, 427-444, doi:10.1175/JAM2208.1',
            'algorithm_reference2': 'DD Turner and RE Holz, 2005: Retrieving '+
                     'cloud fraction in the field-of-view of a high-spectral-resolution infrared radiometer. '+
                     'IEEE Geosci. Remote Sens. Lett., 3, 287-291, doi:10.1109/LGRS.2005.850533',
            'algorithm_reference3': 'DD Turner, 2007: Improved ground-based '+
                     'liquid water path retrievals using a combined infrared and microwave approach. '+
                     'J. Geophys. Res., 112, D15204, doi:10.1029/2007JD008530',
            'algorithm_reference4': 'DD Turner and EW Eloranta 2008: Validating '+
                     'mixed-phase cloud optical depth retrieved from infrared observations with high '+
                     'spectral resolution lidar. IEEE Geosci. Remote Sens. Lett., 5, 285-288, doi:10.1109/LGRS.2008.915940',
            'datafile_created_on_date': strftime("%Y-%m-%d %H:%M:%S", gmtime()),
            'datafile_created_on_machine': os.uname()[-1]}

# Read in the VIP parameters
vip = VIP_Databases_functions.read_mixcra_vip_file(vipfile, globatt = globatt, verbose = verbose, dostop = dostop)
if vip['success'] != 1:
    print(('>>> MIXCRA retrieval on ' + str(date) + ' FAILED and ABORTED <<<'))
    print('---------------------------------------------------------------------')
    print(' ')
    sys.exit()
print(('  Working with the LBLRTM '))
print(('    in the directory ' + vip['lblrtm_home']))
print(('    and the TAPE3 file ' + vip['lbl_tape3']))

# Make sure that the reference wavenumber makes some sense
if (vip['ref_wnum'] <= 0):
    vip['ref_wnum'] = -1

# Determine which IRS spectra to process
if step is None:
    step = vip['irs_sampling']          # Process every (step)th spectrum
else:
    step = int(step)
if step <= 0:
    step = 1                            # Must process in a forward way
vip['irs_sampling'] = step             # Update the VIP file, if passed in via command line

# If the first spectral bands variable is negative, I will assume it was not in
# the VIP file and set it to some reasonable default value here
if vip['spectral_bands'][0,0] < 0:
    spectral_bands = np.array([[530.0,533.0],[558.0,562.0],[817.0,823.0],[898.5,904.7],[959.9,964.3]])
else:
    foo = np.where(vip['spectral_bands'][0,:] > 0)[0]
    spectral_bands = vip['spectral_bands'][:,foo]

# Confirm that the converge_factor is properly set
if vip['converge_factor'] <= 0 or vip['converge_factor'] > 1:
    print('Error: The converge_factor in the VIP file must be in (0,1]')
    sys.exit

# If the first surface emissivity variable is negative, I will assume it was not
# in the VIP file and set it to some reasonable default values here
if vip['sfc_emissivity'][0,0] < 0:
    sfc_emissivity = np.array([[200.,0.92],[500.,0.93],[650,0.95],[3000,0.95]])
else:
    sfc_emissivity = vip['sfc_emissivity']

# Apply some QC to the VIP parameters
if vip['solver'] != 0 and vip['solver'] != -1:
    print('Error: The solver in the VIP file must be either 0 or -1')
    sys.exit()

if any(spectral_bands[1]-spectral_bands[0] < 0):
    print('Error: The spectral bands must always have pairs of wavenumbers that are loWnum, hiWnum')
    sys.exit()

# Indicate what is being retrieved here
if vip['retrieve_lcloud'] == 0 and vip['retrieve_icloud'] == 0:
    print('Error: neither of the two retrievals are turned on -- aborting')
    sys.exit()

retr = ''
if vip['retrieve_lcloud'] == 1:
    retr = retr+'LiqCloud '
if vip['retrieve_icloud'] == 1:
    retr = retr+'IceCloud '
if verbose >= 1:
    print('  Retrieving these parameters: ' + retr)

# Capture the wavenumber range for the LBLRTM calculations, and ensure it can be made
lblwnumdel = 25
lblwnum1 = np.min(spectral_bands) - lblwnumdel
lblwnum2 = np.max(spectral_bands) + lblwnumdel
if lblwnum2 - lblwnum1 > 1999:
    print('Error: The range of wavenumbers in spectral_bands must be less than 2000 cm-1')
    sys.exit()

if verbose >=2:
    print(' The lblwnums are ' + str(lblwnum1) + ' ' + str(lblwnum2))

# Set a flag on whether we will use the solar source spectrum or not
if lblwnum2 < 1800:
    use_sun = 0
else:
    use_sun = 1

# Override the delete temporary in the VIP file, if desired
if keep:
    vip['delete_temporary'] = 0
    if verbose >= 1:
        print('  Override the delete_temporary in the VIP file to keep the working directory contents')

# Read in the TROPoe input file
if verbose >= 0:
    print('  Reading the TROPoe output file')
tropoe_filename = '*tropoe*.' + str(date) + '.*.(nc|cdf)'
files, status = Data_reads.findfile(vip['tropoe_path'],tropoe_filename)
if len(files) != 1:
    print('Error: Unable to find a unique TROPoe file for this date in ' + vip['tropoe_path'])
    print('         that matches this file pattern: '+tropoe_filename)
    sys.exit()
tropoe = Data_reads.read_tropoe(files[0],vip)

# Now read in the IRS and MWR data (if desired)
fail, irs, mwr, mwrscan = Data_reads.read_all_data(date, vip, dostop, verbose)

# Identify the good samples (i.e., apply QC to the IRS data)
qcflag = Other_functions.qc_irs(irs)
if qcflag[0] < -900:
    sys.exit()

# I will aslo define a "entire_spectrum" variables, that is similar to the above
# but only used at the end of the retrieval to capture the results across the entire spectrum
delw = 10
minw = 500
maxw = 1499

if lblwnum2 > maxw:
    maxw = lblwnum2+100
    if lblwnum1 >= lblwnum2:
        lblwnum1 = minw-100

if vip['include_full_calc'] > 0:
    delw = vip['include_full_calc']
    minw = np.max([lblwnum1,np.min(irs['wnum'])])
    maxw = np.min([lblwnum2,np.max(irs['wnum'])])

entire_spectrum = np.array([minw,minw+delw])
minw += delw
while np.max(entire_spectrum) < maxw:
    new_row = np.array([[minw, minw + delw]])
    entire_spectrum = np.vstack([entire_spectrum, new_row])
    minw += delw
entire_spectrum = entire_spectrum.T

# Now avereage the IRS data over the desired spectral bands
yobs1, ysig1, nyobs1 = Other_functions.average_irs(irs,spectral_bands,vip['avg_instant'])       # These are the obs for the retrieval
yobs2, ysig2, nyobs2 = Other_functions.average_irs(irs,entire_spectrum,vip['avg_instant'])      # This is only for the end

# Need to capture the latitude and longitude of the station
if vip['station_alt'] > 0:
    location = {'lat':vip['station_lat'], 'lon':vip['station_lon'],'alt':vip['station_alt']}
else:
    location = {'lat':irs['lat'], 'lon':irs['lon'],'alt':irs['alt']}

# Add the date being processed to the workdir entry; this allows multiple days to be processed simultaneous
vip['workdir'] = vip['workdir'] + '_' + str(date)

# If this flag was set then delete the working subdirectory, and restart from scratch
if vip['delete_temporary'] != 0:
    if verbose >= 1:
        print('    Reinitializing the working directory ' + vip['workdir'])
    if os.path.exists(vip['workdir']):
        shutil.rmtree(vip['workdir'])
            # Test to see if the current working directory is still there, as it should
            # have been deleted in the previous step.  If it is, then abort
    if os.path.exists(vip['workdir']):
        print(f"Error: working directory {vip['workdir']:s} previously exists and can not be deleted -- aborting")
        sys.exit()
    os.mkdir(vip['workdir'])
    
# This should be included in the VIP file. Right now it is always set.
if((vip['compute_lwp'] > 0) and (vip['mwr_type'] > 0) and (vip['retrieve_lcloud'] > 0)):
        # Set up the temporary files needed for the MonoRTM (these will be placed in vip['workdir'])
    monortm_config = 'monortm_config.txt'
    monortm_zfreq  = 'monortm_zfreqs.txt'    # For MWR-zenith calculations
    monortm_tfile  = 'monortm_sonde.cdf'
    create_monortm_config = 1           # Set this flag to create a custom config file for MonoRTM
    create_monortm_zfreq  = 1           # Set this flag to create a custom freq-zenith file for MonoRTM
    monortm_zexec = ('cd ' + vip['workdir'] + ' ; setenv monortm_config ' + monortm_config +
                ' ; setenv monortm_freqs ' + monortm_zfreq + ' ; ' + vip['monortm_wrapper'])

# Select the TROPoe indices to make LBLRTM runs, if we had to recreate the working directory
# This might be able to be cleaned up later
if verbose >= 1:
    print('  Run the LBLRTM')
lidx = [0]
lhour = tropoe['hour'][lidx]
cnt = 1
tres = max(vip['tres_lblrtm'],0.25)
while ((np.max(lhour) <= 23.8 ) and (cnt == 1)):
    foo = np.where(tropoe['hour'] >= np.max(lhour)+tres)[0]
    if len(foo) > 0:
        lidx.append(foo[0])
    else:
        cnt = 0
    lhour = tropoe['hour'][lidx]

if len(tropoe['hour']) > 1:
    if(tropoe['hour'][-1] > np.max(lhour)):
        lidx.append(len(tropoe['hour']) - 1)
        lhour = tropoe['hour'][lidx]

if verbose >= 1:
    # Format the lhour array as comma-separated values with 2 decimal places
    hour_str = ', '.join(f'{hour:5.2f}' for hour in lhour)
    print(f'    LBLRTM calculations will be made at these times: {hour_str} UTC')

for i in range(len(lhour)):
    # Make the LBLRTM calculation, and read in the ODs
    lbl = Jacobian_Functions.make_lblrtm_calc(vip,date,tropoe['hour'][lidx[i]],
                                              tropoe['co2'][lidx[i],0],tropoe['height'],
                                              tropoe['pressure'][lidx[i],:], tropoe['temperature'][lidx[i],:],
                                              tropoe['waterVapor'][lidx[i],:], tropoe['pwv'][lidx[i]], lblwnum1, lblwnum2,
                                              vip['delete_temporary'],verbose)
    
    if lbl['status'] == 0:
        print('Error: problem with the LBLRTM run -- aborting')
        sys.exit()
    
    if i == 0:
        lblout = [np.copy(lbl['filename'])]
    else:
        lblout.append(np.copy(lbl['filename']))
    
if len(lhour) == 1:
    lhour = [lhour[0]-0.5,lhour[0]+0.5]
    lblout = [lblout[0],lblout[0]]

# Create the microwindow file used by LBLDIS in this working directory
    # For the retrieval spectral bands
microwin_file1 = vip['workdir'] + '/microwindows1.txt'
if verbose >= 2:
    print('  Generating mixrowindow file for the retrieval bands')
lun = open(microwin_file1,'w')
lun.write(str(len(spectral_bands[0])) + '\n')
for j in range(len(spectral_bands[0])):
    lun.write('{:7.2f}  {:7.2f}\n'.format(spectral_bands[0,j],spectral_bands[1,j]))
lun.close()

# For the end-game "entire_spectrum"
microwin_file2 = vip['workdir'] + '/microwindows2.txt'
if verbose >= 3:
    print('  Generating microwindow file for the entire spectral band')
lun = open(microwin_file2,'w')
lun.write(str(len(entire_spectrum[0])) + '\n')
for j in range(len(entire_spectrum[0])):
    lun.write('{:7.2f}  {:7.2f}\n'.format(entire_spectrum[0,j],entire_spectrum[1,j]))
lun.close()

# Determine the ranges of the effective radii in the SSP DBs
     # Database #1 first
if os.path.exists(vip['lcloud_ssp']) == False:
    print('Error: Unable to uniquely determine the needed file ' + vip['lcloud_ssp'])
    sys.exit()
else:
    # Don't take the exact_min and max; give room for perturbation calcs
    sspl, flag = VIP_Databases_functions.read_scat_databases(vip['lcloud_ssp'])
    if flag == 1:
        print('Error reading in the SSP DB '+vip['lcloud_ssp'])
        sys.exit()
    min_lReff = sspl['data'][2,1]
    max_lReff = sspl['data'][2,-2]
    print('  The minimum and maximum effective radii in the SSP for ' \
    'lcloud is {:7.2f} and {:7.2f} microns'.format(min_lReff, max_lReff))
    min_lwnum = sspl['data'][1,1]
    max_lwnum = sspl['data'][1,-2]
    print('  The minimum and maximum wavenumber in the SSP for ' \
    'lcloud is {:7.1f} and {:7.1f} cm-1'.format(min_lwnum, max_lwnum))

     # Database #2 next
if os.path.exists(vip['icloud_ssp']) == False:
    print('Error: Unable to uniquely determine the needed file ' + vip['icloud_ssp'])
    sys.exit()
else:
    # Don't take the exact_min and max; give room for perturbation calcs
    sspi, flag = VIP_Databases_functions.read_scat_databases(vip['icloud_ssp'])
    if flag == 1:
        print('Error reading in the SSP DB '+vip['icloud_ssp'])
        sys.exit()
    min_iReff = sspl['data'][2,1]
    max_iReff = sspl['data'][2,-2]
    print('  The minimum and maximum effective radii in the SSP for ' \
    'icloud is {:7.2f} and {:7.2f} microns'.format(min_iReff, max_iReff))
    min_iwnum = sspi['data'][1,1]
    max_iwnum = sspi['data'][1,-2]
    print('  The minimum and maximum wavenumber in the SSP for ' \
    'icloud is {:7.1f} and {:7.1f} cm-1'.format(min_iwnum, max_iwnum))

# Select the LBLRTM version to use
print('  Working with the LBLRTM version in ' + vip['lblrtm_home'])

# Quick check: make sure the LBLRTM path is properly set
if not os.path.exists(vip['lblrtm_home'] + '/bin/lblrtm'):
    print('Error: lblhome is not properly set')
    sys.exit()

# Make sure that the specified TAPE3 file exists
if not os.path.exists(vip['lblrtm_home'] + '/hitran/' + vip['lbl_tape3']):
    print('Error: unable to find the specified TAPE3 file in the LBLRTM_HOME hitran directory')
    sys.exit()

# Make sure that path to the MonoRTM files are set properly. We will only
# stop the code if MWR data is used and the files don't exist
if not os.path.exists(vip['monortm_wrapper']) and ((vip['mwr_type'] > 0) or (vip['mwrscan_type'] > 0)):
    print('Error: unable to find the the specified MonoRTM wrapper')
    sys.exit()

if not os.path.exists(vip['monortm_exec']) and ((vip['mwr_type'] > 0) or (vip['mwrscan_type'] > 0)):
    print('Error: unable to find the the specified MonoRTM executable')
    sys.exit()

if not os.path.exists(vip['monortm_spec']) and ((vip['mwr_type'] > 0) or (vip['mwrscan_type'] > 0)):
    print('Error: unable to find the the specified MonoRTM spectral line file')
    sys.exit()

# We need to use an internal reference wavenumber that is within the spectral ranges 
# in the scattering databases.  I will assumed a desired reference wavenumber
# and check to make sure it is spanned by both SSP DBs
if(vip['ref_wnum'] > 0):
    d_ref_wnum = vip['ref_wnum']
    d_lQext_ratio = 1.0
    d_iQext_ratio = 1.0
else:
    d_ref_wnum    = 900.    # Desired reference wavenumber [cm-1]
if((vip['retrieve_lcloud'] != 0) and (vip['retrieve_icloud'] != 0)):
    if((d_ref_wnum < min_lwnum) or (d_ref_wnum < min_iwnum)):
        print('Error: the reference wavenumber "ref_wnum" is smaller than the smallest wavenumber in the SSP DBs')
        sys.exit()
    if((d_ref_wnum > max_lwnum) or (d_ref_wnum > max_iwnum)):
        print('Error: the reference wavenumber "ref_wnum" is larger than the largesest wavenumber in the SSP DBs')
        sys.exit()
elif(vip['retrieve_lcloud'] != 0):
    if(d_ref_wnum < min_lwnum):
        print('Error: the reference wavenumber "ref_wnum" is smaller than the smallest wavenumber in the liquid SSP DB')
        sys.exit()
    if(d_ref_wnum > max_lwnum):
        print('Error: the reference wavenumber "ref_wnum" is larger than the largesest wavenumber in the liquid SSP DB')
        sys.exit()
elif(vip['retrieve_icloud'] != 0):
    if(d_ref_wnum < min_iwnum):
        print('Error: the reference wavenumber "ref_wnum" is smaller than the smallest wavenumber in the ice SSP DB')
        sys.exit()
    if(d_ref_wnum > max_iwnum):
        print('Error: the reference wavenumber "ref_wnum" is larger than the largesest wavenumber in the ice SSP DB')
        sys.exit()
else:
    print('The VIP should have at least one of retrieve_lcloud or retrieve_icloud set to 1 - aborting')
    sys.exit()

# Build the prior and its uncertainty
minsd = 1e-5
if vip['retrieve_lcloud'] == 0:
    vip['prior_ltau_mn']  = 0.
    vip['prior_ltau_sd']  = minsd
    vip['prior_lreff_sd'] = minsd
    vip['min_ltau']       = 0.
if vip['retrieve_icloud'] == 0:
    vip['prior_itau_mn']  = 0.
    vip['prior_itau_sd']  = minsd
    vip['prior_ireff_sd'] = minsd
    vip['min_itau']       = 0.

# Compute the julian day for these data
jday = np.zeros(len(irs['hour']))
for i in range(len(irs['hour'])):
    dt = datetime.fromtimestamp(irs['secs'][i],tz=timezone.utc)
    jday[i] = dt.timetuple().tm_yday + irs['hour'][i]/24.

# If use sun == 0, then set all solar zenith angles to -1, else to the proper value
solzenang = np.ones(len(irs['hour']))*-1
if use_sun == 1:
    for i in range(len(irs['secs'])):
        dt = datetime.fromtimestamp(irs['secs'][i],tz=timezone.utc)
        sun_angle = solar.get_altitude(location['lat'], location['lon'], dt)
        solzenang[i] = 90-sun_angle

# Define the gamma array
gamm    = np.ones(vip['maxiter'])
sfactor = np.ones(vip['maxiter'])

# This bit is an attempt to help the code converge more smoothly, based upon the Rodger's equations used
if vip['math_choice'] == 0:
    # If first math_choice is 0, then I will tweak the gamma factors and use Eq. 5.9
    gamm[0] = 10
    gamm[1] = 3
elif vip['math_choice'] == 1:
    # If math_choice is 1, then use Eq. 5.8 with no modifications 
    foo = 0       # This does nothing as intended
elif vip['math_choice'] == 2:
    # If math_choice is 2, then use Eq 5.8 with annealing
    if((vip['annealing_step'] <= 0) | (1 < vip['annealing_step'])):
        print('Error -- VIP.annealing_step must be in (0,1] -- aborting')
        sys.exit()
else:
    print('Error -- undefined math_choice in the VIP file -- aborting')
    sys.exit()

if step > 1 and verbose >= 2:
    print('  Will run the retrieval on every ' + str(step) + ' sample')

# This defines the extra vertical layers that will be added to both
# the infrared and microwave radiative transfer calculations
rt_extra_layers = Other_functions.compute_extra_layers(np.max(tropoe['height']))

# This will be used to update the MWR jacobian more quickly
stored_mwr_jacobian = {'computed':0}

# Used to capture the spectral range in the TAPE3 used by the lblrtm
tape3_info = {'success':-2}

#### Main loop over the IRS samples
foo = np.where(irs['hour'] >= shour)[0]
if len(foo) <= 0:
    print('There were not IRS samples at or after this start time ' + str(shour) + ' UTC -- aborting')
    sys.exit()

fsample = 0
starttime = datetime.now()
if ehour <= shour:
    ehour = irs['hour'][foo[0]] + 1./3600        # To perform at least one retrieval

ofilename = ' '
for samp in range(foo[0],len(irs['secs']),step):

    # Only process the samples that exist in the desired window
    if irs['hour'][samp] < shour:
        continue
    if irs['hour'][samp] > ehour:
        continue
    if verbose >= 0:
        if qcflag[samp] == 1:
            print('Processing sample at {:6.3f} UTC'.format(irs['hour'][samp]))
        else:
            print('Skipping sample at {:6.3f} UTC -- did not pass QC'.format(irs['hour'][samp]))
            continue
    
    # Create the LWC profile that distributes the cloud vertically
    if vip['fix_cbh'] > 0:
        cbh = vip['fix_cbh']
    else:
        # Interpolate the CBH from the TROPoe retrievals in time, but don't extrapolate the CBH
        cbh = np.interp(irs['hour'][samp],tropoe['hour'],tropoe['cbh'])
        if irs['hour'][samp] < tropoe['hour'][0]:
            cbh = tropoe['cbh'][0]
        if irs['hour'][samp] > tropoe['hour'][-1]:
            cbh = tropoe['cbh'][-1]
        if cbh <= 0:
            cbh = np.abs(vip['fix_cbh'])

    if len(tropoe['hour']) == 1:
        lwc = Other_functions.create_lwc_profile(max(tropoe['lwp'],5),tropoe['height'], cbh, vip['lwc'])
    else:
        lwc = Other_functions.create_lwc_profile(max(np.interp(irs['hour'][samp],tropoe['hour'],tropoe['lwp']),5), tropoe['height'], cbh, vip['lwc'])

    # Get an estimate of the temperature of the cloud
    tcld = Other_functions.estimate_tcld(lwc,tropoe['temperature'],tropoe['hour'],irs['hour'][samp])

    # Define the a priori mean and covariance
    Xa  = np.array([vip['prior_ltau_mn'],vip['prior_lReff_mn'],vip['prior_itau_mn'],vip['prior_iReff_mn']])
    sig = np.array([vip['prior_ltau_sd'],vip['prior_lReff_sd'],vip['prior_itau_sd'],vip['prior_iReff_sd']])

    # If desired, then modify the liquid and ice optical depths based upon cloud temperature
    if((vip['apply_tcloud_constraints'] == 1) and (vip['retrieve_icloud'] == 1) and (vip['retrieve_lcloud'] == 1)):
        xtcld = np.arange(61)-50.
        xifac = 1-1./(1+np.exp(-(xtcld+ 5)))
        xlfac =   1./(1+np.exp(-(xtcld+35)))
        imult = np.min([np.max([np.interp(tcld,xtcld,xifac),0]),1])
        lmult = np.min([np.max([np.interp(tcld,xtcld,xlfac),0]),1])
        Xa[0]  *= lmult
        sig[0] *= lmult
        Xa[2]  *= imult
        sig[2] *= imult
    
    # Make sure that all of the uncertainties are above the minimum value allowed
    foo = np.where(sig < minsd)[0]
    if len(foo) > 0:
        sig[foo] = minsd
    Ca  = np.diag(np.ones(4))
    Ca[0,1] = vip['prior_corr_ltau_lreff']
    Ca[1,0] = Ca[0,1]
    Ca[0,2] = vip['prior_corr_ltau_itau']
    Ca[2,0] = Ca[0,2]
    Ca[0,3] = vip['prior_corr_ltau_ireff']
    Ca[3,0] = Ca[0,3]
    Ca[1,2] = vip['prior_corr_lreff_itau']
    Ca[2,1] = Ca[1,2]
    Ca[1,3] = vip['prior_corr_lreff_ireff']
    Ca[3,1] = Ca[1,3]
    Ca[2,3] = vip['prior_corr_itau_ireff']
    Ca[3,2] = Ca[2,3]
    Sa = np.copy(Ca)
    for i in range(len(sig)):
        for j in range(len(sig)):
            Sa[i,j] = Ca[i,j]*sig[i]*sig[j]
    invSa = np.linalg.pinv(Sa)

    # Quick QC on the prior's effective radii
    if vip['retrieve_lcloud'] > 0:
        Xa[1] = np.min([np.max([Xa[1],min_lReff]),max_lReff])
    if vip['retrieve_icloud'] > 0:
        Xa[3] = np.min([np.max([Xa[3],min_iReff]),max_iReff])

    # Build the observation vector and its covariance matrix
    dimY  = np.mean(spectral_bands,axis=0)
    flagY = np.ones(dimY.shape)
    Y     = np.squeeze(yobs1[:,samp])
    sigY  = np.squeeze(ysig1[:,samp])
    foo   = np.where(sigY > 0)[0]
    if len(foo) <= 0:
        print('Error: it seems that none of the IRS data were in the spectral bands -- aborting')
        sys.exit()
    minnoise = np.min(sigY[foo])
    foo = np.where(sigY <= 0)[0]
    if len(foo) > 0:
        sigY[foo] = minnoise
    sigY = sigY * vip['irs_noise_inflation']

    # Append the MWR observations to the obs vector, if desired
    if((mwr['n_fields'] > 0) and (vip['compute_lwp'] == 1) and (vip['retrieve_lcloud'] == 1) and (vip['ref_wnum'] < 0)):
        tbsky = np.copy(mwr['tbsky'][:,samp])
        noise = np.copy(mwr['noise'])
        freq  = np.copy(mwr['freq'])
        foo   = np.where(tbsky < 2.7)[0]                # Must be larger than the cosmic background
        if len(foo) > 0:
            tbsky[foo] = -999.
        Y     = np.append(Y,tbsky)
        sigY  = np.append(sigY,noise)
        flagY = np.append(flagY, np.ones(mwr['n_fields'])*2)
        dimY  = np.append(dimY,freq)

    # Now build the observational covariance matrix
    Sm    = np.diag(sigY**2)
    invSm = np.linalg.pinv(Sm)

    # Initialize the first guess to the prior, but update the ltau to be a bit more accurate
    Xn = np.copy(Xa)
    if((vip['retrieve_lcloud'] == 1) and (vip['compute_lwp'] == 1)):
        lwp = np.interp(irs['hour'][samp],tropoe['hour'],tropoe['lwp'])
        Xn[0] = Other_functions.compute_ltau(lwp,Xn[1])
    
    # Now iterate the retrieval
    FXnm1 = np.ones(len(Y))*999.
    di2m  = 9e14
    rms   = 999.
    itern = 0
    conv  = 0
    if verbose >= 1:
        print('   Iter     lTau   lReff    iTau   iReff       RMS         di2m  sfactor')
    
        # Select the correct TROPoe index run to use for this sample time
    delt = np.abs(tropoe['hour']-irs['hour'][samp])
    sidx = np.where(delt == np.min(delt))[0][0]
        # Select the correct LBLRTM run to use for this sample time
    delt = np.abs(lhour-irs['hour'][samp])
    tidx = np.where(delt == np.min(delt))[0][0]
    isza = solzenang[samp]

        # If the solar zenith angle is greater than 82 degrees, then the earth's curvature
        # (when sun is between 82-90 degrees) is too difficult to account for (Thomas and Stamnes)
        # or the sun is below the horizon -- so turn solar input off
    if isza >= 82:
        isza = -1
        
    # Set this to unity before we start the iterations (only used if vip['math_choice'] == 2)
    anneal_step = 1.0

    # Now perfrom the iterations
    while((itern < vip['maxiter']-1) and (conv == 0)):
        # Capture the total optical depth
        total_tau = Xn[0]+Xn[1]

        # Check to make sure the spectral range spanned by the TAPE3 is sufficient
        if(tape3_info['success'] < 0):
            tmplbldir = f'{lblout[tidx]:s}'
            tape3_info = Data_reads.tape6_min_max_wnum(tmplbldir,verbose=verbose)
            if(tape3_info['success'] <= 0):
                print('Error: Unable to determine the spectral range of the LBLRTMs TAPE3 -- (this should not happen)')
                sys.exit()
            if((lblwnum1 < tape3_info['minw']) or (tape3_info['maxw'] < lblwnum2)):
                print('Error: The tape3 selected in the VIP does not span the desired IRS calculation range')
                print(f"            TAPE3 spans from {tape3_info['minw']:.3f} to {tape3_info['maxw']:.3f} cm-1")
                print(f"            Needed LBLRTM range is {lblwnum1:.3f} to {lblwnum2:.3f} cm-1")
                sys.exit()
        # But also ensure that the spectral range used in the actual calculation is also sufficient
        # Note that I only need to do this once, which is why I've placed this test in with the tape3 test
            tmpfile = []
            tmpfile = tmpfile + sorted(glob.glob(tmplbldir+'/OD*'))
            tmps0, tmpv0 = LBLRTM_Functions.lbl_read(tmpfile[0], do_load_data=True)
            if((lblwnum1 < np.min(tmpv0)) or (np.max(tmpv0) < lblwnum2)):
                print('Error: The LBLRTM calc does not span the desired IRS calculation range')
                print(f"            Current LBLRTM calc spans from {np.min(tmpv0):.2f} to {np.max(tmpv0):.2f} cm-1")
                print(f"            Needed LBLRTM range is {lblwnum1:.2f} to {lblwnum2:.2f} cm-1")
                sys.exit()

        # Call the LBLDIS
        flag, FXn, Kij = Jacobian_Functions.mixcra_forward_model(Xn, tropoe['height'],
                        lblout[tidx], lwc, vip, jday[samp], isza,
                        sfc_emissivity,d_ref_wnum, nyobs1, microwin_file1,
                        vip['retrieve_lcloud'], vip['retrieve_icloud'], verbose)
        
        if flag == 0:
            print('problem with the forward model')
        
        # Perform the forward model calculation and compute the Jacobian for the
        # MWR-zenith portion of the observation vector.  Note there will only be
        # MWR data in the obs vector if the LWP retrieval is actually enabled
        foo = np.where(flagY == 2)[0]
        if len(foo) > 0:
            if create_monortm_config == 1:
                # Create the MonoRTM configuration file
                lun = open(vip['workdir'] + '/' + monortm_config, 'w')
                lun.write(vip['monortm_exec'] + '\n')
                lun.write(vip['monortm_spec'] + '\n')
                lun.write('0\n')          # The verbose flag
                lun.write('{:0d}\n'.format(vip['lbl_std_atmos']))
                lun.write('1\n')          # The 'output layer optical depths' flag
                for gg in range(6):       # The 6 continuum multipliers
                    lun.write('1.0\n')
                lun.write('{:7.3f}\n'.format(np.max(tropoe['height']-0.01)))
                lun.write('{:0d}\n'.format(len(tropoe['height'])+len(rt_extra_layers)))
                for gg in range(len(tropoe['height'])):
                    lun.write('{:7.3f}\n'.format(tropoe['height'][gg]))
                for gg in range(len(rt_extra_layers)):
                    lun.write('{:7.3f}\n'.format(rt_extra_layers[gg]))
                lun.close()
                    # Turn the flag off, as we only need to create these files once
                create_monortm_config = 0

            if create_monortm_zfreq == 1:
                # Create the MonoRTM freuency file
                lun = open(vip['workdir'] + '/' + monortm_zfreq, 'w')
                lun.write('\n')
                lun.write('{:0d}\n'.format(len(mwr['freq'])))
                for gg in range(len(mwr['freq'])):
                    lun.write('{:7.3f}\n'.format(mwr['freq'][gg]))
                lun.close()
                    # Turn the flag off, as we only need to create these files once
                create_monortm_zfreq = 0

                # Run the forward model and compute the Jacobian
            tmp_lQext_ratio = 2. / Other_functions.get_scat_properties('Qe', d_ref_wnum, Xn[1], sspl)
            flag, KK, FF, stored_mwr_jacobian = Jacobian_Functions.compute_jacobian_microwave_lwp_only(tropoe['height'],
                            tropoe['pressure'][sidx,:],tropoe['temperature'][sidx,:],tropoe['waterVapor'][sidx,:],
                            mwr['freq'], cbh, Xn, tmp_lQext_ratio, vip, vip['workdir'], monortm_tfile, monortm_zexec,
                            vip['lbl_std_atmos'], location['alt'], stored_mwr_jacobian, verbose)

            # If the Jacobian did not compute properly (i.e., an error ocurred),
            # then we need to abort
            if flag == 0:
                print('-- Skipping this sample due to issue with MonoRTM Jacobian (likely bad input profile)')
                break

            # Now the size of the forward calculation should be the correct size to match
            # the number of MWR observations in the Y vector
            if len(foo) != len(FF):
                print('Problem computing the Jacobian for the microwave radiometer')
                sys.exit()

            # Append the MWR calculated info to the forward calc and jacobian
            Kij = np.append(Kij, KK, axis = 0)
            FXn = np.append(FXn,FF)

        # If the observation data are missing, then set the forward calc and Jacobian 
        # for that obs to missing and 0, respectively
        foo = np.where(Y < -900)[0]
        if len(foo) > 0:
            FXn[foo]   = Y[foo]
            Kij[foo,:] = 0.
        
        # Do the retrieval calculations (matrix math)
        gfac = gamm[itern]
        sfac = sfactor[itern]
        B = (gfac * invSa) + Kij.T.dot(invSm).dot(Kij)
        Binv = np.linalg.pinv(B)
        Gain = Binv.dot(Kij.T).dot(invSm)

#        Output_Functions.write_variable(Y,'/data/Y.nc')
#        Output_Functions.write_variable(Xa,'/data/Xa.nc')
#        Output_Functions.write_variable(invSm,'/data/invSm.nc')
#        Output_Functions.write_variable(invSa,'/data/invSa.nc')
#        Output_Functions.write_variable(Xn,f'/data/Xn.{itern:d}.nc')
#        Output_Functions.write_variable(FXn,f'/data/FXn.{itern:d}.nc')
#        Output_Functions.write_variable(Kij,f'/data/Kij.{itern:d}.nc')
#        Output_Functions.write_variable(Binv,f'/data/Binv.{itern:d}.nc')
#        Output_Functions.write_variable(Gain,f'/data/Gain.{itern:d}.nc')

        if vip['math_choice'] == 0:
            Xnp1 = Xa[:,None] + Gain.dot(Y[:,None] - FXn[:,None] + Kij.dot((Xn-Xa)[:,None]))
        else:
            innovation = Binv.dot(Kij.T.dot(invSm).dot(Y[:,None] - FXn[:,None])-invSa.dot((Xn-Xa)[:,None]))
            if(vip['math_choice'] == 2):
                if((sfactor[itern] >= 1) & (itern > 0)):
                    if(total_tau > vip['annealing_min_tau']):
                        anneal_step = anneal_step * vip['annealing_step']
                        sfac = anneal_step
            Xnp1 = Xn[:,None] + sfac*innovation
        Xnp1 = np.squeeze(Xnp1)

        Sop = Binv.dot(gfac*gfac*invSa + Kij.T.dot(invSm).dot(Kij)).dot(Binv)
        SopInv = np.linalg.pinv(Sop)
        Akern  = (Binv.dot(Kij.T).dot(invSm).dot(Kij)).T

        rms = np.sqrt(np.mean((Y-FXn)**2))
        di2m = ((FXn[:,None] - FXnm1[:,None]).T.dot(
                np.linalg.pinv(Kij.dot(Sop).dot(Kij.T)+Sm)).dot(
                FXn[:,None] - FXnm1[:,None]))[0,0]
        if((di2m < vip['converge_factor']*len(Y)) and (sfactor[itern] >= 0.99)):
            conv = 1
        
        # Capture all of the results for this iteration
        if itern == 0:
            iXn = [np.copy(Xn)]
            iFXn = [np.copy(FXn)]
            irms = [np.copy(rms)]
            idi2m = [np.copy(di2m)]
            # Allocate some space. Do it each time to make sure we have a clean slate
            iKij = []
            iSop = []
            iAkern =[]
        else:
            iXn.append(np.copy(Xn))
            iFXn.append(np.copy(FXn))
            irms.append(np.copy(rms))
            idi2m.append(np.copy(di2m))
        
        iKij.append(Kij)
        iSop.append(Sop)
        iAkern.append(Akern)

        # Get the scale factors, if needed (default values already set)
        if(vip['ref_wnum'] < 0):
            d_lQext_ratio = 2. / Other_functions.get_scat_properties('Qe', d_ref_wnum, Xn[1], sspl)
            d_iQext_ratio = 2. / Other_functions.get_scat_properties('Qe', d_ref_wnum, Xn[3], sspi)

        # Write status of this iteration to stdOut
        if verbose >= 1 and conv != 1:
            print('     {:2d}  {:7.2f} {:7.2f} {:7.2f} {:7.2f}    {:6.2f}   {:10.4e}   {:5.2f}'.format(
            itern,Xn[0]*d_lQext_ratio,Xn[1],Xn[2]*d_iQext_ratio,Xn[3],rms,di2m,sfac))

        # Update for the next iteration
        FXnm1 = np.copy(FXn)
        Xn = np.copy(Xnp1)
        itern += 1

        # Some QC basic limits
        Xn[0] = np.max([Xn[0],vip['min_ltau']*d_lQext_ratio])
        Xn[1] = np.min([np.max([Xn[1],min_lReff]),max_lReff])
        Xn[2] = np.max([Xn[2],vip['min_itau']*d_iQext_ratio])
        Xn[3] = np.min([np.max([Xn[3],min_iReff]),max_iReff])
        
    # If retrieval converged we will store that result. If not, then we
    # will store the iteration that had the smallest RMS relative to the observations
    if conv == 1:
        if verbose >= 1:
            print('         Converged via di2m test')
        itern -= 1
    elif itern >= vip['maxiter'] - 1:
        foo = np.where(irms == np.min(irms))[0]
        itern = foo[0]
        if verbose >= 1:
            print(f'         Did not converge -- reached maxIter -- using iteration {itern:d}')
    else:
        print('Error: not sure why this "not converge and not maxIter happened -- debug code')
        sys.exit()
    # Select the case we will store
    Xn    = np.squeeze(iXn[itern])
    Sop   = np.squeeze(iSop[itern])
    Akern = np.squeeze(iAkern[itern])
    FXn   = np.squeeze(iFXn[itern])
    rms   = np.squeeze(irms[itern])
    di2m  = np.squeeze(idi2m[itern])

    # Get the scale factors, if needed (default values already set)
    if(vip['ref_wnum'] < 0):
        d_lQext_ratio = 2. / Other_functions.get_scat_properties('Qe', d_ref_wnum, Xn[1], sspl)
        d_iQext_ratio = 2. / Other_functions.get_scat_properties('Qe', d_ref_wnum, Xn[3], sspi)
    # Scale the optical depths and their uncertainties appropriately (to the desired reference wavenumber)
    Xn[0]    = Xn[0]*d_lQext_ratio
    Xn[2]    = Xn[2]*d_iQext_ratio
    Sop[0,:] = Sop[0,:] * d_lQext_ratio
    Sop[:,0] = Sop[:,0] * d_lQext_ratio
    Sop[2,:] = Sop[2,:] * d_iQext_ratio
    Sop[:,2] = Sop[:,2] * d_iQext_ratio

    if verbose >= 1:
        tmp1 = np.sqrt(np.diag(Sop))
        tmp2 = np.diag(Akern)
        print('         {:6.2f} {:6.2f} {:6.2f} {:6.2f} : 1-sigma uncertainties'.format(tmp1[0],tmp1[1],tmp1[2],tmp1[3]))
        print('         {:6.2f} {:6.2f} {:6.2f} {:6.2f} : DFS values'.format(tmp2[0],tmp2[1],tmp2[2],tmp2[3]))
    
    # Compute the LWP and its uncertainty.  LWP is computed only if the three tests are true
    lwpm = -999.
    lwpu = -999.
    min_sigma_lwp = 0.001 # Minimum uncertainty in LWP [g/m2]
    if((vip['compute_lwp'] == 1) and (vip['retrieve_lcloud'] == 1) and (vip['ref_wnum'] < 0)):
        # Use Bevington method, for x = auv, then
        # VarX/X^2 = VarU/U^2 + VarV/V^2 + 2*CovUV/(U*V)
        lwpm = Other_functions.compute_lwp(Xn[0],Xn[1])
        uuu2 = lwpm**2 * ((Sop[0,0]/Xn[0]**2) + (Sop[1,1]/Xn[1]**2) + 2*(Sop[0,1]/(Xn[0]*Xn[1])))
        if uuu2 > 0:
            lwpu = np.sqrt(uuu2)

    # Compute the optical depth fraction and its uncertainty
    # Use Bevington method, for x = au/(u+v)
    fracm = Xn[0]/(Xn[0]+Xn[2])
    fracu = ((    Xn[2]**2 / (Xn[0]+Xn[2])**4 ) * Sop[0,0]
          + (     Xn[0]**2 / (Xn[0]+Xn[2])**4 ) * Sop[2,2]
          - ((Xn[0]*Xn[2]) / (Xn[0]+Xn[2])**4 ) * Sop[0,2])
    fracu = np.sqrt(fracu)

    # Generate some additional output to capture the
    # correlations among the different retrieved variables
    Cop = np.copy(Sop)
    for i in range(len(Cop[0,:])):
        for j in range(len(Cop[:,0])):
            Cop[i,j] = Sop[i,j] / (np.sqrt(Sop[i,i])*np.sqrt(Sop[j,j]))
    msig = np.sqrt(np.diag(Sop))
    mdfs = np.diag(Akern)
    mcor = np.array([Cop[0,1],Cop[0,2],Cop[0,3],Cop[1,2],Cop[1,3],Cop[2,3]])

    # Store the data into a growing structure
    mout = {'secs':irs['secs'][samp], 'X':np.copy(Xn), 'sigx':np.copy(msig),
            'dfs':np.copy(mdfs),'y':np.copy(Y),'sigy':np.sqrt(np.diag(Sm)),
            'FXn':np.copy(FXn),'lhour':lhour[tidx],'lwpm':np.copy(lwpm),
            'lwpu':np.copy(lwpu),'fracm':np.copy(fracm),'fracu':np.copy(fracu),
            'delt':delt[tidx],'cbh':np.copy(cbh),'sza':np.copy(isza),'tcld':np.copy(tcld),
            'corr':np.copy(mcor),'iter':np.copy(itern),'conv':np.copy(conv),
            'rms':np.copy(rms),'di2m':np.copy(di2m)}
    
    # Compute the full spectrum if desired
    outfull = {'include':vip['include_full_calc']}
    if vip['include_full_calc'] > 0:
        if(verbose >= 1):
            print('      Computing the full IRS spectrum for this solution')
        flag, full_FXn, full_Kij = Jacobian_Functions.mixcra_forward_model(Xn, tropoe['height'],
                                        lblout[tidx], lwc, vip, jday[samp], isza,
                                        sfc_emissivity,d_ref_wnum, nyobs2, microwin_file2,
                                        0, 0, verbose)
        
        if flag == 0:
            full_FXn = np.ones(nyobs2)*-999.
        fullwnum = np.mean(entire_spectrum,axis=0)
        outfull = {'include':vip['include_full_calc'], 'wnum':np.copy(fullwnum),
                   'y':np.squeeze(yobs2[:,samp]),'sigy':np.squeeze(ysig2[:,samp]),
                   'FXn':np.copy(full_FXn)}
    
    # Write the output to the netCDF file
    currenttime = datetime.now()
    
    flag, ofilename = Output_Functions.mixcra_write_output(mout,vip,location,flagY,dimY,outfull,
                          fsample,(currenttime-starttime).total_seconds(),verbose,globatt,ofilename)
    
    # Increment the pointer, and continue
    fsample += 1

# If this flag was set then delete the working subdirectory
if vip['delete_temporary'] != 0:
    if verbose >= 2:
        print('  Deleting the working directory ' + vip['workdir'])
    shutil.rmtree(vip['workdir'])

# Successful exit
print(('>>> MIXCRA retrieval on ' + str(date) + ' ended properly <<<'))
print('--------------------------------------------------------------------')
print(' ')

