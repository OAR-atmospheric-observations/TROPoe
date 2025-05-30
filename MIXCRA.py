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

__version__ = '0.0.1'

import os
import sys
import numpy as np
import shutil
import scipy.io
import warnings
from netCDF4 import Dataset
from datetime import datetime, timezone
from glob import glob
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

# Create parser for command line arguments
parser = ArgumentParser()

parser.add_argument("date", type=int, help="Date to run the code [YYYYMMDD]")
parser.add_argument("vip_filename", help="Name if the VIP file (string)")
parser.add_argument("--step", help="The step size for Mixcra")
parser.add_argument("--shour", type=float, help="Start hour (decimal, 0-24)")
parser.add_argument("--ehour", type=float, help="End hour (decimal, 0-24) [If ehour<0 process up to last IRS sample]")
parser.add_argument("--verbose",type=int, choices=[0,1,2,3], help="The verbosity of the output (0-very quiet, 3-noisy)")
parser.add_argument("--dostop",action="store_true", help="Set this to stop at the end before exiting")
parser.add_argument("--keep",action="store_true",help="Not really sure what this does")

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

# Capture the version of the code
# TODO Figure out how we want to do the version stuff here. This is just a simple place holder
mixcra_version = __version__
print('  Running MIXCRA version ' + mixcra_version)

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

# Specify some global parameters
globatt = {'algorithm_code': 'MIXCRAv2 retrieval code',
            'algorithm_author': 'Dave Turner, NOAA Global System Laboratory (dave.turner@noaa.gov), ' +
                                'Ported to Python by Josh Gebauer NOAA National Severe Storms Laboratory / CIWRO (joshua.gebauer@noaa.gov)',
            'algorithm_version': mixcra_version,
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

# Make sure that the reference wavenumber makes some sense
if (vip['ref_wnum'] <= 0):
    vip['ref_wnum'] = -1

# Determine which AERI spectra to process
if step is None:
    step = vip['irs_sampling']          # Process every (step)th spectrum
if step <= 0:
    step = 1                            # Must process iin a forward way
vip['aeri_sampling'] = step             # Update the VIP file, if passed in via command line

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
if vip['sovler'] != 0 and vip['solver'] != -1:
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
if verbose >= 1:
    print('   Override the delete_temporary in the VIP file to keep the working directory contents')
if keep:
    vip['delete_temporary'] = 0

# Read in the TROPoe input file
if verbose >= 0:
    print('   Reading the TROPoe output file')
tropoe_filename = '*{tropoe,aerioe}*' + str(date) + '*.{nc,cdf}'
files = Data_reads.findfile(vip['tropoe_path'],tropoe_filename)
if len(files) != 1:
    print('Error: Unable to find a single TROPoe file for this date in ' + vip['tropoe_path'])
    sys.exit()

tropoe = Data_reads.read_tropoe(files[0])

# Confirm AERI spectra and summary files exist for this date
irsch1_filename = '*{aeri,assist}*{ch1,ChA,cha}*' + str(date)+'*.{nc,cdf}'
irsch1_file = Data_reads.findfile(vip['irsch1_path'],irsch1_filename)

irsch2_filename = '*{aeri,assist}*{ch2,ChB,chb}*' + str(date)+'*.{nc,cdf}'
irsch2_file = Data_reads.findfile(vip['irsch2_path'],irsch2_filename)

irssum_filename = '*{aeri,assist}*sum*' + str(date)+'*.{nc,cdf}'
irssum_file = Data_reads.findfile(vip['irssum_path'],irssum_filename)

if len(irssum_file) != 1:
    print('Error: Unable to find a single IRS summary file for this date in ' + vip['irssum_path'])
    sys.exit()

if len(irsch1_file) != len(irssum_file):
    print('Error: the number of IRS ch1 files does not equal the number of IRS summary files - aborting')
    sys.exit()

# Now read in the IRS data
irsch1 = Data_reads.skinny_read_irs_ch(irsch1_file[0],vip)
if use_sun == 1:
    if len(irsch2_file) != len(irssum_file):
        print('Error: the number of IRS ch1 files does not equal the number of IRS summary files - aborting')
        sys.exit()
    irsch2 = Data_reads.skinny_read_irs_ch(irsch2_file[0],vip)
    if irsch2['status'] == 0:
        sys.exit()
else:
    irsch2 = {'status':0}

irssum = Data_reads.skinny_read_irs_sum(irssum_file[0])
if irssum['status'] == 0:
    sys.exit()

irs = Data_reads.combine_irs(irsch1, irsch2, irssum,use_sun)

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
    minw = np.min(irs['wnum'])
    maxw = np.max(irs['wnum'])

entire_spectrum = np.array([minw,minw+delw])
minw += delw
while np.max(entire_spectrum) < maxw:
    new_row = np.array([[minw, minw + delw]])
    entire_spectrum = np.vstack([entire_spectrum, new_row])
    minw += delw

# Now avereage the IRS data over the desired spectral bands
yobs1, ysig1, nyobs1 = Other_functions.average_irs(irs,spectral_bands)       # These are the obs for the retrieval
yobs2, ysig2, nyobs2 = Other_functions.average_irs(irs,entire_spectrum)      # This is only for the end

# Need to capture the latitude and longitude of the station
if vip['station_alt'] > 0:
    location = {'lat':vip['station_lat'], 'lon':vip['station_lon'],'alt':vip['station_alt']}
else:
    location = {'lat':irs['lat'], 'lon':irs['lon'],'alt':irs['alt']}

# Add the date being processed to the workdir entry; this allows multiple days to be processed simultaneous
vip['workdir'] = vip['workdir'] + '_' + str(date)

# If this flag was set then delete the working subdirectory, and restart from scratch
if vip['delete temporary'] != 1:
    if verbose >= 1:
        print('    Reinitializing the working directory ' + vip['workdir'])
    shutil.rmtree(vip['workdir'])
    os.mkdir(vip['workdir'])
    
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
                                              tropoe['pressure'][lidx[i],:], tropoe['temperature'[lidx[i],:]],
                                              tropoe['watervapor'][lidx[i],:], lblwnum1, lblwnum2,
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
for j in len(spectral_bands[0]):
    lun.write('{:7.2}  {:7.2f}\n'.format(spectral_bands[0,j],spectral_bands[1,j]))
lun.close()

# For the end-game "entire_spectrum"
microwin_file2 = vip['workdir'] + '/microwindows2.txt'
if verbose >= 3:
    print('  Generating microwindow file for the entire spectral band')
lun = open(microwin_file2,'w')
lun.write(str(len(entire_spectrum[0])) + '\n')
for j in len(entire_spectrum[0]):
    lun.write('{:7.2}  {:7.2f}\n'.format(entire_spectrum[0,j],entire_spectrum[1,j]))
lun.close()

# Determine the ranges of the effective radii in the SSP DBs
ssp_file = Data_reads.findfile(vip['lcloud_ssp'])
if vip['retrieve_lcloud'] and len(ssp_file) != 1:
    print('Error: Unable to uniquely determine the needed file ' + vip['lcloud_ssp'])
    sys.exit()
else:
    # Don't take the exact_min and max; give room for perturbation calcs
    sspl, flag = VIP_Databases_functions.read_scat_databases(ssp_file[0])
    min_lReff = sspl['data'][2,1]
    max_lReff = sspl['data'][2,-2]
    print('  The minimum and maximum effective radii in the SSP for ' \
    'lcloud is {:6.2f} and {:6.2} microns'.format(min_lReff, max_lReff))

ssp_file = Data_reads.findfile(vip['icloud_ssp'])
if vip['retrieve_icloud'] and len(ssp_file) != 1:
    print('Error: Unable to uniquely determine the needed file ' + vip['icloud_ssp'])
    sys.exit()
else:
    # Don't take the exact_min and max; give room for perturbation calcs
    sspi, flag = VIP_Databases_functions.read_scat_databases(ssp_file[0])
    min_iReff = sspl['data'][2,1]
    max_iReff = sspl['data'][2,-2]
    print('  The minimum and maximum effective radii in the SSP for ' \
    'icloud is {:6.2f} and {:6.2} microns'.format(min_iReff, max_iReff))

# Build the prior and its uncertainty
minsd = 1e-5
if vip['retrieve_lcloud'] == 0:
    vip['prior_ltau_mn'] = 0.
    vip['prior_ltau_sd'] = minsd
    vip['prior_lreff_sd'] = minsd
    vip['min_ltau'] = 0
if vip['retrieve_icloud'] == 0:
    vip['prior_itau_mn'] = 0.
    vip['prior_itau_sd'] = minsd
    vip['prior_ireff_sd'] = minsd
    vip['min_itau'] = 0

# If use sun == 0, then set all solar zenith angles to -1, else to the proper value
solzenang = np.ones(len(irs['hour']))*-1
if use_sun == 1:
    for i in range(len(irs['time'])):
        dt = datetime.fromtimestamp(irs['time'][i],tz=timezone.utc)
        sun_angle = solar.get_altitude(location['lat'], location['lon'], dt)
        solzenang[i] = 90-sun_angle

# Define the gamma array
gamm = np.ones(vip['maxiter'])
sfactor = np.ones(vip['maxiter'])

# This bit is an attempt to help the code converge more smoothly, based upon the Rodger's equations used
if vip['math_choice'] == 0:
    # If first math_choice is 0, then I will tweak the gamma factors and use Eq. 5.9
    gamm[0] = 10
    gamm[1] = 3

    # This doesn't make sense but it is in the IDL code so I left it, but it needs looked at.
    gamm[0] = 1
    gamm[1] = 1
    
elif vip['math_choice'] == 1:
    # If math_choice is 1, then use Eq. 5.8 with no modifications 
    foo = 0       # This does nothing as intended
elif vip['math_choice'] == 2:
    # If math_choice is 2, then use Eq 5.8 with annealing
    sfactor[0] = 0.3
    sfactor[1] = 0.7
else:
    print('Error -- undefined math_choice in the VIP file -- aorting')
    sys.exit()

if step > 1 and verbose >= 2:
    print('  Will run the retrieval on every ' + str(step) + ' sample')

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
        foo = np.where(cbh <= 0)[0]
        if len(foo) <= 0:
            cbh[foo] = tropoe['cbh'][0]
    
    if len(tropoe['hour']) == 1:
        lwc = Other_functions.create_lwc_profile(max(tropoe['lwp'],5),tropoe['height'], cbh, vip['lwc'])
    else:
        lwc = Other_functions.create_lwc_profile(max(np.interp(irs['hour'][samp],tropoe['hour'],tropoe['lwp']),5), tropoe['height'], cbh, vip['lwc'])

    # Get an estimate of the temperature of the cloud
    tcld = Other_functions.estimate_tcld(lwc,tropoe['temperature'],tropoe['hour'],irs['hour'][samp])

    # Define the a priori mean and covariance
    Xa = np.array([vip['prior_ltau_mn'],vip['prior_lreff_mn'],vip['prior_itau_mn'],vip['prior_ireff_mn']])
    sig = np.array([vip['prior_ltau_sd'],vip['prior_lreff_sd'],vip['prior_itau_sd'],vip['prior_ireff_sd']])

    # If desired, then modify the liquid and ice optical depths based upon cloud temperature
    if vip['apply_tcloud_constraints'] == 1 and vip['retrieve_icloud'] == 1 and vip['retrieve_lcloud'] == 1:
        xtcld = np.ones(60)*-50
        xifac = 1-1./(1+np.exp(-(xtcld+ 5)))
        xlfac = 1./(1+np.exp(-(xtcld+35)))
        imult = min(max(np.interp(tcld,xtcld,xifac),0),1)
        lmult = min(max(np.interp(tcld,xtcld,xlfac),0),1)
        Xa[0] *= lmult
        sig[0] *= lmult
        Xa[2] *= imult
        sig[2] *= imult
    
    sig = max(sig,minsd)
    Ca = np.diag(np.ones(4))
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
    for i in len(sig):
        for j in len(sig):
            Sa[i,j] = Ca[i,j]*sig[i]*sig[j]
    invSa = np.linalg.pinv(Sa)

    # Quick QC on the prior's effective radii
    Xa[1] = min(max(Xa[1],min_lReff),max_lReff)
    Xa[3] - min(max(Xa[3],min_iReff),max_iReff)

    # Build the observation vector and its covariance matrix
    cwnum = np.mean(spectral_bands,axis=1)
    Y = np.squeeze(yobs1[:,samp])
    foo = np.where(ysig1[:,samp] > 0)[0]
    if len(foo) <= 0:
        print('Error: it seems that none of the IRS data were in the spectral bands -- aborting')
        sys.exit()
    
    minnoise = np.min(ysig1[foo,samp])
    Sm = np.diag((np.maximum(vip.aeri_noise_multiplier * ysig1[:, samp], minnoise))**2)
    invSm = np.linalg.pinv(Sm)

    # Initialize the first guess to the prior, but update the ltau to be a bit more accurate
    Xn = np.copy(Xa)
    if vip['retrieve_lcloud'] == 1 and vip['compute_lwp'] == 1:
        lwp = np.interp(irs['hour'][samp],tropoe['hour'],tropoe['lwp'])
        Xn[0] = Other_functions.compute_ltau(lwp,Xn[1])
    
    # Now iterate the retrieval
    FXnm1 = np.ones(nyobs1)*999.
    di2m = 9e14
    rms = 999.
    itern = 0
    conv = 0
    if verbose >= 1:
        print('   Iter     lTau   lReff    iTau   iReff       RMS         di2m  sfactor')
        print('     {:2d}  {:7.2f} {:7.2f} {:7.2f} {:7.2f}    {:6.2f}   {:10.4e}   {:5.2f}'.format(
            itern,Xn[0],Xn[1],Xn[2],Xn[3],Xn[4],rms,di2m,0))
    
    # Now perfrom the iterations
    while((itern < vip['maxiter']-1) and (conv == 0)):
        delt = np.abs(lhour-irs['hour'][samp])
        lidx = np.where(delt == np.min(delt))[0][0]
        isza = solzenang[samp]

        # If the solar zenith angle is greater than 90 degrees,
        # it is below the horizon sp turn solar input off
        if isza >= 90:
            isza = -1
        
        # Call the LBLDIS
        flag, FXn, Kij = Jacobian_Functions.mixcra_forward_model(Xn, tropoe['height'], lblout[lidx], lwc, vip, isza,
                                              sfc_emissivity,vip['ref_wnum'], nyobs1, microwin_file1,
                                              vip['retrieve_lcloud'], vip['retrieve_icloud'], verbose)
        
        if flag == 0:
            print('problem with the forward model')
        
        # If the observation data are missing, then set the forward calc and Jacobian 
        # for that obs to missing and 0, respectively
        foo = np.where(Y < -990)[0]
        if len(foo) > 0:
            FXn[foo] = Y[foo]
            Kij[foo,:] = 0.
        
        # Do the retrieval calculations (matrix math)
        gfac = gamm[itern]
        sfac = sfactor[itern]
        B = (gfac * invSa) + Kij.T.dot(invSm).dot(Kij)
        Binv = np.linalg.pinv(B)
        Gain = Binv.dot(Kij.T).dot(invSm)

        if vip['math_choice'] == 0:
            Xnp1   = Xa[:,None] + Gain.dot(Y[:,None] - FXn[:,None] + Kij.dot((Xn-Xa)[:,None]))
        else:
            Xnp1 = Xn[:,None] + sfac*Binv.dot(Kij.T.dot(invSm).dot(Y[:,None] - FXn[:,None])-invSa.dot((Xn-Xa)[:,None]))
        Xnp1 = np.squeeze(Xnp1)

        Sop = Binv.dot(gfac*gfac*invSa + Kij.T.dot(invSm).dot(Kij)).dot(Binv)
        SopInv = np.linalg.pinv(Sop)
        Akern  = (Binv.dot(Kij.T).dot(invSm).dot(Kij)).T

        rms = np.sqrt(np.mean((Y-FXn)**2))
        di2m = ((FXn[:,None] - FXnm1[:,None]).T.dot(
                np.linalg.pinv(Kij.dot(Sop).dot(Kij.T)+Sm)).dot(
                FXn[:,None] - FXnm1[:,None]))[0,0]
        if (di2m < vip['converge_factor']*yobs1) and (sfac >= 0.99):
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

        # Update for the next iteration
        FXnm1 = np.copy(FXn)
        Xn = np.copy(Xnp1)
        itern += 1

        # Some QC basic limits
        Xn[0] = max(Xn[0],vip['min_ltau'])
        Xn[1] = min(max(Xn[1],min_lReff),max_lReff)
        Xn[2] = max(Xn[2],vip['min_ltau'])
        Xn[3] = min(max(Xn[3],min_iReff),max_iReff)

        if verbose >= 1 and conv != 1:
            print('     {:2d}  {:7.2f} {:7.2f} {:7.2f} {:7.2f}    {:6.2f}   {:10.4e}   {:5.2f}'.format(
            itern,Xn[0],Xn[1],Xn[2],Xn[3],Xn[4],rms,di2m,sfac))
        
    # If retrieval converged we will store that result. If not, then we
    # will store the iteration that had the smallest RMS relative to the observations

    if conv == 1:
        if verbose >= 1:
            print('         Converged via di2m test')
        itern -= 1
    
    elif itern >= vip['maxiter'] - 1:
        if verbose >= 1:
            print('         Did not converge -- reached maxIter')
            foo = np.where(irms == np.min(irms))[0]
            itern = foo[0]
    else:
        print('Error: not sure why this "not converge and not maxIter happened -- debug code')
        sys.exit()
    
    Xn = np.squeeze(iXn[itern])
    Sop = np.squeeze(iSop[itern])
    Akern = np.squeeze(iAkern[itern])
    FXn = np.squeeze(iFXn[itern])
    rms = np.squeeze(irms[itern])
    di2m = np.squeeze(idi2m[itern])

    if verbose >= 1:
        tmp1 = np.sqrt(np.diag(Sop))
        tmp2 = np.diag(Akern)
        print('         {:6.2f} {:6.2f} {:6.2f} {:6.2f} : 1-sigma uncertainties'.format(tmp1[0],tmp1[1],tmp1[2],tmp1[3]))
        print('         {:6.2f} {:6.2f} {:6.2f} {:6.2f} : 1-sigma uncertainties'.format(tmp2[0],tmp2[1],tmp2[2],tmp2[3]))
    
    # Compute the LWP and its uncertainty
    lwpm = -999.
    lwpu = -999.
    if vip['compute_lwp'] == 1 and vip['retrieve_lcloud'] == 1:
        # Use Bevington method, for x = auv, then
        # VarX/X^2 = VarU/U^2 + VarV/V^2 + 2*CovUV/(U*V)
        lwpm = Other_functions.compute_lwp(Xn[0],Xn[1])
        uuu2 = lwpm**2 * (Sop[0,0]/Xn[0]**2) + (Sop[1,1]/Xn[1]**2) + 2*(Sop[0,1]/(Xn[0]*Xn[1]))
        lwpu = np.sqrt(uuu2)
    
    # Compute the optical depth fraction and its uncertainty
    # Use Bevington method, for x = au/(u+v)
    fracm = Xn[0]/(Xn[0]*Xn[2])
    fracu = ((    Xn[2]**2 / (Xn[0]+Xn[2])**4 ) * Sop[0,0]
          + (         1    / (Xn[0]+Xn[2])**4 ) * Sop[2,2]
          - ( (2 * Xn[2])  / (Xn[0]+Xn[2])**4 ) * Sop[0,2])
    fracu = np.sqrt(fracu)

    # Generate some additional output to capture the
    # correlations among the different retrieved variables
    Cop = np.copy(Sop)
    for i in range(len(Cop[0])):
        for j in range(len(Cop[:,0])):
            Cop[i,j] = Sop[i,j] / (np.sqrt(Sop[i,i])*np.sqrt(Sop[j,j]))
    msig = np.sqrt(np.diag(Sop))
    mdfs = np.diag(Akern)
    mcor = np.array([Cop(0,1),Cop(0,2),Cop(0,3),Cop(1,2),Cop(1,3),Cop(2,3)])

    # Store the data into a growing structure
    mout = {'secs':irs['sec'][samp], 'X':np.copy(Xn), 'sigx':np.copy(msig),
            'dfs':np.copy(mdfs),'y':np.copy(y),'sigy':np.sqrt(np.diag(Sm)),
            'FXn':np.copy(FXn),'lhour':lhour[lidx],'lwpm':np.copy(lwpm),
            'lwpu':np.copy(lwpu),'fracm':np.copy(fracm),'fracu':np.copy(fracu),
            'delt':delt[lidx],'cbh':np.copy(cbh),'sza':np.copy(isza),'tcld':np.copy(tcld),
            'corr':np.copy(mcor),'iter':np.copy(itern),'conv':np.copy(conv),
            'rms':np.copy(rms),'di2m':np.copy(di2m)}
    
    # Compute the full spectrum if desired
    outfull = {'include':vip['include_full_calc']}
    if vip['include_full_calc'] > 0:
        flag, full_FXn, full_Kij = Jacobian_Functions.mixcra_forward_model(Xn, tropoe['height'], lblout[lidx], lwc, vip, isza,
                                              sfc_emissivity,vip['ref_wnum'], nyobs2, microwin_file2,
                                              0, 0, verbose)
        
        if flag == 0:
            full_FXn = np.ones(nyobs2)*-999.
        fullwnum = np.mean(entire_spectrum,axis=1)
        outfull = {'include':vip['include_full_calc'], 'wnum':np.copy(fullwnum),
                   'y':np.squeeze(yobs2[:,samp]),'sigy':np.squeeze(ysig2[:,samp]),
                   'FXn':np.copy(full_FXn)}
    
    # Write the output to the netCDF file
    currenttime = datetime.now()
    
    flag, ofilename = Output_Functions.mixcra_write_output(mout,vip,location,cwnum,outfull,
                                                           fsample,currenttime-starttime,
                                                           verbose,globatt,ofilename)
    
    # Increment the pointer, and continue
    fsample += 1

# If this flag was set then delete the working subdirectory
if vip['delete_temporary'] == 1:
    if verbose >= 2:
        print('  Deleting the working directory ' + vip['workdir'])

    shutil.rmtree(vip['workdir'])
    os.mkdir(vip['workdir'])


