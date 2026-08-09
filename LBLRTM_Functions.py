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
import struct
import errno
from pathlib import Path
from subprocess import Popen, PIPE


################################################################################
# This file contains the following functions:
# rundecker()
# lbl_read()
# read_tape27()
# run_monortm()
# write_lbldis_parmfile
################################################################################

################################################################################
# This function builds rundecks which can be used for both the LBLRTM
# and the RRTM. Clear sky runs only. Can use standard atmospheres
# for the PTU data (one or all variables). Able to compute either
# radiance/transmittance or optical depths. Layer structure used in the models
# can be specified. Lots of features here.
#
# Profiles for other gasses can be specified to be one of the default
# atmospheres. In fact, its a good idear to always specify the background
# atmosphere when z/p/t/w are entered, so the correct atmosphere is added
# above the highest level in the z/p/t/w profile...
################################################################################

def rundecker(model, aprofile, z, p, t, w, co2_profile= None, o3_profile=None,
           ch4_profile=None, n2o_profile=None, co_profile = None, p_units=None,
           t_units=None, w_units=None, o3_units=None, iout=0, icld=0, numangs=0,
           comment='None', p_comment='User_supplied profile', cntnm=None, xsec=1,
           od_only=0, sc = None, co2_mix=None, ccl4_sfactor=None, f11_sfactor=None,
           f12_sfactor=None, h2o_sfactor=None, co2_sfactor = None, o3_sfactor=None,
           co_sfactor=None, ch4_sfactor=None, n2o_sfactor=None, o2_sfactor=None,
           view_angle=0.0, sfc_emis=None, sfc_refl=None, sfc_temp=None, sfc_type= None, 
           wnum1=None, wnum2=None, short=False, mlayers=None, mlay_pres=False,
           altitude=0.0, v10=False, sample=4.0, tape5=None, tape7=None, monortm=False,
           freqs=None, silent=False, juldate=None, sza=None):

   # For capturing the version number
    rcsid = '$Id: rundecker.py,v 0.1 2019/07/29 Josh.Gebauer Exp $'
    parts = rcsid.split(' ')
    version = parts[2]
   
    if ((aprofile < 1) | (aprofile > 6)):
        print(' Error: Value for keyword aprofile is out of range - aborting')
        return
    
    if ((len(z) != len(p)) | (len(z) != len(t)) | (len(z) != len(w))):
        print('Error: The lenght of the z/p/t/w vectors must be the same - aborting')
        return
    
    if len(z) > 0:
        have_profile = 1
    else:
        have_profile = 0
    
    # Process the gas scale factors to make sur they are in correct
    # format (i.e., 2-element vectors)
    if h2o_sfactor is None:
        h2o_sf = np.array([1.0,0.])
    elif type(h2o_sfactor) is np.ndarray:
        if ((len(h2o_sfactor.shape) == 1) & (h2o_sfactor.shape[0] == 2)):
            h2o_sf = np.copy(h2o_sfactor)
        else:
            print('WARNING: H2O scale factor vector improperly specified; setting to default')
            h2o_sf = np.array([1.0,0.])
    else:
        print('WARNING: H2O scale factor vector improperly specified; setting to default')
        h2o_sf = np.array([1.0,0.]) 
        
    if co2_sfactor is None:
        if co2_profile is None:
            co2_sf = np.array([380.0,1.0])
        else:
            co2_sf = np.array([1.0,0.])
    elif type(co2_sfactor) is np.ndarray:
        if ((len(co2_sfactor.shape) == 1) & (co2_sfactor.shape[0] == 2)):
            co2_sf = np.copy(co2_sfactor)
        else:
            print('WARNING: CO2 scale factor vector improperly specified; setting to default')
            co2_sf = np.array([380.0,1.0])
    else:
        print('WARNING: CO2 scale factor vector improperly specified; setting to default')
        co2_sf = np.array([380.0,1.0]) 
        
    if o3_sfactor is None:
        o3_sf = np.array([1.0,0.])
    elif type(o3_sfactor) is np.ndarray:
        if ((len(o3_sfactor.shape) == 1) & (o3_sfactor.shape[0] == 2)):
            o3_sf = np.copy(o3_sfactor)
        else:
            print('WARNING: O3 scale factor vector improperly specified; setting to default')
            o3_sf = np.array([1.0,0.])
    else:
        print('WARNING: O3 scale factor vector improperly specified; setting to default')
        o3_sf = np.array([1.0,0.]) 
        
    if co_sfactor is None:
        co_sf = np.array([1.0,0.])
    elif type(co_sfactor) is np.ndarray:
        if ((len(co_sfactor.shape) == 1) & (co_sfactor.shape[0] == 2)):
            co_sf = np.copy(co_sfactor)
        else:
            print('WARNING: CO scale factor vector improperly specified; setting to default')
            co_sf = np.array([1.0,0.])
    else:
        print('WARNING: CO scale factor vector improperly specified; setting to default')
        co_sf = np.array([1.0,0.])
        
    if ch4_sfactor is None:
        ch4_sf = np.array([1.0,0.])
    elif type(ch4_sfactor) is np.ndarray:
        if ((len(ch4_sfactor.shape) == 1) & (ch4_sfactor.shape[0] == 2)):
            ch4_sf = np.copy(ch4_sfactor)
        else:
            print('WARNING: CH4 scale factor vector improperly specified; setting to default')
            ch4_sf = np.array([1.0,0.])
    else:
        print('WARNING: CH4 scale factor vector improperly specified; setting to default')
        ch4_sf = np.array([1.0,0.]) 
        
    if n2o_sfactor is None:
        n2o_sf = np.array([1.0,0.])
    elif type(n2o_sfactor) is np.ndarray:
        if ((len(n2o_sfactor.shape) == 1) & (n2o_sfactor.shape[0] == 2)):
            n2o_sf = np.copy(n2o_sfactor)
        else:
            print('WARNING: N2O scale factor vector improperly specified; setting to default')
            n2o_sf = np.array([1.0,0.])
    else:
        print('WARNING: N2O scale factor vector improperly specified; setting to default')
        n2o_sf = np.array([1.0,0.]) 
        
    if o2_sfactor is None:
        o2_sf = np.array([1.0,0.])
    elif type(o2_sfactor) is np.ndarray:
        if ((len(o2_sfactor.shape) == 1) & (o2_sfactor.shape[0] == 2)):
            o2_sf = np.copy(o2_sfactor)
        else:
            print('WARNING: O2 scale factor vector improperly specified; setting to default')
            o2_sf = np.array([1.0,0.])
    else:
        print('WARNING: O2 scale factor vector improperly specified; setting to default')
        o2_sf = np.array([1.0,0.])
        
    # Process the CFC scale factors to make sure they are in the correct format,
    # and then convert them into volume mixing ratios
       
    default_ccl4 = 0.1105 #ppm
    if ccl4_sfactor is None:
        ccl4_sf = np.array([default_ccl4,1.0])
    elif type(ccl4_sfactor) is np.ndarray:
        if ((len(ccl4_sfactor.shape) == 1) & (ccl4_sfactor.shape[0] == 2)):
            if ((ccl4_sfactor[1] < 0) | (ccl4_sfactor[1] > 1)):
                print('WARNING: CCL4 scale factor units are incorrect; setting to default')
                ccl4_sf = np.array([default_ccl4,1.0])
            else:
                ccl4_sf = np.copy(ccl4_sfactor)
            if int(ccl4_sfactor[1]+0.5) == 0:
                ccl4_sf = np.array([default_ccl4*ccl4_sfactor[0],1.0])
        else:
            print('WARNING: CCL4 scale factor vector imporoperly specified; setting to default')
            ccl4_sf = np.array([default_ccl4,1.0])
    else:
        print('WARNING: CCL4 scale factor vector imporoperly specified; setting to default')
        ccl4_sf = np.array([default_ccl4,1.0]) 
    
    default_f11 = 0.2783 #ppm
    if f11_sfactor is None:
        f11_sf = np.array([default_f11,1.0])
    elif type(f11_sfactor) is np.ndarray:
        if ((len(f11_sfactor.shape) == 1) & (f11_sfactor.shape[0] == 2)):
            if ((f11_sfactor[1] < 0) | (f11_sfactor[1] > 1)):
                print('WARNING: F11 scale factor units are incorrect; setting to default')
                f11_sf = np.array([default_f11,1.0])
            else:
                f11_sf = np.copy(f11_sfactor)
            if int(f11_sfactor[1]+0.5) == 0:
                f11_sf = np.array([default_f11*f11_sfactor[0],1.0])
        else:
            print('WARNING: F11 scale factor vector imporoperly specified; setting to default')
            f11_sf = np.array([default_f11,1.0])
    else:
        print('WARNING: F11 scale factor vector imporoperly specified; setting to default')
        f11_sf = np.array([default_f11,1.0]) 
   
    default_f12 = 0.5027 #ppm
    if f12_sfactor is None:
        f12_sf = np.array([default_f12,1.0])
    elif type(f12_sfactor) is np.ndarray:
        if ((len(f12_sfactor.shape) == 1) & (f12_sfactor.shape[0] == 2)):
            if ((f12_sfactor[1] < 0) | (f12_sfactor[1] > 1)):
                print('WARNING: F12 scale factor units are incorrect; setting to default')
                f12_sf = np.array([default_f12,1.0])
            else:
                f12_sf = np.copy(f12_sfactor)
            if int(f12_sfactor[1]+0.5) == 0:
                f12_sf = np.array([default_f12*f12_sfactor[0],1.0])
        else:
            print('WARNING: F12 scale factor vector imporoperly specified; setting to default')
            f12_sf = np.array([default_f12,1.0])
    else:
        print('WARNING: F12 scale factor vector imporoperly specified; setting to default')
        f12_sf = np.array([default_f12,1.0])
        
    if sample < 1.0:
        sample = 1.0
    if sample > 4.0:
        sample = 4.0
        
    if monortm:
        if view_angle != 0:
            print('MonoRTM calcs (in this tool) can only be made in the downwelling direction')
            print('Please set view_angle to zero degrees (the default value)')
            return
    
        # We don't need heavy molecule cross-sections
        if xsec != 0:
            print('There is no need for heavy molecule cross-sections -- turning them off')
            xsec = 0
    
        # We won't be applying any instrument filter function
        if sc != 0:
            print('No filter function is applied to monoRTM calculations -- turning sc off')
            sc = 0
    
        # We can define wnum1 and wnum2 here so that they don't need to be entered.
        # They aren't used anyway...
        wnum1 = 400
        wnum2 = 1400
        
    if ((90 < view_angle % 360) & (view_angle % 360 < 270)):
        direction = 'upwelling'
    else:
        direction = 'downwelling'
    
    if direction == 'upwelling':
        if not silent:
            print('Writing tape5 for an upwelling calculation...')
        if sfc_temp is None:
            print('ERROR: Surface temperature (sfc_temp) is required for upwelling calculations')
            return
        if sfc_type is None:
            sfc_type = 'l'
        if ((sfc_type != 'l') & (sfc_type != 's')):
            print('Error: Incorrectly defined surface (reflectance) type')
            return
    else:
        if not silent:
            print('Writing tape5 for a downwelling calculation...')
    
    iemit = 1                # Calculate radiance and transmittance
    merge = 0                # Normal merge
    if sc is None:
        scan = 3             # Filter function set for FFT-based instrument
        inst = 1             # and instrument is set to the AERI
    elif sc == 0:
        scan = 0
    else:
        scan = 3
        inst = sc
        
    # Handle the continuum options and scale factors
    # TODO -- make sure that the "cntnm" variable is consistent with the IDL code (may be ok, maybe not)
    cntnm_default = 1
    reset_cntnm = False
    if cntnm is None:
        cntnm = cntnm_default
    if type(cntnm) is np.ndarray:
            if len(cntnm) != 7:
                reset_cntnm = True
            else:
                #Now check to make sure cntnm isn't set to negative numbers
                foo = np.where(cntnm < 0)[0]
                if len(foo) > 0:
                    cntnm[foo] = 0
                cntnm_array = np.copy(cntnm)          # The array of coninuum scale factors
                cntnm = 6                     # The flag indicating scale factors will be used
    elif type(cntnm) is int:
        if ((cntnm < 0)  | (cntnm > 5)):
            reset_cntnm = True
    else:
        reset_cntnm = True
    
    if reset_cntnm:
        print('WARNING: continuum flag is reset to the default - continuing')
        cntnm = cntnm_default
    
    
    # If we only want optical depths, then need to reset some flags
    if od_only > 0.5:
        iemit = 0           # Optical depths only
        scan = 0            # No fliter function applied (monocromatic results)
        if od_only > 1.5:
            # layer-to-instrument transmittance profiles in TAPE13
            merge = 4
            iemit = 1
        else:
            # Optical depths only; results from each layer in separate file
            merge = 1
    
    # If this is the RRTM then turn these off as they are meaningless anyway...
    if model == 0:
        xsec = 0
        cntnm = 0
        iemit = 1
    
    # Don't need cross-sections for channel 2
    if model == 2:
        xsec = 0
        
    # Unable to insert cross-sections if we don't have z/p/t/w profiles
    if have_profile == 0:
        xsec = 0
    
    if co2_mix is not None:
        if v10:
            print('Warning: Keyword CO2_MIX is not used with v10; use CO2_SFACTOR instead')
    else:
        co2_mix = 360.0
    
    if ((model < 0) | (model > 3)):
        print('Undefined model - quitting')
        return
    
    if o3_profile is not None:
        if len(o3_profile) != len(p):
            print('Ozone profile does not have same number of levels as pressure profile')
            return

    if co2_profile is not None:
        if len(co2_profile) != len(p):
            print('Carbon dioxide profile does not have same number of levels as pressure profile')
            return
    
    if ch4_profile is not None:
        if len(ch4_profile) != len(p):
            print('Methane profile does not have same number of levels as pressure profile')
            return
    
    if co_profile is not None:
        if len(co_profile) != len(p):
            print('Carbon monoxide profile does not have same number of levels as pressure profile')
            return
    
    if n2o_profile is not None:
        if len(n2o_profile) != len(p):
            print('Nitrous oxide profile does not have same number of levels as pressure profile')
            return
    
    # If TAPE7 is set, validate that the file actually exists. If it does,
    # then use this file as the input into the model run.
    
    if tape7 is not None:
        if not os.path.exists(tape7):
            print('Unable to determine the TAPE7 file ' + tape7)
            print(' and therefore the code is aborting!')
            return
        iatm = 0
    else:
        iatm = 1
        
     # If this is an RRTM run, then MLAYERS and SHORT should have no affect
     # DDT had this commented out for some reason
#    if ((model == 0) & (not monortm)):
#         mlayers = -1
#         short = 0

    # The altitudes for the model layers
    if mlayers is None:
        mlayers = -1
    if type(mlayers) is int:
        if mlayers < 0:
            if short:
                # The old layering used for the SGP QME...  
                mlayers = np.array([0.320, 0.360, 0.400, 0.500, 0.600, 0.700, 0.800,
                    0.900, 1.000, 1.200, 1.400, 1.600, 1.800, 2.000, 2.200, 2.400, 2.600,
      	            2.800, 3.000, 3.200, 3.400, 3.600, 3.800, 4.000, 4.500, 5.000, 5.500,
      	            6.000, 6.500, 7.000, 7.500, 8.000, 8.500, 9.000, 9.500, 10.000, 11.000,
      	            12.000, 13.000, 14.000, 15.000, 16.500, 18.000, 20.000])
      	            
      	        # Better layering in general
                mlayers = np.arange(11)*0.1
                mlayers = np.append(mlayers,np.arange(10)*0.25+1.25)
                mlayers = np.append(mlayers, np.arange(23)*0.5+4.0)
                mlayers = np.append(mlayers, np.arange(5)+16)
            else:
      	        # The old layering used for the SGP QME...
                mlayers = np.array([0.320, 0.360, 0.400, 0.500, 0.600, 0.700, 0.800,
                    0.900, 1.000, 1.200, 1.400, 1.600, 1.800, 2.000, 2.200, 2.400, 2.600,
      	            2.800, 3.000, 3.200, 3.400, 3.600, 3.800, 4.000, 4.500, 5.000, 5.500,
      	            6.000, 6.500, 7.000, 7.500, 8.000, 8.500, 9.000, 9.500, 10.000, 11.000,
      	            12.000, 13.000, 14.000, 15.000, 16.500, 18.000, 20.000, 22.000, 24.000,
                    25.500, 27.000, 28.500, 30.000, 32.000, 34.000, 36.000, 38.000, 40.000,
                    42.000, 44.000, 46.000, 48.000, 50.000, 52.000, 54.000, 56.000, 58.000,
                    60.000, 64.000, 68.000])
	       
	        # Better layering in general
                mlayers = np.arange(11)*0.1
                mlayers = np.append(mlayers,np.arange(10)*0.25+1.25)
                mlayers = np.append(mlayers, np.arange(23)*0.5+4.0)
                mlayers = np.append(mlayers, np.arange(5)+16)
                mlayers = np.append(mlayers, np.arange(10)*2+22)
                mlayers = np.append(mlayers, np.arange(8)*4+42)
      	        
            if not mlay_pres:
      	        if have_profile:
      	            mlayers = mlayers = z[0]
      	        else:
      	            mlayers = mlayers + altitude
    
    # Put this in the rundeck until I am sure I am
    # getting everything in the right place
    numbers0 = '         1         2         3         4         5         6         7         8         9'
    numbers1 = '123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 123456789 '
    
    ############################################################################
    # Standard LBLRTM parameters
    # Meaning of some of these flags:
    #     CN: continuum off/on
    #     AE: aerosols off/on (set to 1 uses LOWTRAN aerosols, others available
    #     EM: 0->optical depth only, 1->radiance and transmittance, other
    #     SC: scanning function (one of 3 types)
    #     FI: filter? no/yes
    #     PL: plot (i.e., write out output) no/yes
    #     XS: Use cross section no/yes
    
    rec_1_2 = ' HI=1 F4=1 CN={:d}'.format(cntnm)
    rec_1_2 = rec_1_2 + ' AE=0 EM={:d} SC={:d} FI=0 PL=0 TS=0 AM={:d} MG={:d} LA=0 MS=0 XS={:d}'.format(iemit,scan,iatm,merge,xsec)
    rec_1_2 = rec_1_2 + '   00   00'
    
    rec_1_3_a = '420.094   1849.855                                          0.0002    0.001'
    rec_1_3_b = '1750.338  3070.000                                          0.0002    0.001'
    
    if model == 3:
        if ((wnum1 is None) | (wnum2 is None)):
            print('Undefined starting or ending wavenumber for custom calculations - aborting')
            return
        if wnum2-wnum1 >= 2020:
            print('Maximum difference allowed between starting and ending wnum is 2020 - aborting')
            return
        if wnum2-wnum1 <= 120:
            print('Ending wnum must be at least 120 cm-1 larger than starting wnum - aborting')
            return
        if v10:
            rec_1_3_c = '{:10.3f}{:10.3f}{:10.3f}                              0.0002    0.001                             7'.format(wnum1,wnum2,sample)
        else:
            rec_1_3_c = '{:10.3f}{:10.3f}{:10.3f}                              0.0002    0.001'.format(wnum1,wnum2,sample)
        
    # Set the boundary temperature, emissivity, and reflectivity.
    # This is only really important for the upwelling calculations.
    # Only need to output these if we are calculating radiances (iemit = 1)
    # Note that we treat the RRTM data slightly differently
        
    if model == 0:
        if sfc_temp is None:
            sfc_temp = -1
        if sfc_emis is None:
            sfc_emis = 1.
        if sfc_emis == 1:
            iemis = 1
        else:
            iemis = 2
        
        ireflect = 0
        rec_1_4 = '{:10.3E} {:1d}  {:1d}'.format(sfc_temp,iemis,ireflect)
        if type(sfc_emis) is int:
            rec_1_4 = rec_1_4 + '{:5.3f} '.format(sfc_emis)
        else:
            sfc_emis_len = len(sfc_emis)
            for j in range(len(sfc_emis_len)):
                rec_1_4 = rec_1_4 + '{:5.3f} '.format(sfc_emis[j])
        
    elif iemit == 1:
        if direction == 'downwelling':
            rec_1_4 = '0.000     0.000'
        else:
            # Get the surface temperature
            print('WARNING: This code is not capable of setting up upwelling LBLRTM run, I have to exit')
            return
    
    if have_profile == 1:
        flag = 0
    else:
        flag = aprofile
    if mlay_pres:
        msign = -1
    else:
        msign = 1
        
    if v10:
        rec_3_1 = '    {:1d}    2{:5d}    1    1    7    1'.format(flag,msign*len(mlayers))
    else:
        rec_3_1 = '    {:1d}    2{:5d}    1    1    7    1                                   {:10.3f}'.format(flag,msign*len(mlayers),co2_mix)
    
    ##################################################
    # Additional portion for the RRTM part
    rec_1_2_1 = ''               # The default setting
    if model == 0:
        rec_1_2 = rec_1_2 + '    {:0d}  {:3d}    {:0d}'.format(numangs,iout,icld)
        if ((juldate is not None) & (sza is not None)):
            rec_1_2_1 = '            {:3d}   {:7.4f}    {:1d}'.format(int(juldate+0.5),sza,0)
    
    # If this is a monoRTM run then override what is in rec_1_2
    # this is very default behavior and should be better treated...
    if monortm:
        rec_1_2 = '    1    0    1    0    1    0    0    1    0    1         0    0    0    0    0    1'
    
    # Get the date for the default comment string
    # Logic to catch various errors that could occur when we spawn to the environment
    try:
        with Popen('date', stdout = PIPE, stderr = PIPE, shell=True, executable = '/bin/csh') as process:
            stdout, stderr = process.communicate()
        # Catches failures occurring during process creation
    except OSError as e:
        if e.errno == errno.EAGAIN:
            print("Popen error with rundecker: Out of PIDs or process limit reached (EAGAIN).")
        elif e.errno == errno.ENOMEM:
            print("Popen error with rundecker: Out of system memory (ENOMEM).")
        elif isinstance(e, FileNotFoundError):
            print("Popen error with rundecker: C-shell executable '/bin/csh' was not found.")
        else:
            print(f"Popen OS error with rundecker ({e.errno}): {e.strerror}")
        sys.exit()
        # Catches non-OS exceptions (e.g., KeyboardInterrupt or I/O errors during communication)
    except Exception as e:
        print(f"Unexpected error running rundecker: {e}")
        sys.exit()
    date = stdout[:-1].decode()
    
    # Get the proper output filename
    if tape5 is None:
        if monortm:
            tape5='MONORTM.IN'
        else:
            if model == 0:
                tape5 = 'INPUT_RRTM'
            elif model == 1:
                tape5 = 'TAPE5_ch1'
            elif model == 2:
                tape5 = 'TAPE5_ch2'
            else:
                tape5 = 'TAPE5_custom'
    
    # Open the file, and write the rundeck
    if not silent:
        print('Writing ' + tape5 + '...')
            
    lun = open(tape5,'w')
    
    acomment = 'Rundeck created on ' + date + ' by rundecker.py (v' + version + ')'
    lun.write(acomment + '\n')
    
    lun.write(numbers0 + '\n')
    lun.write(numbers1 + '\n')
    
    if len(comment) > 75:
        comment = comment[0:76]
    
    lun.write('$ ' + comment + '\n')
    
    lun.write(rec_1_2 + '\n')
    if rec_1_2_1 != '':
        lun.write(rec_1_2_1 + '\n')
    
    if monortm:
        lun.write('-0.100E+00 0.100E+02 0.000E+00 0.100E+00 0.000E+00 0.000E+00 0.000E+00 0.000E+00    0      0.000E+00\n')
        if freqs is not None:
            clight = 2.9979248e10   #cm/s
            lun.write('{:0d}\n'.format(len(freqs)))
            for kk in range(len(freqs)):
                lun.write('{:19.7E}\n'.format(freqs[kk]/clight*1e9))
        else:
            lun.write('5\n')
            lun.write('0.789344\n')
            lun.write('0.79828\n')
            lun.write('1.043027\n')
            lun.write('1.051763\n')
            lun.write('3.000000\n')
        
        lun.write(' 0.275E+01 0.100E+01 0.000E+00 0.000E+00 0.000E+00 0.000E+00 0.000E+00\n')
    
        
    # Add the continuum, scale factors if desired (Record 1_2a)
    if ((cntnm == 6) & (not monortm)):
        lun.write('{:7.4f}  {:7.4f}  {:7.4f}  {:7.4f}  {:7.4f}  {:7.4f}  {:7.4f}  \n'.format(cntnm_array[0],cntnm_array[1],cntnm_array[2],cntnm_array[3],cntnm_array[4],cntnm_array[5],cntnm_array[6]))
        
    if ((model == 1) & (not monortm)):
        lun.write(rec_1_3_a +'\n')
    if ((model == 2) & (not monortm)):
        lun.write(rec_1_3_b +'\n')
    if ((model == 3) & (not monortm)):
        lun.write(rec_1_3_c +'\n')
    
    if ((v10) & (not monortm)):
        if int(h2o_sf[1]+0.5) == 0:
            sf_string = '1'
        elif int(h2o_sf[1]+0.5) == 1:
            sf_string = 'm'
        elif int(h2o_sf[1]+0.5) == 2:
            sf_string = 'p'
        else:
            print('Error: H2O scale factor inappropriately set (units not defined)')
            lun.close()
            os.remove(tape5)
            return
        
        if (int(h2o_sf[1]+0.5)==1):
            h2o_sf[0] = h2o_sf[0]/1e6   # Convert ppmv to ppv
        
        if int(co2_sf[1]+0.5) == 0:
            sf_string = sf_string+'1'
        elif int(co2_sf[1]+0.5) == 1:
            sf_string = sf_string+'m'
        else:
            print('Error: CO2 scale factor inappropriately set (units not defined)')
            lun.close()
            os.remove(tape5)
            return
        
        if (int(co2_sf[1]+0.5)==1):
            co2_sf[0] = co2_sf[0]/1e6   # Convert ppmv to ppv
        
        if int(o3_sf[1]+0.5) == 0:
            sf_string = sf_string+'1'
        elif int(o3_sf[1]+0.5) == 1:
            sf_string = sf_string+'m'    
        elif int(o3_sf[1]+0.5) == 2:
            sf_string = sf_string+'d'
        else:
            print('Error: O3 scale factor inappropriately set (units not defined)')
            lun.close()
            os.remove(tape5)
            return
        
        if (int(o3_sf[1]+0.5)==1):
            o3_sf[0] = o3_sf[0]/1e6   # Convert ppmv to ppv
        
        if int(n2o_sf[1]+0.5) == 0:
            sf_string = sf_string+'1'
        elif int(n2o_sf[1]+0.5) == 1:
            sf_string = sf_string+'m'
        else:
            print('Error: N2O scale factor inappropriately set (units not defined)')
            lun.close()
            os.remove(tape5)
            return
        
        if (int(n2o_sf[1]+0.5)==1):
            n2o_sf[0] = n2o_sf[0]/1e6   # Convert ppmv to ppv
        
        if int(co_sf[1]+0.5) == 0:
            sf_string = sf_string+'1'
        elif int(co_sf[1]+0.5) == 1:
            sf_string = sf_string+'m'
        else:
            print('Error: CO scale factor inappropriately set (units not defined)')
            lun.close()
            os.remove(tape5)
            return
            
        if (int(co_sf[1]+0.5)==1):
            co_sf[0] = co_sf[0]/1e6   # Convert ppmv to ppv
            
        if int(ch4_sf[1]+0.5) == 0:
            sf_string = sf_string+'1'
        elif int(ch4_sf[1]+0.5) == 1:
            sf_string = sf_string+'m'
        else:
            print('Error: CH4 scale factor inappropriately set (units not defined)')
            lun.close()
            os.remove(tape5)
            return
        
        if (int(ch4_sf[1]+0.5)==1):
            ch4_sf[0] = ch4_sf[0]/1e6   # Convert ppmv to ppv]
        
        if int(o2_sf[1]+0.5) == 0:
            sf_string = sf_string+'1'
        elif int(o2_sf[1]+0.5) == 1:
            sf_string = sf_string+'m'
        else:
            print('Error: O2 scale factor inappropriately set (units not defined)')
            lun.close()
            os.remove(tape5)
            return
        
        if (int(o2_sf[1]+0.5)==1):
            o2_sf[0] = o2_sf[0]/1e6   # Convert ppmv to ppv]
            
        lun.write(sf_string + '\n')
        lun.write('{:15.7E}{:15.7E}{:15.7E}{:15.7E}{:15.7E}{:15.7E}{:15.7E}\n'.format(h2o_sf[0],co2_sf[0],o3_sf[0],n2o_sf[0],co_sf[0],ch4_sf[0],o2_sf[0]))
    
    if (((iemit == 1) | (model == 0)) & (not monortm)):
        lun.write(rec_1_4 +'\n')
    
    # If this flag is not set, then add the TAPE7 file to the rundeck
    if ((iatm == 0) & (not monortm)):
        line = ''
        lun2 = open(tape7,'r')
        line = lun2.readline()      # This is the comment line
        line = lun2.readline()      # This line has the number of layers, etc.
        
        # We will replace the SECNTO flag with the direction (view_angle)
        secnto = 1./np.cos(view_angle*np.pi/180.)
        scento_string = '{:9.6f}'.format(secnto)
        line = line[0:11] + scento_string + line[20:len(line)]
        
        # Now place this line in the tape5
        lun.write(line)
        nlayers = int(line[2:5])
        print(str(nlayers) + ' in the gas profiles')
        
        # We replace the IPATH flag to zero that way the SECNTO flag dictates the direction
        for i in range(nlayers):
            line = lun2.readline()
            line = line[0:38] + ' 0' + line[40:len(line)]
            lun.write(line)
            line = lun2.readline()
            lun.write(line)
        
        line = lun2.readline()
        lun.write(line)
        line = lun2.readline()
        lun.write(line)
        line = lun2.readline()
        
        # We will replace the SECNTO flag with the direction (view_angle)
        line = line[0:11] + scento_string + line[20:len(line)]
        lun.write(line)
        nlayers = int(line[2:5])
        print(str(nlayers) + ' in the xsec profiles')
        
        # We replace the IPATH flag to zero that way the SECNTO flag dictates the direction
        for i in range(nlayers):
            line = lun2.readline()
            line = line[0:38] + ' 0' + line[40:len(line)]
            lun.write(line)
            line = lun2.readline()
            lun.write(line)
        lun2.close()
    
    else:
        # Add in the section on the model layers
        lun.write(rec_3_1 + '\n')
        if direction == 'upwelling':
            h1 = mlayers[len(mlayers)-1]
            h2 = mlayers[0]
        else:
            h1 = mlayers[0]
            h2 = mlayers[len(mlayers)-1]
        
        line = '{:10.3f}{:10.3f}{:10.3f}'.format(h1,h2,view_angle)
        lun.write(line + '\n')
        line = ''
        for i in range(1,len(mlayers)+1):
            line = line + '{:10.3f}'.format(mlayers[i-1])
            if i%8 == 0:
                lun.write(line + '\n')
                line=''
        
        if len(mlayers) % 8 != 0:
            lun.write(line + '\n')
            
        # We are assuming that the input profiles are all the same length. We need to
        # verify that they actually extend to the top layer needed for the model
        # else we will use the U.S. Standard Atmosphere value
        
        if have_profile == 1:
            if p_units is None:
                p_units = 'mb'
            if t_units is None:
                t_units = 'K'
            if w_units is None:
                w_units = 'g/kg'
            if o3_units is None:
                o3_units = 'ppmv'
            
            if p_units == 'mb':
                JCHARP = 'A'
            elif p_units == 'atm':
                JCHARP = 'B'
            elif p_units == 'torr':
                JCHARP = 'C'
            elif p_units == 'sa1':
                JCHARP = '1'
            elif p_units == 'sa2':
                JCHARP = '2'
            elif p_units == 'sa3':
                JCHARP = '3'
            elif p_units == 'sa4':
                JCHARP = '4'
            elif p_units == 'sa5':
                JCHARP = '5'
            elif p_units == 'sa6':
                JCHARP = '6'
            else:
                print('Unidentified pressure unit - quitting')
                return
            
            if t_units == 'K':
                JCHART = 'A'
            elif t_units == 'C':
                JCHART = 'B'
            elif t_units == 'sa1':
                JCHART = '1'
            elif t_units == 'sa2':
                JCHART = '2'
            elif t_units == 'sa3':
                JCHART = '3'
            elif t_units == 'sa4':
                JCHART = '4'
            elif t_units == 'sa5':
                JCHART = '5'
            elif t_units == 'sa6':
                JCHART = '6'
            else:
                print('Unidentified temperature unit - quitting')
                return
            
            if w_units == 'ppmv':
                JCHAR = 'A'
            elif w_units == 'cm-3':
                JCHAR = 'B'
            elif w_units == 'g/kg':
                JCHAR = 'C'
            elif w_units == 'g/m3':
                JCHAR = 'D'
            elif w_units == 'mb':
                JCHAR = 'E'
            elif w_units == 'K':
                JCHAR = 'F'
            elif w_units == 'C':
                JCHAR = 'G'
            elif w_units == '%':
                JCHAR = 'H'
            elif w_units == 'sa1':
                JCHAR = '1'
            elif w_units == 'sa2':
                JCHAR = '2'
            elif w_units == 'sa3':
                JCHAR = '3'
            elif w_units == 'sa4':
                JCHAR = '4'
            elif w_units == 'sa5':
                JCHAR = '5'
            elif w_units == 'sa6':
                JCHAR = '6'
            else:
                print('Unidentified moisture profile unit - quitting')
                return
            
            # Work on co2 units
            if co2_profile is not None:
                # The default units for CO2 is 'ppmv'
                JCHAR = JCHAR + 'A'
            else:
                JCHAR = JCHAR + '{:0d}'.format(aprofile) # Use climatology for co2
            
            # Work on ozone units
            if o3_profile is not None:
                if o3_units == 'ppmv':
                    JCHAR = JCHAR + 'A'
                elif o3_units == 'cm-3':
                    JCHAR = JCHAR + 'B'
                elif o3_units == 'g/kg':
                    JCHAR = JCHAR + 'C'
                elif o3_units == 'g/m3':
                    JCHAR = JCHAR + 'D'
                elif o3_units == 'sa1':
                    JCHAR = JCHAR + '1'
                elif o3_units == 'sa2':
                    JCHAR = JCHAR + '2'
                elif o3_units == 'sa3':
                    JCHAR = JCHAR + '3'
                elif o3_units == 'sa4':
                    JCHAR = JCHAR + '4'
                elif o3_units == 'sa5':
                    JCHAR = JCHAR + '5'
                elif o3_units == 'sa6':
                    JCHAR = JCHAR + '6'
                else:
                    print('Unidentified ozone profile unit - quitting')
                    return
            
            else:
                JCHAR = JCHAR + '{:0d}'.format(aprofile)
            
            # Work on n2o units
            if n2o_profile is not None:
                # The default units for N2O is 'ppmv'
                JCHAR = JCHAR + 'A'
            else:
                JCHAR = JCHAR + '{:0d}'.format(aprofile) # Use climatology for N2O
            
            # Work on co units
            if co_profile is not None:
                # The default units for CO is 'ppmv'
                JCHAR = JCHAR + 'A'
            else:
                JCHAR = JCHAR + '{:0d}'.format(aprofile) # Use climatology for CO
            
            # Work on ch4 units
            if ch4_profile is not None:
                # The default units for CH4 is 'ppmv'
                JCHAR = JCHAR + 'A'
            else:
                JCHAR = JCHAR + '{:0d}'.format(aprofile) # Use climatology for CH4
            
            # And use defaults for rest of the gases... (which is just oxygen)
            JCHAR = JCHAR + '{:0d}'.format(aprofile)
            
            # Make and use the copies of the profiles versus the actual profiles
            
            zz = np.copy(z)
            tt = np.copy(t)
            pp = np.copy(p)
            ww = np.copy(w)
            if co2_profile is not None:
                co2 = np.copy(co2_profile)
            else:
                co2 = np.zeros(len(pp))
            if o3_profile is not None:
                oo3 = np.copy(o3_profile)
            else:
                oo3 = np.zeros(len(pp))
            if n2o_profile is not None:
                n2o = np.copy(n2o_profile)
            else:
                n2o =  np.zeros(len(pp))
            if co_profile is not None:
                co = np.copy(co_profile)
            else:
                co = np.zeros(len(pp))
            if ch4_profile is not None:
                ch4 = np.copy(ch4_profile)
            else:
                ch4 = np.zeros(len(pp))
                
            # Need to apply some QC to the pressure profile for the RRTM
            # Make sure the pressures are always decreasing, and that no two
            # levels have the same pressure value
            
            foo = np.argsort(pp)
            foo = np.copy(pp[foo])
            bar = np.where(pp != np.fliplr([foo])[0])[0]
            if len(bar) > 0:
                print('Pressure array is not monotonically increasing - quitting')
                lun.close()
                return
            
            # Now find and keep only the unique pressure levels
            # for all of the input variables
            foo = np.unique(pp, return_index = True)[1]
            foo = np.fliplr([foo])[0]
            pp = pp[foo]
            zz = zz[foo]
            tt = tt[foo]
            ww = ww[foo]
            oo3 = oo3[foo]
            n2o = n2o[foo]
            co = co[foo]
            ch4 = ch4[foo]
            
            inlayers = len(zz)
            if mlay_pres:
                foo = np.where(mlayers < pp[inlayers-1])[0]
            else:
                # Precision errors between mlayers and zz can cause pressure
                # to increase at top of profile. Since the tape5 file only
                # writes to four sig figs, make sure the difference is
                # greater than 0.00009
                foo = np.where(mlayers > zz[inlayers-1]+0.00009)[0]
            if len(foo) >= 1:
                foo = foo[0:len(foo)]
                zz = np.append(zz, mlayers[foo])
                pp = np.append(pp, np.zeros(len(foo)))
                tt = np.append(tt, np.zeros(len(foo)))
                ww = np.append(ww, np.zeros(len(foo)))
                oo3 = np.append(oo3, np.zeros(len(foo)))
                n2o = np.append(n2o, np.zeros(len(foo)))
                co = np.append(co, np.zeros(len(foo)))
                ch4 = np.append(ch4, np.zeros(len(foo)))
                inlayers = inlayers + len(foo)
            
            if len(p_comment) > 23:
                p_comment = p_comment[0:24]
            lun.write('{:5d} '.format(inlayers) + p_comment + '\n')
            for i in range(inlayers-len(foo)):
                lun.write('{:10.4f} {:9.4f}{:10.3E}     '.format(zz[i],pp[i],tt[i]) + JCHARP + JCHART + '   ' + JCHAR + '\n')
                lun.write('{:10.3E}{:10.3E}{:10.3E}{:10.3E}{:10.3E}{:10.3E}{:10.3E}\n'.format(ww[i],co2[i],oo3[i],n2o[i],co[i],ch4[i],0))
            
            JCHARP = '{:0d}'.format(aprofile)
            JCHART = '{:0d}'.format(aprofile)    
            JCHAR = '{:0d}{:0d}{:0d}{:0d}{:0d}{:0d}{:0d}'.format(aprofile,aprofile,aprofile,aprofile,aprofile,aprofile,aprofile)
            
            for i in range(inlayers-len(foo),inlayers):
                lun.write('{:10.4f}{:10.3E}{:10.3E}     '.format(zz[i],pp[i],tt[i]) + JCHARP + JCHART + '   ' + JCHAR + '\n')
                lun.write('{:10.3f}{:10.3E}{:10.3E}{:10.3E}{:10.3E}{:10.3E}{:10.3E}\n'.format(ww[i],oo3[i],0,0,0,0,0))
            
        # If it is ch1 or ch2 rundecks, then we can add options for heavy molecule
        # cross sections, aerosols, and filter functions...
        
        if model > 0:
            # Add in the stuff for the cross sections, if desired, here
            # There are no aprreciable cross sections for ch2...
            if ((xsec == 1) & (have_profile == 1)):
                lun.write('    3    0    0  The following cross-sections were selected:\n')
                lun.write('CCL4      F11       F12\n')
                
                # We can enter these profiles at any height resolution (i.e.,
                # it can be different than the gas profiles defined earlier).
                # Since these are constant profiles, let's just use two levels
                index = np.array([0,len(zz)-1])        # The first and last level of the input profile
                lun.write('{:5d}    0 XS 1995 UNEP values\n'.format(len(index)))
                for i in range(len(index)):
                    lun.write('{:10.3f}     AAA\n'.format(zz[index[i]]))
                    lun.write('{:10.3E}{:10.3E}{:10.3E}\n'.format(ccl4_sf[0]/1.0e3,f11_sf[0]/1.0e3,f12_sf[0]/1.0e3))
    
    if model > 0:
        # Add in the stuff for the aerosols, if desired, here
        
        if inst == 1:
            opd = 1.03702766          # AERI
        elif inst == 2:
            opd = 2.000000000         # IASI
        else:
            print('Error in rundecker -- undefined FTS instrument')
            return
        
        # Construct an array of wavenumbers
        delv = 1.0/(2*opd)
        npts = int(10000/delv+1)
        varray = np.arange(npts)*1.0*delv + delv
        
        # Add in the stuff for the filter functions if desired, here
        # These two lines indicate that the radiance is to be written
        # to TAPE13, the transmittance to TAPE14. The last number
        # indicates the TAPE number
        
        if scan == 3:                 # Only if the filter-function is turned to FFT
            if model == 1:
               lun.write('1.03702766 497.575891803.71268    1   -4     0.48214722   12    1    1   13\n')
               lun.write('1.03702766 497.575891803.71268    0   -4     0.48214722   12    1    1   14\n')
            elif model == 2:
                lun.write('1.037027661796.480423022.09850    1   -4     0.48214722   12    1    1   13\n')
                lun.write('1.037027661796.480423022.09850    0   -4     0.48214722   12    1    1   14\n')
            elif model == 3:
                
                # I will only take the wavenumbers that are +50 cm-1 away
                # from the boundaries of the gaseous optical depth boundaries
                foo1 = np.where(varray > wnum1+50.)[0]
                foo2 = np.where(varray > wnum2-50.)[0]
                if ((len(foo1) == 0) | (len(foo2)==0)):
                    print('Error determining range for _filtering_ in rundeck - aborting')
                    print('Rundeck is incomplete!!!')
                    lun.close()
                    return
                
                lun.write('{:10.8f}{:10.5f}    1   -4     {:10.8f}   12    1    1   13\n'.format(opd,varray[foo1[0]],delv))
                lun.write('{:10.8f}{:10.5f}    0   -4     {:10.8f}   12    1    1   14\n'.format(opd,varray[foo1[0]],delv))
        
        # End of this set of commands to LBLRTM
        lun.write('-1.\n')
        
        # Add in the commands to control the plotting (i.e., the
        # way the output files (TAPE27 and TAPE 28) are created)
        
        if ((iemit == 1) & (scan > 0)):
            lun.write('$ Transfer to ASCII plotting data (TAPES 27 and 28)\n')
            lun.write(' HI=0 F4=0 CN=0 AE 0 EM=0 SC=0 FI=0 PL=1 TS=0 AM=0 MG=0 LA=0 MS=0 XS=0    0    0\n')
            lun.write('# Plot title not used')
            if model == 1:
                lun.write(' 499.986651799.85550   10.2000  100.0000    5    0   13    0     1.000 0  0    0\n')
                lun.write('    0.0000    1.2000    7.0200    0.2000    4    0    1    1    0    0 0    3 27\n')
                lun.write(' 499.986651799.85550   10.2000  100.0000    5    0   14    0     1.000 0  0    0\n')
                lun.write('    0.0000    1.2000    7.0200    0.2000    4    0    1    0    0    0 0    3 28\n')
            elif model == 2:
                lun.write('1800.337523020.16992   10.2000  100.0000    5    0   13    0     1.000 0  0    0\n')
                lun.write('1800.337523020.16992   10.2000  100.0000    5    0   14    0     1.000 0  0    0\n')
                lun.write('    0.0000    1.2000    7.0200    0.2000    4    0    1    0    0    0 0    3 28\n')
            elif model == 3:
                lun.write('{:10.5f}{:10.5f}   10.2000  100.0000    5    0   13    0     1.000 0  0    0\n'.format(varray[foo1[0]],varray[foo2[0]]))
                lun.write('    0.0000    1.2000    7.0200    0.2000    4    0    1    1    0    0 0    3 27\n')
                lun.write('{:10.5f}{:10.5f}   10.2000  100.0000    5    0   14    0     1.000 0  0    0\n'.format(varray[foo1[0]],varray[foo2[0]]))
                lun.write('    0.0000    1.2000    7.0200    0.2000    4    0    1    0    0    0 0    3 28\n')
            
            # End of this set of commands to LBLRTM.
            
            lun.write('1.\n')
        
    # And finally, the closing statement
    lun.write('%%%')
    lun.close()
    
    if not silent:
        print(tape5 + ' completed')
    return

################################################################################
#  This function reads a layer of data from a LBLRTM/FASCODE format.
#  These files are unformatted binary (FUN!) and this is one of the areas that
#  I had to deviate from DDT's scripts since IDL has a binary file read function
#  unlike python. The bones for this function come from panel_file.py in
#  Greg Blumburg's pyLBLRTM,
################################################################################

def lbl_read(lbl_file, do_load_data = False, valid_avg_P = None, valid_avg_T = None):

    # We don't want the default values to be mutable so
    # assign default values here

    if valid_avg_P is None:
        valid_avg_P = [1e-4, 1e4]

    if valid_avg_T is None:
        valid_avg_T = [100, 400]

    # First we need to check the type of the lbl file. It is either 32 or 64 bit
    # and also need to know the size of the junk fields. Has to be done using
    # brute force. Ugh binary files.
    
    # the 4 tested possibilities are:
    #
    # file_type 1: Junk fields are 4 bytes, default floats are 32 bit.
    # file_type 2: Junk fields are 4 bytes, default floats are 64 bit.
    # file_type 3: Junk fields are 8 bytes, default floats are 32 bit.
    # file_type 4: Junk fields are 8 bytes, default floats are 64 bit.
    #
    # It is possible that file types 3 & 4 are solely due to gfortran 
    # compiler problems on 64 bit linux. From my limited experience 
    # attempting to compile LBLRTM on a few machines:
    # Using the Intel ifort compiler produces files of type 1 or 2 only. 
    # Using gfortran on 32 bit Mac OS X produces file type 1 
    # (have been unable to compile it with default double precision).
    # Using f77 on 32 bit linux produces file type 1.
    # Using gfortran on 64 bit linux produces file type 3 and 4.
    
    f = open(lbl_file, 'rb')
    # Only read the max number of bytes we will possibly need
    test_bytes = f.read(112)
    f.close()
    
    # We will use the average temperature and pressure data to do the tests
    
    # Case 1
    junk_type = 'i'
    float_type = 'f'
    int_type = 'i'
    unpack_fmt = '=' + junk_type + '80sd' + 2*float_type
    test_data = struct.unpack(unpack_fmt, test_bytes[0:struct.calcsize(unpack_fmt)])
    if ((test_data[3] < valid_avg_P[0]) | (test_data[3] > valid_avg_P[1]) |
        (test_data[4] < valid_avg_T[0]) | (test_data[4] > valid_avg_T[1])):
        
        # Case 2
        junk_type = 'i'
        float_type = 'd'
        int_type = 'q'
        unpack_fmt = '=' + junk_type + '80sd' + 2*float_type
        test_data = struct.unpack(unpack_fmt, test_bytes[0:struct.calcsize(unpack_fmt)])
        if ((test_data[3] < valid_avg_P[0]) | (test_data[3] > valid_avg_P[1]) |
            (test_data[4] < valid_avg_T[0]) | (test_data[4] > valid_avg_T[1])):
            
            # Case 3
            junk_type = 'q'
            float_type = 'f'
            int_type = 'i'
            unpack_fmt = '=' + junk_type + '80sd' + 2*float_type
            test_data = struct.unpack(unpack_fmt, test_bytes[0:struct.calcsize(unpack_fmt)])
            if ((test_data[3] < valid_avg_P[0]) | (test_data[3] > valid_avg_P[1]) |
                (test_data[4] < valid_avg_T[0]) | (test_data[4] > valid_avg_T[1])):
                
                # Case 4
                junk_type = 'q'
                float_type = 'd'
                int_type = 'q'
                unpack_fmt = '=' + junk_type + '80sd' + 2*float_type
                test_data = struct.unpack(unpack_fmt, test_bytes[0:struct.calcsize(unpack_fmt)])
                if ((test_data[3] < valid_avg_P[0]) | (test_data[3] > valid_avg_P[1]) |
                    (test_data[4] < valid_avg_T[0]) | (test_data[4] > valid_avg_T[1])):
                    
                    raise IOError("Failed to determine field size in LBLRTM output file")
    
    f = open(lbl_file, 'rb')
    
    try:
        unpack_fmt = '=' + junk_type + '80sd' + 2*float_type
        raw_bytes = f.read(struct.calcsize(unpack_fmt))
        data = struct.unpack(unpack_fmt, raw_bytes)
        user_id = data[1]
        secant = data[2]
        p_avg = data[3]
        t_avg = data[4]
    
        unpack_fmt = 64*'8s'
        raw_bytes = f.read(struct.calcsize(unpack_fmt))
        molecule_id = struct.unpack(unpack_fmt, raw_bytes)
        
        unpack_fmt = 64*float_type
        raw_bytes = f.read(struct.calcsize(unpack_fmt))
        mol_col_dens = struct.unpack(unpack_fmt, raw_bytes)
        
        unpack_fmt = 2*float_type + 'dd' + 2*float_type + 11*int_type + 2*float_type + 6*int_type + float_type
        raw_bytes = f.read(struct.calcsize(unpack_fmt))
        data = struct.unpack(unpack_fmt, raw_bytes)
            
        broad_dens = data[0]
        dv = data[1]
        v1 = data[2]
        v2 = data[3]
        t_bound = data[4]
        emis_bound = data[5]
        lblrtm_flag = dict()
        lblrtm_flag['hirac'] = data[6]
        lblrtm_flag['lblf4'] = data[7]
        lblrtm_flag['xscnt'] = data[8]
        lblrtm_flag['aersl'] = data[9]
        lblrtm_flag['emit'] = data[10]
        lblrtm_flag['scan'] = data[11]
        lblrtm_flag['plot'] = data[12]
        lblrtm_flag['path'] = data[13]
        lblrtm_flag['jrad'] = data[14]
        lblrtm_flag['test'] = data[15]
        lblrtm_flag['merge'] = data[16]
        lblrtm_flag['scnid'] = data[17]
        lblrtm_flag['hwhm'] = data[18]
        lblrtm_flag['idabs'] = data[19]
        lblrtm_flag['atm'] = data[20]
        lblrtm_flag['layr1'] = data[21]
        lblrtm_flag['nlayr'] = data[22]
        n_mol = data[23]
        layer = data[24]
        yi1 = data[25]
        
        unpack_fmt = '=8s8s6s8s4s46s' + junk_type
        raw_bytes = f.read(struct.calcsize(unpack_fmt))
        data = struct.unpack(unpack_fmt,raw_bytes)
        # drop last junk word
        yid = data[0:-1]
    
    except:
        raise IOError('Failed to read in the header of the LBLRTM output file')
    
    hdr_size = f.tell()
    f.close()
    
    #try:
    f = open(lbl_file, 'rb')
    f.seek(hdr_size)
        
    # unpack string data into binary, and discard the first and last values
    # since those are junk values.
        
    unpack_fmt = '=' + junk_type + 'dd' + float_type + int_type + junk_type
    size = struct.calcsize(unpack_fmt)
    data = struct.unpack(unpack_fmt,f.read(size))
        
    panel_hdr = data[1:-1]
        
    # these checks are fully hardcoded - the wavenumber values (first and
    # second values) must be between 1 and 50000, the wavenum increment
    # (dv, third value) must be between 1e-9 and 10.0 and the number of
    # points (np), must be between 0 and 10000. The last value seems
    # to be 2400 in practice, at most ... so just guessing that 10000
    # makes sense as an upper limit.
    # also, check  that the wnum's are in the right order (should be wnum
    # min followed by wnum max, so wnum2 > wnum1.)
    
    if ((panel_hdr[0] < 1) | (panel_hdr[0] > 50000) | (panel_hdr[1] < 1) |
        (panel_hdr[1] < 1) | (panel_hdr[1] > 50000) | (panel_hdr[2] < 1e-9) | 
        (panel_hdr[2] > 10.0) | (panel_hdr[3] < 0) | (panel_hdr[3] > 10000) |
        (panel_hdr[1] < panel_hdr[0])):
        
        raise IOError('Panel header not successfully read')
    
    # this is a little clumsy - tell size of header by how much file
    # pointer moves after panel header read
    
    panel_hdr_size = f.tell() - hdr_size
    
    # Now left with 2 ugly cases
    # First there is only one panel in the entire file.
    # In this case, check the size of the file versus the size of the 
    # header plus a single or double format panel.
    
    file_size = os.stat(lbl_file).st_size
    
    panel_fmt = junk_type + panel_hdr[3]*float_type + junk_type
    panel_size_single = struct.calcsize(panel_fmt)
    
    panel_fmt = junk_type + panel_hdr[3]*float_type + junk_type + junk_type + panel_hdr[3]*float_type + junk_type
    panel_size_double = struct.calcsize(panel_fmt)
    
    if file_size == (hdr_size + panel_size_single + panel_hdr_size):
        panel_format = 'single'
        f.close()
    
    elif file_size == (hdr_size + panel_size_double + panel_hdr_size):
        panel_format = 'double'
        f.close()
        
    # file seem to have a "null" panel header at the end sometimes,
    # so check for that
    
    elif file_size == (hdr_size + panel_size_single + 2*panel_hdr_size):
        panel_format = 'single'
        f.close()
    
    elif file_size == (hdr_size + panel_size_double + 2*panel_hdr_size):
        panel_format = 'double'
        f.close()
    
    # Second case - there are multiple panels. In this case, try to read
    # the second panel's header to see if we can get sensible values
    f.seek(hdr_size)
    
    # unpack string data into binary, and discard the first and last values
    # since those are junk values.
    
    unpack_fmt = '=' + junk_type + 'dd' + float_type + int_type + junk_type
    size = struct.calcsize(unpack_fmt)
    data = struct.unpack(unpack_fmt,f.read(size))
    
    panel_hdr = data[1:-1]
    f.seek(panel_size_single,1)
    
    # unpack string data into binary, and discard the first and last values
    # since those are junk values.
    
    unpack_fmt = '=' + junk_type + 'dd' + float_type + int_type + junk_type
    size = struct.calcsize(unpack_fmt)
    data = struct.unpack(unpack_fmt,f.read(size))
    
    panel_hdr = data[1:-1]
    single_panel_valid = ((panel_hdr[0] > 1) & (panel_hdr[0] < 50000) & (panel_hdr[1] > 1) &
        (panel_hdr[1] > 1) & (panel_hdr[1] < 50000) & (panel_hdr[2] > 1e-9) & 
        (panel_hdr[2] <= 10.0) & (panel_hdr[3] > 0) & (panel_hdr[3] < 10000) &
        (panel_hdr[1] > panel_hdr[0]))
    
    # this can cause a read-over-EOF error; when this occurred, the
    # above method did succeed (I think this produced an error only
    # when the file was between 1 and 2 full single panels in length
    
    if not single_panel_valid:
        f.seek(hdr_size)
        # unpack string data into binary, and discard the first and last values
        # since those are junk values.
    
        unpack_fmt = '=' + junk_type + 'dd' + float_type + int_type + junk_type
        size = struct.calcsize(unpack_fmt)
        data = struct.unpack(unpack_fmt,f.read(size))
    
        panel_hdr = data[1:-1]
        f.seek(panel_size_double,1)
        
        # unpack string data into binary, and discard the first and last values
        # since those are junk values.
    
        unpack_fmt = '=' + junk_type + 'dd' + float_type + int_type + junk_type
        size = struct.calcsize(unpack_fmt)
        data = struct.unpack(unpack_fmt,f.read(size))
    
        panel_hdr = data[1:-1]
        double_panel_valid = ((panel_hdr[0] > 1) & (panel_hdr[0] < 50000) & (panel_hdr[1] > 1) &
            (panel_hdr[1] > 1) & (panel_hdr[1] < 50000) & (panel_hdr[2] > 1e-9) & 
            (panel_hdr[2] <= 10.0) & (panel_hdr[3] > 0) & (panel_hdr[3] < 10000) &
            (panel_hdr[1] > panel_hdr[0]))
    else:
        double_panel_valid = False
    
    if single_panel_valid:
        panel_format = 'single'
        f.close()
    elif double_panel_valid:
        panel_format = 'double'
        f.close()
    else:
        f.close()
        raise IOError('Could not determine panel format')
    #except:
    #    raise IOError('Could not determine panel format')
    
    
    # compute number of panels, by sequentially reading the file.
    # Unfortunately, I do not think there is anyway to check this 
    # except for reading through the file. Previous versions attempted 
    # to take a shortcut by assuming the panel structure was always 
    # n L-sized panels (L is usually 2400, but this is not assumed), 
    # with one panel at the end with the remainder;
    # however, since that time I discovered that at some points it 
    # will write the remainder-sized panel at the *front* of the file, 
    # which of course screws everything up.

    # Now, the method is to just repeat reading panels until finding 
    # a panel with -99 points (sometimes seen as an "EOF-like" marker; 
    # reading to EOF; or reading a panel with 0 points, or reaching EOF.
    
    f = open(lbl_file, 'rb')
    f.seek(hdr_size)
    
    # read panels until "end condition" is met: either nonpositive 
    # number of points in panel, or reached EOF. This code 
    # should even catch a case of no panels (if the initial n_pts 
    # count is already nonpositive)    
    
    file_size = os.stat(lbl_file).st_size
    n_panels = 0
    total_n_pts = 0
    
    at_EOF = f.tell() == file_size
    at_last_panel = False
    
    while not at_last_panel and not at_EOF:
        
        # unpack string data into binary, and discard the first and last values
        # since those are junk values.
        
        unpack_fmt = '=' + junk_type + 'dd' + float_type + int_type + junk_type
        size = struct.calcsize(unpack_fmt)
        data = struct.unpack(unpack_fmt,f.read(size))
        
        panel_hdr = data[1:-1]
        n_pts_in_panel = panel_hdr[3]
        if n_pts_in_panel > 0:
            total_n_pts += n_pts_in_panel
            n_panels += 1
            
            if panel_format == 'single':
                panel_fmt = junk_type + n_pts_in_panel*float_type + junk_type
            else:
                panel_fmt = junk_type + n_pts_in_panel*float_type + junk_type + junk_type + panel_hdr[3]*float_type + junk_type
            size = struct.calcsize(panel_fmt)
            f.seek(size,1)
        else:
            at_last_panel = True
        
        # check if at EOF
        at_EOF = f.tell() == file_size
        
    n_pts = total_n_pts
    
    f.close()
    
    if do_load_data:
        f = open(lbl_file, 'rb')
        f.seek(hdr_size)
        
        # allocate numpy arrays for efficiency
        panel_dtype = np.dtype([ ('v1', 'float64'),
                                 ('v2', 'float64'),
                                 ('dv', 'float64'),
                                 ('n_pts', 'int') ])
        
        panel_hdrs = np.zeros(n_panels, panel_dtype)
        data1 = np.zeros(n_pts, 'float32')
        if panel_format == 'double':
            data2 = np.zeros(n_pts, 'float32')
        
        ct = 0
        for n in range(n_panels):
            # unpack string data into binary, and discard the first and last values
            # since those are junk values.
        
            unpack_fmt = '=' + junk_type + 'dd' + float_type + int_type + junk_type
            size = struct.calcsize(unpack_fmt)
            data = struct.unpack(unpack_fmt,f.read(size))
            panel_hdr = data[1:-1]
            panel_hdrs['v1'][n] = panel_hdr[0]
            panel_hdrs['v2'][n] = panel_hdr[1]
            panel_hdrs['dv'][n] = panel_hdr[2]
            panel_hdrs['n_pts'][n] = panel_hdr[3]
            
            if panel_format == 'single':
                panel_fmt = junk_type + panel_hdr[3]*float_type + junk_type
            else:
                panel_fmt = junk_type + panel_hdr[3]*float_type + junk_type + junk_type + panel_hdr[3]*float_type + junk_type
            size = struct.calcsize(panel_fmt)
            raw_bytes = f.read(size)
            data = struct.unpack(panel_fmt, raw_bytes)
            # get rid of junk bytes
            if panel_format == 'single':
                data = [data[1:panel_hdr[3]+1]]
            else:
                data = [data[1:panel_hdr[3]+1], data[panel_hdr[3]+3:2*panel_hdr[3]+3]]
            
            data1[ct:ct+panel_hdr[3]] = data[0]
            if panel_format == 'double':
                data2[ct:ct+panel_hdr[3]] = data[1]
                
            ct = ct + panel_hdr[3]
        
        vmin = panel_hdrs['v1'][0]
        vmax = panel_hdrs['v2'][-1]
        dv = panel_hdrs['dv'][0]
        v = np.linspace(vmin,vmax,n_pts)
        
        if panel_format == 'double':
            return np.hstack([data1,data2]), v
        else:
            return data1, v
        
################################################################################
# This routine reads the radiance/transmittance data found in the 
# TAPE27/TAPE28 output files generated by the LBLRTM. Part of the code comes
# from Greg Blumburgs pyLBLRTM readTAPE27 package
################################################################################

def read_tape27(filen = 'TAPE27'):
    
    if not os.path.exists(filen):
        print('Unable to find ' + filen)
        return
    
    fn = open(filen,'r')
    
    rad = []
    wnum = []
    
    for i, line in enumerate(fn.readlines()):
        if i > 26:
            parsed = line.split()
            rad.append(np.float(parsed[1]))
            wnum.append(np.float(parsed[0]))
            
    fn.close()
    
    return np.asarray(wnum), np.asarray(rad)

################################################################################
# This function runs the MonoRTM (via the command passed in) and reads the 
# output into a structure. Note this requires monortm_v5 that outputs
# the profile of gaseous optical depth also.
################################################################################

def run_monortm(command, freq, z, stdatmos, outputfile):
    
    # Define the error status
    error = {'status':0}
    
    # Run the command
    newcommand = '('+command+') >& /dev/null'
    # Logic to catch various errors that could occur when we spawn off the Monortm runs
    try:
        with Popen(newcommand, stdout=PIPE, stderr=PIPE, shell=True, executable = '/bin/csh') as process:
            stdout, stderr = process.communicate()
                # Catch cases where the process launched, but Monortm exited with an error
            if process.returncode != 0:
                print(f"Monortm execution failed with exit code {process.returncode}")
                print(f"Stderr output: {stderr.decode('utf-8', errors='replace')}")
                return error
        # Catches failures occurring during process creation
    except OSError as e:
        if e.errno == errno.EAGAIN:
            print("Popen error with Monortm: Out of PIDs or process limit reached (EAGAIN).")
        elif e.errno == errno.ENOMEM:
            print("Popen error with Monortm: Out of system memory (ENOMEM).")
        elif isinstance(e, FileNotFoundError):
            print("Popen error with Monortm: C-shell executable '/bin/csh' was not found.")
        else:
            print(f"Popen OS error with Monortm ({e.errno}): {e.strerror}")
        return error
        # Catches non-OS exceptions (e.g., KeyboardInterrupt or I/O errors during communication)
    except Exception as e:
        print(f"Unexpected error running Monortm: {e}")
        return error

    # Make sure the model created an output file; if not, the return error
    file_path = Path(outputfile)
    if not file_path.is_file():
        print('    Error: run_monortm did not run properly')
        return error

    # Read in the output file into "stdout"
    f = open(outputfile)
    stdout = f.readlines()
    f.close()
    # Process / parse the output
    for i in range(len(stdout)):
        stdout[i] = stdout[i].strip('\n').strip()
    stdout = np.array(stdout)
    if len(stdout) < (14 + len(freq) + len(z)):
        print('Error: the MonoRTM calculation was not successful')
        return error
    
    nht = len(stdout) - (14 + len(freq))
    
    # Skip over the header
    for i in range(len(stdout)):
        if len(stdout[i]) >= 4:
           if stdout[i][0:4] == 'Freq':
              break
 
    if i == len(stdout)-1:
        print('Error: unable to find the string "Freq" in the MonoRTM calc -- this is an error')
        return error
    i = i + 1
    
    # Get the brightness temperatures
    nfreq = np.zeros(len(freq))
    tb = np.zeros(len(freq))
    tmr = np.zeros(len(freq))
    for k in range(len(freq)):
        foo = stdout[i+k]
        parts = foo.split()
        if parts[0] == '':
            parts = parts[1:len(parts)]
        nfreq[k] = np.float(parts[0])
        tb[k] = np.float(parts[1])
        tmr[k] = np.float(parts[2])
    
    dell = np.abs(nfreq - freq)
    foo = np.where(dell > 0.001)[0]
    if len(foo) > 0:
        print('Error: The frequency array in the MonoRTM calculation does not match up with desired freqs')
        return error
   
     
    # Now get the optical depth profiles
    i = i + k + 1  + 3
    od = np.zeros((len(freq),nht-1))
    mz = np.zeros((2,nht-1))
    for k in range(nht-1):
        foo = stdout[i+k]
        parts = foo.split()
        parts = np.array(parts)
        if parts[0] == '':
            parts = parts[1:len(parts)]
        mz[:,k] = parts[0:2].astype('float')
        od[:,k] = parts[2:2+len(freq)].astype('float')    
    zz = np.append(mz[0,0], mz[1,:])
    return {'status':1, 'freq':np.copy(freq), 'tb':tb, 'tmr':tmr, 'z':zz, 'od':od}

###################################################################################
# This routine writes out the LBLDIS parameter file. It creates a two-phased
# cloud layer with the profiles that are input
##################################################################################

def write_lbldis_parmfile(parmfile, sza, microwin_file, z, cldlay, Xn,
                          ltau, itau, vip, lblout, sfc_emissivity, ref_wnum):
    
    ncldlay = len(cldlay)
    f = open(parmfile,'w')
    f.write('# Header line (blank)\n')
    f.write('{:0d}\n'.format(vip['nstreams']))
    f.write('{:6.1f} 0.0 1.0\n'.format(sza))
    f.write('180.0\n')
    f.write('-1 0 0 ' + microwin_file +'\n')
    f.write('1\n')
    f.write('{:0d}\n'.format(ncldlay*2))
    for j in range(ncldlay):
        f.write('{:0d} {:6.3f} {:7.3f} {:7.1f} {:9.4f}\n'.format(0,z[cldlay[j]],Xn[1],ref_wnum,ltau[j]))
        f.write('{:0d} {:6.3f} {:7.3f} {:7.1f} {:9.4f}\n'.format(1,z[cldlay[j]],Xn[3],ref_wnum,itau[j]))
    f.write('{:s}\n'.format(lblout))
    f.write(vip['ssf'] + '\n')
    f.write('2\n')
    f.write(vip['lcloud_ssp'] + '\n')
    f.write(vip['icloud_ssp'] + '\n')
    f.write('-1\n')
    foo = np.where(sfc_emissivity[0,:] > 0)[0]
    f.write('{:0d}\n'.format(len(foo)))
    for j in range(len(foo)):
        f.write('{:6.1f} {:5.3f}\n'.format(sfc_emissivity[0,foo[j]],sfc_emissivity[1,foo[j]]))
    f.write('{:0d}\n'.format(vip['solver']))
    f.close()
    return
