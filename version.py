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


################################################################################
# This function is called to get the software version information, 
# and this string is updated via the command in get_version.csh, which is 
# triggered by a post-commit hook

def get_software_version():

   TROPoe_software_version = '1.0-4-gd2f49bb'

   return TROPoe_software_version
