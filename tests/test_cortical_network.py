# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2023-03-23 11:53:33
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-03-23 17:18:17

import logging
import matplotlib.pyplot as plt

from PySONIC.core import PulsedProtocol, AcousticDrive
from PySONIC.utils import logger

from MorphoSONIC.models import CorticalNetwork
from MorphoSONIC.plt import SectionCompTimeSeries

''' Test cortical network model '''

logger.setLevel(logging.INFO)

# Sonophore parameters
a = 32e-9
fs = 1.0

# Cortical network
network = CorticalNetwork(ntot=20, a=a, fs=fs)

# Pulsing parameters
tstart = 1.  # s
tstim = 1.    # s
toffset = 2.  # s
PRF = 100.0    # Hz
DC = .2       # (-)
pp = PulsedProtocol(tstim, toffset, PRF, DC, tstart=tstart)

# US stimulation parameters
Fdrive = 500e3  # Hz
Adrive = 100e3  # Pa
US_drive = AcousticDrive(Fdrive, Adrive)

# Simulate network
data, meta = network.simulate(US_drive, pp)
# Plot comparative membrane charge density profiles
SectionCompTimeSeries([(data, meta)], 'Qm', network.ids).render(cmap=None)

plt.show()