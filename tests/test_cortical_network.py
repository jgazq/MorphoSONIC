# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2023-03-23 11:53:33
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-03-24 18:18:40

import logging
import time
import matplotlib.pyplot as plt

from PySONIC.core import PulsedProtocol, AcousticDrive
from PySONIC.utils import logger

from MorphoSONIC.models import CorticalNodeNetwork
from MorphoSONIC.plt import SectionCompTimeSeries

''' Test cortical network model '''

logger.setLevel(logging.INFO)

# Sonophore parameters
a = 32e-9
fs = 1.0

# Create cortical network
network = CorticalNodeNetwork(ntot=100, a=a, fs=fs)

# Plot network connectivity matrix
fig = network.plot_connectivity_matrix(hue='presyn-celltype')

# US drive parameters
Fdrive = 500e3  # Hz
Adrive = 100e3  # Pa
US_drive = AcousticDrive(Fdrive, Adrive)

# Pulsing parameters
tstart = 1.  # s
tstim = 1.    # s
toffset = 2.  # s
PRF = 100.0    # Hz
DC = .8       # (-)
pp = PulsedProtocol(tstim, toffset, PRF, DC, tstart=tstart)

# Simulate network and record detect spikes
t0 = time.perf_counter()
(data, tspikes), meta = network.simulate(US_drive, pp, record_spikes=True)
tcomp = time.perf_counter() - t0
logger.info(f'simulation completed in {tcomp:.2f} seconds')

# # Plot comparative membrane charge density profiles
# SectionCompTimeSeries([(data, meta)], 'Qm', network.ids).render(cmap=None)

# Plot spikes raster
fig = network.plotSpikesRaster(tspikes, pp)

# Render figures
plt.show()