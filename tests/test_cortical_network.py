# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2023-03-23 11:53:33
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-03-23 17:55:38

import logging
import matplotlib.pyplot as plt
import numpy as np

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
DC = .5       # (-)
pp = PulsedProtocol(tstim, toffset, PRF, DC, tstart=tstart)

# US stimulation parameters
Fdrive = 500e3  # Hz
Adrive = 100e3  # Pa
US_drive = AcousticDrive(Fdrive, Adrive)

# Simulate network
data, meta = network.simulate(US_drive, pp)

# # Plot comparative membrane charge density profiles
# SectionCompTimeSeries([(data, meta)], 'Qm', network.ids).render(cmap=None)

# Detect spikes and plot spikes raster
tspikes = network.detectSpikes(data)

fig, ax = plt.subplots()
ax.set_xlabel('time (s)')
ax.set_ylabel('node')
ax.set_yticks(np.arange(network.size()))
ax.set_yticklabels(network.ids)
ax.set_ylim(-0.4, network.size() - 0.4)

for i, (node_id, node_spikes) in enumerate(tspikes.items()):
    if len(node_spikes) > 0:
        ax.vlines(node_spikes, i - 0.5, i + 0.5, colors='C3')


plt.show()