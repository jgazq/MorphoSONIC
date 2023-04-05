# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2023-03-23 11:53:33
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-04-05 11:19:05

import logging
import time
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from PySONIC.core import PulsedProtocol, AcousticDrive
from PySONIC.utils import logger
from MorphoSONIC.models import CorticalNodeNetwork

''' Test cortical network model with US stimulation '''

logger.setLevel(logging.INFO)

# Parse command line arguments
parser = ArgumentParser()
parser.add_argument(
    '-n', '--nnodes', type=int, default=48, help='network size')
parser.add_argument(
    '-d', '--disconnect', default=False, action='store_true', 
    help='run simulation in disconnected network')
args = parser.parse_args()

# Sonophore parameters
a = 32e-9
fs = 1.0

# Create cortical network
network = CorticalNodeNetwork(ntot=args.nnodes, a=a, fs=fs, connect=not args.disconnect)

# Plot network connectivity matrix
fig1 = network.plot_connectivity_matrix(hue='presyn-celltype')

# US drive parameters
Fdrive = 500e3  # Hz
Adrive = 30e3  # Pa
US_drive = AcousticDrive(Fdrive, Adrive)

# Pulsing parameters
tstart = 2.  # s
tstim = .2    # s
toffset = 2.  # s
PRF = 100.0    # Hz
DC = .5       # (-)
pp = PulsedProtocol(tstim, toffset, PRF, DC, tstart=tstart)

# Simulate network and record detect spikes
t0 = time.perf_counter()
(data, tspikes), meta = network.simulate(US_drive, pp, record_spikes=True)
tcomp = time.perf_counter() - t0
logger.info(f'simulation completed in {tcomp:.2f} seconds')

# Plot spikes raster and firing rate profiles
fig2 = network.plot_spikes_raster(tspikes, pp)
fig3 = network.plot_firing_rates(tspikes, pp, binsize=.3, groupby='celltype')

# Render figures
plt.show()