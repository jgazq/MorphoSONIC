# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2023-03-22 10:16:44
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-03-22 10:22:54

import logging
import matplotlib.pyplot as plt

from PySONIC.core import PulsedProtocol, AcousticDrive
from PySONIC.neurons import getPointNeuron
from PySONIC.utils import logger
from PySONIC.plt import GroupedTimeSeries
from MorphoSONIC.models import DrivenNode

# Set logging level
logger.setLevel(logging.INFO)

# Define point-neuron model
pneuron = getPointNeuron('RS')

# Create driven node model
a = 32e-9  # sonophore radius (m)
fs = 1.    # sonophore coverage fraction (-)
Idrive = -7.0  # driving current 
node = DrivenNode(pneuron, Idrive, a=a, fs=fs)

# Define ultrasonic drive
Fdrive = 500e3  # Hz
Adrive = 100e3  # Pa
US_drive = AcousticDrive(Fdrive, Adrive)

# Set pulsing protocol
tstim = 250e-3   # s
toffset = 50e-3  # s
PRF = 100.       # Hz
DC = 0.5         # -
pp = PulsedProtocol(tstim, toffset, PRF, DC)

# Simulate model and plot results
data, meta = node.simulate(US_drive, pp)
GroupedTimeSeries([(data, meta)]).render()

# Show figures
plt.show()
