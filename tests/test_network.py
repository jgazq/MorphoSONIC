# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-01-13 19:51:33
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-03-22 21:04:21

import logging

from PySONIC.core import PulsedProtocol, ElectricDrive, AcousticDrive
from PySONIC.neurons import getPointNeuron
from PySONIC.test import TestBase
from PySONIC.utils import logger

from MorphoSONIC.plt import SectionCompTimeSeries
from MorphoSONIC.models import Node, DrivenNode, Collection, Network
from MorphoSONIC.net_params import cortical_connections, thalamic_drives
from MorphoSONIC.parsers import TestNetworkParser

''' Create and simulate a small network of nodes. '''

logger.setLevel(logging.INFO)


class TestNetwork(TestBase):

    parser_class = TestNetworkParser

    def runTests(self, testsets, args):
        ''' Run appropriate tests. '''
        for s in args['subset']:
            testsets[s](args['connect'])

    def __init__(self):
        ''' Initialize network components. '''

        # Point-neuron models
        self.pneurons = {k: getPointNeuron(k) for k in ['RS', 'FS', 'LTS']}
        
        # Synaptic connections
        self.connections = cortical_connections

        # Driving currents
        self.idrives = {k: (v * 1e-6) / self.pneurons[k].area for k, v in thalamic_drives.items()}  # mA/m2

        # Pulsing parameters
        tstart = 1.  # s
        tstim = 1.    # s
        toffset = 2.  # s
        PRF = 100.0    # Hz
        DC = .2       # (-)
        self.pp = PulsedProtocol(tstim, toffset, PRF, DC, tstart=tstart)

        # Sonophore parameters
        self.a = 32e-9
        self.fs = 1.0

        # US stimulation parameters
        Fdrive = 500e3  # Hz
        Adrive = 100e3  # Pa
        self.US_drive = AcousticDrive(Fdrive, Adrive)

    def simulate(self, nodes, drives, connect):
        # Create appropriate system
        if connect:
            system = Network(nodes, self.connections)
        else:
            system = Collection(nodes)

        # Simulate system
        data, meta = system.simulate(drives, self.pp)

        # Plot comparative membrane potential and firing rate profiles
        for pltkey in ['Qm', 'FR']:
            SectionCompTimeSeries([(data, meta)], pltkey, system.ids).render(cmap=None)

    def test_el(self, connect):
        ''' Electrical stimulation only '''
        nodes = {k: Node(v) for k, v in self.pneurons.items()}
        EL_drives = {k: ElectricDrive(v) for k, v in self.idrives.items()}
        self.simulate(nodes, EL_drives, connect)

    def test_us(self, connect):
        ''' US stimulation only '''
        nodes = {k: Node(v, a=self.a, fs=self.fs) for k, v in self.pneurons.items()}
        US_drives = {k: self.US_drive for k in self.pneurons.keys()}
        self.simulate(nodes, US_drives, connect)

    def test_combined(self, connect):
        ''' US stimulation with current drive '''
        nodes = {k: DrivenNode(v, self.idrives[k], a=self.a, fs=self.fs) for k, v in self.pneurons.items()}
        US_drives = {k: self.US_drive for k in self.pneurons.keys()}
        self.simulate(nodes, US_drives, connect)


if __name__ == '__main__':
    tester = TestNetwork()
    tester.main()
