# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-01-13 19:51:33
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-03-22 16:01:10

import logging

from PySONIC.core import PulsedProtocol, ElectricDrive, AcousticDrive
from PySONIC.neurons import getPointNeuron
from PySONIC.test import TestBase
from PySONIC.utils import logger

from MorphoSONIC.plt import SectionCompTimeSeries
from MorphoSONIC.models import Node, DrivenNode
from MorphoSONIC.core.synapses import Exp2Synapse, FExp2Synapse, FDExp2Synapse
from MorphoSONIC.models.network import NodeCollection, NodeNetwork
from MorphoSONIC.parsers import TestNodeNetworkParser

''' Create and simulate a small network of nodes. '''

logger.setLevel(logging.INFO)


class TestNodeNetwork(TestBase):

    parser_class = TestNodeNetworkParser

    def runTests(self, testsets, args):
        ''' Run appropriate tests. '''
        for s in args['subset']:
            testsets[s](args['connect'])

    def __init__(self):
        ''' Initialize network components. '''

        # Point-neuron models
        self.pneurons = {k: getPointNeuron(k) for k in ['RS', 'FS', 'LTS']}

        # Synapse models
        RS_syn_base = Exp2Synapse(tau1=0.1, tau2=3.0, E=0.0)
        RS_LTS_syn = FExp2Synapse(
            tau1=RS_syn_base.tau1, tau2=RS_syn_base.tau2, E=RS_syn_base.E, f=0.2, tauF=200.0)
        RS_FS_syn = FDExp2Synapse(
            tau1=RS_syn_base.tau1, tau2=RS_syn_base.tau2, E=RS_syn_base.E, f=0.5, tauF=94.0,
            d1=0.46, tauD1=380.0, d2=0.975, tauD2=9200.0)
        FS_syn = Exp2Synapse(tau1=0.5, tau2=8.0, E=-85.0)
        LTS_syn = Exp2Synapse(tau1=0.5, tau2=50.0, E=-85.0)

        # Synaptic connections
        self.connections = {
            'RS': {
                'RS': (0.002, RS_syn_base),
                'FS': (0.04, RS_FS_syn),
                'LTS': (0.09, RS_LTS_syn)
            },
            'FS': {
                'RS': (0.015, FS_syn),
                'FS': (0.135, FS_syn),
                'LTS': (0.86, FS_syn)
            },
            'LTS': {
                'RS': (0.135, LTS_syn),
                'FS': (0.02, LTS_syn)
            }
        }

        # Driving currents
        I_Th_RS = 0.17  # nA
        Idrives = {  # nA
            'RS': I_Th_RS,
            'FS': 1.4 * I_Th_RS,
            'LTS': 0.0}
        self.idrives = {k: (v * 1e-6) / self.pneurons[k].area for k, v in Idrives.items()}  # mA/m2

        # Pulsing parameters
        tstim = 2.0    # s
        toffset = 1.0  # s
        PRF = 100.0    # Hz
        DC = 1.0       # (-)
        self.pp = PulsedProtocol(tstim, toffset, PRF, DC)

        # Sonophore parameters
        self.a = 32e-9
        self.fs = 1.0

        # US stimulation parameters
        Fdrive = 500e3  # Hz
        Adrive = 30e3  # Pa
        self.US_drive = AcousticDrive(Fdrive, Adrive)

    def simulate(self, nodes, drives, connect):
        # Create appropriate system
        if connect:
            system = NodeNetwork(nodes, self.connections)
        else:
            system = NodeCollection(nodes)

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
    tester = TestNodeNetwork()
    tester.main()
