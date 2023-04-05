# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-01-13 19:51:33
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-04-05 15:50:02

import logging

from PySONIC.core import PulsedProtocol, AcousticDrive
from PySONIC.neurons import getPointNeuron
from PySONIC.test import TestBase
from PySONIC.utils import logger

from MorphoSONIC.plt import SectionCompTimeSeries
from MorphoSONIC.models import Node, NodePopulation, NodeNetwork, CorticalNodeNetwork
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

        # Sonophore parameters
        self.a = 32e-9
        self.fs = 1.0

        # Synaptic weights (in uS) normalized as in Plaksin 2016)
        syn_weights = {
            'RS': {
                'RS': 0.002,
                'FS': 0.04,
                'LTS': 0.09,
            },
            'FS': {
                'RS': 0.015,
                'FS': 0.135,
                'LTS': 0.86,
            },
            'LTS': {
                'RS': 0.135,
                'FS': 0.02,
            }
        }

        # Generate synaptic connections using synaptic models from
        # (Vierling-Classen et al. 2010) model
        self.connections = []
        for presyn, targets in syn_weights.items():
            for postsyn, w in targets.items():
                self.connections.append((
                    presyn, 
                    postsyn,
                    w,
                    CorticalNodeNetwork.syn_models[presyn][postsyn]
                ))

        # Driving currents
        self.idrives = {k: (v * 1e-6) / self.pneurons[k].area
                        for k, v in CorticalNodeNetwork.thalamic_drives.items()}  # mA/m2

        # Corresponding Node models, with appropriate driving currents
        self.nodes = {}
        for k, v in self.pneurons.items():
            self.nodes[k] = Node(v, a=self.a, fs=self.fs)
            self.nodes[k].setConstantDrive(self.idrives[k])
    
        # Pulsing parameters
        tstart = 1.  # s
        tstim = 1.    # s
        toffset = 2.  # s
        PRF = 100.0    # Hz
        DC = .2       # (-)
        self.pp = PulsedProtocol(tstim, toffset, PRF, DC, tstart=tstart)

        # US stimulation parameters
        Fdrive = 500e3  # Hz
        Adrive = 100e3  # Pa
        self.US_drive = AcousticDrive(Fdrive, Adrive)

    def simulate(self, nodes, drives, connect):
        # Create appropriate system
        system = NodeNetwork(nodes, self.connections) if connect else NodePopulation(nodes)

        # Simulate system and record spikes
        (data, tspikes), meta = system.simulate(drives, self.pp, record_spikes=True)

        # Clear any existing conections
        if isinstance(system, NodeNetwork):
            system.disconnect_all()

        # Plot comparative membrane charge density profiles
        SectionCompTimeSeries([(data, meta)], 'Qm', system.ids).render(cmap=None)

        # Plot spikes raster
        system.plot_spikes_raster(tspikes, self.pp)

    def test_drive(self, connect):
        ''' Thalamic drive only '''
        logger.warning('drive only')
        self.simulate(self.nodes, self.US_drive.updatedX(0.), connect)

    def test_us(self, connect):
        ''' US stimulation only '''
        logger.warning('US only')
        for k in self.nodes.keys():
            self.nodes[k].disableConstantDrive()
        self.simulate(self.nodes, self.US_drive, connect)
        for k in self.nodes.keys():
            self.nodes[k].enableConstantDrive()

    def test_combined(self, connect):
        ''' US stimulation with thalamic drive '''
        logger.warning('drive+US')
        self.simulate(self.nodes, self.US_drive, connect)


if __name__ == '__main__':
    tester = TestNetwork()
    tester.main()
