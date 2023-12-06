# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2021-06-14 10:48:04
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-03-30 14:11:20

import abc
import numpy as np
from boltons.strutils import cardinalize
from PySONIC.utils import logger, si_format, isWithin
from PySONIC.threshold import threshold
from PySONIC.postpro import detectSpikes
from PySONIC.neurons import getPointNeuron
from PySONIC.core import Lookup

from ..core import SpatiallyExtendedNeuronModel, addSonicFeatures
from ..sources import *
from ..constants import *


class RealBase(SpatiallyExtendedNeuronModel):
    ''' Generic interface for a realistic NEURON model based on the BBP/Aberra cells. '''

    use_equivalent_currents = True

    def __init__(self, compartments, **kwargs):
        ''' Initialization.

            :param nnodes: number of nodes
        '''
        self.compartments = compartments
        self.nodes = self.compartments['node']
        self.soma = self.compartments['soma']
        self.axon = self.compartments['axon']
        self.basal = self.compartments['basal']
        self.apical = self.compartments['apical']
        self.myelin = self.compartments['myelin']
        self.unmyelin = self.compartments['unmyelin']
        self.nnodes = len(self.nodes)
        #super().__init__(**kwargs) #TODO: fix this

    def copy(self):
        other = self.__class__(self.nnodes)
        other.rs = self.rs
        other.pneuron = self.pneuron
        return other

    @property
    def nnodes(self):
        ''' Number of nodes. '''
        return self._nnodes

    @nnodes.setter
    def nnodes(self, value):
        self.set('nnodes', value)

    @property
    def central_ID(self):
        #TODO: change this that the soma is the central ID
        return f'node{self.nnodes // 2}'

    @property
    def R_node_to_node(self):
        raise NotImplementedError

    @property
    def ga_node_to_node(self):
        ''' Node-to-node axial conductance per node unit area (S/cm2). '''
        Ga_node_to_node = 1 / self.R_node_to_node  # S
        Anode = self.nodes[self.central_ID].Am     # cm2
        return Ga_node_to_node / Anode             # S/cm2

    @property
    def nodeIDs(self):
        ''' IDs of the model nodes sections. '''
        return [f'node{i}' for i in range(self.nnodes)]

    @property
    @abc.abstractmethod
    def is_myelinated(self):
        raise NotImplementedError

    @property
    def refsection(self):
        return self.nodes[self.nodeIDs[0]]

    @property
    def nonlinear_sections(self):
        return self.nodes

    @property
    def nodelist(self):
        return list(self.nodes.values())

    def str_geometry(self):
        return f'TODO'

    def str_nodes(self):
        return f'{self.nnodes} {cardinalize("node", self.nnodes)}'

    def __repr__(self):
        return f'{self.__class__.__name__}({self.str_geometry()}, {self.str_nodes()})'

    @property
    def meta(self):
        return {
            'simkey': self.simkey,
            'nnodes': self.nnodes
        }

    @staticmethod
    def getMetaArgs(meta):
        return [meta['nnodes']], {}

    @property
    def modelcodes(self):
        return {
            'simkey': self.simkey,
            'nnodes': self.str_nodes().replace(' ', '')
        }

    def filecodes(self, source, pp, *args):
        codes = super().filecodes(source, pp, *args)
        codes['tstim'] = f'{pp.tstim * 1e3:.2f}ms'
        return codes

    def getXCoords(self):
        return {k: np.array([e[0] for e in self.compartments[k].values()]) for k in self.sections.keys()}
        #return {k: np.array([e[0] for e in self.compartments[k].values()]) for k in self.compartments.keys()}

    def getXBounds(self):
        xcoords = self.getXCoords()
        xmin = min(v.min() for v in xcoords.values())
        xmax = max(v.max() for v in xcoords.values())
        return xmin, xmax

    def getXZCoords(self):
        return {k: np.vstack((v, np.ones(v.size) * self.z)).T
                for k, v in self.getXCoords().items()}  # m

    @abc.abstractmethod
    def isInternodalDistance(self, d):
        raise NotImplementedError

    @property
    def CV_estimate(self):
        raise NotImplementedError

    @property
    def AP_travel_time_estimate(self):
        ''' Estimated AP travel time (assuming excitation at central node). '''
        return (self.length / 2.) / self.CV_estimate  # s

    def getConductionVelocity(self, data, ids=None, out='median'):
        ''' Compute average conduction speed from simulation results.

            :param data: simulation output dataframe
            :return: conduction speed output (m/s).
        '''
        # By default, consider all fiber nodes
        if ids is None:
            ids = self.nodeIDs.copy()

        # Remove end nodes from calculations if present
        for x in [0, -1]:
            if self.nodeIDs[x] in ids:
                ids.remove(self.nodeIDs[x])

        # Compute spikes timing dataframe (based on nspikes majority voting) and
        # update list of relevant sections accordingly
        tspikes = self.getSpikesTimings({id: data[id] for id in ids})  # (nspikes x nnodes)
        ids = list(tspikes.columns.values)  # (nnodes)

        # Get coordinates of relevant nodes
        indexes = [self.nodeIDs.index(id) for id in ids]  # (nnodes)
        xcoords = self.getNodeCoords()[indexes]  # (nnodes)

        # Compute distances across consecutive nodes only, and associated spiking delays
        # for first spike only
        distances, delays = [], []  # (nnodes - 1)
        for i in range(len(ids) - 1):
            d = xcoords[i] - xcoords[i - 1]
            if self.isInternodalDistance(d):
                distances.append(d)
                dt = np.abs(tspikes.values[0][i] - tspikes.values[0][i - 1])
                delays.append(dt)   # node-to-node delay

        # Compute conduction velocities for each considered node pair
        velocities = np.array(distances) / np.array(delays)  # m/s

        # Return specific output metrics
        if out == 'range':
            return velocities.min(), velocities.max()
        elif out == 'median':
            return np.median(velocities)
        elif out == 'mean':
            return np.mean(velocities)
        else:
            raise AttributeError(f'invalid out option: {out}')

    def getEndSpikeTrain(self, data):
        ''' Detect spikes on end node. '''
        ispikes, *_ = detectSpikes(
            data[self.nodeIDs[-1]], key='Vm', mph=SPIKE_MIN_VAMP, mpt=SPIKE_MIN_DT,
            mpp=SPIKE_MIN_VPROM)
        if len(ispikes) == 0:
            return None
        return data.time[ispikes]

    def getEndFiringRate(self, data):
        ''' Compute firing rate from spikes detected on end node. '''
        tspikes = self.getEndSpikeTrain(data)
        if tspikes is None:
            return np.nan
        # return np.mean(1 / np.diff(tspikes))
        return 1 / np.mean(np.diff(tspikes))

    def isExcited(self, data):
        ''' Determine if neuron is excited from simulation output.

            :param data: dataframe containing output time series
            :return: boolean stating whether neuron is excited or not
        '''
        ids = {
            'proximal': self.nodeIDs[0],
            'distal': self.nodeIDs[-1],
            'central': self.central_ID}
        nspikes = {k: detectSpikes(data[v])[0].size for k, v in ids.items()}
        has_spiked = {k: v > 0 for k, v in nspikes.items()}
        has_spiked['ends'] = has_spiked['proximal'] and has_spiked['distal']
        if not has_spiked['ends'] and has_spiked['central']:
            logger.warning('AP did not reach end nodes')
        return has_spiked['ends']

    def checkForConduction(self, pp):
        ''' Check that a protocol should allow for full AP conduction if excitation is reached. '''
        if pp.toffset < REL_AP_TRAVEL_FACTOR * self.AP_travel_time_estimate:
            AP_travel_str = f'estimated AP travel time: {self.AP_travel_time_estimate * 1e3:.1f} ms'
            raise ValueError(f'offset duration too short for full AP conduction ({AP_travel_str})')

    def titrate(self, source, pp):
        ''' Use a binary search to determine the threshold amplitude needed to obtain
            neural excitation for a given pulsing protocol.

            :param source: source object
            :param pp: pulsed protocol object
            :return: determined threshold amplitude
        '''
        # Check that protocol should allow for full AP conduction, if any
        self.checkForConduction(pp)

        # Run titration procedure
        Arange = self.getArange(source)
        xthr = threshold(
            lambda x: self.titrationFunc(
                self.simulate(source.updatedX(-x if source.is_cathodal else x), pp)[0]),
            Arange,
            x0=self.getStartPoint(Arange),
            eps_thr=self.getAbsConvThr(Arange),
            rel_eps_thr=REL_EPS_THR,
            precheck=source.xvar_precheck)
        if source.is_cathodal:
            xthr = -xthr
        return xthr



class RealNeuronModel(RealBase):
    ''' Generic interface for a realistic NEURON model based on the BBP/Aberra cells. '''

    simkey = 'RealOne'
    is_myelinated = True
    _pneuron = getPointNeuron('realneuron')

    def copy(self):
        other = super().copy()
        return other

    def clearSections(self):
        ''' delete all model sections. '''
        self.nodes = None

    def isInternodalDistance(self, d):
        return np.isclose(d, (self.nodeL + self.interL))

    @property
    def sections(self):
        return {'soma': self.soma, 'axon': self.axon, 'apical': self.apical, 'basal': self.basal, 'node': self.nodes, 'myelin': self.myelin, 'unmyelin': self.unmyelin}

    @property
    def seclist(self):
        return self.nodelist

    def createSections(self):
        self.nodes = {k: self.createSection(
            k, mech=self.mechname, states=self.pneuron.statesNames()) for k in self.nodeIDs}

    def setGeometry(self):
        logger.debug(f'defining sections geometry: {self.str_geometry()}')
        for sec in self.nodelist:
            sec.setGeometry(self.nodeD, self.nodeL)

    def setResistivity(self):
        logger.debug(f'nominal nodal resistivity: rs = {self.rs:.0f} Ohm.cm')
        rho_nodes = np.ones(self.nnodes) * self.rs  # Ohm.cm

        # Adding extra resistivity to account for half-internodal resistance
        # for each connected side of each node
        if self.R_inter > 0:
            logger.debug('adding extra-resistivity to account for internodal resistance')
            rho_extra = self.R_extra * self.rs / self.R_node  # Ohm.cm
            rho_nodes += rho_extra  # Ohm.cm

        # Assigning resistivities to sections
        for sec, rho in zip(self.nodelist, rho_nodes):
            sec.setResistivity(rho)

    @property
    def R_extra(self):
        ''' Vector of extra internodal resistances to add to nodal axial resistances. '''
        return np.ones(self.nnodes) * self.R_inter  # Ohm

    def setTopology(self):
        for i in range(self.nnodes - 1):
            self.connect('node', i, 'node', i + 1)

    def toInjectedCurrents(self, Ve):
        Iinj = np.diff(Ve['node'], 2) / (self.R_node + self.R_inter) * MA_TO_NA  # nA
        return {'node': np.pad(Iinj, (1, 1), 'constant')}  # zero-padding on both extremities