# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-01-13 20:15:35
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-03-22 15:58:30

import pandas as pd
from neuron import h

from PySONIC.neurons import getPointNeuron
from PySONIC.core import Model, getDriveArray
from PySONIC.utils import si_prefixes, filecode, simAndSave
from PySONIC.postpro import prependDataFrame
from PySONIC.core.timeseries import SpatiallyExtendedTimeSeries

from ..core.pyhoc import *
from . import Node, DrivenNode
from ..core import NeuronModel
from ..core.synapses import *


prefix_map = {v: k for k, v in si_prefixes.items()}


class NodeCollection(NeuronModel):

    ''' Collection of node models to be simulated '''

    simkey = 'node_collection'
    tscale = 'ms'  # relevant temporal scale of the model
    titration_var = None

    node_constructor_dict = {
        'ESTIM': (Node, [], []),
        'ASTIM': (Node, [], ['a', 'fs']),
        'DASTIM': (DrivenNode, ['Idrive'], ['a', 'fs']),
    }

    def __init__(self, nodes):
        ''' Constructor.

            :param nodes: dictionary of node objects
        '''
        # Assert consistency of inputs
        ids = list(nodes.keys())
        assert len(ids) == len(set(ids)), 'duplicate node IDs'

        # Assign attributes
        self.nodes = nodes
        self.ids = ids
        self.refnode = self.nodes[self.ids[0]]
        self.pneuron = self.refnode.pneuron
        
        # # Deduce drive type (US or EL) from input node type 
        # if self.refnode.a is not None:
        #     unit, factor = 'Pa', 1e3
        # else:
        #     unit, factor = 'A/m2', 1e-3
        # self.unit = f'{prefix_map[factor]}{unit}'

    def strNodes(self):
        ''' String representation for node list '''
        return f"[{', '.join([repr(x.pneuron) for x in self.nodes.values()])}]"

    def __repr__(self):
        ''' Explicit naming of the model instance. '''
        return f'{self.refnode.__class__.__name__}{self.__class__.__name__}({self.strNodes()})'

    def __getitem__(self, key):
        return self.nodes[key]

    def __delitem__(self, key):
        del self.nodes[key]

    def __setitem__(self, key, value):
        self.nodes[key] = value

    def clear(self):
        for node in self.nodes.values():
            node.clear()

    def size(self):
        return len(self.nodes)

    @classmethod
    def getNodesFromMeta(cls, meta):
        node_class, node_args, node_kwargs = cls.node_constructor_dict[meta['nodekey']]
        nodes = {}
        for k, v in meta['nodes'].items():
            pneuron = getPointNeuron(v['neuron'])
            node_args = [v[x] for x in node_args]
            node_kwargs = {x: v[x] for x in node_kwargs}
            nodes[k] = node_class(pneuron, *node_args, **node_kwargs)
        return nodes

    @classmethod
    def initFromMeta(cls, meta):
        return cls(cls.getNodesFromMeta(meta))

    def inputs(self):
        return self.refnode.pneuron.inputs()

    def setStimValue(self, value):
        for node in self.nodes.values():
            node.setStimValue(value)

    def setDrives(self, drives):
        for id, node in self.nodes.items():
            node.setDrive(drives[id])
    
    def clearDrives(self):
        for node in self.nodes.values():
            node.clearDrives()
    
    def getDriveArray(self, drives):
        return getDriveArray(list(drives.values()))

    def createSections(self):
        pass

    def clearSections(self):
        pass

    def seclist(self):
        pass

    def initToSteadyState(self):
        ''' Initialize model variables to pre-stimulus resting state values. '''
        for id, node in self.nodes.items():
            if self.refvar == 'Qm':
                x0 = node.pneuron.Qm0 * C_M2_TO_NC_CM2  # nC/cm2
            else:
                x0 = self.pneuron.Vm0  # mV
            node.section.v = x0
        h.finitialize()
        self.resetIntegrator()
        h.frecord_init()

    @Model.addMeta
    @Model.logDesc
    def simulate(self, drives, pp, dt=None, atol=None):
        ''' Set appropriate recording vectors, integrate and return output variables.

            :param drives: drive dictionary with node ids as keys
            :param pp: pulse protocol object
            :param dt: integration time step for fixed time step method (s)
            :param atol: absolute error tolerance for adaptive time step method.
            :return: output dataframe
        '''
        if len(drives) != self.size():
            raise ValueError(f'number of drives ({len(drives)}) does not match number of nodes {self.size()}')
        if list(drives.keys()) != self.ids:
            raise ValueError('mismatch ')
        
        # logger.info(self.desc(self.meta(drives, pp)))

        # Set recording vectors
        t = self.refnode.setTimeProbe()
        stim = self.refnode.section.setStimProbe()
        probes = {k: v.section.setProbes() for k, v in self.nodes.items()}

        # Set distributed stimulus amplitudes
        self.setDrives(drives)

        # Integrate model
        self.integrate(pp, dt, atol)
        self.clearDrives()

        # Return output dataframe dictionary
        t = t.to_array()  # s
        stim = self.fixStimVec(stim.to_array(), dt)
        return SpatiallyExtendedTimeSeries({
            id: self.outputDataFrame(t, stim, probes) for id, probes in probes.items()})

    @property
    def meta(self):
        return {
            'simkey': self.simkey,
            'nodes': {k: v.meta for k, v in self.nodes.items()},
            'nodekey': self.refnode.simkey
        }

    def desc(self, meta):
        darray = self.getDriveArray(meta['drives'])
        return f'{self}: simulation @ {darray.desc}, {meta["pp"].desc}'

    def modelCodes(self):
        return {
            'simkey': self.simkey,
            'neurons': '_'.join([x.pneuron.name for x in self.nodes.values()])
        }

    def filecodes(self, drives, pp, *_):
        return {
            **self.modelCodes(),
            **self.getDriveArray(drives).filecodes,
            'nature': 'CW' if pp.isCW else 'PW',
            **pp.filecodes
        }

    def getPltVars(self, *args, **kwargs):
        ref_pltvars = self.refnode.pneuron.getPltVars(*args, **kwargs)
        keys = set(ref_pltvars.keys())
        for node in self.nodes.values():
            node_keys = list(node.pneuron.getPltVars(*args, **kwargs).keys())
            keys = keys.intersection(node_keys)
        return {k: ref_pltvars[k] for k in keys}

    def getPltScheme(self, *args, **kwargs):
        ref_pltscheme = self.refnode.pneuron.getPltScheme(*args, **kwargs)
        keys = set(ref_pltscheme.keys())
        for node in self.nodes.values():
            node_keys = list(node.pneuron.getPltScheme(*args, **kwargs).keys())
            keys = keys.intersection(node_keys)
        return {k: ref_pltscheme[k] for k in keys}

    def simAndSave(self, *args, **kwargs):
        return simAndSave(self, *args, **kwargs)


class NodeNetwork(NodeCollection):

    simkey = 'node_network'

    def __init__(self, nodes, connections, presyn_var='Qm'):
        ''' Construct network.

            :param nodes: dictionary of node objects
            :param connections: {presyn_node: postsyn_node} dictionary of (syn_weight, syn_model)
            :param presyn_var: reference variable for presynaptic threshold detection (Vm or Qm)
        '''
        # Construct node collection
        super().__init__(nodes)

        # Assert consistency of inputs
        for presyn_node_id, targets in connections.items():
            assert presyn_node_id in self.ids, f'invalid pre-synaptic node ID: "{presyn_node_id}"'
            for postsyn_node_id, (syn_weight, syn_model) in targets.items():
                assert postsyn_node_id in self.ids, f'invalid post-synaptic node ID: "{postsyn_node_id}"'
                assert isinstance(syn_model, Synapse), f'invalid synapse model: {syn_model}'

        # Assign attributes
        self.connections = connections
        self.presyn_var = presyn_var

        # Connect nodes
        self.syn_objs = []
        self.netcon_objs = []
        for presyn_node_id, targets in self.connections.items():
            for postsyn_node_id, (syn_weight, syn_model) in targets.items():
                self.connect(presyn_node_id, postsyn_node_id, syn_model, syn_weight)
    
    @property
    def meta(self):
        return {
            **super().meta,
            'simkey': self.simkey,
            'connections': self.connections,
            'presyn_var': self.presyn_var
        }

    @classmethod
    def initFromMeta(cls, meta):
        return cls(cls.getNodesFromMeta(meta), meta['connections'], meta['presyn_var'])

    def connect(self, source_id, target_id, syn_model, syn_weight, delay=0.0):
        ''' Connect a source node to a target node with a specific synapse model
            and synaptic weight.

            :param source_id: ID of the pre-synaptic node
            :param target_id: ID of the post-synaptic node
            :param syn_model: synapse model
            :param weight: synaptic weight (uS)
            :param delay: synaptic delay (ms)
        '''
        for id in [source_id, target_id]:
            assert id in self.ids, f'invalid node ID: "{id}"'
        syn = syn_model.attach(self.nodes[target_id])
        if self.presyn_var == 'Vm':
            hoc_var = f'Vm_{self.nodes[source_id].mechname}'
        else:
            hoc_var = 'v'
        nc = h.NetCon(
            getattr(self.nodes[source_id].section(0.5), f'_ref_{hoc_var}'),
            syn,
            sec=self.nodes[source_id].section)

        # Normalize synaptic weight
        syn_weight *= self.nodes[target_id].getAreaNormalizationFactor()

        # Assign netcon attributes
        nc.threshold = syn_model.Vthr  # pre-synaptic voltage threshold (mV)
        nc.delay = syn_model.delay     # synaptic delay (ms)
        nc.weight[0] = syn_weight          # synaptic weight (uS)

        self.syn_objs.append(syn)
        self.netcon_objs.append(nc)

