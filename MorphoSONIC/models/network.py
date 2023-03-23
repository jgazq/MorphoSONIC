# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-01-13 20:15:35
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-03-23 17:42:04

from itertools import product
import random
import numpy as np
import pandas as pd
from neuron import h
from tqdm import tqdm

from PySONIC.neurons import getPointNeuron
from PySONIC.core import Model, Drive, getDriveArray, SpatiallyExtendedTimeSeries
from PySONIC.utils import simAndSave
from PySONIC.postpro import detectSpikes

from ..core.pyhoc import *
from . import Node, DrivenNode
from ..core import NeuronModel
from ..core.synapses import *


class Collection(NeuronModel):

    ''' Generic interface to node collection '''

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

            :param nodes: dictionary of {node id: node object}
        '''
        # Assert consistency of inputs
        ids = list(nodes.keys())
        assert len(ids) == len(set(ids)), 'duplicate node IDs'

        # Assign attributes
        logger.info(f'assigning {len(nodes)} nodes to collection')
        self.nodes = nodes
        self.ids = ids
        self.refnode = self.nodes[self.ids[0]]
        self.pneuron = self.refnode.pneuron

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
        if isinstance(drives, dict):
            return getDriveArray(list(drives.values()))
        else:
            return getDriveArray(drives)

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

            :param drives: single drive object, or dictionary of {node_id: drive object}
            :param pp: pulse protocol object
            :param dt: integration time step for fixed time step method (s)
            :param atol: absolute error tolerance for adaptive time step method.
            :return: output dataframe
        '''
        # Vectorize input drive if needed
        if isinstance(drives, Drive):
            drives = {k: drives for k in self.nodes.keys()}
        # Check validity of input drives 
        if len(drives) != self.size():
            raise ValueError(f'number of drives ({len(drives)}) does not match number of nodes {self.size()}')
        if list(drives.keys()) != self.ids:
            raise ValueError('mismatch ')

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
        if isinstance(meta['drives'], Drive):
            drive_desc = meta['drives'].desc
        else:
            darray = self.getDriveArray(meta['drives']).desc
            drive_desc = darray.desc
        return f'{self}: simulation @ {drive_desc}, {meta["pp"].desc}'
        pass

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

    def detectSpikes(self, data):
        tspikes = {}
        logger.info(f'detecting spikes on {data.size} nodes:')
        for node_id, node_data in tqdm(data.items()):
            ispikes, *_ = detectSpikes(node_data)
            tspikes[node_id] = node_data.time[ispikes.astype(int)]
        return tspikes


class Network(Collection):

    ''' Generic interface to node network '''

    simkey = 'node_network'

    def __init__(self, nodes, connections, presyn_var='Qm'):
        ''' Construct network.

            :param connections: list of (presyn_node_id, postsyn_node_id syn_weight, syn_model)
                for each connection to be instantiated
            :param presyn_var: reference variable for presynaptic threshold detection (Vm or Qm)
        '''
        # Construct node collection
        super().__init__(nodes)
        # Assign attributes
        self.connections = connections
        self.presyn_var = presyn_var
        # Connect nodes
        self.connect_all()
    
    def __repr__(self):
        ''' Explicit naming of the model instance. '''
        return f'{super().__repr__()[:-1]}, {len(self.connections)} connections)'    

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

    def checkConnection(self, presyn_id, postsyn_id, syn_model):
        ''' Check validity of tentative synaptic conection '''
        assert presyn_id in self.ids, f'invalid pre-synaptic node ID: "{presyn_id}"'
        assert postsyn_id in self.ids, f'invalid post-synaptic node ID: "{postsyn_id}"'
        assert isinstance(syn_model, Synapse), f'invalid synapse model: {syn_model}'
    
    @property
    def connections(self):
        return self._connections

    @connections.setter
    def connections(self, value):
        for presyn_id, postsyn_id, _, syn_model in value:
            self.checkConnection(presyn_id, postsyn_id, syn_model)
        self._connections = value

    def connect(self, source_id, target_id, syn_model, syn_weight, delay=None):
        ''' Connect a source node to a target node with a specific synapse model
            and synaptic weight.

            :param source_id: ID of the pre-synaptic node
            :param target_id: ID of the post-synaptic node
            :param syn_model: synapse model
            :param weight: synaptic weight (uS)
            :param delay (optional): synaptic delay (ms)
        '''
        # Assert vaildity of source and target IDs
        for id in [source_id, target_id]:
            assert id in self.ids, f'invalid node ID: "{id}"'
        # Create synapse instance from model, and attach it to target node
        syn = syn_model.attach(self.nodes[target_id])
        # Determine relevant hoc variable for pre-synaptic trigger
        if self.presyn_var == 'Vm':
            hoc_var = f'Vm_{self.nodes[source_id].mechname}'
        else:
            hoc_var = 'v'
        # Generate network-connection between pre and post synaptic nodes
        nc = h.NetCon(
            getattr(self.nodes[source_id].section(0.5), f'_ref_{hoc_var}'),  # trigger variable 
            syn,  # synapse object (already attached to post-synaptic node)
            sec=self.nodes[source_id].section  # pre-synaptic node
        )

        # Normalize synaptic weight according to ratio of assigned vs. theoretical membrane area 
        syn_weight *= self.nodes[target_id].getAreaNormalizationFactor()

        # Assign netcon attributes
        nc.threshold = syn_model.Vthr  # pre-synaptic voltage threshold (mV)
        if delay is None:
            nc.delay = syn_model.delay  # synaptic delay (ms)
        else:
            nc.delay = delay
        nc.weight[0] = syn_weight      # synaptic weight (uS)

        # Append synapse and netcon objects to network class atributes 
        self.syn_objs.append(syn)
        self.netcon_objs.append(nc)
    
    def connect_all(self):
        ''' Form all specific connections between network nodes '''
        logger.info(f'instantiating {len(self.connections)} connections between nodes')
        self.syn_objs = []
        self.netcon_objs = []
        for presyn_node_id, postsyn_node_id, syn_weight, syn_model in self.connections:
            self.connect(presyn_node_id, postsyn_node_id, syn_model, syn_weight)
    
    def clear_connections(self):
        ''' Clear all synapses and network connection objects '''
        self.syn_objs = None
        self.netcon_objs = None


class SmartNetwork(Network):
    ''' Network model with automated generation of nodes and connections '''

    simkey = 'smart_network'

    def __init__(self, ntot, proportions, conn_rates, syn_models, syn_weights, **node_kwargs):
        '''
        Initialization
        
        :param ntot: total number of cells in the network
        :param proportions: dictionary of proportions of each cell type in the network
        :param conn_rates: 2-level dictionary of connection rates for each connection type
        :param syn_models: 2-level dictionary of synapse models for each connection type
        :param syn_weights: 2-level dictionary of synaptic weights (in uS) for each connection type
        :param node_kwargs (optional): additional node initialization parameters
        '''
        # Compute number of cells of each type
        self.ncells = {k: int(np.round(v * ntot)) for k, v in proportions.items()}
        
        # Get reference point neuron for each cell type
        pneurons = {k: getPointNeuron(k) for k in self.ncells.keys()}

        # Instantiate nodes
        nfmt = int(np.ceil(np.log10(max(self.ncells.values()))))
        fmt = f'{{:0{nfmt}}}'
        self.nodesdict = {}
        for k, n in self.ncells.items():
            self.nodesdict[k] = {}
            for i in range(n):
                self.nodesdict[k][f'{k}{fmt.format(i)}'] = DrivenNode(pneurons[k], 0., **node_kwargs)
        logger.info(f'instantiated {sum(self.ncells.values())} nodes ({self.strNodeCount()})')
        
        # Generate connections lists structured by connection types
        self.conndict = {}
        for presyn, targets in conn_rates.items():
            self.conndict[presyn] = {}
            for postsyn, rate in targets.items():
                pairs = self.generate_connection_pairs(presyn, postsyn, rate)
                if len(pairs) > 0:
                    w, model = syn_weights[presyn][postsyn], syn_models[presyn][postsyn]
                    self.conndict[presyn][postsyn] = [(*pair, w, model) for pair in pairs]
        concounts = self.conCountMatrix()
        logger.info(f'generated {concounts.values.sum()} random connection pairs:\n{concounts}')

        # Initialize parent class
        super().__init__(self.get_all_nodes(), self.get_all_connections())
    
    def strNodeCount(self):
        return ', '.join([f'{n} {k} cell{"s" if n > 1 else ""}' for k, n in self.ncells.items()])
    
    def conCountMatrix(self):
        return (pd.DataFrame(
            {k: {kk: len(vv) for kk, vv in v.items()} for k, v in self.conndict.items()})
            .fillna(0)
            .astype(int)
        )

    def __repr__(self):
        ''' Explicit naming of the model instance. '''
        return f'{self.refnode.__class__.__name__}{self.__class__.__name__}({self.strNodeCount()}, {len(self.connections)} connections)'

    def get_nodeids(self, celltype=None):
        '''
        Get list of node IDs
        
        :param celltype (optional): cell type of interest
        :return: list of node IDs
        '''
        if celltype is not None:
            keys = [celltype]
        else:
            keys = list(self.nodesdict.keys()) 
        ids = []
        for k in keys:
            ids = ids + list(self.nodesdict[k].keys())
        return ids

    def generate_connection_pairs(self, presyn, postsyn, rate): 
        '''
        Generate list of connection pairs to be instantiated between two cell types
        
        :param presyn: pre-synaptic cell type
        :param postsyn: post-synaptic cell type
        :param rate: connection rate
        :return: list of (presyn_id, postsyn_id) connections to be instantiated
        '''
        # Extract node IDs of pre- and post-synaptic populations
        pre_ids, post_ids = [self.get_nodeids(celltype=k) for k in [presyn, postsyn]]
        # print(pre_ids, post_ids)
        # Compute all candidate connections between the two populations
        candidates = list(product(pre_ids, post_ids))
        # Remove self-connections
        candidates = list(filter(lambda x: x[0] != x[1], candidates))
        # Compute number of connections to instantiate based on connection rate
        nconns = int(np.round(len(candidates) * rate))
        # Randomly select connections from candidates list
        conns = random.sample(candidates, nconns)
        # Make sure all selected connections are unique
        assert len(conns) == len(set(conns)), 'duplicate connections'
        # Trick: Remove single connections:
        if len(conns) == 1:
            conns = []
        # print(presyn, postsyn, len(candidates), len(conns))
        return conns
    
    def get_all_nodes(self):
        ''' Return nodes as a single-level (node_id: node_obj) dictionary '''
        d = {}
        for ndict in self.nodesdict.values():
            d = {**d, **ndict}
        return d
    
    def get_all_connections(self):
        ''' Return all connections to be instantiated as a single list '''
        l = []
        for projs in self.conndict.values():
            for conns in projs.values():
                l = l + conns
        return l

    @classmethod
    def initFromMeta(cls, meta):
        return Network(cls.getNodesFromMeta(meta), meta['connections'], meta['presyn_var'])
    

class CorticalNetwork(SmartNetwork):
    '''
    Cortical network model as defined in (Vierling-Classen et al., 2010)
    
    Reference: 
    *Vierling-Claassen, D., Cardin, J.A., Moore, C.I., and Jones, S.R. (2010). Computational 
    modeling of distinct neocortical oscillations driven by cell-type selective optogenetic
    drive: separable resonant circuits controlled by low-threshold spiking and fast-spiking 
    interneurons. Front Hum Neurosci 4, 198. 10.3389/fnhum.2010.00198.*
    '''
    simkey = 'cortical_network'
    
    # Proportions of each cell type in the network
    proportions = {
        'RS': .75,
        'FS': .125,
        'LTS': .125
    }

    # Connection rates for each connection type
    conn_rates = {
        'RS': {
            'RS': .06,
            'FS': .43,
            'LTS': .57
        },
        'FS': {
            'RS': .44,
            'FS': .51,
            'LTS': .36
        },
        'LTS': {
            'RS': .35,
            'FS': .61,
            'LTS': .04
        }
    }

    # Constant parameters for excitatory (AMPA) synapses
    AMPA_params = dict(
        tau1=0.1,  # rise time constant (ms)
        tau2=3.0,  # decay time constant (ms)
        E=0.  # pre-synaptic voltage threshold (mV)
    )

    # Constant parameters for inhibitory (GABA) synapses
    GABA_params = dict(
        tau1=0.5,  # rise time constant (ms)
        E=-80.  # threshold potential (mV)   # set to -85 mV in Plaksin 2016
    )

    # RS-to-RS excitatory connections: basic AMPA
    RS_RS_syn = Exp2Synapse(**AMPA_params)

    # RS-to-LTS: AMPA with short-term facilitation mechanism
    RS_LTS_syn = FExp2Synapse(**AMPA_params,
        f=0.2,  # facilitation factor (-)
        tauF=200.0  #  facilitation time constant (ms)
    )

    # RS-to-FS: AMPA with short-term with facilitation & short 
    # + long-term depression mechanisms
    RS_FS_syn = FDExp2Synapse(**AMPA_params,
        f=0.5,  # facilitation factor (-)
        tauF=94.0,  #  facilitation time constant (ms)
        d1=0.46,  # short-term depression factor (-)
        tauD1=380.0,  # short-term depression time constant (ms)
        d2=0.975,  # long-term depression factor (-)
        tauD2=9200.0 # long-term depression time constant (ms)
    )

    # FS projections: GABA-A mechanism (short)
    FS_syn = Exp2Synapse(**GABA_params,
        tau2=8.0,  # decay time constant (ms)
    )

    # LTS projections: GABA-B mechanism (long)
    LTS_syn = Exp2Synapse(**GABA_params,
        tau2=50.0,  # decay time constant (ms)
    )

    # Synapse models for each connection type
    syn_models = {
        'RS': {
            'RS': RS_RS_syn,
            'FS': RS_FS_syn,
            'LTS': RS_LTS_syn
        },
        'FS': {
            'RS': FS_syn,
            'FS': FS_syn,
            'LTS': FS_syn
        },
        'LTS': {
            'RS': LTS_syn,
            'FS': LTS_syn
        }
    }

    # Synaptic weights (in uS) for each connection type
    # TODO: modify weigths to reflect differences in neurons membrane area between
    # (Vierling-Classen et. al, 2010) and (Plaksin et al., 2016)
    syn_weights = {
        'RS': {
            'RS': 0.001635,
            'FS': 0.001316,
            'LTS': 0.00099512
        },
        'FS': {
            'RS': 0.008,
            'FS': 0.023421,
            'LTS': 0.1
        },
        'LTS': {
            'RS': 0.1059,
            'FS': 0.002714
        }
    }

    def __init__(self, ntot, **node_kwargs):
        super().__init__(
            ntot, 
            self.proportions, self.conn_rates, self.syn_models, self.syn_weights,
            **node_kwargs
        )