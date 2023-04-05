# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-01-13 20:15:35
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-04-05 16:29:29

from itertools import product
import random
import re
import numpy as np
import pandas as pd
from neuron import h
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import seaborn as sns

from PySONIC.neurons import getPointNeuron
from PySONIC.core import Model, Drive, getDriveArray, SpatiallyExtendedTimeSeries
from PySONIC.postpro import detectSpikes
from PySONIC.plt import TimeSeriesPlot
from PySONIC.utils import add_indent, isIterable

from ..core.pyhoc import *
from . import Node
from ..core import NeuronModel
from ..core.synapses import *


class NodePopulation(NeuronModel):

    ''' Generic interface to population of nodes '''

    simkey = 'node_pop'
    tscale = 'ms'  # relevant temporal scale of the model
    titration_var = None
    nodeid_pattern = '([a-zA-Z]+)(?:([0-9]+))?'  # Regexp node ID pattern
    node_constructor_dict = {
        'ESTIM': (Node, [], []),
        'ASTIM': (Node, [], ['a', 'fs'])
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

    def str_nodes(self):
        ''' String representation for node list '''
        return f"[{', '.join([repr(x.pneuron) for x in self.nodes.values()])}]"

    def __repr__(self):
        ''' Explicit naming of the model instance. '''
        return f'{self.__class__.__name__}({self.str_nodes()})'

    def __getitem__(self, key):
        ''' Get node by its reference id '''
        return self.nodes[key]

    def __delitem__(self, key):
        ''' Delete node by its reference id '''
        del self.nodes[key]

    def __setitem__(self, key, value):
        ''' Set node by its reference id '''
        self.nodes[key] = value

    def clear(self):
        ''' Clear all constituent nodes '''
        for node in self.nodes.values():
            node.clear()

    def size(self):
        ''' Return number of nodes '''
        return len(self.nodes)

    @classmethod
    def getNodesFromMeta(cls, meta):
        ''' Initialize nodes from meta dictionary '''
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
        ''' Initialize population from meta dictionary '''
        return cls(cls.getNodesFromMeta(meta))

    def inputs(self):
        ''' Return population inputs '''
        return self.refnode.pneuron.inputs()

    def setStimValue(self, value):
        ''' Set stim value for each constituent node '''
        for node in self.nodes.values():
            node.setStimValue(value)

    def setDrives(self, drives):
        ''' Set drives for each constituent node '''
        for id, node in self.nodes.items():
            node.setDrive(drives[id])
    
    def clearDrives(self):
        ''' Clear drives for each constituent node '''
        for node in self.nodes.values():
            node.clearDrives()

    def createSections(self):
        ''' Null override of parent abstract method '''
        pass

    def clearSections(self):
        ''' Null override of parent abstract method '''
        pass

    def seclist(self):
        ''' Null override of parent abstract method '''
        pass

    @classmethod
    def parse_cell_type(cls, node_id):
        ''' Parse cell type from node id '''
        if isIterable(node_id):
            return [cls.parse_cell_type(x) for x in node_id]
        mo = re.match(cls.nodeid_pattern, node_id)
        if mo is None:
            raise ValueError(f'"{node_id}" does not match node ID pattern: {cls.nodeid_pattern}')
        return mo.group(1)

    def parse_cell_types(self, unique=False):
        '''
        Parse cell types from constituent node IDs
        
        :param unique: whether to only return unique cell types
        :return: list of cell types
        '''
        ctypes = map(self.parse_cell_type, self.ids)
        if unique:
            ctypes = dict.fromkeys(ctypes)
        return list(ctypes)

    def initToSteadyState(self):
        ''' Initialize model variables to pre-stimulus resting state values. '''
        for node in self.nodes.values():
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
    def simulate(self, drives, pp, dt=None, atol=None, record_spikes=False):
        ''' Set appropriate recording vectors, integrate and return output variables.

            :param drives: single drive object, or dictionary of {node_id: drive object}
            :param pp: pulse protocol object
            :param dt: integration time step for fixed time step method (s)
            :param atol: absolute error tolerance for adaptive time step method.
            :param record_spikes: boolean specifying whether or not to record detected spikes
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

        # Set spike detectors, if specified
        if record_spikes:
            for node in self.nodes.values():
                node.setSpikeDetector()

        # Set distributed stimulus amplitudes
        self.setDrives(drives)

        # Integrate model
        self.integrate(pp, dt, atol)
        self.clearDrives()

        # Extract and format output timeseries data 
        t = t.to_array()  # s
        stim = self.fixStimVec(stim.to_array(), dt)
        data = SpatiallyExtendedTimeSeries({
            id: self.outputDataFrame(t, stim, probes) for id, probes in probes.items()})
        
        # Extract detected spike times and clear spike detectors, if specified
        if record_spikes:
            logger.info(f'extracting detected spikes on {data.size} nodes')
            tspikes = {id: node.getSpikeTimes() for id, node in self.nodes.items()}
            for node in self.nodes.values():
                node.clearSpikedetector()

        # Return output(s)
        if record_spikes:
            return data, tspikes
        else:
            return data

    @property
    def meta(self):
        ''' Return dictionary of meta information about the model '''
        return {
            'simkey': self.simkey,
            'nodes': {k: v.meta for k, v in self.nodes.items()},
            'nodekey': self.refnode.simkey
        }

    def desc(self, meta):
        ''' Describe model and simulation based on meta-information dictionary '''
        if isinstance(meta['drives'], Drive):
            drive_desc = meta['drives'].desc
        else:
            darray = getDriveArray(meta['drives']).desc
            drive_desc = darray.desc
        return f'{self}: simulation @ {drive_desc}, {meta["pp"].desc}'

    def filecodes(self, drives, pp, *_):
        return {
            'simkey': self.simkey,
            'nodes': '_'.join([x.pneuron.name for x in self.nodes.values()]),
            **getDriveArray(drives).filecodes,
            'nature': 'CW' if pp.isCW else 'PW',
            **pp.filecodes
        }

    def getPltVars(self, *args, **kwargs):
        ''' Get union of plot variables of constituent nodes '''
        ref_pltvars = self.refnode.pneuron.getPltVars(*args, **kwargs)
        keys = set(ref_pltvars.keys())
        for node in self.nodes.values():
            node_keys = list(node.pneuron.getPltVars(*args, **kwargs).keys())
            keys = keys.intersection(node_keys)
        return {k: ref_pltvars[k] for k in keys}

    def getPltScheme(self, *args, **kwargs):
        ''' Get intersection of plot schemes of constituent nodes '''
        ref_pltscheme = self.refnode.pneuron.getPltScheme(*args, **kwargs)
        keys = set(ref_pltscheme.keys())
        for node in self.nodes.values():
            node_keys = list(node.pneuron.getPltScheme(*args, **kwargs).keys())
            keys = keys.intersection(node_keys)
        return {k: ref_pltscheme[k] for k in keys}

    def detectSpikes(self, data):
        ''' Detect spikes on timeseries of each constituent nodes '''
        tspikes = {}
        logger.info(f'detecting spikes on {data.size} nodes:')
        for node_id, node_data in tqdm(data.items()):
            ispikes, *_ = detectSpikes(node_data)
            tspikes[node_id] = node_data.time[ispikes.astype(int)]
        return tspikes

    def compute_firing_rates(self, tspikes, protocol, binsize=None, groupby=None):
        '''
        Compute firing rates from spikes timings 

        :param tspikes: dictionary of spikes timings
        :param binsize (optional): binning time interval for spike counting (defaults to 10% of total stimulation time)
        :param groupby (optional): grouping variable for firing rate computation (e.g. "celltype"). If no
            grouping variable is provided, firing rates are index by node ID
        :return: pandas firing rate object, indexed by time and grouping variable (or node ID).
            - If no grouping variable is used, a simple pandas Series is returned.
            - If a grouping variable is used, a pandas DataFrame is returned with mean and standard error columns
        '''
        # Compute bin edges and mid points
        if binsize is None:
            binsize = protocol.tstop / 10
        bins = np.arange(0, protocol.tstop + binsize / 2, binsize)
        mids = (bins[:-1] + bins[1:]) / 2

        # Log
        s = 'computing firing rates'
        if groupby is not None:
            s = f'{s} by {groupby}'
        if binsize is not None:
            s = f'{s} in {binsize} s bins'
        logger.info(s)

        # Count spikes in each bin, for each constituent node
        nspikes = {k: np.histogram(v, bins=bins)[0] for k, v in tspikes.items()}

        # Assemble into DataFrame, and stack along node IDs
        tkey = 'time (s)'
        nodekey = 'node ID'
        nspikes = (
            pd.DataFrame(nspikes)
            .assign(**{tkey: mids})
            .set_index(tkey)
            .rename_axis(nodekey, axis='columns')
            .stack()
            .rename('# spikes')
        )

        # Compute corresponding firing rates
        rates = (nspikes / binsize).rename('spike rate (Hz)')

        # Group results by variable, if specified
        if groupby is not None:
            if groupby == 'celltype':
                rates = rates.to_frame()
                rates['celltype'] = self.parse_cell_type(rates.index.get_level_values(nodekey))
                rates = (rates
                    .groupby([tkey, 'celltype'])
                    ['spike rate (Hz)']
                    .agg(['mean', 'sem'])
                    .stack()
                    .unstack()
                )
            else:
                raise ValueError(f'invalid grouping variable: "{groupby}"')
        
        # Return
        return rates
    
    @staticmethod
    def add_stim_mark(ax, protocol, mark_type='auto'):
        '''
        Add stimulus mark on timeseries axis
        
        :param ax: axis object
        :param protocol: protocol object
        :param mark_type (default = "auto"): how to represent stimulus:
            - "all": mark each individual pulse
            - "span": mark overall stimulus span
            - "auto": automatically choose between "all" and "span" depending on stimulus complexity
            - False: no mark
        '''
        # Extract stimulation events from protocol
        evs = protocol.stimEvents()
        # Classify into ON and OFF events
        on_events = list(filter(lambda x: x[1] != 0., evs))
        off_events = list(filter(lambda x: x[1] == 0., evs))
        assert len(on_events) == len(off_events), 'imbalanced protocol'
        # Assemble into pulse list
        pulses = [(ton, toff, val) for (ton, val), (toff, _) in zip(on_events, off_events)]
        # If "auto" option specified
        if mark_type == 'auto':
            # Extract event times, and compute average time delta between each event
            avg_time_delta = np.diff([x[0] for x in evs]).mean()
            # choose between "span" and "all" mode depending on ratio between
            # average time delta and simulation time span
            rel_avg_time_delta = avg_time_delta / protocol.tstop
            mark_type = 'span' if rel_avg_time_delta < .005 else 'all'
        # If "span" mode, reduce protocol to single pulse
        if mark_type == 'span':
            pulses = [(pulses[0][0], pulses[-1][1], pulses[0][2])]
        # Draw patches for each selected pulses
        TimeSeriesPlot.addPatches(ax, pulses, {'factor': 1})
    
    def plot_spikes_raster(self, tspikes, protocol, ymargin=0.2, stim_mark='auto', nodelabel='auto', ax=None):
        '''
        Plot spikes raster for each node, color-coded by cell type
        
        :param tspikes: dictionary of spike times (in s) per node
        :param protocol: pulsing protocol used during the simulation
        :param ymargin (optional): relative vertical margin between reach raster line
        :param stim_mark (optional): how to represent stimulus
        :param nodelabel: how to label nodes:
            - "all": label all nodes on y-axis
            - "type": indicate only cell types via legend
            - "auto": automatically choose between "all" and "type" depending on network size
        :param ax (optional): plotting axis
        :return: figure handle
        '''
        logger.info('plotting spikes rasters')

        if nodelabel == 'auto':
            nodelabel = 'all' if len(tspikes) <= 30 else 'type'
        
        # Create figure backbone
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        sns.despine(ax=ax)
        ax.set_title('spikes raster')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('node')
        if nodelabel == 'all':
            ax.set_yticks(np.arange(self.size()))
            ax.set_yticklabels(self.ids[::-1])
        else:
            ax.set_yticks([0, self.size() - 1])
            ax.set_yticklabels([self.size(), 1])
        ax.set_xlim(0, protocol.tstop)
        ax.set_ylim(-0.5, self.size() - 0.5)

        # If stimulus mark is specified
        if stim_mark != False:
            self.add_stim_mark(ax, protocol, mark_type=stim_mark)

        # Construct colormap with cell-type color code
        ctypes = self.parse_cell_types(unique=True)
        cmap = dict(zip(ctypes, plt.get_cmap('tab10').colors))

        # For each node in spikes dictionary
        handles = {}
        for i, (node_id, node_spikes) in enumerate(tspikes.items()):
            # If at least 1 spike detected
            if len(node_spikes) > 0:
                # Extract color and plot raster on appropriate line
                cell_type = self.parse_cell_type(node_id)
                yspan = (1 - ymargin) / 2
                yref = self.size() - 1 - i
                ax.vlines(node_spikes, yref - yspan, yref + yspan, colors=cmap[cell_type])
        
        # Add cell type legend if needed
        if nodelabel == 'type':
            handles = {k: Rectangle((0, 0), 1, 1, color=v) for k, v in cmap.items()}
            ax.legend(handles=handles.values(), labels=handles.keys(), 
                      loc='upper left', bbox_to_anchor=(1.0, 1.0), frameon=False)
            fig.tight_layout()

        # Return
        return fig
    
    def plot_firing_rates(self, tspikes, protocol, binsize=None, groupby=None, render='auto'):
        ''' 
        Compute and plot firing rate profiles.

        :param tspikes: dictionary of spike times (in s) per node
        :param protocol: pulsing protocol used during the simulation
        :param binsize: binning time interval for spike counting
        :param groupby: grouping variable for firing rate computation (e.g. "celltype"). If no
            grouping variable is provided, firing rates are index by node ID        
        :param render: how to render FR profiles (in case of individual node rendering):
            - "lines": plot each profile as a line
            - "map": plot profiles on a heatmap
            - "auto": automatically choose between "lines" and "map" depending on network size
        :return: figure handle
        '''
        # Compute firing rates from spikes timings
        FRs = self.compute_firing_rates(tspikes, protocol, binsize=binsize, groupby=groupby)
        # Create figure backbone
        fig, ax = plt.subplots()
        sns.despine(ax=ax)
        
        # If grouping variable provided
        if groupby is not None:
            if render == 'map':
                raise ValueError(f'heatmap rendering not implemented with "{groupby}" grouping')
            render = 'lines'
            # Define discrete color mapping for group values
            if groupby == 'celltype':
                ctypes = self.parse_cell_types(unique=True)
                cdict = dict(zip(ctypes, plt.get_cmap('tab10').colors))
            else:
                raise ValueError(f'no color mapping defined for "{groupby}" grouping variable')
            # For each group
            for key, group in FRs.groupby(groupby):
                gdata = group.droplevel('celltype')
                # Plot mean trace with appropriate color
                gdata.plot(y='mean', c=cdict[key], label=key, ax=ax)
                # If available, plot +/-sem shading
                if not gdata['sem'].isna().any():
                    ax.fill_between(
                        gdata.index, gdata['mean'] - gdata['sem'], gdata['mean'] + gdata['sem'],
                        fc=cdict[key], ec=None, alpha=0.3)

        # If no grouping variable provided
        else:
            gby = FRs.index.names[-1]
            # Define rendering strategy 
            if render == 'auto':
                gidxs = FRs.index.unique(gby).values
                render = 'lines' if len(gidxs) <= 10 else 'map'

            # Line rendering: render each nod FR profile as its own line
            if render == 'lines':
                sns.lineplot(
                    ax=ax, data=FRs.to_frame(), x='time (s)', y='spike rate (Hz)', hue=gby)
            # Heatmap rendering: render FR profiles on common heatmap
            else:
                fig.subplots_adjust(right=0.8)
                pos = ax.get_position()
                cbar_ax = fig.add_axes([pos.x1 + 0.05, pos.y0, .05, pos.y1 - pos.y0])
                cbar_ax.set_title('spike rate (Hz)')
                data = FRs.unstack().T
                data.columns = data.columns.map(lambda x: np.round(x, 4))
                sns.heatmap(
                    data=data,
                    ax=ax,
                    cbar_ax=cbar_ax,
                    cmap='viridis'
                )
        
        # Post-process figure
        if render == 'lines':
            self.add_stim_mark(ax, protocol)
            ax.set_xlim(0, protocol.tstop)
            ax.set_ylabel('spike rate (Hz)')

        # Return
        return fig


class NodeNetwork(NodePopulation):

    ''' Generic interface to neural network of nodes '''

    simkey = 'node_network'
    CON_MAPPINGS = [  # possible connectivity mappings 
            'presyn-celltype', 
            'postsyn-celltype',
            'syntype',
            'synmodel',
            'synweight',
            'weighted-syntype'
        ]

    def __init__(self, nodes, connections, presyn_var='Qm', connect=True):
        ''' Construct network.

            :param connections: list of (presyn_node_id, postsyn_node_id syn_weight, syn_model)
                for each connection to be instantiated
            :param presyn_var: reference variable for presynaptic threshold detection (Vm or Qm)
            :param connect (default=True): whether to wire specified connections upon instantiation
        '''
        # Construct node collection
        super().__init__(nodes)
        # Assign attributes
        self.connections = connections
        self.presyn_var = presyn_var
        # Connect nodes
        if connect:
            self.connect_all()
    
    def __repr__(self):
        ''' Explicit naming of the model instance. '''
        return f'{super().__repr__()[:-1]}, {len(self.connections)} connections)'    

    @property
    def meta(self):
        ''' Return dictionary of meta information about the model '''
        return {
            **super().meta,
            'simkey': self.simkey,
            'connections': self.connections,
            'presyn_var': self.presyn_var
        }

    @classmethod
    def initFromMeta(cls, meta):
        ''' Initialize network from meta dictionary '''
        return cls(cls.getNodesFromMeta(meta), meta['connections'], meta['presyn_var'])

    def checkConnection(self, presyn_id, postsyn_id, syn_model):
        ''' Check validity of tentative synaptic conection '''
        assert presyn_id in self.ids, f'invalid pre-synaptic node ID: "{presyn_id}"'
        assert postsyn_id in self.ids, f'invalid post-synaptic node ID: "{postsyn_id}"'
        assert isinstance(syn_model, Synapse), f'invalid synapse model: {syn_model}'
    
    @property
    def connections(self):
        ''' Getter for connections attribute '''
        return self._connections

    @connections.setter
    def connections(self, value):
        ''' Setter for connections attribute '''
        for presyn_id, postsyn_id, _, syn_model in value:
            self.checkConnection(presyn_id, postsyn_id, syn_model)
        self._connections = value

    def connect(self, presyn_id, postsyn_id, syn_model, syn_weight, delay=None):
        ''' Connect a source node to a target node with a specific synapse model
            and synaptic weight.

            :param presyn_id: ID of the pre-synaptic node
            :param postsyn_id: ID of the post-synaptic node
            :param syn_model: synapse model
            :param weight: synaptic weight (uS)
            :param delay (optional): synaptic delay (ms)
        '''
        self.checkConnection(presyn_id, postsyn_id, syn_model)
        # Create synapse instance from model, and attach it to target node
        syn = syn_model.attach(self.nodes[postsyn_id])
        # Determine relevant hoc variable for pre-synaptic trigger
        if self.presyn_var == 'Vm':
            hoc_var = f'Vm_{self.nodes[presyn_id].mechname}'
        else:
            hoc_var = 'v'
        # Generate network-connection between pre and post synaptic nodes
        nc = h.NetCon(
            getattr(self.nodes[presyn_id].section(0.5), f'_ref_{hoc_var}'),  # trigger variable 
            syn,  # synapse object (already attached to post-synaptic node)
            sec=self.nodes[presyn_id].section  # pre-synaptic node
        )

        # Normalize synaptic weight according to ratio of assigned vs. theoretical membrane area 
        syn_weight *= self.nodes[postsyn_id].getAreaNormalizationFactor()

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
    
    def disconnect_all(self):
        ''' Clear all synapses and network connection objects '''
        logger.info(f'removing all {len(self.netcon_objs)} connections between nodes')
        self.syn_objs = None
        self.netcon_objs = None
    
    def init_connectivity_matrix(self, fill=0):
        '''
        Initalize connectivity matrix filled with constant value
        
        :param fill: value used to fill the matrix
        :return: n-by-n matrix, with n being the network size
        '''
        keys = ['pre-syn', 'post-syn']
        candidate_pairs = list(product(self.ids, self.ids))
        return (
            pd.DataFrame(data=candidate_pairs, columns=keys)
            .set_index(keys)
            .assign(x=fill)['x']
            .unstack()
            .reindex(self.ids).reindex(self.ids, axis='columns')
        )

    def get_connectivity_matrix(self, criterion=None):
        ''' 
        Get network connectivity matrix
        
        :param criterion (optional): criterion according to which matrix values are defined:
            - 'presyn-celltype': map according to pre-synaptic cell type
            - 'postsyn-celltype': map according to post-synaptic cell type
            - 'syntype': map according to synapse tyep (E/I)
            - 'synmodel': map according to synapse model
            - 'synweight': map according to synaptic weight
            - 'weighted-syntype': map according to synapse type (E/I) and weight
            If no criterion is provided, a simple (connection / no connection) binary mapping is used
        :return: n-by-n connectivity matrix, with n = network size
        '''
        # Check mapping criterion validity 
        if criterion is not None and criterion not in self.CON_MAPPINGS:
            raise ValueError(
                f'invalid mapping criterion: "{criterion}". Candidates are {self.CON_MAPPINGS}')

        # Construct discrete mapping onto category, if required 
        if criterion is not None:
            refobjs = None
            if 'celltype' in criterion:
                refobjs = self.parse_cell_types(unique=True)
            elif criterion == 'synmodel':
                refobjs = list(set([con[-1] for con in self.connections]))
            if refobjs is not None:
                mymap = dict(zip(refobjs, np.arange(len(refobjs)) + 1))

        # Initalize connectivity matrix filled with zeros
        con_matrix = self.init_connectivity_matrix()

        # Mark established connections within matrix
        for presyn_id, postsyn_id, syn_weight, syn_model in self.connections:
            if criterion is None:  # no key provided -> binary output 
                val = 1
            elif criterion == 'presyn-celltype':  # map according to pre-synaptic cell type
                val = mymap[self.parse_cell_type(presyn_id)]
            elif criterion == 'postsyn-celltype':  # map according to post-synaptic cell type
                val = mymap[self.parse_cell_type(postsyn_id)]
            elif criterion == 'syntype':  # map according to synapse tyep (E/I)
                val = syn_model.syntype
            elif criterion == 'synmodel':  # map according to synapse model
                val = mymap[syn_model]
            elif criterion == 'synweight':  # map according to synaptic weight
                val = syn_weight
            elif criterion == 'weighted-syntype':  # map according to synapse type (E/I) and weight
                val = syn_weight * syn_model.syntype
            con_matrix.loc[presyn_id, postsyn_id] = val

        # Return 
        return con_matrix

    def plot_connectivity_matrix(self, hue=None, background_color='lightgray', ax=None):
        '''
        Plot connectivity matrix
        
        :param hue (optional): color-coding criterion
        :param background_color (optional): matrix background color
        :param ax (optional): plotting axis
        :return figure handle
        '''
        # Extract connectivity matrix with appropriate mapping criterion
        matrix = self.get_connectivity_matrix(criterion=hue)
        # Extract matrix data type
        dtype = matrix.values.dtype
        # Replace zeros by NaNs
        if hue is not None:
            matrix = matrix.replace(0, np.nan)
        # Extract unique non-NaN matrix values and recast them to original data types
        uniques = np.sort(np.unique(matrix.values))
        uniques = uniques[~np.isnan(uniques)].astype(dtype)
        # Initialize default heatmap arguments
        cbar = True
        ticklabels = None
        vmin, vmax = None, None
        cmap = None
        center = None
        # No hue provided -> simple black - white heatmap without colorbar
        if hue is None:
            cmap = [background_color, 'k']
            cbar = False
        else:
            # Integer input (i.e. discrete mapping)
            if dtype == int: 
                # Set colormap type and colorbar tick labels as a function of mapping criterion
                if 'celltype' in hue:
                    cmap = 'tab10'
                    ticklabels = self.parse_cell_types(unique=True)
                elif hue == 'syntype':
                    cmap = 'Set1'
                    ticklabels = ['inh', 'exc']
                elif hue == 'synmodel':
                    cmap = 'tab10'
                    ticklabels = uniques
                
                # Expand colormap range, and construct discrete colormap
                vmin = min(uniques) - 0.5
                vmax = max(uniques) + 0.5
                colors = list(plt.get_cmap(cmap).colors[:len(uniques)])
                cmap = ListedColormap(colors)
            
            # Float input (i.e. continous mapping)
            else:
                # If mapping contains negative values, set appropriate colormap
                # and update center value
                if uniques.min() < 0:
                    cmap = plt.get_cmap('coolwarm_r')
                    center = 0
                # Otherwise, use default colormap
                else:
                    cmap = plt.get_cmap('viridis')

            # Map NaNs to background color
            cmap.set_bad(background_color)
        
        # Create figure backbone, and add colorbar axis if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig = ax.get_figure()
        ax.set_aspect(1.)
        ax.set_title(f'{self.size()}-by-{self.size()} connectivity matrix')
        if cbar:
            fig.subplots_adjust(right=0.8)
            pos = ax.get_position()
            cbar_ax = fig.add_axes([pos.x1 + 0.05, pos.y0, .05, pos.y1 - pos.y0])
            cbar_ax.set_title(hue.replace('-', ' '))
        else:
            cbar_ax = None
        
        # Plot heatmap
        sns.heatmap(
            ax=ax, 
            data=matrix,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            center=center,
            cbar=cbar,
            cbar_ax=cbar_ax,
        )

        # Update colorbar ticks and tick labels, if needed
        if ticklabels is not None:
            cbar_ax.set_yticks(uniques)
            cbar_ax.set_yticklabels(ticklabels)
        
        # Return figure handle
        return fig


class SmartNodeNetwork(NodeNetwork):
    ''' Network model with automated generation of nodes and connections '''

    simkey = 'smart_node_network'

    def __init__(self, ntot, proportions, conn_rates, syn_models, syn_weights, drives=None, connect=True, **node_kwargs):
        '''
        Initialization
        
        :param ntot: total number of cells in the network
        :param proportions: dictionary of proportions of each cell type in the network
        :param conn_rates: 2-level dictionary of connection rates for each connection type
        :param syn_models: 2-level dictionary of synapse models for each connection type
        :param syn_weights: 2-level dictionary of synaptic weights (in uS) for each connection type
        :param drives: dictionary of input drives (in nA) for each cell type
        :param node_kwargs (optional): additional node initialization parameters
        '''
        # Compute number of cells of each type
        self.ncells = {k: int(np.round(v * ntot)) for k, v in proportions.items()}
        
        # Get reference point neuron for each cell type
        pneurons = {k: getPointNeuron(k) for k in self.ncells.keys()}

        # Instantiate nodes with appriopriate drives
        if drives is None:
            drives = {}
        nfmt = int(np.ceil(np.log10(max(self.ncells.values()))))
        fmt = f'{{:0{nfmt}}}'
        self.nodesdict = {}
        for k, n in self.ncells.items():
            self.nodesdict[k] = {}
            Idrive = drives.get(k, 0.)  # nA
            idrive = Idrive * 1e-6 / pneurons[k].area  # mA/m2
            for i in range(n):
                self.nodesdict[k][f'{k}{fmt.format(i)}'] = Node(pneurons[k], **node_kwargs)
                if idrive != 0:
                    self.nodesdict[k][f'{k}{fmt.format(i)}'].setConstantDrive(idrive)
        logger.info(f'instantiated {sum(self.ncells.values())} nodes ({self.str_node_count()})')
        
        # Generate connections lists structured by connection types
        self.conndict = {}
        for presyn, targets in conn_rates.items():
            self.conndict[presyn] = {}
            for postsyn, rate in targets.items():
                pairs = self.generate_connection_pairs(presyn, postsyn, rate)
                if len(pairs) > 0:
                    ignore = False
                    try:
                        w = syn_weights[presyn][postsyn]
                    except KeyError:
                        logger.warning(f'no synaptic weight defined for {presyn}-{postsyn} connection -> ignoring')
                        ignore = True
                    try:
                        model = syn_models[presyn][postsyn]
                    except KeyError:
                        logger.warning(f'no synaptic model defined for {presyn}-{postsyn} connection -> ignoring')
                        ignore = True
                    if not ignore:
                        self.conndict[presyn][postsyn] = [(*pair, w, model) for pair in pairs]
        concounts = self.con_count_matrix()
        logger.info(f'generated {concounts.values.sum()} random connection pairs:\n{add_indent(repr(concounts), nspaces=3)}')

        # Initialize parent network class
        super().__init__(self.get_all_nodes(), self.get_all_connections(), connect=connect)
    
    @classmethod
    def initFromMeta(cls, meta):
        ''' Initialize network from meta dictionary '''
        return NodeNetwork(cls.getNodesFromMeta(meta), meta['connections'], meta['presyn_var'])
    
    def __repr__(self):
        ''' Explicit naming of the model instance. '''
        return f'{self.__class__.__name__}({self.str_node_count()}, {len(self.connections)} connections)'
    
    def str_node_count(self):
        ''' String representation for nodes count per cell type '''
        return ', '.join([f'{n} {k} cell{"s" if n > 1 else ""}' for k, n in self.ncells.items()])
    
    def con_count_matrix(self):
        ''' Get matrix of number of connections between each pair of cell types '''
        return (pd.DataFrame(
            {k: {kk: len(vv) for kk, vv in v.items()} for k, v in self.conndict.items()})
            .fillna(0)
            .astype(int)
        )

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
    

class CorticalNodeNetwork(SmartNodeNetwork):
    '''
    Cortical network model derived from (Vierling-Classen et al., 2010)
    
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

    # Synaptic weights (in uS) for each connection type from (Vierling-Classen et. al, 2010), 
    # modified to compensate for differences in post-synaptic neuron membrane area between
    # their multi-compartment models and the NICE point-neuron models from (Plaksin et al., 2016).
    syn_weights = {
        'RS': {
            'RS': 0.001368638,
            'FS': 0.002539801,
            'LTS': 0.004301499
        },
        'FS': {
            'RS': 0.0066967,
            'FS': 0.045201132,
            'LTS': 0.432259291
        },
        'LTS': {
            'RS': 0.088647563,
            'FS': 0.005237858
        }
    }

    # Thalamic drive to cortical cells (from in Plaksin 2016), meant to
    # produce 7 Hz firing rate in RS neuron(s) to mimic spontaneous activity 
    # during anesthetized experiments.
    # TODO:
    # - adapt to match spontanteous baseline during awake experiments
    # - replace by NetStim?
    # - add noise to drive?
    I_Th_RS = 0.17  # nA
    thalamic_drives = {
        'RS': I_Th_RS,
        'FS': 1.4 * I_Th_RS,
        'LTS': 0.0
    }

    def __init__(self, ntot, connect=True, **node_kwargs):
        super().__init__(
            ntot, 
            self.proportions, self.conn_rates, self.syn_models, self.syn_weights,
            drives=self.thalamic_drives, connect=connect,
            **node_kwargs
        )