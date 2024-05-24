# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-08-18 21:14:43
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-03-29 21:10:27

import re
import matplotlib.pyplot as plt
from PySONIC.parsers import *

from .plt import SectionGroupedTimeSeries, SectionCompTimeSeries
from .models import models_dict
from .constants import *


class SpatiallyExtendedParser(Parser):

    secid_pattern = '([a-zA-Z]+)(?:([0-9]+))?'  # Regexp section ID pattern

    def __init__(self):
        super().__init__()
        self.addSection()
        self.addWiring()

    def addResistivity(self):
        self.add_argument(
            '--rs', nargs='+', type=float, help='Intracellular resistivity (Ohm.cm)')

    def addSection(self):
        self.add_argument(
            '--section', nargs='+', type=str, help='Section of interest for plot')

    def addSectionID(self):
        self.add_argument(
            '--secid', nargs='+', type=str, help='ID of morphological section targeted by stimulus')
    
    def parseSecID(self, args):
        if not isIterable(args['secid']):
            args['secid'] = [args['secid']]
        return args
    
    def addWiring(self):
        self.add_argument(
            '--wiring', type=str, default='sonic', choices=('sonic', 'default'), 
            help='Internal wiring scheme')

    def parse(self, args=None):
        if args is None:
            args = super().parse()
        return args

    @staticmethod
    def parseSimInputs(args):
        return [args[k] for k in ['rs']]
    
    @classmethod
    def parse_section_type(cls, sec_id):
        ''' Parse section type from its id '''
        if isIterable(sec_id):
            return [cls.parse_section_type(x) for x in sec_id]
        mo = re.match(cls.secid_pattern, sec_id)
        if mo is None:
            raise ValueError(f'"{sec_id}" does not match section ID pattern: {cls.secid_pattern}')
        return mo.group(1)

    @classmethod    
    def parse_section_dict(cls, seclist):
        ''' Parse raw sections list into model-relevant sections dictionary '''
        sectypes = cls.parse_section_type(seclist)
        secdict = {k: [] for k in dict.fromkeys(sectypes)}
        for secid, stype in zip(seclist, sectypes):
            secdict[stype].append(secid)
        return secdict
    
    @classmethod
    def parsePlot(cls, args, outputs):
        render_args = {}
        if 'spikes' in args:
            render_args['spikes'] = args['spikes']
        if args['section'] is None:
            raise ValueError('names of sections to plot must be explicitly specified')
        if args['compare']:
            if args['plot'] == ['all']:
                logger.error('Specific variables must be specified for comparative plots')
                return
            for key in ['cmap', 'cscale']:
                render_args[key] = args[key]
            if render_args['cmap'] is None:
                del render_args['cmap']
            # For each simulation output 
            for output in outputs:
                # Parse section dict from output
                secdict = cls.parse_section_dict(list(output[0].keys()))
                sections = args['section']
                # Filter section dict based on section selection
                if sections == ['all']:
                    filtered_secdict = secdict
                else:
                    filtered_secdict = {}
                    for sec in sections:
                        if sec in secdict:
                            filtered_secdict[sec] = secdict[sec]
                        else:
                            stype = cls.parse_section_type(sec)
                            if sec not in secdict[stype]:
                                raise ValueError(f'section "{sec}" not found in output')
                            if stype not in filtered_secdict:
                                filtered_secdict[stype] = [sec]
                            else:
                                filtered_secdict[stype].append(sec)
                filtered_secdict = {k: sorted(v) for k, v in filtered_secdict.items()}
                for k, v in filtered_secdict.items():
                    if len(v) < 2:
                        raise ValueError(f'at least 2 {k} sections must be provided for a comparative plot')

                # Generate 1 comparative plot per plot variable
                for pltvar in args['plot']:
                    for sectype, seclist in filtered_secdict.items():
                        # Adjust colormap based on completeness of section list
                        if len(seclist) < len(secdict[sectype]):
                            render_args['cmap'] = 'viridis'
                        else:
                            render_args.pop('cmap', None) 
                        comp_plot = SectionCompTimeSeries([output], pltvar, seclist)
                        comp_plot.render(**render_args)
        else:
            if args['section'] == ['all']:
                raise ValueError('"--section all" not available in standard plotting mode. Try adding "--compare" to enable comparative mode')
            for key in args['section']:
                scheme_plot = SectionGroupedTimeSeries(key, outputs, pltscheme=args['pltscheme'])
                if ABERRA:
                    render_args['fiber'] = args['fiber']
                    render_args['ref_loc'] = args['ref_loc']
                scheme_plot.render(**render_args)
        plt.show()


class FiberParser(SpatiallyExtendedParser):

    def __init__(self):
        super().__init__()
        self.defaults.update({'type': 'senn', 'fiberD': 20., 'nnodes': 21})
        self.factors.update({'fiberD': 1 / M_TO_UM})
        self.addType()
        self.addFiberDiameter()
        self.addNnodes()

    def addResistivity(self):
        pass

    def addType(self):
        self.add_argument(
            '--type', nargs='+', type=str, help='Fiber model type')

    def addFiberDiameter(self):
        self.add_argument(
            '-d', '--fiberD', nargs='+', type=float, help='Fiber diameter (um)')

    def addNnodes(self):
        self.add_argument(
            '--nnodes', nargs='+', type=int, help='Number of nodes of Ranvier')

    def parsePlot(self, args, output):
        if args['section'] is None:
            if args['compare']:
                logger.warning('no section specified for plot -> showing profiles from all recorded sections')
                args['section'] = ['all']
            else:
                logger.warning('no section specified for plot -> plotting from central section only')
                args['section'] = ['center']
        if args['section'] == ['center'] and args['compare']:
            raise ValueError(
                '"center" placeholder for section ID cannot be used when comparing results from models of different sizes')
        if len(args['section']) > 5 and args['plot'] != ['all'] and not args['compare']:
            logger.warning(
                'More than 5 morphological sections were specified for plots -> falling back to comparative plot(s)')
            args['compare'] = True
        return SpatiallyExtendedParser.parsePlot(args, output)

    @staticmethod
    def parseSimInputs(args):
        return SpatiallyExtendedParser.parseSimInputs(args)

    def parse(self, args=None):
        args = super().parse(args=args)#;print(models_dict)
        args['type'] = [models_dict[model_key] for model_key in args['type']]
        if args['wiring'] == 'default':
            args['type'] = [fclass.__original__ for fclass in args['type']]
        for key in ['fiberD']: #this needs to be args['fiberD'] right? #joa
            if len(args[key]) > 1 or args[key][0] is not None:
                args[key] = self.parse2array(args, key, factor=self.factors[key])
        return args


class EStimFiberParser(FiberParser, PWSimParser):

    def __init__(self):
        PWSimParser.__init__(self)
        FiberParser.__init__(self)
        self.defaults.update({'tstim': 0.1, 'toffset': 3.})
        self.allowed.update({'mode': ['cathode', 'anode']})
        self.addElectrodeMode()
        self.addAstim()

    def addElectrodeMode(self):
        self.add_argument(
            '--mode', type=str, help='Electrode polarity mode ("cathode" or "anode")')

    def addAstim(self):
        self.add_argument(
            '-A', '--amp', nargs='+', type=float,
            help=f'Point-source current amplitude ({self.amp_unit})')
        self.add_argument(
            '--Arange', type=str, nargs='+',
            help=f'Point-source current amplitude range {self.dist_str} ({self.amp_unit})')
        self.to_parse['amp'] = self.parseAmplitude

    def parseAmplitude(self, args):
        return EStimParser.parseAmplitude(self, args)

    def parse(self):
        args = FiberParser.parse(self, args=PWSimParser.parse(self))
        if isIterable(args['mode']):
            args['mode'] = args['mode'][0]
        return args

    @staticmethod
    def parseSimInputs(args):
        return PWSimParser.parseSimInputs(args) + SpatiallyExtendedParser.parseSimInputs(args)

    def parsePlot(self, *args):
        return FiberParser.parsePlot(self, *args)


class IextraFiberParser(EStimFiberParser):

    amp_unit = 'mA'

    def __init__(self):
        super().__init__()
        self.defaults.update({'xps': 0., 'zps': None, 'mode': 'cathode', 'amp': -0.7})
        self.factors.update({'amp': MA_TO_A, 'xps': 1 / M_TO_MM, 'zps': 1 / M_TO_MM})
        self.addPointSourcePosition()

    def addPointSourcePosition(self):
        self.add_argument(
            '--xps', nargs='+', type=float, help='Point source x-position (mm)')
        self.add_argument(
            '--zps', nargs='+', type=float, help='Point source z-position (mm)')

    def parse(self):
        args = super().parse()
        for key in ['xps', 'zps']:
            if len(args[key]) > 1 or args[key][0] is not None:
                args[key] = self.parse2array(args, key, factor=self.factors[key])
        return args


class IintraFiberParser(EStimFiberParser):

    amp_unit = 'nA'

    def __init__(self):
        super().__init__()
        self.defaults.update({'secid': None, 'mode': 'anode', 'amp': 2.0})
        self.factors.update({'amp': 1 / A_TO_NA})
        self.addSectionID()
    
    def parse(self):
        return self.parseSecID(super().parse())


class AStimFiberParser(FiberParser, AStimParser):

    def __init__(self):
        AStimParser.__init__(self)
        FiberParser.__init__(self)
        for x in [self.defaults, self.allowed, self.to_parse]:
            x.pop('method')
        self.defaults.update({'tstim': 0.1, 'toffset': 3.})

    @staticmethod
    def parseSimInputs(args):
        return AStimParser.parseSimInputs(args) + SpatiallyExtendedParser.parseSimInputs(args)

    def parsePlot(self, *args):
        return FiberParser.parsePlot(self, *args)


class SectionAStimFiberParser(AStimFiberParser):

    amp_unit = 'kPa'

    def __init__(self):
        super().__init__()
        self.defaults.update({'secid': None})
        self.addSectionID()

    def parseAmplitude(self, args):
        return AStimParser.parseAmplitude(self, args)

    def parse(self):
        return self.parseSecID(super().parse())


class SpatiallyExtendedTimeSeriesParser(TimeSeriesParser):

    def __init__(self):
        super().__init__()
        self.addSection()

    def addSection(self):
        SpatiallyExtendedParser.addSection(self)


class TestNetworkParser(TestParser):

    def __init__(self, valid_subsets):
        super().__init__(valid_subsets)
        self.addConnect()

    def addConnect(self):
        self.add_argument(
            '--connect', default=False, action='store_true', help='Connect nodes')
        

#added by Joa -------------------------------------------------------------------------------------------------------------------------------|
class NeuronParser(SpatiallyExtendedParser):

    def __init__(self):
        super().__init__()
        self.defaults.update({'type': 'realistic_cort', 'x0': 0., 'sigma': 1.}) #sigma is 1mm by default
        self.factors.update({'sigma': 1 / M_TO_MM})
        self.addType()
        self.addx0()
        self.addSigma()

    def addResistivity(self):
        pass

    def addType(self):
        self.add_argument(
            '--type', nargs='+', type=str, help='Neuron model type')

    def addx0(self):
        self.add_argument(
            '--x0', '--gausscenter', nargs='+', type=float, help='Gaussian center coordinate (m)')

    def addSigma(self):
        self.add_argument(
            '--sigma', nargs='+', type=float, help='Gaussian RMS width (m)')

    def parsePlot(self, args, output):
        if args['section'] is None:
            if args['compare']:
                logger.warning('no section specified for plot -> showing profiles from all recorded sections')
                args['section'] = ['all']
            else:
                logger.warning('no section specified for plot -> plotting from central section only')
                args['section'] = ['center']
        if args['section'] == ['center'] and args['compare']:
            raise ValueError(
                '"center" placeholder for section ID cannot be used when comparing results from models of different sizes')
        if len(args['section']) > 5 and args['plot'] != ['all'] and not args['compare']:
            logger.warning(
                'More than 5 morphological sections were specified for plots -> falling back to comparative plot(s)')
            args['compare'] = True
        return SpatiallyExtendedParser.parsePlot(args, output)

    @staticmethod
    def parseSimInputs(args):
        return SpatiallyExtendedParser.parseSimInputs(args)

    def parse(self, args=None):
        args = super().parse(args=args)#;print(models_dict)
        args['type'] = [models_dict[model_key] for model_key in args['type']]
        if args['wiring'] == 'default':
            args['type'] = [fclass.__original__ for fclass in args['type']]
        # for key in ['fiberD']:
        #     if len(args[key]) > 1 or args[key][0] is not None:
        #         args[key] = self.parse2array(args, key, factor=self.factors[key])
        return args
        

class AStimNeuronParser(NeuronParser, AStimParser):

    def __init__(self):
        AStimParser.__init__(self)
        NeuronParser.__init__(self)
        for x in [self.defaults, self.allowed, self.to_parse]:
            x.pop('method')
        self.defaults.update({'tstim': 0.1, 'toffset': 3.})

    @staticmethod
    def parseSimInputs(args):
        return AStimParser.parseSimInputs(args) + SpatiallyExtendedParser.parseSimInputs(args)

    def parsePlot(self, *args):
        return NeuronParser.parsePlot(self, *args)


class AStimRealisticNeuronParser(AStimNeuronParser):

    amp_unit = 'kPa'

    def __init__(self):
        super().__init__()
        #self.defaults.update({'secid': None})
        #self.addSectionID()

    def parseAmplitude(self, args):
        return AStimParser.parseAmplitude(self, args)

    # def parse(self):
    #     return self.parseSecID(super().parse())