# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-02-19 14:42:20
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-03-30 14:07:09

import abc
from neuron import h, numpy_element_ref
import numpy as np
import pandas as pd
from scipy import stats
import time
from scipy.optimize import least_squares, minimize, show_options
import ctypes
from scipy.interpolate import interp1d, interpn
import pprint
import sys
import matplotlib.pyplot as plt
import pickle

from PySONIC.core import Model, PointNeuron, BilayerSonophore, EffectiveVariablesDict
from PySONIC.core.timeseries import TimeSeries, SpatiallyExtendedTimeSeries
from PySONIC.postpro import detectSpikes
from PySONIC.utils import logger, si_format, filecode, simAndSave, isIterable
from PySONIC.constants import *
from PySONIC.threshold import threshold, titrate, Thresholder

from .pyhoc import *
from ..sources import *
from ..utils import array_print_options, load_mechanisms, getNmodlDir
from ..constants import *

cvode = h.CVode()


class NeuronModel(metaclass=abc.ABCMeta):
    ''' Generic interface for NEURON models. '''

    tscale = 'ms'  # relevant temporal scale of the model
    refvar = 'Qm'  # default reference variable
    is_constructed = False
    fixed_dt = FIXED_DT
    passive_mechname = CLASSIC_PASSIVE_MECHNAME

    # integration methods
    int_methods = {
        0: 'backward Euler method',
        1: 'Crank-Nicholson method',
        2: 'Crank-Nicholson method with fixed currents at mid-steps',
        3: 'CVODE multi order variable time step method',
        4: 'DASPK (Differential Algebraic Solver with Preconditioned Krylov) method'
    }

    def __init__(self, construct=True):
        ''' Initialization. '''
        logger.debug(f'Creating {self} model')
        load_mechanisms(getNmodlDir(), self.modfile)
        if construct:
            self.construct()

    def set(self, attrkey, value):
        ''' Set attribute if not existing or different, and reset model if already constructed. '''
        realkey = f'_{attrkey}'
        if not hasattr(self, realkey) or value != getattr(self, realkey):
            setattr(self, realkey, value)
            if self.is_constructed:
                logger.debug(f'resetting model with {attrkey} = {value}')
                self.reset()

    def isEditableProperty(self, k):
        ''' Check if a key corresponds to an editable property of the model. '''
        return k.startswith('_') and hasattr(self, k) and hasattr(self, k[1:])

    def mirror(self, other):
        ''' Modify self properties to match those of another model instance. '''
        logger.debug(f'mirroring {self} to {other}')
        for k, v in other.__dict__.items():  # loop through model properties
            if self.isEditableProperty(k):  # if editable property in self
                if v != getattr(self, k):  # if value differ -> modify in self
                    logger.debug(f'setting {self}.{k} to {v}')
                    self.set(k[1:], v)

    def mirrored(self, other_cls, **kwargs):
        ''' Return an instance from another model class modified to mirror self. '''
        other = other_cls(*self.initargs[0], **self.initargs[1], **kwargs)
        other.mirror(self)
        return other

    @property
    @abc.abstractmethod
    def simkey(self):
        ''' Keyword used to characterize stimulation modality. '''
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def meta(self):
        raise NotImplementedError

    @property
    def modelcodes(self):
        return {
            'simkey': self.simkey,
            'neuron': self.pneuron.name
        }

    @property
    def modelcode(self):
        return '_'.join(self.modelcodes.values())

    @property
    def pneuron(self):
        return self._pneuron

    @pneuron.setter
    def pneuron(self, value):
        if not isinstance(value, PointNeuron):
            raise TypeError(f'{value} is not a valid PointNeuron instance')
        self.set('pneuron', value)

    @property
    def modfile(self):
        return f'{self.pneuron.name}.mod'

    @property
    def mechname(self):
        return f'{self.pneuron.name}auto'

    @staticmethod
    def axialSectionArea(d_out, d_in=0.):
        ''' Compute the cross-section area of a axial cylinder section expanding between an
            inner diameter (presumably zero) and an outer diameter.

            :param d_out: outer diameter (m)
            :param d_in: inner diameter (m)
            :return: cross-sectional area (m2)
        '''
        return np.pi * ((d_out)**2 - d_in**2) / 4.

    @classmethod
    def axialResistancePerUnitLength(cls, rho, *args, **kwargs):
        ''' Compute the axial resistance per unit length of a cylindrical section.

            :param rho: axial resistivity (Ohm.cm)
            :return: resistance per unit length (Ohm/cm)
        '''
        return rho / cls.axialSectionArea(*args, **kwargs) / M_TO_CM**2  # Ohm/cm

    @classmethod
    def axialResistance(cls, rho, L, *args, **kwargs):
        ''' Compute the axial resistance of a cylindrical section.

            :param rho: axial resistivity (Ohm.cm)
            :param L: cylinder length (m)
            :return: resistance (Ohm)
        '''
        return cls.axialResistancePerUnitLength(rho, *args, **kwargs) * L * M_TO_CM  # Ohm

    def setCelsius(self, celsius=None):
        if celsius is None:
            try:
                celsius = self.pneuron.celsius
            except AttributeError:
                raise ValueError('celsius value not provided and not found in PointNeuron class')
        h.celsius = celsius

    @property
    @abc.abstractmethod
    def seclist(self):
        raise NotImplementedError

    @property
    def nsections(self):
        return len(self.seclist)

    def construct(self):
        ''' Create, specify and connect morphological model sections. '''
        self.createSections()
        self.setGeometry()
        self.setResistivity()
        self.setBiophysics()
        self.setTopology()
        self.setExtracellular()
        self.is_constructed = True

    @abc.abstractmethod
    def createSections(self):
        ''' Create morphological sections. '''
        raise NotImplementedError

    def setGeometry(self):
        ''' Set sections geometry. '''
        pass

    def setResistivity(self):
        ''' Set sections axial resistivity. '''
        pass

    def setBiophysics(self):
        ''' Set the membrane biophysics of all model sections. '''
        if self.refvar == 'Qm':
            pass
            #self.setFuncTables() #RS with overtones

    def setTopology(self):
        ''' Connect morphological sections. '''
        pass

    def setExtracellular(self):
        ''' Set the sections' extracellular mechanisms. '''
        pass

    def clearLookups(self):
        ''' Clear model-related lookups. '''
        self.Aref = None
        self.Qref = None
        self.lkp = None
        self.pylkp = None

    @abc.abstractmethod
    def clearSections(self):
        ''' Clear model-related neuron sections. '''
        raise NotImplementedError

    @abc.abstractmethod
    def clearDrives(self):
        ''' Clear model-related drives. '''
        raise NotImplementedError

    def clear(self):
        ''' Clear all model related NEURON objects. '''
        self.clearSections()
        self.clearLookups()
        self.clearDrives()
        constituent_secs = list(filter(lambda s: s.cell() == self, getAllSecs()))
        if len(constituent_secs) > 0:
            s = '\n'.join([f'  - {s}' for s in constituent_secs])
            raise ValueError(f'clearing error: remaining {self} sections:\n{s}')

    def reset(self):
        ''' Delete and re-construct all model sections. '''
        self.clear()
        self.construct()

    def getSectionClass(self, mechname):
        ''' Get the correct section class according to mechanism name. '''
        if mechname is None:
            d = {'Vm': VSection, 'Qm': QSection}
        else:
            d = {'Vm': MechVSection, 'Qm': MechQSection}
        return d[self.refvar]

    def createSection(self, id, *args, mech=None, states=None, Cm0=None, nrnsec=None): #nrnsec is added to enable creation of Aberra cell Sections
        ''' Create a model section with a given id. '''
        args = [x for x in args if x is not None]
        if Cm0 is None:
            Cm0 = self.pneuron.Cm0 * F_M2_TO_UF_CM2  # uF/cm2
        kwargs = {'name': id, 'cell': self, 'Cm0': Cm0}
        if mech is not None:
            kwargs.update({'mechname': mech, 'states': states})
        if nrnsec:
            kwargs.update({'nrnsec': nrnsec})
        secclass = self.getSectionClass(mech)
        #print('secclass: ',secclass) #to check which section class is chosen 
        return secclass(*args, **kwargs) #BREAKPOINT

    def setTimeProbe(self):
        ''' Set time probe. '''
        return Probe(h._ref_t, factor=1 / S_TO_MS)

    def setIntegrator(self, dt, atol):
        ''' Set CVODE integration parameters. '''
        #print(f'dt: {dt}, atol: {atol},cvode_active: {cvode.active()}')
        if dt is not None:
            h.secondorder = 0  # using backward Euler method if fixed time step
            h.dt = dt * S_TO_MS
            if cvode.active():
                cvode.active(0)
        else:
            if not cvode.active():
                pass
                cvode.active(1) #apparently it is better to use h.cvode_active(1) instead of h.cvode.active(1) 
                #cvode.use_daspk(1) #added daspk option -> this happens automatic because of LinMech and extracellular
            if atol is not None:
                cvode.atol(atol)
        #print('SET INTEGRATOR:') #LOG OUTPUT
        #print(f'before: h.dt = {h.dt*1e3} us'); h.dt *= 0.1; print(f'after: h.dt = {h.dt*1e3} us')
        #print(f'current methods: {cvode.current_method()}')
        #print(self.getIntegrationMethod())
        #print(f'h.secondorder = {h.secondorder}')
        #print('')

    def resetIntegrator(self):
        ''' Re-initialize the integrator. '''
        # If adaptive solver: re-initialize the integrator
        if cvode.active():
            cvode.re_init()
        # Otherwise, re-align currents with states and potential
        else:
            h.fcurrent()

    def getIntegrationMethod(self):
        ''' Get the method used by NEURON for the numerical integration of the system. '''
        method_type_code = cvode.current_method() % 1000 // 100
        method_type_str = self.int_methods[method_type_code]
        if cvode.active():
            return f'{method_type_str} (atol = {cvode.atol()})'
        else:
            return f'{method_type_str} (fixed dt = {h.dt} ms)'

    def fi3(self):
        logger.debug('finitialize: initialization started')

    def fi0(self):
        logger.debug('finitialize: internal structures checked')
        logger.debug('finitialize: t set to 0')
        logger.debug('finitialize: event queue cleared')
        logger.debug('finitialize: play values assigned to variables')
        logger.debug('finitialize: initial v set in all sections')

    def fi1(self):
        logger.debug('finitialize: mechanisms BEFORE INITIAL blocks called')
        logger.debug('finitialize: mechanisms INITIAL blocks called')
        logger.debug('finitialize: LinearMechanism states initialized')
        logger.debug('finitialize: INITIAL blocks inside NETRECEIVE blocks called')
        logger.debug('finitialize: mechanisms AFTER INITIAL blocks are called.')

    def fi2(self):
        logger.debug('finitialize: net_send events delivered')
        logger.debug('finitialize: integrator initialized')
        logger.debug('finitialize: record functions called at t = 0')
        logger.debug('finitialize: initialization completed')

    def initToSteadyState(self):
        ''' Initialize model variables to pre-stimulus resting state values. '''
        self.setStimValue(0)
        if self.refvar == 'Qm':
            x0 = self.pneuron.Qm0 * C_M2_TO_NC_CM2  # nC/cm2
            unit = 'nC/cm2'
        else:
            x0 = self.pneuron.Vm0  # mV
            unit = 'mV'
        logger.debug(f'initializing system at {x0} {unit}')
        if PRINT_FINITIALIZE_STEPS:
            self.fih = [
                h.FInitializeHandler(3, self.fi3),
                h.FInitializeHandler(0, self.fi0),
                h.FInitializeHandler(1, self.fi1),
                h.FInitializeHandler(2, self.fi2)
            ]
        #print('starting h.finitialize')
        for sec in h.allsec():
            sec.v = self.pneuron.Vm0*sec.cm 
        h.finitialize() #(x0) #BREAKPOINT
        h.fcurrent()
        #self.resetIntegrator()
        #print('ended h.finitialize')

    def fadvanceLogger(self):
        logger.debug(f'fadvance return at t = {h.t:.3f} ms')

    def setStimValue(self, value):
        ''' Set stimulation ON or OFF.

            :param value: new stimulation state (0 = OFF, 1 = ON)
            :return: new stimulation state
        '''
        logger.debug(f't = {h.t:.3f}ms, setting x = {value}')
        # Set "stimon" attribute in all model sections
        for sec in self.seclist:
            sec.setStimON(value)
        # Set multiplying factor of all model drives
        for drive in self.drives:
            drive.set(value)

    def setTimeStep(self, dt):
        h.dt = dt * S_TO_MS

    def update(self, value, new_dt):
        self.setStimValue(value)
        self.resetIntegrator()
        if new_dt is not None:
            self.setTimeStep(new_dt)

    def createStimSetter(self, value, new_dt):
        return lambda: self.update(value, new_dt)

    # def setDriveModulator(self, events, tstop):
    #     ''' Drive temporal modulation vector. '''
    #     times, values = zip(*events)
    #     times, values = np.array(times), np.array(values)
    #     if times[0] > 0:
    #         times = np.hstack(([0.], times))
    #         values = np.hstack(([0.], values))
    #     self.tmod = h.Vector(np.append(np.sort(np.hstack((
    #         times - TRANSITION_DT / 2, times + TRANSITION_DT / 2))), tstop) * S_TO_MS)
    #     self.xmod = h.Vector(np.hstack((0., values.repeat(2))))
    #     h('stimflag = 0')  # reference stim flag HOC variable
    #     self.xmod.play(h._ref_stimflag, self.tmod, True)

    def setTransitionEvent(self, t, value, new_dt):
        # cvode.event((t - TRANSITION_DT) * S_TO_MS, self.fadvanceLogger)
        cvode.event((t - TRANSITION_DT) * S_TO_MS)
        cvode.event(t * S_TO_MS, self.createStimSetter(value, new_dt))

    def setTransitionEvents(self, events, tstop, dt):
        ''' Set integration events for transitions. '''
        times, values = zip(*events)
        times, values = np.array(times), np.array(values)
        if dt is not None:
            Dts = np.diff(np.append(times, tstop))
            dts = np.array([min(dt, Dt / MIN_NSAMPLES_PER_INTERVAL) for Dt in Dts])
        else:
            dts = [None] * len(times)
        for t, value, new_dt in zip(times, values, dts):
            if t == 0:  # add infinitesimal offset in case of event at time zero
                t = 2 * TRANSITION_DT
            self.setTransitionEvent(t, value, new_dt)

    def integrateUntil(self, tstop):
        logger.debug(f'integrating system using {self.getIntegrationMethod()}')
        h.t = 0
        if OVERTONES:
            self.init_overtones()
        t_next, t_step = 1, 1 #1, 1 #0.0010, 0.0010
        T_up_next, T_up_step = 1, 1 #0.05, 0.05 #next update moment and update step of overtones
        print(f'tstop = {tstop} ms') #LOG OUTPUT
        print(F"T_up = {T_up_step} ms")
        t_end = [0]
        i_end = [0]
        while h.t < tstop: #BREAKPOINT
            for e in self.seclist:
                continue
                #print(str(e).split('.')[-1],e.v)
            #print('\n')
            if DEBUG_OV:
                if h.t > 1:
                    #print(h.t,flush=1)
                    # v= 0
                    for sec in self.seclist:
                        nrnsec = sec.nrnsec
                        if nrnsec.v > 40:
                            pass
                            #print(f"{nrnsec}: v = {nrnsec.v}")
                        #print(nrnsec.psection()['density_mechs'].keys())
                            #print(sec.relevant_mechs)
                        for e in sec.relevant_mechs:
                            try:
                                em = eval(f"nrnsec.m_{e}")
                                if em <= 0 or em >= 1:
                                #if nrnsec.v > 40:
                                    print(f"{nrnsec}: m_{e} = {em}")
                                    sys.exit(1)
                                
                                ha = eval(f"nrnsec.h_{e}")
                                if ha <= 0 or ha >= 1:
                                #if nrnsec.v > 40:
                                    print(f"{nrnsec}: h_{e} = {ha}")
                                    sys.exit(1)
                            except:
                                try:
                                    em = eval(f"nrnsec.m_{e}2")
                                    if em <= 0 or em >= 1:
                                    #if nrnsec.v > 40:
                                        print(f"{nrnsec}: m_{e} = {em}")
                                        sys.exit(1)

                                    ha = eval(f"nrnsec.h_{e}2")
                                    if ha <= 0 or ha >= 1:
                                    #if nrnsec.v > 40:
                                        print(f"{nrnsec}: h_{e} = {ha}")
                                        sys.exit(1)
                                except:
                                    pass
                pass
            if DEBUG_OV:
                #print(f'\n\n new timestep: {h.t}\n\n')
                #print(self.all_probes['soma0'].keys()); quit()
                for sec in self.seclist:
                    nrnsec = sec.nrnsec
                    #print(f"stimon = {nrnsec.stimon_pas_eff}")
                    if h.t > 3:
                        pass
                        #plt.plot(np.array(self.t), np.array(self.all_probes['soma0']['Vm']),label='Vm')
                    #print(f"v = {nrnsec.v}")
                    break
            if h.t > t_next:
                print(f'h.t = {h.t}') #LOG OUTPUT
                if DEBUG_OV:
                    # for sec in self.seclist:
                    #     nrnsec = sec.nrnsec
                    #     print(nrnsec.psection()["density_mechs"].keys())
                    #     print(f"v = {nrnsec.v}, Vm: {nrnsec.Vm_pas_eff}, A_t: {nrnsec.A_t_pas_eff}")
                    #     if OVERTONES:
                    #         print(f"a1: {nrnsec.a1_pas_eff}, b1: {nrnsec.b1_pas_eff}")
                    #         print(f"V_val = {nrnsec.V_val_pas_eff}")
                    #         print(f"A1_val = {nrnsec.A_1_val_pas_eff}")
                    #     print(f"stimon = {nrnsec.stimon_pas_eff}")
                    #     import matplotlib.pyplot as plt
                    #     from sklearn.preprocessing import MinMaxScaler
                    #     scaler = MinMaxScaler()
                    #     print(np.array(self.all_probes['soma0'].keys()))
                    #     r = self.t
                    #     t = np.array(r)
                    #     for i,e in zip(i_end, t_end):
                    #         t[i:] += e
                        #print(i_end, t_end)
                        #vm = np.array(self.all_probes['soma0']['Vm'])
                        #qm = np.array(self.all_probes['soma0']['Qm'])
                        #vm_scaled = scaler.fit_transform(vm.reshape(-1, 1)).flatten()
                        #if len(i_end) < 2:
                        #    plt.plot(t[i_end[-1]:], vm[i_end[-1]:], label='Vm')
                        #else:
                        #    plt.plot(t[i_end[-2]:], vm[i_end[-2]:], label='Vm')
                        #    plt.plot(t[i_end[-2]:], qm[i_end[-2]:], label='Qm')
                        #print(h.t)
                        #print(len(t))
                        # for e in t:
                        #     print(e)
                        #print('\n')
                        #t_end += [t[-1]]
                        #i_end += [len(t)]
                        #plt.show()

                        # ipas = np.array(self.all_probes['soma0'][('i', 'pas_eff')])
                        # ipas_scaled = scaler.fit_transform(ipas.reshape(-1, 1)).flatten()
                        # #plt.plot(np.array(self.t), ipas,label='ipas')
                        # icaH = np.array(self.all_probes['soma0'][('ica', 'Ca_HVA')])
                        # icaH_scaled = scaler.fit_transform(icaH.reshape(-1, 1)).flatten()
                        # #plt.plot(np.array(self.t), icaH,label='icaH')
                        # icaL = np.array(self.all_probes['soma0'][('ica', 'Ca_LVAst')])
                        # icaL_scaled = scaler.fit_transform(icaL.reshape(-1, 1)).flatten()
                        # #plt.plot(np.array(self.t), icaL,label='icaL')
                        # ih = np.array(self.all_probes['soma0'][('ihcn', 'Ih')])
                        # ih_scaled = scaler.fit_transform(ih.reshape(-1, 1)).flatten()                        
                        # #plt.plot(np.array(self.t), ih,label='ih')
                        # ik2 = np.array(self.all_probes['soma0'][('ik', 'SK_E2')])
                        # ik2_scaled = scaler.fit_transform(ik2.reshape(-1, 1)).flatten()
                        # #plt.plot(np.array(self.t), ik2,label='ik2')
                        # ik3 = np.array(self.all_probes['soma0'][('ik', 'SKv3_1')])
                        # ik3_scaled = scaler.fit_transform(ik3.reshape(-1, 1)).flatten()
                        # #plt.plot(np.array(self.t),ik3 ,label='ik3')
                        # #plt.plot(np.array(self.t), np.array(self.all_probes['soma0'][('ik', 'SKv3_1')]),label='ik3')
                        # stimon = (1 - self.all_probes['soma0'][('stimon', 'pas_eff')]) * -85 + self.all_probes['soma0'][('stimon', 'pas_eff')] * np.array(self.all_probes['soma0']['Vm'])
                        #plt.plot(np.array(self.t), np.array(stimon))
                        #plt.legend()
                        #plt.show()
                        #plt.plot(np.array(self.t), np.array(self.all_probes['soma0'][('a1', 'pas_eff')]),label='a1')
                        #plt.plot(np.array(self.t), np.array(self.all_probes['soma0'][('b1', 'pas_eff')]),label='b1')
                        #plt.plot(np.array(self.t), np.array(self.all_probes['soma0'][('A_1_val', 'pas_eff')]),label='A1')
                        #plt.plot(np.array(self.t), np.array(self.all_probes['soma0'][('B_1_val', 'pas_eff')]),label='B1')
                        #plt.legend()
                        #plt.show()
                        #break
                    pass
                t_next += t_step
            if OVERTONES:
                if h.t > T_up_next:
                    #print('overtones update')
                    T_up_next += T_up_step
                    self.advance(1) #1
                else:
                    self.advance(1) #0
            else:
                self.advance()
    
    def init_overtones(self):
        """ init """

        start = 0
        start_reversed = 0
        
        connections = self.connections
        connections_reversed = self.connections_reversed
        indexes = self.indexes

        self.nseg = len(self.segments) #number of segments
        self.nov = OVERTONES #number of overtones
        "dimensions order: 2(A/B), overtones, nseg"

        self.AB = np.zeros(2*self.nov*self.nseg) #input overtones a and b for all segments -> initialize at 0 #C/m2
        #self.A, self.B = np.zeros((OVERTONES,len(self.segments))), np.zeros((OVERTONES,len(self.segments)))

        self.Identity = np.identity(2*self.nov*self.nseg)
        Rsingle = np.full((self.nseg,self.nseg),np.inf) 
        R = np.full((self.nseg,self.nseg),np.inf) 
        C_flat = np.zeros((self.nseg,self.nov,2)) #in opposite order because flatten works in this order
        #G = np.zeros((2*self.nov*self.nseg,self.nseg))

        iterator = 0
        for i,seci in enumerate(self.seclist): #iterate over all sections i -> sec_i
            sec_i = seci.nrnsec #neuron section
            nseg_i = sec_i.nseg #number of segments that the section i contains
            midpoints_i = [(1+2*i)/(2*nseg_i) for i in range(nseg_i)] #midpoints of the different segments of section i
            for g, seg in enumerate(sec_i): #iterate over all the segments in section i
                #print(iterator,nseg_i)
                for k in range(self.nov):
                    C_flat[iterator,k,0] = 1/((k+1)*seg.area()*1e-12*(2*np.pi*self.fref)) #seg.area: um2 = 1e-12 m2 #s/m2 = 1/(Hz*m2)
                    C_flat[iterator,k,1] = -1/((k+1)*seg.area()*1e-12*(2*np.pi*self.fref)) #eq (13) has a minus sign #s/m2 = 1/(Hz*m2)

                "CASE 3: segment and subsequent segment of subsequent section"
                if g == nseg_i - 1: #if it is a segment at the end of the section (last segment before terminal 1) -> case 3
                    for c,i_c in enumerate(connections[start:]): #iterate over all connections where they are stored as (parent, child)
                        if i_c[0] > i: #if we are past section i in the connections
                            start += c #offset c is used to avoid iterating over the whole list for every section
                            break #break if we are past section i
                        if i == i_c[0]: #if the connection is where the segment of section i is the parent, and c is the child #secc = (i,c)
                            secc = self.seclist[i_c[1]]
                            sec_c = secc.nrnsec # neuron section
                            nseg_c = sec_c.nseg #number of segments that section c contains
                            R_ic = sec_i(1).ri() + sec_c(1/(2*nseg_c)).ri() #resistance between segment of section i and segment of section c
                            R[indexes[i_c[0]][-1],indexes[i_c[1]][0]] = Rsingle[indexes[i_c[0]][-1],indexes[i_c[1]][0]] = R_ic
                            R[indexes[i_c[1]][0],indexes[i_c[0]][-1]] = R_ic
                            #print(f'case3: ({indexes[i_c[0]][-1]}, {indexes[i_c[1]][0]})')

                "CASE 2: segment and preceding segment of preceding section"
                if g == 0: #if it is a segment at the beginning of the section (first section after terminal 0) -> case 2
                    for c,i_c in enumerate(connections_reversed[start_reversed:]): #iterate over all connections where they are stored as (child, parent)
                        if i_c[0] > i: #if we are past section i in the connections
                            start_reversed += c #avoiding whole list iteration
                            break #stop i we are past section i
                        if i == i_c[0]: #where segment i is the child, and c is the parent #sec = (i,c)
                            secc = self.seclist[i_c[1]]
                            sec_c = secc.nrnsec #neuron section
                            nseg_c = sec_c.nseg #number of segments in section c             
                            R_ic = sec_c(1).ri() + sec_i(1/(2*nseg_i)).ri() #resistance between segment of section c and segment of section i
                            R[indexes[i_c[1]][-1],indexes[i_c[0]][0]] = Rsingle[indexes[i_c[1]][-1],indexes[i_c[0]][0]] = R_ic
                            R[indexes[i_c[0]][0],indexes[i_c[1]][-1]] = R_ic
                            #print(f'case2: ({indexes[i_c[1]][-1]}, {indexes[i_c[0]][0]})')

                "CASE 1"
                if nseg_i != 1:  #if it is a segment next to another segments of the same section -> case 1: segment is between 2 other segments (which are not the terminal segments)
                    "segment and preceding segment of the same section"
                    if g != 0: #not the first segment so we look at connection between the segment and the one before it
                        R_ic = sec_i(midpoints_i[g]).ri() #resistance between segment and the previous one (the one before it)
                        R[iterator-1,iterator] = Rsingle[iterator-1,iterator] = R_ic
                        R[iterator,iterator-1] = R_ic
                        #print(f'case1.2: ({iterator-1}, {iterator})')

                    "segment and subsequent segment of the same section"
                    if g != nseg_i - 1: #not the last segment so we look at connection between segment and the one after it
                        R_ic = sec_i(midpoints_i[g+1]).ri() #resistance between segment and the next one (the one after it)
                        R[iterator,iterator+1] = Rsingle[iterator,iterator+1] = R_ic
                        R[iterator+1,iterator] = R_ic
                        #print(f'case1.3: ({iterator}, {iterator+1})')

                iterator += 1
        self.C_flat = C_flat #actually not flat, has dimensions: nseg, ov, AB #s/m2
        self.C = np.diag(self.C_flat.flatten()) #s/m2
        R /= OHM_TO_MOHM #seg.ri: MOhm = 1e6 Ohm #Ohm
        G = 1/R #S
        Grep = np.repeat(G, 2*self.nov , axis=0) #S
        self.Gext = np.repeat(Grep, 2*self.nov , axis=1) #S
        Gsum = np.sum(Grep,axis=1) #S
        self.Gsumdiag = np.diag(Gsum) #S
        self.G = G #S
        #print(R)
        #print(sum(np.sum((G>0)*1,axis=1)))
        return
    

    def AB_to_ampphase(self, A,B):
        amp = np.sqrt(A**2 + B**2)
        phase = -np.arctan2(B,A) % (2*np.pi)
        return amp, phase
    
    def ampphase_to_AB(self, amp,phase):
        A = amp * np.cos(phase)
        B = - amp * np.sin(phase)
        return A,B


    def calc_jac(self, ABovseg):
        """ calculation of jacobian
            :ABovseg: contains all the input overtone data stuctured in a way for Jacobian simplification (first AB, then ov, then segments)  """

        mult = 2*self.nov
        "dimensions order: 2(A/B), overtones, nseg"
        LUTseci_1 = np.identity(self.nseg) #simplified structure with a 1 representing a filled block and a 0 representing an empty block
        LUTsecc_1 = (self.G>0)*1 #simplified structure with a 1 representing a filled block and a 0 representing an empty block
        self.LUTseci = np.repeat(np.repeat(LUTseci_1,mult,axis=0),mult,axis=1) #create the block matrix
        self.LUTsecc = np.repeat(np.repeat(LUTsecc_1,mult,axis=0),mult,axis=1) #create a block matrix
        self.LUTseci_eq = np.copy(self.LUTseci) #for eq
        self.LUTsecc_eq = np.copy(self.LUTsecc) #for eq

        #LUTs = []
        Jac = self.pylkp.Jac
        lkp = self.pylkp #for eq
        refs = list(Jac.refs.values())
        refs_ext = refs.copy()
        refs_ext[1] = self.pylkp.Q_ext #refs_ext are the same refs but with the extended Q ref

        A = ABovseg[::2] #C/m2
        B = ABovseg[1::2] #C/m2
        self.Qs = []
        # for nov in range(OVERTONES):
        #     Ak = A[nov::OVERTONES] #list containing the A's for a specific overtone for all segments
        #     Bk = B[nov::OVERTONES] #start with the desired overtone an take steps in the size of the number of overtones to take the same overtone every time

        #     iterator = 0
        #     for sec in self.seclist:
        #         for seg in sec.nrnsec:
        #             for mech in sec.relevant_mechs:
        #                 setattr(seg,f'a1_{mech}',Ak[iterator])
        #                 setattr(seg,f'b1_{mech}',Bk[iterator])
        #             iterator += 1

        iterator = 0
        for secindex,sec in enumerate(self.seclist):
            for segindex, seg in enumerate(sec.nrnsec): #iterate over the sections
                Cm0 = Cm0_map[seg.cm] #determine the cm0-string
                dV_dQ_i = np.array([[f'd{ABV}V{vov+1}{Cm0}_d{ABQ}Q{qov+1}{Cm0}' for ABQ in ['A', 'B'] for vov in range(self.nov)] for ABV in ['B','A'] for qov in range(self.nov)]) #block matrix containing the string versions or visual versions of the derivatives
                block = np.zeros((mult, mult)) #actual derivative block
                for ir, row in enumerate(dV_dQ_i): #iterate over the output overtones (numerator of the derivative)
                    for ic, column in enumerate(row): #iterate over the input overtones (denominator of the derivative)
                        xi = [getattr(seg,f'A_t_{sec.random_mechname}')/PA_TO_KPA, seg.v/C_M2_TO_NC_CM2] #determine the input values for the LUT from the segment, start with A and Q
                        for i in range(self.nov): 
                            Ak = A[i::self.nov] #list containing the A's for a specific overtone for all segments
                            Bk = B[i::self.nov] #start with the desired overtone an take steps in the size of the number of overtones to take the same overtone every time
                            a = Ak[iterator//mult] #a = getattr(seg,f'a{i+1}_{sec.random_mechname}')
                            b = Bk[iterator//mult] #b = getattr(seg,f'b{i+1}_{sec.random_mechname}')
                            xi += [a, b] #add overtones to input values of the LUT
                            # if ir == ic == 0: #only add once per block as this will be the same every time
                            #     self.Qs += [a, b] #for eq
                        if seg.cm == 2:
                            #column contains the string representation of the derivative
                            block[ir,ic] = interpn(refs_ext,Jac[column],xi)[0] #,fill_value=None,bounds_error=0)[0] #interpolate the Jacobian for the point where the segment is currently at #[0] because return is array with single value
                            #block_eq[ir,ic] = interpn(refs_ext,lkp[V_i[ir,ic]],xi)[0] #for eq
                        elif seg.cm == 0.02: #in the case of myelin, the derivative is 1/cm if X=Y in dX/dY, otherwise it is 0
                            split4 = column.split('_') #the string should be splitted in 4 strings
                            XY = (split4[0].replace('V','Q') == split4[2])
                            block[ir,ic] = 1/seg.cm if XY else 0
                        else:
                            block[ir,ic] = interpn(refs,Jac[column],xi)[0] #,fill_value=None,bounds_error=0) -> to extrapolate #[0] is sliced because it returns an array with a single value -> take that single value
                            #block_eq[ir,ic] = interpn(refs,lkp[V_i[ir,ic]],xi)[0] #for eq
                self.LUTseci[iterator:iterator+mult] = np.tile(block,self.nseg) * self.LUTseci[iterator:iterator+mult] #multiply the block by the repeated identity matrix rows
                self.LUTsecc[iterator:iterator+mult] = np.tile(block,self.nseg) * self.LUTsecc[iterator:iterator+mult] #multiply the block by the repeated connection matrix rows

                iterator += mult
        self.LUTsecc = np.transpose(self.LUTsecc) #transpose because we actually needed to do column multiplication but easier to implement row multiplication in numpy
        #print(f'I:{self.Identity.shape} + C: {self.C.shape} * Gext: {self.Gext.shape} *~ LUTsecc:{self.LUTsecc.shape} - C:{self.C.shape} *Gsumdiag: {self.Gsumdiag.shape} * LUTseci:{self.LUTseci.shape}')
        Jac = self.Identity + np.multiply(np.matmul(self.C, self.Gext), self.LUTsecc) + np.matmul(np.matmul(self.C, self.Gsumdiag), self.LUTseci)
        eqs = self.solve_overtone(A,B)[0].flatten()
        return np.linalg.norm(np.transpose(2*eqs*np.transpose(Jac)),axis=0)**2 #the columns of the Jacobian matrix need to be multiplied with the 'equations' vector
    

    def solve_overtone(self, A, B):

        print(f'A,B: {A}, {B}')

        "use SECONIC LUT"
        with open(r"C:\Users\jgazquez\PySONIC\PySONIC\lookups\test_joa\sonictable_cfit.pkl", 'rb') as fh:
            pkl = pickle.load(fh)
        sonictable = pkl['tables']
        sonicrefs = [f for f in  pkl['refs'].values()]
        sonictable = sonictable[:,0,:,0,:,:]
        A1 = sonictable[:,:,:,1] #Qm, psi1, delta1
        B1 = sonictable[:,:,:,2]
        A1 = np.swapaxes(A1,1,2) #Qm, delta1, psi1
        B1 = np.swapaxes(B1,1,2)
        print('sonictable')
        # plt.plot(B1[5, 8, :])
        # plt.show()


        #print(self.segments[0],self.segments[1])
        #print(f'self.segments[0].A_t_RSauto = Adrive*stimon: {self.segments[0].A_t_RSauto} = {self.segments[0].Adrive_RSauto} * {self.segments[0].stimon_RSauto}')
        #print(f'self.segments[1].A_t_RSauto = Adrive*stimon: {self.segments[1].A_t_RSauto} = {self.segments[1].Adrive_RSauto} * {self.segments[1].stimon_RSauto}')
        self.eqs = np.zeros((self.nseg,self.nov,2)) #C/m2
        residual = 0 #the residual of all overtone functions (for all overtones and sections) #C2/m4 (unit is not of interest here)

        connections = self.connectdictseg
        refs = list(self.pylkp.Jac.refs.values())
        refs_ext = refs.copy()
        refs_ext[1] = self.pylkp.Q_ext

        for k in range(self.nov):
            Ak = A[k::self.nov] #list containing the A's for a specific overtone for all segments #C/m2
            Bk = B[k::self.nov] #start with the desired overtone and take steps in the size of the number of overtones to take the same overtone every time #C/m2

            #print(len(Q_k), len(phi_k))
            for iseg, seg in enumerate(self.segments):
                #print(seg)
                #print(seg, seg.cm)
                xi = [getattr(seg,f'A_t_{self.seclist[self.seginsec[iseg]].random_mechname}')/PA_TO_KPA,seg.v/C_M2_TO_NC_CM2] #Pa, C/m2
                #xi[0] = 1e5 #doesnt matter in SECONIC table
                for n in range(self.nov):
                    A_n = A[n::self.nov] #C/m2
                    B_n = B[n::self.nov] #C/m2
                    a = A_n[iseg] #C/m2
                    b = B_n[iseg] #C/m2
                    print('xi:')
                    xi += [*self.AB_to_ampphase(a,b)] #Pa, C/m2, rad #xi += [a, b] #Pa, C/m2
                LHS_12 = Ak[iseg] #C/m2
                LHS_13 = Bk[iseg] #C/m2
                RHS_12 = 0
                RHS_13 = 0
                if seg.cm == 0.02:
                    BVki = Bk[iseg]/(seg.cm*1e-2) * V_TO_MV #mV
                    AVki = Ak[iseg]/(seg.cm*1e-2) * V_TO_MV #mV
                elif seg.cm == 1:
                    print(xi)
                    #print([Qm,deltas,phases])
                    BVki = interpn(refs, self.pylkp[f'B_{k+1}'], xi) #mV
                    AVki = interpn(refs, self.pylkp[f'A_{k+1}'], xi) #mV
                    print(f'python LUT: AVki = {AVki}, BVki = {BVki}')
                    BVki = interpn(sonicrefs, B1, xi[1:]) #mV
                    AVki = interpn(sonicrefs, A1, xi[1:]) #mV
                    print(f'matlab LUT: AVki = {AVki}, BVki = {BVki}')
                    print(f'{xi} -> {BVki,AVki}')
                    print(f'BVki,AVki: {BVki,AVki}')
                elif seg.cm == 2:
                    BVki = interpn(refs_ext, self.pylkp[f'B_{k+1}2'], xi) #mV
                    AVki = interpn(refs_ext, self.pylkp[f'A_{k+1}2'], xi) #mV
                else:
                    raise TypeError('Wrong seg.cm')
                for connectie in connections[iseg]:
                    #print(connectie, self.segments[connectie], self.segments[connectie].cm)
                    xc = [getattr(self.segments[connectie],f'A_t_{self.seclist[self.seginsec[connectie]].random_mechname}')/PA_TO_KPA,self.segments[connectie].v/C_M2_TO_NC_CM2] #Pa, C/m2
                    #xc[0] = 100000 #test: periphery section in RS is also LIFUS sensitive
                    for n in range(self.nov):
                        A_n = A[n::self.nov] #C/m2
                        B_n = B[n::self.nov] #C/m2
                        a = A_n[connectie] #C/m2
                        b = B_n[connectie] #C/m2
                        print('xc:')
                        xc += [*self.AB_to_ampphase(a,b)]
                    if self.segments[connectie].cm == 0.02:
                        BVkc = Bk[connectie]/(seg.cm*1e-2) * V_TO_MV #mV
                        AVkc = Ak[connectie]/(seg.cm*1e-2) * V_TO_MV #mV
                    elif self.segments[connectie].cm == 1:
                        print(xc)
                        print(refs)
                        BVkc = interpn(refs, self.pylkp[f'B_{k+1}'], xc) #mV
                        AVkc = interpn(refs, self.pylkp[f'A_{k+1}'], xc) #mV
                        print(f'python LUT: AVkc = {AVkc}, BVkc = {BVkc}')
                        BVkc = interpn(sonicrefs, B1, xc[1:]) #mV
                        AVkc = interpn(sonicrefs, A1, xc[1:]) #mV
                        print(f'matlab LUT: AVkc = {AVkc}, BVkc = {BVkc}')
                    elif self.segments[connectie].cm == 2:
                        BVkc = interpn(refs_ext, self.pylkp[f'B_{k+1}2'], xc) #mV
                        AVkc = interpn(refs_ext, self.pylkp[f'A_{k+1}2'], xc) #mV
                    else:
                        raise TypeError('Wrong seg.cm')
                    print(f'xc, xi: {xc, xi}')
                    print('before')
                    print(f'BVkc,BVki: {BVkc,BVki}')
                    print(f'AVkc,AVki: {AVkc,AVki}')
                    #SECONIC enforcing
                    BVki = np.array([-50.482187]) #np.array([40.085600])
                    BVkc = np.array([-40.085600]) #np.array([50.482187])
                    AVki = np.array([2.590251]) #np.array([-1.405800])
                    AVkc = np.array([1.405800]) #np.array([-2.590251])
                    print('after')
                    print(f'BVkc,BVki: {BVkc,BVki}')
                    print(f'AVkc,AVki: {AVkc,AVki}')
                    RHS_12 += self.G[iseg, connectie] * (BVkc-BVki) * 1e-3 #mA = 1e-3 A #A
                    RHS_13 += self.G[iseg, connectie] * (AVkc-AVki) * 1e-3 #mA = 1e-3 A #A
                    #print(f'RHS_12 = {self.G[iseg, connectie]*self.C_flat[iseg, k, 0]} * {(BVkc-BVki) * 1e-3} = {self.G[iseg, connectie]*self.C_flat[iseg, k, 0]*(BVkc-BVki) * 1e-3}')
                    #print(f'RHS_13 = {self.G[iseg, connectie]*self.C_flat[iseg, k, 1]} * {(AVkc-AVki) * 1e-3} = {self.G[iseg, connectie]*self.C_flat[iseg, k, 1]*(AVkc-AVki) * 1e-3}')
                    print(f'(BVkc-BVki) = {BVkc} -{BVki} = {(BVkc-BVki)}')
                    print(f'(AVkc-AVki) = {AVkc} -{AVki} = {(AVkc-AVki)}')
                #print(f'xi: {xi}, xc: {xc}')
                #print(f'LHS, RHS = {LHS_12}, {(self.C_flat[iseg, k, 0] * RHS_12)[0]}')
                #print(f'LHS, RHS = {LHS_13}, {(self.C_flat[iseg, k, 1] * RHS_13)[0]}')
                #print(f'G*C = {self.G[iseg, connectie] * self.C_flat[iseg, k, 0]}')
                eq12 = LHS_12 + self.C_flat[iseg, k, 0] * RHS_12 #C/m2
                eq13 = LHS_13 + self.C_flat[iseg, k, 1] * RHS_13 #C/m2
                #print(self.C_flat[iseg, k, 0] * RHS_12)
                #print(self.C_flat[iseg, k, 1] * RHS_13)
                
                self.eqs[iseg, k, 0] = eq12 #C/m2
                self.eqs[iseg, k, 1] = eq13 #C/m2
                #print(f'residuals: {eq12}, {eq13}')
                #print(f'res12**2: {eq12**2}, res13**2: {eq13**2}')
                print(eq12**2+eq13**2)
                quit()
                residual += eq12**2 + eq13**2 #C2/m4
        #print(self.eqs); quit()

        #print(f'residual: {residual}')
        return self.eqs, residual


    def solve_overtones(self,ABovseg):
        """ equation that returns LHS-RHS of eq. (12) and eq. (13) for ALL overtones
            :ABovseg: all the input overtone data stuctured in a way for Jacobian simplification (first AB, then ov, then segments)"""
        A = ABovseg[::2] #A_Q's #C/m2
        B = ABovseg[1::2] #B_Q's #C/m2

        _, residual = self.solve_overtone(A,B)
            
        return residual

    def update_overtones(self, solution):
        """ updating q_k and f_k: this is done before h.fadvance() as update() is also done before anything else in the NMODL file
            :solution: the solved overtones"""

        #AB_matrix = np.zeros((2,self.nov,self.nseg))
        ABovseg = solution #order of variables: AB, overtones, segments #C/m2
        self.AB = ABovseg.copy() #C/m2
        A = ABovseg[::2] #C/m2
        B = ABovseg[1::2] #C/m2
        for nov in range(OVERTONES):
            Ak = A[nov::OVERTONES] #list containing the A's for a specific overtone for all segments #C/m2
            Bk = B[nov::OVERTONES] #start with the desired overtone an take steps in the size of the number of overtones to take the same overtone every time #C/m2

            iterator = 0
            for sec in self.seclist:
                for seg in sec.nrnsec:
                    for mech in sec.relevant_mechs:
                        setattr(seg,f'a1_{mech}', Ak[iterator]) #setattr(seg,f'a1_{mech}', 0) #setattr(seg,f'a1_{mech}', Ak[iterator]) #C/m2
                        setattr(seg,f'b1_{mech}', Bk[iterator]) #setattr(seg,f'b1_{mech}', 0) #setattr(seg,f'b1_{mech}', Bk[iterator]) #C/m2
                    iterator += 1


    def advance(self, update_ov=0):
        ''' Advance simulation onto the next time step. '''

        if update_ov:

            print('START UPDATE',flush=1)
            #jac = self.calc_jac()
            if DEBUG_OV:
                #self.update_overtones(self.AB)
                #h.fadvance()
                #return
                pass
            A_bounds = (-0.000606762745, 0.000309016993) #C/m2
            B_bounds = (-0.00029388, 0.00095105) #C/m2
            bounds_AB = [x for _ in self.AB[::2] for x in (A_bounds, B_bounds)] #input overtone values for all segments (initialized at 0) #C/m2

            "debug calc_jac"
            #Jac = self.calc_jac(self.AB)
            # print(np.sum(Jac>0))
            # print(np.sum(Jac<0))
            # print(np.sum(Jac==0))
            #quit()

            "debug solve_overtones"
            # res = self.solve_overtones(self.AB)
            # print(res)
            # A = [-0.000000606762745]*(len(self.AB)//2)
            # B = [0.00000095105]*(len(self.AB)//2)
            # self.AB[::2] = A
            # self.AB[1::2] = B
            # res = self.solve_overtones(self.AB)
            # print(res)
            # quit()
            
            "plotting the residu in function of the bounds range"

            #print(f'self.calc_jac(self.AB): {self.calc_jac(self.AB)}')
            # for e in np.linspace(B_bounds[0],B_bounds[-1],50):
            #     self.AB[3] = e #e
            #     continue
            #     print(f'self.solve_overtones(self.AB): {e} -> {self.solve_overtones(self.AB)}')
            # quit()

            print(f'h.t: {h.t}')
            for e in self.segments:
                if 'center' in str(e):
                    centerarea = e.area()
                if 'periphery' in str(e):
                    peripherarea = e.area()

            self.AB = np.array([-1.4058*1e-5, 40.0856*1e-5, 1.4058*1e-5*centerarea/peripherarea, -40.0856*1e-5*centerarea/peripherarea])
            print(f'self.AB: {self.AB*1e5}')
            print(f'residual of matlab solution: {self.solve_overtones(self.AB)}')
            quit()
            
            result = minimize(self.solve_overtones, self.AB, bounds = bounds_AB, method = 'Nelder-Mead')
            #result = minimize(self.solve_overtones, self.AB, bounds = bounds_AB, method = 'Nelder-Mead', options = {'xatol': 1e-10, 'fatol': 1e-10})
            #result = minimize(self.solve_overtones, self.AB, bounds = bounds_AB,method = 'BFGS', options={'gtol' : 1e-10})#minimize(self.solve_overtones, self.AB, jac=self.calc_jac, bounds = bounds_AB)
            solution = result.x
            if not result.success:
                print(f"solution: {result}")
                raise ValueError
            #solution = Q_phi #for debugging
            #print(len(solution))
            print(f'solution*1e5: {solution*1e5}')
            if h.t > 0.1:
                quit()
            A = solution[::2]
            B = solution[1::2]
            Q = np.sqrt(A**2+B**2)
            f = -np.arctan2(B,A)
            print(f'A = {A*1e5} nC/cm2\nB= {B*1e5} nC/cm2')
            print(f'Q = {Q*1e5} nC/cm2\nf= {f/np.pi*180} °')
            a1c, a1p = A #it should be: a1c*areac = -a1p*areap
            b1c, b1p = B #it should be: b1c*areac = -b1p*areap
            print(f'sanity check: a1_c*A_c = a1_p*A_p <=> {a1c}*{centerarea} = {a1p}*{peripherarea} <=> {a1c*centerarea} = {a1p*peripherarea}')
            print(f'sanity check: b1_c*A_c = b1_p*A_p <=> {b1c}*{centerarea} = {b1p}*{peripherarea} <=> {b1c*centerarea} = {b1p*peripherarea}')

            #print(f'center: {a1c}, {b1c}')
            #print(f'center (periphery): {-a1p*peripherarea/centerarea}, {-b1p*peripherarea/centerarea}')
            solution[0] = -a1p*peripherarea/centerarea #enforcing
            solution[1] = -b1p*peripherarea/centerarea #enforcing
            #print(f'difference: {a1c+a1p*peripherarea/centerarea}, {b1c+b1p*peripherarea/centerarea}')
            #print(solution)

            print(f'residual after solving: {self.solve_overtones(self.AB)}')


            if DEBUG_OV:
                print(f'solution range A: [{min(solution[::2])}, {max(solution[::2])}]')
                print(f'solution range B: [{min(solution[1::2])}, {max(solution[1::2])}]')
                if 0:
                    A = solution[::2]
                    B = solution[1::2]
                    for n in range(OVERTONES):
                        print(f'\n\nOVERTONE {n}')
                        Ak = A[n::OVERTONES] 
                        Bk = B[n::OVERTONES] 
                        for i,seg in enumerate(self.segments):
                            print(f'{seg}: A = {Ak[i]}, B = {Bk[i]}')

            self.update_overtones(solution)
            print('END UPDATE',flush=1)

        h.fadvance()

    def integrate(self, pp, dt, atol):
        ''' Integrate a model differential variables for a given duration, while updating the
            value of the boolean parameter stimon during ON and OFF periods throughout the numerical
            integration, according to stimulus parameters.

            Integration uses an adaptive time step method by default.

            :param pp: pulsed protocol object
            :param dt: integration time step (s). If provided, the fixed time step method is used.
            :param atol: absolute error tolerance. If provided, the adaptive
                time step method is used.
        '''
        self.setIntegrator(dt, atol)
        self.initToSteadyState()
        # self.setDriveModulator(pp.stimEvents(), pp.tstop)
        self.setTransitionEvents(pp.stimEvents(), pp.tstop, dt)
        # cvode.print_event_queue()
        self.integrateUntil(pp.tstop * S_TO_MS)
        return 0

    def psection(self, sec, lkp=0):
        "stolen from C:\nrn\lib\python\neuron\psection.py"

        #refs = list(self.pylkp.Jac.refs.values())
        if lkp: #if a lookup table is given an no (temporary) dummy pointers are used
            if sec.cm == 2:
                Q_arr2 = {'Q_arr2': self.Qextref._ref_x[0]}
                Q_s2 = {'Q_s2': self.Qextref.size()}
            else:
                Q_arr2 = {}
                Q_s2 = {}
            arrays = {'A_arr': self.Aref._ref_x[0], 'Q_arr': self.Qref._ref_x[0], **Q_arr2} #create dictionary with all arrays
            sizes = {'A_s': self.Aref.size(), 'Q_s': self.Qref.size(), **Q_s2} #create dictionary with all array sizes
            for i in range(len(self.overtones)//4): #for every overtone we need 2 arrays and 2 sizes in the NMODL files
                sizes[f'amp{i+1}_s'] = self.overtones[4*i] #sizes[f'A{i+1}_s'] = self.overtones[4*i]
                arrays[f'amp{i+1}_arr'] = self.overtones[4*i+1]._ref_x[0] #arrays[f'A{i+1}_arr'] = self.overtones[4*i+1]._ref_x[0]
                sizes[f'phi{i+1}_s'] = self.overtones[4*i+2] #sizes[f'B{i+1}_s'] = self.overtones[4*i+2]
                arrays[f'phi{i+1}_arr'] = self.overtones[4*i+3]._ref_x[0] #arrays[f'B{i+1}_arr'] = self.overtones[4*i+3]._ref_x[0]
            #print(sizes); print(arrays); quit()

        mname = h.ref("")
        center_seg_dir = dir(sec(0.5))
        mechs_present = []

        # membrane mechanisms
        mt = h.MechanismType(0)

        for i in range(int(mt.count())):
            mt.select(i)
            mt.selected(mname)
            name = mname[0]
            if name in center_seg_dir:
                mechs_present.append(name)


        for mech in mechs_present:
            ms = h.MechanismStandard(mech, 0)
            for j in range(int(ms.count())):
                n = int(ms.name(mname, j))
                name = mname[0]
                pvals = []
                if mech.endswith("_ion"):
                    pvals = [getattr(seg, name) for seg in sec]
                else:
                    mechname = name  # + '_' + mech
                    for seg in sec: #iterate over all segments in this particular section
                        cm = '' if seg.cm==1 else '2' if seg.cm==2 else 'WrongCmValue'
                        #print(seg)
                        if '_table' in mechname or '_arr' in mechname: #added code to reference to a pointer
                            mechname_nosuffix = mechname[:-len(mech)-1] #remove the suffix from the mechname
                            if lkp: #reference to the LUT numpy array
                                if n > 1:
                                    TypeError('2 values for a table pointer?')
                                    pvals.append([getattr(seg, mechname)[i] for i in range(n)]) #test
                                else:
                                    if 'table' in mechname:
                                        #print(mechname)
                                        key = mechname_nosuffix.split('_table')[0]
                                        if not key.endswith(cm): # for V and overtones
                                            key += cm
                                        ref = numpy_element_ref(self.pylkp[key],0)
                                        #print(ref,mechname_nosuffix,getattr(seg,mech))
                                        h.setpointer(ref,mechname_nosuffix,getattr(seg,mech))
                                    if 'arr' in mechname:
                                        #print(mechname)
                                        key = mechname_nosuffix#.split('_arr')[0]
                                        if key == 'Q_arr' and cm == '2':
                                            key = 'Q_arr2'
                                        ref = arrays[key]
                                        #print(ref,mechname_nosuffix,getattr(seg,mech))
                                        h.setpointer(ref,mechname_nosuffix,getattr(seg,mech))
                                    getattr(seg, mechname) #test
                            else: #reference to a dummy pointer: h.t
                                if n > 1:
                                    TypeError('2 values for a table pointer?')
                                    pvals.append([getattr(seg, mechname)[i] for i in range(n)]) #test
                                else:
                                    #print(h._ref_t,mechname_nosuffix,getattr(seg,mech))
                                    h.setpointer(h._ref_t,mechname_nosuffix,getattr(seg,mech)) #just use the time reference as a dummy pointer
                                    #maybe it is because of this that the time vector is reset when plotting in matplotlib? 
                                    #probably not because this is only done in the beginning and time was reset with every plot
                                    #getattr(seg, mechname) #test
                        elif '_s' in mechname and lkp:
                            mechname_nosuffix = mechname[:-len(mech)-1] 
                            if mechname_nosuffix == 'Q_s' and cm == '2':
                                mechname_nosuffix = 'Q_s2'
                            #print(seg, mechname, sizes[mechname_nosuffix])
                            setattr(seg, mechname, sizes[mechname_nosuffix]) #this is no pointer, just a value


    def Py2ModLookup(self, pylkp):
        ''' Convert a 2D python lookup into amplitude (kPa) and charge (nC/cm2) reference vectors
            and a dictionary of 2D hoc matrices for potential (mV) and rate constants (ms-1).
        '''
        assert pylkp.ndims == 2, 'can only convert 2D lookups'

        # Convert lookups independent variables to hoc vectors
        Aref = h.Vector(pylkp.refs['A'] * PA_TO_KPA)
        Qref = h.Vector(pylkp.refs['Q'] * C_M2_TO_NC_CM2)

        # Convert lookup tables to hoc matrices
        local_S_TO_MS = 1 if ABERRA else S_TO_MS # if ABERRA: no conversion needed
        matrix_dict = {'V': Matrix.from_array(pylkp['V'])}  # mV
        if Cm0_var2: #and also Cm0_var? NO
            matrix_dict['V2'] = Matrix.from_array(pylkp['V2']) #mV
        for ratex in self.pneuron.alphax_list.union(self.pneuron.betax_list):
            matrix_dict[ratex] = Matrix.from_array(pylkp[ratex] / local_S_TO_MS)
            if Cm0_var2:
                matrix_dict[ratex+'2'] = Matrix.from_array(pylkp[ratex+'2'] / local_S_TO_MS)
        for taux in self.pneuron.taux_list:
            matrix_dict[taux] = Matrix.from_array(pylkp[taux] * local_S_TO_MS)
        for xinf in self.pneuron.xinf_list:
            matrix_dict[xinf] = Matrix.from_array(pylkp[xinf])

        return Aref, Qref, matrix_dict
    
    def Py2PyLookup(self, pylkp, vbt = 0):
        ''' Convert a 2D python lookup into amplitude (kPa) and charge (nC/cm2) python array
            and a dictionary of (2+2*ov)D python matrices for potential (mV) and rate constants (ms-1).
        '''
        assert pylkp.ndims == 2 + 2*OVERTONES, f'can only convert (2+2ov)D lookups'
        
        # Convert lookups independent variables to hoc vectors
        Aref = h.Vector(pylkp.refs['A'] * PA_TO_KPA) #kPa
        Qref = h.Vector(pylkp.refs['Q'] * C_M2_TO_NC_CM2) #nC/cm2
        overtones = []
        if vbt:
            for ov in range(OVERTONES):
                #print(f'pylkp.refs: {pylkp.refs}')
                overtones.append(len(pylkp.refs[f'AQ{ov+1}'])) #C/m2 #overtones.append(len(pylkp.refs[f'A_{ov+1}'])) #C/m2
                overtones.append(h.Vector(pylkp.refs[f'AQ{ov+1}'])) #C/m2 #overtones.append(h.Vector(pylkp.refs[f'A_{ov+1}'])) #C/m2
                overtones.append(len(pylkp.refs[f'phiQ{ov+1}'])) #rad #overtones.append(len(pylkp.refs[f'B_{ov+1}'])) #C/m2
                overtones.append(h.Vector(pylkp.refs[f'phiQ{ov+1}'])) #rad #overtones.append(h.Vector(pylkp.refs[f'B_{ov+1}'])) #C/m2
        else:
            for ov in range(OVERTONES): #this will never happen as 'VERBATIM = 1 if OVERTONES else 0' (verbatim is necessary when using overtones as the LUT become > 2D)
                #print(f'pylkp.refs: {pylkp.refs}')
                overtones.append(len(pylkp.refs[f'A_{ov+1}'])) #overtones.append(len(pylkp.refs[f'AQ{ov+1}'])) #C/m2
                overtones.append(h.Vector(pylkp.refs[f'A_{ov+1}'])._ref_x[0]) #overtones.append(h.Vector(pylkp.refs[f'AQ{ov+1}'])._ref_x[0]) #C/m2
                overtones.append(len(pylkp.refs[f'B_{ov+1}'])) #overtones.append(len(pylkp.refs[f'phiQ{ov+1}'])) #C/m2
                overtones.append(h.Vector(pylkp.refs[f'B_{ov+1}'])._ref_x[0]) #overtones.append(h.Vector(pylkp.refs[f'phiQ{ov+1}'])._ref_x[0]) #C/m2

        # Convert lookup tables to hoc matrices
        local_S_TO_MS = 1 if ABERRA else S_TO_MS # if ABERRA: no conversion needed
        matrix_dict = {'V': pylkp['V']}  # mV
        if Cm0_var2: #and also Cm0_var? NO
            matrix_dict['V2'] = pylkp['V2'] #mV
        for ov in range(OVERTONES):
            A, B = f'A_{ov+1}', f'B_{ov+1}' #amp, ph = f'A_V{ov+1}', f'phi_V{ov+1}'
            matrix_dict[A] = pylkp[A] #mV #matrix_dict[amp] = pylkp[amp] #mV
            matrix_dict[B] = pylkp[B] #mV #matrix_dict[ph] = pylkp[ph] #rad
            if Cm0_var2:
                matrix_dict[A+'2'] = pylkp[A+'2'] #mV #matrix_dict[amp+'2'] = pylkp[amp+'2'] #mV
                matrix_dict[B+'2'] = pylkp[B+'2'] #mV #matrix_dict[ph+'2'] = pylkp[ph+'2'] #rad
        for ratex in self.pneuron.alphax_list.union(self.pneuron.betax_list):
            matrix_dict[ratex] = pylkp[ratex] / local_S_TO_MS #1/ms
            if Cm0_var2:
                matrix_dict[ratex+'2'] = pylkp[ratex+'2'] / local_S_TO_MS #1/ms
        for taux in self.pneuron.taux_list:
            matrix_dict[taux] = pylkp[taux] * local_S_TO_MS #ms
        for xinf in self.pneuron.xinf_list:
            matrix_dict[xinf] = pylkp[xinf]

        return Aref, Qref, matrix_dict, overtones

    def getBaselineLookup(self):
        ''' Get zero amplitude lookup . '''
        pylkp = self.pneuron.getLookup()  # get 1D charge-dependent lookup
        pylkp.refs = {'A': np.array([0.]), **pylkp.refs}  # add amp as first dimension
        pylkp.tables = EffectiveVariablesDict(
            {k: np.array([v]) for k, v in pylkp.items()})  # add amp dimension to tables
        return pylkp

    def setPyLookup(self):
        ''' Set the appropriate model 2D lookup. '''
        if not hasattr(self, 'pylkp') or self.pylkp is None:
            self.pylkp = self.getBaselineLookup()

    def setModLookup(self, *args, **kwargs):
        ''' Get the appropriate model 2D lookup and translate it to Hoc. '''
        # Set Lookup
        self.setPyLookup(*args, **kwargs)

        # Convert to HOC equivalents and store them as class attributes
        if OVERTONES:
            self.Aref, self.Qref, self.lkp, self.overtones = self.Py2PyLookup(self.pylkp, vbt = VERBATIM) #expand 2D to multiD (depending on overtones)
        else:
            self.Aref, self.Qref, self.lkp = self.Py2ModLookup(self.pylkp)
        if Cm0_var2:
            self.Qextref = h.Vector(self.pylkp.Q_ext * C_M2_TO_NC_CM2)

    @staticmethod
    def setFuncTable(mechname, fname, matrix, xref, yref, Cm0=None):
        ''' Set the content of a 2-dimensional FUNCTION TABLE of a density mechanism.

            :param mechname: name of density mechanism
            :param fname: name of the FUNCTION_TABLE reference in the mechanism
            :param matrix: HOC Matrix object with values to be linearly interpolated
            :param xref: HOC Vector object with reference values for interpolation in 1st dimension
            :param yref: HOC Vector object with reference values for interpolation in 2nd dimension
            :Cm0: resting membrane capacitance
            :return: the updated HOC object
        '''
        # Check conformity of inputs
        dims_not_matching = 'reference vector size ({}) does not match matrix {} dimension ({})'
        nx, ny = int(matrix.nrow()), int(matrix.ncol())
        #nx, ny are the dimensions of the LUT matrix, only 2D?
        assert xref.size() == nx, dims_not_matching.format(xref.size(), '1st', nx)
        assert yref.size() == ny, dims_not_matching.format(yref.size(), '2nd', ny)

        # Get the HOC function that fills in a specific FUNCTION_TABLE in a mechanism
        #ASSUMPTION: all the LUT for all mechanisms are loaded in, even if not all function tables need to be used for a specific compartment/cell #POTENTIAL RISK?
        #(different compartment types contain different mech & cells don't have all the available mechs)
        #print(f'setFuncTable mechname:{mechname}, fname: {fname}')
        if 'real' in mechname and 'neuron' in mechname: #also "if ABERRA:" can be used #BREAKPOINT
            if fname == 'V' or fname == 'V2':
                for mech in mech_mapping.values():
                    try:
                        if Cm0:
                            fillTable = getattr(h, f'table_{fname}_{mech}{Cm0_map[Cm0]}')
                        elif Cm0_var2: #when using Cm0_var2, no iteration over Cm0's so Cm0 = None
                            #print(f'table_V_{mech}')
                            if fname == 'V':
                                fillTable = getattr(h, f'table_V_{mech}') #both variants have 'V' as defined table but V or V2 is loaded depending on which variant
                            else:
                                fillTable = getattr(h, f'table_V_{mech}2')
                        else:
                            fillTable = getattr(h, f'table_{fname}_{mech}')
                        #print(f"fillTable1: {fillTable}")
                        fillTable(matrix._ref_x[0][0], nx, xref._ref_x[0], ny, yref._ref_x[0]) #call the table filler for every mechanisms that contains V LUT
                    except:
                        pass
                        #print(f'{mech} has no attribute {fname}') #LOG OUTPUT
                #fillTable = getattr(h, f'table_{fname}_K_Pst') #this assumes that the mechanism K_Pst is always present in the chosen cell -> replaced with iteration over all mechanisms
                return #stop after iterating over different mechanisms as the remainder of the code is not for V LUTs
            else:
                #print(f'table_{fname}_{mech_mapping[fname.split("_")[-1]]}')
                if Cm0:
                    #fillTable = getattr(h, f'table_{fname}{Cm0_map[Cm0]}_{mech_mapping[fname.split("_")[-1]]}{Cm0_map[Cm0]}') #to distinguish tables from 0.01-variant and 0.02-variant (LUT2)
                    fillTable = getattr(h, f'table_{fname}_{mech_mapping[fname.split("_")[-1]]}{Cm0_map[Cm0]}')
                elif Cm0_var2:
                    mech = fname.split("_")[-1]
                    if mech in mech_mapping.keys(): #if the mechname is in the keys: 0.01-variant
                        #print(f'table_{fname}_{mech_mapping[fname.split("_")[-1]]}')
                        fillTable = getattr(h, f'table_{fname}_{mech_mapping[fname.split("_")[-1]]}')
                    else: #if the mechname is not in the keys, it is the 0.02-variant, remove 2 and add it afterwards
                        #print(f'table_{fname}_{mech_mapping[fname.split("_")[-1][:-1]]}2')
                        fillTable = getattr(h, f'table_{fname}_{mech_mapping[fname.split("_")[-1][:-1]]}2')

                else:
                    fillTable = getattr(h, f'table_{fname}_{mech_mapping[fname.split("_")[-1]]}')
        else:
            fillTable = getattr(h, f'table_{fname}_{mechname}') #original line: in case of a single mechanism -> not the case for Aberra cells (multiple mechanisms)
                                                                #the function table has the following structure in hoc: table_variable_mechname

        # Call function
        #print(f"fillTable2: {fillTable}")
        fillTable(matrix._ref_x[0][0], nx, xref._ref_x[0], ny, yref._ref_x[0])

    @staticmethod
    def setPyFuncTable(mechname, fname, matrix, xref, yref, Cm0=None, overtones=None):
        ''' Set the content of a 2-dimensional FUNCTION TABLE of a density mechanism with a python (>2D) matrix output table and overtone input vectors.

            :param mechname: name of density mechanism
            :param fname: name of the FUNCTION_TABLE reference in the mechanism
            :param matrix: HOC Matrix object with values to be linearly interpolated
            :param xref: HOC Vector object with reference values for interpolation in 1st dimension
            :param yref: HOC Vector object with reference values for interpolation in 2nd dimension
            :Cm0: resting membrane capacitance
            :overtones: overtone input vectors
            :return: the updated HOC object
        '''
        # Check conformity of inputs
        dims_not_matching = 'reference vector size ({}) does not match matrix {} dimension ({})'
        nx, ny = int(matrix.shape[0]), int(matrix.shape[1])
        #nx, ny are the dimensions of the LUT matrix, only 2D?
        assert len(xref) == nx, dims_not_matching.format(xref.size, '1st', nx)
        assert len(yref) == ny, dims_not_matching.format(yref.size, '2nd', ny)

        # Get the HOC function that fills in a specific FUNCTION_TABLE in a mechanism
        #ASSUMPTION: all the LUT for all mechanisms are loaded in, even if not all function tables need to be used for a specific compartment/cell #POTENTIAL RISK?
        #(different compartment types contain different mech & cells don't have all the available mechs)
        #print(f'setFuncTable mechname:{mechname}, fname: {fname}')

        if 'A_1' in fname or 'B_1' in fname:
            print(f'{fname} is not loaded into the NMODL files tables.')
            return

        if 'real' in mechname and 'neuron' in mechname: #also "if ABERRA:" can be used #BREAKPOINT
            if fname == 'V' or fname == 'V2' or 'A_V' in fname or 'phi_V' in fname or 'A_' in fname or 'B_' in fname: #V0, V0_2, Vk, phik, Ak, Bk
                for mech in mech_mapping.values():
                    try:
                        if Cm0:
                            fillTable = getattr(h, f'table_{fname}_{mech}{Cm0_map[Cm0]}')
                        elif Cm0_var2: #when using Cm0_var2, no iteration over Cm0's so Cm0 = None
                            #print(f'table_V_{mech}')
                            if fname.endswith('2'):
                                fillTable = getattr(h, f'table_{fname[:-1]}_{mech}2') #both variants have 'V' as defined table but V or V2 is loaded depending on which variant
                            else:
                                fillTable = getattr(h, f'table_{fname}_{mech}')
                        else:
                            fillTable = getattr(h, f'table_{fname}_{mech}')
                        #print(f"fillTable1: {fillTable}")
                        mat_ptr = numpy_element_ref(matrix, 0) #pass the pointer of the numpy matrix to the FUNCTION TABLE
                        #fillTable(mat_ptr, nx, min(xref), max(xref), ny, min(yref), max(yref), float(overtones[0]), float(min(overtones[1])), float(max(overtones[1])), float(overtones[2]), float(min(overtones[3])), float(max(overtones[3]))) #call the table filler for every mechanisms that contains V LUT
                        fillTable(mat_ptr, nx, xref._ref_x[0], ny, yref._ref_x[0],*overtones) #call the table filler for every mechanisms that contains V LUT
                    except:
                        pass
                        #print(f'{mech} has no attribute {fname}') #LOG OUTPUT
                #quit() 
                #fillTable = getattr(h, f'table_{fname}_K_Pst') #this assumes that the mechanism K_Pst is always present in the chosen cell -> replaced with iteration over all mechanisms
                return #stop after iterating over different mechanisms as the remainder of the code is not for V LUTs
            else:
                #print(f'table_{fname}_{mech_mapping[fname.split("_")[-1]]}')
                if Cm0:
                    #fillTable = getattr(h, f'table_{fname}{Cm0_map[Cm0]}_{mech_mapping[fname.split("_")[-1]]}{Cm0_map[Cm0]}') #to distinguish tables from 0.01-variant and 0.02-variant (LUT2)
                    fillTable = getattr(h, f'table_{fname}_{mech_mapping[fname.split("_")[-1]]}{Cm0_map[Cm0]}')
                elif Cm0_var2:
                    mech = fname.split("_")[-1]
                    if mech in mech_mapping.keys(): #if the mechname is in the keys: 0.01-variant
                        #print(f'table_{fname}_{mech_mapping[fname.split("_")[-1]]}')
                        fillTable = getattr(h, f'table_{fname}_{mech_mapping[fname.split("_")[-1]]}')
                    else: #if the mechname is not in the keys, it is the 0.02-variant, remove 2 and add it afterwards
                        #print(f'table_{fname}_{mech_mapping[fname.split("_")[-1][:-1]]}2')
                        fillTable = getattr(h, f'table_{fname}_{mech_mapping[fname.split("_")[-1][:-1]]}2')

                else:
                    fillTable = getattr(h, f'table_{fname}_{mech_mapping[fname.split("_")[-1]]}')
        else:
            fillTable = getattr(h, f'table_{fname}_{mechname}') #original line: in case of a single mechanism -> not the case for Aberra cells (multiple mechanisms)
                                                                #the function table has the following structure in hoc: table_variable_mechname

        # Call function
        #print(f"fillTable2: {fillTable}")
        mat_ptr = numpy_element_ref(matrix, 0)
        #fillTable(mat_ptr, nx, min(xref), max(xref), ny, min(yref), max(yref),overtones[0],min(overtones[1]), max(overtones[1]), overtones[2], min(overtones[3]), max(overtones[3]))
        fillTable(mat_ptr, nx, xref._ref_x[0], ny, yref._ref_x[0],*overtones)

    def setFuncTables(self, *args, **kwargs):
        ''' Set neuron-specific interpolation tables along the charge dimension,
            and link them to FUNCTION_TABLEs in the MOD file of the corresponding
            membrane mechanism.
        '''
        if Cm0_var2:
            self.setModLookup(*args, **kwargs)
            logger.debug(f'setting {self.mechname} function tables')
            if OVERTONES and VERBATIM:
                for sec in self.seclist:
                    #print(sec.nrnsec)
                    self.psection(sec.nrnsec,lkp=1)
            else:
                for k, v in self.lkp.items():
                    if OVERTONES:
                        if v.shape[1] == len(self.Qref):
                            self.setPyFuncTable(self.mechname, k, v, self.Aref, self.Qref,overtones=self.overtones)
                        else:
                            self.setPyFuncTable(self.mechname, k, v, self.Aref, self.Qextref, overtones=self.overtones) 
                    else: 
                        if v.ncol() == self.Qref.size():
                            self.setFuncTable(self.mechname, k, v, self.Aref, self.Qref)
                        else:
                            self.setFuncTable(self.mechname, k, v, self.Aref, self.Qextref)
        else:
            self.setModLookup(*args, **kwargs)
            logger.debug(f'setting {self.mechname} function tables')
            if OVERTONES and VERBATIM:
                for sec in self.seclist:
                    #print(sec.nrnsec)
                    self.psection(sec.nrnsec,lkp=1)
            else:
                if OVERTONES:
                    for k, v in self.lkp.items():
                        self.setPyFuncTable(self.mechname, k, v, self.Aref, self.Qref, overtones=self.overtones)    
                else:
                    for k, v in self.lkp.items():
                        self.setFuncTable(self.mechname, k, v, self.Aref, self.Qref)

    @staticmethod
    def fixStimVec(stim, dt):
        ''' Quick fix for stimulus vector discrepancy for fixed time step simulations. '''
        if dt is None:
            return stim
        else:
            return np.hstack((stim[1:], stim[-1]))

    @staticmethod
    def outputDataFrame(t, stim, probes):
        ''' Return output in dataframe with prepended initial conditions (prior to stimulation). '''
        sol = TimeSeries(t, stim, {k: v.to_array() for k, v in probes.items()})
        if 'Vext' in sol:  # add "Vin" field if solution has both "Vm" and "Vext" fields
            sol['Vin'] = sol['Vm'] + sol['Vext']
        sol['Cm'] = sol['Qm'] / sol['Vm'] * V_TO_MV
        sol.prepend(t0=0)
        return sol

    @Model.logNSpikes
    @Model.checkTitrate
    @Model.addMeta
    @Model.logDesc
    def simulate(self, drive, pp, dt=None, atol=None):
        ''' Set appropriate recording vectors, integrate and return output variables.

            :param drive: drive object
            :param pp: pulse protocol object
            :param dt: integration time step for fixed time step method (s)
            :param atol: absolute error tolerance for adaptive time step method.
            :return: output dataframe
        '''
        # Set recording vectors
        t = self.setTimeProbe()
        stim = self.section.setStimProbe()
        probes = self.section.setProbes()

        # Set drive and integrate model
        self.setDrive(drive) #added by Joaquin: this function is not defined (yet) (simulate in SENN is redefined and uses setDrives which is only defined in the SENN class)
        self.integrate(pp, dt, atol)
        self.clearDrives()

        # Return output dataframe
        return self.outputDataFrame(t.to_array(), self.fixStimVec(stim.to_array(), dt), probes)

    @property
    def titrationFunc(self):
        return self.pneuron.titrationFunc

    def titrate(self, *args, **kwargs):
        return titrate(self, *args, **kwargs)

    def getPltVars(self, *args, **kwargs):
        Cm_pltvar = BilayerSonophore.getPltVars()['Cm']
        del Cm_pltvar['func']
        Cm_pltvar['bounds'] = (0.0, 1.5 * self.pneuron.Cm0 * F_M2_TO_UF_CM2)
        return {
            **self.pneuron.getPltVars(*args, **kwargs),
            'Cm': Cm_pltvar,
            'Vext': {
                'desc': 'extracellular potential',
                'label': 'V_{ext}',
                'unit': 'mV',
                # 'strictbounds': (-0.2, 0.2)
            },
            'Vin': {
                'desc': 'intracellular potential',
                'label': 'V_{in}',
                'unit': 'mV'
            }
        }

    @property
    def pltScheme(self):
        return self.pneuron.pltScheme

    @property
    @abc.abstractmethod
    def filecodes(self, *args):
        raise NotImplementedError

    def filecode(self, *args):
        return filecode(self, *args)

    def simAndSave(self, *args, **kwargs):
        return simAndSave(self, *args, **kwargs)

    @property
    def has_ext_mech(self):
        #print('self.seclist = ',self.seclist)
        return any(sec.has_ext_mech for sec in self.seclist)


class SpatiallyExtendedNeuronModel(NeuronModel):
    ''' Generic interface for spatially-extended NEURON models. '''

    # Whether to use equivalent currents for imposed extracellular voltage fields
    use_equivalent_currents = False
    has_passive_sections = False

    @abc.abstractstaticmethod
    def getMetaArgs(meta):
        raise NotImplementedError

    @classmethod
    def initFromMeta(cls, meta, construct=False):
        args, kwargs = cls.getMetaArgs(meta)
        return cls(*args, **kwargs, construct=construct)

    @staticmethod
    def inputs():
        return {
            'section': {
                'desc': 'section',
                'label': 'section',
                'unit': ''
            }
        }

    def filecodes(self, source, pp, *_):
        return {
            **self.modelcodes,
            **source.filecodes,
            'nature': pp.nature,
            **pp.filecodes
        }

    @property
    def rmin(self):
        ''' Lower bound for axial resistance * membrane area (Ohm/cm2). '''
        return None

    @property
    def rs(self):
        return self._rs

    @rs.setter
    def rs(self, value):
        if value <= 0:
            raise ValueError('longitudinal resistivity must be positive')
        self.set('rs', value)

    def str_resistivity(self):
        return f'rs = {si_format(self.rs)}Ohm.cm'

    @property
    @abc.abstractmethod
    def refsection(self):
        ''' Model reference section (used mainly to monitor stimon parameter). '''
        raise NotImplementedError

    @property
    def sectypes(self):
        return list(self.sections.keys())

    def getSectionsDetails(self):
        ''' Get details about the model's sections. '''
        d = {}
        for secdict in self.sections.values():
            sec = secdict[list(secdict.keys())[0]]
            dd = sec.getDetails()
            if len(d) == 0:
                d = {'nsec': [], **{k: [] for k in dd.keys()}}
            d['nsec'].append(len(secdict))
            for k, v in dd.items():
                d[k].append(v)
        return pd.DataFrame(d, index=self.sectypes)

    def logSectionsDetails(self):
        return f'sections details:\n{self.getSectionsDetails().to_markdown()}'

    def printTopology(self):
        ''' Print the model's topology. '''
        h.topology()

    @property
    @abc.abstractmethod
    def nonlinear_sections(self):
        ''' Sections that contain nonlinear dynamics. '''
        raise NotImplementedError

    @property
    def drives(self):
        if not hasattr(self, '_drives'):
            self._drives = []
        return self._drives

    @drives.setter
    def drives(self, value):
        if value is None:
            self._drives = None
        else:
            if not isIterable(value):
                raise ValueError('drives must be an iterable')
            for item in value:
                if not hasattr(item, 'set'):
                    raise ValueError(f'drive {item} has no "set" method')
            self._drives = value

    def desc(self, meta):
        return f'{self}: simulation @ {meta["source"]}, {meta["pp"].desc}'

    def connect(self, k1, i1, k2, i2):
        ''' Connect two sections referenced by their type and index.

            :param k1: type of parent section
            :param i1: index of parent section in subtype dictionary
            :param k2: type of child section
            :param i2: index of child section in subtype dictionary
        '''
        self.sections[k2][f'{k2}{i2:d}'].connect(self.sections[k1][f'{k1}{i1:d}'])

    def setIClamps(self, Iinj_dict):
        ''' Set distributed intracellular current clamps. '''
        logger.debug(f'Intracellular currents:')
        with np.printoptions(**array_print_options):
            for k, Iinj in Iinj_dict.items():
                logger.debug(f'{k}: Iinj = {Iinj} nA')
        iclamps = []
        for k, Iinj in Iinj_dict.items():
            iclamps += [IClamp(sec, I) for sec, I in zip(self.sections[k].values(), Iinj)]
        return iclamps

    def toInjectedCurrents(self, Ve):
        ''' Convert extracellular potential array into equivalent injected currents.

            :param Ve: model-sized vector of extracellular potentials (mV)
            :return: model-sized vector of intracellular injected currents (nA)
        '''
        raise NotImplementedError

    def setVext(self, Ve_dict):
        ''' Set distributed extracellular voltages. '''
        logger.debug(f'Extracellular potentials:')
        with np.printoptions(**array_print_options):
            for k, Ve in Ve_dict.items():
                logger.debug(f'{k}: Ve = {Ve} mV')
        # Variant 1: inject equivalent intracellular currents
        if self.use_equivalent_currents:
            return self.setIClamps(self.toInjectedCurrents(Ve_dict))
        # Variant 2: insert extracellular mechanisms
        else:
            emechs = []
            for k, Ve in Ve_dict.items():
                emechs += [ExtField(sec, v) for sec, v in zip(self.sections[k].values(), Ve)]
            return emechs

    @property
    def drive_funcs(self):
        return {
            IntracellularCurrent: self.setIClamps,
            ExtracellularCurrent: self.setVext,
            GaussianVoltageSource: self.setVext,
            UniformVoltageSource: self.setVext
        }

    def setDrives(self, source):
        ''' Set distributed stimulus amplitudes. '''
        self.drives = []
        amps_dict = source.computeDistributedAmps(self)
        match = False
        for source_class, drive_func in self.drive_funcs.items():
            if isinstance(source, source_class):
                self.drives = drive_func(amps_dict)
                match = True
        if not match:
            raise ValueError(f'Unknown source type: {source}')

    def clearDrives(self):
        self.drives = None

    @property
    def Aranges(self):
        return {
            IntracellularCurrent: IINJ_RANGE,
            ExtracellularCurrent: VEXT_RANGE,
            GaussianVoltageSource: VEXT_RANGE,
            UniformVoltageSource: VEXT_RANGE
        }

    def getArange(self, source):
        ''' Get the stimulus amplitude range allowed at the fiber level. '''
        for source_class, Arange in self.Aranges.items():
            if isinstance(source, source_class):
                return [source.computeSourceAmp(self, x) for x in Arange]
        raise ValueError(f'Unknown source type: {source}')

    @Model.checkTitrate
    @Model.addMeta
    @Model.logDesc
    def simulate(self, source, pp, dt=None, atol=None):
        ''' Set appropriate recording vectors, integrate and return output variables.

            :param source: source object
            :param pp: pulsed protocol object
            :param dt: integration time step for fixed time step method (s)
            :param atol: absolute error tolerance for adaptive time step method.
            :return: output dataframe
        '''
        # Reset time to zero (redundant, but helps with clarity during debugging)
        h.t = 0
        #disconnect in hoc
        # for sec in h.allsec():
        #     print(sec.Ra)
        #     h.disconnect(sec=sec)
        
        # Set distributed drives
        self.setDrives(source)

        # Set recording vectors
        t = self.setTimeProbe()
        stim = self.refsection.setStimProbe()
        all_probes = {}
        no_probes = []
        for sectype, secdict in self.sections.items():
            for k, sec in secdict.items():
                if ABERRA:
                    if sec.random_mechname: #only do this for sections that have mechanisms -> POTENTIAL RISK (the other ones are skipped and will return None when called later on)
                        all_probes[k] = sec.setProbes() #puts a probe for every section in a dictionary    
                    else:
                        no_probes.append(sec.nrnsec)             
                else:
                    all_probes[k] = sec.setProbes() #puts a probe for every section in a dictionary
        if len(no_probes) > 0: #only print if there is a probeless section
            print(f'No probes for these sections: {no_probes}\n')
        if DEBUG_OV:
            self.t = t
            self.all_probes = all_probes

        # Integrate model
        self.integrate(pp, dt, atol)
        print(f'simulation time: {h.t}')
        self.clearDrives()

        # Return output dataframe dictionary
        t = t.to_array()  # s
        stim = self.fixStimVec(stim.to_array(), dt)
        return SpatiallyExtendedTimeSeries({
            id: self.outputDataFrame(t, stim, probes) for id, probes in all_probes.items()})

    def getSpikesTimings(self, data, zcross=True, spikematch='majority'):
        ''' Return an array containing occurence times of spikes detected on a collection of sections.

            :param data: simulation output dataframe
            :param zcross: boolean stating whether to use ascending zero-crossings preceding peaks
                as temporal reference for spike occurence timings
            :return: dictionary of spike occurence times (s) per section.
        '''
        tspikes = {}
        nspikes = np.zeros(len(data.items()))
        errmsg = 'Ascending zero crossing #{} (t = {:.2f} ms) not prior to peak #{} (t = {:.2f} ms)'

        for i, (id, df) in enumerate(data.items()):

            # Detect spikes on current trace
            ispikes, *_ = detectSpikes(df, key='Vm', mph=SPIKE_MIN_VAMP, mpt=SPIKE_MIN_DT,
                                       mpp=SPIKE_MIN_VPROM)
            nspikes[i] = ispikes.size

            if ispikes.size > 0:
                # Extract time vector
                t = df['t'].values  # s
                if zcross:
                    # Consider spikes as time of zero-crossing preceding each peak
                    Vm = df['Vm'].values  # mV
                    i_zcross = np.where(np.diff(np.sign(Vm)) > 0)[0]  # ascending zero-crossings
                    # If mismatch, remove irrelevant zero-crossings by taking only the ones
                    # preceding each detected peak
                    if i_zcross.size > ispikes.size:
                        i_zcross = np.array([i_zcross[(i_zcross - i1) < 0].max() for i1 in ispikes])
                    # Compute slopes (mV/s)
                    slopes = (Vm[i_zcross + 1] - Vm[i_zcross]) / (t[i_zcross + 1] - t[i_zcross])
                    # Interpolate times of zero crossings
                    tzcross = t[i_zcross] - (Vm[i_zcross] / slopes)
                    for ispike, (tzc, tpeak) in enumerate(zip(tzcross, t[ispikes])):
                        assert tzc < tpeak, errmsg.format(
                            ispike, tzc * S_TO_MS, ispike, tpeak * S_TO_MS)
                    tspikes[id] = tzcross
                else:
                    tspikes[id] = t[ispikes]

        if spikematch == 'strict':
            # Assert consistency of spikes propagation
            assert np.all(nspikes == nspikes[0]), 'Inconsistent spike number across sections'
            if nspikes[0] == 0:
                logger.warning('no spikes detected')
                return None
        else:
            # Use majority voting
            nfrequent = stats.mode(nspikes, keepdims=False).mode.astype(int)
            tspikes = {k: v for k, v in tspikes.items() if len(v) == nfrequent}

        return pd.DataFrame(tspikes)

    def getSpikeAmp(self, data, ids=None, key='Vm', out='range'):
        # By default, consider all sections with nonlinear dynamics
        if ids is None:
            ids = list(self.nonlinear_sections.keys())
        amps = np.array([np.ptp(data[id][key].values) for id in ids])
        if out == 'range':
            return amps.min(), amps.max()
        elif out == 'median':
            return np.median(amps)
        elif out == 'mean':
            return np.mean(amps)
        else:
            raise AttributeError(f'invalid out option: {out}')

    def titrationFunc(self, data):
        return self.isExcited(data)

    def getStartPoint(self, Arange):
        scale = 'lin' if Arange[0] == 0 else 'log'
        return Thresholder.getStartPoint(Arange, x=REL_START_POINT, scale=scale)

    def getAbsConvThr(self, Arange):
        return np.abs(Arange[1] - Arange[0]) / 1e4

    def titrate(self, source, pp):
        ''' Use a binary search to determine the threshold amplitude needed to obtain
            neural excitation for a given pulsing protocol.

            :param source: source object
            :param pp: pulsed protocol object
            :return: determined threshold amplitude
        '''
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

    def getCurrentsDict(self, df):
        ''' Compute currents dictionary. '''
        Vm = df['Vm'].values
        states = {k: df[k].values for k in self.pneuron.states.keys()}
        membrane_currents = {k: cfunc(Vm, states) for k, cfunc in self.pneuron.currents().items()}
        cdict = {
            'Ax': df['iax'],
            'Leak': membrane_currents.pop('iLeak')
        }
        cdict.update({k[1:]: v for k, v in membrane_currents.items()})
        cdict['Net'] = self.pneuron.iNet(Vm, states) + cdict['Ax']
        return cdict

    def getBuildupContributions(self, df, tthr):
        t, currents = df['t'].values, self.getCurrentsDict(df)
        del currents['Net']
        # Interpolate currents at regular time step during build-up interval
        tsub = np.linspace(0, tthr, 100)  # s
        buildup_currents = {k: np.interp(tsub, t, v) for k, v in currents.items()}  # mA/m2

        # Compute charge variation associated to each current during build-up
        dt = np.diff(tsub)[0]  # s
        buildup_charges = {k: -np.sum(v) * dt * MA_TO_A for k, v in buildup_currents.items()}  # C/m2

        # Return charge variations normalized by resting capacitance
        return {k: v / self.pneuron.Cm0 * V_TO_MV for k, v in buildup_charges.items()}  # mV


