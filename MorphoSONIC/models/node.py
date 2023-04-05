# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2018-08-27 09:23:32
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-04-05 16:15:23

import numpy as np
from neuron import h

from PySONIC.core import PointNeuron, ElectricDrive
from PySONIC.utils import logger

from ..constants import *
from ..core import IClamp, NeuronModel, addSonicFeatures


@addSonicFeatures
class Node(NeuronModel):
    ''' Node model. '''

    def __init__(self, pneuron, **kwargs):
        ''' Initialization.

            :param pneuron: point-neuron model
        '''
        self.pneuron = pneuron
        super().__init__(**kwargs)

    def __repr__(self):
        s = f'{self.__class__.__name__}({self.pneuron}'
        if self.Idrive != 0.:
            s = f'{s}, Idrive = {self.Idrive:.2f} mA/m2'
        return f'{s})'

    def createSections(self):
        self.section = self.createSection(
            'node', mech=self.mechname, states=self.pneuron.statesNames())

    def clearSections(self):
        self.section = None

    def clearDrives(self):
        self.drive = None

    @property
    def simkey(self):
        return self.pneuron.simkey

    @property
    def seclist(self):
        return [self.section]

    @property
    def drives(self):
        return [self.drive]

    def getAreaNormalizationFactor(self):
        ''' Return area normalization factor '''
        A0 = self.section.Am / M_TO_CM**2  # section area (m2)
        A = self.pneuron.area              # neuron membrane area (m2)
        return A0 / A

    @staticmethod
    def getNSpikes(data):
        return PointNeuron.getNSpikes(data)

    @property
    def meta(self):
        meta = self.pneuron.meta
        if self.Idrive != 0.:
            meta['Idrive'] = self.Idrive
        return meta

    def desc(self, meta):
        return f'{self}: simulation @ {meta["drive"].desc}, {meta["pp"].desc}'

    def currentDensityToCurrent(self, i):
        ''' Convert an intensive current density to an extensive current.

            :param i: current density (mA/m2)
            :return: current (nA)
        '''
        Iinj = i * self.section.Am / M_TO_CM**2 * MA_TO_NA  # nA
        logger.debug(f'Equivalent injected current: {Iinj:.1f} nA')
        return Iinj

    def setIClamp(self, drive):
        ''' Set intracellular electrical stimulation drive

            :param drive: electric drive object.
        '''
        return IClamp(self.section, self.currentDensityToCurrent(drive.I))

    @property
    def Idrive(self):
        ''' Idrive getter that returns 0 if undefined '''
        try:
            return self._Idrive
        except AttributeError:
            return 0.
    
    def setConstantDrive(self, value):
        ''' 
        Set a constant driving current.
        
        :param value: current density (mA/m2)
        '''
        self._Idrive = value
        # Convert current density (in mA/m2) to injected current (in nA)
        Iinj = self.currentDensityToCurrent(value)
        # Create and activate IClamp object if it does not exist already
        if not hasattr(self, 'iclamp') or self.iclamp is None:
            self.idrive_clamp = IClamp(self.section, Iinj)
            self.idrive_clamp.set(1)
        # Otherwise, update amplitude
        else:
            self.idrive_clamp.amp = Iinj
    
    def enableConstantDrive(self):
        ''' Enable constant driving current '''
        if self.Idrive == 0.:
            logger.warning('no constant driving current assigned -> ignoring')
        self.idrive_clamp.set(1)
    
    def disableConstantDrive(self):
        if self.Idrive == 0.:
            logger.warning('no constant driving current assigned -> ignoring')
        self.idrive_clamp.set(0)
    
    def clear(self):
        self.idrive_clamp = None
        super().clear()
    
    def construct(self):
        super().construct()
        if self.Idrive != 0:
            self.setConstantDrive(self.Idrive)

    @property
    def drive_funcs(self):
        return {ElectricDrive: self.setIClamp}

    def setDrive(self, drive):
        ''' Set stimulation drive.

            :param drive: drive object.
        '''
        logger.debug(f'Stimulus: {drive}')
        self.drive = None
        match = False
        for drive_class, drive_func in self.drive_funcs.items():
            if isinstance(drive, drive_class):
                self.drive = drive_func(drive)
                match = True
        if not match:
            raise ValueError(f'Unknown drive type: {drive}')

    @property
    def Arange_funcs(self):
        return {ElectricDrive: self.pneuron.getArange}

    def getArange(self, drive):
        ''' Get the stimulus amplitude range allowed. '''
        for drive_class, Arange_func in self.Arange_funcs.items():
            if isinstance(drive, drive_class):
                return Arange_func(drive)
        raise ValueError(f'Unknown drive type: {drive}')

    def filecodes(self, *args):
        codes = self.pneuron.filecodes(*args)
        if self.Idrive != 0.:
            codes['Idrive'] = f'Idrive{self.Idrive:.1f}mAm2'
        codes['method'] = 'NEURON'
        return codes

    def setSpikeDetector(self, thr=0.):
        '''
        Set up a spike detector
        
        :param thr: threshold voltage in the positive direction for spike detection
        '''
        self.apc = h.APCount(self.section(0.5))
        self.apc.thresh = thr
        self.aptimes = h.Vector()
        self.apc.record(self.aptimes)
    
    def clearSpikedetector(self):
        ''' Clear spike detector '''
        self.apc = None
        self.aptimes = None
    
    def getSpikeTimes(self):
        ''' Extract detect spike times (in s) as a numpy array '''
        return np.array(self.aptimes.to_python()) * 1e-3


@addSonicFeatures
class PassiveNode(Node.__original__):

    has_passive_sections = True

    def createSections(self):
        self.section = self.createSection('node', mech='pas', Cm0=self.pneuron.Cm0 * 1e2)
        self.section.setPassiveG(self.pneuron.gLeak * 1e-4)  # S/cm2
        self.section.setPassiveE(self.pneuron.Vm0)  # mV
