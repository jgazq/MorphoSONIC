# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-04 18:26:42
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-04-05 16:28:49

''' Utilities to manipulate HOC objects. '''

import numpy as np
from neuron import h, hclass, nrn

from PySONIC.utils import isWithin, logger, npformat
from ..constants import *
from ..utils import seriesGeq


def getNetStim(freq, start=0., number=1e9):
    '''
    Wrapper around NetStim allowing to set parameters in 1-liner.
    
    :param freq: spiking frequency of pre-synaptic drive (Hz)
    :param start (optional): start time (s)
    :param number (optional): total number of pre-synaptic spikes
    :return: NetStim object
    '''
    stim = h.NetStim()
    stim.number = number
    stim.start = start * S_TO_MS  # ms
    stim.interval = S_TO_MS / freq  # ms
    return stim


class Probe(hclass(h.Vector)):
    ''' Interface to a Hoc vector recording a particular variable. '''

    def __new__(cls, *args, **kwargs):
        ''' Instanciation. '''
        return super(Probe, cls).__new__(cls)

    def __init__(self, variable, factor=1.):
        ''' Initialization. '''
        super().__init__()
        self.factor = factor
        self.record(variable)

    def to_array(self):
        ''' Return itself as a numpy array. '''
        return np.array(self.to_python()) * self.factor

    def clear(self):
        super().__init__()

    def __del__(self):
        self.clear()


class Matrix(hclass(h.Matrix)):
    ''' Interface to Hoc Matrix with extra functionalities. '''

    def to_array(self):
        ''' Return itself as a numpy array. '''
        return np.array([self.getrow(i).to_python() for i in range(int(self.nrow()))])

    @classmethod
    def from_array(cls, arr):
        ''' Create matrix object from numpy array. '''
        m = cls(*arr.shape)
        for i, row in enumerate(arr):
            m.setrow(i, h.Vector(row))
        return m

    @property
    def nRow(self):
        ''' Number of rows in matrix with integer casting. '''
        return int(self.nrow())

    @property
    def nCol(self):
        ''' Number of columns in matrix with integer casting. '''
        return int(self.ncol())

    def setVal(self, *args, **kwargs):
        ''' Set value to an element. '''
        self.setval(*args, **kwargs)

    def getVal(self, *args, **kwargs):
        ''' Get value of an element. '''
        return self.getval(*args, **kwargs)

    def addVal(self, irow, jcol, val):
        ''' Add value to an element. '''
        self.setVal(irow, jcol, self.getVal(irow, jcol) + val)

    def setRow(self, *args, **kwargs):
        ''' Set values to a row. '''
        self.setrow(*args, **kwargs)

    def getRow(self, *args, **kwargs):
        ''' Get values of a row. '''
        return self.getrow(*args, **kwargs)

    def setCol(self, *args, **kwargs):
        ''' Set values to a column. '''
        self.setcol(*args, **kwargs)

    def getCol(self, *args, **kwargs):
        ''' Get values of a column. '''
        return self.getcol(*args, **kwargs)

    def mulRow(self, i, x):
        ''' Multiply a row by a scalar. '''
        self.setRow(i, self.getRow(i).mul(x))

    def mulCol(self, j, x):
        ''' Multiply a column by a scalar. '''
        self.setCol(j, self.getCol(j).mul(x))

    def mulRows(self, v):
        ''' Multiply rows by independent scalar values from a vector. '''
        assert v.size == self.nRow, f'Input vector must be of size {self.nRow}'
        for i, x in enumerate(v):
            self.mulRow(i, x)

    def mulCols(self, v):
        ''' Multiply columns by independent scalar values from a vector. '''
        assert v.size == self.nCol, f'Input vector must be of size {self.nCol}'
        for j, x in enumerate(v):
            self.mulCol(j, x)

    def addRowSlice(self, i, v, col_offset=0):
        ''' Add a vector to a given column with optional row offset. '''
        for j, x in enumerate(v):
            self.addVal(i, j + col_offset, x)

    def addColSlice(self, j, v, row_offset=0):
        ''' Add a vector to a given column with optional row offset. '''
        for i, x in enumerate(v):
            self.addVal(i + row_offset, j, x)

    def emptyClone(self):
        ''' Return empty matrix of identical shape. '''
        return Matrix(self.nRow, self.nCol)

    def fullClone(self):
        ''' Return full matrix clone. '''
        mout = self.emptyClone()
        for i in range(self.nRow):
            mout.setRow(i, self.getRow(i))
        return mout

    def copyTo(self, mout, i, j):
        ''' Copy matrix to appropriate location in another matrix. '''
        self.bcopy(0, 0, self.nRow, self.nCol, i, j, mout)

    def sub(self, m):
        ''' Subtract another matrix to the current matrix. '''
        self.add(m.c().muls(-1))

    def onFullUpdate(self):
        ''' Method called manually to execute operations after a full matrix update. '''
        pass


class SquareMatrix(Matrix):
    ''' Interface to a square matrix object. '''

    def __new__(cls, n):
        ''' Instanciation. '''
        return super(SquareMatrix, cls).__new__(cls, n, n)

    def emptyClone(self):
        ''' Return empty matrix of identical shape. '''
        return SquareMatrix(self.nRow)


class DiagonalMatrix(SquareMatrix):
    ''' Interface to a diagonal matrix. '''

    def __new__(cls, x):
        ''' Instanciation. '''
        return super(DiagonalMatrix, cls).__new__(cls, x.size)

    def __init__(self, x):
        ''' Initialization.

            :param x: vector used to fill the diagonal.
        '''
        self.setdiag(0, h.Vector(x))



class PointerVector(hclass(h.Vector)):
    ''' Interface to a pointing vector, i.e. a vector that can "point" toward
        subsets of other vectors.
    '''

    def __new__(cls, *args, **kwargs):
        ''' Instanciation. '''
        return super(PointerVector, cls).__new__(cls)

    def __init__(self, size):
        ''' Initialization. '''
        self.refs = []
        super().__init__()
        self.resize(size)
    
    def __repr__(self):
        vals = npformat(np.array(self.to_python()))
        return f'{self.__class__.__name__}({len(self.refs)} ref{"s" if len(self.refs) > 1 else ""}, x={vals})'

    def setVal(self, i, x):
        ''' Item setter, adding the difference between the old an new value to all
            reference vectors with their respective offsets.
        '''
        xdiff = x - self.x[i]
        self.x[i] = x
        for v, offset in self.refs:
            v.x[i + offset] = v.x[i + offset] + xdiff

    def addVal(self, i, x):
        self.setVal(i, self.x[i] + x)

    def addTo(self, v, offset, fac=1):
        ''' Add the vector to a destination vector with a specific offset and factor. '''
        for i, x in enumerate(self):
            v.x[i] = v.x[i] + x * fac

    def addRef(self, v, offset):
        ''' Add a reference vector to "point" towards. '''
        assert self.size() + offset <= v.size(), 'exceeds reference dimensions'
        self.addTo(v, offset)
        self.refs.append((v, offset))

    def removeRef(self, iref):
        ''' Remove a reference vector from the list. '''
        self.addTo(*self.refs[iref], fac=-1)
        del self.refs[iref]


def pointerMatrix(MatrixBase):
    ''' Interface to a pointing matrix, i.e. a matrix that can "point" toward
        subsets of other matrices.
    '''

    class PointerMatrix(MatrixBase):

        def __init__(self, *args, **kwargs):
            ''' Initialization. '''
            self.refs = []
            super().__init__(*args, **kwargs)

        @property
        def nrefs(self):
            return len(self.refs)

        def addRef(self, mref, row_offset, col_offset):
            ''' Add a reference matrix to "point" towards. '''
            assert self.nRow + row_offset <= mref.nRow, 'exceeds reference dimensions'
            assert self.nCol + col_offset <= mref.nCol, 'exceeds reference dimensions'
            self.refs.append((mref, mref.emptyClone(), row_offset, col_offset))
            self.updateRef(self.nrefs - 1)

        def removeRef(self, iref):
            ''' Remove a reference matrix. '''
            mref, mholder, *_ = self.refs[iref]
            mref.sub(mholder)  # remove old values from ref matrix
            del self.refs[iref]

        def updateRef(self, iref):
            ''' Update a reference matrix. '''
            mref, mholder, row_offset, col_offset = self.refs[iref]
            mref.sub(mholder)                             # remove old values from ref matrix
            self.copyTo(mholder, row_offset, col_offset)  # update holder matrix
            mref.add(mholder)                             # add new values to ref matrix

        def updateRefs(self):
            ''' Update all reference matrices. '''
            for i in range(self.nrefs):
                self.updateRef(i)

        def onFullUpdate(self):
            self.updateRefs()

    PointerMatrix.__name__ = f'Pointer{MatrixBase.__name__}'

    return PointerMatrix


class IClamp(hclass(h.IClamp)):
    ''' IClamp object that takes section with default relative position as input
        and allows setting parameters on creation.
    '''

    def __new__(cls, section, *args, x=0.5):
        ''' Instanciation. '''
        return super(IClamp, cls).__new__(cls, section(x))

    def __init__(self, section, amplitude, x=0.5):
        super().__init__(section)
        self.delay = 0  # we want to exert control over amp starting at 0 ms
        self.dur = 1e9  # dur must be long enough to span all our changes
        self.amp = 0.  # initially, we set the amplitude to zero
        self.xamp = amplitude
        self.secref = h.SectionRef(sec=section)

    def set(self, value):
        self.amp = value * self.xamp
        self.secref.sec.setIstim(self.amp)


class ExtField():
    ''' Extracellular field object that allows setting parameters on creation. '''

    def __init__(self, section, amplitude, **kwargs):
        self.section = section
        if not self.section.has_ext_mech:
            self.section.insertVext(**kwargs)
        self.xamp = amplitude

    def __repr__(self):
        return f'{self.__class__.__name__}({self.section.shortname()}, {self.xamp:.2f} mV)'

    def set(self, value):
        self.section.setVext(self.xamp * value)


class Section(nrn.Section):
    ''' Interface to a Hoc Section with nseg=1. '''

    stimon_var = 'stimon'

    def __init__(self, name=None, cell=None, Cm0=1.):
        ''' Initialization.

            :param name: section name
            :param cell: section cell
            :param Cm0: resting membrane capacitance (uF/cm2)
        '''
        if name is None:
            raise ValueError('section must have a name')
        if cell is not None:
            super().__init__(name=name, cell=cell)
        else:
            super().__init__(name=name)
        self.passive_mechname = cell.passive_mechname
        self.Cm0 = Cm0
        self.nseg = 1
        self.has_ext_mech = False

    @property
    def cfac(self):
        raise NotImplementedError

    def setIstim(self, I):
        pass

    def shortname(self):
        s = self.name()
        if '.' in s:
            s = s.split('.')[-1]
        return s

    def setGeometry(self, diam, L):
        ''' Set section diameter and length.

            :param diam: section diameter (m)
            :param L: section length (m)
        '''
        self.diam = diam * M_TO_UM  # um
        self.L = L * M_TO_UM        # um

    def setResistivity(self, value):
        ''' Set section resistivity.

            Applies cfac-correction to Q-based sections, such that
            Iax = dV / r = dQ / (r * cm)

            :param value: longitudinal resistivity (Ohm.cm)
        '''
        self.Ra = value * self.cfac

    @property
    def Am(self):
        ''' membrane surface area (cm2). '''
        return np.pi * self.diam * self.L / CM_TO_UM**2

    @property
    def Ax(self):
        ''' axial area (cm2). '''
        return 1 / 4 * np.pi * self.diam**2 / CM_TO_UM**2

    @property
    def Ga(self):
        ''' axial conductance (S) with optional bounding to ensure (conductance / membrane area)
            stays below a specific threshold and limit the magnitude of axial currents.

            :return: axial conductance value (S)
        '''
        Ga = self.Ax / (self.Ra * self.L / CM_TO_UM)
        if hasattr(self, 'gmax'):
            Am = self.Am      # cm2
            Gmax = self.gmax * Am  # S
            s = f'{self}: Ga / Am = {Ga / Am:.1e} S/cm2'
            if Ga > Gmax:
                s = f'{s} -> bounded to {self.gmax:.1e} S/cm2'
                Ga = Gmax
                logger.warning(s)
            else:
                s = f'{s} -> not bounded'
                logger.debug(s)
        return Ga

    def target(self, x):
        ''' Return x-dependent resolved object (section's self of section's only segment).

            :param x: optional relative position over the section's length (0 <= x <= 1)
            :return: section's self (if x not provided) or of section's only segment (if x provided)
        '''
        return self(x) if x is not None else self

    def setValue(self, key, value, x=None):
        ''' Set the value of a section's (or segment's) attribute.

            :param key: attribute name
            :param value: attribute value
            :param x: relative position over the section's length
        '''
        setattr(self.target(x), key, value)

    def getValue(self, key, x=None):
        ''' Get the value of a section's (or segment's) attribute.

            :param key: attribute name
            :param x: relative position over the section's length
            :return: attribute value
        '''
        return getattr(self.target(x), key)

    def setProbe(self, var, loc=0.5, **kwargs):
        ''' Set recording vector for a range variable in a specific section location.

            :param var: range variable to record
            :return: recording probe object
        '''
        return Probe(self.getValue(f'_ref_{var}', x=loc), **kwargs)

    @property
    def rx(self):
        return self.xraxial[0]

    @property
    def gx(self):
        return self.xg[0]

    @property
    def cx(self):
        return self.xc[0]

    def insertVext(self, xr=None, xg=None, xc=None):
        ''' Insert extracellular mechanism with specific parameters.

            Applies cfac-correction to Q-based sections.

            :param xr: axial resistance per unit length of first extracellular layer (MOhm/cm)
            :param xg: transverse conductance of first extracellular layer (S/cm2)
            :param xc: transverse capacitance of first extracellular layer (uF/cm2)
        '''
        self.insert('extracellular')
        self.xraxial[0] = xr if xr is not None else XR_DEFAULT  # MOhm/cm
        self.xg[0] = xg if xg is not None else XG_DEFAULT       # S/cm2
        self.xc[0] = xc if xc is not None else XC_DEFAULT       # S/cm2
        for i in range(2):
            self.xraxial[i] *= self.cfac
            self.xc[i] /= self.cfac
            self.xg[i] /= self.cfac
        self.has_ext_mech = True

    def setVext(self, Ve):
        ''' Set the extracellular potential just outside of a section. '''
        self.e_extracellular = Ve * self.cfac

    def getVextRef(self):
        ''' Get reference to section's extracellular voltage. '''
        return self(0.5)._ref_vext[0]

    def setVextProbe(self):
        ''' Set a probe for the section's extracellular voltage. '''
        return Probe(self.getVextRef(), factor=1 / self.cfac)

    def connect(self, parent):
        ''' Connect proximal end to the distal end of a parent section.

            :param parent: parent sectio object
        '''
        super().connect(parent, 1, 0)

    def insertPassive(self):
        ''' Insert passive (leakage) mechanism. '''
        self.insert(self.passive_mechname)
        if self.passive_mechname == CUSTOM_PASSIVE_MECHNAME:
            self.mechname = self.passive_mechname

    @property
    def mechname(self):
        if not hasattr(self, '_mechname'):
            raise AttributeError(f'{self} does not have an attached mechanism')
        return self._mechname

    @mechname.setter
    def mechname(self, value):
        self._mechname = value

    def hasPassive(self):
        ''' Determine if section contains a passive membrane mechanism. '''
        return h.ismembrane(self.passive_mechname, sec=self)

    def setPassive(self, key, value):
        ''' Set an attribute of the passive (leakage) mechanism. '''
        setattr(self, f'{key}_{self.passive_mechname}', value)

    def setPassiveG(self, value):
        ''' Set the passive conductance (S/cm2). '''
        if self.passive_mechname != CUSTOM_PASSIVE_MECHNAME:
            value /= self.cfac
        self.setPassive('g', value)

    def setPassiveE(self, value):
        ''' Set the passive reversal potential (mV). '''
        if self.passive_mechname != CUSTOM_PASSIVE_MECHNAME:
            value *= self.cfac
        self.setPassive('e', value)

    def getDetails(self):
        ''' Get details of section parameters. '''
        d = {
            'nseg': self.nseg,
            'diam (um)': self.diam,
            'L (um)': self.L,
            'cm (uF/cm2)': self.cm,
            'Ra (Ohm.cm)': self.Ra
        }
        if self.has_ext_mech:
            d.update({
                'rx (MOhms)': self.rx,
                'gx (S/cm2)': self.gx,
                'cx (uF/cm2)': self.cx
            })
        return d

    def setVmProbe(self):
        raise NotImplementedError

    def setQmProbe(self):
        raise NotImplementedError

    def setProbes(self):
        ''' Set recording vectors for all variables of interest in the section.

            :return: probes object dictionary
        '''
        d = {
            'Qm': self.setQmProbe(),
            'Vm': self.setVmProbe()}
        if self.has_ext_mech:
            d['Vext'] = self.setVextProbe()
        return d

    def setMechValue(self, key, value, **kwargs):
        ''' Set the value of the section's mechanism attribute.

            :param key: attribute name
            :param value: attribute value
        '''
        self.setValue(f'{key}_{self.mechname}', value, **kwargs)

    def getMechValue(self, key, **kwargs):
        ''' Get the value of an attribute related to the section's mechanism.

            :param key: attribute name
            :return: attribute value
        '''
        return self.getValue(f'{key}_{self.mechname}', **kwargs)

    def setStimON(self, value):
        ''' Set stimulation ON or OFF.

            :param value: new stimulation state (0 = OFF, 1 = ON)
        '''
        try:
            self.setMechValue(self.stimon_var, value)
        except AttributeError as err:
            if self.hasPassive() and self.passive_mechname == CLASSIC_PASSIVE_MECHNAME:
                pass
            else:
                raise err

    def setMechProbe(self, var, loc=0.5, **kwargs):
        ''' Set recording vector for a mechanism specific variable. '''
        return self.setProbe(f'{var}_{self.mechname}', loc=loc, **kwargs)

    def setStimProbe(self):
        ''' Set recording vector for stimulation state. '''
        return self.setMechProbe(self.stimon_var)


class VSection(Section):
    ''' Interface to a Hoc Section with voltage based transmembrane dynamics. '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cm = self.Cm0

    @property
    def cfac(self):
        return 1.

    def setVmProbe(self):
        return self.setProbe('v')

    def setQmProbe(self):
        return self.setProbe('v', factor=self.Cm0 / C_M2_TO_NC_CM2)


class QSection(Section):
    ''' Interface to a Hoc Section with charge-density based transmembrane dynamics.

        In this section type, the differential variable "v" stands for charge density (in nC/cm2).
    '''

    @property
    def cfac(self):
        return self.Cm0

    def connect(self, parent):
        ''' Connect two sections together, provided they have the same membrane capacitance. '''
        if self.Cm0 != parent.Cm0:
            raise ValueError('Cannot connect Q-based sections with differing capacitance')
        super().connect(parent)

    def setVmProbe(self):
        return self.setProbe('v', factor=1 / self.Cm0)

    def setQmProbe(self):
        return self.setProbe('v', factor=1 / C_M2_TO_NC_CM2)


class MechSection(Section):
    ''' Interface to a section with associated point-neuron mechanism. '''

    # Aliases for NMODL-protected variable names
    NEURON_aliases = {'O': 'O1', 'C': 'C1'}

    def __init__(self, mechname, states=None, **kwargs):
        ''' Initialization.

            :param mechname: mechanism name
            :param states: list of mechanism time-varying states
        '''
        if kwargs:
            super().__init__(**kwargs)
        self.mechname = mechname
        self.states = states

    @property
    def mechname(self):
        return self._mechname

    @mechname.setter
    def mechname(self, value):
        if value == 'pas':
            value = self.passive_mechname
        self.insert(value)
        self._mechname = value

    @property
    def states(self):
        return self._states

    @states.setter
    def states(self, value):
        if value is None:
            value = []
        self._states = value

    def alias(self, state):
        ''' Return NEURON state alias.

            :param state: state name
        '''
        return self.NEURON_aliases.get(state, state)

    def setProbes(self):
        return {
            **super().setProbes(),
            **{k: self.setMechProbe(self.alias(k)) for k in self.states}
        }


class MechVSection(MechSection, VSection):
    ''' Interface to a V-based Section with associated point-neuron mechanism. '''

    def __init__(self, **kwargs):
        name = kwargs.pop('name', None)
        cell = kwargs.pop('cell', None)
        Cm0 = kwargs.pop('Cm0', 1.)
        VSection.__init__(self, name=name, cell=cell, Cm0=Cm0)
        MechSection.__init__(self, **kwargs)

    def setQmProbe(self):
        return self.setMechProbe('Qm', factor=1 / C_M2_TO_NC_CM2)


class MechQSection(MechSection, QSection):
    ''' Interface to a Q-based Section with associated point-neuron mechanism. '''

    def __init__(self, **kwargs):
        name = kwargs.pop('name', None)
        cell = kwargs.pop('cell', None)
        Cm0 = kwargs.pop('Cm0', 1.)
        QSection.__init__(self, name=name, cell=cell, Cm0=Cm0)
        MechSection.__init__(self, **kwargs)

    def setVmProbe(self):
        return self.setMechProbe('Vm')


def getCustomConnectSection(section_class):

    class CustomConnectSection(section_class):

        def __init__(self, *args, **kwargs):
            ''' Initialization. '''
            super().__init__(*args, **kwargs)
            try:
                key = self.mechname
            except AttributeError:
                key = self.passive_mechname
            self.vref = f'Vm_{key}'
            self.ex = 0.       # mV
            self.ex_last = 0.  # mV

        @property
        def cfac(self):
            return 1.

        def getVm(self, **kwargs):
            ''' Get the value of the section's reference voltage variable used to compute
                intracellular currents.

                :param return: reference variable value
            '''
            return self.getValue(self.vref, **kwargs)

        def getVmRef(self, **kwargs):
            ''' Get the reference to the section's reference voltage variable used to compute
                intracellular currents.

                :param return: reference variable value
            '''
            return self.getValue(f'_ref_{self.vref}', **kwargs)

        def getCm(self, **kwargs):
            return self.getValue('v', **kwargs) / self.getVm(**kwargs)

        def connect(self, parent):
            self.cell().registerConnection(parent, self)

        def getVextRef(self):
            ''' Get reference to section's extracellular voltage. '''
            return self.cell().getVextRef(self)

        @property
        def rx(self):
            return self._rx

        @rx.setter
        def rx(self, value):
            self._rx = isWithin('rx', value, XR_BOUNDS) * self.cfac

        @property
        def gx(self):
            return self._gx

        @gx.setter
        def gx(self, value):
            ''' Add NEURON's default xg in series to mimick 2nd extracellular layer. '''
            self._gx = seriesGeq(isWithin('gx', value, XG_BOUNDS), XG_DEFAULT) / self.cfac

        @property
        def cx(self):
            return self._cx

        @cx.setter
        def cx(self, value):
            self._cx = isWithin('cx', value, XC_BOUNDS) / self.cfac

        @property
        def Gp(self):
            ''' Section extracellular axial conductance (S). '''
            return 1 / (self.rx / OHM_TO_MOHM * self.L / CM_TO_UM)

        def insertVext(self, xr=None, xg=None, xc=None):
            self.rx = xr if xr is not None else XR_DEFAULT  # MOhm/cm
            self.gx = xg if xg is not None else XG_DEFAULT  # S/cm2
            self.cx = xc if xc is not None else XC_DEFAULT  # S/cm2
            self.has_ext_mech = True

        def setIstim(self, I):
            ''' Set the stimulation current density (in mA/cm2) corresponding
                to a stimulation current clamp (in nA) '''
            self.cell().setIstim(self, I)

        def setVext(self, Ve):
            ''' Set the extracellular potential just outside of a section. '''
            new_ex = Ve * self.cfac  # mV
            self.cell().setEx(self, self.ex, new_ex)
            self.ex = new_ex

    # Correct class name for consistency with input class
    CustomConnectSection.__name__ = f'CustomConnect{section_class.__name__}'

    return CustomConnectSection


def getAllSecs():
    return list(h.allsec())
