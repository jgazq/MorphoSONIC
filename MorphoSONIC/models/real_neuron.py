# h.load_file("axons/SENN_final.hoc")
# h.load_file("stdrun.hoc")

import sys
from neuron import h, gui
import re

#from ..core import SpatiallyExtendedNeuronModel, addSonicFeatures
from PySONIC.neurons import getPointNeuron
from MorphoSONIC.core import SpatiallyExtendedNeuronModel, addSonicFeatures #, FiberNeuronModel


class nrn(SpatiallyExtendedNeuronModel):
    """ Neuron class with methods for E-field stimulation """

    _pneuron = getPointNeuron('realneuron')

    #TT -> this function gives problems without errors, when assigning or calling variables in SonicMorpho __init__, code "crashes"
    # def __getattr__(self, attr):
    #     # Pass neuron hoc attributes to the Python nrn class
    #     if hasattr(self.cell, attr):
    #         return getattr(self.cell, attr)
    #     else:
    #         raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")
        
    #TT   
    def __dell__(self):
        # Delete all sections at gc
        for sec in self.all:
            h.delete_section(sec=sec)

    #TT
    def __repr__(self):
        return repr(self.cell)
    
    #TT
    @property
    def node(self):
        return [self.cell.node[i] for i in range(self.numberNodes)]
    
    #TT
    @property
    def internode(self):
        return [self.cell.internode[i] for i in range(self.numberNodes-1)]   # number internodes = number nodes - 1
    
    #TT
    def inactivateTerminal(self):
        "Method of Aberra et al. (2020) to inactivate axon at the terminal"
        self.cell.node[0](0).diam = 1000
        self.cell.node[self.numberNodes-1](1).diam = 1000

    #TT
    def set_xtra_coordinates(self,track_group,track,axonCoords):
        # Set the segment coordinates in the xtra-mechanism for coupling with E-fields
        for icoord, sec in enumerate(self.cell.all):
            if h.ismembrane("xtra",sec=sec):
                sec.x_xtra, sec.y_xtra, sec.z_xtra = tuple(1e3*axonCoords[1][track_group][track][icoord][:])    # [mm]

    #TT
    def set_pt3d_coordinates(self,track_group,track,axonCoords):          
        # Set the 3D points for visualisation
        h.define_shape()                            # Create 3d points to set the diameter
        for icoord, sec in enumerate(self.cell.all):
            if h.ismembrane("xtra",sec=sec):
                    if sec.n3d() != 3:              # Assuming there are 3 points: begin, halfway and end point
                        raise NotImplementedError
                    for i_pt3d in range(sec.n3d()):
                        sec.pt3dchange(i_pt3d,*tuple(1e3*axonCoords[i_pt3d][track_group][track][icoord][:]),sec.diam)

    #TT
    def interp2hoc_Efield(self,track_group,track,Efield_int):
        # Interpolate the
        allsecList = h.List()    # Necessary, because a sectionList can not be subscripted
        for sec in self.cell.all:
            allsecList.append(h.SectionRef(sec=sec))   # Note, make a list of sectionRefs (a list of sections is not supported by HOC)
 
        # Set the electric field in the NEURON xtra-mechanism
        for icoord, sec in enumerate(self.cell.all):
            if h.ismembrane("xtra",sec=sec):
                sec.Ex_xtra, sec.Ey_xtra, sec.Ez_xtra = (Efield_int[track_group][track][icoord,dim] for dim in range(3))
               
        # Use hoc functions to obtain the pseudo-potentials
        h.calc_pseudo_es(allsecList)  

    #TT
    def setcoords_interp(self,track_group,track,axonCoords,Efield_int):
        # Sets all coordinates (pt3d and xtra), interpolates the electric field and calculates the pseudopotentials
        self.set_xtra_coordinates(track_group,track,axonCoords)
        self.set_pt3d_coordinates(track_group,track,axonCoords)
        self.interp2hoc_Efield(track_group,track,Efield_int)

    #print all attributes of a class instance, both the ones defined in this file as the ones defined in the template.hoc file
    def print_attr(self):
        print(f"\n{'-'*75} python attributes {'-'*75}\n")
        print(*dir(self),sep=',\t')
        print(f"\n{'-'*75} hoc attributes {'-'*75}\n")
        print(*dir(self.cell),sep=',\t')
        print(f"\n{'-'*75}-----------------{'-'*75}")

    #case insensitive search for attributes 
    def search_attr(self,query):
        hits = []
        query = query.lower()
        for e in dir(self):
            if query in re.sub('[^a-zA-Z]+', '', e).lower():
                hits.append(e)
        for e in dir(self.cell):
            if query in re.sub('[^a-zA-Z]+', '', e).lower():
                hits.append(e)       
        print(f'\"{query}\" found in the following attributes: {hits}') if hits else print(f'{query} not found in attributes')
    
    #def clearSections, createSections, nonlinear_sections, refsection, seclist, simkey #NeuronModel

    #def getMetaArgs, meta #SENM -> added by addSonicFeatures decorator/wrapper

    #create sections by choosing the given cell and put the cell defined in hoc in the variable 'cell' of the class
    def createSections(self):
        print('creating sections in nrn')
        h.setParamsAdultHuman()
        h.cell_chooser(self.cell_nr); print('')
        #h("forall delete_section()") #to delete all sections in hoc -> __dell__ doesn't work because these sections are not assigned to the class (self)
        # new_dir = h.getcwd()+"cells/"+h.cell_names[cell_nr-1].s #to change the directory in the morphology file -> doesn't work
        # h(f"chdir({new_dir})") #change directory
        #self.cell = h.cADpyr229_L23_PC_8ef1aa6602(se) #class object based on defined template
        self.cell = h.cell
    
    def clearSections(self):
        self.__dell__()
        del self.cell #alternative: self.cell = None

    def nonlinear_sections(self):
        return {re.findall('[a-zA-Z]*\[[0-9]*\]',str(e))[-1] : e for e in self.cell.all} #or self.all
    
    def refsection(self):
        for e in self.cell.all:
            if 'soma' in str(e):
                return e
    
    def seclist(self):
        return list(self.nonlinear_sections().values())

    @property
    def meta(self):
        return {
            'simkey': self.simkey,
            'cell_nr': self.cell_nr,
            'se': self.synapses_enabled
        }

    @staticmethod
    def getMetaArgs(meta):
        return [meta['cell_nr'], meta['se']], {}


@addSonicFeatures
class Realnrn(nrn):
    """ Realistic Cortical Neuron class - python wrapper around the BBP_neuron hoc-template"""

    simkey = 'realistic_cort'

    def __init__(self,cell_nr,se,**kwargs):
        print(f'Realnrn init: {super()}')
        self.synapses_enabled = se
        self.cell_nr = cell_nr
        #h("strdef cell_name") #variable is defined to get assigned below
        h.load_file("init.hoc")
        #h(f"cell_name = \"{h.cell_names[cell_nr-1].s}\"") #this variable is unfortunately not recognized in the morphology.hoc file
        self.createSections()
        #self.inactivateTerminal() #TT
        #print(getattr(h, f'table_V_Ca_HVA')) #test if this attribute can be accessed -> OK

""""TESTING"""
#realnrn = Realnrn()#(cell_nr=7,se=0)      

#---search for a certain attribute
# realnrn.search_attr('bio')

#---print the attributes of the cell defined in the class and the one defined in hoc to compare
# print(dir(realnrn.cell))
# print(dir(h.cell))
#---print all the sections defined in hoc and all the sections assigned to the class, complemented with the number of segments
# for e in h.allsec():
#     print('\t',e,e.nseg)
# for e in realnrn.all:
#     print('\t',e,e.nseg)

# input()