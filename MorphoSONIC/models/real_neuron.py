# h.load_file("axons/SENN_final.hoc")
# h.load_file("stdrun.hoc")

import sys
from neuron import h#, gui
import re
import copy
import numpy as np

#from ..core import SpatiallyExtendedNeuronModel, addSonicFeatures
from PySONIC.neurons import getPointNeuron
from MorphoSONIC.core import SpatiallyExtendedNeuronModel, addSonicFeatures, MechQSection #, FiberNeuronModel
from ..constants import *


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

    def print_attr(self):
        """print all attributes of a class instance, both the ones defined in this file as the ones defined in the template.hoc file"""
        print(f"\n{'-'*75} python attributes {'-'*75}\n")
        print(*dir(self),sep=',\t')
        print(f"\n{'-'*75} hoc attributes {'-'*75}\n")
        print(*dir(self.cell),sep=',\t')
        print(f"\n{'-'*75}-----------------{'-'*75}")
 
    def search_attr(self,query):
        """case insensitive search for attributes"""
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
        
    def mech_Cm0(self, distr_mech):
        """replace the generic mechanism with the specific mechanism based on Cm0"""
        existing_mech = []
        unexisting_mech = []
        for sec in h.allsec():
            #print(sec.psection()['density_mechs'].keys()) #to print all sections with their respective insterted mechanisms
            #print(f'Cm0_{sec}: {sec.cm}, Ra_{sec}: {sec.Ra}')
            for mech in sec.psection()['density_mechs'].keys():
                mech_ext = f"{mech}{Cm0_map[sec.cm]}"
                if mech == 'pas':
                    suffix = f'pas_eff{Cm0_map[sec.cm]}'
                    # to only insert 0.02-variant mechs
                    if not suffix.endswith('2'):
                        suffix+= '2'
                    #to only insert 0.01-variant mechs
                    # if suffix.endswith('2') and not suffix.endswith('02'):
                    #     suffix = suffix[:-1]
                    sec.insert(suffix)
                    exec(f'sec.g_{suffix} = sec.g_pas')
                    exec(f'sec.e_{suffix} = sec.e_pas')
                    sec.uninsert('pas')
                    if suffix not in existing_mech:
                        existing_mech.append(suffix)
                elif mech_ext in distr_mech:
                    if sec.cm != 1:
                        #to only use 0.01-variants, only put 'pass' out of comments in next section
                        pass
                        sec.uninsert(mech)
                        sec.insert(mech_ext)
                    elif 'xtra' not in mech and 'Dynamics' not in mech:
                        #to only use 0.02-variants, put everything out of comments (pass doesn't matter)
                        pass
                        sec.uninsert(mech)
                        #print(mech+'2')
                        sec.insert(mech+'2')                        
                    if mech_ext not in existing_mech:
                        existing_mech.append(mech_ext)
                else:
                    if mech_ext not in unexisting_mech:
                        unexisting_mech.append(mech_ext)

            #sec.Ra = 1e20 # to decouple the different sections from each other
        existing_mech.sort()
        unexisting_mech.sort()
        print(f'existing mechs: {existing_mech}')
        print(f'unexisting mechs: {unexisting_mech}')

    def createSections(self):
        """create sections by choosing the given cell and put the cell defined in hoc in the variable 'cell' of the class"""
        print('creating sections in nrn')
        h.setParamsAdultHuman() 
        h.cell_chooser(self.cell_nr); print('')
        #h("forall delete_section()") #to delete all sections in hoc -> __dell__ doesn't work because these sections are not assigned to the class (self)
        # new_dir = h.getcwd()+"cells/"+h.cell_names[cell_nr-1].s #to change the directory in the morphology file -> doesn't work
        # h(f"chdir({new_dir})") #change directory
        #self.cell = h.cADpyr229_L23_PC_8ef1aa6602(se) #class object based on defined template
        self.cell = h.cell
        print("self.mechname: ",self.mechname)

        distr_mech, point_mech = [], []

        mt0 = h.MechanismType(0)
        mname  = h.ref('')
        for i in range(mt0.count()):
            mt0.select(i)
            mt0.selected(mname)
            distr_mech.append(mname[0])
            
        mt1 = h.MechanismType(1)
        for i in range(mt1.count()):
            mt1.select(i)
            mt1.selected(mname)
            point_mech.append(mname[0])

        #print(f'distributed mechs: {distr_mech}, point process mechs: {point_mech}')

        if Cm0_var:
            self.mech_Cm0(distr_mech)

        #first create a dictionary for every type of compartment by creating a python section wrapper around the nrn section

        somas = {eval(f"'soma{i}'"): self.createSection(eval(f"'soma[{i}]'"),mech=self.mechname,states=self.pneuron.statesNames(),nrnsec=e) for i,e in enumerate(self.cell.soma)}
        apicals = {eval(f"'apical{i}'"): self.createSection(eval(f"'apical[{i}]'"),mech=self.mechname,states=self.pneuron.statesNames(),nrnsec=e) for i,e in enumerate(self.cell.apical)}
        basals = {eval(f"'basal{i}'"): self.createSection(eval(f"'basal[{i}]'"),mech=self.mechname,states=self.pneuron.statesNames(),nrnsec=e) for i,e in enumerate(self.cell.basal)}
        nodes = {eval(f"'node{i}'"): self.createSection(eval(f"'node[{i}]'"),mech=self.mechname,states=self.pneuron.statesNames(),nrnsec=e) for i,e in enumerate(h.Node)}
        myelins = {eval(f"'myelin{i}'"): self.createSection(eval(f"'myelin[{i}]'"),mech=self.mechname,states=self.pneuron.statesNames(),nrnsec=e) for i,e in enumerate(h.Myelin)}
        unmyelins = {eval(f"'unmyelin{i}'"): self.createSection(eval(f"'unmyelin[{i}]'"),mech=self.mechname,states=self.pneuron.statesNames(),nrnsec=e) for i,e in enumerate(h.Unmyelin)}

        #self.sections: dicionary contain dictionaries for each type
        self.sections = {'soma': somas, 'apical': apicals, 'basal': basals, 'node': nodes, 'myelin': myelins, 'unmyelin': unmyelins} #no axon -> replaced with Node, Myelin and Unmyelin #
        #self.seclist: contains all python sections in a list
        self.seclist = [*list(somas.values()), *list(apicals.values()), *list(basals.values()), *list(nodes.values()), *list(myelins.values()), *list(unmyelins.values())] #
        #self.nrnseclist: contains all (original) nrn (hoc) sections in a list 
        self.nrnseclist = [e.nrnsec for e in self.seclist]
        #print("len(seclist): ",len(self.seclist)) #to check how many sections are defined
        #print('self.nrnseclist',self.nrnseclist) #to check if the creation has been conducted correctly

        self.connections = []

        for sec in self.seclist:
            #print('sec: ',sec)
            parent, children = sec.nrnsec, sec.nrnsec.children()
            for child in children:
                if ('axon' in str(parent) or 'axon' in str(child)): #or ('Myelin' in str(parent) or 'Myelin' in str(child)): #there is an axon that is still a child of the soma even if they are replaced with nodes, myelin and unmyelin
                    continue
                #print('parent: ',parent,'child',child)
                self.connections.append((self.nrnseclist.index(parent),self.nrnseclist.index(child)))
            "lines are moved to init of CustomConnectSection"
        #print(self.connections)
        print(f'CELL IS CREATED: {len(self.seclist)} sections')

    
    def clearSections(self):
        self.__dell__()
        del self.cell #alternative: self.cell = None

    def nonlinear_sections(self):
        return {re.findall('[a-zA-Z]*\[[0-9]*\]',str(e))[-1] : e for e in self.cell.all} #or self.all
    
    @property
    def refsection(self):
        return self.sections['soma']['soma0']
        #code below is not possible because of the init of new unnamed sections
        # for e in self.cell.all:
        #     if 'soma' in str(e):
        #         return e
    
    def seclist(self):
        return #self.sections #list(self.nonlinear_sections().values())

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

    # @property
    # def sections(self):
    #     #nrnsec = {'soma': {eval(f"'soma{i}'"): e for i,e in enumerate(self.cell.soma)}, 'apical': {eval(f"'apical{i}'"): e for i,e in enumerate(self.cell.apical)}, 'basal': {eval(f"'basal{i}'"): e for i,e in enumerate(self.cell.basal)}, 'node': {eval(f"'node{i}'"): e for i,e in enumerate(h.Node)}, 'myelin': {eval(f"'myelin{i}'"): e for i,e in enumerate(h.Myelin)}, 'unmyelin': {eval(f"'unmyelin{i}'"): e for i,e in enumerate(h.Unmyelin)}} #no axon -> replaced with Node, Myelin and Unmyelin
    #     return self.sections
    
    def getXCoords(self):
        # print(self.sections.items())
        # print(self.sections['soma']['soma0'].x_xtra)
        return {k: np.array([e.nrnsec.x_xtra*1e-6 for e in l.values()]) for k,l in self.sections.items()} #1e-6: um -> m
    
    def getYCoords(self):
        return {k: np.array([e.nrnsec.z_xtra*1e-6 for e in l.values()]) for k,l in self.sections.items()} #m
    
    def getZCoords(self):
        return {k: np.array([e.nrnsec.z_xtra*1e-6 for e in l.values()]) for k,l in self.sections.items()} #m


@addSonicFeatures
class Realnrn(nrn):
    """ Realistic Cortical Neuron class - python wrapper around the BBP_neuron hoc-template"""

    simkey = 'realistic_cort'

    def __init__(self,cell_nr,se=0,**kwargs):
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