# h.load_file("axons/SENN_final.hoc")
# h.load_file("stdrun.hoc")

import sys
from neuron import h#, gui
import re
import copy
import numpy as np
import time

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
            #print(f'Cm0_{sec}: {sec.cm}, Ra_{sec}: {sec.Ra}') #to print the membrane capacity and axial resistance
            for i,mech in enumerate(sec.psection()['density_mechs'].keys()): #iterate over the different inserted mechanisms of a particular section
                # if mech == relevant_mechs[-1]: #to check which section is used to probe?
                #     print(sec, i, mech)
                mech_ext = f"{mech}{Cm0_map[sec.cm]}"
                if mech == 'pas': #different treatment for the passive mechanism
                    suffix = f'pas_eff{Cm0_map[sec.cm]}'
                    "following 4 lines: -all in comment: both -last two in comment: only 0.02 -first two in comment or nothing in comment: only 0.01 (#end)"
                    # if not suffix.endswith('2'):
                    #     suffix+= '2'
                    # if suffix.endswith('2') and not suffix.endswith('02'):
                    #     suffix = suffix[:-1]
                    #end
                    sec.insert(suffix)
                    exec(f'sec.g_{suffix} = sec.g_pas')
                    exec(f'sec.e_{suffix} = sec.e_pas')
                    sec.uninsert('pas')
                    if suffix not in existing_mech:
                        existing_mech.append(suffix)
                elif mech_ext in distr_mech: #to insert the right mechanisms variant (0.01 or 0.02)
                    if self.increased_gNa: #to increase the conductivity of a certain mechanism, for debugging
                        if 'Na' in mech:
                            #print(f"before\t\t{mech}: {eval(f'sec.g{mech}bar_{mech}')}")
                            exec(f'sec.g{mech}bar_{mech} = 0') #this is actually decreasing but can be anything -> just for debugging
                            #exec(f'sec.g{mech}bar_{mech} = 10*sec.g{mech}bar_{mech}')
                            #print(f"after\t\t{mech}: {eval(f'sec.g{mech}bar_{mech}')}")
                            
                    "following 4 (un)insert lines: -all in comment: only 0.01 -nothing in comment: only 0.02 -last two in comment: both -first two in comment: all 0.01 become 0.02 and vice versa (#end)"
                    if sec.cm != 1:
                        pass
                        sec.insert(mech_ext)
                        exec(f'sec.g{mech}bar_{mech_ext} = sec.g{mech}bar_{mech} ')
                        sec.uninsert(mech)
                    elif 'xtra' not in mech and 'Dynamics' not in mech:
                        pass
                        #print(mech+'2')
                        # sec.uninsert(mech)
                        # sec.insert(mech+'2')    
                    #end     
   
                    "for only 0.01: also adapt setFuncTables in nmodel.py"            
                    if mech_ext not in existing_mech:
                        existing_mech.append(mech_ext) #a list containing all the distributed mechanisms and passive mechanisms (which is also a distributed one)
                else:
                    if mech_ext not in unexisting_mech:
                        unexisting_mech.append(mech_ext) #all the other mechanisms
            #sec.cm = 1
            sec.v = -75*sec.cm #to initialize the section starting value properly
            #print(sec.v)
            if self.decoupling: #and 'Node' not in str(sec): #('apic' not in str(sec) and 'dend' not in str(sec) and 'soma ' not in str(sec)): #to decouple all sections by putting the axial resistance very high so there is no axial currents to influence other sections
                sec.Ra = 1e20 # to decouple the different sections from each other
            # else:   
            #     print(f'connected: {sec}')
        for sec_soma in self.cell.soma: #redefine the voltage of the soma as this value adapts when changing v of other sections (which is done in this line: sec.v = -75*sec.cm)
            sec_soma.v = -75*sec_soma.cm

        existing_mech.sort()
        unexisting_mech.sort()
        #print(f'existing mechs: {existing_mech}')
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
        #print("self.mechname: ",self.mechname)

        "same code as in psection() but looks at all mechanisms that are loaded into hoc instead of a specific section"
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
        if VERBATIM:
            for sec in h.allsec():
                self.psection(sec)
        if Cm0_var or Cm0_var2:
            self.mech_Cm0(distr_mech) #BREAKPOINT
        else:
            for sec in h.allsec():
                for mech in sec.psection()['density_mechs'].keys():
                    if mech == 'pas':
                        sec.insert('pas_eff')
                        exec(f'sec.g_pas_eff = sec.g_pas')
                        exec(f'sec.e_pas_eff = sec.e_pas')
                        sec.uninsert('pas')
                if self.decoupling: #('apic' not in str(sec) and 'dend' not in str(sec)):
                    sec.Ra = 1e20
                # else:   
                #     print(f'connected: {sec}')
        if VERBATIM:
            for sec in h.allsec():
                self.psection(sec)

        "first create a dictionary for every type of compartment by creating a python section wrapper around the nrn section"

        somas = {eval(f"'soma{i}'"): self.createSection(eval(f"'soma[{i}]'"),mech=self.mechname,states=self.pneuron.statesNames(),nrnsec=e) for i,e in enumerate(self.cell.soma)}
        axons, nodes, myelins, unmyelins = {}, {}, {}, {}
        for e in self.cell.axonal:
            if 'axon' in str(e):
                axons[eval(f"'axon{len(axons.keys())}'")] = self.createSection(eval(f"'axon[{len(axons.keys())}]'"),mech=self.mechname,states=self.pneuron.statesNames(),nrnsec=e)
            elif 'Node' in str(e):
                nodes[eval(f"'node{len(nodes.keys())}'")] = self.createSection(eval(f"'node[{len(nodes.keys())}]'"),mech=self.mechname,states=self.pneuron.statesNames(),nrnsec=e)
            elif 'Myelin' in str(e):
                myelins[eval(f"'myelin{len(myelins.keys())}'")] = self.createSection(eval(f"'myelin[{len(myelins.keys())}]'"),mech=self.mechname,states=self.pneuron.statesNames(),nrnsec=e)
            elif 'Unmyelin' in str(e):
                unmyelins[eval(f"'unmyelin{len(unmyelins.keys())}'")] = self.createSection(eval(f"'unmyelin[{len(unmyelins.keys())}]'"),mech=self.mechname,states=self.pneuron.statesNames(),nrnsec=e)
            else:
                raise TypeError(f'Undefined section ecountered: {str(e)}')
        #axons = {eval(f"'axon{i}'"): self.createSection(eval(f"'axon[{i}]'"),mech=self.mechname,states=self.pneuron.statesNames(),nrnsec=e) for i,e in enumerate(self.cell.axonal)}
        apicals = {eval(f"'apical{i}'"): self.createSection(eval(f"'apical[{i}]'"),mech=self.mechname,states=self.pneuron.statesNames(),nrnsec=e) for i,e in enumerate(self.cell.apical)}
        basals = {eval(f"'basal{i}'"): self.createSection(eval(f"'basal[{i}]'"),mech=self.mechname,states=self.pneuron.statesNames(),nrnsec=e) for i,e in enumerate(self.cell.basal)}
        #nodes = {eval(f"'node{i}'"): self.createSection(eval(f"'node[{i}]'"),mech=self.mechname,states=self.pneuron.statesNames(),nrnsec=e) for i,e in enumerate(h.Node)}
        #myelins = {eval(f"'myelin{i}'"): self.createSection(eval(f"'myelin[{i}]'"),mech=self.mechname,states=self.pneuron.statesNames(),nrnsec=e) for i,e in enumerate(h.Myelin)}
        #unmyelins = {eval(f"'unmyelin{i}'"): self.createSection(eval(f"'unmyelin[{i}]'"),mech=self.mechname,states=self.pneuron.statesNames(),nrnsec=e) for i,e in enumerate(h.Unmyelin)}

        #self.sections: dicionary contain dictionaries for each type
        self.sections = {'soma': somas, 'axon': axons, 'apical': apicals, 'basal': basals, 'node': nodes, 'myelin': myelins, 'unmyelin': unmyelins} #no axon -> replaced with Node, Myelin and Unmyelin #
        #self.seclist: contains all python sections in a list
        self.seclist = [*list(somas.values()), *list(axons.values()), *list(apicals.values()), *list(basals.values()), *list(nodes.values()), *list(myelins.values()), *list(unmyelins.values())] #
        #self.nrnseclist: contains all (original) nrn (hoc) sections in a list 
        self.nrnseclist = [e.nrnsec for e in self.seclist]
        self.segments = []
        self.indexes = []
        iterator = 0
        for sec in self.nrnseclist:
            nseg = sec.nseg
            self.indexes.append([e for e in range(iterator,iterator+nseg)])
            iterator += nseg
            for seg in sec:
                self.segments.append(seg)
        #print("len(seclist): ",len(self.seclist)) #to check how many sections are defined
        #print('self.nrnseclist',self.nrnseclist) #to check if the creation has been conducted correctly

        self.connections = []

        for sec in self.seclist:
            #print('sec: ',sec)
            parent, children = sec.nrnsec, sec.nrnsec.children()
            for child in children:
                #print(parent, child); continue
                #if ('axon' in str(parent) or 'axon' in str(child)): #or ('Myelin' in str(parent) or 'Myelin' in str(child)): #there is an axon that is still a child of the soma even if they are replaced with nodes, myelin and unmyelin
                    #continue
                #print('parent: ',parent,'child',child)
                #disconnect in Morpho
                self.connections.append((self.nrnseclist.index(parent),self.nrnseclist.index(child))) #str(parent),str(child)
            "lines are moved to init of CustomConnectSection"
        self.connections_reversed = [] #this list contains both the (x,y)=(parent,child) connection as the (y,x)=(child,parent) connection
        for e in self.connections:
            self.connections_reversed.append((e[1],e[0])) # creating and adding the (y,x) connections
        self.connections_double = self.connections + self.connections_reversed #adding the original (x,y) connections
        
        self.connections.sort()
        self.connections_reversed.sort()
        self.connections_double.sort() #sort them so they are in order

        #disconnect in hoc
        for sec in h.allsec():
            h.disconnect(sec=sec)
        print(f'CELL IS CREATED: {len(self.seclist)} sections')
        self.loc_soma = (somas['soma0'].x_xtra, somas['soma0'].y_xtra, somas['soma0'].z_xtra)

        #print(f"self.connections: {self.connections}")
        #print(f'len(self.segments): {len(self.segments)}')
        

    
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
        return {k: np.array([e.nrnsec.y_xtra*1e-6 for e in l.values()]) for k,l in self.sections.items()} #m
    
    def getZCoords(self):
        return {k: np.array([e.nrnsec.z_xtra*1e-6 for e in l.values()]) for k,l in self.sections.items()} #m


@addSonicFeatures
class Realnrn(nrn):
    """ Realistic Cortical Neuron class - python wrapper around the BBP_neuron hoc-template"""

    simkey = 'realistic_cort'

    def __init__(self,cell_nr,se=0,**kwargs):
        #print(f'Realnrn init: {super()}')
        self.synapses_enabled = se
        self.cell_nr = cell_nr
        ''''DEBUG variables'''
        self.decoupling = 0
        self.increased_gNa = 0
        #h("strdef cell_name") #variable is defined to get assigned below
        h.load_file("init.hoc")
        #h(f"cell_name = \"{h.cell_names[cell_nr-1].s}\"") #this variable is unfortunately not recognized in the morphology.hoc file
        self.createSections()
        #self.inactivateTerminal() #TT
        #print(getattr(h, f'table_V_Ca_HVA')) #test if this attribute can be accessed -> OK
        self.folder = 'cells\\'+h.cell_names[cell_nr-1].s

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