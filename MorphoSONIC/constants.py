# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-04 18:26:42
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-06-22 09:44:58

# Conversions
V_TO_MV = 1e3            # V to  mV
PA_TO_KPA = 1e-3         # Pa -> kPa
HZ_TO_KHZ = 1e-3         # Hz -> kHz
C_M2_TO_NC_CM2 = 1e5     # C/m2 -> nC/cm2
S_M2_TO_S_CM2 = 1e-4     # S/m2 -> S/cm2
S_TO_MS = 1e3            # s -> ms
S_TO_US = 1e6            # s -> us
M_TO_CM = 1e2            # m -> cm
M_TO_MM = 1e3            # m -> mm
M_TO_UM = 1e6            # m to um
M_TO_NM = 1e9            # m to nm
CM_TO_UM = 1e4           # cm -> um
A_TO_NA = 1e9            # A -> nA
MA_TO_A = 1e-3           # mA -> A
MA_TO_NA = 1e6           # mA -> nA
F_M2_TO_UF_CM2 = 1e2     # F/m2 -> uF/cm2
UF_CM2_TO_MF_CM2 = 1e-3  # F/m2 -> uF/cm2
OHM_TO_MOHM = 1e-6       # Ohm-> MOhm

# Numerical integration
FIXED_DT = 1e-5                  # default time step for fixed integration (s)
MIN_NSAMPLES_PER_INTERVAL = 10   # minimum number of integration steps per interval
TRANSITION_DT = 1e-15            # time step for ON-OFF and OFF-ON sharp transitions (s)
PRINT_FINITIALIZE_STEPS = False  # flag stating whether to print the different finitialize steps

# Titration
IINJ_RANGE = (1e-12, 1e-7)  # intracellular current range allowed at the fiber level (A)
VEXT_RANGE = (1e0, 1e5)     # extracellular potential range allowed at the fiber level (mV)
REL_EPS_THR = 1e-2          # relative convergence threshold
REL_START_POINT = 0.2       # relative position of starting point within search range
REL_AP_TRAVEL_FACTOR = 1.5  # relative minimal duration of stimulus offset compared to estimated AP travel time

# Passive mechanism
CLASSIC_PASSIVE_MECHNAME = 'pas'
CUSTOM_PASSIVE_MECHNAME = 'custom_pas'

# Extracellular mechanism parameters
XR_DEFAULT = 1e9          # MOhm/cm
XG_DEFAULT = 1e9          # S/cm2
XC_DEFAULT = 0.           # uF/cm2
XR_BOUNDS = (1e-9, 1e15)  # MOhm/cm
XG_BOUNDS = (0., 1e15)    # S/cm2
XC_BOUNDS = (0., 1e15)    # uF/cm2

# NEURON matrix types
MFULL = 1
MSPARSE = 2
MBAND = 3

THR_VM_DIV = 1.0  # mV

# Model geometry
MIN_FIBERL_FWHM_RATIO = 2.0  # minimal fiber length compared to gaussian full-width at half-maximum

# Timeseries
NTRACES_MAX = 30  # max number of traces to be displayed simulatanously

#added by Joaquin
#mech_mapping to switch between the mechanism names with and without underscores in it
mech_mapping = {'Ca':'Ca', 'CaHVA':'Ca_HVA', 'CaLVAst':'Ca_LVAst', 'CaDynamicsE2':'CaDynamics_E2', 'Ih':'Ih', 'Im':'Im', 'KPst':'K_Pst', 'KTst':'K_Tst', 'KdShu2007':'KdShu2007', 'NapEt2':'Nap_Et2', 'NaTat':'NaTa_t', 'NaTs2t':'NaTs2_t', 'ProbAMPANMDAEMS':'ProbAMPANMDA_EMS', 'ProbGABAABEMS':'ProbGABAAB_EMS', 'SKE2':'SK_E2', 'SKv31':'SKv3_1', 'StochKv':'StochKv', 'xtra':'xtra', 'paseff':'pas_eff'}
mech_mapping_inv = {'Ca': 'Ca', 'Ca_HVA': 'CaHVA', 'Ca_LVAst': 'CaLVAst', 'CaDynamics_E2': 'CaDynamicsE2', 'Ih': 'Ih', 'Im': 'Im', 'K_Pst': 'KPst', 'K_Tst': 'KTst', 'KdShu2007': 'KdShu2007', 'Nap_Et2': 'NapEt2', 'NaTa_t': 'NaTat', 'NaTs2_t': 'NaTs2t', 'ProbAMPANMDA_EMS': 'ProbAMPANMDAEMS', 'ProbGABAAB_EMS': 'ProbGABAABEMS', 'SK_E2': 'SKE2', 'SKv3_1': 'SKv31', 'StochKv': 'StochKv', 'xtra': 'xtra', 'pas_eff': 'paseff'}

ABERRA = 1 #to enable code adaptations for an Aberra cell which has multiple mechanisms (mechanism files)
Cm0_var = 0 #to enable adaptations in the code when taking into account Cm0 variations along the sections #DEPRECATED
sep_12 = 0 #lookups for 0.01 and 0.02 are separated (done in parallel instead of in series) #DEPRECATED
Cm0_var2 = 1 #also considers Cm0 variations but different implementation: LUT and LUT2
SIMPLIFIED = 1 #this removes the probes of the gating parameters so they are not recorded during the simulation
OVERTONES = 0 #the number of overtones that need to be included
DEBUG_OV = 0 #debugging the code with overtone implementation
VERBATIM = 1 if OVERTONES else 0 #LUT are implemented using an interpolation method defined in the VERBATIM block #only allow verbatim to work if overtones are used

Cm0_map = {0.02: '0_02', 1.: '', 2.: '2'}
Cm0_actual = {1.: '', 2.: '2'}