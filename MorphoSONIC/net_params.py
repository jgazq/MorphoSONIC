# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2023-03-22 17:14:29
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-03-22 17:47:05

''' Synaptic models and parameters for cortical network, as defined in Vierling-Classen et al. 2010 '''

from .core import Exp2Synapse, FExp2Synapse, FDExp2Synapse

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

# FS projections: GABA-A mechanism
FS_syn = Exp2Synapse(**GABA_params,
    tau2=8.0,  # decay time constant (ms)
)

# LTS projections: GABA-B mechanism
LTS_syn = Exp2Synapse(**GABA_params,
    tau2=50.0,  # decay time constant (ms)
)

# Cortical synaptic connections (with normalized weights as in Plaksin 2016)
cortical_connections = {
    'RS': {
        'RS': (0.002, RS_RS_syn),  # (synaptic conductance (uS), synaptic model)
        'FS': (0.04, RS_FS_syn),
        'LTS': (0.09, RS_LTS_syn)
    },
    'FS': {
        'RS': (0.015, FS_syn),
        'FS': (0.135, FS_syn),
        'LTS': (0.86, FS_syn)
    },
    'LTS': {
        'RS': (0.135, LTS_syn),
        'FS': (0.02, LTS_syn)
    }
}

# Thalamic drive to cortical cells (as in Plaksin 2016)
I_Th_RS = 0.17  # nA
thalamic_drives = {
    'RS': I_Th_RS,
    'FS': 1.4 * I_Th_RS,
    'LTS': 0.0
}