# Description

`MorphoSONIC` is a Python+NEURON implementation of **spatially extended representations** of the **multi-Scale Optimized Neuronal Intramembrane Cavitation (SONIC) model [1]**. It enables the simulation of the distributed electrical response of morphologically realistic neuron representations to acoustic stimuli, as predicted by the *intramembrane cavitation* hypothesis. The details of the MorphoSONIC framework and of its application to study ultrasound neuromodulation in peripheral nerve fibers are described in [2].

This package expands features from the `PySONIC` package (https://github.com/tjjlemaire/PySONIC).

# Content of repository

## Models

The `models` module defines a variety of neuron models:

- the `Node` class that provides a *NEURON* wrapper around the point-neuron models defined in the `PySONIC` package, and can be simulated with both punctual electrical and acoustic drives.
- the `RadialModel` class defines a **nanoscale radially-symmetric model** with central and peripheral compartments. It can be used to model the coupling between an "ultrasound-responsive" sonophore and an "ultrasound-resistant" surrounding membrane (see `surroundedSonophore` function). As this model is radially symmetric, some adaptation was needed in order to represent it within the *NEURON* environment (check [this link](docs/NEURON_radial_geometry.md) for more details).

The module also contains morphologically-structured models of **unmyelinated and myelinated peripheral fibers**:
- `SennFiber` implements a spatially-extended nonlinear node (SENN) myelinated fiber model, as defined in Reilly 1985.
- `SweeneyFiber` implements the SENN model variant defined in Sweeney 1987.
- `MRGFiber` implements the double-cable myelinated fiber model defined as in McIntyre 2002.
- `UnmyelinatedFiber` implements an unmyelinated fiber model defined as in Sundt 2015.

## `@addSonicFeatures` decorator

One of the main advantages of the *NEURON* simulation environment is the presence of optimized numerical integration pipelines that are completely abstracted from the model definition. However, this abstraction imposes a default cable representation for models implemented in *NEURON*, one that assumes (1) a voltage-casted electrical system and (2) a constant membrane capacitance throughout simulations. Unfortunately, both these assumptions are incompatible with the underlying equations of the SONIC model. 

Therefore, to enable simulations of models incorporating the SONIC paradigm in *NEURON*, we defined an alternative cable representation that is intrinsically compatible with the SONIC model and can be substituted to *NEURON*'s default representation upon model construction (see [2] for more details). In practice, this substitution is achieved by assigning a simple decorator (`@addSonicFeatures`) to the model class, such that one can easily switch between the default and SONIC-compatible implementations. Importantly, the SONIC-compatible version can also be used for model simulations with "conventional" electrical stimuli.

## Custom membrane mechanisms

`MorphoSONIC` leverages *NEURON*'s architecture to define a range of membrane mechanisms that can be independently assigned to any morphological section. These mechanisms are each implemented in a specific NMODL (`.mod`) file, stored in the `nmodl` subfolder. Each mechanism defines a set of parameters and equations governing the membrane dynamics of the sections it is attached to. 

These equations often involve the use of voltage-dependent rate constants that are typically defined by analytical functions in the `mod` file. To enable compatibility with the SONIC model, these analytical functions have been substituted by *function tables*, i.e. 2-dimensional tables dynamically populated at runtime, either from SONIC lookup tables or by simple interpolation of the original analyical functions of the model. 

Most point-neuron models defined in the `PySONIC` package have been translated to equivalent `mod` files so they can be readily used here. If you have implemented additional point-neuron models and wish to translate them into MOD files, use the `generate_mod_file.py` script.

## Sources

The `sources` module defines a variety of analytical models of electrical and acoustic exposure distributions that can be used to stimulate spatially-extended models:

- `IntracellularCurrent` for local intracellular current injection at a specific section.
- `ExtracellularCurrent` for distributed voltage perturbation resulting from current injection at a distant point-source electrode.
- `GaussianVoltageSource` for a distributed voltage perturbation defined by a Gaussian distribution
- `SectionAcousticSource` for local acoustic perturbation at a specific section.
- `PlanarDiskTransducerSource` for distributed acoustic perturbation resulting from sonication by a distant planar acoustic transducer.
- `GaussianAcousticSource` for a distributed acoustic perturbation defined by a Gaussian distribution


## Other modules

- `pyhoc`: defines utilities for Python-NEURON communication
- `pymodl`: defines a parser to translate point-neuron models defined in Python into NMODL membrane mechanisms
- `parsers`: command line parsing utilities
- `plt`: graphing utilities
- `constants`: algorithmic constants
- `utils`: generic utilities

# Requirements

- Python 3.6+
- NEURON 8+
- `PySONIC` package (https://github.com/tjjlemaire/PySONIC)

# Installation

### NEURON

If you are using a **Windows computer**, you'll first need to **install NEURON manually**:
1. Go to the [NEURON website](https://neuron.yale.edu/neuron/download/) and download the appropriate *NEURON* installer for Windows
2. Start the *NEURON* installer, confirm the installation directory (`c:\nrn`), check the "Set DOS environment" option, and click on "Install". After completion you should see a new folder named `NEURON 8.x x86_64` on your Desktop.
3. Log out and back in to make sure your environment variables are updated.

For Mac OSx / Linux users, NEURON will be automatically installed as a Python dependency, so you don't need to pre-install anything. 

### MorphoSONIC package

- Open a terminal
- Clone the `PySONIC` repository and install it as a python package in a separate conda environment called `sonic`. The detailed installation instructions are on the [PySONIC](https://github.com/tjjlemaire/PySONIC) webpage.
- If not already done, activate the `sonic` anaconda environment: `conda activate sonic`
- Clone the `MorphoSONIC` repository and install it as a python package:

```
git clone https://github.com/tjjlemaire/MorphoSONIC.git
cd MorphoSONIC
pip install -e .
```

All package dependencies (numpy, scipy, ...) should be installed automatically.

### Pre-compilation of NEURON membrane mechanisms

In order to use the package, you will need to compile a specific set of equations describing the membrane dynamics of the different neuron types.

For **Windows users**:
- In the folder named `NEURON 8.x x86_64` on your Desktop, run the `mknrndll` executable.
- In the displayed window, select the directory containing the source files for the membrane mechanisms: `.../MorphoSONIC/MorphoSONIC/nmodl/`
- Click on `make nrnmech.dll`
- Upon completion, hit enter in the terminal to close it.

For **Mac OSx / Linux users**:
- Open a terminal window
- Move to the directory containing the source files for the membrane mechanisms, and run the `nrnivmodl` executable:

```
cd <path_to_MorphoSONIC_package>/MorphoSONIC/nmodl/
nrnivmodl
```

# Usage

## Python scripts

You can easily run simulations of any implemented point-neuron model under both electrical and ultrasonic stimuli, and visualize the simulation results, in just a few lines of code:

```python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-03-31 13:56:36
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-07-27 17:45:54

import logging
import matplotlib.pyplot as plt

from PySONIC.core import PulsedProtocol, ElectricDrive, AcousticDrive
from PySONIC.neurons import getPointNeuron
from PySONIC.utils import logger
from PySONIC.plt import GroupedTimeSeries
from MorphoSONIC.models import Node

# Set logging level
logger.setLevel(logging.INFO)

# Define point-neuron model
pneuron = getPointNeuron('RS')

# Define sonophore parameters
a = 32e-9  # sonophore radius (m)
fs = 1.    # sonophore coverage fraction (-)

# Create node model
node = Node(pneuron, a=a, fs=fs)

# Define electric and ultrasonic drives
EL_drive = ElectricDrive(20.)  # mA/m2
US_drive = AcousticDrive(
    500e3,  # Hz
    100e3)  # Pa

# Set pulsing protocol
tstim = 250e-3   # s
toffset = 50e-3  # s
PRF = 100.       # Hz
DC = 0.5         # -
pp = PulsedProtocol(tstim, toffset, PRF, DC)

# Simulate model with each drive modality and plot results
for drive in [EL_drive, US_drive]:
    data, meta = node.simulate(drive, pp)
    GroupedTimeSeries([(data, meta)]).render()

# Show figures
plt.show()

```

Similarly, you can run simulations of myelinated and unmyelinated fiber models under extracellular electrical and ultrasonic stimulation, and visualize the simulation results:

```python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-03-31 13:56:36
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-07-27 17:45:43

import logging
import matplotlib.pyplot as plt

from PySONIC.core import PulsedProtocol, BalancedPulsedProtocol
from PySONIC.utils import logger

from MorphoSONIC.models import SennFiber
from MorphoSONIC.sources import *
from MorphoSONIC.plt import SectionCompTimeSeries

# Set logging level
logger.setLevel(logging.INFO)

# Define sonophore parameters
a = 32e-9  # sonophore radius (m)
fs = 1.    # sonophore coverage fraction (-)

# Define fiber parameters
fiberD = 20e-6  # m
nnodes = 21

# Create fiber model
fiber = SennFiber(fiberD, nnodes, a=a, fs=fs)

# Define various sources
iintra_source = IntracellularCurrent(
    sec_id=fiber.central_ID,  # target section
    I=3.0e-9)                 # current amplitude (A)
iextra_source = ExtracellularCurrent(
    x=(0., fiber.interL),  # point-source electrode position (m)
    I=-0.70e-3)            # current amplitude (A)
voltage_source = GaussianVoltageSource(
    0,                   # gaussian center (m)
    fiber.length / 10.,  # gaussian width (m)
    Ve=-80.)             # peak extracellular voltage (mV)
section_US_source = SectionAcousticSource(
    fiber.central_ID,  # target section
    500e3,             # US frequency (Hz)
    A=100e3)           # peak acoustic amplitude (Pa)
gaussian_US_source = GaussianAcousticSource(
    0,                   # gaussian center (m)
    fiber.length / 10.,  # gaussian width (m)
    500e3,               # US frequency (Hz)
    A=100e3)             # peak acoustic amplitude (Pa)
transducer_source = PlanarDiskTransducerSource(
    (0., 0., 'focus'),  # transducer position (m)
    500e3,              # US frequency (Hz)
    r=2e-3,             # transducer radius (m)
    u=0.04)             # m/s

# Define pulsing protocols
tpulse = 100e-6  # s
xratio = 0.2     # (-)
toffset = 3e-3   # s
standard_pp = PulsedProtocol(tpulse, toffset)                  # (for US sources)
balanced_pp = BalancedPulsedProtocol(tpulse, xratio, toffset)  # (for electrical sources)

# Define source-protocol pairs
pairs = [
    (iintra_source, balanced_pp),
    (iextra_source, balanced_pp),
    (voltage_source, balanced_pp),
    (section_US_source, standard_pp),
    (gaussian_US_source, standard_pp),
    (transducer_source, standard_pp)
]

# Simulate model with each source-protocol pair, and plot results
for source, pp in pairs:
    data, meta = fiber.simulate(source, pp)
    SectionCompTimeSeries([(data, meta)], 'Vm', fiber.nodeIDs).render()

plt.show()

```

## From the command line

### Running simulations

You can easily run simulations of punctual and spatially-extended models using the dedicated command line scripts. To do so, open a terminal in the `scripts` directory.

- Use `run_node_estim.py` for simulations of **point-neuron models** upon **intracellular electrical stimulation**. For instance, a regular-spiking (RS) neuron injected with 10 mA/m2 intracellular current for 30 ms:

```python run_node_estim.py -n RS -A 10 --tstim 30 -p Vm```

- Use `run_node_astim.py` for simulations of **point-neuron models** upon **ultrasonic stimulation**. For instance, a 32 nm radius bilayer sonophore within a regular-spiking (RS) neuron membrane, sonicated at 500 kHz and 100 kPa for 150 ms:

```python run_node_astim.py -n RS -a 32 -f 500 -A 100 --tstim 150 --method sonic -p Qm```

- Use `run_fiber_iextra.py` for simulations of a **peripheral fiber models** (myelinated or unmyelinated) of any diameter and with any number of nodes upon **extracellular electrical stimulation**. For instance, to simulate a 20 um diameter, 11 nodes SENN-type myelinated fiber, stimulated at 0.6 mA for 0.1 ms by a cathodal point-source electrode located one internodal distance above the central node, and plot the resulting membrane potential profiles across all model sections:

```python run_fiber_iextra.py --type senn -d 20 --nnodes 11 -A -0.6 --tstim 0.1 -p Vm --compare```

- Use `run_fiber_iintra.py` for simulations of a **peripheral fiber models** (myelinated and unmyelinated) of any diameter and with any number of nodes upon **intracellular electrical stimulation**. For instance, to simulate a 20 um diameter, 11 nodes SENN-type fiber, stimulated at 3 nA for 0.1 ms by a anodic current injected intracellularly at the central node, and plot the resulting membrane potential profiles across all model sections:

```python run_fiber_iintra.py --type senn -d 20 --nnodes 11 -A 3 --tstim 0.1 --secid center -p Vm --compare```

- Use `run_fiber_astim_section.py` for simulations of a **peripheral fiber models** (myelinated and unmyelinated) of any diameter and with any number of nodes upon **acoustic stimulation restricted to a particular morphological section**. For instance, a 20 um diameter, 11 nodes SENN-type fiber, sonicated at 500 kHz and 100 kPa for 0.1 ms on its central node, and plot the resulting membrane potential profiles across all model sections:

```python run_fiber_astim_section.py --type senn -d 20 --nnodes 11 -f 500 -A 100 --tstim 0.1 --secid center -p Vm --compare```

### Saving and visualizing results

By default, simulation results are neither shown, nor saved.

To view results directly upon simulation completion, you can use the `-p [xxx]` option, where `[xxx]` can be `all` (to plot all resulting variables) or a given variable name (e.g. `Z` for membrane deflection, `Vm` for membrane potential, `Qm` for membrane charge density).

To save simulation results in binary `.pkl` files, you can use the `-s` option. You will be prompted to choose an output directory, unless you also specify it with the `-o <output_directory>` option. Output files are automatically named from model and simulation parameters to avoid ambiguity.

When running simulation batches, it is highly advised to specify the `-s` option in order to save results of each simulation. You can then visualize results at a later stage.

To visualize results, use the `plot_ext_timeseries.py` script. You will be prompted to select the output files containing the simulation(s) results. By default, separate figures will be created for each simulation, showing the time profiles of all resulting variables in the default morphological section of the model. Here again, you can choose to show only a subset of variables using the `-p [xxx]` option, and specify morphological sections of interest with the `--section [xxx]` option. Moreover, if you select a subset of variables, you can visualize resulting profiles across sections in comparative figures wih the `--compare` option.

# Authors

Code written and maintained by Theo Lemaire (theo.lemaire1@gmail.com).

# License & citation

This project is licensed under the MIT License - see the LICENSE file for details.

If this code base contributes to a project that leads to a scientific publication, please acknowledge this fact by citing [Lemaire, T., Vicari E., Neufeld, E., Kuster, N., and Micera, S. (2021). MorphoSONIC: a morphologically structured intramembrane cavitation model reveals fiber-specific neuromodulation by ultrasound. iScience](https://www.cell.com/iscience/fulltext/S2589-0042(21)01053-1)

# References

- [1] Lemaire, T., Neufeld, E., Kuster, N., and Micera, S. (2019). Understanding ultrasound neuromodulation using a computationally efficient and interpretable model of intramembrane cavitation. J. Neural Eng. [DOI](https://doi.org/10.1088/1741-2552/ab1685)
- [2] Lemaire, T., Vicari E., Neufeld, E., Kuster, N., and Micera, S. (2021). MorphoSONIC: a morphologically structured intramembrane cavitation model reveals fiber-specific neuromodulation by ultrasound. iScience. [DOI](https://doi.org/10.1016/j.isci.2021.103085)