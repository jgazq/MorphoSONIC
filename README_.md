# Description

`MorphoSONIC` is a Python+NEURON implementation of **spatially extended representations** of the **multi-Scale Optimized Neuronal Intramembrane Cavitation (SONIC) model [1]**. It enables the simulation of the distributed electrical response of morphologically realistic neuron representations to acoustic stimuli, as predicted by the *intramembrane cavitation* hypothesis. The details of the MorphoSONIC framework and of its application to study ultrasound neuromodulation in peripheral nerve fibers are described in [2].

This package expands features from the `PySONIC` package (https://github.com/tjjlemaire/PySONIC).

# Content of repository

## Models

The `models` module defines a variety of neuron models:

- the `Node` class that provides a NEURON wrapper around the point-neuron models defined in the `PySONIC` package, and can be simulated with both punctual electrical and acoustic drives.
- the `RadialModel` class defines a **nanoscale radially-symmetric model** with central and peripheral compartments. It can be used to model the coupling between an "ultrasound-responsive" sonophore and an "ultrasound-resistant" surrounding membrane (see `surroundedSonophore` function). As this model is radially symmetric, some adaptation was needed in order to represent it within the *NEURON* environment (check [this link](docs/NEURON_radial_geometry.md) for more details).

The module also contains morphologically-structured models of **unmyelinated and myelinated peripheral fibers**:
- `SennFiber` implements a spatially-extended nonlinear node (SENN) myelinated fiber model, as defined in Reilly 1985.
- `SweeneyFiber` implements the SENN model variant defined in Sweeney 1987.
- `MRGFiber` implements the double-cable myelinated fiber model defined as in McIntyre 2002.
- `UnmyelinatedFiber` implements an unmyelinated fiber model defined as in Sundt 2015.

By default, multi-compartment models are wired using the conventional *NEURON* cable representation that assumes a voltage-casted electrical system and a constant membrane capacitance throughout simulations. This wiring strategy is incompatible with the SONIC model, where both these assumptions are violated. Therefore, to enable the simulation of these models under acoustic perturbations, we defined an alternative wiring scheme that is intrinsically compatible with the SONIC model and can be readily substituted to *NEURON*'s default wiring scheme (see [2] for more details). This substitution is achieved by assigning a simple decorator (`addSonicFeatures`) to each model class.

## Sources

The `sources` module defines a variety of analytical models of electrical and acoustic exposure distributions that can be used to stimulate spatially-extended models:

- `IntracellularCurrent` for local intracellular current injection at a specific section.
- `ExtracellularCurrent` for distributed voltage perturbation resulting from current injection at a distant point-source electrode.
- `GaussianVoltageSource` for a distributed voltage perturbation defined by a Gaussian distribution
- `SectionAcousticSource` for local acoustic perturbation at a specific section.
- `PlanarDiskTransducerSource` for distributed acoustic perturbation resulting from sonication by a distant planar acoustic transducer.
- `GaussianAcousticSource` for a distributed acoustic perturbation defined by a Gaussian distribution

## Membrane mechanisms (NMODL)

Most point-neuron models defined in the `PySONIC` package have been translated to equivalent membrane mechanisms in **NMODL** language, stored as MOD files in the `nmodl` folder. If you implemented additional point-neuron models and wish to translate them into MOD files, use the `generate_mod_file.py` script.

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

!INCLUDECODE "examples/node.py" (python)

Similarly, you can run simulations of myelinated and unmyelinated fiber models under extracellular electrical and ultrasonic stimulation, and visualize the simulation results:

!INCLUDECODE "examples/fiber.py" (python)

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