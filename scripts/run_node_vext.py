# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-08-24 11:55:07
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-07-01 17:23:32

''' Run E-STIM simulations of a specific point-neuron. '''

import matplotlib.pyplot as plt

from PySONIC.core import Batch, PointNeuron
from PySONIC.utils import logger
from PySONIC.plt import GroupedTimeSeries
from PySONIC.parsers import EStimParser
from ExSONIC.core import VextNode


def main():
    # Parse command line arguments
    parser = EStimParser()
    parser.add_argument(
        '-V', '--Vext', nargs='+', default=10., help='Extracellular potential (mV)')
    args = parser.parse()
    logger.setLevel(args['loglevel'])
    if args['mpi']:
        logger.warning('NEURON multiprocessing disabled')

    # Run E-STIM batch
    logger.info("Starting Vext-STIM simulation batch")
    inputs = [args[k] for k in ['Vext', 'tstim', 'toffset', 'PRF', 'DC']]
    queue = PointNeuron.simQueue(*inputs, outputdir=args['outputdir'])
    output = []
    for pneuron in args['neuron']:
        node = VextNode(pneuron)
        batch = Batch(node.simAndSave if args['save'] else node.simulate, queue)
        output += batch(loglevel=args['loglevel'])

    # Plot resulting profiles
    if args['plot'] is not None:
        parser.parsePlot(args, output)

if __name__ == '__main__':
    main()