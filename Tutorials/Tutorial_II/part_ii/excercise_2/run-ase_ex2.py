import os
import sys
import numpy as np
from ase.calculators.socketio import SocketClient
from ase.io import read
from mace.calculators.mace import MACECalculator

init_geo = 'init.xyz'
model_name = '../../MODELS/MACE_MLIP.model'

# Create atoms object
atoms = read(init_geo, '0')

# Run using CPU
model_calc = MACECalculator(model_name, device='cpu', default_dtype='float32')

# Run using GPU
# model_calc = MACECalculator(model_name, device='cuda', default_dtype='float32')

# Attach the calculator to the atoms object
atoms.calc = model_calc

# Create unix client. The address must match <address> in input.xml;
# run_ex2.sh replaces both by a name unique to this folder, so that students
# sharing a machine in the school do not end up on the same socket.
host = os.environ.get("IPI_ADDRESS", "mace-ex2")
print("Setting up socket.")
client = SocketClient(unixsocket=host)
print("Running socket.")
client.run(atoms, use_stress=False)
