import os
import sys
from ase.io import read
from mace.calculators import MACECalculator

sys.path.insert(0, '../../scripts')      # location of socketIO_mdp.py
from socketIO_mdp import MDP_SocketClient

init_geo = '../excercise_2/init.xyz'
model_name = '../../MODELS/MACE-MDP.model'

# Create atoms object (same atoms as in the trajectory that is replayed)
atoms = read(init_geo, '0')

# Dipole and polarizability model. Run using CPU
model_calc = MACECalculator(model_name, device='cpu', model_type='DipolePolarizabilityMACE')

# Run using GPU
# model_calc = MACECalculator(model_name, device='cuda', model_type='DipolePolarizabilityMACE')

# Attach the calculator to the atoms object
atoms.calc = model_calc

# Create unix client: instead of energies and forces, it sends the total
# dipole and polarizability of each frame to i-PI. The address must match
# <address> in input_mdp.xml; run_ex3.sh replaces both by a name unique to
# this folder, so that students sharing a machine in the school do not end up
# on the same socket.
host = os.environ.get("IPI_ADDRESS", "mdp-ex3")
print("Setting up socket.")
client = MDP_SocketClient(unixsocket=host, has_dipole=True, has_polarizability=True,
                          dipole_units='eang')   # MACECalculator returns dipoles in e*Angstrom
print("Running socket.")
client.run(atoms, use_stress=False)
