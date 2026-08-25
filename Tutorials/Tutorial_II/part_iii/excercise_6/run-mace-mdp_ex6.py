import sys
from ase.io import read
from mace.calculators import MACECalculator

sys.path.insert(0, '../../scripts')      # location of socketIO_mdp.py
from socketIO_mdp import MDP_SocketClient

init_geo = 'template.xyz'                # first frame of the slab trajectory
model_name = '../../MODELS/MACE-MDP.model'

# Create atoms object (same atoms as in the trajectory that is replayed)
atoms = read(init_geo, '0')

# Dipole and polarizability model. Run using CPU
model_calc = MACECalculator(model_name, device='cpu', model_type='DipolePolarizabilityMACE')

# Run using GPU
# model_calc = MACECalculator(model_name, device='cuda', model_type='DipolePolarizabilityMACE')

# Attach the calculator to the atoms object
atoms.calc = model_calc

# Create unix client. Besides the total dipole and polarizability it sends
# their ATOMIC decomposition (charges, atomic dipoles and atomic
# polarizabilities), which the SFG analysis needs (has_atomic=True)
host = "mdp-ex6"  # must match <address> in input_mdp.xml
print("Setting up socket.")
client = MDP_SocketClient(unixsocket=host, has_dipole=True, has_polarizability=True,
                          has_atomic=True,
                          dipole_units='eang')   # MACECalculator returns dipoles in e*Angstrom
print("Running socket.")
client.run(atoms, use_stress=False)
