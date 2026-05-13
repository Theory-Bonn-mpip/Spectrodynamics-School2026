import os
import sys
import numpy as np
from ase.calculators.socketio import SocketClient
from ase.io import read
from mace.calculators import mace_mp
#from mace.calculators.mace import MACECalculator

#model_name=sys.argv[1]
#atoms = read('init.xyz', '0')
#atoms.set_cell([12.42,12.42,12.42])
atoms = read('data/water_32.pdb', '0')
atoms.pbc = True
#atoms.calc = MACECalculator(model_name, device='cuda', default_dtype='float32')
#macemp = mace_mp(model="large")
#macemp = mace_mp(model="medium")
macemp = mace_mp(model="small",device='cpu',default_dtype='float32')
atoms.calc = macemp



# Create Client
# unix
port = 11111
host = "h2o-mace"
print ("Setting up socket.")
client = SocketClient(unixsocket=host)
print ("Running socket.")
client.run(atoms,use_stress=False )
print ("setting up calculator")






