#!/bin/bash
# This script is used only by the tutor to install DFTB+ and MARADO code.
#This has to be used just once.

cd $HOME/Codes

export DFTB_INSTALL=$(pwd)/dftbplus/_build/_install

git clone https://github.com/charlyqchm/MARADO.git
cd MARADO
make

echo "Installation of DFTB+ and MARADO code is completed."
