# MACE Machine-Learned Models

Machine-learned models trained with the [MACE](https://github.com/ACEsuit/mace) architecture for the Spectrodynamics2026 School.

The models were trained with a locally modified version of MACE developed by the STREAM group.

## Overview

| | |
|---|---|
| **Model architecture** | MACE |
| **MACE version** | Locally modified version 0.3.16 ([MDP-derivatives-with-bec_training branch](https://github.com/MPIP-STREAM/mace/tree/MDP-derivatives-with-bec_training)) |
| **Reference data level of theory** | revPBE + D3(0) *(zero damping)* |
| **Reference data code** | FHI-aims |
| **FHI-aims settings** | Light basis set |

## Files

The two `.model` files are **not kept in git**: run `./download_models.sh`
(or `./setup.sh`) in the tutorial root to fetch them into this folder.

| File | Description |
|---|---|
| `MACE_MLIP.model` | MLIP (energies and forces) for water |
| `MACE-MDP.model` | MACE model for dipole moments and polarizabilities |
| `README.md` | This file |

## Chemical system and scope

These models were trained on water systems, covering bulk water and the water/air interface at room temperature and a density of 1.00 g/cm³.

## Accuracy / validation

`MACE_MLIP.model`

| Property | Test error |
|---|---|
| Energy (meV/atom) | *<!-- TODO -->* |
| Forces (meV/Å) | *<!-- TODO -->* |

`MACE-MDP.model`

| Property | Test error |
|---|---|
| Dipole moment (*<!-- TODO: units, e.g. mD/atom or me·Å/atom -->*) | *<!-- TODO -->* |
| Polarizability (*<!-- TODO: units, e.g. mÅ³/atom -->*) | *<!-- TODO -->* |

## How to cite

The paper describing the SFG extension of MACE-MDP is in preparation:

> *Ab initio SFG simulations made easy* — Litman *et al.* (in preparation)

Until it is published, please contact the authors before using or redistributing these models outside the school.

If you use MACE, please also cite:

- Batatia et al., *MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields*, NeurIPS 2022.
- Batatia et al., *The design space of E(3)-equivariant atom-centred interatomic potentials*, 2022.

## Authors / contact

Yair Litman, Max Planck Institute for Polymer Research, litmany@mpip-mainz.mpg.de

## License

CC-BY-4.0
