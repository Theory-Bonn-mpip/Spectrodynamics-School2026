Installation and Setup
======================

:Author: Yair Litman (`litmany@mpip-mainz.mpg.de <mailto:litmany@mpip-mainz.mpg.de>`_)

Quick start
-----------

.. code-block:: bash

   ./setup.sh

Does all of the below in one go — safe to re-run.

Individual steps
----------------

.. code-block:: bash

   conda env create -n Tutorial_II --file environment.yml   # create the environment
   conda activate Tutorial_II                               # activate it
   ./generate_notebooks.sh                                  # *.py -> tutorial_II.ipynb, 3_vsfg.ipynb
   ./download_models.sh                                     # MACE models (~47 MB)
   ./download_trajectories.sh                               # Part III trajectory data (~1 GB), re-run to resume

Run the tutorial
----------------

.. code-block:: bash

   conda activate Tutorial_II
   jupyter-lab tutorial_II.ipynb

Parts I and II are in ``tutorial_II.ipynb``. The optional Part III
(sum-frequency generation at the water/air interface) is a separate
notebook, ``3_vsfg.ipynb``, which builds on them.

Acknowledgements
----------------

The structure and style of this tutorial are inspired by the
`Atomistic Cookbook <https://atomistic-cookbook.org>`_ and by the
hands-on material of the
`i-PI schools and workshops <https://github.com/i-pi/tutorials-schools>`_.
