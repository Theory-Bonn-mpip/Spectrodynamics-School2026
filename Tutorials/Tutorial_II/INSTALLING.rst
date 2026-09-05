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

``setup.sh`` also switches JupyterLab's notebook rendering to ``none``
(an ``overrides.json`` inside the environment): the default virtualized
rendering can intermittently leave cells blank, especially on remote/cloud
machines. To apply just that fix to an existing environment:

.. code-block:: bash

   conda run -n Tutorial_II python -c "import json,os,sys; d=os.path.join(sys.prefix,'share','jupyter','lab','settings'); os.makedirs(d,exist_ok=True); p=os.path.join(d,'overrides.json'); cfg=json.load(open(p)) if os.path.exists(p) else {}; cfg.setdefault('@jupyterlab/notebook-extension:tracker',{}).update(windowingMode='none'); json.dump(cfg,open(p,'w'),indent=2); print(p)"

(then reload the browser tab). The first code cell of each notebook applies
the same setting at the user level, so simply running the notebooks once
also fixes it.

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
