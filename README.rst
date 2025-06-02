Position: Constants are Critical in Regret Bounds for Reinforcement Learning
****************************************************************************

Simone Drago, Marco Mussi and Alberto Maria Metelli

Running Experiments
===================

The code requires *python3* along with *numpy* and *matplotlib*.

The configurations used for the experiments in the main paper are in the *configs* folder.

To run the experiments on the illustrative environment, from the root directory, call the python script *parallel_runner.py* with as parameter the pathname of the configuration file (also with ".json"). The bash script *runner_sequence.sh* allows to run in sequence all the configurations in the *configs* folder.

To run the RiverSwim experiment, call the python script *parallel_runner_riverswim.py*.

To run the experiment on MABs, call the python script *runner_bandit.py* with 4 parameters: number of actions, time horizon, number of trials, number of cores.

Cite this Work
==============

If you are using this code for your scientific publications, please cite:

.. code:: bibtex

    @inproceedings{drago2025position,
      author    = {Drago, Simone and
                   Mussi, Marco and
                   Metelli, Alberto Maria},
      title        = {Position: Constants are Critical in Regret Bounds for Reinforcement Learning},
      booktitle    = {International Conference on Machine Learning (ICML)},
      series       = {Proceedings of Machine Learning Research},
      volume       = {267},
      publisher    = {{PMLR}},
      year         = {2025}
   }

Contact Us
==========

For any question, drop an e-mail at marco.mussi@polimi.it
