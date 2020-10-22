# EcoLight

EcoLight is a reinforcement learning agent for AI based separate and independent (without network communication) 
traffic signal control. It intends to match state-of-the-art-performance using low computational resources. 
It utilizes offline DRL training and paves the way for efficient deployment. 
The corresponding research paper is accepted at NeurIPS-2020.

It uses an open source project Presslight and my another project SachinLight as reference.

Start an experiment by:

``python3 code/runexp.py --goodness=G --quantize=Q``

where G can be 1 for FairShare or 2 for DecisionConsistency
and Q is desired quantization level

## Preparing build environment

  The below commands should do the needful -

  ``conda create -n AnyName python=3.6``

  ``conda install Keras==2.3.1``
  
## Datasets

  EcoLight uses real data for the evaluations. Traffic file and road networks of New York City and New Delhi 
  can be found in ``data``, it contains two networks of NYC at different scales (16 intersections and 48 
  intersections) taken from Presslight, and New Delhi data for a 3-approach single intersection is taken 
  from SachinLight.

## Modules

* ``runexp.py``

  Run the pipeline under different traffic flows with various arguments/parameters. Specific traffic flow files as well as basic configuration can be assigned in this file. For details about config, please turn to ``config.py``.

  For most cases, you might only modify traffic files and config parameters in ``runexp.py`` via commandline parameters.

* ``config.py``

  The default configuration of this project. Note that most of the useful parameters can be updated in ``runexp.py`` via command line arguments.

* ``dqn.py``

  A DQN based agent build atop ``agent.py``

* ``pipeline.py``

  The whole pipeline is implemented in this module:

  Start a simulator environment, run a simulation for certain time (one round), construct samples from raw log data, update the model etc.

* ``generator.py``

  A generator to load a model, start a simulator environment, conduct a simulation and log the results.

* ``anon_env.py``

  Define a simulator environment to interact with the simulator and obtain needed data.

* ``construct_sample.py``

  Construct training samples from data received from simulator. Select desired state features in the config and compute the corresponding reward.

* ``updater.py``

  Define a class of updater for model updating.
  
## Simulator

  The project uses an open source simulator CityFlow to get the impact of EcoLight's actions in the environment. 
  The default library is built for Python 3.6, and a Python 3.7 variant is also provided. 
  If the given libraries don't work for you, please build the Cityflow simulator from source code.
