#Global minimum time control for stiff van der Pol oscillator using Janus Dual nunber tensor library

import torch
import janus_nlp as jnlp
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from smac.facade.smac_ac_facade import SMAC4AC
from smac.scenario.scenario import Scenario
import numpy as np

jnlp.set_global_dtype(torch.float64)
jnlp.set_global_backend('cpu')
jnlp.vdpc_solve()