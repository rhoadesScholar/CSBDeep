# Headless version of the notebook for long unattended runs
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np

from tifffile import imread
from csbdeep.utils import axes_dict, plot_some, plot_history
from csbdeep.utils.tf import limit_gpu_memory
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE

# SET DATA PATH
datapath = "/n/groups/htem/ESRF_id16a/tomo_ML/ReducedAnglesXray/CARE/mCTX/"

# LOAD TRAINING DATA
(X,Y), (X_val,Y_val), axes = load_training_data(datapath + 'my_training_data.npz', validation_split=0.1, verbose=True)

c = axes_dict(axes)['C']
n_channel_in, n_channel_out = X.shape[c], Y.shape[c]

# SET CONFIG VARIABLES
config = Config(axes, n_channel_in, n_channel_out, train_steps_per_epoch=400)#, unet_kern_size=3, unet_n_depth=3)
print(config)
vars(config)

# MAKE MODEL
model = CARE(config, 'kernSize3_depth2_450p', basedir='models')

# TRAIN
history = model.train(X,Y, validation_data=(X_val,Y_val))

# EXPORT TO WORK WITH ImageJ CSBDeep Plugin 
model.export_TF()