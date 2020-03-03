import scipy.io
import os
import numpy as np

pathtomatfile = '/home/carlos/PycharmProjects/PublicDatasets/LeedsSportPoseDatasetExtended/joints.mat'
pathtoimages = '/home/carlos/PycharmProjects/PublicDatasets/LeedsSportPoseDataset/images'
mat = scipy.io.loadmat(pathtomatfile)

joints = mat['joints']

matt = np.swapaxes(joints, 2,0)
matt = np.swapaxes(matt, 2,1)
y=2
#works