from easydict import EasyDict as edict
from detection3d.utils.normalizer import AdaptiveNormalizer

__C = edict()
cfg = __C

##################################
# general parameters
##################################
__C.general = {}

__C.general.training_image_list_file = '/shenlab/lab_stor6/projects/CT_Dental/dataset/landmark_detection/face/train.csv'

__C.general.validation_image_list_file = ''

# landmark label starts from 1, 0 represents the background.
__C.general.target_landmark_label = {
    'OR-R': 9,
    'OR-L': 10,
    'Nt-L': 20,
    'Ss': 25,
    'Ch-R': 31,
    'Ch-L': 32,
    'Li': 33,
    'Me': 38
}

__C.general.save_dir = '/shenlab/lab_stor6/qinliu/projects/CT_Dental/models/model_0531_2020/batch_4'

__C.general.resume_epoch = -1

__C.general.num_gpus = 1

##################################
# dataset parameters
##################################
__C.dataset = {}

__C.dataset.crop_spacing = [1.5, 1.5, 1.5]      # mm

__C.dataset.crop_size = [96, 96, 96]   # voxel

__C.dataset.sampling_size = [6, 6, 6]      # voxel

__C.dataset.positive_upper_bound = 3    # voxel

__C.dataset.negative_lower_bound = 6    # voxel

__C.dataset.num_pos_patches_per_image = 8

__C.dataset.num_neg_patches_per_image = 16


# sampling method:
# 1) GLOBAL: sampling crops randomly in the entire image domain
__C.dataset.sampling_method = 'GLOBAL'

# linear interpolation method:
# 1) NN: nearest neighbor interpolation
# 2) LINEAR: linear interpolation
__C.dataset.interpolation = 'LINEAR'

##################################
# data augmentation parameters
##################################
__C.augmentation = {}

__C.augmentation.turn_on = False

# [x,y,z], axis = [0,0,0] will set it as random axis.
__C.augmentation.orientation_axis = [0, 0, 0]

# range of rotation in degree, 1 degree = 0.0175 radian
__C.augmentation.orientation_radian = [-30, 30]

__C.augmentation.translation = [10, 10, 10]  # mm

##################################
# loss function
##################################
__C.landmark_loss = {}

__C.landmark_loss.name = 'Focal'          # 'Dice', or 'Focal'

# class balancing weight for focal loss
__C.landmark_loss.focal_obj_alpha = [0.75] * 9

# gamma in pow(1-p,gamma) for focal loss
__C.landmark_loss.focal_gamma = 2

##################################
# net
##################################
__C.net = {}

__C.net.name = 'vdnet'

##################################
# training parameters
##################################
__C.train = {}

__C.train.epochs = 2001

__C.train.batch_size = 4

__C.train.num_threads = 4

__C.train.lr = 1e-4

__C.train.betas = (0.9, 0.999)

__C.train.save_epochs = 100

##################################
# debug parameters
##################################
__C.debug = {}

# random seed used in training
__C.debug.seed = 0

# whether to save input crops
__C.debug.save_inputs = False
