from easydict import EasyDict as edict


__C = edict()
cfg = __C


##################################
# general parameters
##################################

__C.general = {}

# image-segmentation pair list
# '/home/admin/test/data/train_face.txt'  # '/home/admin/test/data_v2/train_teeth.txt' #'/home/admin/test/T4.0/Medical-Segmentation3d-Toolkit/segmentation3d/dentaltest/train.txt' #
__C.general.imseg_list = '/home/admin/test/DentalEngine-main-20210215/segmentation3d/config/train_poc.txt'

# cd /home/admin/test/data/data_bone; dest='../train_bone.txt'; find . -type f -exec realpath '{}' \+ | sort | tee $dest.temp | wc -l > $dest; bc <<< "$(cat "$dest")/2" > $dest; cat $dest.temp >> $dest; rm $dest.temp

# the output of training models and logs
# '/shenlab/lab_stor6/qinliu/CT_Pancreas/model/model_0908_2020'
__C.general.save_dir = '/data/deng/model/rd_first_train'

# the model scale
__C.general.model_scale = 'coarse'

# continue training from certain epoch, -1 to train from scratch
__C.general.resume_epoch = -1

# the number of GPUs used in training. Set to 0 if using cpu only.
__C.general.num_gpus = 4

# random seed used in training (debugging purpose)
__C.general.seed = 0


##################################
# data set parameters
##################################

__C.dataset = {}

# the number of classes
__C.dataset.num_classes = 5

# the resolution on which segmentation is performed
__C.dataset.spacing = [2.0, 2.0, 2.0]

# the sampling crop size, e.g., determine the context information
__C.dataset.crop_size = [96, 96, 96]

# sampling method:
# 1) GLOBAL: sampling crops randomly in the entire image domain
# 2) MASK: sampling crops randomly within segmentation mask
# 3) HYBRID: Sampling crops randomly with both GLOBAL and MASK methods
# 4) CENTER: sampling crops in the image center
__C.dataset.sampling_method = 'HYBRID'

# linear interpolation method:
# 1) NN: nearest neighbor interpolation
# 2) LINEAR: linear interpolation
__C.dataset.interpolation = 'LINEAR'


##################################
# data augmentation parameters
##################################

# translation augmentation (unit: mm)
__C.dataset.random_translation = [6, 6, 6]

# spacing scale augmentation, spacing scale will be randomly selected from [min, max]
# during training, the image spacing will be spacing * scale
__C.dataset.random_scale = [0.9, 1.1]


##################################
# training loss
##################################

__C.loss = {}

# the name of loss function to use
# Focal: Focal loss, supports binary-class and multi-class segmentation
# Dice: Dice Similarity Loss which supports binary and multi-class segmentation
# CE: Cross Entropy loss
__C.loss.name = 'Focal'

# the weight for each class including background class
# weights will be normalized
__C.loss.obj_weight = [1/3, 1/3, 1/3, 1/3, 1/3]

# the gamma parameter in focal loss
__C.loss.focal_gamma = 2


##################################
# net
##################################

__C.net = {}

# the network name
__C.net.name = 'vbnet'

##################################
# training parameters
##################################

__C.train = {}

# the number of training epochs
__C.train.epochs = 4001

# the number of samples in a batch
__C.train.batchsize = 12

# the number of threads for IO
__C.train.num_threads = 12

# the learning rate
__C.train.lr = 1e-4

# the beta in Adam optimizer
__C.train.betas = (0.9, 0.999)

# the number of batches to save model
__C.train.save_epochs = 50


###################################
# debug parameters
###################################

__C.debug = {}

# whether to save input crops
__C.debug.save_inputs = False
