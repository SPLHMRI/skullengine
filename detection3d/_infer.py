import os

import numpy as np

from .core.lmk_det_infer import detection_single_image, load_det_model
from .utils.image_tools import crop_image

__all__ = ['model_init', 'teeth_model_init', 'infer']


READY = None
DET_MODEL_DIR = None
GPU_ID = None
MODEL1 = None
MODEL2 = None
MODEL3 = None


TEETH_READY = None
TEETH_MODEL_DIR = None
TEETH_GPU_ID = None
MODEL_T1 = None
MODEL_T2 = None
MODEL_T3 = None
MODEL_T4 = None


def model_init(gpu_id: int, checkpoint_dir=None):

    global DET_MODEL_DIR, GPU_ID, MODEL1, MODEL2, MODEL3, READY
    
    print('Initiating landmark model')

    if not checkpoint_dir:

        if 'SkullEngineCheckpointDir' in os.environ:
            checkpoint_dir = os.environ['SkullEngineCheckpointDir']

        else:
            checkpoint_dir = os.path.join(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))), 'checkpoints')

            if not os.path.isdir(os.path.join(checkpoint_dir, 'detection', 'model_0514_2020')):
                raise ValueError(
                    'Cannot find valid checkpoint directory for landmark generation.')

    DET_MODEL_DIR = os.path.join(
        checkpoint_dir, 'detection', 'model_0514_2020')

    GPU_ID = gpu_id
    
    try:
        MODEL1 = load_det_model(
            os.path.join(DET_MODEL_DIR, 'batch_1'), GPU_ID)
        MODEL2 = load_det_model(
            os.path.join(DET_MODEL_DIR, 'batch_2'), GPU_ID)
        MODEL3 = load_det_model(
            os.path.join(DET_MODEL_DIR, 'batch_3'), GPU_ID)
    except:
        MODEL1 = None
        MODEL2 = None
        MODEL3 = None

    READY = True

    teeth_model_init(gpu_id, checkpoint_dir)

    return None



def teeth_model_init(gpu_id: int, checkpoint_dir=None):

    global TEETH_MODEL_DIR, TEETH_GPU_ID, MODEL_T1, MODEL_T2, MODEL_T3, MODEL_T4, TEETH_READY
    
    print('Initiating landmark model')

    if not checkpoint_dir:

        if 'SkullEngineCheckpointDir' in os.environ:
            checkpoint_dir = os.environ['SkullEngineCheckpointDir']

        else:
            checkpoint_dir = os.path.join(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))), 'checkpoints')

            if not os.path.isdir(os.path.join(checkpoint_dir, 'detection', 'model_0514_2020')):
                raise ValueError(
                    'Cannot find valid checkpoint directory for landmark generation.')

    TEETH_MODEL_DIR = os.path.join(
        checkpoint_dir, 'detection', 'model_0514_2020')

    TEETH_GPU_ID = gpu_id
    
    try:
        MODEL_T1 = load_det_model(
            os.path.join(TEETH_MODEL_DIR, 'batch_4_lower_1'), TEETH_GPU_ID)
        MODEL_T2 = load_det_model(
            os.path.join(TEETH_MODEL_DIR, 'batch_4_lower_2'), TEETH_GPU_ID)
        MODEL_T3 = load_det_model(
            os.path.join(TEETH_MODEL_DIR, 'batch_4_upper_1'), TEETH_GPU_ID)
        MODEL_T4 = load_det_model(
            os.path.join(TEETH_MODEL_DIR, 'batch_4_upper_2'), TEETH_GPU_ID)
        
    except:
        MODEL_T1 = None
        MODEL_T2 = None
        MODEL_T3 = None
        MODEL_T4 = None

    TEETH_READY = True
    return None


def infer(img):
    # Note: landmarks L6CF-R, L6CF-L, L6DC-R, L6DC-L won't be detected because they are too close.

    assert READY, 'Landmark models are not initiated properly.'

    # detect batch 1
    print('Start detecting the landmarks ...')

    if MODEL1:
        landmark_batch_1 = detection_single_image(
            img, MODEL1)

    else:
        batch_1_model = load_det_model(os.path.join(
            DET_MODEL_DIR, 'batch_1'), GPU_ID)
        landmark_batch_1 = detection_single_image(
            img, batch_1_model)
        del batch_1_model

    if MODEL2:
        landmark_batch_2 = detection_single_image(
            img, MODEL2)

    else:
        batch_2_model = load_det_model(os.path.join(
            DET_MODEL_DIR, 'batch_2'), GPU_ID)
        landmark_batch_2 = detection_single_image(
            img, batch_2_model)
        del batch_2_model

    if MODEL3:
        landmark_batch_3 = detection_single_image(
            img, MODEL3)

    else:
        batch_3_model = load_det_model(os.path.join(
            DET_MODEL_DIR, 'batch_3'), GPU_ID)
        landmark_batch_3 = detection_single_image(
            img, batch_3_model)
        del batch_3_model

    _lmk = np.vstack(
        (landmark_batch_1.values, landmark_batch_2.values, landmark_batch_3.values))
    _ind = _lmk[:, 0].argsort()
    lmk = dict(zip(_lmk[_ind, 0], _lmk[_ind, 1:].astype(np.float32)))

    if TEETH_READY:
        l0 = landmark_batch_3[landmark_batch_3['name'] == 'L0'].values[0,1:]
        if l0.all():
            print('Start detecting the teeth landmarks')
            teeth_lmk = _infer_teeth(img, l0)
    
            lmk.update(teeth_lmk)

    return lmk


def _infer_teeth(img, L0):

    def is_voxel_coordinate_valid(coord_voxel, image_size):
        """
        Check whether the voxel coordinate is out of bound.
        """
        for idx in range(3):
            if coord_voxel[idx] < 0 or coord_voxel[idx] >= image_size[idx]:
                return False
        return True


    def is_world_coordinate_valid(coord_world):
        """
        Check whether the world coordinate is valid.
        The world coordinate is invalid if it is (0, 0, 0), (1, 1, 1), or (-1, -1, -1).
        """
        coord_world_npy = np.array(coord_world)

        if np.linalg.norm(coord_world_npy, ord=1) < 1e-6 or \
                np.linalg.norm(coord_world_npy - np.ones(3), ord=1) < 1e-6 or \
                np.linalg.norm(coord_world_npy - -1 * np.ones(3), ord=1) < 1e-6:
            return False

        return True

    lmk = {}

    if is_world_coordinate_valid(L0):
        voxel_coord_l0 = img.TransformPhysicalPointToIndex(L0)
        if is_voxel_coordinate_valid(voxel_coord_l0, img.GetSize()):
            img_spacing, img_size = img.GetSpacing(), img.GetSize()
            crop_spacing, crop_size = [0.8, 0.8, 0.8], [128, 96, 96]
            resample_ratio = [crop_spacing[idx] / img_spacing[idx] for idx in range(3)]

            offset = [-int(64 * resample_ratio[0]), -int(16 * resample_ratio[1]), -int(48 * resample_ratio[2])]
            left_bottom_voxel = [voxel_coord_l0[idx] + offset[idx] for idx in range(3)]
            right_top_voxel = [left_bottom_voxel[idx] + int(crop_size[idx] * resample_ratio[idx]) - 1 for idx in range(3)]
            for idx in range(3):
                left_bottom_voxel[idx] = max(0, left_bottom_voxel[idx])
                right_top_voxel[idx] = min(img_size[idx] - 1, right_top_voxel[idx])

            for idx in range(3):
                left_bottom_voxel[idx] = max(0, left_bottom_voxel[idx])
                right_top_voxel[idx] = min(img_size[idx] - 1, right_top_voxel[idx])

            crop_voxel_center = [(left_bottom_voxel[idx] + right_top_voxel[idx]) // 2 for idx in range(3)]
            crop_world_center = img.TransformContinuousIndexToPhysicalPoint(crop_voxel_center)
            cropped_img = crop_image(img, crop_world_center, crop_size, crop_spacing, 'LINEAR')


            if MODEL_T1:
                landmark_teeth_1 = detection_single_image(
                    cropped_img, MODEL_T1)

            else:
                teeth_1_model = load_det_model(os.path.join(
                    TEETH_MODEL_DIR, 'batch_4_lower_1'), TEETH_GPU_ID)
                landmark_teeth_1 = detection_single_image(
                    cropped_img, teeth_1_model)
                del teeth_1_model


            if MODEL_T2:
                landmark_teeth_2 = detection_single_image(
                    cropped_img, MODEL_T2)

            else:
                teeth_2_model = load_det_model(os.path.join(
                    TEETH_MODEL_DIR, 'batch_4_lower_2'), TEETH_GPU_ID)
                landmark_teeth_2 = detection_single_image(
                    cropped_img, teeth_2_model)
                del teeth_2_model


            if MODEL_T3:
                landmark_teeth_3 = detection_single_image(
                    cropped_img, MODEL_T3)

            else:
                teeth_3_model = load_det_model(os.path.join(
                    TEETH_MODEL_DIR, 'batch_4_upper_1'), TEETH_GPU_ID)
                landmark_teeth_3 = detection_single_image(
                    cropped_img, teeth_3_model)
                del teeth_3_model


            if MODEL_T4:
                landmark_teeth_4 = detection_single_image(
                    cropped_img, MODEL_T4)

            else:
                teeth_4_model = load_det_model(os.path.join(
                    TEETH_MODEL_DIR, 'batch_4_upper_2'), TEETH_GPU_ID)
                landmark_teeth_4 = detection_single_image(
                    cropped_img, teeth_4_model)
                del teeth_4_model

            _lmk = np.vstack(
                (landmark_teeth_1.values, landmark_teeth_2.values, landmark_teeth_3.values, landmark_teeth_4.values))
            _ind = _lmk[:, 0].argsort()
            lmk = dict(zip(_lmk[_ind, 0], _lmk[_ind, 1:].astype(np.float32)))

    return lmk