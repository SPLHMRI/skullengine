#!/usr/bin/env python
import argparse
import numpy as np
import SimpleITK as sitk
import os
import sys
import time
import csv

from .core.seg_infer import segmentation_internal, load_models
from .utils.image_tools import crop_image

__all__ = ['model_init', 'infer_coarse', 'infer_refine']


READY = None
GPU_ID = None
SEG_MODEL_DIR_BASE = None
SEG_MODEL_DIR_COR = None
SEG_MODEL_DIR_A = None
MODEL_BASE = None
MODEL_COR = None
MODEL_A = None


def model_init(gpu_id: int, checkpoint_dir=None, patch_refine=True):

    global SEG_MODEL_DIR_BASE, SEG_MODEL_DIR_COR, SEG_MODEL_DIR_A, GPU_ID, MODEL_BASE, MODEL_COR, MODEL_A, READY

    print('Initiating segmentation model')

    if not checkpoint_dir:

        if 'SkullEngineCheckpointDir' in os.environ:
            checkpoint_dir = os.environ['SkullEngineCheckpointDir']

        else:
            checkpoint_dir = os.path.join(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))), 'checkpoints')

            if not os.path.isdir(os.path.join(checkpoint_dir, 'segmentation')):
                raise ValueError(
                    'Cannot find valid checkpoint directory for landmark generation.')

    SEG_MODEL_DIR_BASE = os.path.join(
        checkpoint_dir, 'segmentation', 'model_0429_2020')
    SEG_MODEL_DIR_COR = os.path.join(
        checkpoint_dir, 'segmentation', 'model_COR')
    SEG_MODEL_DIR_A = os.path.join(
        checkpoint_dir, 'segmentation', 'model_whole')

    GPU_ID = gpu_id

    try:
        MODEL_BASE = load_models(SEG_MODEL_DIR_BASE, GPU_ID)
        if patch_refine:
            MODEL_COR = load_models(SEG_MODEL_DIR_COR, GPU_ID)
            MODEL_A = load_models(SEG_MODEL_DIR_A, GPU_ID)
    except:
        MODEL_BASE = None
        MODEL_COR = None
        MODEL_A = None

    READY = True
    return None


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


def refine_facial_region(img: sitk.Image, seg: sitk.Image, models, left_face_anchor, right_face_anchor):
    """
    This function aims to refine the segmentation results in the facial region.
    """

    crop_physical_volume = [38.4, 38.4, 38.4]  # unit mm
    crop_spacing = [img.GetSpacing()[idx] for idx in range(3)]
    crop_size = [crop_physical_volume[idx] // crop_spacing[idx]
                 for idx in range(3)]

    if is_world_coordinate_valid(left_face_anchor):
        left_voxel_coord = img.TransformPhysicalPointToIndex(
            left_face_anchor)
        if is_voxel_coordinate_valid(left_voxel_coord, img.GetSize()):
            cropped_left_image = crop_image(
                img, left_face_anchor, crop_size, crop_spacing, 'LINEAR')
            _, left_COR_mask = segmentation_internal(
                cropped_left_image, models)

            left_COR_mask = left_COR_mask[:-20, 5:-5, 32:-32]
            left_COR_start_voxel = seg.TransformPhysicalPointToIndex(
                left_COR_mask.GetOrigin())
            seg = sitk.Paste(seg, left_COR_mask, left_COR_mask.GetSize(), [
                0, 0, 0], left_COR_start_voxel)

    if is_world_coordinate_valid(right_face_anchor):
        right_voxel_coord = img.TransformPhysicalPointToIndex(
            right_face_anchor)
        if is_voxel_coordinate_valid(right_voxel_coord, img.GetSize()):
            cropped_right_image = crop_image(
                img, right_face_anchor, crop_size, crop_spacing, 'LINEAR')
            _, right_COR_mask = segmentation_internal(
                cropped_right_image, models)

            right_COR_mask = right_COR_mask[:, 5:-5, 32:-32]
            right_COR_start_voxel = seg.TransformPhysicalPointToIndex(
                right_COR_mask.GetOrigin())
            seg = sitk.Paste(seg, right_COR_mask, right_COR_mask.GetSize(), [
                0, 0, 0], right_COR_start_voxel)

    return seg


def refine_whole_region(img: sitk.Image, seg: sitk.Image, models, A_anchor):
    """
    This function aims to refine the segmentation results in the palate  region.
    """

    crop_physical_volume = [134.4, 72, 60]  # [38.4, 38.4, 38.4]  # unit mm
    crop_spacing = [img.GetSpacing()[idx] for idx in range(3)]
    crop_size = [crop_physical_volume[idx] // crop_spacing[idx]
                 for idx in range(3)]

    if is_world_coordinate_valid(A_anchor):
        left_voxel_coord = np.array(
            img.TransformPhysicalPointToIndex(A_anchor))
        if is_voxel_coordinate_valid(left_voxel_coord, img.GetSize()):
            image_spacing = img.GetSpacing()
            voxel_offset = np.array(
                [0 * 0.3 / image_spacing[0], 100 * 0.3 / image_spacing[1], 30 * 0.3 / image_spacing[1]])
            cropped_center = img.TransformContinuousIndexToPhysicalPoint(
                left_voxel_coord + voxel_offset)
            cropped_palate_image = crop_image(
                img, cropped_center, crop_size, crop_spacing, 'LINEAR')
            _, A_mask = segmentation_internal(
                cropped_palate_image, models)

            A_mask = A_mask[:, :, :-2]
            A_start_voxel = seg.TransformPhysicalPointToIndex(
                A_mask.GetOrigin())
            # mask_A_fill = sitk.Paste(seg, A_mask, A_mask.GetSize(), [0, 0, 0], A_start_voxel)
            # sitk.WriteImage(mask_A_fill, os.path.join(save_dicom_folder, 'seg_A_fill.nii.gz'), True) #.format(mask_name)

            crop_size = A_mask.GetSize()
            cropping_origin = A_mask.GetOrigin()
            cropping_direction = seg.GetDirection()

            transform = sitk.Transform(3, sitk.sitkIdentity)
            cropped_A_mask = sitk.Resample(seg, crop_size, transform, sitk.sitkNearestNeighbor, cropping_origin, image_spacing,
                                           cropping_direction)

            # , A_mask.GetSize(), [0, 0, 0], A_start_voxel)
            mask_A_xor = sitk.Or(cropped_A_mask, A_mask)
            arr1 = sitk.GetArrayFromImage(cropped_A_mask)
            arr2 = sitk.GetArrayFromImage(A_mask)
            # arr = np.zeros_like(arr1)
            # arr[arr1]
            arr = np.maximum(arr1, arr2).astype(np.int8)
#            arr = np.logical_or(arr1, arr2).astype(np.int8)
            mask_A_xor = sitk.GetImageFromArray(arr)
            mask_A_xor.CopyInformation(A_mask)

            # sitk.WriteImage(mask_A_xor, os.path.join(save_dicom_folder, '{}_A_XOR_pre.nii.gz'.format(mask_name)), True)
            seg = sitk.Paste(seg, mask_A_xor, mask_A_xor.GetSize(), [
                0, 0, 0], A_start_voxel)

    return seg



def infer_coarse(img):
    # Note: landmarks L6CF-R, L6CF-L, L6DC-R, L6DC-L won't be detected because they are too close.

    assert READY, 'Landmark models are not initiated properly.'

    if MODEL_BASE:
        _, seg = segmentation_internal(img, MODEL_BASE)

    else:
        models_base = load_models(SEG_MODEL_DIR_BASE, GPU_ID)
        _, seg = segmentation_internal(img, models_base)
        del models_base

    return seg


def infer_refine(img, seg, lmk):
    
    assert READY, 'Landmark models are not initiated properly.'

    A = lmk['A'].tolist()
    COR_L = lmk['COR-L'].tolist()
    COR_R = lmk['COR-R'].tolist()

    if MODEL_COR:
        seg = refine_facial_region(
            img, seg, MODEL_COR, COR_L, COR_R)

    else:
        models_cor = load_models(SEG_MODEL_DIR_COR, GPU_ID)
        seg = refine_facial_region(
            img, seg, models_cor, COR_L, COR_R)
        del models_cor

    if MODEL_A:
        seg = refine_whole_region(img, seg, MODEL_A, A)

    else:
        models_a = load_models(SEG_MODEL_DIR_A, GPU_ID)
        seg = refine_whole_region(img, seg, models_a, A)
        del models_a
        
    return seg
