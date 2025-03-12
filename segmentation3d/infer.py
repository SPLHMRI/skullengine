#!/usr/bin/env python
import argparse
import numpy as np
import SimpleITK as sitk
import os
import sys
import time
import csv

from .core.seg_infer import segmentation_internal, load_models, check_input
from .utils.dicom_helper import write_binary_dicom_series
from .utils.dicom_helper import read_dicom_series, write_dicom_series, dicom_tags_dict
from .utils.image_tools import crop_image


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


def refine_facial_region(img: sitk.Image, seg: sitk.Image, model_dir, gpu_id, left_face_anchor, right_face_anchor):
    """
    This function aims to refine the segmentation results in the facial region.
    """

    crop_physical_volume = [38.4, 38.4, 38.4]  # unit mm
    crop_spacing = [img.GetSpacing()[idx] for idx in range(3)]
    crop_size = [crop_physical_volume[idx] // crop_spacing[idx]
                 for idx in range(3)]

    models = load_models(model_dir, gpu_id)

    if is_world_coordinate_valid(left_face_anchor):
        left_voxel_coord = img.TransformPhysicalPointToIndex(
            left_face_anchor)
        if is_voxel_coordinate_valid(left_voxel_coord, img.GetSize()):
            cropped_left_image = crop_image(
                img, left_face_anchor, crop_size, crop_spacing, 'LINEAR')
            _, left_COR_mask = segmentation_internal(
                cropped_left_image, models, gpu_id)

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
                cropped_right_image, models, gpu_id)

            right_COR_mask = right_COR_mask[:, 5:-5, 32:-32]
            right_COR_start_voxel = seg.TransformPhysicalPointToIndex(
                right_COR_mask.GetOrigin())
            seg = sitk.Paste(seg, right_COR_mask, right_COR_mask.GetSize(), [
                0, 0, 0], right_COR_start_voxel)

    return seg


def refine_whole_region(img: sitk.Image, seg: sitk.Image, model_dir, gpu_id, A_anchor):
    """
    This function aims to refine the segmentation results in the palate  region.
    """

    crop_physical_volume = [134.4, 72, 60]  # [38.4, 38.4, 38.4]  # unit mm
    crop_spacing = [img.GetSpacing()[idx] for idx in range(3)]
    crop_size = [crop_physical_volume[idx] // crop_spacing[idx]
                 for idx in range(3)]

    models = load_models(model_dir, gpu_id)

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
                cropped_palate_image, models, gpu_id)

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


def dental_refine_bone_region(img: sitk.Image, seg: sitk.Image, model_dir, gpu_id, left_bone_anchor, right_bone_anchor):
    """
    This function aims to refine the segmentation results in the facial region.
    """
    crop_physical_volume = [38.4, 38.4, 38.4]  # unit mm
    crop_spacing = [img.GetSpacing()[idx] for idx in range(3)]
    crop_size = [crop_physical_volume[idx] // crop_spacing[idx]
                 for idx in range(3)]

    models = load_models(model_dir, gpu_id)

    left_bone_mask = None
    if is_world_coordinate_valid(left_bone_anchor):
        left_voxel_coord = np.array(
            img.TransformPhysicalPointToIndex(left_bone_anchor))
        if is_voxel_coordinate_valid(left_voxel_coord, img.GetSize()):
            img_spacing = img.GetSpacing()
            voxel_offset = np.array(
                [15 * 0.3 / img_spacing[0], 65 * 0.3 / image_spacing[1], -15 * 0.3 / image_spacing[2]])
            cropped_center = img.TransformContinuousIndexToPhysicalPoint(
                left_voxel_coord + voxel_offset)
            cropped_left_image = crop_image(
                img, cropped_center, crop_size, crop_spacing, 'LINEAR')
            _, left_bone_mask = segmentation_internal(
                cropped_left_image, models, gpu_id)

    right_bone_mask = None
    if is_world_coordinate_valid(right_bone_anchor):
        right_voxel_coord = np.array(
            img.TransformPhysicalPointToIndex(right_bone_anchor))
        if is_voxel_coordinate_valid(right_voxel_coord, img.GetSize()):
            image_spacing = img.GetSpacing()
            voxel_offset = np.array(
                [-15 * 0.3 / image_spacing[0], 65 * 0.3 / image_spacing[1], -15 * 0.3 / image_spacing[2]])
            cropped_center = img.TransformContinuousIndexToPhysicalPoint(
                right_voxel_coord + voxel_offset)
            cropped_right_image = crop_image(
                img, cropped_center, crop_size, crop_spacing, 'LINEAR')
            _, right_bone_mask = segmentation_internal(
                cropped_right_image, models, gpu_id)

    return left_bone_mask, right_bone_mask


def dental_refine_IC_region(img: sitk.Image, seg: sitk.Image, model_dir, gpu_id, palate_anchor):
    """
    This function aims to refine the segmentation results in the palate  region.
    """

    crop_physical_volume = [48, 48, 48]  # [38.4, 38.4, 38.4]  # unit mm
    crop_spacing = [img.GetSpacing()[idx] for idx in range(3)]
    crop_size = [crop_physical_volume[idx] // crop_spacing[idx]
                 for idx in range(3)]

    models = load_models(model_dir, gpu_id)

    palate_mask = None
    if is_world_coordinate_valid(palate_anchor):
        left_voxel_coord = np.array(
            img.TransformPhysicalPointToIndex(palate_anchor))
        if is_voxel_coordinate_valid(left_voxel_coord, img.GetSize()):
            image_spacing = img.GetSpacing()
            voxel_offset = np.array(
                [45 * 0.3 / image_spacing[0], -40 * 0.3 / image_spacing[1], 0])
            cropped_center = img.TransformContinuousIndexToPhysicalPoint(
                left_voxel_coord + voxel_offset)
            cropped_palate_image = crop_image(
                img, cropped_center, crop_size, crop_spacing, 'LINEAR')
            _, palate_mask = segmentation_internal(
                cropped_palate_image, models, gpu_id)

    return palate_mask


def dental_refine_chin_region(img: sitk.Image, seg: sitk.Image, model_dir, gpu_id, chin_anchor):
    """
    This function aims to refine the segmentation results in the chin  region.
    """
    crop_physical_volume = [48, 38.4, 38.4]  # [38.4, 38.4, 38.4]  # unit mm
    crop_spacing = [img.GetSpacing()[idx] for idx in range(3)]
    crop_size = [crop_physical_volume[idx] // crop_spacing[idx]
                 for idx in range(3)]

    models = load_models(model_dir, gpu_id)

    chin_mask = None
    if is_world_coordinate_valid(chin_anchor):
        left_voxel_coord = np.array(
            img.TransformPhysicalPointToIndex(chin_anchor))
        if is_voxel_coordinate_valid(left_voxel_coord, img.GetSize()):
            image_spacing = img.GetSpacing()
            voxel_offset = np.array(
                [0 * 0.3 / image_spacing[0], 0 * 0.3 / image_spacing[1], 0])
            cropped_center = img.TransformContinuousIndexToPhysicalPoint(
                left_voxel_coord + voxel_offset)
            cropped_chin_image = crop_image(
                img, cropped_center, crop_size, crop_spacing, 'LINEAR')
            _, chin_mask = segmentation_internal(
                cropped_chin_image, models, gpu_id)

    return chin_mask


def segmentation_bone(img, seg_model_parent_dir, gpu_id):

    seg_global_model_dir = os.path.join(
        seg_model_parent_dir, 'model_0429_2020')
    models = load_models(seg_global_model_dir, gpu_id)
    _, mask = segmentation_internal(img, models, 0)
    return mask


def segmentation_teeth(img, seg_model_parent_dir, gpu_id):
    seg_teeth_model_folder = os.path.join(
        seg_model_parent_dir, 'model_0803_2020_dental')
    models = load_models(seg_teeth_model_folder, gpu_id)
    _, mask = segmentation_internal(img, models, 0)
    return mask


def refine_segmentation_bone(img, seg, lmk, seg_model_parent_dir, gpu_id):

    seg_local_model4_folder = os.path.join(seg_model_parent_dir, 'model_whole')
    seg = refine_whole_region(
        img, seg, seg_local_model4_folder, gpu_id, lmk['A'])

    seg_local_model_COR_folder = os.path.join(
        seg_model_parent_dir, 'model_COR')
    seg = refine_facial_region(
        img, seg, seg_local_model_COR_folder, gpu_id, lmk['COR-L'], lmk['COR-R'])

    return seg


def main(argv):

    long_description = 'Inference interface for segmentation.'

    parser = argparse.ArgumentParser(description=long_description)
    parser.add_argument('-i', '--input',
                        help='input dicom dir or volumetric data')

    parser.add_argument('-o', '--output-dir',
                        help='output dir to save masks')

    parser.add_argument('--landmark',
                        help='landmark used for refining the segmentation')

    parser.add_argument('--checkpoint-dir',
                        help='segmentation model dir')

    args, unknown_pars = parser.parse_known_args(argv)

    img_name = os.path.basename(args.input)

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # load image
    if os.path.isdir(args.input):

        series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(args.input)
        if not series_IDs:
            raise ValueError("ERROR: given directory \"" + args.input +
                             "\" does not contain a DICOM series.")

        if len(series_IDs) > 1:
            raise ValueError("ERROR: given directory \"" + args.input +
                             "\" contains multiple DICOM series.")

        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
            args.input, series_IDs[0])

        series_reader = sitk.ImageSeriesReader()
        series_reader.SetFileNames(series_file_names)

        # Configure the reader to load all of the DICOM tags (public + private):
        # By default tags are not loaded (saves time). If tags are loaded, the private tags are not loaded.
        # We explicitly configure the reader to load tags, including the private ones.

        series_reader.MetaDataDictionaryArrayUpdateOn()
        series_reader.LoadPrivateTagsOn()
        series_reader.SetOutputPixelType(sitk.sitkFloat32)
        img = series_reader.Execute()

    else:
        img = sitk.ReadImage(args.input, sitk.sitkFloat32)

    with open(args.landmark, 'r') as f:
        lmk = list(csv.reader(f))
        lmk = {x[0]: x[1:] for x in lmk}

    seg_model_parent_dir = os.path.join(args.checkpoint_dir, 'segmentation')
    seg = segmentation_bone(img, seg_model_parent_dir, 0)
    seg = refine_segmentation_bone(img, seg, lmk, seg_model_parent_dir, 0)
    seg_teeth = segmentation_teeth(img, seg_model_parent_dir, 0)

    sitk.WriteImage(seg, os.path.join(args.output_dir, 'seg-'+img_name))
    sitk.WriteImage(seg_teeth, os.path.join(
        args.output_dir, 'seg-teeth-'+img_name))


if __name__ == '__main__':
    main(sys.argv[1:])
