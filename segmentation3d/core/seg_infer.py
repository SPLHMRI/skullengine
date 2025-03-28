from collections import OrderedDict
import copy
import glob
import importlib
import torch.nn as nn
import os
import pandas as pd
import SimpleITK as sitk
import time
import torch
import numpy as np
from easydict import EasyDict as edict

from ..utils.dicom_helper import read_dicom_series, write_dicom_series, dicom_tags_dict
from ..utils.file_io import load_config, readlines
from ..utils.model_io import get_checkpoint_folder
from ..utils.image_tools import resample, convert_image_to_tensor, convert_tensor_to_image, \
    add_image_region, image_partition_by_fixed_size, resample_spacing, add_image_value, pick_largest_connected_component, \
    remove_small_connected_component, get_bounding_box
from ..utils.normalizer import FixedNormalizer, AdaptiveNormalizer


def read_test_txt(txt_file):
    """ read single-modality txt file
    :param txt_file: image list txt file path
    :return: a list of image path list, list of image case names
    """
    lines = readlines(txt_file)
    case_num = int(lines[0])

    if len(lines) - 1 != case_num:
        raise ValueError('case num do not equal path num!')

    file_name_list, file_path_list = [], []
    for i in range(case_num):
        im_msg = lines[1 + i]
        im_msg = im_msg.strip().split()
        im_name = im_msg[0]
        im_path = im_msg[1]
        if not os.path.isfile(im_path):
            raise ValueError('image not exist: {}'.format(im_path))
        file_name_list.append(im_name)
        file_path_list.append(im_path)

    return file_name_list, file_path_list


def read_test_csv(imlist_file, mode='test'):
    """ read single-modality csv file
    :param imlist_file: image list file path
    :return: a list of image path list, list of segmentation paths
    """
    images_df = pd.read_csv(imlist_file)
    image_name_list = images_df['image_name'].tolist()
    image_path_list = images_df['image_path'].tolist()

    if mode == 'test':
        return image_name_list, image_path_list

    elif mode == 'train' or mode == 'validation':
        mask_path_list = images_df['mask_path'].tolist()
        return image_path_list, mask_path_list

    else:
        raise ValueError('Unsupported mode type.')


def read_test_folder(folder_path, is_dicom_folder):
    """ read single-modality input folder
    :param folder_path: image file folder path
    :param is_dicom_folder: whether it is a dicom folder
    :return: a list of image path list, list of image case names
    """
    if is_dicom_folder:
        _, im_name = os.path.split(folder_path)
        file_name_list = [im_name]
        file_path_list = [folder_path]

    else:
        suffix = ['.mhd', '.nii', '.hdr', '.nii.gz', '.mha', '.image3d']
        file = []
        for suf in suffix:
            file += glob.glob(os.path.join(folder_path, '*' + suf))

        file_name_list, file_path_list = [], []
        for im_pth in sorted(file):
            _, im_name = os.path.split(im_pth)
            for suf in suffix:
                idx = im_name.find(suf)
                if idx != -1:
                    im_name = im_name[:idx]
                    break
            file_name_list.append(im_name)
            file_path_list.append(im_pth)

    return file_name_list, file_path_list


def check_input(input_path):
    """
    Check the input path and return the file list for testing.
    """
    # load test images
    is_dicom_folder = False
    if os.path.isfile(input_path):
        if input_path.endswith('.txt'):
            file_name_list, file_path_list = read_test_txt(input_path)
        elif input_path.endswith('.csv'):
            file_name_list, file_path_list = read_test_csv(input_path)
        else:
            if input_path.endswith('.mhd') or input_path.endswith('.mha') or input_path.endswith('.nii.gz') or \
                    input_path.endswith('.nii') or input_path.endswith('.hdr') or input_path.endswith('.image3d'):
                im_name = os.path.basename(input_path)
                file_name_list = [im_name]
                file_path_list = [input_path]

            else:
                raise ValueError('Unsupported input path.')

    else:
        if os.path.isdir(input_path):
            # test if it is a DICOM folder
            file_names = glob.glob(os.path.join(input_path, '*'))
            if len(file_names) > 0:
                is_dicom_folder = True

            file_name_list, file_path_list = read_test_folder(
                input_path, is_dicom_folder)
            if len(file_name_list) == 0:
                raise ValueError('Empty test folder!')

        else:
            raise ValueError('The file {} does not exist.'.format(input_path))

    return is_dicom_folder, file_name_list, file_path_list


def load_single_model(model_folder, gpu_id):
    """ load single segmentation model from folder
    :param model_folder:    the folder containing the segmentation model
    :param gpu_id:          the gpu device id to run the segmentation model
    :return: a dictionary containing the model and inference parameters
    """
    assert os.path.isdir(
        model_folder), 'Model folder does not exist: {}'.format(model_folder)

    model = edict()

    # load model state
    latest_checkpoint_dir = get_checkpoint_folder(
        os.path.join(model_folder, 'checkpoints'), -1)
    chk_file = os.path.join(latest_checkpoint_dir, 'params.pth')

    if gpu_id >= 0:
        # load network module
        state = torch.load(chk_file)
        net_module = importlib.import_module(
            f"skullengine.segmentation3d.network.{state['net']}")
        net = net_module.SegmentationNet(
            state['in_channels'], state['out_channels'])
        net = nn.parallel.DataParallel(net)
        net.load_state_dict(state['state_dict'])
        net = net.module.to(f"cuda:{gpu_id}")
        net.eval()

    else:
        state = torch.load(chk_file, map_location='cpu')
        net_module = importlib.import_module(
            f"skullengine.segmentation3d.network.{state['net']}")
        net = net_module.SegmentationNet(
            state['in_channels'], state['out_channels'])

        state_dict = state['state_dict']
        is_trained_on_cpu = True
        for k, _ in state_dict.items():
            if 'module' in k:
                is_trained_on_cpu = False
                break

        if not is_trained_on_cpu:
            state_dict_cpu = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove the string `module.`
                state_dict_cpu[name] = v
            state_dict = state_dict_cpu

        net.load_state_dict(state_dict)
        net.eval()

    model.net = net
    model.device = gpu_id
    model.spacing, model.max_stride, model.interpolation = state[
        'spacing'], state['max_stride'], state['interpolation']
    model.in_channels, model.out_channels = state['in_channels'], state['out_channels']

    model.crop_normalizers = []
    for crop_normalizer in state['crop_normalizers']:
        if crop_normalizer['type'] == 0:
            mean, stddev, clip = crop_normalizer['mean'], crop_normalizer['stddev'], crop_normalizer['clip']
            model.crop_normalizers.append(FixedNormalizer(mean, stddev, clip))

        elif crop_normalizer['type'] == 1:
            clip_sigma = crop_normalizer['clip_sigma']
            model.crop_normalizers.append(AdaptiveNormalizer(clip_sigma))

        else:
            raise ValueError('Unsupported normalization type.')

    return model


def load_models(model_folder, gpu_id):
    """ load segmentation model from folder
    :param model_folder:    the folder containing the segmentation model
    :param gpu_id:          the gpu device id to run the segmentation model
    :return: a dictionary containing the model and inference parameters
    """
    assert os.path.isdir(
        model_folder), 'Model folder does not exist: {}'.format(model_folder)

    # load inference config file
    infer_cfg = load_config(os.path.join(model_folder, 'infer_config.py'))
    models = edict()
    models.infer_cfg = infer_cfg

    # load coarse model if it is enabled
    if models.infer_cfg.general.single_scale == 'coarse':
        coarse_model_folder = os.path.join(
            model_folder, models.infer_cfg.coarse.model_name)
        coarse_model = load_single_model(coarse_model_folder, gpu_id)
        models.coarse_model = coarse_model
        models.fine_model = None

    elif models.infer_cfg.general.single_scale == 'fine':
        fine_model_folder = os.path.join(
            model_folder, models.infer_cfg.fine.model_name)
        fine_model = load_single_model(fine_model_folder, gpu_id)
        models.fine_model = fine_model
        models.coarse_model = None

    elif models.infer_cfg.general.single_scale == 'DISABLE':
        coarse_model_folder = os.path.join(
            model_folder, models.infer_cfg.coarse.model_name)
        coarse_model = load_single_model(coarse_model_folder, gpu_id)
        models.coarse_model = coarse_model

        fine_model_folder = os.path.join(
            model_folder, models.infer_cfg.fine.model_name)
        fine_model = load_single_model(fine_model_folder, gpu_id)
        models.fine_model = fine_model

    else:
        raise ValueError('Unsupported single scale type!')

    return models


def segmentation_voi(model, iso_image, start_voxel, end_voxel):
    """ Segment the volume of interest
    :param model:           the loaded segmentation model.
    :param iso_image:       the image volume that has the same spacing with the model's resampling spacing.
    :param start_voxel:     the start voxel of the volume of interest (inclusive).
    :param end_voxel:       the end voxel of the volume of interest (exclusive).
    :param use_gpu:         whether to use gpu or not, bool type.
    :return:
      mean_prob_maps:        the mean probability maps of all classes
      std_maps:              the standard deviation maps of all classes
    """
    assert isinstance(iso_image, sitk.Image)

    roi_image = iso_image[start_voxel[0]:end_voxel[0],
                          start_voxel[1]:end_voxel[1], start_voxel[2]:end_voxel[2]]

    if model['crop_normalizers'] is not None:
        roi_image = model.crop_normalizers[0](roi_image)

    roi_image_tensor = convert_image_to_tensor(roi_image).unsqueeze(0)
    if model.device >= 0:
        roi_image_tensor = roi_image_tensor.cuda(model.device)

    with torch.no_grad():
        probs = model['net'](roi_image_tensor)
        probs = torch.unsqueeze(probs, 0)
        probs = torch.cat((probs, torch.unsqueeze(
            model['net'](roi_image_tensor), 0)), 0)
        mean_probs = torch.mean(probs, 0)

    num_classes = model['out_channels']
    assert num_classes == mean_probs.shape[1]

    # return the average probability map
    mean_prob_maps = []
    for idx in range(num_classes):
        mean_prob = convert_tensor_to_image(
            mean_probs[0][idx].data, dtype=float)
        mean_prob.CopyInformation(roi_image)
        mean_prob_maps.append(mean_prob)

    return mean_prob_maps


def segmentation_volume(model, cfg, image, bbox_start_voxel, bbox_end_voxel):
    """ Segment the volume
    :param model:             the loaded segmentation model.
    :param image:             the image volume.
    :param bbox_start_voxel:  the start voxel of the bounding box (inclusive).
    :param bbox_end_voxel:    the end voxel of the bounding box (exclusive).
    :param use_gpu:           whether to use gpu or not, bool type.
    :return:
      mean_prob_maps:         the mean probability maps of all classes
      std_maps:               the standard deviation maps of all classes
    """
    assert isinstance(image, sitk.Image)

    model_spacing = copy.deepcopy(model['spacing'])

    iso_image = resample_spacing(
        image, model_spacing, model['max_stride'], model['interpolation'])

    num_classes = model['out_channels']
    iso_mean_probs = []
    for idx in range(num_classes):
        iso_mean_prob = sitk.Image(iso_image.GetSize(), sitk.sitkFloat32)
        iso_mean_prob.CopyInformation(iso_image)
        iso_mean_probs.append(iso_mean_prob)

    partition_type = cfg.partition_type
    if partition_type == 'DISABLE':
        start_voxels = [[0, 0, 0]]
        end_voxels = [[int(iso_image.GetSize()[idx]) for idx in range(3)]]

    elif partition_type == 'SIZE':
        partition_stride = copy.deepcopy(cfg.partition_stride)
        partition_size = copy.deepcopy(cfg.partition_size)
        max_stride = model['max_stride']

        # convert bounding box to the iso image frame
        if bbox_start_voxel is not None and bbox_end_voxel is not None:
            bbox_start_voxel_double = [
                float(bbox_start_voxel[idx]) for idx in range(3)]
            bbox_start_world = image.TransformContinuousIndexToPhysicalPoint(
                bbox_start_voxel_double)
            bbox_start_voxel = iso_image.TransformPhysicalPointToIndex(
                bbox_start_world)

            bbox_end_voxel_double = [
                float(bbox_end_voxel[idx]) for idx in range(3)]
            bbox_end_world = image.TransformContinuousIndexToPhysicalPoint(
                bbox_end_voxel_double)
            bbox_end_voxel = iso_image.TransformPhysicalPointToIndex(
                bbox_end_world)

            bbox_start_voxel = [max(0, bbox_start_voxel[idx])
                                for idx in range(3)]
            bbox_end_voxel = [min(bbox_end_voxel[idx], iso_image.GetSize()[
                                  idx]) for idx in range(3)]
        else:
            bbox_start_voxel, bbox_end_voxel = [0, 0, 0], [
                iso_image.GetSize()[idx] for idx in range(3)]

        start_voxels, end_voxels = image_partition_by_fixed_size(
            iso_image, bbox_start_voxel, bbox_end_voxel, partition_size, partition_stride, max_stride
        )

    else:
        raise ValueError('Unsupported partition type!')

    iso_partition_overlap_count = sitk.Image(
        iso_image.GetSize(), sitk.sitkFloat32)
    iso_partition_overlap_count.CopyInformation(iso_image)
    for idx in range(len(start_voxels)):
        start_voxel, end_voxel = start_voxels[idx], end_voxels[idx]

        voi_mean_probs = segmentation_voi(
            model, iso_image, start_voxel, end_voxel)
        for idy in range(num_classes):
            iso_mean_probs[idy] = add_image_region(
                iso_mean_probs[idy], start_voxel, end_voxel, voi_mean_probs[idy])

        iso_partition_overlap_count = add_image_value(
            iso_partition_overlap_count, start_voxel, end_voxel, 1.0)
        print('{:0.2f}%'.format((idx + 1) / len(start_voxels) * 100))

    iso_partition_overlap_count = sitk.Cast(
        1.0 / iso_partition_overlap_count, sitk.sitkFloat32)
    for idx in range(num_classes):
        iso_mean_probs[idx] = iso_mean_probs[idx] * iso_partition_overlap_count

    # resample to the original spacing
    mean_probs = []
    for idx in range(num_classes):
        padding_value = 1.0 if idx == 0 else 0.0
        mean_probs.append(
            resample(iso_mean_probs[idx], image, 'LINEAR', padding_value))

    # get segmentation mask from the mean_probability maps
    mean_probs_tensor = convert_image_to_tensor(mean_probs)
    _, mask = mean_probs_tensor.max(0)
    mask = convert_tensor_to_image(mask, dtype=np.int8)
    mask.CopyInformation(image)

    # pick the largest component
    if cfg.pick_largest_cc:
        mask = pick_largest_connected_component(
            mask, list(range(1, num_classes)))

    # remove small connected component
    if cfg.remove_small_cc > 0:
        threshold = cfg.remove_small_cc
        mask = remove_small_connected_component(
            mask, list(range(1, num_classes)), threshold)

    return mean_probs, mask


def segmentation_internal(image, models):
    """
    volumetric image segmentation engine
    :param input_path:          The path of text file, a single image file or a root dir with all image files
    :param model_folder:        The path of trained model
    :param output_folder:       The path of out folder
    :param gpu_id:              Which gpu to use, by default, 0
    :param return_mask:         Whether to return mask
    :param save_mask:           Whether to save mask
    :param save_image:          Whether to save original image
    :param save_prob:           Whether to save all probability maps
    :return: None
    """

    if models['infer_cfg'].general.single_scale == 'coarse':
        mean_probs, mask = segmentation_volume(
            models['coarse_model'], models['infer_cfg'].coarse, image, None, None)

    elif models['infer_cfg'].general.single_scale == 'fine':
        mean_probs, mask = segmentation_volume(
            models['fine_model'], models['infer_cfg'].fine, image, None, None)

    elif models['infer_cfg'].general.single_scale == 'DISABLE':
        print('Coarse segmentation: ')
        _, mask = segmentation_volume(
            models['coarse_model'], models['infer_cfg'].coarse, image, None, None)

        start_voxel, end_voxel = get_bounding_box(mask, None)

        # compute bbox ratio
        bbox_ratio = 100
        for idx in range(3):
            bbox_ratio *= (end_voxel[idx] -
                           start_voxel[idx]) / mask.GetSize()[idx]

        print('Fine segmentation (bbox ratio: {:.2f}%): '.format(bbox_ratio))
        mean_probs, mask = segmentation_volume(
            models['fine_model'], models['infer_cfg'].fine, image, start_voxel, end_voxel)

    else:
        raise ValueError('Unsupported scale type!')

    return mean_probs, mask
