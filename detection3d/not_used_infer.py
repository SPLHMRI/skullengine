#!/usr/bin/env python
import argparse
import SimpleITK as sitk
import os
import sys

from .utils.landmark_utils import is_voxel_coordinate_valid, is_world_coordinate_valid
from .utils.image_tools import crop_image
from .core.lmk_det_infer import detection_single_image, load_det_model
from .utils.landmark_utils import merge_landmark_dataframes


def lmk_detection_bone(image, image_name, model_folder, gpu_id, output_csv_file):
    
    """
    :param input:
    :param model_folder:
    :param gpu_id:
    :param output_csv_file:
    :return:
    """
    
    assert isinstance(image, sitk.Image)
    assert output_csv_file.endswith('.csv')

    landmark_dataframes = []

    # detect batch 1
    print('Start detecting the non-teeth landmarks ...')
    batch_1_model = load_det_model(
        os.path.join(model_folder, 'batch_1'), gpu_id)
    landmark_batch_1 = detection_single_image(
        image, image_name, batch_1_model, gpu_id, None, None)
    del batch_1_model
    landmark_dataframes.append(landmark_batch_1)

    # detect batch 2
    batch_2_model = load_det_model(os.path.join(model_folder, 'batch_2'))
    landmark_batch_2 = detection_single_image(
        image, image_name, batch_2_model, gpu_id, None, None)
    del batch_2_model
    landmark_dataframes.append(landmark_batch_2)

    # detect batch 3
    batch_3_model = load_det_model(os.path.join(model_folder, 'batch_3'))
    landmark_batch_3 = detection_single_image(
        image, image_name, batch_3_model, gpu_id, None, None)
    del batch_3_model
    landmark_dataframes.append(landmark_batch_3)

    # crop the teeth region according to landmark 'L0'
    print('Start detecting the teeth landmarks')
    l0 = landmark_batch_3[landmark_batch_3['name'] == 'L0']
    world_coord_l0 = [l0['x'].values[0], l0['y'].values[0], l0['z'].values[0]]
    if is_world_coordinate_valid(world_coord_l0):
        voxel_coord_l0 = image.TransformPhysicalPointToIndex(world_coord_l0)
        if is_voxel_coordinate_valid(voxel_coord_l0, image.GetSize()):
            image_spacing, image_size = image.GetSpacing(), image.GetSize()
            crop_spacing, crop_size = [0.8, 0.8, 0.8], [128, 96, 96]
            resample_ratio = [crop_spacing[idx] /
                              image_spacing[idx] for idx in range(3)]

            offset = [-int(64 * resample_ratio[0]), -int(16 *
                                                         resample_ratio[1]), -int(48 * resample_ratio[2])]
            left_bottom_voxel = [voxel_coord_l0[idx] +
                                 offset[idx] for idx in range(3)]
            right_top_voxel = [left_bottom_voxel[idx] +
                               int(crop_size[idx] * resample_ratio[idx]) - 1 for idx in range(3)]
            for idx in range(3):
                left_bottom_voxel[idx] = max(0, left_bottom_voxel[idx])
                right_top_voxel[idx] = min(
                    image_size[idx] - 1, right_top_voxel[idx])

            crop_voxel_center = [
                (left_bottom_voxel[idx] + right_top_voxel[idx]) // 2 for idx in range(3)]
            crop_world_center = image.TransformContinuousIndexToPhysicalPoint(
                crop_voxel_center)
            cropped_image = crop_image(
                image, crop_world_center, crop_size, crop_spacing, 'LINEAR')

            # detect batch 4-lower teeth batch 1
            batch_4_lower_1_model = load_det_model(
                os.path.join(model_folder, 'batch_4_lower_1'))
            landmark_batch_4_lower_1 = detection_single_image(
                cropped_image, image_name, batch_4_lower_1_model, gpu_id, None, None)
            del batch_4_lower_1_model
            landmark_dataframes.append(landmark_batch_4_lower_1)

            # detect batch 4-lower teeth batch 2
            batch_4_lower_2_model = load_det_model(
                os.path.join(model_folder, 'batch_4_lower_2'))
            landmark_batch_4_lower_2 = detection_single_image(
                cropped_image, image_name, batch_4_lower_2_model, gpu_id, None, None)
            del batch_4_lower_2_model
            landmark_dataframes.append(landmark_batch_4_lower_2)

            # detect batch 4-upper teeth batch 1
            batch_4_upper_1_model = load_det_model(
                os.path.join(model_folder, 'batch_4_upper_1'))
            landmark_batch_4_upper_1 = detection_single_image(
                cropped_image, image_name, batch_4_upper_1_model, gpu_id, None, None)
            del batch_4_upper_1_model
            landmark_dataframes.append(landmark_batch_4_upper_1)

            # detect batch 4-upper teeth batch 2
            batch_4_upper_2_model = load_det_model(
                os.path.join(model_folder, 'batch_4_upper_2'))
            landmark_batch_4_upper_2 = detection_single_image(
                cropped_image, image_name, batch_4_upper_2_model, gpu_id, None, None)
            del batch_4_upper_2_model
            landmark_dataframes.append(landmark_batch_4_upper_2)

    merged_landmark_dataframes = merge_landmark_dataframes(landmark_dataframes)
    merged_landmark_dataframes.sort_values(by=['name'], inplace=True)
    merged_landmark_dataframes.to_csv(output_csv_file, index=False)

    return merged_landmark_dataframes


def main(argv):

    long_description = 'Inference interface for landmark detection.'

    parser = argparse.ArgumentParser(description=long_description)
    parser.add_argument('-i', '--input',
                        help='input dicom dir or volumetric data')
    parser.add_argument('-o', '--output-dir',
                        help='output dir to save binary masks')
    parser.add_argument('--checkpoint-dir',
                        help='detection model dir')

    args, unknown_pars = parser.parse_known_args(argv)

    img_name = os.path.basename(args.input)
    output_csv_file = os.path.join(args.output_dir, f'{img_name}.csv')

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
        print(img.GetOutputPixelType())
        # img = sitk.Cast(img, sitk.sitkFloat32)
        
    else:
        img = sitk.ReadImage(args.input, sitk.sitkFloat32)

    det_model_dir = os.path.join(args.checkpoint_dir, 'detection', 'model_0514_2020')
    
    # Note: landmarks L6CF-R, L6CF-L, L6DC-R, L6DC-L won't be detected because they are too close.

    landmark_df = lmk_detection_bone(
        img, img_name, det_model_dir, 0, output_csv_file)


if __name__ == '__main__':
    main(sys.argv[1:])
