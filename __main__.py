#!/usr/bin/env python
import argparse
import numpy as np
import SimpleITK as sitk
import os
import sys
import time

from .detection3d.infer import lmk_detection_bone
from .segmentation3d.infer import segmentation_bone, segmentation_teeth, refine_segmentation_bone


def main(argv):

    begin = time.perf_counter()

    long_description = 'Inference interface for landmark detection.'
    checkpoint_dir = os.environ['SkullEngineCheckpointDir'] if 'SkullEngineCheckpointDir' in os.environ else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'checkpoints')

    parser = argparse.ArgumentParser(description=long_description)
    parser.add_argument('-i', '--input',
                        help='input dicom dir or volumetric data')
    parser.add_argument('-o', '--output-dir',
                        help='output dir to save binary masks')
    parser.add_argument('--checkpoint-dir',
                        default=checkpoint_dir,
                        help='detection model dir')
    parser.add_argument('--cass', action='store_true')

    args, unknown_pars = parser.parse_known_args(argv)

    # Note: landmarks L6CF-R, L6CF-L, L6DC-R, L6DC-L won't be detected because they are too close.

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

    else:
        img = sitk.ReadImage(args.input, sitk.sitkFloat32)

    det_model_dir = os.path.join(
        args.checkpoint_dir, 'detection', 'model_0514_2020')
    lmk = lmk_detection_bone(img, img_name, det_model_dir, 0, os.path.join(args.output_dir, f'{img_name}.csv'))
    lmk = dict(zip(lmk.iloc[:,0].values, lmk.iloc[:,1:].values))

    seg_model_parent_dir = os.path.join(args.checkpoint_dir, 'segmentation')
    seg = segmentation_bone(img, seg_model_parent_dir, 0)
    seg = refine_segmentation_bone(img, seg, lmk, seg_model_parent_dir, 0)
    seg_teeth = segmentation_teeth(img, seg_model_parent_dir, 0)

    sitk.WriteImage(seg, os.path.join(args.output_dir, 'seg-'+img_name))
    sitk.WriteImage(seg_teeth, os.path.join(
        args.output_dir, 'seg-teeth-'+img_name))

    print('Process Time: {}'.format(time.perf_counter() - begin))


if __name__ == '__main__':
    main(sys.argv[1:])
