import csv
import argparse
import os
import sys
import time

import torch
import SimpleITK as sitk

from . import detection3d, segmentation3d, util


def main(argv):

    begin = time.perf_counter()

    long_description = 'Inference interface for landmark detection.'
    checkpoint_dir = os.environ['SkullEngineCheckpointDir'] if 'SkullEngineCheckpointDir' in os.environ else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'checkpoints')

    parser = argparse.ArgumentParser(description=long_description)
    parser.add_argument('-i', '--input', nargs='+',
                        help='input dicom dir or volumetric data')

    parser.add_argument('-o', '--output-dir',
                        help='output dir to save binary masks')

    parser.add_argument('--lmk', action='store_true',
                        help='Generate landmarks.')

    parser.add_argument('--seg', action='store_true',
                        help='Generate bone segmentation without later refinement.')

    parser.add_argument('--seg-refine', action='store_true',
                        help='Generate refined bone segmentation.')

    parser.add_argument('--checkpoint-dir',
                        default=checkpoint_dir,
                        help='detection model dir')

    parser.add_argument('--all', action='store_true',
                        help='Generate everything.')

    args, unknown_pars = parser.parse_known_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)

    num_gpu = torch.cuda.device_count()
    load_lmk = args.all or args.seg_refine or args.lmk
    load_seg = args.all or args.seg_refine or args.seg
    patch_refine = args.seg_refine or args.all
    
    if load_lmk:
        detection3d.model_init(gpu_id=0)

    if load_seg:
        segmentation3d.model_init(gpu_id=num_gpu-1, patch_refine=patch_refine)

    for img_file in args.input:

        img_name = os.path.basename(img_file)

        # load image
        img = util.read_image(img_file)

        # landmarks
        if args.lmk or args.all:

            lmk = detection3d.infer(img)
            with open(os.path.join(
                    args.output_dir, f'lmk-{img_name.removesuffix('.nii.gz')}.csv'), 'w') as f:

                csv.writer(f).writerows(zip(lmk.keys(), *zip(*lmk.values())))

        # segmentation
        if args.seg or args.seg_refine or args.all:

            seg = segmentation3d.infer_coarse(img)
            sitk.WriteImage(seg, os.path.join(
                args.output_dir, 'seg-no-refine-'+img_name.removesuffix('.nii.gz')+'.nii.gz'))

            if patch_refine:
                seg = segmentation3d.infer_refine(img, seg, lmk)
                sitk.WriteImage(seg, os.path.join(
                    args.output_dir, 'seg-'+img_name.removesuffix('.nii.gz')+'.nii.gz'))


    print('Process Time: {}'.format(time.perf_counter() - begin))


if __name__ == '__main__':
    main(sys.argv[1:])
