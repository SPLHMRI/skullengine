import os
import numpy as np
import SimpleITK as sitk

from .dicom_helper import read_dicom_series, write_dicom_series, write_binary_dicom_series, \
  dicom_tags_dict



def test_merge_mask(in_path,out_path):
  """
  Merge two masks.
  mask1: a mask which includes midface, mandible and soft tissue
  mask2: a mask which includes the upper teeth and the lower teeth. The upper teeth is a part of midface, and the lower teeth
         is a part of mandible.
  merged_mask: the mask merged by mask1 and mask2.
  """
  # read the first mask
  print('Read Original Dicom')
  org_path = os.path.join(in_path, 'img')
  org = read_dicom_series(org_path)

  print('Read Mask Dicom')
  mask_path_1 = os.path.join(in_path, 'maxi-new') #'/home/qinliu/debug/seg1_dicom'
  mask1 = read_dicom_series(mask_path_1)

  # read the second mask
  mask_path_2 = os.path.join(in_path, 'mand-new') #'/home/qinliu/debug/seg2_dicom'
  mask2 = read_dicom_series(mask_path_2)

  print('Finish Reading')
  # the two masks should have the same size
  size_mask_1, size_mask_2 = mask1.GetSize(), mask2.GetSize()
  assert size_mask_1[0] == size_mask_2[0]
  assert size_mask_1[1] == size_mask_2[1]
  assert size_mask_1[2] == size_mask_2[2]

  # merge two masks
  mask1_npy = sitk.GetArrayFromImage(mask1)
  mask2_npy = sitk.GetArrayFromImage(mask2)

  maxi_label, mand_label = 1, 2
  mask1_npy[mask1_npy == 100] = maxi_label
  mask1_npy[mask2_npy == 100] = mand_label

  print('Write Files')
  img_path = os.path.join(out_path, 'img.nii.gz') 
  sitk.WriteImage(org, img_path, True)


  merged_mask = sitk.GetImageFromArray(mask1_npy)
  merged_mask.CopyInformation(mask1)
  merged_mask_path = os.path.join(out_path, 'seg_filled.nii.gz') #'/home/qinliu/debug/merged_seg.mha'
  sitk.WriteImage(merged_mask, merged_mask_path, True)

def main(input_path, output_path):
  case_list = os.listdir(input_path)
  for case in case_list:
    print('Current Folder: {}'.format(case))
    case_path = os.path.join(input_path,case)
    result_path = os.path.join(output_path, case)
    if not os.path.isdir(result_path) :
      print('Make Folder: {}'.format(result_path))
      os.mkdir(result_path)
    test_merge_mask(case_path, result_path)

if __name__ == '__main__':

  # test_save_dicom_series()

  # test_save_binary_dicom_series()
  #
  # test_merge_mask()
  main('/data/deng/New_GT/data/Josh', '/data/deng/New_GT/data/test')