import numpy as np
import SimpleITK as sitk
import os

from .dicom_helper import read_dicom_series, write_dicom_series, write_binary_dicom_series, \
  dicom_tags_dict


def test_save_two_masks(input_path, output_path):
  if not os.path.isdir(output_path):
    os.mkdir(output_path)

  # read mha image
  case_list=os.listdir(input_path)
  midface_label, mandible_label, softtissue_label, spine_label = 1, 2, 3, 4
  for case in case_list:
    current_path = os.path.join(input_path, case)
    if not os.path.isdir(current_path):
        continue
    org_path=  os.path.join(input_path, case, 'cropped_org.mha')
    org = sitk.ReadImage(org_path, sitk.sitkInt16)
    
    seg_path=os.path.join(input_path, case,'cropped_seg.mha')
    seg = sitk.ReadImage(seg_path, sitk.sitkInt16)
    masks = sitk.GetArrayFromImage(seg)
    masks[masks == softtissue_label] = 0
    masks[masks == spine_label] = 0
    masks_twolable = sitk.GetImageFromArray(masks)
    masks_twolable_path= os.path.join(output_path, case)
    print('Path: {}'.format(masks_twolable_path))
    if not os.path.isdir(masks_twolable_path) :
      print('Make Folder: {}'.format(masks_twolable_path))
      os.mkdir(masks_twolable_path)
    org_output_path = os.path.join(masks_twolable_path,'cropped_org.mha')
    masks_twolable_path=os.path.join(masks_twolable_path,'cropped_seg.mha')
   
    sitk.WriteImage(org, org_output_path, True)
    sitk.WriteImage(masks_twolable, masks_twolable_path, True)

    # # save mha to binary dicom series
    # tags = dicom_tags_dict()
    # dicom_save_folder = '/home/qinliu/debug/seg_dicom_maxilla'
    # write_binary_dicom_series(seg, dicom_save_folder, in_label=1, out_label=100, tags=tags)

    # dicom_save_folder = '/home/qinliu/debug/seg_dicom_mandible'
    # write_binary_dicom_series(seg, dicom_save_folder, in_label=2, out_label=100, tags=tags)


# def test_merge_mask():

#   """
#   Merge two masks.
#   mask1: a mask which includes midface, mandible and soft tissue
#   mask2: a mask which includes the upper teeth and the lower teeth. The upper teeth is a part of midface, and the lower teeth
#          is a part of mandible.
#   merged_mask: the mask merged by mask1 and mask2.
#   """
#   # read the first mask
#   mask_path_1 = '/home/qinliu/debug/seg1_dicom'
#   mask1 = read_dicom_series(mask_path_1)

#   # read the second mask
#   mask_path_2 = '/home/qinliu/debug/seg2_dicom'
#   mask2 = read_dicom_series(mask_path_2)

#   # the two masks should have the same size
#   size_mask_1, size_mask_2 = mask1.GetSize(), mask2.GetSize()
#   assert size_mask_1[0] == size_mask_2[0]
#   assert size_mask_1[1] == size_mask_2[1]
#   assert size_mask_1[2] == size_mask_2[2]

#   # merge two masks
#   mask1_npy = sitk.GetArrayFromImage(mask1)
#   mask2_npy = sitk.GetArrayFromImage(mask2)

#   upper_teeth_label, lower_teeth_label = 1, 2
#   mask1_npy[mask2_npy == upper_teeth_label] = upper_teeth_label
#   mask1_npy[mask2_npy == lower_teeth_label] = lower_teeth_label

#   merged_mask = sitk.GetImageFromArray(mask1_npy)
#   merged_mask.CopyInformation(mask1)
#   merged_mask_path = '/home/qinliu/debug/merged_seg.mha'
#   sitk.WriteImage(merged_mask, merged_mask_path, True)


if __name__ == '__main__':

  test_save_two_masks('/data/deng/New_GT/Crop_before_select_mask/crop_ION_R_pre_p','/data/deng/New_GT/crop_ION_R_p')

  # test_save_binary_dicom_series()
  #
  # test_merge_mask()
