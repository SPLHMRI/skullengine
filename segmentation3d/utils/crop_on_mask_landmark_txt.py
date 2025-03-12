import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
from .image_tools import crop_image, resample_spacing, \
  pick_largest_connected_component, get_bounding_box

def get_anchor(lm_path):
  if lm_path.endswith('.csv'):
    landmark_all = pd.read_csv(lm_path,header=None)
    # landmark_all = pd.read_csv('/home/admin/test/data_v1_ldmk/case_17_cbct_patient.csv')
    # landmark_list = landmark_all['name'].tolist()
    # landmark = landmark_all[landmark_all['name']=='Me']
    # landmark = [landmark['x'].values[0], landmark['y'].values[0], landmark['z'].values[0]]
    print(landmark_all)
    landmark_list = landmark_all.iloc[:,0]#.tolist
    print(landmark_list)
    # landmark = landmark.all.iloc[landmark_all]
    # landmark = landmark_all[landmark_list=='Me']
    landmark = landmark_all[landmark_list=='Me']
    # landmark = [landmark[1].values[0], landmark[2].values[0], landmark[3].values[0]]
    if len(landmark)==0:
      return
    
    # print('{}landmark'.format(landmark))
    landmark = [landmark[1].values[0], landmark[2].values[0], landmark[3].values[0]]
  elif lm_path.endswith('.xlsx'):
    landmark_all = pd.read_excel(lm_path, engine='openpyxl')
    landmark_list = landmark_all['Landmark Name'].tolist()
    landmark_offset = landmark_all[landmark_all['Landmark Name']=='Landmark_Offset']
    landmark_offset = [landmark_offset['Original_X'].values[0], landmark_offset['Original_Y'].values[0], landmark_offset['Original_Z'].values[0]]
    landmark_temp = landmark_all[landmark_all['Landmark Name']=='Po-L']
    landmark_temp = [landmark_temp['Original_X'].values[0], landmark_temp['Original_Y'].values[0], landmark_temp['Original_Z'].values[0]]
    landmark = [ landmark_offset[i]+landmark_temp[i] for i in range(len(landmark_offset))]
    print('Coordinate: {}'.format(landmark))
  elif lm_path.endswith('.txt'):
    landmark_all = pd.read_csv(lm_path, lineterminator=';', names=['name','x','y','z'])
    landmark_list = landmark_all['name'].tolist()
    landmark = landmark_all[landmark_all['name']=='ION-R']
    landmark = [landmark['x'].values[0], landmark['y'].values[0], landmark['z'].values[0]]


  return landmark

# def test_copy_image():
#   seg_path = '/home/qinliu/debug/seg.mha'
#   seg = sitk.ReadImage(seg_path)

#   assert isinstance(seg, sitk.Image)

#   seg_empty = sitk.Image(seg.GetSize(), seg.GetPixelID())
#   seg_empty.CopyInformation(seg)

#   # crop from seg
#   cropping_center_voxel = [int(seg.GetSize()[idx] // 2) for idx in range(3)]
#   cropping_center_world = seg.TransformIndexToPhysicalPoint(cropping_center_voxel)
#   cropping_size = [128, 128, 128]
#   cropping_spacing = [1.0, 1.0, 1.0]
#   interp_method = 'NN'
#   seg_cropped = crop_image(seg, cropping_center_world, cropping_size, cropping_spacing, interp_method)

#   seg_cropped_path = '/home/qinliu/debug/seg_cropped.mha'
#   sitk.WriteImage(seg_cropped, seg_cropped_path)

#   # copy_image(seg_cropped, cropping_center_world, cropping_size, seg_empty)
#   # seg_copy_path = '/home/qinliu/debug/seg_empty_copy.mha'
#   # sitk.WriteImage(seg_empty, seg_copy_path)

#   seg_origin = seg.GetOrigin()
#   seg_empty_origin = list(map(int, seg_empty.GetOrigin()))
#   seg_cropped_size = list(map(int, seg_cropped.GetSize()))
#   seg_cropped_origin = list(map(int, seg_cropped.GetSize()))
#   seg_pasted = sitk.Paste(seg_empty, seg_cropped, seg_cropped_size, [0, 0, 0], [100, 100, 100])
#   seg_paste_path = '/home/qinliu/debug/seg_empty_paste.mha'
#   sitk.WriteImage(seg_pasted, seg_paste_path)


def test_resample_spacing():
  seg_path = '/home/qinliu/debug/org.mha'
  seg = sitk.ReadImage(seg_path)

  resampled_seg = resample_spacing(seg, [0.5, 0.5, 0.5], 'LINEAR')
  resampled_seg_path = '/home/qinliu/debug/resampled_seg.mha'
  sitk.WriteImage(resampled_seg, resampled_seg_path)


def crop_patch(image_path, patch_path, cropping_center):
  # image_path = '/home/admin/test/data_twomasks/case_1_cbct_trauma/org.nii.gz'
  # seg_path = '/home/admin/test/data_twomasks/case_1_cbct_trauma/seg_head_twomask.nii.gz'

  org_path = os.path.join(image_path,'img.nii.gz') #'/home/admin/test/data_v1/case_17_cbct_patient/org.mha'
  seg_path = os.path.join(image_path,'seg.nii.gz') #''/home/admin/test/data_v1/case_17_cbct_patient/seg.mha'

  image = sitk.ReadImage(org_path)
  image_org=image.GetOrigin
  seg = sitk.ReadImage(seg_path)

  # # ION# 
  # crop_physical_volume = [48, 38.4, 38.4]  #[48, 48, 48] #  unit mm
  # cropping_spacing = [image.GetSpacing()[idx] for idx in range(3)]
  # cropping_size = [crop_physical_volume[idx] // cropping_spacing[idx] for idx in range(3)]
  # print('crop size:{}'.format(cropping_size))

  # Po-L # 
  voxel_coord = np.array(image.TransformPhysicalPointToIndex(cropping_center))
  crop_physical_volume = [38.4, 38.4, 38.4]  #[48, 48, 48] #  unit mm
  cropping_spacing = [image.GetSpacing()[idx] for idx in range(3)]
  image_spacing = image.GetSpacing()
  cropping_size = [crop_physical_volume[idx] // cropping_spacing[idx] for idx in range(3)]
  voxel_offset = np.array([0 * 0.3 / image_spacing[0], 35 * 0.3 / image_spacing[1], 0 * 0.3 / image_spacing[2]])
  cropping_center = image.TransformContinuousIndexToPhysicalPoint(voxel_coord + voxel_offset)
  print('crop size:{}'.format(cropping_size))

  # ### posterior pterygoid patch ### # ION # 
  # voxel_coord = np.array(image.TransformPhysicalPointToIndex(cropping_center))
  # image_spacing = image.GetSpacing()
  # print('Image Spacing:{}'.format(image_spacing))
  # voxel_offset = np.array([-15 * 0.3 / image_spacing[0], 65 * 0.3 / image_spacing[1], -15 * 0.3 / image_spacing[2]])
  # cropping_center = image.TransformContinuousIndexToPhysicalPoint(voxel_coord + voxel_offset)
  # ### end ###

  # ### IC patch ##### 
  # voxel_coord = np.array(image.TransformPhysicalPointToIndex(cropping_center))
  # image_spacing = image.GetSpacing()
  # print('Image Spacing:{}'.format(image_spacing))
  # voxel_offset = np.array([45 * 0.3 / image_spacing[0], -40 * 0.3 / image_spacing[1], 0])
  # cropping_center = image.TransformContinuousIndexToPhysicalPoint(voxel_coord + voxel_offset)
  # ### end ###

  # # #### cranio Patch   #######      # A #  
  # crop_physical_volume = [134.4, 72, 60] #67.2 86.4,
  # cropping_spacing = [image.GetSpacing()[idx] for idx in range(3)]
  # cropping_size = [crop_physical_volume[idx] // cropping_spacing[idx] for idx in range(3)]
  # print('crop size:{}'.format(cropping_size))
  # voxel_coord = np.array(image.TransformPhysicalPointToIndex(cropping_center))
  # image_spacing = image.GetSpacing()
  # print('Image Spacing:{}'.format(image_spacing))
  # # voxel_offset = np.array([80 * 0.3 / image_spacing[1], 70 * 0.3 / image_spacing[1], -20 * 0.3 / image_spacing[1]])
  # voxel_offset = np.array([0 * 0.3 / image_spacing[1], 100 * 0.3 / image_spacing[1], 30 * 0.3 / image_spacing[1]])
  # cropping_center = image.TransformContinuousIndexToPhysicalPoint(voxel_coord + voxel_offset)


  # ####   end   #####
    
  cropped_image = crop_image(image, cropping_center, cropping_size, cropping_spacing, 'LINEAR')
  cropped_seg = crop_image(seg, cropping_center, cropping_size, cropping_spacing, 'NN')

  print('Result Folder: {}'.format(patch_path))
  cropped_image_path = os.path.join(patch_path,'cropped_org_po_L.mha') #'/home/admin/test/test_crop/cropped_org.mha'
  cropped_seg_path = os.path.join(patch_path, 'cropped_seg_po_L.mha') #'/home/admin/test/test_crop/cropped_seg.mha'

  if not os.path.isdir(patch_path) :
      print('Make Folder: {}'.format(patch_path))
      os.mkdir(patch_path)
 
  sitk.WriteImage(cropped_image, cropped_image_path, True)
  sitk.WriteImage(cropped_seg, cropped_seg_path, True)

def main(input_path, landmark_path, saving_path):
  if not os.path.isdir(saving_path):
    os.mkdir(saving_path)

  case_list = os.listdir(input_path)
  # case_list = os.path.splitext(case_list)[0]
  for case in case_list:

    lmk_path = os.path.join(landmark_path, case) #'data_ldmk')
    image_path = os.path.join(input_path, case)

    print('Current File:{}'.format(case))
    # landmark_case_path = os.path.join(landmark_path,case)

    # verify file path is valid
    verify_path = os.path.join(image_path,'img.nii.gz')
    if not os.path.isfile(verify_path):
      continue
    # verify end

    lm_path = os.path.join(lmk_path, 'lmk.xlsx')
    if not os.path.isfile(lm_path):
      lm_path = os.path.join(lmk_path, 'lmk.csv')

    case_name = os.path.splitext(case)[0]
    img_path = image_path # os.path.join(input_path, case_name)
    patch_path = os.path.join(saving_path, case)
    if os.path.isdir(patch_path):
      continue

    # seg_file = os.path.join(img_path,'seg.nii.gz')
    # if os.path.isfile(seg_file):
    #   print('file')
      
    # else:
    #   print('skip')
    #   continue


    center = get_anchor(lm_path)
    print('center:{}'.format(center))
    if center is not None:
      crop_patch(img_path, patch_path, center)

if __name__ == '__main__':

  # test_copy_image()

  # test_resample_spacing()

  # center = get_anchor()

  # crop_patch(center)
  # main('/home/admin/test/', '/home/admin/test/test_crop_COR_L')
  # main('/data/gt-cbct', '/data/deng/CBCT_patchtraining','/data/deng/CBCT_patch/IC')
  # main('/data/gt-cbct', '/data/gt-cbct','/data/deng/CBCT_patch/ION_R_p_large')
  # main('/data/deng/test', '/data/deng/test','/data/deng/CBCT_patch/whole_new')
  main('/data/gt-cbct', '/data/gt-cbct','/data/deng/CBCT_patch/patch_Po')

