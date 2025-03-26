import os

import torch
import SimpleITK as sitk
from . import detection3d
from . import segmentation3d


def read_image(file_or_dcmdir) -> sitk.Image:
    if os.path.isdir(file_or_dcmdir):

        dcmdir = file_or_dcmdir

        series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dcmdir)
        if not series_IDs:
            raise ValueError("ERROR: given directory \"" + dcmdir +
                             "\" does not contain a DICOM series.")

        if len(series_IDs) > 1:
            raise ValueError("ERROR: given directory \"" + dcmdir +
                             "\" contains multiple DICOM series.")

        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
            dcmdir, series_IDs[0])

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
        img_file = file_or_dcmdir
        img = sitk.ReadImage(img_file, sitk.sitkFloat32)

    return img





def crop_image(image, cropping_center, cropping_size, cropping_spacing, interp_method):
    """
    Crop a patch from a volume given the cropping center, cropping size, cropping spacing, and the interpolation method.
    This function DO NOT consider the transformation of coordinate systems, which means the cropped patch has the same
    coordinate system with the given volume.

    :param image: the given volume to be cropped.
    :param cropping_center: the center of the cropped patch in the world coordinate system of the given volume.
    :param cropping_size: the voxel coordinate size of the cropped patch.
    :param cropping_spacing: the voxel spacing of the cropped patch.
    :param interp_method: the interpolation method, only support 'NN' and 'Linear'.
    :return a cropped patch
    """
    assert isinstance(image, sitk.Image)

    cropping_center = [float(cropping_center[idx]) for idx in range(3)]
    cropping_size = [int(cropping_size[idx]) for idx in range(3)]
    cropping_spacing = [float(cropping_spacing[idx]) for idx in range(3)]

    cropping_physical_size = [cropping_size[idx] *
                              cropping_spacing[idx] for idx in range(3)]
    cropping_start_point_world = [
        cropping_center[idx] - cropping_physical_size[idx] / 2.0 for idx in range(3)]
    for idx in range(3):
        cropping_start_point_world[idx] += cropping_spacing[idx] / 2.0

    cropping_origin = cropping_start_point_world
    cropping_direction = image.GetDirection()

    if interp_method == 'LINEAR':
        interp_method = sitk.sitkLinear
    elif interp_method == 'NN':
        interp_method = sitk.sitkNearestNeighbor
    else:
        raise ValueError('Unsupported interpolation type.')

    transform = sitk.Transform(3, sitk.sitkIdentity)
    outimage = sitk.Resample(image, cropping_size, transform, interp_method, cropping_origin, cropping_spacing,
                             cropping_direction)

    return outimage
