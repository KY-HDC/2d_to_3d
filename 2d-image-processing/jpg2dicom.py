import os
import pydicom
from pydicom.dataset import FileDataset
from PIL import Image


def convert_images_to_dicom(jpg_folder, dicom_folder):
    # Create the output DICOM folder if it doesn't exist
    if not os.path.exists(dicom_folder):
        os.makedirs(dicom_folder)

    # Iterate over all JPG files in the input folder
    for filename in os.listdir(jpg_folder):
        if filename.endswith('.png'):
            # Load the PNG image
            jpg_path = os.path.join(jpg_folder, filename)
            image = Image.open(jpg_path)

            # Create a new DICOM dataset
            dicom_path = os.path.join(dicom_folder, f'{filename[:-4]}.dcm')
            ds = FileDataset(dicom_path, {}, file_meta=pydicom.Dataset())

            # Set the DICOM file meta information
            ds.file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.7'  # Secondary Capture Image Storage
            ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
            ds.file_meta.ImplementationClassUID = pydicom.uid.generate_uid()

            # Set the DICOM dataset fields
            ds.PatientName = 'Anonymous'
            ds.PatientID = '12345'
            ds.Modality = 'OT'
            ds.SamplesPerPixel = 3
            ds.PhotometricInterpretation = 'RGB'
            ds.Rows, ds.Columns, _ = image.size
            ds.PixelSpacing = [1.0, 1.0]
            ds.BitsAllocated = 8
            ds.BitsStored = 8
            ds.HighBit = 7
            ds.PixelRepresentation = 0
            ds.SeriesInstanceUID = pydicom.uid.generate_uid()
            ds.StudyInstanceUID = pydicom.uid.generate_uid()
            ds.FrameOfReferenceUID = pydicom.uid.generate_uid()

            # Convert the image to a byte array
            image_bytes = image.tobytes()

            # Set the pixel data
            ds.PixelData = image_bytes

            # Save the DICOM dataset
            ds.save_as(dicom_path)


if __name__ == '__main__':
    jpg_folder = '/data2/gayrat/vs-projects/image_dataset_stacks/mednerf_drr_dataset/knee_xrays'
    dicom_folder = '/data2/gayrat/vs-projects/image_dataset_stacks/knees_dicom'
    print(convert_images_to_dicom(jpg_folder, dicom_folder))