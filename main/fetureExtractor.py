import os
import numpy as np
import pandas as pd
from radiomics import featureextractor
import SimpleITK as sitk


def extract_feature(img_path, label_path):
    """
    Extract features from the given image and label paths.

    :param img_path: Path to the image file
    :param label_path: Path to the label file
    :return: Extracted features as a DataFrame
    """
    # Read image
    itk_img = sitk.ReadImage(img_path)
    img = sitk.GetArrayFromImage(itk_img)
    itk_label = sitk.ReadImage(label_path)
    label = sitk.GetArrayFromImage(itk_label)


    settings = {}

    settings['resampledPixelSpacing'] = [1, 1, 1]  # Resampling to 1mm^3
    settings['voxelArrayShift'] = 300  # Shift for intensity normalization
    settings['normalize'] = True
    settings['normalizeScale'] = 100

    # Feature extraction
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.enableImageTypes(Original={})
    extractor.enableFeaturesByName(firstorder=[])
    extractor.enableFeaturesByName(shape3D=[])
    extractor.enableFeaturesByName(glcm=[])
    extractor.enableFeaturesByName(glrlm=[])
    extractor.enableFeaturesByName(glszm=[])
    extractor.enableFeaturesByName(ngtdm=[])
    extractor.enableFeaturesByName(gldm=[])

    feature = extractor.execute(sitk.GetImageFromArray(img), sitk.GetImageFromArray(label))

    df_temp = pd.DataFrame({k: str(v) for k, v in feature.items()}, index=[0])

    return df_temp


def batch_preprocess(rootpath):
    i = 0
    total_patient_dir = []
    total_group = []
    df_feature_all = None
    for f1 in os.listdir(rootpath):
        if f1.endswith(".7z") or f1 == "code":
            continue
        f1path = os.path.join(rootpath, f1)
        for f2 in os.listdir(f1path):
            abnpath = os.path.join(f1path, f2)
            dicompath = os.path.join(abnpath, "A150kev", "DICOM")
            t = 0
            try:
                for file in os.listdir(dicompath):
                    if file.endswith(".nii.gz"):
                        if t == 1:
                            continue
                        t = t + 1
                        i = i + 1
                        filepath = os.path.join(dicompath, file)

                        f3 = next((os.path.join(dicompath, d) for d in os.listdir(dicompath) if
                                   os.path.isdir(os.path.join(dicompath, d))), None)

                        f4 = os.path.join(f3, os.listdir(f3)[0])

                        for mask in os.listdir(f4):
                            if mask.endswith(".nii.gz"):
                                maskpath = os.path.join(f4, mask)
                                df_feature = extract_feature(filepath, maskpath)
                                if df_feature_all is None:
                                    df_feature_all = df_feature
                                else:
                                    df_feature_all = pd.concat([df_feature_all, df_feature], axis=0)
                                total_patient_dir.append(f2)
                                if mask.startswith("F"):
                                    total_group.append(0)
                                else:
                                    total_group.append(1)
            except FileNotFoundError:
                print("File not found")

    df_feature_all.insert(len(df_feature_all.columns), "group", total_group)
    df_feature_all.insert(0, "patient", total_patient_dir)
    df_feature_all.to_csv("feature_extracted.csv", index=False, encoding='utf-8-sig')
    print('Feature extraction completed')
    print(i)


if __name__ == '__main__':
    rootpath = r"F:\dataset\brain\DECT+PETCT"
    batch_preprocess(rootpath)
