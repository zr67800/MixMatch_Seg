import nibabel as nib
import numpy as np
import os


def save_as_image(v, filename, image = False):
    '''
    save v in to .nii or 3 centre-cut images along x, y, z axis
    v should be in shape (n, c, d1, d2, d3)

    '''
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    for i,x in enumerate(v):
        if not image:
            nib.save(nib.Nifti1Image(x[0,:,:,:].astype(np.float32), np.eye(4)), f"{filename}_{i}.nii.gz")

