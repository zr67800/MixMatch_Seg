import os, argparse
import nibabel as nib
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/2018LA_Seg_Training Set/', help='Root path of data')
parser.add_argument('--img_root_path', type=str, default='../evaluation/img/', help='Root path of images to eval')
parser.add_argument('--remove', type=bool, default=False, help='Remove a certain strange img')

FLAGS = parser.parse_args()

root_path = FLAGS.root_path
with open(root_path + '/../test.list', 'r') as f:
    image_list = f.readlines()
image_list = list(map(str.strip, image_list))
if FLAGS.remove:
    image_list.remove("VG4C826RAAKVMV9BQLVD")

def load_nii(filename):
    '''Load nii file, return ndarray'''
    f = nib.load(filename)
    d = f.get_fdata()
    return d

def dice_coefficient(pred, gt):
    pred = pred.flatten()
    gt = gt.flatten()
    dice = 2*gt.dot(pred)/(gt.dot(gt)+pred.dot(pred)+1e-5)
    return dice

def compare_nii(path, id):
    gt = load_nii(f"{path}/{id}_gt.nii.gz")
    pred = load_nii(f"{path}/{id}_pred.nii.gz")
    dice = dice_coefficient(pred, gt)
    print(f"id {id}: dice = {dice:.4f}")
    return dice

def compare_all(path):
    sum_dice = 0
    for id in image_list:
        sum_dice += compare_nii(path, id)
    avg = sum_dice/len(image_list)
    print (f"{path}: avg dice = {avg:.4f}")



if __name__ == "__main__":
    #print (image_list)
    img_root_path = FLAGS.img_root_path
    with os.scandir(img_root_path) as it:
        for entry in it:
            if not entry.name.startswith('.') and entry.is_dir():
                print(entry.name)
                try:
                    compare_all(img_root_path+entry.name)
                except:
                    print("Error")