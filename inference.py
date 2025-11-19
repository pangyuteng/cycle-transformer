import sys
import glob
import os
import numpy as np
import pandas as pd
import pydicom
import torch

#from skimage.metrics import structural_similarity as ssim
from models import create_model
from options.train_options import TrainOptions
import SimpleITK as sitk
import matplotlib.pyplot as plt

@torch.no_grad()
def main(device='cuda'): # device='cuda',cpu
    tagA='ARTERIAL'
    tagB='NATIVE'
    # root_path - is the path to the raw Coltea-Lung-CT-100W data set.

    opt = TrainOptions().parse()
    nifti_file = opt.inputnifti

    opt.load_iter = 40
    opt.isTrain = False
    opt.device = device

    model = create_model(opt)
    model.setup(opt)
    gen = model.netG_A
    gen.eval()

    img_obj = sitk.ReadImage(nifti_file)
    img_arr = sitk.GetArrayFromImage(img_obj)
    print(img_arr.shape)

    orig_img_orig = img_arr[100,:,:].squeeze()
    orig_img = orig_img_orig + 1024
    print(orig_img.shape)
     
    # Scale original image, which is transform

    orig_img[orig_img < 0] = 0 
    orig_img = orig_img / 1e3
    orig_img = orig_img - 1
    print(np.min(orig_img),np.max(orig_img))

    orig_img_in = np.expand_dims(orig_img, 0).astype(np.float)
    orig_img_in = torch.from_numpy(orig_img_in).float().to(device)
    orig_img_in = orig_img_in.unsqueeze(0)
    print(orig_img_in.shape)

    native_fake = gen(orig_img_in)[0, 0].detach().cpu().numpy()
    print(np.min(native_fake),np.max(native_fake))
    print(native_fake.shape)

    # lungs W:1500 L:-600
    # -600-750,-600+750
    # minval, maxval = ((-600-750+1024)/1000)-1,((-600+750+1024)/1000)-1
    
    # scale back to HU
    minval, maxval = -600-750, -600+750
    native_fake = (((native_fake+1)*1000)-1024).clip(-1024,1024)
    
    print(minval, maxval)
    print(np.min(orig_img),np.max(orig_img))

    real_obj = sitk.GetImageFromArray(orig_img_orig.astype(np.int32))
    sitk.WriteImage(real_obj,'real-post-contrast-slice-100.nii.gz')
    fake_obj = sitk.GetImageFromArray(native_fake.astype(np.int32))
    sitk.WriteImage(fake_obj,'fake-pre-contrast-slice-100.nii.gz')

    plt.figure()
    plt.subplot(121)
    plt.title("real (with contrast)")
    plt.imshow(orig_img_orig,cmap='gray',vmin=minval,vmax=maxval,interpolation='nearest')
    plt.subplot(122)
    plt.title("fake")
    plt.imshow(native_fake,cmap='gray',vmin=minval,vmax=maxval,interpolation='nearest')
    plt.savefig("ok.png")


if __name__ == '__main__':
    main()


"""

docker run -it --shm-size=10g \
-v /cvibraid:/cvibraid -v /dingo_data:/dingo_data \
-w $PWD pangyuteng/cycle-transformer bash

cd checkpoints/cytran
cp ../arterial-native/cytran/90_net_G_A.pth 40_net_G_A.pth
cp ../arterial-native/cytran/90_net_G_B.pth 40_net_G_B.pth

python inference.py \
--dataroot /radraid/pteng-public/Coltea-Lung-CT-100W \
--checkpoints_dir checkpoints \
--inputnifti /dingo_data/cvib-airflow/RESEARCH/10156/images/10156_ICI_002/1.3.12.2.1107.5.1.4.55174.30000022090111444695300000102/image.nii.gz

"""