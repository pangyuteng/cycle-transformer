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
def main(device='cuda',batch_size=25): # device='cuda',cpu
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
    
    orig_img_orig = img_arr.copy()
    orig_img = orig_img_orig + 1024
    orig_img[orig_img < 0] = 0 
    orig_img = orig_img / 1e3
    orig_img = orig_img - 1
    orig_img = np.expand_dims(orig_img, 1).astype(np.float)
    blank_arr = np.zeros_like(orig_img)
    print(np.min(orig_img),np.max(orig_img))

    
    myoutputlist = []
    mydataset = torch.utils.data.TensorDataset(
        torch.from_numpy(orig_img), torch.from_numpy(blank_arr)
    )
    mydataloader = torch.utils.data.DataLoader(
        mydataset, batch_size=batch_size, shuffle=False
    )
    for step, (inputs, _) in enumerate(mydataloader):
        gpu_tensor = inputs.float().to(device)
        native_fake = gen(gpu_tensor).detach().cpu().numpy()
        print(np.min(native_fake),np.max(native_fake))
        print(native_fake.shape)
        myoutputlist.append(native_fake)

    output_arr = np.concatenate(myoutputlist,axis=0)
    output_arr = output_arr.squeeze()
    # scale back to HU
    minval, maxval = -600-750, -600+750
    output_arr = (((output_arr+1)*1000)-1024).clip(-1024,1024)
    print(output_arr.shape)
    out_obj = sitk.GetImageFromArray(output_arr.astype(np.int32))
    out_obj.CopyInformation(img_obj)
    sitk.WriteImage(out_obj,'fake.nii.gz')

if __name__ == '__main__':
    main()


"""

docker run -it --shm-size=10g \
-v /cvibraid:/cvibraid -v /dingo_data:/dingo_data \
-w $PWD pangyuteng/cycle-transformer bash

cd checkpoints/cytran
cp ../arterial-native/cytran/90_net_G_A.pth 40_net_G_A.pth
cp ../arterial-native/cytran/90_net_G_B.pth 40_net_G_B.pth

python inference_nifti.py \
--dataroot /radraid/pteng-public/Coltea-Lung-CT-100W \
--checkpoints_dir checkpoints \
--inputnifti /dingo_data/cvib-airflow/RESEARCH/10156/images/10156_ICI_002/1.3.12.2.1107.5.1.4.55174.30000022090111444695300000102/image.nii.gz

"""