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
import matplotlib.pyplot as plt

@torch.no_grad()
def main(device='cuda'): # device='cuda',cpu
    tagA='ARTERIAL'
    tagB='NATIVE'
    # root_path - is the path to the raw Coltea-Lung-CT-100W data set.

    opt = TrainOptions().parse()
    inputdicom = opt.inputdicom # dicom image
    outputdicom = opt.outputdicom # dicom image

    opt.load_iter = 40
    opt.isTrain = False
    opt.device = device

    model = create_model(opt)
    model.setup(opt)
    gen = model.netG_A
    gen.eval()

    orig_img = pydicom.dcmread(inputdicom).pixel_array
    org_dtype = orig_img.dtype

    orig_img[orig_img < 0] = 0
    orig_img = orig_img / 1e3
    orig_img = orig_img - 1

    orig_img_in = np.expand_dims(orig_img, 0).astype(np.float)
    orig_img_in = torch.from_numpy(orig_img_in).float().to(device)
    print(orig_img_in.shape)
    orig_img_in = orig_img_in.unsqueeze(0)
    print(orig_img_in.shape)

    native_fake = gen(orig_img_in)[0, 0].detach().cpu().numpy()
    print(np.min(native_fake),np.max(native_fake))
    print(native_fake.shape)
    print(np.min(native_fake),np.max(native_fake))
    native_fake = ((native_fake+1)*1000).clip(0,2048)
    print(np.min(native_fake),np.max(native_fake))
    
    print(native_fake.dtype)
    target_arr = native_fake.astype(np.uint16)
    ds = pydicom.dcmread(inputdicom)
    ds.PixelData = target_arr.tobytes()
    ds.save_as(outputdicom)

if __name__ == '__main__':
    main()


"""

docker run -it --shm-size=10g \
-v /cvibraid:/cvibraid -v /dingo_data:/dingo_data \
-w $PWD pangyuteng/cycle-transformer bash

cd checkpoints/cytran
cp ../arterial-native/cytran/90_net_G_A.pth 40_net_G_A.pth
cp ../arterial-native/cytran/90_net_G_B.pth 40_net_G_B.pth

CUDA_VISIBLE_DEVICES=7 python arterial_to_native_dcm.py \
--dataroot /radraid/pteng-public/Coltea-Lung-CT-100W \
--checkpoints_dir checkpoints \
--inputdicom /dingo_data/cvib-airflow/RESEARCH/10156/images/10156_ICI_002/1.3.12.2.1107.5.1.4.55174.30000022090111444695300000102/dicom/10156_ICI_002/2022-09-01/1.3.12.2.1107.5.1.4.55174.30000022090111444695300000102/8a3cc270-26d574bf-55620207-a433d6d8-45327a33.dcm \
--outputdicom ok.dcm

"""