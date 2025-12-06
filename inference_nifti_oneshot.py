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


#@torch.no_grad()
def main(device='cuda',batch_size=5): # device='cuda',cpu
    tagA='ARTERIAL'
    tagB='NATIVE'
    # root_path - is the path to the raw Coltea-Lung-CT-100W data set.
    opt = TrainOptions().parse()
    input_nifti_file = opt.inputnifti
    input_aorta_nifti_file = opt.inputaortanifti
    output_nifti_file = opt.outputnifti

    dirpath = os.path.dirname(output_nifti_file)
    os.makedirs(dirpath,exist_ok=True)

    opt.load_iter = 40
    opt.isTrain = True
    opt.device = device

    model = create_model(opt)
    model.setup(opt)
    model.netG_B.eval()
    model.netD_A.eval()
    model.netD_B.eval()
    gen = model.netG_A
    gen.train()

    img_obj = sitk.ReadImage(input_nifti_file)
    img_arr = sitk.GetArrayFromImage(img_obj)
    print(img_arr.shape)
    aorta_obj = sitk.ReadImage(input_aorta_nifti_file)
    aorta_arr_org = sitk.GetArrayFromImage(aorta_obj)

    orig_img_orig = img_arr.copy()
    orig_img = orig_img_orig + 1024
    orig_img[orig_img < 0] = 0 
    orig_img = orig_img / 1e3
    orig_img = orig_img - 1
    orig_img = np.expand_dims(orig_img, 1).astype(np.float)
    aorta_arr = np.expand_dims(aorta_arr_org, 1).astype(np.float)
    #blank_arr = np.zeros_like(orig_img)
    blank_arr = orig_img.copy()
    print(np.min(orig_img),np.max(orig_img))
    
    # NOTE: maybe filter only aorta slices?
    mydataset = torch.utils.data.TensorDataset(
        torch.from_numpy(orig_img), torch.from_numpy(blank_arr), torch.from_numpy(aorta_arr)
    )
    mydataloader = torch.utils.data.DataLoader(
        mydataset, batch_size=5, shuffle=True
    )

    #for x in range(5):
    myloss = 100
    while myloss > 1:
        print(myloss)
        loss_list = []
        for i, data in enumerate(mydataloader):
            model.set_custom_input(data)
            model.optimize_parameters(is_compute_aorta_hu_loss=True)
            loss_list.append(model.loss_aorta_mean_hu_G_A.cpu().detach().numpy())
            print(loss_list)
        myloss = np.mean(loss_list)
        print(myloss,model.loss_aorta_mean_hu_G_A,'!!!!!!!!!!!!!!!')

    gen.eval()

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
        #print(np.min(native_fake),np.max(native_fake))
        #print(native_fake.shape)
        myoutputlist.append(native_fake)

    output_arr = np.concatenate(myoutputlist,axis=0)
    output_arr = output_arr.squeeze()
    # scale back to HU
    minval, maxval = -600-750, -600+750
    output_arr = (((output_arr+1)*1000)-1024).clip(-1024,1024)
    print(output_arr.shape)
    out_obj = sitk.GetImageFromArray(output_arr.astype(np.int32))
    print('real aorta mean hu',np.mean(img_arr[aorta_arr_org==1]))
    print('fake aorta mean hu',np.mean(output_arr[aorta_arr_org==1]))
    #sys.exit(1)
    out_obj.CopyInformation(img_obj)
    sitk.WriteImage(out_obj,output_nifti_file)

if __name__ == '__main__':
    main()


"""

docker run -it --shm-size=10g \
-u $(id -u):$(id -g) -v /cvibraid:/cvibraid -v /dingo_data:/dingo_data \
-w $PWD pangyuteng/cycle-transformer bash

cd checkpoints/cytran
cp ../arterial-native/cytran/90_net_G_A.pth 40_net_G_A.pth
cp ../arterial-native/cytran/90_net_G_B.pth 40_net_G_B.pth
cp ../arterial-native/cytran/90_net_D_A.pth 40_net_D_A.pth
cp ../arterial-native/cytran/90_net_D_B.pth 40_net_D_B.pth

CUDA_VISIBLE_DEVICES=7 python inference_nifti_oneshot.py \
--dataroot /radraid/pteng-public/Coltea-Lung-CT-100W \
--checkpoints_dir checkpoints \
--inputnifti /dingo_data/cvib-airflow/RESEARCH/10156/images/10156_ICI_002/1.3.12.2.1107.5.1.4.55174.30000022090111444695300000102/image.nii.gz \
--inputaortanifti /dingo_data/cvib-airflow/RESEARCH/10156/totalsegmentator/results/10156_ICI_002/1.3.12.2.1107.5.1.4.55174.30000022090111444695300000102/aorta.nii.gz \
--outputnifti ./fake-one-shot.nii.gz \
--continue_train --lr 0.000001

CUDA_VISIBLE_DEVICES=7 python inference_nifti_oneshot.py \
--dataroot /radraid/pteng-public/Coltea-Lung-CT-100W \
--checkpoints_dir checkpoints \
--inputnifti /dingo_data/cvib-airflow/RESEARCH/10156/images/10156_ICI_002/1.3.12.2.1107.5.1.4.55174.30000022090111444695300000102/image.nii.gz \
--inputaortanifti /dingo_data/cvib-airflow/RESEARCH/10156/totalsegmentator/results/10156_ICI_002/1.3.12.2.1107.5.1.4.55174.30000022090111444695300000102/aorta.nii.gz \
--outputnifti ./fake-one-shot.nii.gz --continue_train --lr 0.00000001

0.000001
0.00000001

reduced lr, added all loss terms, set B image to A, not blank, set shuffle to true.
backprop with n=1, 5 iterations

"""