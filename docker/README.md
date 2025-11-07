

# NOTES


### download 

downloaded with gdown

/radraid/pteng-public/Coltea-Lung-CT-100W

### container related notes.

docker run -it -u $(id -u):$(id -g) -v /cvibraid:/cvibraid -w $PWD pangyuteng/cycle-transformer bash

docker run -it --shm-size=10g -v /cvibraid:/cvibraid -v /radraid:/radraid -w $PWD pangyuteng/cycle-transformer bash


NOTE: json file is not respecting original author train/val/test 
/radraid/pteng-public/Coltea-Lung-CT-100W/*.csv


python train.py --dataroot /cvibraid/cvib2/apps/personal/pteng/github/cycle-transformer/docker/ct_data.json \
--Aclass ARTERIAL --Bclass NATIVE \
--checkpoints_dir /cvibraid/cvib2/apps/personal/pteng/github/cycle-transformer/checkpoints/arterial-native

python train.py --dataroot /cvibraid/cvib2/apps/personal/pteng/github/cycle-transformer/docker/ct_data.json \
--Aclass VENOUS --Bclass NATIVE \
--checkpoints_dir /cvibraid/cvib2/apps/personal/pteng/github/cycle-transformer/checkpoints/venous-native

python train.py --dataroot /cvibraid/cvib2/apps/personal/pteng/github/cycle-transformer/docker/ct_data.json \
--Aclass ARTERIAL --Bclass VENOUS \
--checkpoints_dir /cvibraid/cvib2/apps/personal/pteng/github/cycle-transformer/checkpoints/arterial-venous