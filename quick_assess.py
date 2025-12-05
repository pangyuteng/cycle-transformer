import SimpleITK as sitk
import numpy as np

img_file = "/dingo_data/cvib-airflow/RESEARCH/10156/images/10156_ICI_002/1.3.12.2.1107.5.1.4.55174.30000022090111444695300000102/image.nii.gz"
aorta_file = "/dingo_data/cvib-airflow/RESEARCH/10156/totalsegmentator/results/10156_ICI_002/1.3.12.2.1107.5.1.4.55174.30000022090111444695300000102/aorta.nii.gz"
fake_file = "fake.nii.gz"
fake_oneshot_file = "fake-one-shot.nii.gz"

img_obj = sitk.ReadImage(img_file)
aorta_obj = sitk.ReadImage(aorta_file)
fake_obj = sitk.ReadImage(fake_file)
fake_oneshot_obj = sitk.ReadImage(fake_oneshot_file)

real = sitk.GetArrayFromImage(img_obj)
aorta = sitk.GetArrayFromImage(aorta_obj)
fake = sitk.GetArrayFromImage(fake_obj)
fake_oneshot = sitk.GetArrayFromImage(fake_oneshot_obj)

print('real',np.mean(real[aorta==1]))
print('fake',np.mean(fake[aorta==1]))
print('fake_oneshot',np.mean(fake_oneshot[aorta==1]))