

+ started from trying to replicate article results, via downloading data.

+ post training, running inference

    + `inference.py` - single slice

    + `inference_nifti.py` - load volume for inference
    
    + crude assessment shows HU in aorta is not consistently low
      when we attempt to generate fake pre-contrast images.

    + `inference_nifti_oneshot.py` attempt to train during inference
      somehow incorporating reduced mean in aorta as loss function?