# A Closer Look at Weak Supervision’s Limitations in WSI Recurrence Score Prediction

<details>
  <summary>Abstract
  </summary>
  TBD
</details>

### Pre-processing

We follow pre-processing from [CLAM](https://github.com/mahmoodlab/CLAM)
```
#segmentation and create patches
'''
convert the images to HSV colour space and generate binary masks for tissue regions using thresholding of the saturation channel
extract 256 × 256 patches from the segmented foreground contours and store their coordinates
'''
python create_patches_fp.py --source DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --patch_size 256 --seg --process_list CSV_FILE_NAME --patch --stitch

#extract features
'''
employ ImageNet pretrained ResNet50 that converts each patch into a 1024-dimensional feature vector representation
extracted features serve as inputs during the training
'''
CUDA_VISIBLE_DEVICES=0,1 python extract_features_fp.py --data_h5_dir DIR_TO_COORDS --data_slide_dir DATA_DIRECTORY --csv_path CSV_FILE_NAME --feat_dir FEATURES_DIRECTORY --batch_size 512 --slide_ext .svs

#create splits
python create_splits_seq.py --task task_2_tumor_subtyping  --k 10
```

### Installation Guide

We perform experiments on a consumer-grade workstation equipped with the NVIDIA GeForce 3090 GPU. 

Create a [virtual environment](https://docs.python.org/3/library/venv.html) and run the following:

```
pip install -r requirements.txt
```

### Details

1. Run the vanilla version of CLAM
   ```
   python main_vanilla_CLAM.py --drop_out --early_stopping --lr 2e-4 --k 10 --label_frac 1.0 --exp_code task_2_tumor_subtyping_CLAM_50 --weighted_sample --bag_loss ce --inst_loss svm --task task_2_tumor_subtyping --model_type clam_mb --log_data --data_root_dir <base-data-dir>
   ```

2. Generate heatmaps from CLAM repository [create_heatmaps.py](https://github.com/mahmoodlab/CLAM/blob/master/create_heatmaps.py)


testing
