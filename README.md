# A Closer Look at Weak Supervision’s Limitations in WSI Recurrence Score Prediction

<details>
  <summary>Abstract
  </summary>
  Histological examination and derived ancillary testing remain the gold standard for breast cancer diagnosis, prognosis assessment and treatment guidance. Currently, a commercial molecular signature test ONCOTYPEDX®, based on RNA quantitation and providing a recurrence score (RS) ranging from 0 to 100, is routinely utilized to predict the probabilities of response to chemotherapy and disease recurrence. We attempted to predict RS using digital pathology and weakly supervised AI models. In tissue samples, the malignant component is haphazardly admixed with the non-malignant component in variable proportions. This represents a challenge for weakly supervised AI models to identify high-valued diagnostic/prognostic areas within whole slide images (WSIs). To address this, we propose an interactive approach with a human in the middle by creating a user-friendly Graphical User Interface (GUI) that allows an expert pathologist to annotate heatmaps generated by any attention-based model. We aim to enhance the model’s learning capabilities and performance by incorporating the feedback from the GUI as expected scores in the successive training process. We train [CLAM](https://github.com/mahmoodlab/CLAM) (traditional convolution) and [TransMIL](https://github.com/szc19990412/TransMIL) (transformer) models on our in-house dataset before and after the expert feedback. We observe an improvement in RS prediction after retraining both models with the pathologist’s annotation- a 5% rise in validation-test AUC and 4% in validation-test accuracy for CLAM and a 4.5% increase in validation-test AUC and 3% in validation-test accuracy for TransMIL. 
  Furthermore, we analyzed the generated heatmaps and observed how additional supervision from a domain expert enhanced the learning capacity of the models. We notice an improvement in cosine similarity between the pathologist’s GUI-based attention scores and trained models’ attention maps after feedback - 5% and 10% increase for CLAM and TransMIL, respectively. Our adaptive, interactive system harmonizes attention scores with expert intuition and instills higher confidence in the system’s predictions. This study establishes a potent synergy between AI and expert collaboration, addressing the constraints of weak supervision by enhancing the discrimination of diagnostic features and making an effort to generate predictions according to clinical diagnostic norms.
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
