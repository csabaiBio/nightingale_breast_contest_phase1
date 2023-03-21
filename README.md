# csabAIbio 1st place solution for Nightingale High Risk Breast Cancer Prediction Contest 1

### Data processing

- generate .npy files from level4/level5/level7 images from the original ndpi files - big images
	- patch_gen/generate_lvl4_npy.py
	- patch_gen/generate_lvl5_npy.py
	- patch_gen/generate_lvl7_npy.py
	
- generate slide bags (.npy format) from level4/level5 images (.npy format) that includes all the non-masked patches in a shape of 224x224x3 with the masking made on level 7
	- patch_gen/generate_all_patches_maskimproved_lvl7_fromnpy.py (run with level4/level5)
	
- generate .npy files from level4/level5/level7 of the original [svs bracs images](https://www.bracs.icar.cnr.it/)
	- bracs/bracs_save_level4equivalent.ipynb
	- bracs/bracs_save_level5equivalent.ipynb
	- bracs/bracs_save_level7equivalent.ipynb

- extract 224x224x3 sized patches from bracs images within annotations
	- bracs/bracs_extract_patches_from_roi_with_mask_level4.ipynb
	
- train embedder/classifier on bracs dataset (for transfer learning)
	- bracs/bracs_level4_train_embeddings.ipynb

- generate embeddings from slide bags (.npy format) with ImageNet embedder/classifier
	- patch_gen/imagenet_embedding_generator.py

- generate embeddings from slide bags (.npy format) with bracs embedder/classifier
	- patch_gen/bracs_embedding_generator.py
	
- reorganise data into biopsy bags with the corresponding patches and generate train/valid folds
	- re_pack_biopsies_training.ipynb -> test_biopsy_unbalenced.csv, train_biopsy_unbalenced.csv
	- re_pack_biopsies_holdout.ipynb -> final_splits/HOLD_OUT.csv

### Training
- BiopsyMIL_efficientloader.py (uses model.py)
	
### Prediction
- inference.ipynb
- ensemble_optim.ipynb

