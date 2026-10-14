# Semantic-Centric Cross-Domain Learning for Dual-Incremental Object Detection

This repository contains the MMDetection-based implementation of **Semantic-Centric Cross-Domain Learning for Dual-Incremental Object Detection (SCL-IOD)**.  checked-in experiment configuration covers the BDD100K **Nighttime Sunny classes 1–4 → Daytime Sunny classes 5–7** setting. The class order used by the BDD configs is `bike`, `bus`, `car`, `motor`, `person`, `rider`, `truck`.

## Paper
- Paper: [ACM Digital Library (DOI)](https://doi.org/10.1145/3767308.3835279)

## Requirements and installation

The checked-in environment instructions use Python 3.8, PyTorch 1.13.1, CUDA 11.6, MMEngine 0.7.3, and MMCV 2.0.0. Install the repository in editable mode after installing those dependencies:

```bash
conda create -n CLOD python=3.8 -y
conda activate CLOD

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install tqdm
pip install -U openmim
mim install mmengine==0.7.3
mim install mmcv==2.0.0

mkdir -p cache temp_cheakpoints
pip install -v -e .
```

## BDD100K data

Download the BDD100K images and detection labels, then arrange the image files and generated COCO annotations as follows:

```text
dataset/bdd100k/
├── images/100k/
│   ├── train/
│   └── val/
└── annotations/
    ├── bdd100k_labels_images_nighttime_clear_train_coco_format.json
    ├── bdd100k_labels_images_nighttime_clear_val_coco_format.json
    ├── bdd100k_labels_images_daytime_clear_train_coco_format.json
    └── bdd100k_labels_images_daytime_clear_val_coco_format.json
```

The data preparation scripts are in the repository root. They use input and output filenames defined in the scripts rather than command-line arguments. Update those paths for each BDD100K train/validation JSON before running them:

1. [`1-fliter_by_weather_and_timeofdat.py`](1-fliter_by_weather_and_timeofdat.py) filters one input JSON. As checked in, it selects clear-weather nighttime images from the train labels. To produce all four splits above, run it for train and validation and adapt the condition to select clear-weather daytime images as well.
2. [`2-transfer_bdd100k_json_to_coco_json.py`](2-transfer_bdd100k_json_to_coco_json.py) converts one filtered JSON to COCO format. Set its input and output paths for each split. It keeps the seven detection categories and assigns IDs in this order: `bike`, `bus`, `car`, `motor`, `person`, `rider`, `truck`.

Keep the image filenames from the BDD100K labels unchanged so they resolve under `images/100k/train` or `images/100k/val`. The configs use `dataset/bdd100k` as the data root.

## Training: Nighttime Sunny [1–4] → Daytime Sunny [5–7]

Run the stages in order. The first two train the old-class and new-class experts. The feature extraction stage uses the old-class model to save ROI features; the clustering script turns those features into 10 prototypes per class. The final stage loads the two expert checkpoints and the generated prototype file.

```bash
# 1. Train the Nighttime Sunny old-class expert (classes 1–4).
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29533 ./tools/dist_train.sh \
  configs/faster_rcnn_increase/NS-DS/SCL-IOD/nighttimeSunny1_4.py 4

# 2. Train the Daytime Sunny new-class expert (classes 5–7).
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29533 ./tools/dist_train.sh \
  configs/faster_rcnn_increase/NS-DS/SCL-IOD/daytimeSunny5_7.py 4

# 3. Extract ROI features using the old-class expert.
CUDA_VISIBLE_DEVICES=0 PORT=29533 ./tools/dist_train.sh \
  configs/faster_rcnn_increase/NS-DS/get_feature/nighttimeSunny1_4.py 1

# 4. Cluster the extracted features into per-class prototypes.
python util/select_protype_and_cluster.py

# 5. Train the joint incremental model.
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29533 ./tools/dist_train.sh \
  configs/faster_rcnn_increase/NS-DS/SCL-IOD/nighttimeSunny1_4_to_daytimeSunny5_7.py 4
```

The default training batch size is 4 images per GPU. With the four-GPU commands above, the total batch size is 16, matching `auto_scale_lr.base_batch_size` in the configs. If you change the GPU count, keep the total batch size at 16 or update the learning-rate scaling settings accordingly.

The prototype extraction and clustering steps expect the configured working directory under `work_dir/faster_rcnn_increase/NS-DS/SCL-IOD/`. The final config expects the expert checkpoints `nighttimeSunny1_4/epoch_12.pth` and `daytimeSunny5_7/epoch_12.pth`, and the prototype file `nighttimeSunny1_4/prototype/class_centroids_10_per_class.pt`. Check that these artifacts exist before starting the final stage.

## Citation

If you use this project, please cite the paper:

```bibtex
@inproceedings{luo2026semanticcentric,
  author    = {Luo, Wenlong and Zhang, Shizhou and Wang, Chenxu and
               Xing, Yinghui and Xu, Di and Cheng, De and Wang, Peng},
  title     = {Semantic-Centric Cross-Domain Learning for Dual-Incremental Object Detection},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia (MM '26)},
  year      = {2026},
  address   = {Rio de Janeiro, Brazil},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3767308.3835279},
  url       = {https://doi.org/10.1145/3767308.3835279}
}
```
