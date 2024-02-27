# CrossDiff

### [Project Page](https://wonderno.github.io/CrossDiff-webpage/) | [Arxiv](https://arxiv.org/abs/2312.10993)

This is the official PyTorch implementation of the paper "Realistic Human Motion Generation with Cross-Diffusion Models". Our method leverages intricate 2D motion knowledge and builds a cross-diffusion mechanism to enhance 3D motion generation.

![teaser](https://github.com/wonderNo/crossdiff/blob/master/assets/teaser.png)

## 1 Setup 

### 1.1 Environment

This code has been tested with Python 3.8 and PyTorch 1.11.

```shell
conda create -n crossdiff python=3.8
conda activate crossdiff
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

### 1.2 Dependencies

Execute the following script to download the necessary materials:

```shell
mkdir data/
bash prepare/download_smpl_files.sh
bash prepare/download_glove.sh
bash prepare/download_t2m_evaluators.sh
```

### 1.3 Pre-train model

Run the script below to download the pre-trained model:

```shell
bash prepare/download_pretrained_models.sh
```

## 2 Train

### 2.1 Prepare data

**HumanML3D** - Follow the instructions provided in the [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git). Afterward, execute the following command to obtain the corresponding 2D motion:
```shell
python prepare/project.py --data_root YOUR_DATA_ROOT
```
Additionally, please set the `data_root` in the configuration file `configs/base.yaml` for subsequent training.

**UCF101** - This dataset is used to train the model with real-world 2D motion.

Download the original data from the [UCF101 project page](https://www.crcv.ucf.edu/data/UCF101.php#Results_on_UCF101). Then, estimate the 2D pose using the off-the-shelf model [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) and process the 2D data in the same manner as HumanML3D.

### 2.2 Train the model

For the first stage, execute the following command:

```shell
python train.py --cfg configs/crossdiff_pre.yaml
```
The results will be stored in `./save/crossdiff_pre`. Locate the best checkpoint and set the `resume_checkpoint` in `configs/crossdiff_finetune.yaml`.

For the second stage, run:
```shell
python train.py --cfg configs/crossdiff_finetune.yaml
```
The final results will be saved in `./save/crossdiff_finetune`

## 3 Test

After training, run the following command to test the model:
```shell
python test.py --cfg configs/crossdiff_finetune.yaml
```
By default, the code will use the final model for testing. Alternatively, you can set the `test_checkpoint` in the configuration file to test a specific model.

You may also configure the following options:
* `test_mm`: Test Multimodality.
* `eval_part`: Choose from `all`,`upper`, or `lower` to test metrics for different body parts.

## 4 Generate

To generate motion from text, use:

```shell
python generate.py --cfg configs/crossdiff_finetune.yaml test_checkpoint=./data/checkpoints/pretrain.pt
```

You can edit the text in the configuration file using the `captions` parameter. The output will be saved in `./save/crossdiff_finetune/eval`. Then, execute:

```shell
python fit_smpl.py -f YOUR_KEYPOINT_FILE
```
This will fit the selected `.npy` file of body keypoints, and you will obtain the mesh file `_mesh.npy`.

For visualizing SMPL results, refer to [MLD-Visualization](https://github.com/ChenFengYe/motion-latent-diffusion) and [TEMOS-Rendering motions](https://github.com/Mathux/TEMOS) for Blender setup.

Run the following command to visualize SMPL:

```shell
blender --background --python render_blender.py -- --file=YOUR_MESH_FILE
```

## Acknowledgments

We express our gratitude to [MDM](https://github.com/GuyTevet/motion-diffusion-model), [MLD](https://github.com/ChenFengYe/motion-latent-diffusion), [T2M-GPT](https://github.com/Mael-zys/T2M-GPT), [TEMOS](https://github.com/Mathux/TEMOS). Our code is partially adapted from their work.

## Bibtex

If you find this code useful in your research, please cite:

```
@article{ren2023realistic,
  title={Realistic Human Motion Generation with Cross-Diffusion Models},
  author={Ren, Zeping and Huang, Shaoli and Li, Xiu},
  journal={arXiv preprint arXiv:2312.10993},
  year={2023}
}
```
