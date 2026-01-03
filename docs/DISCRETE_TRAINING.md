# Training the discrete diffusion model

This guide walks through preparing data, configuring the discrete diffusion scheduler, and running training/evaluation for the discrete Image-to-3D pipeline.

## 1) Prepare the datasets
- **RealEstate10K (scenes)**: Download and unzip the raw data, then run the provided preprocessing. Example:
  ```bash
  python process_data.py --base_path /path/to/re10k_raw --output_dir /path/to/re10k_processed --mode train
  python process_data.py --base_path /path/to/re10k_raw --output_dir /path/to/re10k_processed --mode test
  ```
  The resulting `full_list.txt` files are referenced as `local_dir` (train) and `local_eval_dir` (eval) in your scene config.
- **Objaverse / G-Objaverse (objects)**: Follow the G-Objaverse README to download the dataset and create `train.json`, `val.json`, `test.json`. Point `data.local_dir` to the JSON folder and `data.image_dir` to the images root. If you are on Colab, you can enable the optional auto-download flags in the config to fetch the JSONs/images into Drive automatically.

## 2) Configure the discrete scheduler
- The object config at `diffusionGS/configs/diffusionGS_rel.yaml` already selects the discrete scheduler via `system.noise_scheduler_type: diffusionGS.models.scheduler.discrete_scheduler.DiscreteScheduler`.
- Required edits before training:
  - Set `data.local_dir` and `data.image_dir` to your Objaverse paths.
  - If you want the model to consume octree tokens instead of continuous inputs, set `data.discrete_tokenize: true` and pick an `data.discrete_octree_depth` (e.g., 3–4). This will feed `octree_tokens_input` into the model alongside the discrete scheduler.
  - For stage 2, point `system.shape_model.pretrained_model_name_or_path` to the checkpoint produced by stage 1.

## 3) Launch training
You can reuse the provided scripts and adjust GPU/process counts as needed. Typical commands:
- **Object stage 1 (256px, discrete scheduler):**
  ```bash
  TORCHELASTIC_TIMEOUT=18000 torchrun --standalone --nproc-per-node=8 \
    launch.py --train --use_ema --gpu 0,1,2,3,4,5,6,7 \
    --config diffusionGS/configs/diffusionGS_rel.yaml
  ```
- **Object stage 2 (higher resolution fine-tune):**
  ```bash
  TORCHELASTIC_TIMEOUT=18000 torchrun --standalone --nproc-per-node=8 \
    launch.py --train --use_ema --gpu 0,1,2,3,4,5,6,7 \
    --config diffusionGS/configs/diffusionGS_rel.yaml
  ```
  Replace `--nproc-per-node` and `--gpu` with your available devices (e.g., `--nproc-per-node=1 --gpu 0` for a single GPU). Ensure stage 2 references the stage-1 checkpoint as noted above.

## 4) Evaluate and compute FID/KID
- After generating images, compute quantitative metrics with the included helper:
  ```bash
  python scripts/compute_fid_kid.py --reference /path/to/ground_truth_images --generated /path/to/generated_images \
    --batch-size 8 --image-size 299 --kid-subset-size 100 --output metrics.json
  ```
- If you saved rendered `.pt` files from validation, first extract the images (or re-render to disk), then point `--reference` and `--generated` to the corresponding folders.

## 5) Tips for Colab runs
- Mount Google Drive and set `data.local_dir`, `data.image_dir`, and any scene `local_dir/local_eval_dir` inside `/content/drive/...`.
- Optionally enable the commented `auto_download`, `cache_dir`, and `gdrive_folder_id` fields in the config to fetch Objaverse splits automatically.
- Reduce `batch_size`, `num_workers`, and GPU counts to fit Colab hardware.
