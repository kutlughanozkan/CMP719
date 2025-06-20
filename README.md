# CMP719

# 🔍 Feature Matching Benchmark on HPatches & Oxford-Affine

Evaluate and compare four feature matching algorithms — **SIFT**, **ORB**, **SuperPoint+LightGlue**, and **LoFTR** — on the HPatches and Oxford-Affine datasets.

The script computes matching **precision** based on reprojection error after **RANSAC**, along with **execution time** and **number of matches**.

---

## Requirements

- Python 3.8+
- PyTorch with CUDA (for GPU acceleration)
- OpenCV
- NumPy
- tqdm

Install dependencies:

'''
pip install torch torchvision opencv-python numpy tqdm

pip install torch einops yacs kornia

git clone https://github.com/cvg/LightGlue.git

git clone https://github.com/zju3dv/LoFTR.git
'''
#Usage

python run_patches.py \
  --root /path/to/datasets \
  --out_dir /path/to/save/results \
  --dataset both \
  --lg_path /path/to/LightGlue \
  --lft_path /path/to/LoFTR
  
  
#Outputs


├── sift_pairs.json
├── orb_pairs.json
├── sp_lg_pairs.json
├── loftr_pairs.json
└── summary.csv  
  
*_pairs.json: contains per-pair statistics:

precision

number of matches

time taken

summary.csv: aggregated performance summary across the dataset.
  
