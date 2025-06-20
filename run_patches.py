#!/usr/bin/env python3
"""
Run SIFT, ORB, SuperPoint+LightGlue, and LoFTR on the HPatches dataset.

Metric:  precision  (matches with reprojection error ≤ 3 px after RANSAC).
Outputs:
    results/{algo}_pairs.json   – per-pair stats
    results/summary.csv         – overall table
    visualisations/*            – optional match visualisations (disabled by default)
"""

from __future__ import annotations
import argparse, json, time, csv
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


import sys
#sys.path.append("/home/db-20997/Documents/hacettepe/CMP712 - MACHINE LEARNING/LightGlue")
#sys.path.append("/home/db-20997/Documents/hacettepe/CMP719 - COMPUTER VISION/FinalProject/LoFTR/")  # lightglue klasörünün tam path'i

# ---------- Learned models ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SuperPoint + LightGlue

#default_cfg['coarse']['resolution'] = [240, 320] 


def list_hpatches_sequences(root: Path, ext: str):
    """Return [(seq_name, ref_path, [(tgt_path, H_txt_path), …])]"""
    for seq in sorted(root.iterdir()):
        imgs = sorted(seq.glob(f"*{ext}"))
        if len(imgs) < 2:
            continue
        pairs = []
        for i, img_t in enumerate(imgs[1:], 2):
            H_txt = seq / f"H_1_{i}"
            if H_txt.exists():
                pairs.append((img_t, H_txt))
        if pairs:
            yield seq.name, imgs[0], pairs

def list_deneme(root: Path, ext: str):
    """Return [(seq_name, ref_path, [(tgt_path, H_txt_path), …])]"""
    for seq in sorted(root.iterdir()):
        imgs = sorted(seq.glob(f"*{ext}"))
        if len(imgs) < 2:
            continue
        pairs = []
        for i, img_t in enumerate(imgs[1:], 2):
            H_txt = seq / f"H1to{i}p"
            if H_txt.exists():
                pairs.append((img_t, H_txt))
        if pairs:
            yield seq.name, imgs[0], pairs

def list_oxford_sequences(root: Path):
    """
    Same idea for Oxford-Affine:
      seq/
        img1.ppm … img6.ppm
        H1to2p  H1to3p …
    """
    for seq in sorted(root.iterdir()):
        img1 = seq / "img1.ppm"
        if not img1.exists():
            continue
        pairs = []
        for i in range(2, 7):
            img_t = seq / f"img{i}.ppm"
            H_txt = seq / f"H1to{i}p"
            if img_t.exists() and H_txt.exists():
                pairs.append((img_t, H_txt))
        if pairs:
            yield seq.name, img1, pairs

def glue_keypoints(f0, f1, out):
    """
    Return 2×N NumPy arrays with the matched key-point coordinates, no matter
    which LightGlue revision is installed (0.4.0 … current HEAD).
    """

    # --- remove batch-dim & move to CPU for every tensor dict -----------
    f0, f1, out = [rbd(x) for x in (f0, f1, out)]

    # ---------- new & old API detection ---------------------------------
    if "matches" in out:                       # newest versions
        pairs = out["matches"]                 # tensor | list | np
    elif "matches0" in out:                    # 0.4.0 legacy
        idx0   = out["matches0"].cpu().numpy()
        valid  = idx0 > -1
        kp0 = f0["keypoints"].cpu().numpy()[valid]
        kp1 = f1["keypoints"].cpu().numpy()[idx0[valid]]
        return kp0, kp1
    else:
        raise RuntimeError("LightGlue output has no matches key")

    # convert pairs → CPU int64 NumPy array   (handles tensor / list)
    pairs = torch.as_tensor(pairs, device="cpu", dtype=torch.long).numpy()
    if pairs.size == 0:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

    kp0_all = f0["keypoints"].cpu().numpy()
    kp1_all = f1["keypoints"].cpu().numpy()

    # guard against out-of-range indices
    mask = (pairs[:, 0] < len(kp0_all)) & (pairs[:, 1] < len(kp1_all))
    pairs = pairs[mask]
    if pairs.size == 0:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

    return kp0_all[pairs[:, 0]], kp1_all[pairs[:, 1]]

# ---------- Helper utils ----------
def load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img

def ransac_inliers(kp1, kp2, h_gt, thresh=3.0):
    """Return boolean mask of matches that reproject within `thresh` px."""
    pts1 = cv2.convertPointsToHomogeneous(kp1)[:, 0, :]
    proj = (h_gt @ pts1.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    err = np.linalg.norm(proj - kp2, axis=1)
    return err < thresh

def evaluate_pair(img1, img2, H_1_2, algo: str):
    img1, H_adj = resize_and_scaleH(img1,  np.eye(3), 640)   # reference is identity
    img2, H_1_2 = resize_and_scaleH(img2, H_1_2, 640)        # target + adjusted H

    
    t0 = time.time()

    if algo == "sift":
        sift = cv2.SIFT_create()
        k1, d1 = sift.detectAndCompute(img1, None)
        k2, d2 = sift.detectAndCompute(img2, None)
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(d1, d2)

        kp1 = np.float32([k1[m.queryIdx].pt for m in matches])
        kp2 = np.float32([k2[m.trainIdx].pt for m in matches])

    elif algo == "orb":
        orb = cv2.ORB_create(5000)
        k1, d1 = orb.detectAndCompute(img1, None)
        k2, d2 = orb.detectAndCompute(img2, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(d1, d2)
        kp1 = np.float32([k1[m.queryIdx].pt for m in matches])
        kp2 = np.float32([k2[m.trainIdx].pt for m in matches])

    elif algo == "sp_lg":
        with torch.no_grad():
            t1 = torch.from_numpy(img1).float().div_(255.)[None, None].to(device)
            t2 = torch.from_numpy(img2).float().div_(255.)[None, None].to(device)
    
            f1 = sp.extract(t1)
            f1["image_size"] = torch.tensor(t1.shape[-2:], device=device)
            f2 = sp.extract(t2)
            f2["image_size"] = torch.tensor(t2.shape[-2:], device=device)
    
            out = lg_matcher({"image0": f1, "image1": f2})
            kp1, kp2 = glue_keypoints(f1, f2, out)

    elif algo == "loftr":
        # ❶ pad both images so H,W are multiples of 8
        img1_p = pad8(img1)
        img2_p = pad8(img2)
    
        data = {
            "image0": torch.from_numpy(img1_p).float().div_(255.)[None, None].to(device),
            "image1": torch.from_numpy(img2_p).float().div_(255.)[None, None].to(device),
        }
    
        with torch.no_grad(), torch.cuda.amp.autocast():
            loftr(data)                       # writes output in-place
    
        kp1 = data["mkpts0_f"].cpu().numpy()  # matched key-points
        kp2 = data["mkpts1_f"].cpu().numpy()

    else:
        raise ValueError(algo)

    if len(kp1) < 4:
        return {"n_matches": 0, "precision": 0.0, "time": time.time() - t0}

    inliers = ransac_inliers(kp1, kp2, H_1_2)
    precision = inliers.mean()
    
    # ---------- flush unused GPU memory ----------
    #torch.cuda.ipc_collect()   # releases CUDAPy objects from previous seq
    torch.cuda.empty_cache()      # ❷ release cached blocks
#    gc.collect()                  # ❸ let Python free CPU tensors too

    return {
        "n_matches": len(kp1),
        "precision": float(precision),
        "time": time.time() - t0,
    }

def pad8(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    nh = ((h + 7) // 8) * 8       # next multiple of 8
    nw = ((w + 7) // 8) * 8
    if nh == h and nw == w:
        return img                # already OK
    return np.pad(img, ((0, nh - h), (0, nw - w)),
                  mode="constant", constant_values=0)

def resize_and_scaleH(img: np.ndarray, H: np.ndarray,
                      max_side: int = 640) -> tuple[np.ndarray, np.ndarray]:
    h, w = img.shape
    scale = max(h, w) / max_side if max(h, w) > max_side else 1.0
    if scale == 1.0:
        return img, H
    nh, nw = int(round(h / scale)), int(round(w / scale))
    img_rs = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    # update homography so that P' = S · H · P,  where S = diag(s,s,1)
    S = np.diag([1/scale, 1/scale, 1])
    H_rs = S @ H @ np.linalg.inv(S)        # keeps 3×3 structure
    return img_rs, H_rs

def run_one_dataset(args, kind):
    """kind ∈ {'hpatches','oxford'}  – returns per-algo result dict"""
    root = Path(args.root).expanduser()
    ext  = args.ext
    if kind == "hpatches":
        seq_iter = list_hpatches_sequences(root / "hpatches-sequences-release", ext)
    else:
        seq_iter = list_oxford_sequences(root / "oxford-affine")

    results = {a: [] for a in args.algos}

    for seq_name, ref_path, tgt_pairs in tqdm(list(seq_iter), desc=f"{kind} seq"):
        ref_img = load_gray(ref_path)
        for img_path, H_path in tgt_pairs:
            img_i = load_gray(img_path)
            H = np.loadtxt(H_path)
            for algo in args.algos:
                st = evaluate_pair(ref_img, img_i, H, algo)
                st.update(sequence=f"{kind}:{seq_name}",
                          pair=f"{ref_path.name}->{img_path.name}")
                results[algo].append(st)
    return results
# ---------- Main loop ----------
def run(args):
    all_results = {a: [] for a in args.algos}

    datasets = ["hpatches", "oxford"] if args.dataset == "both" else [args.dataset]
    for kind in datasets:
        res = run_one_dataset(args, kind)
        for a in args.algos:
            all_results[a].extend(res[a])

    # -------- save per-algo JSON ----------
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for algo, recs in all_results.items():
        with open(out_dir / f"{algo}_pairs.json", "w") as f:
            json.dump(recs, f, indent=2)

    # -------- combined summary.csv --------
    with open(out_dir / "summary.csv", "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["algo","mean_precision","mean_time(s)","avg_matches"])
        for algo, recs in all_results.items():
            if not recs: continue
            mp = np.mean([r["precision"] for r in recs])
            mt = np.mean([r["time"]      for r in recs])
            mm = np.mean([r["n_matches"] for r in recs])
            wr.writerow([algo, f"{mp:.3f}", f"{mt:.3f}", f"{mm:.1f}"])
    print("Done – results in", out_dir.resolve())

# ---------- CLI ----------
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run several matchers.")
    parser.add_argument("--root", required=True, help="Parent folder that contains both datasets")
    parser.add_argument("--out_dir", default="/home/db-20997/Documents/hacettepe/CMP719 - COMPUTER VISION/FinalProject")
    parser.add_argument("--ext", default=".ppm", help="Image file extension in HPatches (e.g., .ppm | .png)")
    parser.add_argument("--algos", nargs="+", default=["sift","orb","sp_lg","loftr"],
                        choices=["sift","orb","sp_lg","loftr"])
    parser.add_argument("--dataset", default="both", choices=["hpatches", "oxford", "both"],
                        help="Dataset selector: hpatches | oxford | both (default)")
    parser.add_argument("--lg_path", required=True, help="Path to LightGlue repo")
    parser.add_argument("--lft_path", required=True, help="Path to LoFTR repo")
    
    args = parser.parse_args()
    
    # Append repo paths
    sys.path.append(args.lg_path)
    sys.path.append(str(Path(args.lft_path).resolve()))
    sys.path.append(str((Path(args.lft_path) / "src").resolve()))

    from lightglue import SuperPoint, LightGlue
    from lightglue.utils import rbd 
    #loftr = torch.hub.load("zju3dv/LoFTR", "loftr_outdoor", pretrained=True).eval().to(device)
    
    #from src.utils.plotting import make_matching_figure
    from src.loftr import LoFTR, default_cfg
    
    sp = SuperPoint(max_num_keypoints=2048).eval().to(device)
    lg_matcher = LightGlue(features="superpoint").eval().to(device)
    matcher = LoFTR(config=default_cfg)
      
    image_type = 'outdoor'  
    loftr_ckpt_path = Path(args.lft_path) / f"{image_type}_ds.ckpt"
    matcher.load_state_dict(torch.load(loftr_ckpt_path)['state_dict'])  
      
    loftr = matcher.eval().to(device)
    
    run(args)