"""
Generate SPIN predictions using official preprocessed 3DPW test data.

Uses SPIN's official 3dpw_test.npz for proper GT alignment.
"""

import sys
import os

SPIN_DIR = os.environ.get("SPIN_ROOT", "third_party/SPIN")
os.chdir(SPIN_DIR)
sys.path.insert(0, SPIN_DIR)

import torch
import numpy as np
import pickle
import cv2
from pathlib import Path
from tqdm import tqdm
from torchvision.transforms import Normalize
import smplx
import argparse

from models import hmr
import config
import constants
from utils.imutils import crop


def process_image(img, center, scale, input_res=224):
    """Process image with given center and scale."""
    normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)

    img_cropped = crop(img, center, scale, (input_res, input_res))
    img_tensor = img_cropped.astype(np.float32) / 255.
    img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1)
    norm_img = normalize_img(img_tensor.clone())[None]

    return norm_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=os.path.join(os.environ.get("SPIN_ROOT", "third_party/SPIN"), "data/model_checkpoint.pt"))
    parser.add_argument('--data_npz', default=os.path.join(os.environ.get("SPIN_ROOT", "third_party/SPIN"), "data/dataset_extras/3dpw_test.npz"))
    parser.add_argument('--img_dir', default=os.environ.get("THREEDPW_ROOT", "data/3DPW"))
    parser.add_argument('--output', default=os.path.join(os.environ.get("PRED_DIR", "data"), "spin_3dpw_official.pkl"))
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_samples', type=int, default=None)
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else \
             torch.device('mps') if torch.backends.mps.is_available() else \
             torch.device('cpu')
    print(f"Using device: {device}")

    # Load SPIN model
    print("\n1. Loading SPIN model...")
    model = hmr(config.SMPL_MEAN_PARAMS).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    print("   SPIN model loaded")

    # Load SMPL model
    print("\n2. Loading SMPL model...")
    smpl = smplx.create(
        model_path=os.environ.get("SMPL_MODELS_DIR", os.path.expanduser("~/.smplx/models")),
        model_type='smpl',
        gender='neutral',
        batch_size=1
    ).to(device)

    # Load J_regressor for H36M joints (same as official eval)
    J_regressor_h36m = torch.tensor(
        np.load(os.path.join(os.environ.get("SPIN_ROOT", "third_party/SPIN"), "data/J_regressor_h36m.npy")),
        dtype=torch.float32
    ).to(device)
    print("   SMPL + J_regressor loaded")

    # Load preprocessed data
    print("\n3. Loading preprocessed 3DPW test data...")
    data = np.load(args.data_npz)
    imgnames = data['imgname']
    centers = data['center']
    scales = data['scale']
    gt_poses = data['pose']
    gt_shapes = data['shape']
    genders = data['gender']

    n_total = len(imgnames)
    if args.max_samples:
        n_total = min(n_total, args.max_samples)
    print(f"   Total samples: {n_total}")

    # Storage
    all_predictions = {
        'pred_rotmat': [],
        'pred_betas': [],
        'pred_camera': [],
        'pred_vertices': [],
        'pred_joints_h36m': [],  # H36M 17 joints from regressor
        'gt_pose': [],
        'gt_betas': [],
        'gt_joints_h36m': [],    # H36M 17 joints from regressor
        'imgname': [],
        'center': [],
        'scale': [],
        'gender': [],
    }

    img_dir = Path(args.img_dir)
    processed = 0
    skipped = 0

    print("\n4. Processing images...")
    for i in tqdm(range(n_total)):
        imgname = imgnames[i]
        center = centers[i]
        scale = scales[i]
        gt_pose = gt_poses[i]
        gt_shape = gt_shapes[i]
        gender = genders[i]

        # Load image
        img_path = img_dir / imgname
        if not img_path.exists():
            skipped += 1
            continue

        try:
            img = cv2.imread(str(img_path))[:,:,::-1].copy()
            norm_img = process_image(img, center, scale)

            with torch.no_grad():
                # SPIN prediction
                pred_rotmat, pred_betas, pred_camera = model(norm_img.to(device))

                # Get pred vertices and joints
                pred_output = smpl(
                    betas=pred_betas,
                    body_pose=pred_rotmat[:,1:],
                    global_orient=pred_rotmat[:,0].unsqueeze(1),
                    pose2rot=False
                )
                pred_vertices = pred_output.vertices

                # H36M joints from vertices (same as official eval)
                pred_joints_h36m = torch.matmul(J_regressor_h36m[None], pred_vertices)

                # Get GT vertices and joints
                gt_pose_t = torch.tensor(gt_pose[None], dtype=torch.float32, device=device)
                gt_shape_t = torch.tensor(gt_shape[None], dtype=torch.float32, device=device)

                gt_output = smpl(
                    betas=gt_shape_t,
                    global_orient=gt_pose_t[:, :3],
                    body_pose=gt_pose_t[:, 3:]
                )
                gt_vertices = gt_output.vertices
                gt_joints_h36m = torch.matmul(J_regressor_h36m[None], gt_vertices)

            # Store
            all_predictions['pred_rotmat'].append(pred_rotmat.cpu().numpy())
            all_predictions['pred_betas'].append(pred_betas.cpu().numpy())
            all_predictions['pred_camera'].append(pred_camera.cpu().numpy())
            all_predictions['pred_vertices'].append(pred_vertices.cpu().numpy())
            all_predictions['pred_joints_h36m'].append(pred_joints_h36m.cpu().numpy())
            all_predictions['gt_pose'].append(gt_pose)
            all_predictions['gt_betas'].append(gt_shape)
            all_predictions['gt_joints_h36m'].append(gt_joints_h36m.cpu().numpy())
            all_predictions['imgname'].append(imgname)
            all_predictions['center'].append(center)
            all_predictions['scale'].append(scale)
            all_predictions['gender'].append(gender)

            processed += 1

        except Exception as e:
            print(f"   Error on {imgname}: {e}")
            skipped += 1
            continue

    # Convert to numpy
    print(f"\n5. Converting to numpy ({processed} samples)...")
    for key in ['pred_rotmat', 'pred_betas', 'pred_camera', 'pred_vertices',
                'pred_joints_h36m', 'gt_joints_h36m']:
        all_predictions[key] = np.concatenate(all_predictions[key], axis=0)

    all_predictions['gt_pose'] = np.array(all_predictions['gt_pose'])
    all_predictions['gt_betas'] = np.array(all_predictions['gt_betas'])
    all_predictions['center'] = np.array(all_predictions['center'])
    all_predictions['scale'] = np.array(all_predictions['scale'])

    # Calculate metrics
    print("\n6. Calculating metrics...")
    H36M_TO_J14 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]

    pred_j14 = all_predictions['pred_joints_h36m'][:, H36M_TO_J14, :]
    gt_j14 = all_predictions['gt_joints_h36m'][:, H36M_TO_J14, :]

    # Root-relative
    pred_j14_rel = pred_j14 - all_predictions['pred_joints_h36m'][:, [0], :]
    gt_j14_rel = gt_j14 - all_predictions['gt_joints_h36m'][:, [0], :]

    # MPJPE
    mpjpe = np.sqrt(np.sum((pred_j14_rel - gt_j14_rel) ** 2, axis=2)).mean() * 1000

    # PA-MPJPE
    def compute_similarity_transform(S1, S2):
        transposed = False
        if S1.shape[0] != 3:
            S1 = S1.T
            S2 = S2.T
            transposed = True
        mu1 = S1.mean(axis=1, keepdims=True)
        mu2 = S2.mean(axis=1, keepdims=True)
        X1 = S1 - mu1
        X2 = S2 - mu2
        var1 = np.sum(X1**2)
        K = X1.dot(X2.T)
        U, s, Vh = np.linalg.svd(K)
        V = Vh.T
        Z = np.eye(U.shape[0])
        Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
        R = V.dot(Z.dot(U.T))
        scale = np.trace(R.dot(K)) / var1
        t = mu2 - scale*(R.dot(mu1))
        S1_hat = scale*R.dot(S1) + t
        if transposed:
            S1_hat = S1_hat.T
        return S1_hat

    pa_errors = []
    for i in range(len(pred_j14_rel)):
        pred_pa = compute_similarity_transform(pred_j14_rel[i], gt_j14_rel[i])
        error = np.sqrt(np.sum((pred_pa - gt_j14_rel[i]) ** 2, axis=1)).mean()
        pa_errors.append(error)
    pa_mpjpe = np.mean(pa_errors) * 1000

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(all_predictions, f)

    print("\n" + "="*60)
    print("SPIN Baseline Results on 3DPW Test (Official Preprocessing)")
    print("="*60)
    print(f"  Processed: {processed}")
    print(f"  Skipped: {skipped}")
    print(f"  MPJPE:    {mpjpe:.1f} mm")
    print(f"  PA-MPJPE: {pa_mpjpe:.1f} mm (paper: 59.2 mm)")
    print(f"\n  Saved to: {output_path}")
    print("="*60)


if __name__ == '__main__':
    main()
