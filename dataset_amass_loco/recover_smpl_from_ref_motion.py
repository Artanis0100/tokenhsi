import os
import os.path as osp
import argparse

import numpy as np
import torch

from lpanlib.poselib.skeleton.skeleton3d import SkeletonMotion
from lpanlib.poselib.core.rotation3d import quat_angle_axis


def motion_to_smpl_like_params(motion: SkeletonMotion):
    """
    Convert a `SkeletonMotion` (saved in `ref_motion.npy`) back to a
    SMPL-like parameter dict with keys:
        - 'poses':  (T, J*3)  angle-axis / exp-map, root first
        - 'trans':  (T, 3)    root translation
        - 'fps':    scalar

    IMPORTANT:
        This does NOT exactly reconstruct the original AMASS `raw_params`
        used in `generate_motion.py`. Information is lost during:
          - retargeting to a different skeleton,
          - ground/height adjustments,
          - joint projection (`project_joints[_simple]`).

        What we return is a *consistent* SMPL-like representation of the
        motion contained in `ref_motion.npy`, in the joint space of the
        motion's own skeleton.
    """
    # Local joint rotations as quaternions in xyzw format
    local_rot = motion.local_rotation  # (T, J, 4)
    root_trans = motion.root_translation  # (T, 3)

    # Convert quaternions (xyzw) -> (angle, axis)
    # quat_angle_axis expects unit quaternions with real part at index 3
    angles, axes = quat_angle_axis(local_rot)
    # Build angle-axis 3-vectors: axis * angle
    expmap = axes * angles.unsqueeze(-1)  # (T, J, 3)

    T, J, _ = expmap.shape
    poses_aa = expmap.reshape(T, J * 3)  # (T, J*3)

    out = {
        "poses": poses_aa.cpu().numpy().astype(np.float32),
        "trans": root_trans.cpu().numpy().astype(np.float32),
        "fps": float(motion.fps),
    }
    return out


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert a ref_motion.npy (SkeletonMotion) back to a SMPL-like "
            "trajectory file with 'poses', 'trans', and 'fps'."
        )
    )
    parser.add_argument(
        "--ref_motion",
        type=str,
        required=True,
        help="Path to ref_motion.npy produced by generate_motion.py",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output .npy path for recovered SMPL-like params. "
            "If not given, writes 'recovered_smpl_params.npy' "
            "next to this script."
        ),
    )

    args = parser.parse_args()

    ref_motion_path = args.ref_motion
    if args.output is None:
        script_dir = osp.dirname(osp.abspath(__file__))
        out_path = osp.join(script_dir, "recovered_smpl_params.npy")
    else:
        out_path = args.output

    if not osp.isfile(ref_motion_path):
        raise FileNotFoundError(f"ref_motion file not found: {ref_motion_path}")

    # Load SkeletonMotion from file (Serializable.from_file)
    motion = SkeletonMotion.from_file(ref_motion_path)

    params = motion_to_smpl_like_params(motion)

    os.makedirs(osp.dirname(out_path), exist_ok=True)
    np.save(out_path, params)
    print(f"Saved recovered SMPL-like params to: {out_path}")


if __name__ == "__main__":
    main()


