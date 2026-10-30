import argparse
import os

import h5py
import pandas as pd

from elf.evaluation import matching, symmetric_best_dice_score
from synapse_net.file_utils import read_mrc  # adjust if your read_mrc lives elsewhere

MRC_EXTENSIONS = (".mrc", ".rec")
H5_EXTENSIONS = (".h5", ".hdf5")
SUPPORTED_EXTENSIONS = MRC_EXTENSIONS + H5_EXTENSIONS


def _is_mrc(path):
    return path.lower().endswith(MRC_EXTENSIONS)


def _is_h5(path):
    return path.lower().endswith(H5_EXTENSIONS)


def load_volume(path, group, key):
    """Load a volume from an mrc/rec file or from `group/key` inside an h5 file.

    For mrc/rec files the group and key are ignored.
    """
    if _is_mrc(path):
        data, _ = read_mrc(path)
        return data
    if _is_h5(path):
        with h5py.File(path, "r") as f:
            return f[group][key][:]
    raise ValueError(f"Unsupported file format: {path}. Expected one of {SUPPORTED_EXTENSIONS}.")


def evaluate(labels, vesicles):
    assert labels.shape == vesicles.shape, f"Shape mismatch: labels {labels.shape} vs vesicles {vesicles.shape}"
    stats = matching(vesicles, labels)
    sbd = symmetric_best_dice_score(vesicles, labels)
    return [stats["f1"], stats["precision"], stats["recall"], sbd]


def summarize_eval(results):
    metrics = ["f1-score", "precision", "recall", "SBD score"]
    summary = results[["dataset"] + metrics].groupby("dataset").mean().reset_index("dataset")
    total = results[metrics].mean().values.tolist()
    # Append the overall mean as an extra row (instead of overwriting the last dataset row).
    summary.loc[len(summary)] = ["all"] + total
    table = summary.to_markdown(index=False)
    print(table)


def evaluate_file(labels_path, vesicles_path, model_name, segment_key, anno_key, mask_key=None):
    print(f"Evaluate labels {labels_path} and vesicles {vesicles_path}")

    ds_name = os.path.basename(os.path.dirname(labels_path))
    tomo = os.path.basename(labels_path)

    # Get the ground truth (and optional mask).
    gt = load_volume(labels_path, "labels", anno_key)

    mask = None
    if mask_key is not None:
        if not _is_h5(labels_path):
            raise ValueError("--mask_key is only supported when the labels are stored in an h5 file.")
        mask = load_volume(labels_path, "labels", mask_key)

    # Get the vesicle segmentation.
    vesicles = load_volume(vesicles_path, "vesicles", segment_key)

    if mask is not None:
        gt[mask == 0] = 0
        vesicles[mask == 0] = 0

    # Evaluate the match of ground truth and vesicles.
    scores = evaluate(gt, vesicles)

    # Store results.
    result_folder = "/scratch/htc/digbinedion/synapse-net/scripts/cooper/evaluation_results"
    os.makedirs(result_folder, exist_ok=True)
    result_path = os.path.join(result_folder, f"evaluation_{model_name}.csv")
    print("Evaluation results are saved to:", result_path)

    if os.path.exists(result_path):
        results = pd.read_csv(result_path)
    else:
        results = None

    res = pd.DataFrame(
        [[ds_name, tomo] + scores], columns=["dataset", "tomogram", "f1-score", "precision", "recall", "SBD score"]
    )
    if results is None:
        results = res
    else:
        results = pd.concat([results, res])
    results.to_csv(result_path, index=False)

    # Print results.
    summarize_eval(results)


def _stem(fname):
    return os.path.splitext(fname)[0]


def evaluate_folder(labels_path, vesicles_path, model_name, segment_key, anno_key, mask_key=None):
    print(f"Evaluating folder {vesicles_path}")
    print(f"Using labels stored in {labels_path}")

    # Match files by name without extension, so e.g. tomo1.h5 labels pair with tomo1.mrc vesicles.
    label_files = {
        _stem(f): f for f in sorted(os.listdir(labels_path)) if f.lower().endswith(SUPPORTED_EXTENSIONS)
    }
    vesicle_files = [f for f in sorted(os.listdir(vesicles_path)) if f.lower().endswith(SUPPORTED_EXTENSIONS)]

    for vesicle_file in vesicle_files:
        label_file = label_files.get(_stem(vesicle_file))
        if label_file is None:
            print(f"No labels found for {vesicle_file}, skipping.")
            continue
        evaluate_file(os.path.join(labels_path, label_file), os.path.join(vesicles_path, vesicle_file),
                      model_name, segment_key, anno_key, mask_key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--labels_path", required=True, help="Labels file or folder (.h5/.hdf5/.mrc/.rec).")
    parser.add_argument("-v", "--vesicles_path", required=True,
                        help="Vesicle segmentation file or folder (.h5/.hdf5/.mrc/.rec).")
    parser.add_argument("-n", "--model_name", required=True)
    parser.add_argument("-sk", "--segment_key", help="Key under 'vesicles' in the h5 segmentation (ignored for mrc).")
    parser.add_argument("-ak", "--anno_key", help="Key under 'labels' in the h5 labels (ignored for mrc).")
    parser.add_argument("-m", "--mask_key", help="Key under 'labels' for a mask (h5 labels only).")
    args = parser.parse_args()

    vesicles_path = args.vesicles_path
    if os.path.isdir(vesicles_path):
        evaluate_folder(args.labels_path, vesicles_path, args.model_name, args.segment_key, args.anno_key,
                        args.mask_key)
    else:
        evaluate_file(args.labels_path, vesicles_path, args.model_name, args.segment_key, args.anno_key, args.mask_key)


if __name__ == "__main__":
    main()