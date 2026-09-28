import argparse
import os

import h5py
import pandas as pd
import mrcfile
import numpy as np

from elf.evaluation import matching, symmetric_best_dice_score


def evaluate(labels, vesicles):
    assert labels.shape == vesicles.shape
    stats = matching(vesicles, labels)
    sbd = symmetric_best_dice_score(vesicles, labels)
    return [stats["f1"], stats["precision"], stats["recall"], sbd]


def load_data(file_path, key=None):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.mrc', '.rec']:
        with mrcfile.open(file_path, permissive=True) as f:
            data = f.data.copy()
        if np.issubdtype(data.dtype, np.floating):
            if data.max() <= 1.0:
                raise ValueError(
                    f"{file_path} looks like a probability map, not a segmentation. "
                    "Threshold + run connected components first."
                )
            data = np.round(data)  # float-encoded integer IDs
        return data.astype(np.int64)
    elif ext in ['.h5', '.hdf5']:
        with h5py.File(file_path, 'r') as f:
            if key is None:
                raise ValueError(f"Key required for HDF5 file: {file_path}")
            return f[key][:]
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def summarize_eval(results):
    summary = results[["dataset", "f1-score", "precision", "recall", "SBD score"]].groupby("dataset").mean().reset_index("dataset")
    total = results[["f1-score", "precision", "recall", "SBD score"]].mean().values.tolist()
    summary = pd.concat([summary, pd.DataFrame([["all"] + total], columns=summary.columns)], ignore_index=True)
    table = summary.to_markdown(index=False)
    print(table)


def evaluate_file(labels_path, vesicles_path, model_name, output_dir,
                   segment_key=None, anno_key=None, mask_path=None):
    print(f"Evaluate labels {labels_path} and vesicles {vesicles_path}")

    ds_name = os.path.basename(os.path.dirname(labels_path))
    tomo = os.path.basename(labels_path)

    # Load ground truth labels (mrc or h5)
    gt = load_data(labels_path, anno_key)

    # Load predicted vesicles (mrc or h5)
    vesicles = load_data(vesicles_path, segment_key)

    # Load mask if provided (separate file, mrc or h5)
    if mask_path is not None:
        mask = load_data(mask_path)
        gt[mask == 0] = 0
        vesicles[mask == 0] = 0

    # Evaluate the match of ground truth and vesicles
    scores = evaluate(gt, vesicles)

    # Store results
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, f"evaluation_{model_name}.csv")
    print("Evaluation results are saved to:", result_path)

    if os.path.exists(result_path):
        results = pd.read_csv(result_path)
    else:
        results = None

    res = pd.DataFrame(
        [[ds_name, tomo] + scores],
        columns=["dataset", "tomogram", "f1-score", "precision", "recall", "SBD score"]
    )
    if results is None:
        results = res
    else:
        results = pd.concat([results, res])

    # Avoid duplicate rows from re-running the same tomogram
    results = results.drop_duplicates(subset=["dataset", "tomogram"], keep="last")

    results.to_csv(result_path, index=False)

    # Print results
    summarize_eval(results)


def evaluate_folder(labels_path, vesicles_path, model_name, output_dir,
                    segment_key=None, anno_key=None, mask_path=None):
    print(f"Evaluating folder {vesicles_path}")
    print(f"Using labels stored in {labels_path}")

    label_files = set(os.listdir(labels_path))
    vesicles_files = sorted(os.listdir(vesicles_path))

    n_matched = 0
    for vesicle_file in vesicles_files:
        stem, ext = os.path.splitext(vesicle_file)
        # prediction files are named <tomo>_prediction.mrc, GT is <tomo>.mrc
        candidates = [vesicle_file, stem.removesuffix("_prediction") + ext]

        label_file = next((c for c in candidates if c in label_files), None)
        if label_file is None:
            print(f"  no matching ground truth for {vesicle_file}, skipping")
            continue

        n_matched += 1
        evaluate_file(
            os.path.join(labels_path, label_file),
            os.path.join(vesicles_path, vesicle_file),
            model_name,
            output_dir,
            segment_key,
            anno_key,
            mask_path,
        )

    if n_matched == 0:
        raise RuntimeError(
            f"No prediction/ground-truth pairs matched between {vesicles_path} and {labels_path}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--labels_path", required=True)
    parser.add_argument("-v", "--vesicles_path", required=True)
    parser.add_argument("-n", "--model_name", required=True)
    parser.add_argument("-o", "--output_dir", required=True)
    parser.add_argument("-sk", "--segment_key", default=None)
    parser.add_argument("-ak", "--anno_key", default=None)
    parser.add_argument("-m", "--mask_path", default=None)
    args = parser.parse_args()

    vesicles_path = args.vesicles_path
    if os.path.isdir(vesicles_path):
        evaluate_folder(
            args.labels_path, vesicles_path, args.model_name, args.output_dir,
            args.segment_key, args.anno_key, args.mask_path,
        )
    else:
        evaluate_file(
            args.labels_path, vesicles_path, args.model_name, args.output_dir,
            args.segment_key, args.anno_key, args.mask_path,
        )


if __name__ == "__main__":
    main()