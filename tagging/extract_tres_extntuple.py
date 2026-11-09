import argparse
import glob
import os

import numpy as np
import uproot


ENERGY_CUTS = {
    "Bi210": (0.7, 1.1),
    "Bi214": (1.25, 3.0),
    "Po214": (0.7, 1.1),
}

BRANCHES_BY_PARTICLE = {
    "Bi210": {
        "energy": "energy",
        "posr": "posr_av",
        "tres": "tres",
    },
    "Bi214": {
        "energy": "energy",
        "posr": "posr_av",
        "tres": "tres",
    },
    "Po214": {
        "energy": "energy",
        "posr": "posr_av",
        "tres": "tres",
    },
}

TREE_BY_PARTICLE = {
    "Bi210": "DelayT",
    "Bi214": "BiT",
    "Po214": "PoT",
}


def _resolve_input_files(input_pattern):
    if os.path.isdir(input_pattern):
        return sorted(glob.glob(os.path.join(input_pattern, "*.root")))

    matched = sorted(glob.glob(input_pattern))
    if matched:
        return [path for path in matched if path.endswith(".root")]

    if input_pattern.endswith(".root") and os.path.exists(input_pattern):
        return [input_pattern]

    return []


def _flatten_tres_array(tres_values):
    if len(tres_values) == 0:
        return np.array([], dtype=np.float64)

    first = tres_values[0]
    if np.isscalar(first):
        return np.asarray(tres_values, dtype=np.float64)

    chunks = []
    for entry in tres_values:
        arr = np.asarray(entry, dtype=np.float64).ravel()
        if arr.size > 0:
            chunks.append(arr)

    if not chunks:
        return np.array([], dtype=np.float64)

    return np.concatenate(chunks)


def extract_residuals(input_pattern, particletype):
    """Extract tres values from ntuple ROOT files using uproot."""
    if (
        particletype not in ENERGY_CUTS
        or particletype not in BRANCHES_BY_PARTICLE
        or particletype not in TREE_BY_PARTICLE
    ):
        raise ValueError(f"Wrong input Particle Type {particletype} in extract_residuals()")

    branch_names = BRANCHES_BY_PARTICLE[particletype]
    energy_branch = branch_names["energy"]
    posr_branch = branch_names["posr"]
    tres_branch = branch_names["tres"]
    tree_name = TREE_BY_PARTICLE[particletype]
    e_min, e_max = ENERGY_CUTS[particletype]

    file_list = _resolve_input_files(input_pattern)
    if not file_list:
        print(f"No ROOT files found for input: {input_pattern}")
        return np.array([], dtype=np.float64)

    required_branches = {posr_branch, energy_branch, tres_branch}
    sources = []
    skipped_count = 0

    for file_path in file_list:
        try:
            with uproot.open(file_path) as root_file:
                if tree_name not in root_file:
                    skipped_count += 1
                    continue

                tree = root_file[tree_name]
                if not required_branches.issubset(set(tree.keys())):
                    skipped_count += 1
                    continue

            sources.append(f"{file_path}:{tree_name}")
        except Exception:
            skipped_count += 1
            continue

    if skipped_count > 0:
        print(f"Skipped {skipped_count} files missing required tree/branches or unreadable files")

    if not sources:
        print("No valid files found after checking tree/branch availability")
        return np.array([], dtype=np.float64)

    # Build a cut expression so uproot will select only surviving events while reading
    #cut_expr = f"({posr_branch} <= 4000.0) & ({energy_branch} >= {e_min}) & ({energy_branch} <= {e_max})"

    try:
        arrays = uproot.concatenate(
            sources,
            [posr_branch, energy_branch, tres_branch],
            library="np"
        )
    except Exception as exc:
        print(f"Failed to read {tree_name} from {len(sources)} files (first: {sources[0]}): {exc}")
        return np.array([], dtype=object)

    posr_av = np.asarray(arrays[posr_branch], dtype=np.float64)
    energy = np.asarray(arrays[energy_branch], dtype=np.float64)
    #print("posrav", posr_av)
    # arrays[tres_branch] may be a jagged array (array of arrays). Build a list
    # of 1D numpy arrays and concatenate them safely.
    tres_raw = arrays[tres_branch]

    # Apply manual masks; an event must pass BOTH:
    # - energy between e_min and e_max MeV (based on particle type)
    # - posr_av <= 4000.0
    energy_mask = (energy >= e_min) & (energy <= e_max)
    posr_mask = (posr_av <= 4000.0)
    mask = energy_mask & posr_mask
    try:
        num_surviving = int(np.count_nonzero(mask))
    except Exception:
        num_surviving = 0
    print(f"Events before mask: {len(posr_av)}, surviving events after AND-mask: {num_surviving}")
    print(f"  energy mask count: {int(np.count_nonzero(energy_mask))}, posr mask count: {int(np.count_nonzero(posr_mask))}")

    # Diagnostic: print tres structure and event counts
    try:
        num_tres_events = len(tres_raw)
    except Exception:
        num_tres_events = "unknown"

    # compute event sizes (may be expensive for many events) and a small sample
    try:
        event_sizes = [np.size(evt) for evt in tres_raw]
        total_hits = int(sum(event_sizes))
        sample_sizes = event_sizes[:10]
    except Exception:
        event_sizes = None
        total_hits = "unknown"
        sample_sizes = None

    print(f"tres raw events: {num_tres_events}, total hits: {total_hits}, sample event sizes: {sample_sizes}")

    if num_surviving == 0:
        print(f"No tres entries after applying manual mask for {particletype}.")
        return np.array([], dtype=np.float64)

    tres_list = []
    surv_indices = np.nonzero(mask)[0]
    for idx in surv_indices:
        try:
            evt = tres_raw[idx]
        except Exception:
            continue
        try:
            arr = np.asarray(evt, dtype=np.float64).ravel()
            if arr.size > 0:
                tres_list.append(arr)
        except Exception:
            # skip malformed entries
            continue

    if not tres_list:
        print(f"No tres entries after filtering for {particletype}.")
        return np.array([], dtype=np.float64)

    tres_flat = np.concatenate(tres_list)
    print(f"Concatenated {num_surviving} surviving entries for {particletype} from {len(sources)} files.")
    print(f"Flattened tres length: {len(tres_flat)} for {particletype}.")

    return np.asarray(tres_flat, dtype=np.float64)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract time residuals from SNO+ ROOT files and save as Numpy.")
    
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Input path pattern (e.g., '/data/snoplus/*.root'). Wrap in quotes if using wildcards.")
    
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output path for the .npy file (e.g., ./output.npy).")
    
    parser.add_argument("-p", "--particle", type=str, choices=["Bi210", "Bi214", "Po214"], required=True,
                        help="Particle type to filter for: Bi210, Bi214, or Po214")

    args = parser.parse_args()

    # 2. Run Extraction using parsed arguments
    results = extract_residuals(args.input, args.particle)

    # 3. Save Output
    if results is not None and len(results) > 0:
        # Ensure directory exists
        out_dir = os.path.dirname(args.output)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Ensure .npy extension
        final_output = args.output if args.output.endswith('.npy') else args.output + ".npy"

        np.save(final_output, results)
        print("-" * 30)
        print(f"Success! Saved {len(results)} residuals to {final_output}")
        print("-" * 30)
    else:
        print("No residuals found or error in processing.")