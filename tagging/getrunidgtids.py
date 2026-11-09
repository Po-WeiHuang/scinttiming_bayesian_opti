import uproot
import os
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


PARTICLE_BRANCHES = {
    "Bi214": {
        "tree": "BiT",
        "eventid": "Bi_eventid",
    },
    "Po214": {
        "tree": "PoT",
        "eventid": "Po_eventid",
    },
}

def process_single_run(filename, ntuple_dir, output_dir, particletype):
    """Processes a single ROOT file to extract event IDs."""
    try:
        branch_info = PARTICLE_BRANCHES[particletype]

        # 1. Construct full paths
        input_path = os.path.join(ntuple_dir, filename)
        
        # 2. Extract Run ID (assuming filename is runid.ntuple.root or similar)
        # Change this split if your naming convention is different
        run_id = filename.split('.')[0] 
        output_path = os.path.join(output_dir, f"{run_id}.txt")

        # 3. Open ROOT and extract the particle-specific event ID branch
        with uproot.open(input_path+".ntuple.root") as f:
            tree = f[branch_info["tree"]]
            event_ids = tree[branch_info["eventid"]].array(library="np")

        # 4. Write to the output directory
        with open(output_path, 'w') as out:
            for eid in event_ids:
                out.write(f"{run_id},{eid}\n")
        
        return f"Done: {run_id}"

    except Exception as e:
        return f"Error on {filename}: {e}"

def main():
    parser = argparse.ArgumentParser(description="Extract event IDs from SNO+ ROOT files.")
    parser.add_argument("-r", required=True, help="Path to the text file containing filenames.")
    parser.add_argument("-n", required=True, help="Directory where the .root files are stored.")
    parser.add_argument("-o", required=True, help="Directory to save the resulting .txt files.")
    parser.add_argument("-p", "--particle", required=True, choices=["Bi214", "Po214"],
                        help="Particle type to select: Bi214 or Po214.")
    
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.o, exist_ok=True)

    # Read the runlist
    with open(args.r, 'r') as f:
        # Filter out empty lines or whitespace
        filenames = [line.strip() for line in f if line.strip()]

    print(f"Processing {len(filenames)} files using multiprocessing...")

    # Use a ProcessPoolExecutor for speed
    with ProcessPoolExecutor() as executor:
        # We pass the directory arguments to each function call using a list comprehension
        futures = [
            executor.submit(process_single_run, fname, args.n, args.o, args.particle) 
            for fname in filenames
        ]
        
        # Use tqdm to show a progress bar
        for future in tqdm(futures, total=len(filenames)):
            result = future.result()
            if "Error" in result:
                print(result)

if __name__ == "__main__":
    main()