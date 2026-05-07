import os
import subprocess
import argparse
import sys
import glob

def create_rat_wrapper(output_path):
    """Creates a shell script to source SNOplus env and run RAT."""
    wrapper_path = os.path.join(output_path, "rat_wrapper.sh")
    rat_root = os.environ.get('RATROOT', '')
    if not rat_root:
        print("Error: RATROOT not found. Source your SNOplus env first!")
        sys.exit(1)
        
    env_sh = "/home/huangp/fresh_rat/rat/wei_env_Will.sh"
    shell_content = f"#!/bin/bash\nsource {env_sh}\nrat \"$@\"\n"
    
    with open(wrapper_path, 'w') as f:
        f.write(shell_content)
    os.chmod(wrapper_path, 0o755)
    return wrapper_path

def submit_jobs(gtid_folder, ratds_path, output_path, macro_template):
    log_dir = os.path.join(output_path, "logs")
    macro_dir = os.path.join(output_path, "macros")
    for d in [log_dir, macro_dir]:
        if not os.path.exists(d): os.makedirs(d)

    wrapper_path = create_rat_wrapper(output_path)

    with open(macro_template, 'r') as f:
        template_content = f.read()

    count = 0
    for filename in os.listdir(gtid_folder):
        if filename.endswith(".txt"):
            run_id = filename.rsplit('.', 1)[0]
            
            # 1. FIND ALL SUBRUNS
            # Looks for files like: path/354109.root OR path/354109_*.root
            search_pattern = os.path.join(ratds_path, f"*{run_id}*.root")
            subrun_files = glob.glob(search_pattern)

            if not subrun_files:
                print(f"Warning: No input files found for Run {run_id} in {ratds_path}. Skipping.")
                continue

            # 2. PREPARE MACRO LINES
            # Replace the single placeholder with multiple load commands
            load_commands = "\n".join([f"/rat/inroot/load {f}" for f in subrun_files])
            #print("load_commands ",load_commands)
            
            gtid_full_path = os.path.abspath(os.path.join(gtid_folder, filename))
            output_root = os.path.join(output_path, f"{run_id}.root")
            generated_macro = os.path.join(macro_dir, f"run_{run_id}.mac")

            # Replace placeholders
            # Note: We replaced the specific line '/rat/inroot/load {IN}.root' 
            # with our new block of multiple load commands.
            new_macro = template_content.replace('/rat/inroot/load "{IN}.root"', load_commands)
            new_macro = new_macro.replace('/rat/inroot/load {IN}.root', load_commands)
            new_macro = new_macro.replace("{GRID}", gtid_full_path.replace(".txt", ""))
            new_macro = new_macro.replace("{OUT}", output_root.replace(".root", ""))

            with open(generated_macro, 'w') as f:
                f.write(new_macro)

            # 3. CONDOR SUBMISSION
            sub_content = f"""
executable              = {wrapper_path}
arguments               = {generated_macro}
JobBatchName            = "pruneratds"
output                  = {log_dir}/{run_id}.out
error                   = {log_dir}/{run_id}.err
log                     = {log_dir}/{run_id}.log
universe                = vanilla
getenv                  = True
queue
"""
            sub_filename = f"temp_{run_id}.sub"
            with open(sub_filename, 'w') as f:
                f.write(sub_content)
            
            try:
                subprocess.run(["condor_submit", sub_filename], check=True)
                count += 1
            except subprocess.CalledProcessError:
                print(f"Failed to submit Run {run_id}")
            finally:
                if os.path.exists(sub_filename): os.remove(sub_filename)
            
    print(f"\nDone. Submitted {count} jobs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("gtid_folder")
    parser.add_argument("ratds_path")
    parser.add_argument("output_path")
    parser.add_argument("--template", default="runeventlist.mac")
    args = parser.parse_args()

    submit_jobs(
        os.path.abspath(args.gtid_folder),
        os.path.abspath(args.ratds_path),
        os.path.abspath(args.output_path),
        args.template,
    )