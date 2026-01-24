#!/usr/bin/env python3
# inputting params in current_params.json
# will output rads .root files in the directory
import string
from pathlib import Path
import json

def create_macro(params, name, particletype):
    """
    For each round of simulations, we load in a template macro and fill in the
    parameters to sample at. This macro is then used by each of the simulation
    jobs, with a different output file specified in the executable.
    """
    if particletype == "Bi214":
        with open("macros/bi214_template.mac", "r") as infile:
            raw_text = string.Template(infile.read())
    elif particletype == "Po214":
        with open("macros/po214_template.mac", "r") as infile:
            raw_text = string.Template(infile.read())
    else:
        print(f"Wrong input Particle Type {particletype} in create_macro. Should be either Bi214 or Po214")
        exit(1)
    
    #output_text = raw_text.substitute(MATERIAL = "labppo_2p2_scintillator", T1 = params["T1"], T2 = params["T2"],
    #                                  T3 = params["T3"], T4 = params["T4"], TR = params["TR"], A1 = params["A1"],
    #                                  A2 = params["A2"], A3 = params["A3"], A4 = params["A4"])
    output_text = raw_text.substitute(MATERIAL = "labppo_2p2_bismsb_scintillator", T1 = params["T1"], T2 = params["T2"],
                                      T3 = params["T3"], T4 = params["T4"], TR = params["TR"], A1 = params["A1"],
                                      A2 = params["A2"], A3 = params["A3"], A4 = params["A4"])

    with open(f"results/macros/{name}.mac", "w") as outfile:
        outfile.write(output_text)
    
    # dynamically create the submit file for simulations to point to the correct macro file
    
    with open("submit_files/simulate.submit", "w") as outfile:
        outfile.write("""
        executable = executables/submit_simulations.sh
        arguments  = $(Process) {MAC_NAME}.mac
        log        = results/condor_logs/sim_logs/$(ClusterId)_$(Process).log
        output     = results/condor_logs/sim_outputs/$(ClusterId)_$(Process).out
        error      = results/condor_logs/sim_errors/$(ClusterId)_$(Process).err
        priority   = 5
        request_memory = 1024MB
        queue 20
        """.format(MAC_NAME = f"{name}"))
if __name__ == "__main__":

    # Read JSON file
    with open("currentparams.json", "r") as f:
        params = json.load(f)
    iter = params["iter"] + 1
    with open("paramsbound.json", "r") as f:
        boundparams = json.load(f)
    create_macro(params, iter, boundparams["Type"])    


    submit_file = Path("submit_files/simulate.submit")
    text = submit_file.read_text()
    text = text.replace("${MAC_NAME}", f"{iter}.mac")
    submit_file.write_text(text)
