# making executable script with user-defined environment

import os
from string import Template

def makedir():
    os.makedirs("executables", exist_ok = True, mode=0o755)
    os.makedirs("results", exist_ok = True, mode=0o755)
    os.makedirs("results/condor_logs", exist_ok = True, mode=0o755)
    os.makedirs("results/condor_logs/residuals_errors", exist_ok = True, mode=0o755)
    os.makedirs("results/condor_logs/residuals_logs", exist_ok = True, mode=0o755)
    os.makedirs("results/condor_logs/residuals_outputs", exist_ok = True, mode=0o755)
    os.makedirs("results/condor_logs/select_errors", exist_ok = True, mode=0o755)
    os.makedirs("results/condor_logs/select_logs", exist_ok = True, mode=0o755)
    os.makedirs("results/condor_logs/select_outputs", exist_ok = True, mode=0o755)
    os.makedirs("results/condor_logs/sim_errors", exist_ok = True, mode=0o755)
    os.makedirs("results/condor_logs/sim_logs", exist_ok = True, mode=0o755)
    os.makedirs("results/condor_logs/sim_outputs", exist_ok = True, mode=0o755)
    os.makedirs("results/macros", exist_ok = True, mode=0o755)
    os.makedirs("results/pars", exist_ok = True, mode=0o755)
    os.makedirs("results/plots", exist_ok = True, mode=0o755)

if __name__ == "__main__":

    virtpythonpath = input("Enter VIRTPYTHONPATH: ").strip()
    ratpath = input("Enter RATPATH: ").strip()
    simsavepath = input("Enter SIMSAVEPATH: ").strip()
    runid = input("RUNID(6-digit): ").strip()
    working_dir = os.getcwd()
    #print(virtpythonpath)
    #print(ratpath)
    #print(simsavepath)
    makedir()

    
    with open("executables_template/checkconverge.sh") as f:
        tempconv_text = f.read()
    with open("executables_template/delete_simulations.sh") as f:
        tempdsim_text = f.read()
    with open("executables_template/select_parameters.sh") as f:
        tempsel_text = f.read()
    with open("executables_template/submit_simulations.sh") as f:
        tempsim_text = f.read()
    with open("executables_template/time_residuals.sh") as f:
        temptres_text = f.read()
    with open("set_env_template.sh") as f:
        tempsetenv_text = f.read()

    conv_text = Template(tempconv_text)
    dsim_text = Template(tempdsim_text)
    sel_text  = Template(tempsel_text)
    sim_text  = Template(tempsim_text)
    tres_text = Template(temptres_text)
    setenv_text=Template(tempsetenv_text)


    conv_out = conv_text.substitute(RATPATH=ratpath,VIRTPYTHONPATH=virtpythonpath)
    dsim_out = dsim_text.substitute(VIRTPYTHONPATH=virtpythonpath,SIMSAVEPATH=simsavepath)
    sel_out = sel_text.substitute(RATPATH=ratpath,VIRTPYTHONPATH=virtpythonpath)
    sim_out = sim_text.substitute(RATPATH=ratpath,VIRTPYTHONPATH=virtpythonpath,SIMSAVEPATH=simsavepath,RUNID=runid)
    tres_out = tres_text.substitute(RATPATH=ratpath,VIRTPYTHONPATH=virtpythonpath)
    env_out = setenv_text.substitute(RATPATH=ratpath,VIRTPYTHONPATH=virtpythonpath,WORKINGDIR=working_dir)


    with open("executables/checkconverge.sh","w") as f:
        f.write(conv_out)
    with open("executables/delete_simulations.sh","w") as f:
        dsim_out = f.write(dsim_out)
    with open("executables/select_parameters.sh","w") as f:
        sel_out = f.write(sel_out)
    with open("executables/submit_simulations.sh","w") as f:
        sim_out = f.write(sim_out)
    with open("executables/time_residuals.sh","w") as f:
        tres_out = f.write(tres_out)
    with open("executables/set_env.sh","w") as f:
        tres_out = f.write(tres_out)

    os.chmod("executables/checkconverge.sh", 0o755)
    os.chmod("executables/delete_simulations.sh", 0o755)
    os.chmod("executables/select_parameters.sh", 0o755)
    os.chmod("executables/submit_simulations.sh", 0o755)
    os.chmod("executables/time_residuals.sh", 0o755)

