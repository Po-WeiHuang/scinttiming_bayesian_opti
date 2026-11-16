"""Generate executables/*.sh (and set_env.sh) for the BO loop from executables_template/.

Run from the repository top level. Every setting has a default; give it as a flag, or answer the
prompt (press Enter to keep the default). --yes accepts all defaults without prompting.

    python initial.py                                   # interactive
    python initial.py --rat-version 8.3.2 --yes          # official RAT 8.3.2 env, other defaults
    python initial.py --rat-env /path/to/env.sh --sim-dir /data/.../sims --nevents 200 --yes

The run number is not asked for: it comes from "RunID" in paramsbound.json.
The optics (PPO / bis-MSB) is chosen afterwards with ./clean_files.sh ppo|bismsb.
"""
import argparse
import glob
import os
import re
from string import Template

SNOCAVE = "/data/snoplus/software/snocave_Alma9"
DEFAULTS = {
    "rat_env": f"{SNOCAVE}/set_env.sh",
    "venv": (os.environ["BOVENV"].rstrip("/") + "/bin/activate") if os.environ.get("BOVENV")
            else "/data/snoplus3/weiii/botorch_venv/bin/activate",
    "sim_dir": "/data/snoplus2/weiiiii/pytor_bayesian_opti",
    "nevents": "120",
}
SCRIPTS = ["checkconverge.sh", "delete_simulations.sh", "pre_simulation.sh", "select_parameters.sh",
           "submit_simulations.sh", "time_residuals.sh"]


def makedir():
    for d in ["executables", "results/macros", "results/pars", "results/plots"] + [
            f"results/condor_logs/{step}_{kind}" for step in ("residuals", "select", "sim")
            for kind in ("errors", "logs", "outputs")]:
        os.makedirs(d, exist_ok=True, mode=0o755)


def rat_version_of(env_script):
    """RAT version an env script sources (last active 'rat-X.Y.Z' line), or None."""
    try:
        lines = [l for l in open(env_script) if l.strip() and not l.lstrip().startswith("#")]
    except OSError:
        return None
    found = [m for l in lines for m in re.findall(r"rat-(\d+\.\d+\.\d+)", l)]
    return found[-1] if found else None


def env_for_version(version):
    """Find the official snocave env script that sources the requested RAT version."""
    available = {}
    for script in sorted(glob.glob(f"{SNOCAVE}/*.sh")):
        v = rat_version_of(script)
        if v:
            available.setdefault(v, script)
    if version not in available:
        listing = ", ".join(f"{v} ({s})" for v, s in available.items()) or "none"
        raise SystemExit(f"No official env script for RAT {version}. Available: {listing}. "
                         f"Use --rat-env to give an env script directly.")
    return available[version]


def ask(label, value, default, interactive):
    if value is not None or not interactive:
        return value if value is not None else default
    answer = input(f"{label} [{default}]: ").strip()
    return answer or default


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate the BO loop executables from executables_template/.")
    rat = p.add_mutually_exclusive_group()
    rat.add_argument("--rat-version", help=f"RAT version, e.g. 8.3.2; picks the official env script in {SNOCAVE}")
    rat.add_argument("--rat-env", help=f"RAT environment script to source (default {DEFAULTS['rat_env']})")
    p.add_argument("--venv", help=f"python venv activate script (default {DEFAULTS['venv']})")
    p.add_argument("--sim-dir", help=f"directory for simulated ntuples (default {DEFAULTS['sim_dir']})")
    p.add_argument("--nevents", help=f"events per simulation job (default {DEFAULTS['nevents']})")
    p.add_argument("--yes", action="store_true", help="accept defaults for anything not given; no prompts")
    args = p.parse_args()

    if not os.path.isdir("executables_template") or not os.path.isdir("src"):
        raise SystemExit("Run initial.py from the repository top level (where src/ and executables_template/ are).")
    interactive = not args.yes

    rat_env = env_for_version(args.rat_version) if args.rat_version else \
        ask("RAT env script", args.rat_env, DEFAULTS["rat_env"], interactive)
    venv = ask("Python venv activate script", args.venv, DEFAULTS["venv"], interactive)
    sim_dir = ask("Simulation output directory", args.sim_dir, DEFAULTS["sim_dir"], interactive).rstrip("/")
    nevents = ask("Events per simulation job", args.nevents, DEFAULTS["nevents"], interactive)

    # checks before anything is written
    problems = [f"RAT env script not found: {rat_env}"] * (not os.path.isfile(rat_env)) + \
               [f"venv activate script not found: {venv}"] * (not os.path.isfile(venv)) + \
               [f"events per job must be a positive integer: {nevents}"] * (not nevents.isdigit() or int(nevents) == 0) + \
               [f"simulation directory must be an absolute path: {sim_dir!r}"] * (not sim_dir.startswith("/") or sim_dir == "")
    if problems:
        raise SystemExit("Not written:\n  " + "\n  ".join(problems))
    if not os.path.isdir(sim_dir):
        os.makedirs(sim_dir, exist_ok=True)
        print(f"Created simulation directory {sim_dir}")

    values = dict(RATPATH=rat_env, VIRTPYTHONPATH=venv, SIMSAVEPATH=sim_dir,
                  NEVENTS=nevents, WORKINGDIR=os.getcwd())
    makedir()
    for name in SCRIPTS:
        with open(f"executables_template/{name}") as f:
            text = Template(f.read()).substitute(values)
        with open(f"executables/{name}", "w") as f:
            f.write(text)
        os.chmod(f"executables/{name}", 0o755)
    with open("set_env_template.sh") as f:
        text = Template(f.read()).substitute(values)
    with open("set_env.sh", "w") as f:
        f.write(text)

    print("Wrote executables/" + ", executables/".join(SCRIPTS) + " and set_env.sh with:")
    print(f"  RAT env      : {rat_env}  (RAT {rat_version_of(rat_env) or 'version not detected'})")
    print(f"  venv         : {venv}")
    print(f"  sim output   : {sim_dir}")
    print(f"  events/job   : {nevents}")
    print(f"  repo (PYTHONPATH): {os.getcwd()}")
    print("Next: ./clean_files.sh ppo|bismsb, then condor_submit_dag dag_running/training.dag")
