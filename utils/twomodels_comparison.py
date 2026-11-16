import argparse
import glob
import json
import os
import subprocess
import sys
sys.path.insert(0, "/home/huangp/pytor_bayesian_opti")
from utils.plot_tres import threetime_residual_agreement
from PlottingStyle.SNOplus_PythonPublicationStyle import SNOplus_style
import rat
import ROOT
from ROOT import RAT
import numpy as np

def oldextract_residuals(ratdsdir,particletype: str):
    """
    Calculates the time residuals from a given round of simulation .root files.
    """

    time_residuals = []
    counter        = 0 
    for ientry, _ in rat.dsreader(f"{ratdsdir}/*.root"):

        # setup time residual calculator and point3d classes to handle AV offset
        PMTCalStatus = RAT.DU.Utility.Get().GetPMTCalStatus()
        light_path = rat.utility().GetLightPathCalculator()
        #group_velocity = rat.utility().GetGroupVelocity()
        effective_velocity = rat.utility().GetEffectiveVelocity()
        pmt_info = rat.utility().GetPMTInfo()
        psup_system_id = RAT.DU.Point3D.GetSystemId("innerPMT")
        av_system_id = RAT.DU.Point3D.GetSystemId("av")

        if ientry.GetEVCount() == 0:
            continue

        reconEvent = ientry.GetEV(0)

        # check reconstruction is valid
        fit_name = reconEvent.GetDefaultFitName()
        if not reconEvent.FitResultExists(fit_name):
            continue

        vertex = reconEvent.GetFitResult(fit_name).GetVertex(0)
        if (not vertex.ContainsPosition() or
            not vertex.ContainsTime() or
            not vertex.ValidPosition() or
            not vertex.ValidTime() or
            not vertex.ContainsEnergy() or
            not vertex.ValidEnergy()):
            continue
        # print("Reconstruction checks PASSED!")
        # reconstruction valid so get reconstructed position and energy
        reconPosition  = vertex.GetPosition() # returns in PSUP coordinates
        reconEnergy    = vertex.GetEnergy()        
        reconEventTime = vertex.GetTime()
        
        # apply AV offset to position
        event_point = RAT.DU.Point3D(psup_system_id, reconPosition)
        event_point.SetCoordinateSystem(av_system_id)
        if event_point.Mag() > 4000:
            continue
        # convert back to PSUP coordinates
        event_point.SetCoordinateSystem(psup_system_id)

        # apply energy tagging cuts the same as that in data
        if particletype == "Bi210":
            if reconEnergy < 0.7 or reconEnergy > 1.1:
                continue
        elif particletype == "Bi214":
            if reconEnergy < 1.25 or reconEnergy > 3.00:
                continue
        elif particletype == "Po214":
            if reconEnergy < 0.7 or reconEnergy > 1.1:
                continue
        else:
            print(f"Wrong input Particle Type {particletype} in extract_residuals(). Should be either Bi210 or Bi214 or Po214")
            exit(1)

        # event has passed all the cuts so we can extract the time residuals
        calibratedPMTs = reconEvent.GetCalPMTs()
        pmtCalStatus = rat.utility().GetPMTCalStatus()
        for j in range(calibratedPMTs.GetCount()):
            pmt = calibratedPMTs.GetPMT(j)
            if pmtCalStatus.GetHitStatus(pmt) != 0:
                continue
            
            # residual_recon = timeResCalc.CalcTimeResidual(pmt, reconPosition, reconEventTime, True)
            pmt_point = RAT.DU.Point3D(psup_system_id, pmt_info.GetPosition(pmt.GetID()))
            light_path.CalcByPosition(event_point, pmt_point)
            inner_av_distance = light_path.GetDistInInnerAV()
            av_distance = light_path.GetDistInAV()
            water_distance = light_path.GetDistInWater()
            #transit_time = group_velocity.CalcByDistance(inner_av_distance, av_distance, water_distance)
            transit_time = effective_velocity.CalcByDistance(inner_av_distance, av_distance, water_distance)
            residual_recon = pmt.GetTime() - transit_time - reconEventTime
            
            time_residuals.append(residual_recon)
        
        counter += 1
        if counter % 100 == 0:
            print("COMPLETED {} / {}".format(counter, 5000))

    return time_residuals




'''
Compare the data time residuals of each (phase, particle) to the best-fit MC simulated at two run IDs.

  python3 utils/twomodels_comparison.py all                    # 4 plots: {ppo,bismsb} x {Bi214,Po214}
  python3 utils/twomodels_comparison.py plot bismsb Po214      # one plot
  python3 utils/twomodels_comparison.py extract SIMDIR Po214 OUT.npy   # MC tres of one run -> .npy

MC tres are cached as <bestntuple>/<run>_tres.npy; each run is extracted in its own process because the
RAT DB / Utility singletons (material, AV offset) are set up once per run.
'''

ENERGY_CUTS = {
    "Bi210": (0.7, 1.1),
    "Bi214": (1.25, 3.0),
    "Po214": (0.7, 1.1),
}

REPO = "/home/huangp/pytor_bayesian_opti"
BEST_DIR = "/data/snoplus2/weiiiii/pytor_bayesian_opti/rat-8.3"
N_BEST_FILES = 50
# per phase: the optics config holding the data tres paths, the MC run IDs, and the legend text
PHASES = {
    "ppo": {
        "paramsbound": f"{REPO}/paramsbound_ppo.json",
        "runs": (300000, 309477),
        "text": "2.2 g/L PPO\nR < 4.0 m",
    },
    "bismsb": {
        "paramsbound": f"{REPO}/paramsbound_bismsb.json",
        "runs": (354109, 363962),
        "text": "2.2 g/L PPO + 2.2 g/L bis-MSB\nR < 4.0 m",
    },
}


def extract_residuals(input_pattern, particletype, run=None):
    """
    Time residuals of the simulated events in input_pattern/*.ntuple.root, with the same selection and
    tres calculation as src/time_residuals.py: fitValid, evIndex == 0, energy cut, posr_av <= 4 m,
    effective velocity of the simulation's GEO[inner_av] material. run defaults to the ntuples' runID.
    """
    if particletype not in ENERGY_CUTS:
        raise ValueError(
            f"Wrong input Particle Type {particletype} in extract_residuals(). "
            "Should be either Bi210, Bi214, or Po214."
        )

    emin, emax = ENERGY_CUTS[particletype]
    time_residuals = []

    file_pattern = os.path.join(input_pattern, "*.ntuple.root")
    file_list = sorted(glob.glob(file_pattern))

    # skip zombie / unclosed / incomplete files (e.g. a rat job killed mid-write) instead of crashing
    good_files, bad_files = [], []
    for f in file_list:
        try:
            tfile = ROOT.TFile.Open(f)
        except OSError:
            tfile = None
        ok = bool(tfile and not tfile.IsZombie() and not tfile.TestBit(ROOT.TFile.kRecovered)
                  and tfile.GetKey("output") and tfile.GetKey("meta"))
        (good_files if ok else bad_files).append(f)
        if tfile:
            tfile.Close()
    if bad_files:
        print(f"Skipping {len(bad_files)} unreadable/zombie ntuple(s): {bad_files}")
    file_list = good_files

    if not file_list:
        print(f"No readable ROOT files found for pattern: {file_pattern}")
        return np.array([])

    chain = ROOT.TChain("output")
    for f in file_list:
        chain.Add(f)

    if chain.GetEntries() == 0:
        print("TChain has zero entries after adding files.")
        return np.array([])

    if run is None:
        chain.GetEntry(0)
        run = int(chain.runID)

    # A TChain does not apply the simulation macro's /rat/db/set commands (rat.dsreader does), so
    # GEO[inner_av].material would stay at the default "labppo_scintillator" and the time-residual
    # calculator would use that material's effective velocity. Load the macro's DB commands from the
    # file's meta, as RAT::DU::DSReader does, before Utility::BeginOfRun() caches the velocity.
    db = RAT.DB.Get()
    db.LoadDefaults()
    meta_file = ROOT.TFile.Open(file_list[0])
    meta = meta_file.Get("meta")
    if not meta:
        raise RuntimeError(f"No 'meta' object in {file_list[0]}; cannot recover the simulation DB settings.")
    RAT.DBCommandLoader.LoadCommands(meta)
    meta_file.Close()

    ds_run = RAT.DS.Run()
    ds_run.SetRunID(int(run))
    db.BeginOfRun(ds_run)
    ratroot = os.environ["RATROOT"]
    db.Load(f"{ratroot}/data/geo/snoplusnative.geo")
    db.Load(f"{ratroot}/data/pmt/PMTINFO_sept2018.ratdb")

    utility = rat.utility().Get()
    utility.BeginOfRun()
    db.EndOfBeginOfRun()

    material = db.GetLink("GEO", "inner_av").GetS("material")
    inner_av_velocity = rat.utility().GetEffectiveVelocity().GetInnerAVVelocity(2000.0)
    print(f"Run {run}: {len(file_list)} ntuples, GEO[inner_av].material = {material}, "
          f"inner-AV effective velocity (r=2 m) = {inner_av_velocity:.4f} mm/ns")

    psup_system_id = RAT.DU.Point3D.GetSystemId("innerPMT")
    av_system_id = RAT.DU.Point3D.GetSystemId("av")
    tres_calc = rat.utility().GetTimeResidualCalculator()

    n_selected = 0
    for iEntry in range(chain.GetEntries()):
        chain.GetEntry(iEntry)

        if int(chain.fitValid) != 1:
            continue
        if int(chain.evIndex) != 0:
            continue

        reco_energy = float(chain.energy)
        if reco_energy < emin or reco_energy > emax:
            continue

        event_position = ROOT.TVector3(chain.posx, chain.posy, chain.posz)
        event_point = RAT.DU.Point3D(psup_system_id, event_position)
        # fiducial cut matching the data (posr_av <= 4000 mm): radius in AV coordinates, then back to PSUP
        event_point.SetCoordinateSystem(av_system_id)
        if event_point.Mag() > 4000.0:
            continue
        event_point.SetCoordinateSystem(psup_system_id)
        t = chain.time

        hittimes = chain.hittimes
        hitids = chain.hitids
        for iPMT in range(len(hittimes)):
            time_residuals.append(tres_calc.CalcTimeResidual(hitids[iPMT], hittimes[iPMT], event_point, t))
        n_selected += 1

    print(f"Run {run}: {n_selected} / {chain.GetEntries()} events selected, {len(time_residuals)} hits")
    return np.asarray(time_residuals, dtype=np.float64)


def load_residuals(filepath):
    data = np.load(filepath, allow_pickle=True)
    if data.dtype == object:
        return np.concatenate([np.asarray(item, dtype=np.float64).ravel() for item in data])
    return np.asarray(data, dtype=np.float64).ravel()


def mc_residuals(phase, particletype, run, refresh=False):
    """Cached MC tres of one best-fit run; extracted in a subprocess so each run gets a fresh RAT DB."""
    simdir = f"{BEST_DIR}/{phase}/{particletype}/results/bestntuple/{run}"
    cache = f"{simdir}/../{run}_tres.npy"
    n_files = len(glob.glob(f"{simdir}/*.ntuple.root"))
    if n_files < N_BEST_FILES:
        print(f"WARNING: {phase}/{particletype} run {run} has only {n_files}/{N_BEST_FILES} ntuples")
    if refresh or not os.path.exists(cache):
        subprocess.run([sys.executable, os.path.abspath(__file__), "extract", simdir, particletype, cache,
                        "--run", str(run)], check=True)
    return load_residuals(cache)


def plot_phase(phase, particletype, output_dir, refresh=False, make_log=False):
    config = PHASES[phase]
    with open(config["paramsbound"]) as f:
        data_path = json.load(f)["DataTres"][particletype]
    data = load_residuals(data_path)
    run1, run2 = config["runs"]
    mc1 = mc_residuals(phase, particletype, run1, refresh)
    mc2 = mc_residuals(phase, particletype, run2, refresh)
    print(f"{phase} {particletype}: data {len(data)} hits ({data_path}), MC run {run1} {len(mc1)}, MC run {run2} {len(mc2)}")

    isotope = {"Bi214": r"$^{214}$Bi", "Po214": r"$^{214}$Po", "Bi210": r"$^{210}$Bi"}[particletype]
    threetime_residual_agreement(
        data, mc1, mc2,
        label0=f"{isotope} Data", label1=f"{isotope} MC (run {run1})", label2=f"{isotope} MC (run {run2})",
        output_dir=output_dir, output_tag=f"tres_{phase}_{particletype}_data_vs_mc_runs",
        make_log=make_log, text=config["text"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data vs best-fit MC (two run IDs) time-residual comparison.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_ext = sub.add_parser("extract", help="MC tres of one ntuple directory -> .npy")
    p_ext.add_argument("simdir")
    p_ext.add_argument("particle_type", choices=sorted(ENERGY_CUTS))
    p_ext.add_argument("output")
    p_ext.add_argument("--run", type=int, default=None)
    for name in ("plot", "all"):
        p = sub.add_parser(name)
        if name == "plot":
            p.add_argument("phase", choices=sorted(PHASES))
            p.add_argument("particle_type", choices=["Bi214", "Po214"])
        p.add_argument("--output-dir", default=f"{REPO}/results/plots/best_rat83")
        p.add_argument("--refresh", action="store_true", help="re-extract the MC tres instead of using the cache")
        p.add_argument("--log", action="store_true", help="also make the log-scale plot")
    args = parser.parse_args()

    if args.cmd == "extract":
        tres = extract_residuals(args.simdir, args.particle_type, args.run)
        np.save(args.output, tres)
        print(f"Saved {len(tres)} residuals to {args.output}")
    else:
        SNOplus_style()
        jobs = [(args.phase, args.particle_type)] if args.cmd == "plot" else \
               [(ph, pt) for ph in PHASES for pt in ("Bi214", "Po214")]
        for phase, particletype in jobs:
            plot_phase(phase, particletype, args.output_dir, args.refresh, args.log)
