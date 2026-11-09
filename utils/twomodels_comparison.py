import sys
sys.path.insert(0, "/home/huangp/pytor_bayesian_opti")
from utils.plot_tres import threetime_residual_agreement
from PlottingStyle.SNOplus_PythonPublicationStyle import SNOplus_style
import rat
import ROOT
from ROOT import RAT
import numpy as np
import glob
import os
import sys

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
Run as: python3 plot_hit_time_residuals.py /path/to/file.ntuple.root
'''

ENERGY_CUTS = {
    "Bi210": (0.7, 1.1),
    "Bi214": (1.25, 3.0),
    "Po214": (0.7, 1.1),
}


def extract_residuals(input_pattern, run, particletype):
    """
    input_pattern: a directory like "/data/...."
    inside the function we search for:
        /data/..../*{run}*.ntuple.root
    """
    if particletype not in ENERGY_CUTS:
        raise ValueError(
            f"Wrong input Particle Type {particletype} in extract_residuals(). "
            "Should be either Bi210, Bi214, or Po214."
        )

    emin, emax = ENERGY_CUTS[particletype]
    time_residuals = []

    # Build file pattern from directory + run number
    #file_pattern = os.path.join(input_pattern, f"*.root")
    file_pattern = os.path.join(input_pattern, f"*.ntuple.root")
    file_list = sorted(glob.glob(file_pattern))

    if not file_list:
        print(f"No ROOT files found for pattern: {file_pattern}")
        return []

    chain = ROOT.TChain("output")
    for f in file_list:
        chain.Add(f)

    if chain.GetEntries() == 0:
        print("TChain has zero entries after adding files.")
        return []

    # Setup database and utilities
    db = RAT.DB().Get()
    db.BeginOfRun(run)
    db.LoadDefaults()
    db.Load("/home/huangp/fresh_rat/rat/data/geo/snoplusnative.geo")
    db.Load("/home/huangp/fresh_rat/rat/data/pmt/PMTINFO_sept2018.ratdb")

    utility = rat.utility().Get()
    utility.BeginOfRun()
    db.EndOfBeginOfRun()

    psup_system_id = RAT.DU.Point3D.GetSystemId("innerPMT")
    tres_calc = rat.utility().GetTimeResidualCalculator()
    chan_status = rat.utility().GetPMTCalStatus()

    for iEntry in range(chain.GetEntries()):
        chain.GetEntry(iEntry)

        if not hasattr(chain, "fitValid") or int(chain.fitValid) != 1:
            continue
        if not hasattr(chain, "evIndex") or int(chain.evIndex) != 0:
            continue

        if hasattr(chain, "energy"):
            reco_energy = float(chain.energy)
        elif hasattr(chain, "reconEnergy"):
            reco_energy = float(chain.reconEnergy)
        elif hasattr(chain, "fitEnergy"):
            reco_energy = float(chain.fitEnergy)
        else:
            raise AttributeError(
                "Could not find an energy branch in the chain. "
                "Tried: energy, reconEnergy, fitEnergy"
            )

        if reco_energy < emin or reco_energy > emax:
            continue

        hittimes = chain.hittimes
        hitids = chain.hitids

        event_position = ROOT.TVector3(chain.posx, chain.posy, chain.posz)
        event_point = RAT.DU.Point3D(psup_system_id, event_position)
        t = chain.time

        for iPMT in range(len(hittimes)):
            pmttime = hittimes[iPMT]
            pmtid = hitids[iPMT]

            #if chan_status.GetChannelStatus(pmtid) < 1.0:
            #    continue

            tres = tres_calc.CalcTimeResidual(pmtid, pmttime, event_point, t)
            time_residuals.append(tres)

    return time_residuals
    
if __name__ == "__main__":
    SNOplus_style()
   
    #data_residuals = np.load("/data/snoplus2/weiiiii/BiPo214_tune_cleaning/residuals_801/PPOPo214FV4000.npy", allow_pickle = True)
    data_residuals1 = np.load("/data/snoplus3/weiii/BiPo214/rat-8.2.0/ppo/data/Po214_300000_300024.npy", allow_pickle = True)
    #data_residuals2 = np.load("/data/snoplus3/weiii/BiPo214/rat-8.2.0/ppo/data/Po214_301409_308873.npy", allow_pickle = True)
    
    #mc = extract_residuals("/data/snoplus2/weiiiii/pytor_bayesian_opti/PPOBi214Trial1/FromRATDBVal/",300001,"Bi214")
    #mc = extract_residuals("/data/snoplus2/weiiiii/pytor_bayesian_opti/PPOPo214_nopporeem_rat820/FromBO",300001,"Po214")
    
    #bay_res= oldextract_residuals("/data/snoplus2/weiiiii/pytor_bayesian_opti/PPOPo214Trial1/FromRATDBVal/","Po214")
    #threetime_residual_agreement(data_residuals2, data_residuals1, mc,label0="Bi214 Data (301409-308873)", label1="Bi214 Data (300000-300024)", label2="Bi214 MC (300000-300024)")
    
    #threetime_residual_agreement(data_residuals1, data_residuals1, mc,label0="Po214 Data (rat-8.0.1)", label1="Po214 Data (rat-8.2.0,300000-300024)", label2="Po214 MC (rat-8.2.0,300000-300024)")
    

    
    data_residuals1 = np.load("/data/snoplus3/weiii/BiPo214/rat-8.2.0/bismsb/tresnpy/Bi214_354099_354139.npy", allow_pickle = True)
    data_residuals2 = np.load("/data/snoplus3/weiii/BiPo214/rat-8.2.0/bismsb/tresnpy/Bi214_354939_369792.npy", allow_pickle = True)
    #data_residuals0 = np.load("/data/snoplus3/weiii/BiPo214/fullFill/rat-8.0.1/bismsb/npy/Po214.npy", allow_pickle = True)
    mc = extract_residuals("/data/snoplus2/weiiiii/pytor_bayesian_opti/bismsbBi214_rat820/bestpar",354109,"Bi214")
    threetime_residual_agreement(data_residuals2, data_residuals1, mc,label0="Bi214 Data (354939-369792)", label1="Bi214 Data (354099-354139)", label2="Bi214 MC (354109)")
    #threetime_residual_agreement(data_residuals1, data_residuals0,mc,label0="Bi214 Data (rat-8.2.0)", label1="Bi214 Data (rat-8.0.1)",label2="Bi214 Mc (rat-8.2.0)")