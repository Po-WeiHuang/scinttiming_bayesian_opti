import numpy as np
import rat
from ROOT import RAT
import json
import os
import rat
from datetime import datetime
import sys
sys.path.insert(0, "/home/huangp/pytor_bayesian_opti")
from utils.plot_tres import time_residual_agreement

def extract_residuals(particletype: str):
    """
    Calculates the time residuals from a given round of simulation .root files.
    """

    time_residuals = []
    counter        = 0 
    for ientry, _ in rat.dsreader("/data/snoplus2/weiiiii/pytor_bayesian_opti/*.root"):

        # setup time residual calculator and point3d classes to handle AV offset
        PMTCalStatus = RAT.DU.Utility.Get().GetPMTCalStatus()
        light_path = rat.utility().GetLightPathCalculator()
        group_velocity = rat.utility().GetGroupVelocity()
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
        if particletype == "Bi214":
            if reconEnergy < 1.25 or reconEnergy > 3.00:
                continue
        elif particletype == "Po214":
            if reconEnergy < 0.7 or reconEnergy > 1.1:
                continue
        else:
            print(f"Wrong input Particle Type {particletype} in extract_residuals(). Should be either Bi214 or Po214")
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
            transit_time = group_velocity.CalcByDistance(inner_av_distance, av_distance, water_distance)
            residual_recon = pmt.GetTime() - transit_time - reconEventTime
            
            time_residuals.append(residual_recon)
        
        counter += 1
        if counter % 100 == 0:
            print("COMPLETED {} / {}".format(counter, 5000))

    return time_residuals


def cal_objective():
    with open("algo_log.txt", "a") as outlog:
        outlog.write("\n\n")
        outlog.write(f"\nExtracted simulated residuals.")

        with open("paramsbound.json", "r") as f:
            boundparams = json.load(f)
        time_residuals = extract_residuals(boundparams["Type"])

        print(f"\nExtracted simulated residuals.")

        # load up the data residuals to compare to
        #data_residuals = np.load("/data/snoplus2/weiiiii/BiPo214_tune_cleaning/detector_data/bismsb_batch4_bi_4000.0.npy", allow_pickle = True)
        data_residuals = np.load("/data/snoplus2/weiiiii/BiPo214_tune_cleaning/detector_data/bismsb_batch4_po_4000.0.npy", allow_pickle = True)
        #data_residuals = np.concatenate(data_residuals)

        # create the binned histograms and calculate the chi2
        binning           = np.arange(-5, 250, 1) # only focus on the PEAK region
        binned_sim, edges = np.histogram(time_residuals, bins = binning, density = False)
        binned_data, _    = np.histogram(data_residuals, bins = binning, density = False)
        # Condition: bin content < 10 in either histogram
        low_mask = (binned_data < 10) | (binned_sim < 10)

        # Only consider bins >= 150
        candidate_mask = low_mask & (np.arange(len(binned_data)) >= 150)

        # Find first such bin
        if np.any(candidate_mask):
            thebin = np.where(candidate_mask)[0][0]
        else:
            thebin = 150

        print(f"\nFirst bin >=150 with content <10 is: {thebin}")
        # calculate the chi2 for this iteration
        binned_data = binned_data[:thebin]
        binned_sim = binned_sim[:thebin]

        binned_data_sigma = np.sqrt(binned_data); binned_sim_sigma = np.sqrt(binned_sim)
        # normalise the MC to the counts in the data
        int_dat = np.sum(binned_data)
        print("Data normalisation is: ", int_dat)
        scale= int_dat/np.sum(binned_sim)
        print("Before normalisation: MC normalisation is: ", np.sum(binned_sim))
        binned_sim = (binned_sim * scale) ; binned_sim_sigma =  binned_sim_sigma*scale
        print("After normalisation: MC normalisation is: ", np.sum(binned_sim))

    
    # using the new objective which is the sum of the squared residuals only
    objective = np.sum((binned_data - binned_sim)**2/(binned_data_sigma**2 + binned_sim_sigma**2))
    
    return data_residuals,time_residuals, objective
if __name__ == "__main__":
    with open("algo_log.txt", "a") as outlog:
        outlog.write("\n\n")
        outlog.write("### THIS IS TIME_RESIDUALS SCRIPT ###\n")
        print("### THIS IS TIME_RESIDUALS SCRIPT ###\n")
        savepath  = "results/pars"

        with open("currentparams.json", "r") as f:
            params = json.load(f)
        iteration =  params["iter"] + 1
        data_residuals,sim_residuals,chi2 = cal_objective()
        outlog.write(f"In iteration {iteration}: chi2 {chi2}")
        print(f"In iteration {iteration}: chi2 {chi2}")
        #now = datetime.now().replace(microsecond=0)
        now = datetime.now().timestamp()
        params["datetime"] = now 
        params["iter"]     = iteration
        params["chi2"]     = chi2
        np.save(f"{savepath}/{iteration}.npy",np.array([params[par] for par in params]))

        with open("currentparams.json", "w") as f:
            json.dump(params, f, indent=2)
        theparams = np.array([params[i] for i in params])
        time_residual_agreement(data_residuals,sim_residuals,iteration, theparams[:-3])

        outlog.write(f"\nFINISHING TRES PLOTTING IN ITER {iteration}")