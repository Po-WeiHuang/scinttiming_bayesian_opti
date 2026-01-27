import sys
sys.path.insert(0, "/home/huangp/pytor_bayesian_opti")
from utils.plot_tres import threetime_residual_agreement
import rat
from ROOT import RAT
import numpy as np

def extract_residuals(ratdsdir,particletype: str):
    """
    Calculates the time residuals from a given round of simulation .root files.
    """

    time_residuals = []
    counter        = 0 
    for ientry, _ in rat.dsreader(f"{ratdsdir}/*.root"):

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

    
if __name__ == "__main__":
    data_residuals = np.load("/data/snoplus2/weiiiii/BiPo214_tune_cleaning/detector_data/bismsb_batch4_bi_4000.0.npy", allow_pickle = True)
    bay_res = extract_residuals("/data/snoplus2/weiiiii/pytor_bayesian_opti/Bi214Trial2/bestparratds","Bi214")
    gridscan_res= extract_residuals("/data/snoplus2/weiiiii/pytor_bayesian_opti/Bi214Trial2/gridscanparratds","Bi214")
    #data_residuals = np.load("/data/snoplus2/weiiiii/BiPo214_tune_cleaning/detector_data/bismsb_batch4_po_4000.0.npy", allow_pickle = True)
    #bay_res = extract_residuals("/data/snoplus2/weiiiii/pytor_bayesian_opti/Po214Trial1/bestparratds","Po214")
    #gridscan_res= extract_residuals("/data/snoplus2/weiiiii/pytor_bayesian_opti/Po214Trial1/gridscanparratds","Po214")

    threetime_residual_agreement(data_residuals, bay_res,gridscan_res)