import rat
from ROOT import RAT
import numpy as np
import os
import sys
import  argparse
def extract_residuals(input_pattern, particletype):

    """
    Calculates the time residuals from a given round of simulation .root files.
    """

    time_residuals = []
    counter        = 0 
    for ientry, _ in rat.dsreader(f"{input_pattern}/*.root"):

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
        if particletype == "Bi210":
            if reconEnergy < 0.7 or reconEnergy > 1.1:
                continue
        elif particletype == "Bi214":
            if reconEnergy < 1.25 or reconEnergy > 3.00:
                continue
        else:
            print(f"Wrong input Particle Type {particletype} in extract_residuals()")
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

    return np.array(time_residuals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract time residuals from SNO+ ROOT files and save as Numpy.")
    
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Input path pattern (e.g., '/data/snoplus/*.root'). Wrap in quotes if using wildcards.")
    
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output path for the .npy file (e.g., ./output.npy).")
    
    parser.add_argument("-p", "--particle", type=str, choices=["Bi210", "Bi214"], required=True,
                        help="Particle type to filter for: Bi210 or Bi214")

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