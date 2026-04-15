// compile command:  g++ -g -std=c++1y antinurecon.cpp -o antinurecon.exe `root-config --cflags --libs` -I${RATROOT}/include/libpq -I${RATROOT}/include -I${RATROOT}/include/external -L${RATROOT}/lib -lRATEvent_Linux
// test sim: /home/huangp/AntiNu/antinurecon.exe "/data/snoplus3/griddata/Processing_7_0_8_Preliminary_Scintillator_Gold_300000_308097/ntuples/*300000*.root" 300000  /data/snoplus3/weiii/antinu/mycuts/Ntuple_data/300000.ntuple.root rat-7.0.15 1
// test data: ./antinurecon.exe "/data/snoplus3/griddata/Processing_7_0_8_Preliminary_Scintillator_Gold_300000_308097/ntuples/*306498*.root" 306498 306498.root rat-7.0.8 1

#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TProfile.h"
#include "TRandom.h"
#include "TTree.h"
#include "TChain.h"
#include "TMath.h"
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <glob.h>
#include <RAT/DU/DSReader.hh>                                               
#include <RAT/DU/Utility.hh>
#include <RAT/DS/Entry.hh>
#include <RAT/GeoUtils.hh>
std::vector<std::string> glob(const char *);
double GetZAVoffset(int);
void printdelayinfo(bool&fitValid, double&posX, double&posY, double&posZ,double delayR, double&posz_av, double&energy,ULong64_t&clock50, ULong64_t&DCapplied,ULong64_t&DCflagged, Int_t&triggerWord, bool is_data);
bool IsDelayEv(bool&fitValid, double &itr,double&posX, double&posY, double&posZ,  double delayR,double&posz_av, double&energy,ULong64_t&clock50, ULong64_t&DCapplied,ULong64_t&DCflagged, Int_t&triggerWord, bool is_data);
void Livetime(int run, int num_vetos, int num_spallationvetos,ULong64_t& loneFollowerTime, ULong64_t& pileupTime, std::string Proc_rat);
void livetimecalculator(int runID, double deadtime, std::string Proc_rat);
bool Muon_Pileup_Follower_Veto(ULong64_t&DCapplied,ULong64_t&DCflagged,ULong64_t& clock50, int&nhitsCleaned, bool& lone_follower_flag, bool& veto_flag, ULong64_t& lone_start_time, ULong64_t& veto_start_time, ULong64_t& loneFollowerTime, ULong64_t& pileupTime, int& num_vetos);
bool BiPoVeto(std::vector<int>& eventid,int gtid);
struct Event{
        int Ev;
        TVector3 pos;
        ULong64_t clock50;
        double dR;
        double dt;
        bool saveflag = true;
        double Rav;
        double Ecorr;
        int ncandidates;
        bool Multiplicity_Flag = false;
        double dt_tostart_prompt;
        double dt_tostart_delay;
};


double GetZAVoffset(int runID){
    RAT::DB *db = RAT::DB::Get();
    db->LoadDefaults();
    RAT::DS::Run run;
    run.SetRunID(runID);
    db->BeginOfRun(run);
    std::vector<double> AVPos = RAT::GeoUtil::UpdateAVOffsetVectorFromDB();
    double zOff = AVPos[2];
    std::cout << "AV Shift is: " << zOff << std::endl;
    return zOff;
}

void Livetime(int run, int num_vetos, int num_spallationvetos,ULong64_t& loneFollowerTime, ULong64_t& pileupTime, std::string Proc_rat){

    double deadtime =  (20 * pow(10, 9) * num_vetos) + (10 * pow(10, 3) * num_spallationvetos) + pileupTime + loneFollowerTime; // in ns
    std::cout<<"num_vetos: "<<num_vetos<<"num_spallationvetos: "<<num_spallationvetos<<" pileupTime: " << pileupTime<<" loneFollowerTime: "<<loneFollowerTime<<" deadtime: "<<deadtime<<std::endl;
    deadtime = deadtime*pow(10,-9);//sec
    livetimecalculator(run, deadtime, Proc_rat);
    
}
void livetimecalculator(int runID, double deadtime, std::string Proc_rat){ //input deadtime in sec
    double start_day, start_sec, start_nsec, end_day, end_sec, end_nsec;
    long long start_time, end_time, duration;
    RAT::DB *db = RAT::DB::Get();
    db->LoadDefaults();
    RAT::DS::Run run;
    run.SetRunID(runID);
    db->BeginOfRun(run);
    RAT::DBLinkPtr dblink = db->GetLink("RUN");
    start_day = dblink->GetD("start_day");
    start_sec = dblink->GetD("start_sec");
    start_nsec = dblink->GetD("start_nsc");
    end_day = dblink->GetD("stop_day");
    end_sec = dblink->GetD("stop_sec");
    end_nsec = dblink->GetD("stop_nsc");
    //std::cout<<start_day<<" "<<start_sec<<" "<<start_nsec<<std::endl;
    //start_time = start_day * 24 * 3600 * pow(10, 9) + start_sec * pow(10, 9) + start_nsec;
    //end_time = end_day * 24 * 3600 * pow(10, 9) + end_sec * pow(10, 9) + end_nsec;
    start_time = start_day * 24 * 3600  + start_sec ;
    end_time = end_day * 24 * 3600  + end_sec;
    duration = end_time - start_time;
    //std::cout << "Duration of run: " << duration * pow(10, -9) << " s" << std::endl;
    std::cout << "Duration of run: " << duration << " s" << std::endl;
    double adjusted_livetime = duration - deadtime; 

    //std::cout << "Adjusted Livetime: " << adjusted_livetime * pow(10, -9) << " s" << std::endl;
    std::cout << "Adjusted Livetime: " << adjusted_livetime  << " s" << std::endl;
    
    // calculate date
    //double year; double month; double day;
    //DateCalculator(start_day, year, month, day);
    //std::cout<<" "<<year<<" "<<month<<" "<<day<<std::endl;

    // create output txt file containing the livetime
    std::ofstream livetime;
    if(  Proc_rat == "rat-7.0.8")                          livetime.open("/data/snoplus3/weiii/antinu/mycuts/2p2goldlistlivetime/" + std::to_string(runID) + ".txt",std::ofstream::out);
    //else if(Proc_rat == "rat-7.0.15")                      livetime.open("/data/snoplus3/weiii/antinu/rat-7.0.15/Livetime/" + std::to_string(runID) + ".txt",std::ofstream::out);
    
    //livetime << std::to_string(int(year))<<" , "<<std::to_string(int(month))<<" , "<<std::to_string(int(day))<<" , "<<std::to_string(runID)<<" , "<<std::to_string(adjusted_livetime * pow(10, -9))<<std::endl;
    livetime << std::to_string(runID)<<" , "<<std::to_string(adjusted_livetime)<<std::endl;
    livetime.close();
}


bool Muon_Pileup_Follower_Veto(ULong64_t&DCapplied,ULong64_t&DCflagged,ULong64_t& clock50, int&nhitsCleaned, bool& lone_follower_flag, bool& veto_flag, ULong64_t& lone_start_time, ULong64_t& veto_start_time, ULong64_t& loneFollowerTime, ULong64_t& pileupTime, int& num_vetos){
    
    
    if ((DCflagged & 0x80) != 0x80 && nhitsCleaned > 3000){
            //std::cout << "nhitsCleaned: "<< nhitsCleaned <<std::endl;
            //std::cout<< "DCflaggedPass: "<<DCflaggedPass<<std::endl;
            
        
          // check if we are inside a lone muon follower veto window
          if (lone_follower_flag == true){
              // add the dT and switch off the muon follower veto
              ULong64_t deltaT = ((clock50-lone_start_time) & 0x7FFFFFFFFFF)*20.0;
              loneFollowerTime += deltaT;
              lone_follower_flag = false;
          }

          if (veto_flag == false){
              num_vetos++;
              veto_flag = true;
              veto_start_time = clock50;
              return false;
          }
          else{
              // we have pileup! Need to calculate the additional pileup time.
              
              ULong64_t deltaT = ((clock50-veto_start_time) & 0x7FFFFFFFFFF)*20.0;
              std::cout<<"we have pielup!!! "<<"increment pileuptime: "<<deltaT<<std::endl;
              pileupTime += deltaT;

              // reset the veto window
              veto_start_time = clock50;
              return false;
          }
      }

      // now handle the veto window for follower events from highE / muon veto
      if (veto_flag == true){
          ULong64_t deltaT = ((clock50-veto_start_time) & 0x7FFFFFFFFFF)*20.0;

          // check if event falls inside veto window
          if (deltaT < 2.0e10){
            return false;
          }
          else{
            // we are no longer inside the veto time window --> switch it off!
            veto_flag = false;
          }
      }

      // check if we have a lone muon follower (i.e. muon at end of previous run)
      // this can only trigger if there was a muon at the end of the previous run
      if ((DCflagged&0x4000)!=0x4000){
          if (lone_follower_flag == false){
            lone_follower_flag = true; 
            lone_start_time = clock50;
            return false;
          } else{
            // we are within a lone follower veto period!
            ULong64_t deltaT = ((clock50 - lone_start_time) & 0x7FFFFFFFFFF)*20.0;
            loneFollowerTime += deltaT;
            lone_start_time = clock50;
            return false;
          }
      }
      else{
        if(lone_follower_flag == true){
            lone_follower_flag = false;
        }
      }
      return true;
}


double EnergyCorrection(int RunID,const double E, RAT::DU::Point3D pos, const bool is_data, const RAT::DU::ReconCalibrator*e_cal) {
    // Data vs MC energy correction (Tony's)
    double Ecorr = 0.;
    if (E == 0) return 0;
    if (RunID< 350000) Ecorr = e_cal->CalibrateEnergyRTF(is_data, E, pos,"labppo_2p2_scintillator",3);
    else if (RunID >=350000) Ecorr = e_cal->CalibrateEnergyRTF(is_data, E, pos,"labppo_2p2_bismsb_2p2_scintillator",3);
    return Ecorr;
}




















bool IsDelayEv(bool&fitValid, double &itr ,double&posX, double&posY, double&posZ,double delayR, double&posz_av, double&energy, ULong64_t&clock50, ULong64_t&DCapplied,ULong64_t&DCflagged, Int_t&triggerWord, bool is_data){
    // 0 : not pass all the cuts; 1: is a delay candidates

    // orphan check
    if (is_data == 1){
        if (triggerWord == 0) return false;
    }
    if(!fitValid ) return false;
    if(itr<0.2)  return false; 
    // FV check    
    if( delayR > 4000.0 ) return false;
    //std::cout<< "R: "<<R <<std::endl;
    // apply DC mask
    
    //std::cout<< "pass the DC"<<std::endl;
    if (is_data == 1){
        if (((DCapplied & 0xD82100000162C6) & DCflagged ) != (DCapplied & 0xD82100000162C6)) return false;
    }
    // apply delayed energy cuts
    if(energy < 0.7 || energy > 1.1)  return false;
    
    return true;
}
void printdelayinfo(bool&fitValid, double&posX, double&posY, double&posZ,double delayR, double&posz_av, double&energy,ULong64_t&clock50, ULong64_t&DCapplied,ULong64_t&DCflagged, Int_t&triggerWord, bool is_data){
    std::cout <<"fitValid "<< fitValid<<std::endl;
    std::cout <<"delayR "<< delayR<<std::endl;
    std::cout <<" posz_av "<< posz_av<<std::endl;
    std::cout <<" energy "<< energy<<std::endl;
    std::cout <<" triggerWord "<<triggerWord <<std::endl;
    
}




void ReconBi210(std::string inputpath, int RUN_NUM,std::string output_root_address, std::string Proc_rat, bool is_data){
    TChain *Tc   = new TChain("output");
    // Open file to print results muon veto times and info to
    std::ofstream outTxtFile;
    if(is_data == 1){
        size_t last_slash = output_root_address.find_last_of("/");

        // Extract directory path
        std::string directory = output_root_address.substr(0, last_slash + 1); 
        outTxtFile.open(directory+std::to_string(RUN_NUM)+".txt", std::ofstream::out | std::ofstream::trunc);
        outTxtFile << "run_number GTID Nhit clockCount50 clockCount10 UTDays UTSecs UTNSecs Veto_length" << std::endl;
    }
    TChain *PoTc = new TChain("PoT"); TChain *BiTc = new TChain("BiT");
    std::string filepath;//input filepath
    if(is_data == 0 ){
        
        Tc->Add(inputpath.c_str());
    }
    else if(is_data == 1 && Proc_rat == "rat-8.0.1" && RUN_NUM<350000 ){
        Tc->Add(inputpath.c_str());
        BiTc->Add(("/data/snoplus3/weiii/antinu/mycuts/rat-8.0.1/2p2/BiPo214/Ntuple/data/*"+std::to_string(RUN_NUM)+"*").c_str());
        PoTc->Add(("/data/snoplus3/weiii/antinu/mycuts/rat-8.0.1/2p2/BiPo214/Ntuple/data/*"+std::to_string(RUN_NUM)+"*").c_str());
        if(Tc->GetEntries() == 0){
            std::cerr << "Error opening files or files corruption"  << std::endl;
            exit(-1);
        }

        if(BiTc->GetEntries() == 0 || PoTc->GetEntries() == 0){
            std::cerr << "Error opening BiPo tagging files or corruption" << std::endl;
            //exit(-1);
        }
    }
    else if(is_data == 1 && Proc_rat == "rat-8.0.1" && RUN_NUM>=350000 ){
        Tc->Add(inputpath.c_str());
        BiTc->Add(("/data/snoplus3/weiii/antinu/mycuts/rat-8.0.1/bismsb/BiPo214/Ntuple/data/*"+std::to_string(RUN_NUM)+"*").c_str());
        PoTc->Add(("/data/snoplus3/weiii/antinu/mycuts/rat-8.0.1/bismsb/BiPo214/Ntuple/data/*"+std::to_string(RUN_NUM)+"*").c_str());
        if(Tc->GetEntries() == 0){
            std::cerr << "Error opening files or files corruption"  << std::endl;
            exit(-1);
        }

        if(BiTc->GetEntries() == 0 || PoTc->GetEntries() == 0){
            std::cerr << "Error opening BiPo tagging files or corruption" << std::endl;
            //exit(-1);
        }
    }
    else{
        std::cerr << "Error Processing RAT Argument" << std::endl;
        exit(-1);
    }
    std::cout<< "Ready to process files in "<< filepath <<std::endl; 
    

// define BiPo variables to load in
    int Po_eventid; int Bi_eventid;
    BiTc->SetBranchAddress("Bi_eventid", &Bi_eventid);
    PoTc->SetBranchAddress("Po_eventid", &Po_eventid);
    std::vector<int> eventid;
    for(int lEntry = 0; lEntry< BiTc->GetEntries(); lEntry++){
        BiTc->GetEntry(lEntry); PoTc->GetEntry(lEntry);
        eventid.push_back(Bi_eventid);
        eventid.push_back(Po_eventid);
        //std::cout<<"Bi_eventid "<<Bi_eventid<<std::endl;
    }
    std::sort(eventid.begin(), eventid.end());
    
    

// define variables to save the event information
    double posr_av, posX, posY, posZ,posz_av, energy, delayedEcorr, promptEcorr;
    int nhits; int nhitsCleaned; Double_t correctedNhits;
    bool fitValid, Multiplicity_Flag;
    ULong64_t clock50, clock10, startclock50;
    ULong64_t DCapplied;
    ULong64_t DCflagged;
    Int_t triggerWord;
    Int_t owlnhits;
    double alphaNReactorIBD;
    double delayR;
    int ncandidates; // num of Bi per Po
    int gtid; int runID; Double_t itr;
    double berkeleyAlphaBeta; UInt_t berkeleyAlphaBetaNhit; bool berkeleyAlphaBetaRetrigger;
    // mcinfo
    Double_t parentKE1; TString *parentMeta1 = NULL; Int_t parentpdg1;
    Int_t mcIndex; Int_t evIndex; Double_t mcPosx; Double_t mcPosy; Double_t mcPosz;

    //initialize all parameters to trck livetime
    bool lone_follower_flag = false; bool veto_flag = false; 
    ULong64_t lone_start_time=0.; ULong64_t veto_start_time=0.; 
    ULong64_t loneFollowerTime=0.; ULong64_t pileupTime=0.;
    int num_vetos=0;int num_spallationvetos=0;


    Int_t UTDays, UTSecs, UTNSecs;
    int64_t delayedTime, promptTime;
    double highNhitDelay, owlNhitDelay,highNhitPrompt, owlNhitPrompt;
    int64_t highNhitTime = -99999999999;int64_t highNhitDelayTime = -99999999999;
    int64_t owlNhitTime = -99999999999; int64_t owlNhitDelayTime = -99999999999;

    
    TFile *f_out = new TFile( output_root_address.c_str(),"RECREATE");
    TTree *DelayT = new TTree("DelayT","Delay After Cuts TTree");
    

    DelayT->Branch("fitValid", &fitValid);
    DelayT->Branch("posX", &posX);
    DelayT->Branch("posY", &posY);
    DelayT->Branch("posZ", &posZ);
    DelayT->Branch("posz_av",&posz_av);
    DelayT->Branch("clockCount50", &clock50);
    DelayT->Branch("startclockCount50", &startclock50);
    DelayT->Branch("energy", &energy);
    DelayT->Branch("R", &delayR);
    DelayT->Branch("Delayposr_av",&posr_av);
    DelayT->Branch("nhits", &nhits);
    DelayT->Branch("nhitsCleaned", &nhitsCleaned);
    DelayT->Branch("correctedNhits", &correctedNhits);
    DelayT->Branch("ncandidates", &ncandidates);
    DelayT->Branch("Multiplicity_Flag", &Multiplicity_Flag);
    DelayT->Branch("eventid", &gtid);
    DelayT->Branch("triggerWord", &triggerWord);
    DelayT->Branch("owlnhits", &owlnhits);
    DelayT->Branch("runID", &runID);
    //DelayT->Branch("delayedEcorr",&delayedEcorr);
    DelayT->Branch("parentKE1",&parentKE1);
    DelayT->Branch("parentMeta1",&parentMeta1);
    DelayT->Branch("parentpdg1",&parentpdg1);
    DelayT->Branch("mcIndex",&mcIndex);
    DelayT->Branch("evIndex",&evIndex);
    DelayT->Branch("mcPosx",&mcPosx);
    DelayT->Branch("mcPosy",&mcPosy);
    DelayT->Branch("mcPosz",&mcPosz);
    DelayT->Branch("itr",&itr);
    DelayT->Branch("alphaNReactorIBD",&alphaNReactorIBD);
    DelayT->Branch("berkeleyAlphaBeta",&berkeleyAlphaBeta);
    DelayT->Branch("berkeleyAlphaBetaNhit",&berkeleyAlphaBetaNhit);
    DelayT->Branch("berkeleyAlphaBetaRetrigger",&berkeleyAlphaBetaRetrigger);

    // to check the reconstruction was a) run at all and b) converged to a good solution
    Tc->SetBranchAddress("fitValid", &fitValid);
    // get the reconstructed position to check event falls within FV and work out the dR between Bi and Po candidate
    Tc->SetBranchAddress("posx", &posX);
    Tc->SetBranchAddress("posy", &posY);
    Tc->SetBranchAddress("posz", &posZ);
    Tc->SetBranchAddress("posz_av",&posz_av);
    Tc->SetBranchAddress("posr_av",&posr_av);
    // We use the clockCount50 (50 MHz) clock to find the inter-event time
    Tc->SetBranchAddress("clockCount50", &clock50);
    Tc->SetBranchAddress("clockCount10", &clock10);
    Tc->SetBranchAddress("uTDays", &UTDays);
    Tc->SetBranchAddress("uTSecs", &UTSecs);
    Tc->SetBranchAddress("uTNSecs", &UTNSecs);

    // Classifier values
    Tc->SetBranchAddress("berkeleyAlphaBeta",&berkeleyAlphaBeta);
    Tc->SetBranchAddress("berkeleyAlphaBetaNhit",&berkeleyAlphaBetaNhit);
    Tc->SetBranchAddress("berkeleyAlphaBetaRetrigger",&berkeleyAlphaBetaRetrigger);
    Tc->SetBranchAddress("alphaNReactorIBD",&alphaNReactorIBD);


    // Use the reconstructed energy to tag the Po and then the Bi
    Tc->SetBranchAddress("energy", &energy);
    Tc->SetBranchAddress("nhits", &nhits);
    Tc->SetBranchAddress("owlnhits", &owlnhits);
    Tc->SetBranchAddress("nhitsCleaned", &nhitsCleaned);
    Tc->SetBranchAddress("correctedNhits", &correctedNhits);

    // DC mask variables
    Tc->SetBranchAddress("dcApplied", &DCapplied);
    Tc->SetBranchAddress("dcFlagged", &DCflagged);
    Tc->SetBranchAddress("eventID", &gtid);
    Tc->SetBranchAddress("runID", &runID);
    Tc->SetBranchAddress("itr",&itr);

    // trigger word
    Tc->SetBranchAddress("triggerWord", &triggerWord);

    // true info
    Tc->SetBranchAddress("parentKE1",&parentKE1);
    Tc->SetBranchAddress("parentMeta1",&parentMeta1);
    Tc->SetBranchAddress("parentpdg1",&parentpdg1);
    Tc->SetBranchAddress("mcIndex",&mcIndex);
    Tc->SetBranchAddress("evIndex",&evIndex);
    Tc->SetBranchAddress("mcPosx",&mcPosx);
    Tc->SetBranchAddress("mcPosy",&mcPosy);
    Tc->SetBranchAddress("mcPosz",&mcPosz);

    int Pocounts = 0;

    // induce energy correction
    // Load RATDB tables into ReconCalibrator (VERY IMPORTANT!)
    //RAT::DU::Utility::Get()->LoadDBAndBeginRun();
    // Get the ReconCalibrator object to do the position-dependent energy correction
    //const RAT::DU::ReconCalibrator& e_cal = RAT::DU::Utility::Get()->GetReconCalibrator();
    RAT::DU::ReconCalibrator* e_cal = RAT::DU::ReconCalibrator::Get();
    Event promptEv;  Event delayEv;
    const size_t av_id = RAT::DU::Point3D::GetSystemId("av");
    int OldCoin_MCIndex = 0;
    Tc->GetEntry(0);
    startclock50 = clock50;
    // now iterate through every event in the ntuple to look for the DELAYED (Po) events
    for (int iEntry = 0; iEntry < Tc->GetEntries(); iEntry++){
        
        
        ncandidates = 0; // refresh the Bi count
        Tc->GetEntry(iEntry);
        if (is_data == 1){
            if(triggerWord == 0) continue; // reject orphans
            if (((DCapplied & 0x18000000004080) & DCflagged ) != (DCapplied & 0x18000000004080)) continue;
        
        }
        // apply hinits Veto
        delayedTime = int64_t(clock50);
        if (nhits > 3000){
                highNhitDelayTime = delayedTime;
                std::cout << RUN_NUM << " " << gtid << " " << energy<< " "<< nhits <<" "<<owlnhits<<" "<<itr<< " " << clock50 << " " << clock10 << " " << UTDays << " " << UTSecs << " " << (double)(UTNSecs) << " " << 1.0 * 1E9 << std::endl;
                outTxtFile << RUN_NUM << " " << gtid << " " << nhits << " " << clock50 << " " << clock10 << " " << UTDays << " " << UTSecs << " " << (double)(UTNSecs) << " " << 1.0 * 1E9 << std::endl;
            }
            highNhitDelay = ((delayedTime - highNhitDelayTime) & 0x7FFFFFFFFFF) / 50E6;  // [s] dealing with clock rollover
        if (highNhitDelay < 1) continue; //1s nhits veto
        if (is_data == 1){
            
            //if( !Muon_Pileup_Follower_Veto(DCapplied,DCflagged,clock50, nhitsCleaned, lone_follower_flag, veto_flag, lone_start_time, veto_start_time, loneFollowerTime, pileupTime, num_vetos)){
              //  continue;
            //}
            if (owlnhits > 3) owlNhitTime = delayedTime;
            owlNhitDelay = ((delayedTime - owlNhitTime) & 0x7FFFFFFFFFF) / 50.0;  // [us] dealing with clock rollover
        
            if (owlNhitDelay < 10){
                num_spallationvetos++;
                continue;
            }
        }
        

        
        //BiPo veto
        
        if (is_data == 1){
            if ( std::find(eventid.begin(), eventid.end(), gtid) != eventid.end() ){
            continue;    
            } 
            
        }
        
        delayR = pow(( posX*posX + posY*posY + posz_av*posz_av),0.5 );
        //std::cout<<"delayR "<<delayR<<" "<<"posr_av "<<posr_av<<std::endl;
        TVector3 delayedPos = TVector3(posX, posY, posZ);
        RAT::DU::Point3D delayposition(av_id, posX, posY, posz_av);
        
        
        if (!IsDelayEv(fitValid,itr, posX, posY, posZ, delayR,posz_av, energy, clock50,DCapplied,DCflagged,triggerWord, is_data)){
            continue;
        }
        
        

        
        
        DelayT->Fill();
         
        
        

    }
    //std::cout<<"Entries:  "<<Tc->GetEntries()<<std::endl;

    std::cout<<"End of Coincidence tagging"<<std::endl;

    
   
    f_out->Write();  
    f_out->Close();
    outTxtFile.close();
    
}

int main(int argc, char** argv) {
    std::string inputfilepath = argv[1];
    int RUN_NUM = std::stoi(argv[2]);
    std::string output_root_address = argv[3];
    std::string ratversion = argv[4];  // Old legacy entry, left for backwards compatibility with wrapper code (unused here)
    bool is_data = std::stoi(argv[5]); 
    
    
    // Addresses of simulation output files to be analysed
    std::vector<std::string> input_files;
    
    ReconBi210(inputfilepath, RUN_NUM,output_root_address,ratversion,is_data);


    return 0;
}
