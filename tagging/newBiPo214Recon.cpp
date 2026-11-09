
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
#include <RAT/DU/TimeResidualCalculator.hh>
std::vector<std::string> glob(const char *);
double GetZAVoffset(int);
void Livetime(int run, int& num_vetos, ULong64_t& loneFollowerTime, ULong64_t& pileupTime, std::string Proc_rat);
void livetimecalculator(int runID, double deadtime, std::string Proc_rat);
void DateCalculator(double , double &, double &, double &);
bool Muon_Pileup_Follower_Veto(ULong64_t&DCapplied,ULong64_t&DCflagged,ULong64_t& clock50, int&nhitsCleaned, bool& lone_follower_flag, bool& veto_flag, ULong64_t& lone_start_time, ULong64_t& veto_start_time, ULong64_t& loneFollowerTime, ULong64_t& pileupTime, int& num_vetos);
bool RejectReTrigger(ULong64_t&vClockPrevious,ULong64_t&clock50);
bool IsDelayEv(int run, bool&fitValid, double&posX, double&posY, double&posZ,double delayR, double&posZAfterAVoffset, Double_t& correctedNhits,double&energy,double&delayedEcorr,ULong64_t&clock50, ULong64_t&DCapplied,ULong64_t&DCflagged, Int_t&triggerWord, bool is_data);
bool IsPromptEv(int run,bool&fitValid, double&posX, double&posY, double&posZ, double promptR, double&posZAfterAVoffset,Double_t& correctedNhits,double&energy,double&promptEcorr,double&dR, ULong64_t&DCapplied,ULong64_t&DCflagged, Int_t&triggerWord,  bool is_data);
double EnergyCorrection(int RunID,const double E, RAT::DU::Point3D pos, const bool is_data, RAT::DU::ReconCalibrator* e_cal);
// use glob to match those files passing the wildcard
std::vector<std::string> glob(const char *pattern) {
    glob_t g;
    glob(pattern, GLOB_TILDE, nullptr, &g); // one should ensure glob returns 0!
    std::vector<std::string> filelist;
    filelist.reserve(g.gl_pathc);// reserve enough space for putting each filename inside vector
    for (size_t i = 0; i < g.gl_pathc; ++i) {
        filelist.emplace_back(g.gl_pathv[i]);//append file in vector
    }
    return filelist;
}

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

void Livetime(int run, int& num_vetos, ULong64_t& loneFollowerTime, ULong64_t& pileupTime, std::string Proc_rat){

    double deadtime =  (20 * pow(10, 9) * num_vetos) + pileupTime + loneFollowerTime; // in ns
    std::cout<<"num_vetos: "<<num_vetos<<" pileupTime: " << pileupTime<<" loneFollowerTime: "<<loneFollowerTime<<" deadtime: "<<deadtime<<std::endl;
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
    if(adjusted_livetime < 0.) adjusted_livetime = duration;
    //std::cout << "Adjusted Livetime: " << adjusted_livetime * pow(10, -9) << " s" << std::endl;
    std::cout << "Adjusted Livetime: " << adjusted_livetime  << " s" << std::endl;
    
    // calculate date
    double year; double month; double day;
    DateCalculator(start_day, year, month, day);
    //std::cout<<" "<<year<<" "<<month<<" "<<day<<std::endl;

    // create output txt file containing the livetime
    std::ofstream livetime;
    livetime.open("/data/snoplus3/weiii/BiPo214/fullFill/"+Proc_rat+"/Livetime/Result/" + std::to_string(runID) + ".txt",std::ofstream::out);
    //if(Proc_rat == "rat-7.0.8")    livetime.open("/data/snoplus3/weiii/BiPo214/fullFill/rat-7.0.8/Livetime/Result/" + std::to_string(runID) + ".txt",std::ofstream::out);
    //else if(Proc_rat == "rat-7.0.15")   livetime.open("/data/snoplus3/weiii/BiPo214/fullFill/rat-7.0.15/Livetime/Result/" + std::to_string(runID) + ".txt",std::ofstream::out);
    //livetime << std::to_string(int(year))<<" , "<<std::to_string(int(month))<<" , "<<std::to_string(int(day))<<" , "<<std::to_string(runID)<<" , "<<std::to_string(adjusted_livetime * pow(10, -9))<<std::endl;
    livetime << std::to_string(int(year))<<" , "<<std::to_string(int(month))<<" , "<<std::to_string(int(day))<<" , "<<std::to_string(runID)<<" , "<<std::to_string(adjusted_livetime)<<std::endl;
    livetime.close();
}
void DateCalculator(double days, double &year, double &month, double &day){  // return year month day hour min sec since 0:00:00 Jan 1st, 2010
    int year_counts = 0; int month_counts = 0; //track_month from January(0) to Dec(11)
    while(days > 365+1){
        year_counts++;
        if((year_counts-2)%4==0){
           // std::cout<<"leap year"<<std::endl;
            days-=366;
        }
        else days-= 365;
    }
    double montharr_ofleapyear[12] = {31,29,31,30,31,30,31,31,30,31,30,31};
    double montharr_nominal[12]    = {31,28,31,30,31,30,31,31,30,31,30,31};
    year = 2010. + year_counts;
    //std::cout<<days<<std::endl;
    if(int(year)%4 == 0){
        while(days > montharr_ofleapyear[month_counts]+1){
            days-= montharr_ofleapyear[month_counts];
            month_counts++;
        }
    }
    else{
        while(days > montharr_nominal[month_counts]+1){
            days-= montharr_nominal[month_counts];
            month_counts++;
            //std::cout<<days<<std::endl;
        }
    }
    day = days;
    month = month_counts + 1;  

}

bool Muon_Pileup_Follower_Veto(ULong64_t&DCapplied,ULong64_t&DCflagged,ULong64_t& clock50, int&nhitsCleaned, bool& lone_follower_flag, bool& veto_flag, ULong64_t& lone_start_time, ULong64_t& veto_start_time, ULong64_t& loneFollowerTime, ULong64_t& pileupTime, int& num_vetos){
    if ((DCflagged & 0x80) != 0x80 && nhitsCleaned > 5000){
            //std::cout << "nhitsCleaned: "<< nhitsCleaned <<std::endl;
            
        
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
double EnergyCorrection(int RunID,const double E, RAT::DU::Point3D pos, const bool is_data, RAT::DU::ReconCalibrator* e_cal) {
    // Data vs MC energy correction (Tony's)
    double Ecorr = 0.;
    if (RunID< 350000) Ecorr = e_cal->CalibrateEnergyRTF(is_data, E, pos,"labppo_2p2_scintillator",3);
    else if (RunID >=350000) Ecorr = e_cal->CalibrateEnergyRTF(is_data, E, pos,"labppo_2p2_bismsb_2p2_scintillator",3);
    return Ecorr;
}


bool RejectReTrigger(ULong64_t&vClockPrevious,ULong64_t&clock50){
    
    Double_t timeDiff = ((clock50 - vClockPrevious )& 0x7FFFFFFFFFF)*20.0; //ns
    if( abs(timeDiff) < 505.0 ){
        //std::cout<<"retrigger time window "<< timeDiff <<std::endl;
        return false;
    }
    
    return true;
}

bool IsDelayEv(int run, bool&fitValid, double&posX, double&posY, double&posZ,double delayR, double&posZAfterAVoffset, Double_t& correctedNhits,double&energy,double&delayedEcorr,ULong64_t&clock50, ULong64_t&DCapplied,ULong64_t&DCflagged, Int_t&triggerWord, bool is_data){
    // 0 : not pass all the cuts; 1: is a delay candidates

    // orphan check
    if (is_data == 1){
        if (triggerWord == 0) return false;
    }
    if(!fitValid ) return false;
    
    // FV check    
    if( delayR > 6000.0 ) return false;
    //std::cout<< "R: "<<R <<std::endl;
    // apply DC mask
    if (is_data == 1){
        
        //for rat-8.0.1, run≥364311 -> include dtretriggercut and  tppotentialmuonfollowercut-short
        if (((DCapplied & 0xD82100000162C6) & DCflagged ) != (DCapplied & 0xD82100000162C6)) return false;
    }
    // for rat-8.0.1 after renergy recordination
    if(delayedEcorr < 0.7 || delayedEcorr > 1.1)  return false;
    
    return true;
}
bool IsPromptEv(int run,bool&fitValid, double&posX, double&posY, double&posZ, double promptR, double&posZAfterAVoffset,Double_t& correctedNhits,double&energy,double&promptEcorr,double&dR, ULong64_t&DCapplied,ULong64_t&DCflagged, Int_t&triggerWord,  bool is_data){

    // orphan check
    if (is_data == 1){
        if (triggerWord == 0) return false;
    }

    if(!fitValid ) return false; 
    // dR check
    if(dR > 1000) return false; 
    //std::cout<< "pass the dR cuts"<<std::endl;
    // pass DC mask
    //std::cout<<DCapplied<<endl;
    if (is_data == 1){
        if (((DCapplied & 0xD82100000162C6) & DCflagged ) != (DCapplied & 0xD82100000162C6)) return false;
    }
    if( promptR > 6000.0 ) return false;

    
    if(promptEcorr < 1.25 || promptEcorr > 3.5) return false;
        
    

    return true;
    
}

void BiPo214Recon(std::string inputpath, int RUN_NUM, std::string output_root_address, std::string Proc_rat, bool is_data){// detector_status: 0 is water phase, 1 is partial fill, 2 is  fulfill
    
    RAT::DB *db = RAT::DB::Get();
    db->LoadDefaults();
    db->SetServer("postgres://snoplus@pgsql.snopl.us:5400/ratdb");

    RAT::DS::Run ratRun;
    ratRun.SetRunID(RUN_NUM);
    db->BeginOfRun(ratRun);

    db->Load("/home/huangp/fresh_rat/rat/data/geo/snoplusnative.geo");
    db->Load("/home/huangp/fresh_rat/rat/data/pmt/PMTINFO_sept2018.ratdb");

    // The ntuple does not carry the processing macro's /rat/db/set overrides, so without this
    // GEO[inner_av].material stays at the default "labppo_scintillator" and the time-residual
    // calculator uses that material's effective velocity instead of the phase's.
    // Must be set before Utility::BeginOfRun(), which caches the effective velocity.
    const std::string inner_av_material = (RUN_NUM >= 350000)
        ? "labppo_2p2_bismsb_2p2_scintillator"   // 2.2 g/L PPO + bis-MSB phase (runs >= 350000)
        : "labppo_2p2_scintillator";             // 2.2 g/L PPO phase
    db->SetS("GEO", "inner_av", "material", inner_av_material);
    std::cout << "Run " << RUN_NUM << ": GEO[inner_av].material = " << inner_av_material << std::endl;

    RAT::DU::Utility::Get()->BeginOfRun();
    
    db->EndOfBeginOfRun();

    TChain *Tc = new TChain("output");
    std::cout<<"Start Processing file "<< inputpath <<std::endl;
    Tc->Add(inputpath.c_str());
    
    
    // check this run number exists or not
    if(Tc->GetEntries() == 0){
        std::cerr << "Error opening file" << std::endl;
        exit(-1);
    }
    std::cout<< "Ready to process files in "<< inputpath <<std::endl; 
    std::ofstream outTxtFile;
    if(is_data == 1){
        size_t last_slash = output_root_address.find_last_of("/");

        // Extract directory path
        std::string directory = output_root_address.substr(0, last_slash + 1); 
        outTxtFile.open(directory+std::to_string(RUN_NUM)+"vetooutput.txt", std::ofstream::out | std::ofstream::trunc);
        outTxtFile << "run_number GTID Nhit clockCount50 clockCount10 UTDays UTSecs UTNSecs Veto_length" << std::endl;
    }
    
    // define variables to save the event information
    double posr_av, posX, posY, posZ,posz_av, energy, delayedEcorr, promptEcorr;
    int nhits; int nhitsCleaned; Double_t correctedNhits;
    bool scintFit, fitValid, partialFit;
    double alphaNReactorIBD;
    ULong64_t clock50, clock10;
    ULong64_t DCapplied;
    ULong64_t DCflagged;
    Int_t triggerWord;
    Double_t itr;
    double promptR; double delayR;
    double dR; // record distance difference between Po/Bi
    double dt;// time difference between Po/Bi
    int ncandidates; // num of Bi per Po
    int gtid;
    Double_t mcPosx; Double_t mcPosy; Double_t mcPosz;Double_t PomcPosx; Double_t PomcPosy; Double_t PomcPosz;
    int evIndex; int runID;
    double berkeleyAlphaBeta; UInt_t berkeleyAlphaBetaNhit; bool berkeleyAlphaBetaRetrigger;

    //initialize all parameters to trck livetime
    bool lone_follower_flag = false; bool veto_flag = false; 
    ULong64_t lone_start_time=0.; ULong64_t veto_start_time=0.; 
    ULong64_t loneFollowerTime=0.; ULong64_t pileupTime=0.;
    int num_vetos=0; 
    Int_t UTDays, UTSecs, UTNSecs;
    int64_t delayedTime, promptTime;
    Double_t time;
    double highNhitDelay, highNhitPrompt;
    int64_t highNhitTime = -99999999999;int64_t highNhitDelayTime = -99999999999;
    std::vector<UShort_t>* hitPMTs = nullptr;
    std::vector<Float_t>* hitTimes = nullptr;
    std::vector<Float_t> Po_tres;
    std::vector<Float_t> Bi_tres;
    // mc truth
    Double_t mcke1; Double_t mcke2;
    Double_t parentKE1; Double_t parentKE2;
    Int_t parentpdg1; Int_t parentpdg2;
    TString *parentMeta1 = NULL;
    // Create a output file, TTree and branches
    std::string outputfilepath; //output filepath
    TFile *f_out = new TFile( output_root_address.c_str(),"RECREATE");
    
    TTree *PoT = new TTree("PoT","Po After Cuts TTree");
    TTree *BiT = new TTree("BiT","Bi After Cuts TTree");

    
    

    PoT->Branch("fitValid", &fitValid);
    PoT->Branch("scintFit", &scintFit);
    PoT->Branch("partialFit", &partialFit);
    PoT->Branch("posX", &posX);
    PoT->Branch("posY", &posY);
    PoT->Branch("posZ", &posZ);
    PoT->Branch("posz_av", &posz_av);
    PoT->Branch("clockCount50", &clock50);
    PoT->Branch("energy", &energy);
    PoT->Branch("dR", &dR);
    PoT->Branch("dt", &dt);
    PoT->Branch("Po_R", &delayR);
    PoT->Branch("posr_av",&posr_av);
    PoT->Branch("nhits", &nhits);
    PoT->Branch("correctedNhits", &correctedNhits);
    PoT->Branch("nhitsCleaned", &nhitsCleaned);
    PoT->Branch("ncandidates", &ncandidates);
    PoT->Branch("eventid", &gtid);
    PoT->Branch("triggerWord", &triggerWord);
    PoT->Branch("evIndex", &evIndex);
    PoT->Branch("delayedEcorr",&delayedEcorr);
    PoT->Branch("parentKE1", &parentKE1);
    PoT->Branch("parentKE1",&parentKE2);
    PoT->Branch("parentpdg1", &parentpdg1);
    PoT->Branch("parentpdg1",&parentpdg1);
    PoT->Branch("mcke1", &mcke1);
    PoT->Branch("mcke2",&mcke2);
    PoT->Branch("runID",&runID);
    PoT->Branch("parentMeta1",&parentMeta1);
    PoT->Branch("alphaNReactorIBD",&alphaNReactorIBD);
    PoT->Branch("berkeleyAlphaBeta",&berkeleyAlphaBeta);
    PoT->Branch("berkeleyAlphaBetaNhit",&berkeleyAlphaBetaNhit);
    PoT->Branch("berkeleyAlphaBetaRetrigger",&berkeleyAlphaBetaRetrigger);
    PoT->Branch("Po_mcPosx",&PomcPosx);
    PoT->Branch("Po_mcPosy",&PomcPosy);
    PoT->Branch("Po_mcPosz",&PomcPosz);
    PoT->Branch("Bi_mcPosx",&mcPosx);
    PoT->Branch("Bi_mcPosy",&mcPosy);
    PoT->Branch("Bi_mcPosz",&mcPosz);
    PoT->Branch("itr",&itr);
    PoT->Branch("hitPMTs",&hitPMTs);
    PoT->Branch("hitTimes",&hitTimes);
    PoT->Branch("tres",&Po_tres);

    BiT->Branch("fitValid", &fitValid);
    BiT->Branch("scintFit", &scintFit);
    BiT->Branch("partialFit", &partialFit);
    BiT->Branch("posX", &posX);
    BiT->Branch("posY", &posY);
    BiT->Branch("posZ", &posZ);
    BiT->Branch("posz_av", &posz_av);
    BiT->Branch("clockCount50", &clock50);
    BiT->Branch("energy", &energy);
    BiT->Branch("dR", &dR);
    BiT->Branch("dt", &dt);
    BiT->Branch("R", &promptR);
    BiT->Branch("posr_av",&posr_av);
    BiT->Branch("nhits", &nhits);
    BiT->Branch("nhitsCleaned", &nhitsCleaned);
    BiT->Branch("correctedNhits", &correctedNhits);
    BiT->Branch("eventid", &gtid);
    BiT->Branch("triggerWord", &triggerWord);
    BiT->Branch("evIndex", &evIndex);
    BiT->Branch("promptEcorr",&promptEcorr);
    BiT->Branch("parentKE1", &parentKE1);
    BiT->Branch("parentKE1",&parentKE2);
    BiT->Branch("parentpdg1", &parentpdg1);
    BiT->Branch("parentpdg1",&parentpdg1);
    BiT->Branch("mcke1", &mcke1);
    BiT->Branch("mcke2",&mcke2);
    BiT->Branch("runID",&runID);
    BiT->Branch("itr",&itr);
    BiT->Branch("Bi_mcPosx",&mcPosx);
    BiT->Branch("Bi_mcPosy",&mcPosy);
    BiT->Branch("Bi_mcPosz",&mcPosz);
    BiT->Branch("Po_mcPosx",&PomcPosx);
    BiT->Branch("Po_mcPosy",&PomcPosy);
    BiT->Branch("Po_mcPosz",&PomcPosz);
    BiT->Branch("parentMeta1",&parentMeta1);
    BiT->Branch("alphaNReactorIBD",&alphaNReactorIBD);
    BiT->Branch("berkeleyAlphaBeta",&berkeleyAlphaBeta);
    BiT->Branch("berkeleyAlphaBetaNhit",&berkeleyAlphaBetaNhit);
    BiT->Branch("berkeleyAlphaBetaRetrigger",&berkeleyAlphaBetaRetrigger);
    BiT->Branch("hitPMTs",&hitPMTs);
    BiT->Branch("hitTimes",&hitTimes);
    BiT->Branch("tres",&Bi_tres);




    // to check the reconstruction was a) run at all and b) converged to a good solution
    Tc->SetBranchAddress("scintFit", &scintFit);
    Tc->SetBranchAddress("fitValid", &fitValid);
    Tc->SetBranchAddress("partialFit", &partialFit);
    // get the reconstructed position to check event falls within FV and work out the dR between Bi and Po candidate
    Tc->SetBranchAddress("posx", &posX);
    Tc->SetBranchAddress("posy", &posY);
    Tc->SetBranchAddress("posz", &posZ);
    Tc->SetBranchAddress("posr_av",&posr_av);
    Tc->SetBranchAddress("posz_av",&posz_av);
    // We use the clockCount50 (50 MHz) clock to find the inter-event time
    Tc->SetBranchAddress("clockCount50", &clock50);

    // Use the reconstructed energy to tag the Po and then the Bi
    Tc->SetBranchAddress("energy", &energy);
    Tc->SetBranchAddress("nhits", &nhits);
    Tc->SetBranchAddress("nhitsCleaned", &nhitsCleaned);
    Tc->SetBranchAddress("correctedNhits", &correctedNhits);
    Tc->SetBranchAddress("berkeleyAlphaBeta",&berkeleyAlphaBeta);
    Tc->SetBranchAddress("berkeleyAlphaBetaNhit",&berkeleyAlphaBetaNhit);
    Tc->SetBranchAddress("berkeleyAlphaBetaRetrigger",&berkeleyAlphaBetaRetrigger);
    Tc->SetBranchAddress("alphaNReactorIBD",&alphaNReactorIBD);

    // DC mask variables & mc truth
    Tc->SetBranchAddress("dcApplied", &DCapplied);
    Tc->SetBranchAddress("dcFlagged", &DCflagged);
    Tc->SetBranchAddress("eventID", &gtid);
    Tc->SetBranchAddress("runID", &runID);
    Tc->SetBranchAddress("evIndex", &evIndex);
    Tc->SetBranchAddress("mcke1", &mcke1);
    Tc->SetBranchAddress("mcke2", &mcke2);
    Tc->SetBranchAddress("parentKE1", &parentKE1);
    Tc->SetBranchAddress("parentKE2", &parentKE2);
    Tc->SetBranchAddress("parentpdg1", &parentpdg1);
    Tc->SetBranchAddress("parentpdg2", &parentpdg2);
    Tc->SetBranchAddress("parentMeta1", &parentMeta1);
    // trigger word
    Tc->SetBranchAddress("triggerWord", &triggerWord);
    Tc->SetBranchAddress("mcPosx",&mcPosx);
    Tc->SetBranchAddress("mcPosy",&mcPosy);
    Tc->SetBranchAddress("mcPosz",&mcPosz);
    Tc->SetBranchAddress("hitids",&hitPMTs);
    Tc->SetBranchAddress("hittimes",&hitTimes);
    Tc->SetBranchAddress("time",&time);

    int Pocounts = 0;
    int orphancounts = 0;
    //double ZAVoffset;
    // get AV offset in zaxis to do correct R=x^2+y^2+z^2 calculation
    //ZAVoffset = GetZAVoffset(RUN_NUM);
    // induce energy correction
    RAT::DU::ReconCalibrator* e_cal = RAT::DU::ReconCalibrator::Get();
    const size_t av_id = RAT::DU::Point3D::GetSystemId("av");
    RAT::DU::TimeResidualCalculator tres_calc = RAT::DU::Utility::Get()->GetTimeResidualCalculator(); 
    ULong64_t vClockPrevious;
    
    // now iterate through every event in the ntuple to look for the DELAYED (Po) events
    for (int iEntry = 0; iEntry < Tc->GetEntries(); iEntry++){
        Tc->GetEntry(iEntry);
        ULong64_t Delay_clock50 = clock50;
        
        if(iEntry > 0){
            Tc->GetEntry(iEntry-1);
            vClockPrevious = clock50;
            if(!RejectReTrigger(vClockPrevious,Delay_clock50)){
                //std::cout<<"a retrigger event is found in delay!!!!!!"<<std::endl;
                continue;
            }
        }
        
        Tc->GetEntry(iEntry);
        ncandidates = 0; // refresh the Bi count
        
        // apply muon Veto
        
        if (is_data == 1){
            if(triggerWord == 0) continue; // reject orphans
            /*
            if (((DCapplied & 0x18000000004080) & DCflagged ) != (DCapplied & 0x18000000004080)) continue;
            //if( !Muon_Pileup_Follower_Veto(DCapplied,DCflagged,clock50, nhitsCleaned, lone_follower_flag, veto_flag, lone_start_time, veto_start_time, loneFollowerTime, pileupTime, num_vetos)) continue;
            
            if (nhits > 5000){
                highNhitDelayTime = delayedTime;
                std::cout << RUN_NUM << " " << gtid << " " << energy<< " "<< nhits <<" "<<itr<< " " << clock50 << " " << clock10 << " " << UTDays << " " << UTSecs << " " << (double)(UTNSecs) << " " << 1.0 * 1E9 << std::endl;
                outTxtFile << RUN_NUM << " " << gtid << " " << nhits << " " << clock50 << " " << clock10 << " " << UTDays << " " << UTSecs << " " << (double)(UTNSecs) << " " << 1.0 * 1E9 << std::endl;
            }
            highNhitDelay = ((delayedTime - highNhitDelayTime) & 0x7FFFFFFFFFF) / 50E6;  // [s] dealing with clock rollover
            if (highNhitDelay < 20) continue; //20s nhits veto
            */
        }
        
        delayR = pow(( posX*posX + posY*posY + posz_av*posz_av),0.5 );
        TVector3 delayedPos = TVector3(posX, posY, posZ);
        RAT::DU::Point3D delayposition(av_id, posX, posY, posz_av);
        delayedEcorr = EnergyCorrection(RUN_NUM,energy, delayposition, is_data, e_cal);
        //std::cout<<"IsDelayEv "<<IsDelayEv(fitValid, posX, posY, posZ, posZAfterAVoffset, energy, clock50,DCapplied,DCflagged,triggerWord)<<std::endl;
        if (!IsDelayEv(RUN_NUM,fitValid, posX, posY, posZ, delayR,posz_av, correctedNhits,energy,delayedEcorr, clock50,DCapplied,DCflagged,triggerWord, is_data)){
            continue;
        }
        
        
        
        
        
        
        // record x y z of Po
        double delayX = posX; double delayY =  posY ; double delayZ = posz_av;
        PomcPosx = mcPosz;PomcPosy = mcPosy; PomcPosz = mcPosz;
        
        // if successful, loop backwards in time to find the Bi candidate
        for (int jEntry = iEntry - 1; jEntry >= 0; jEntry --){
            Tc->GetEntry(jEntry);
            ULong64_t Prompt_clock50 = clock50;
            // used for tagging retriggers
            /*
            if(jEntry>0){
                Tc->GetEntry(jEntry-1);
                vClockPrevious = clock50;
                if(!RejectReTrigger(vClockPrevious,Prompt_clock50)){
                    //std::cout<<"a retrigger event is found in prompt!!!!!!"<<std::endl;
                    continue;
                }
            }
            
             
            Tc->GetEntry(jEntry);
            if (is_data == 1){
                highNhitPrompt = ((promptTime - highNhitDelayTime) & 0x7FFFFFFFFFF) / 50E6;  // [s] dealing with clock rollover
                if (highNhitPrompt < 20) break;
            }
            */
            dt = ((Delay_clock50 - Prompt_clock50) & 0x7FFFFFFFFFF)* 20.0/pow(10,3);//us
            //std::cout<<"dt: "<<dt<<std::endl;
            if( dt > 1000 || dt < 0.4 ) break;
            
    
            // apply dR in mm
            dR = pow( (posX- delayX)*(posX- delayX)+ (posY- delayY)*(posY- delayY)+ (posz_av- delayZ)*(posz_av- delayZ) ,0.5);
            //std::cout<< "dR: "<<dR<<std::endl;

            promptR = posr_av;
            TVector3 promptPos = TVector3(posX, posY, posZ);
            RAT::DU::Point3D promptposition(av_id, posX, posY, posz_av);
            promptEcorr = EnergyCorrection(RUN_NUM,energy, promptposition, is_data, e_cal);

            if (!IsPromptEv(RUN_NUM,fitValid, posX, posY, posZ, promptR,posz_av,correctedNhits,energy,promptEcorr,dR, DCapplied,DCflagged, triggerWord, is_data)){
                continue;
            }

            ncandidates++;
            std::cout<<ncandidates<<std::endl;

            // clear per-candidate vectors so we don't accumulate across candidates
            Bi_tres.clear();
            Po_tres.clear();

            if (!hitPMTs || !hitTimes) {
                continue;
            }
            
            std::cout<<"Prompt hitPMTs size: "<<hitPMTs->size()<<std::endl;
            if (hitPMTs->size() != hitTimes->size()){
                std::cerr << "Warning: hitPMTs and hitTimes size mismatch: " << hitPMTs->size() << " vs " << hitTimes->size() << std::endl;
            }

            for (size_t iPMT = 0; iPMT < hitPMTs->size(); iPMT++){
                double pmttime = hitTimes->at(iPMT);
                const UInt_t pmtid = hitPMTs->at(iPMT);
                Float_t tres = tres_calc.CalcTimeResidual(pmtid, pmttime, promptposition, time);
                Bi_tres.push_back(tres);
            }
            Tc->GetEntry(iEntry);
            
            std::cout<<"Po hitPMTs size: "<<hitPMTs->size()<<std::endl;
            if (hitPMTs->size() != hitTimes->size()){
                std::cerr << "Warning: hitPMTs and hitTimes size mismatch before Po loop: " << hitPMTs->size() << " vs " << hitTimes->size() << std::endl;
            }

            for (size_t iPMT = 0; iPMT < hitPMTs->size(); iPMT++){
                double pmttime = hitTimes->at(iPMT);
                unsigned int pmtid = hitPMTs->at(iPMT);
                Float_t tres = tres_calc.CalcTimeResidual(pmtid, pmttime, delayposition, time);
                Po_tres.push_back(tres);
                
            }
            Tc->GetEntry(jEntry);
            BiT->Fill();
            Tc->GetEntry(iEntry);
            PoT->Fill();

            // Clear vectors after filling the trees so they don't accumulate.
            Bi_tres.clear();
            Po_tres.clear();
            
            
        } 
        
        

    }
    f_out->Write();  
    f_out->Close();  
    std::cout<<"End of Coincidence Tagging"<<std::endl;
    // calculate lifetime and save results into file

    
}

int main(int argc, char** argv) {
    std::string inputfilepath = argv[1];
    int RUN_NUM = std::stoi(argv[2]);
    std::string output_root_address = argv[3];
    std::string ratversion = argv[4];  // Old legacy entry, left for backwards compatibility with wrapper code (unused here)
    //int runid = std::stoi(argv[3]);
    bool is_data = std::stoi(argv[5]); 
    
    
    // Addresses of simulation output files to be analysed
    std::vector<std::string> input_files;
    BiPo214Recon(inputfilepath, RUN_NUM,output_root_address,ratversion,is_data);


    return 0;
}
