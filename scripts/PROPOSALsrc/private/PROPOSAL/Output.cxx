#include "PROPOSAL/Output.h"

using namespace std;

vector<Particle*> Output::secondarys_;
vector<double> Output::secondarys_parent_E_;
bool Output::store_in_root_trees_ =   false;
bool Output::store_in_ASCII_file_ =   false;

void Output::SetLoggingConfigurationFile(std::string file)
{
    PropertyConfigurator::doConfigure(LOG4CPLUS_TEXT(file));
}


//----------------------------------------------------------------------------//
//----------------------------------------------------------------------------//


void Output::FillSecondaryVector(Particle *particle, int secondary_id, pair<double,string> energy_loss, double distance)
{
    callback_(particle, secondary_id, energy_loss, distance);

    string secondary_name;

    if(energy_loss.second.compare("Epairproduction")==0)
    {
        secondary_name  =    "epair";
    }
    else if(energy_loss.second.compare("Ionization")==0)
    {
        secondary_name  =    "delta";
    }
    else if(energy_loss.second.compare("Bremsstrahlung")==0)
    {
        secondary_name  =    "brems";
    }
    else if(energy_loss.second.compare("Photonuclear")==0)
    {
        secondary_name  =    "munu";
    }
    else  //decay
    {
        secondary_name  =   energy_loss.second;
    }

    Particle *particle_to_store   =   new Particle(particle->GetParticleId(), secondary_id, secondary_name, particle->GetX(), particle->GetY(), particle->GetZ(), particle->GetTheta(), particle->GetPhi(), energy_loss.first, particle->GetT(), distance); 
    secondarys_.push_back(particle_to_store);
    secondarys_parent_E_.push_back(particle_to_store->GetEnergy() + particle->GetEnergy());

    #if ROOT_SUPPORT
        if(store_in_root_trees_)
        {
            secondary_x_                    =   particle_to_store->GetX();
            secondary_y_                    =   particle_to_store->GetY();
            secondary_z_                    =   particle_to_store->GetZ();
            secondary_t_                    =   particle_to_store->GetT();
            secondary_theta_                =   particle_to_store->GetTheta();
            secondary_phi_                  =   particle_to_store->GetPhi();
            secondary_energy_               =   particle_to_store->GetEnergy();
            secondary_parent_particle_id_   =   particle_to_store->GetParentParticleId();
            secondary_particle_id_          =   particle_to_store->GetParticleId();
            secondary_name_                 =   particle_to_store->GetName();
            secondary_tree_->Fill();
        }
    #endif
        if(store_in_ASCII_file_)
        {
            secondary_ascii_ <<
            particle_to_store->GetX()<< "\t" <<
            particle_to_store->GetY()<< "\t" <<
            particle_to_store->GetZ()<< "\t" <<
            //particle_to_store->GetT()<< "\t" <<
            //particle_to_store->GetTheta()<< "\t" <<
            //particle_to_store->GetPhi()<< "\t" <<
            particle_to_store->GetEnergy()<< "\t" <<
            particle->GetEnergy() << "\t" <<
            particle_to_store->GetParentParticleId()<< "\t"
            //particle_to_store->GetParticleId()<< "\t" <<
            //particle_to_store->GetName()
            << endl;
        }

}


//----------------------------------------------------------------------------//
//----------------------------------------------------------------------------//

void Output::SetSecondaryCallback(std::function<void(Particle*,int,std::pair<double,std::string>,double)> callback) {
    callback_ = callback;
}

//----------------------------------------------------------------------------//
//----------------------------------------------------------------------------//


void Output::ClearSecondaryVector()
{
    for(unsigned int i = 0 ; i< secondarys_.size() ; i++)
    {
        delete secondarys_.at(i);
    }
    secondarys_.clear();
    secondarys_parent_E_.clear();
}


//----------------------------------------------------------------------------//
//----------------------------------------------------------------------------//


void Output::Close()
{
    #if ROOT_SUPPORT
        if(store_in_root_trees_)
        {
            secondary_tree_->Write();
            primary_tree_->Write();
            propagated_primary_tree_->Write();
            rootfile_->Close();
        }
    #endif
        if(store_in_ASCII_file_)
        {
            secondary_ascii_.close();
            primary_ascii_.close();
            propagated_primary_ascii_.close();
        }
}


//----------------------------------------------------------------------------//
//----------------------------------------------------------------------------//


#if ROOT_SUPPORT

    void Output::EnableROOTOutput(std::string rootfile_name)
    {
        rootfile_               =   new TFile(rootfile_name.c_str(),"RECREATE");

        secondary_tree_             =   new TTree("secondarys","secondarys");
        primary_tree_               =   new TTree("primarys","primarys");
        propagated_primary_tree_    =   new TTree("propagated_primarys","propagated_primarys");

        store_in_root_trees_     =   true;

        //Set the Branch Addresses;
        secondary_tree_->Branch("x",&secondary_x_,"x/D");
        secondary_tree_->Branch("y",&secondary_y_,"y/D");
        secondary_tree_->Branch("z",&secondary_z_,"z/D");
        secondary_tree_->Branch("t",&secondary_t_,"t/D");
        secondary_tree_->Branch("theta",&secondary_theta_,"theta/D");
        secondary_tree_->Branch("phi",&secondary_phi_,"phi/D");
        secondary_tree_->Branch("energy",&secondary_energy_,"energy/D");
        secondary_tree_->Branch("parent_particle_id",&secondary_parent_particle_id_,"parent_particle_id/I");
        secondary_tree_->Branch("particle_id",&secondary_particle_id_,"particle_id/I");
        secondary_tree_->Branch("name",&secondary_name_);

        primary_tree_->Branch("x",&primary_x_,"x/D");
        primary_tree_->Branch("y",&primary_y_,"y/D");
        primary_tree_->Branch("z",&primary_z_,"z/D");
        primary_tree_->Branch("t",&primary_t_,"t/D");
        primary_tree_->Branch("theta",&primary_theta_,"theta/D");
        primary_tree_->Branch("phi",&primary_phi_,"phi/D");
        primary_tree_->Branch("energy",&primary_energy_,"energy/D");
        primary_tree_->Branch("parent_particle_id",&primary_parent_particle_id_,"parent_particle_id/I");
        primary_tree_->Branch("particle_id",&primary_particle_id_,"particle_id/I");
        primary_tree_->Branch("name",&primary_name_);

        propagated_primary_tree_->Branch("x",&prop_primary_x_,"x/D");
        propagated_primary_tree_->Branch("y",&prop_primary_y_,"y/D");
        propagated_primary_tree_->Branch("z",&prop_primary_z_,"z/D");
        propagated_primary_tree_->Branch("t",&prop_primary_t_,"t/D");
        propagated_primary_tree_->Branch("theta",&prop_primary_theta_,"theta/D");
        propagated_primary_tree_->Branch("phi",&prop_primary_phi_,"phi/D");
        propagated_primary_tree_->Branch("energy",&prop_primary_energy_,"energy/D");
        propagated_primary_tree_->Branch("parent_particle_id",&prop_primary_parent_particle_id_,"parent_particle_id/I");
        propagated_primary_tree_->Branch("particle_id",&prop_primary_particle_id_,"particle_id/I");
        propagated_primary_tree_->Branch("name",&prop_primary_name_);
        propagated_primary_tree_->Branch("xi",&prop_primary_xi_,"xi/D");
        propagated_primary_tree_->Branch("yi",&prop_primary_yi_,"yi/D");
        propagated_primary_tree_->Branch("zi",&prop_primary_zi_,"zi/D");
        propagated_primary_tree_->Branch("ti",&prop_primary_ti_,"ti/D");
        propagated_primary_tree_->Branch("ei",&prop_primary_ei_,"ei/D");
        propagated_primary_tree_->Branch("xf",&prop_primary_xf_,"xf/D");
        propagated_primary_tree_->Branch("yf",&prop_primary_yf_,"yf/D");
        propagated_primary_tree_->Branch("zf",&prop_primary_zf_,"zf/D");
        propagated_primary_tree_->Branch("tf",&prop_primary_tf_,"tf/D");
        propagated_primary_tree_->Branch("ef",&prop_primary_ef_,"ef/D");
        propagated_primary_tree_->Branch("xc",&prop_primary_xc_,"xc/D");
        propagated_primary_tree_->Branch("yc",&prop_primary_yc_,"yc/D");
        propagated_primary_tree_->Branch("zc",&prop_primary_zc_,"zc/D");
        propagated_primary_tree_->Branch("tc",&prop_primary_tc_,"tc/D");
        propagated_primary_tree_->Branch("ec",&prop_primary_ec_,"ec/D");
        propagated_primary_tree_->Branch("propagated_distance",&prop_primary_propagated_distance_,"propagated_distance/D");
        propagated_primary_tree_->Branch("energy_lost",&prop_primary_elost_,"energy_lost/D");

    }


//----------------------------------------------------------------------------//
//----------------------------------------------------------------------------//


    void Output::DisableROOTOutput()
    {
        store_in_root_trees_ =   false;
    }

//----------------------------------------------------------------------------//
//----------------------------------------------------------------------------//


    void Output::StorePrimaryInTree(Particle *primary)
    {
        if(store_in_root_trees_)
        {
            primary_x_                    =   primary->GetX();
            primary_y_                    =   primary->GetY();
            primary_z_                    =   primary->GetZ();
            primary_t_                    =   primary->GetT();
            primary_theta_                =   primary->GetTheta();
            primary_phi_                  =   primary->GetPhi();
            primary_energy_               =   primary->GetEnergy();
            primary_parent_particle_id_   =   primary->GetParentParticleId();
            primary_particle_id_          =   primary->GetParticleId();
            primary_name_                 =   primary->GetName();

            primary_tree_->Fill();
        }
    }


    //----------------------------------------------------------------------------//
    //----------------------------------------------------------------------------//

    void Output::StorePropagatedPrimaryInTree(Particle *prop_primary)
    {
        if(store_in_root_trees_)
        {
            prop_primary_x_                    =   prop_primary->GetX();
            prop_primary_y_                    =   prop_primary->GetY();
            prop_primary_z_                    =   prop_primary->GetZ();
            prop_primary_t_                    =   prop_primary->GetT();
            prop_primary_theta_                =   prop_primary->GetTheta();
            prop_primary_phi_                  =   prop_primary->GetPhi();
            prop_primary_energy_               =   prop_primary->GetEnergy();
            prop_primary_parent_particle_id_   =   prop_primary->GetParentParticleId();
            prop_primary_particle_id_          =   prop_primary->GetParticleId();
            prop_primary_name_                 =   prop_primary->GetName();
            prop_primary_xi_                   =   prop_primary->GetXi();
            prop_primary_yi_                   =   prop_primary->GetYi();
            prop_primary_zi_                   =   prop_primary->GetZi();
            prop_primary_ti_                   =   prop_primary->GetTi();
            prop_primary_ei_                   =   prop_primary->GetEi();
            prop_primary_xf_                   =   prop_primary->GetXf();
            prop_primary_yf_                   =   prop_primary->GetYf();
            prop_primary_zf_                   =   prop_primary->GetZf();
            prop_primary_tf_                   =   prop_primary->GetTf();
            prop_primary_ef_                   =   prop_primary->GetEf();
            prop_primary_xc_                   =   prop_primary->GetXc();
            prop_primary_yc_                   =   prop_primary->GetYc();
            prop_primary_zc_                   =   prop_primary->GetZc();
            prop_primary_tc_                   =   prop_primary->GetTc();
            prop_primary_ec_                   =   prop_primary->GetEc();
            prop_primary_propagated_distance_  =   prop_primary->GetPropagatedDistance();
            prop_primary_elost_                =   prop_primary->GetElost();

            propagated_primary_tree_->Fill();
        }
    }

#endif

    //----------------------------------------------------------------------------//
    //----------------------------------------------------------------------------//
    //------------------------------ASCII OUTPUT----------------------------------//
    //----------------------------------------------------------------------------//
    //----------------------------------------------------------------------------//

    void Output::EnableASCIIOutput(string ASCII_Prefix, bool append)
     {
         if(store_in_ASCII_file_)
         {
             log_info("Storing in ascii file is already enabled!");
             return;
         }

         stringstream ss;

         ss.str("");
         ss << ASCII_Prefix << "_secondarys.txt";
         if(append)
         {
             secondary_ascii_.open(ss.str().c_str(),ios::app);
         }
         else
         {
             secondary_ascii_.open(ss.str().c_str(),ios::out);
         }

         ss.str("");
         ss << ASCII_Prefix << "_primarys.txt";
         if(append)
         {
             primary_ascii_.open(ss.str().c_str(),ios::app);
         }
         else
         {
             primary_ascii_.open(ss.str().c_str(),ios::out);
         }

         ss.str("");
         ss << ASCII_Prefix << "_propagated_primarys.txt";
         if(append)
         {
             propagated_primary_ascii_.open(ss.str().c_str(),ios::app);
         }
         else
         {
             propagated_primary_ascii_.open(ss.str().c_str(),ios::out);
         }


         if( !(propagated_primary_ascii_.is_open() && secondary_ascii_.is_open() && primary_ascii_.is_open()))
         {
             log_error("Could not open ASCII streams. ASCII output disabled.");
             store_in_ASCII_file_ = false;
         }

         store_in_ASCII_file_ = true;
     }


 //----------------------------------------------------------------------------//
 //----------------------------------------------------------------------------//


     void Output::DisableASCIIOutput()
     {
         store_in_ASCII_file_ =   false;
         secondary_ascii_.close();
         primary_ascii_.close();
         propagated_primary_ascii_.close();
     }

 //----------------------------------------------------------------------------//
 //----------------------------------------------------------------------------//


     void Output::StorePrimaryInASCII(Particle *primary)
     {
         if(store_in_ASCII_file_)
         {
             primary_ascii_ <<
             primary->GetX()<< "\t" <<
             primary->GetY()<< "\t" <<
             primary->GetZ()<< "\t" <<
             //primary->GetT()<< "\t" <<
             //primary->GetTheta()<< "\t" <<
             //primary->GetPhi()<< "\t" <<
             primary->GetEnergy()<< "\t" <<
             //primary->GetParentParticleId()<< "\t" <<
             primary->GetParticleId()<< "\t" << endl;
             //primary->GetName() << endl;
         }
     }


 //----------------------------------------------------------------------------//
 //----------------------------------------------------------------------------//

     void Output::StorePropagatedPrimaryInASCII(Particle *prop_primary)
     {
         if(store_in_ASCII_file_)
         {
             propagated_primary_ascii_ <<
             prop_primary->GetX()<< "\t" <<
             prop_primary->GetY()<< "\t" <<
             prop_primary->GetZ()<< "\t" <<
             
             prop_primary->GetElost()<< "\t" <<
             prop_primary->GetEnergy()<< "\t" <<
             prop_primary->GetParticleId()<< "\t" <<
             prop_primary->GetPropagatedDistance()<< "\t"
             
             //prop_primary->GetT()<< "\t" <<
             //prop_primary->GetTheta()<< "\t" <<
             //prop_primary->GetPhi()<< "\t" <<
             //prop_primary->GetEnergy()<< "\t" <<
             //prop_primary->GetParentParticleId()<< "\t" <<
             //prop_primary->GetParticleId()<< "\t" <<
             //prop_primary->GetName()<< "\t" <<
             //prop_primary->GetXi()<< "\t" <<
             //prop_primary->GetYi()<< "\t" <<
             //prop_primary->GetZi()<< "\t" <<
             //prop_primary->GetTi()<< "\t" <<
             //prop_primary->GetEi()<< "\t" <<
             //prop_primary->GetXf()<< "\t" <<
             //prop_primary->GetYf()<< "\t" <<
             //prop_primary->GetZf()<< "\t" <<
             //prop_primary->GetTf()<< "\t" <<
             //prop_primary->GetEf()<< "\t" <<
             //prop_primary->GetXc()<< "\t" <<
             //prop_primary->GetYc()<< "\t" <<
             //prop_primary->GetZc()<< "\t" <<
             //prop_primary->GetTc()<< "\t" <<
             //prop_primary->GetEc()<< "\t" <<
             //prop_primary->GetPropagatedDistance()<< "\t" <<
             //prop_primary->GetElost()<< endl;
             << endl;
         }
     }




























