#include <iostream>
#include <fstream>
#include "PROPOSAL/Bremsstrahlung.h"
#include "PROPOSAL/Integral.h"
#include "PROPOSAL/Medium.h"
#include "PROPOSAL/Interpolant.h"
#include "PROPOSAL/Ionization.h"
#include "PROPOSAL/Epairproduction.h"
#include "PROPOSAL/Propagator.h"
#include "PROPOSAL/ContinuousRandomization.h"
#include "PROPOSAL/Geometry.h"
#include "PROPOSAL/Output.h"
#include "PROPOSAL/StandardNormal.h"

//Stuff for LOG4CPLUS
#include <log4cplus/logger.h>
#include <log4cplus/loggingmacros.h>
#include <log4cplus/configurator.h>
#include <log4cplus/layout.h>
#include <iomanip>

#include <time.h>
#include <boost/math/special_functions/erf.hpp>
#define erfInv(x)   boost::math::erf_inv(x, boost::math::policies::policy<boost::math::policies::overflow_error<boost::math::policies::ignore_error> >())

using namespace log4cplus;
using namespace std;




int main(int argc, char** argv)
{


    // Output::getInstance().EnableASCIIOutput("ascii_test");
    Output::getInstance();
    int row = 0;

	std::ifstream config("config.txt");
	std::vector<double> energies;
	std::vector<int> energy_count;
	double energy;
	int count;
	while(!config.eof()) {
		std::string str;
	    std::getline(config, str);
	    std::stringstream ss(str);
		ss >> energy;
		ss >> count;
		energies.push_back(energy);
		energy_count.push_back(count);
	}
	config.close();

	std::vector<double> distances;
    distances.push_back(1000*100);

    int total = 0;
    int energy_index;
	for(int i=0; i<energy_count.size(); ++i) {
        total += energy_count[i];
        if(count < total) {
            energy_index = i;
            total -= energy_count[i];
            break;
        }
	}


    std::function<void(int,int,double)> save = [&] (int energy_i, int distance_i, double final_energy) -> void {
        std::cout << energy_i << " " << distance_i << " " << final_energy << std::endl;
        return;
    };

    Propagator* propa = new Propagator("resources/configuration");
    int pos = count-total;
    int id = total;
    for(int i=energy_index; i<energy_count.size(); ++i) {
        for(;pos<energy_count[i]; ++pos) {
            id += 1;
            Particle* part = new Particle(id,pos,"mu",0,0,0,0,0,0,0,0);
            part->SetEnergy(energies[i]);
            propa->SetParticle(part);
            propa->ChooseCurrentCollection(part);
            double result;
            for(unsigned int j=0; j<distances.size(); ++j) {
                result = propa->Propagate(distances[j]);
                if(result < 0) {
                    // failed to propagate the full distance
                    for(;j<distances.size(); ++j) {
                        save(i, j, 0.0);
                    }
                    break;
                }
                else {
                    // propagated the full distance
                    // result is the energy
                    save(i, j, result);
                }
            }
            delete part;
        }
        pos = 0;
    }
}


