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
    std::cout << erfInv(-1) << std::endl;
}


