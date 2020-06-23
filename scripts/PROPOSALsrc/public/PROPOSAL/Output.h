#ifndef OUTPUT_H
#define OUTPUT_H
#ifndef ICECUBE

//Stuff for LOG4CPLUS
#include <log4cplus/logger.h>
#include <log4cplus/loggingmacros.h>
#include <log4cplus/configurator.h>
#include <iomanip>
#include <string>
#include <sstream>
#include <boost/preprocessor/control/if.hpp>
#include <boost/preprocessor/comparison/less.hpp>
#include <boost/preprocessor/array/elem.hpp>
#include <boost/preprocessor/comparison/equal.hpp>
#include <boost/preprocessor/config/limits.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <functional>
#include "PROPOSAL/Particle.h"
#include "utility"

#define VA_COUNT(...) VA_COUNT0(__VA_ARGS__, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)
#define VA_COUNT0(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, N,...) N
#define VA_ARG(_i, ...) BOOST_PP_ARRAY_ELEM(_i, (VA_COUNT(__VA_ARGS__), (__VA_ARGS__)))

#define log_error(...) BOOST_PP_IF( BOOST_PP_EQUAL( 1, VA_COUNT(__VA_ARGS__) ),\
    LOG4CPLUS_ERROR(Output::getInstance().logger, VA_ARG(0, __VA_ARGS__) ) , \
    LOG4CPLUS_ERROR_FMT(Output::getInstance().logger, __VA_ARGS__ ) )

#define log_fatal(...) BOOST_PP_IF( BOOST_PP_EQUAL( 1, VA_COUNT(__VA_ARGS__) ),\
    LOG4CPLUS_FATAL(Output::getInstance().logger, VA_ARG(0, __VA_ARGS__) ) ; exit(1) , \
    LOG4CPLUS_FATAL_FMT(Output::getInstance().logger, __VA_ARGS__ ) ) ; exit(1)

#define log_warn(...) BOOST_PP_IF( BOOST_PP_EQUAL( 1, VA_COUNT(__VA_ARGS__) ),\
    LOG4CPLUS_WARN(Output::getInstance().logger, VA_ARG(0, __VA_ARGS__) ) , \
    LOG4CPLUS_WARN_FMT(Output::getInstance().logger, __VA_ARGS__ ) )

#define log_info(...) BOOST_PP_IF( BOOST_PP_EQUAL( 1, VA_COUNT(__VA_ARGS__) ),\
    LOG4CPLUS_INFO(Output::getInstance().logger, VA_ARG(0, __VA_ARGS__) ) , \
    LOG4CPLUS_INFO_FMT(Output::getInstance().logger, __VA_ARGS__ ) )

#define log_trace(...) BOOST_PP_IF( BOOST_PP_EQUAL( 1, VA_COUNT(__VA_ARGS__) ),\
    LOG4CPLUS_TRACE(Output::getInstance().logger, VA_ARG(0, __VA_ARGS__) ) , \
    LOG4CPLUS_TRACE_FMT(Output::getInstance().logger, __VA_ARGS__ ) )

#define log_debug(...) BOOST_PP_IF( BOOST_PP_EQUAL( 1, VA_COUNT(__VA_ARGS__) ),\
    LOG4CPLUS_DEBUG(Output::getInstance().logger, VA_ARG(0, __VA_ARGS__) ) , \
    LOG4CPLUS_DEBUG_FMT(Output::getInstance().logger, __VA_ARGS__ ) )

#define log_notice(...) BOOST_PP_IF( BOOST_PP_EQUAL( 1, VA_COUNT(__VA_ARGS__) ),\
    LOG4CPLUS_NOTICE(Output::getInstance().logger, VA_ARG(0, __VA_ARGS__) ) , \
    LOG4CPLUS_NOTICE_FMT(Output::getInstance().logger, __VA_ARGS__ ) )


#if ROOT_SUPPORT
    #include "TTree.h"
    #include "TFile.h"
#endif

using namespace log4cplus;

class Output
{
private:

    Output()
    {
        PropertyConfigurator::doConfigure(LOG4CPLUS_TEXT("resources/log4cplus.conf"));
        logger = Logger::getInstance(LOG4CPLUS_TEXT("PORPOSAL"));

    }

    Output(Output const&);         // Don't Implement.
    void operator=(Output const&); // Don't implement

    static std::vector<Particle*> secondarys_;
    static std::vector<double> secondarys_parent_E_;

    static bool store_in_root_trees_;

    //ASCII Output
    std::ofstream secondary_ascii_;
    std::ofstream primary_ascii_;
    std::ofstream propagated_primary_ascii_;

    #if ROOT_SUPPORT

        TTree *secondary_tree_;
        TTree *primary_tree_;
        TTree *propagated_primary_tree_;
        TFile *rootfile_;

        double secondary_x_;
        double secondary_y_;
        double secondary_z_;
        double secondary_t_;
        double secondary_theta_;
        double secondary_phi_;
        double secondary_energy_;
        int secondary_parent_particle_id_;
        int secondary_particle_id_;
        std::string secondary_name_;

        double primary_x_;
        double primary_y_;
        double primary_z_;
        double primary_t_;
        double primary_theta_;
        double primary_phi_;
        double primary_energy_;
        int primary_parent_particle_id_;
        int primary_particle_id_;
        std::string primary_name_;

        double prop_primary_x_;
        double prop_primary_y_;
        double prop_primary_z_;
        double prop_primary_t_;
        double prop_primary_theta_;
        double prop_primary_phi_;
        double prop_primary_energy_;
        int prop_primary_parent_particle_id_;
        int prop_primary_particle_id_;
        std::string prop_primary_name_;
        double prop_primary_propagated_distance_;
        double prop_primary_xi_;
        double prop_primary_yi_;
        double prop_primary_zi_;
        double prop_primary_ti_;
        double prop_primary_ei_;
        double prop_primary_xf_;
        double prop_primary_yf_;
        double prop_primary_zf_;
        double prop_primary_tf_;
        double prop_primary_ef_;
        double prop_primary_xc_;
        double prop_primary_yc_;
        double prop_primary_zc_;
        double prop_primary_tc_;
        double prop_primary_ec_;
        double prop_primary_elost_;

    #endif

    std::function<void(Particle*,int,std::pair<double,std::string>,double)> callback_;

public:

    Logger logger;

    //ASCII
    static bool store_in_ASCII_file_;

//----------------------------------------------------------------------------//

    static Output& getInstance()
    {
        static Output instance;
        return instance;
    }
//----------------------------------------------------------------------------//

    void SetLoggingConfigurationFile(std::string file);

//----------------------------------------------------------------------------//

    void FillSecondaryVector(Particle *particle, int secondary_id, std::pair<double,std::string> energy_loss, double distance);
    
//----------------------------------------------------------------------------//

    void SetSecondaryCallback(std::function<void(Particle*,int,std::pair<double,std::string>,double)> callback);

//----------------------------------------------------------------------------//

    void ClearSecondaryVector();

//----------------------------------------------------------------------------//

    void Close();

//----------------------------------------------------------------------------//

    #if ROOT_SUPPORT

        void EnableROOTOutput(std::string rootfile_name);

//----------------------------------------------------------------------------//

        void DisableROOTOutput();

//----------------------------------------------------------------------------//

        void StorePrimaryInTree(Particle *primary);

//----------------------------------------------------------------------------//

        void StorePropagatedPrimaryInTree(Particle *prop_primary);

    #endif

// ASCII OUTPUT
        void EnableASCIIOutput(std::string ASCII_Prefix, bool append = false);

     //----------------------------------------------------------------------------//

         void DisableASCIIOutput();

     //----------------------------------------------------------------------------//

         void StorePrimaryInASCII(Particle *primary);

     //----------------------------------------------------------------------------//

         void StorePropagatedPrimaryInASCII(Particle *prop_primary);

     //----------------------------------------------------------------------------//

    //Getter
    std::vector<Particle*> GetSecondarys() const
    {
        return secondarys_;
    }
    
    //Getter
    std::vector<double> GetSecondarysParentE() const
    {
        return secondarys_parent_E_;
    }

 };

#endif //ICECUBE

#endif //OUTPUT_H























