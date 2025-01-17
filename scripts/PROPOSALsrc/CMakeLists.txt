cmake_minimum_required(VERSION 2.8)

project( PROPOSAL)

FIND_PACKAGE( Boost COMPONENTS program_options REQUIRED )

# Load some basic macros which are needed later on
include(FindROOT.cmake)

#################################################################
#################           log4cpp     #########################
#################################################################

include(resources/FindLog4cplus.cmake)

#################################################################
#################           GTest       #########################
#################################################################
IF(DEFINED ENV{GTEST_PATH})
  enable_testing()
  #1. Path
  #2. Path where the binarys will be copied to
  add_subdirectory($ENV{GTEST_PATH} ${CMAKE_CURRENT_BINARY_DIR}/gtestbin)
  set(DoTesting true)
  MESSAGE(STATUS "Found gtest-path")
ELSE()
    message("GTEST_PATH is not defined.  To build testsuite you have to set GTEST_PATH to gtest root directory")
    message("Remember to add gtest/include to C_INCLUDE_PATH and CPP_INCLUDE_PATH")
ENDIF()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

SET(DEBUG OFF)
option (DEBUG "DEBUG" OFF)
set( CMAKE_CXX_FLAGS "-std=c++11")
if(DEBUG)
        set( CMAKE_CXX_FLAGS "-g -pg -O0" )
endif()

SET(ICETRAY_INCLUDE_PATH "/home/koehne/Simulation/icesim4_candidate/V04-00-01-RC/icetray/public/")

include_directories("${PROJECT_SOURCE_DIR}/public" ${GTEST_INCLUDE_DIR} ${LOG4CPLUS_INCLUDE_DIR} ${Boost_INCLUDE_DIR} )

#if ROOT is found ROOT files with ROOT trees can be written
IF(ROOT_FOUND)
    add_definitions(-DROOT_SUPPORT=1)

    set(INCLUDE_DIRECTORIES
    ${ROOT_INCLUDE_DIR}
    )

    include_directories( ${INCLUDE_DIRECTORIES})

    set(LINK_DIRECTORIES
    ${ROOT_LIBRARY_DIR}
    ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    )

    link_directories( ${LINK_DIRECTORIES})

ELSE(ROOT_FOUND)

    add_definitions(-DROOT_SUPPORT=0)

    MESSAGE(STATUS "ROOT not found...")
    MESSAGE(STATUS "ROOT examples will not be builded.")
    MESSAGE(STATUS "No ROOT Output is available.")
    MESSAGE(STATUS "Make sure you have ROOT installed and ROOTSYS is set.")

ENDIF(ROOT_FOUND)

add_library(PROPOSAL
        private/PROPOSAL/Integral.cxx
        private/PROPOSAL/methods.cxx
        private/PROPOSAL/MathModel.cxx
        private/PROPOSAL/Bremsstrahlung.cxx
        private/PROPOSAL/CrossSections.cxx
        private/PROPOSAL/Decay.cxx
        private/PROPOSAL/Epairproduction.cxx
        private/PROPOSAL/Ionization.cxx
        private/PROPOSAL/Photonuclear.cxx
        private/PROPOSAL/Medium.cxx
        private/PROPOSAL/Particle.cxx
        private/PROPOSAL/EnergyCutSettings.cxx
	private/PROPOSAL/Interpolant.cxx
        private/PROPOSAL/StandardNormal.cxx
        private/PROPOSAL/RootFinder.cxx
        private/PROPOSAL/ProcessCollection.cxx
        private/PROPOSAL/Propagator.cxx
        private/PROPOSAL/ContinuousRandomization.cxx
        private/PROPOSAL/Geometry.cxx
        private/PROPOSAL/Scattering.cxx
        private/PROPOSAL/ScatteringFirstOrder.cxx
        private/PROPOSAL/ScatteringMoliere.cxx
        private/PROPOSAL/Output.cxx
	)

set_target_properties(PROPOSAL PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -Wall -Werror")# -Wextra -pedantic")

IF(ROOT_FOUND)
    target_link_libraries(PROPOSAL ${Boost_LIBRARIES} ${LOG4CPLUS_LIBRARIES} ${ROOT_LIBRARIES})
ELSE(ROOT_FOUND)
    target_link_libraries(PROPOSAL ${Boost_LIBRARIES} ${LOG4CPLUS_LIBRARIES})
ENDIF(ROOT_FOUND)

add_executable(PROPOSALtest
        private/test/PROPOSAL.cxx
)
#set_target_properties(PROPOSALtest PROPERTIES LINK_FLAGS "-L/home/koehne/Simulation/icesim4_candidate/buildV04-00-01-RC/lib")

target_link_libraries(PROPOSALtest PROPOSAL)

add_executable(PROPOSAL1km
        private/test/PROPOSAL_1km.cxx
)
#set_target_properties(PROPOSALtest PROPERTIES LINK_FLAGS "-L/home/koehne/Simulation/icesim4_candidate/buildV04-00-01-RC/lib")

target_link_libraries(PROPOSAL1km PROPOSAL)

add_executable(errtest
        private/test/err.cxx
)
#set_target_properties(PROPOSALtest PROPERTIES LINK_FLAGS "-L/home/koehne/Simulation/icesim4_candidate/buildV04-00-01-RC/lib")

target_link_libraries(errtest PROPOSAL)

execute_process(COMMAND ln -s ${CMAKE_SOURCE_DIR}/resources ${CMAKE_BINARY_DIR}/resources)

add_executable(WriteSectorsFromDomList
        private/test/WriteSectorsFromDomList.cxx
)
target_link_libraries(WriteSectorsFromDomList PROPOSAL)


IF(DoTesting)
  execute_process(COMMAND mkdir ${PROPOSAL_BINARY_DIR}/bin/ OUTPUT_VARIABLE _output OUTPUT_STRIP_TRAILING_WHITESPACE)

  #create tar directory with "tar -czvf TestFiles.tar.Z TestFiles/" and put it in Test directory
  execute_process(COMMAND  tar -xvf ${PROJECT_SOURCE_DIR}/Test/TestFiles.tar.Z -C ${PROPOSAL_BINARY_DIR}/bin/
                    OUTPUT_VARIABLE _output OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND  tar -xvf ${PROJECT_SOURCE_DIR}/Test/TestFiles2.tar.Z -C ${PROPOSAL_BINARY_DIR}/bin/
                    OUTPUT_VARIABLE _output OUTPUT_STRIP_TRAILING_WHITESPACE)

  add_executable(UnitTest_Scattering Test/Scattering_TEST.cxx)
  add_executable(UnitTest_StandardNormal Test/StandardNormal_Test.cxx)
  add_executable(UnitTest_Photonuclear Test/Photonuclear_TEST.cxx)
  add_executable(UnitTest_Integral Test/Integral_TEST.cxx)
  add_executable(UnitTest_Interpolant Test/Interpolant_TEST.cxx)
  add_executable(UnitTest_Bremsstrahlung Test/Bremsstrahlung_TEST.cxx)
  add_executable(UnitTest_Epairproduction Test/Epairproduction_TEST.cxx)
  add_executable(UnitTest_Ionization Test/Ionization_TEST.cxx)
  add_executable(UnitTest_RootFinder Test/RootFinder_TEST.cxx)
  add_executable(UnitTest_Medium Test/Medium_TEST.cxx)
  add_executable(UnitTest_Particle Test/Particle_TEST.cxx)
  add_executable(UnitTest_EnergyCutSettings Test/EnergyCutSettings_TEST.cxx)
  add_executable(UnitTest_Decay Test/Decay_TEST.cxx)
  add_executable(UnitTest_ProcessCollection Test/ProcessCollection_TEST.cxx)
  add_executable(UnitTest_ContinuousRandomization Test/ContinuousRandomization_TEST.cxx)
  add_executable(UnitTest_Geometry Test/Geometry_TEST.cxx)

  target_link_libraries(UnitTest_Scattering PROPOSAL gtest gtest_main)
  target_link_libraries(UnitTest_StandardNormal PROPOSAL gtest gtest_main)
  target_link_libraries(UnitTest_Integral PROPOSAL gtest gtest_main)
  target_link_libraries(UnitTest_Interpolant PROPOSAL gtest gtest_main)
  target_link_libraries(UnitTest_Ionization PROPOSAL gtest gtest_main)
  target_link_libraries(UnitTest_Bremsstrahlung PROPOSAL gtest gtest_main)
  target_link_libraries(UnitTest_Epairproduction PROPOSAL gtest gtest_main)
  target_link_libraries(UnitTest_Photonuclear PROPOSAL gtest gtest_main)
  target_link_libraries(UnitTest_RootFinder PROPOSAL gtest gtest_main)
  target_link_libraries(UnitTest_Medium PROPOSAL gtest gtest_main)
  target_link_libraries(UnitTest_Particle PROPOSAL gtest gtest_main)
  target_link_libraries(UnitTest_EnergyCutSettings PROPOSAL gtest gtest_main)
  target_link_libraries(UnitTest_Decay PROPOSAL gtest gtest_main)
  target_link_libraries(UnitTest_ProcessCollection PROPOSAL gtest gtest_main)
  target_link_libraries(UnitTest_ContinuousRandomization PROPOSAL gtest gtest_main)
  target_link_libraries(UnitTest_Geometry PROPOSAL gtest gtest_main)

  add_test(UnitTest_Scattering bin/UnitTest_Scattering)
  add_test(UnitTest_ContinuousRandomization bin/UnitTest_ContinuousRandomization)
  add_test(UnitTest_ProcessCollection bin/UnitTest_ProcessCollection)
  add_test(UnitTest_Decay bin/UnitTest_Decay)
  add_test(UnitTest_RootFinder bin/UnitTest_RootFinder)
  add_test(UnitTest_StandardNormal bin/UnitTest_StandardNormal)
  add_test(UnitTest_Integral bin/UnitTest_Integral)
  add_test(UnitTest_Medium bin/UnitTest_Medium)
  add_test(UnitTest_Particle bin/UnitTest_Particle)
  add_test(UnitTest_EnergyCutSettings bin/UnitTest_EnergyCutSettings)
  add_test(UnitTest_Interpolant bin/UnitTest_Interpolant)
  add_test(UnitTest_Epairproduction bin/UnitTest_Epairproduction)
  add_test(UnitTest_Ionization bin/UnitTest_Ionization)
  add_test(UnitTest_Bremsstrahlung bin/UnitTest_Bremsstrahlung)
  add_test(UnitTest_Photonuclear bin/UnitTest_Photonuclear)
  add_test(UnitTest_Geometry bin/UnitTest_Geometry)

ENDIF()

ADD_SUBDIRECTORY( doc )
IF(ROOT_FOUND)
    ADD_SUBDIRECTORY( root_examples )
ENDIF(ROOT_FOUND)
