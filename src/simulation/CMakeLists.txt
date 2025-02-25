add_library(simulations STATIC 
    basicSimulation.cpp
    simulation.cpp
    multirankSimulation.cpp
    multirankSimulationBoundaries.cpp
    )
target_link_libraries(simulations utilities)
