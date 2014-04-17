project(directIntegrationSolver)
cmake_minimum_required(VERSION 2.8)
aux_source_directory(. SRC_LIST)
add_executable(${PROJECT_NAME} ${SRC_LIST})

if(UNIX)
    list(APPEND CMAKE_CXX_FLAGS "-std=c++11 -Wall -Wextra ${CMAKE_CXX_FLAGS}")
endif()

find_package(sbf REQUIRED)
include_directories(${SBF_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${SBF_LIBRARIES})

find_package(Boost COMPONENTS program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
target_link_libraries(${PROJECT_NAME} ${MPI_LIBRARIES})
set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()
