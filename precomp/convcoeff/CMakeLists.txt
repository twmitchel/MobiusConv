cmake_minimum_required(VERSION 3.9)
project("convcoeff")

file(GLOB SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c)
file(GLOB HED_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
file(GLOB INCL_HED ${CMAKE_CURRENT_SOURCE_DIR}/../include/*.h ${CMAKE_CURRENT_SOURCE_DIR}/../include/*.inl)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/../include)
#source_group("Misha" FILES ${MISHA_HED})

add_executable(${PROJECT_NAME} ${SRC_FILES} ${HED_FILES} ${INCL_HED})

