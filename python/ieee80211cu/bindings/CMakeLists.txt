# Copyright 2020 Free Software Foundation, Inc.
#
# This file is part of GNU Radio
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

########################################################################
# Check if there is C++ code at all
########################################################################
if(NOT ieee80211cu_sources)
    MESSAGE(STATUS "No C++ sources... skipping python bindings")
    return()
endif(NOT ieee80211cu_sources)

########################################################################
# Check for pygccxml
########################################################################
GR_PYTHON_CHECK_MODULE_RAW(
    "pygccxml"
    "import pygccxml"
    PYGCCXML_FOUND
    )

include(GrPybind)

########################################################################
# Python Bindings
########################################################################

list(APPEND ieee80211cu_python_files
    preproccu_python.cc
    demodcu_python.cc
    demodcu2_python.cc
    modcu_python.cc
    genpkt_python.cc python_bindings.cc)

GR_PYBIND_MAKE_OOT(ieee80211cu
   ../../..
   gr::ieee80211cu
   "${ieee80211cu_python_files}")

# copy in bindings .so file for use in QA test module
add_custom_target(
  copy_bindings_for_tests ALL
  COMMAND
    ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/*.so"
    ${CMAKE_BINARY_DIR}/test_modules/gnuradio/ieee80211cu/
  DEPENDS ieee80211cu_python)

install(TARGETS ieee80211cu_python DESTINATION ${GR_PYTHON_DIR}/gnuradio/ieee80211cu COMPONENT pythonapi)
