/*
 * Copyright 2023 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

/***********************************************************************************/
/* This file is automatically generated using bindtool and can be manually edited  */
/* The following lines can be configured to regenerate this file during cmake      */
/* If manual edits are made, the following tags should be modified accordingly.    */
/* BINDTOOL_GEN_AUTOMATIC(0)                                                       */
/* BINDTOOL_USE_PYGCCXML(0)                                                        */
/* BINDTOOL_HEADER_FILE(modcu.h)                                        */
/* BINDTOOL_HEADER_FILE_HASH(7a68c63384f8399cc4139113b3fd356f)                     */
/***********************************************************************************/

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <gnuradio/ieee80211cu/modcu.h>
// pydoc.h is automatically generated in the build directory
#include <modcu_pydoc.h>

void bind_modcu(py::module& m)
{

    using modcu    = gr::ieee80211cu::modcu;


    py::class_<modcu, gr::block, gr::basic_block,
        std::shared_ptr<modcu>>(m, "modcu", D(modcu))

        .def(py::init(&modcu::make),
           D(modcu,make)
        )
        



        ;




}








