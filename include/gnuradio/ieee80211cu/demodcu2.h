/* -*- c++ -*- */
/*
 * Copyright 2023 Zelin Yun.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_IEEE80211CU_DEMODCU2_H
#define INCLUDED_IEEE80211CU_DEMODCU2_H

#include <gnuradio/ieee80211cu/api.h>
#include <gnuradio/block.h>

namespace gr {
  namespace ieee80211cu {

    /*!
     * \brief <+description of block+>
     * \ingroup ieee80211cu
     *
     */
    class IEEE80211CU_API demodcu2 : virtual public gr::block
    {
     public:
      typedef std::shared_ptr<demodcu2> sptr;

      /*!
       * \brief Return a shared_ptr to a new instance of ieee80211cu::demodcu2.
       *
       * To avoid accidental use of raw pointers, ieee80211cu::demodcu2's
       * constructor is in a private implementation
       * class. ieee80211cu::demodcu2::make is the public interface for
       * creating new instances.
       */
      static sptr make();
    };

  } // namespace ieee80211cu
} // namespace gr

#endif /* INCLUDED_IEEE80211CU_DEMODCU2_H */
