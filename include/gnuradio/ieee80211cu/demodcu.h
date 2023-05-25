/* -*- c++ -*- */
/*
 * Copyright 2023 Zelin Yun.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_IEEE80211CU_DEMODCU_H
#define INCLUDED_IEEE80211CU_DEMODCU_H

#include <gnuradio/ieee80211cu/api.h>
#include <gnuradio/block.h>

namespace gr {
  namespace ieee80211cu {

    /*!
     * \brief <+description of block+>
     * \ingroup ieee80211cu
     *
     */
    class IEEE80211CU_API demodcu : virtual public gr::block
    {
     public:
      typedef std::shared_ptr<demodcu> sptr;

      /*!
       * \brief Return a shared_ptr to a new instance of ieee80211cu::demodcu.
       *
       * To avoid accidental use of raw pointers, ieee80211cu::demodcu's
       * constructor is in a private implementation
       * class. ieee80211cu::demodcu::make is the public interface for
       * creating new instances.
       */
      static sptr make(int mupos, int mugid, bool ifdebug);
    };

  } // namespace ieee80211cu
} // namespace gr

#endif /* INCLUDED_IEEE80211CU_DEMODCU_H */
