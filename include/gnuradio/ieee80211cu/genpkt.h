/* -*- c++ -*- */
/*
 * Copyright 2022 Zelin Yun.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_IEEE80211CU_GENPKT_H
#define INCLUDED_IEEE80211CU_GENPKT_H

#include <gnuradio/ieee80211cu/api.h>
#include <gnuradio/tagged_stream_block.h>

namespace gr {
  namespace ieee80211cu {

    /*!
     * \brief <+description of block+>
     * \ingroup ieee80211cu
     *
     */
    class IEEE80211CU_API genpkt : virtual public gr::tagged_stream_block
    {
     public:
      typedef std::shared_ptr<genpkt> sptr;

      /*!
       * \brief Return a shared_ptr to a new instance of ieee80211cu::genpkt.
       *
       * To avoid accidental use of raw pointers, ieee80211cu::genpkt's
       * constructor is in a private implementation
       * class. ieee80211cu::genpkt::make is the public interface for
       * creating new instances.
       */
      static sptr make(const std::string& lengthtagname = "packet_len");
    };

  } // namespace ieee80211cu
} // namespace gr

#endif /* INCLUDED_IEEE80211CU_GENPKT_H */
