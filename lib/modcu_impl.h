/*
 *
 *     GNU Radio IEEE 802.11a/g/n/ac 2x2
 *     Coding and Modulation for 1x1 and 2x2 transmitter cuda ver
 *     Copyright (C) Dec 1, 2022  Zelin Yun
 *
 *     This program is free software: you can redistribute it and/or modify
 *     it under the terms of the GNU Affero General Public License as published
 *     by the Free Software Foundation, either version 3 of the License, or
 *     (at your option) any later version.
 *
 *     This program is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY; without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU Affero General Public License for more details.
 *
 *     You should have received a copy of the GNU Affero General Public License
 *     along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef INCLUDED_IEEE80211CU_MODCU_IMPL_H
#define INCLUDED_IEEE80211CU_MODCU_IMPL_H

#include <gnuradio/ieee80211cu/modcu.h>
#include <gnuradio/fft/fft.h>
#include <uhd/types/time_spec.hpp>
#include <boost/crc.hpp>
#include "cloud80211phy.h"
#include "cloud80211phycu.cuh"

#define MODCU_S_RDTAG 1
#define MODCU_S_RDPKT 2
#define MODCU_S_MOD 3
#define MODCU_S_COPY 4
#define MODCU_S_CLEAN 5

#define MODCU_GAP_HEAD 80
#define MODCU_GAP_TAIL 80

#define MODCU_GR_PAD 160

namespace gr {
  namespace ieee80211cu {

    class modcu_impl : public modcu
    {
     private:
      int d_sModcu;
      int d_nProc;
      int d_nGen;
      int d_nUsed;
      int d_nPassed;
      // input pkt
      std::vector<gr::tag_t> d_tags;
      int d_pktFormat;
      int d_pktSeq;
      int d_pktMcs0;
      int d_pktNss0;
      int d_pktLen0;
      int d_pktMcs1;
      int d_pktNss1;
      int d_pktLen1;
      int d_pktMuGroupId;
      int d_nPktTotal;
      int d_nPktRead;
      uint8_t d_pktVhtSigBCrc[8];
      uint8_t d_pktVhtSigBCrc1[8];
      uint8_t d_pkt[CUDEMOD_B_MAX + MODCU_GR_PAD];
      gr_complex d_sig0[CUDEMOD_S_MAX * 80 + MODCU_GAP_HEAD + MODCU_GAP_TAIL];
      gr_complex d_sig1[CUDEMOD_S_MAX * 80 + MODCU_GAP_HEAD + MODCU_GAP_TAIL];
      // modulation
      c8p_mod d_m;
      c8p_preamble d_pream;
      cloud80211modcu d_modcu;
      uint8_t d_vhtBfQBR[1024];
      uint8_t d_vhtBfQBI[1024];
      gr_complex d_vhtBfQ[256]; // fft shifted
      // copy samples out
      int d_nSampTotal;
      int d_nSampCopied;

     public:
      modcu_impl();
      ~modcu_impl();

      // Where all the action really happens
      void forecast (int noutput_items, gr_vector_int &ninput_items_required);

      int general_work(int noutput_items,
           gr_vector_int &ninput_items,
           gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);
      void addTag();
    };

  } // namespace ieee80211cu
} // namespace gr

#endif /* INCLUDED_IEEE80211CU_MODCU_IMPL_H */
