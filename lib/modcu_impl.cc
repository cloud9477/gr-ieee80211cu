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

#include <gnuradio/io_signature.h>
#include "modcu_impl.h"

namespace gr {
  namespace ieee80211cu {

    modcu::sptr
    modcu::make()
    {
      return gnuradio::make_block_sptr<modcu_impl>(
        );
    }


    /*
     * The private constructor
     */
    modcu_impl::modcu_impl()
      : gr::block("modcu",
              gr::io_signature::make(1, 1, sizeof(uint8_t)),
              gr::io_signature::make(2, 2, sizeof(gr_complex)))
    {
      d_sModcu = MODCU_S_RDTAG;
    }

    /*
     * Our virtual destructor.
     */
    modcu_impl::~modcu_impl()
    {
    }

    void
    modcu_impl::forecast (int noutput_items, gr_vector_int &ninput_items_required)
    {
      if(d_sModcu == MODCU_S_RDPKT)
      {
        ninput_items_required[0] = noutput_items + (d_nPktTotal - d_nPktRead);
      }
      else
      {
        ninput_items_required[0] = noutput_items;
      }
    }

    int
    modcu_impl::general_work (int noutput_items,
                       gr_vector_int &ninput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items)
    {
      const uint8_t* inPkt = static_cast<const uint8_t*>(input_items[0]);
      // gr_complex* outSig1 = static_cast<gr_complex*>(output_items[0]);
      // gr_complex* outSig2 = static_cast<gr_complex*>(output_items[1]);
      d_nProc = ninput_items[0];
      d_nGen = noutput_items;
      d_nUsed = 0;
      d_nPassed = 0;

      if(d_sModcu == MODCU_S_RDTAG)
      {
        get_tags_in_range(d_tags, 0, nitems_read(0) , nitems_read(0) + 1);
        if (d_tags.size())
        {
          pmt::pmt_t d_meta = pmt::make_dict();
          for (auto tag : d_tags){
            d_meta = pmt::dict_add(d_meta, tag.key, tag.value);
          }
          d_pktFormat = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("format"), pmt::from_long(-1)));
          d_pktMcs0 = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("mcs0"), pmt::from_long(-1)));
          d_pktNss0 = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("nss0"), pmt::from_long(-1)));
          d_pktLen0 = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("len0"), pmt::from_long(-1)));
          d_pktSeq = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("seq"), pmt::from_long(-1)));
          d_nPktTotal = d_pktLen0;
          if(d_pktFormat == C8P_F_VHT_MU)
          {
            d_pktMcs1 = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("mcs1"), pmt::from_long(-1)));
            d_pktNss1 = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("nss1"), pmt::from_long(-1)));
            d_pktLen1 = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("len1"), pmt::from_long(-1)));
            d_pktMuGroupId = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("gid"), pmt::from_long(-1)));
            d_nPktTotal += d_pktLen1;
          }
          d_nPktRead = 0;
          d_sModcu = MODCU_S_RDPKT;
        }
      }

      if(d_sModcu == MODCU_S_RDPKT)
      {
        if(d_nProc >= (d_nPktTotal - d_nPktRead))
        {
          memcpy(d_pkt + d_nPktRead, inPkt, (d_nPktTotal - d_nPktRead));
          d_nUsed += (d_nPktTotal - d_nPktRead);
          d_sModcu = MODCU_S_MOD;
        }
        else
        {
          memcpy(d_pkt + d_nPktRead, inPkt, d_nProc);
          d_nPktRead += d_nProc;
          d_nUsed += d_nProc;
        }
      }
      
      if(d_sModcu == MODCU_S_MOD)
      {
        std::cout<<"ieee80211 modcu, pkt format:"<<d_pktFormat<<", seq:"<<d_pktSeq<<std::endl;
        /* mod legacy sig and non-legacy sig, concatenate with training field*/

        /* mod data with cuda*/
        d_sModcu = MODCU_S_RDTAG;
      }
      
      if(d_sModcu == MODCU_S_COPY)
      {

      }

      consume_each (d_nUsed);
      return d_nPassed;
    }

  } /* namespace ieee80211cu */
} /* namespace gr */
