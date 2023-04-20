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
      memset(d_pktVhtSigBCrc, 0, 8);
      memset(d_pktVhtSigBCrc1, 0, 8);
      memset(d_pkt, 0, CUDEMOD_B_MAX);
      memset((uint8_t*) d_sig0, 0, sizeof(gr_complex) * CUDEMOD_S_MAX * 80 + MODCU_GAP_HEAD + MODCU_GAP_TAIL);
      memset((uint8_t*) d_sig1, 0, sizeof(gr_complex) * CUDEMOD_S_MAX * 80 + MODCU_GAP_HEAD + MODCU_GAP_TAIL);
      d_nSampTotal = 0;
      d_nSampCopied = 0;
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
      gr_complex* outSig0 = static_cast<gr_complex*>(output_items[0]);
      gr_complex* outSig1 = static_cast<gr_complex*>(output_items[1]);
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
            std::cout<<"ieee80211 modcu, mu #"<<d_pktSeq<<", mcs0:"<<d_pktMcs0<<", nss0:"<<d_pktNss0<<", len0:"<<d_pktLen0<<", mcs1:"<<d_pktMcs1<<", nss1:"<<d_pktNss1<<", len1:"<<d_pktLen1<<std::endl;
            d_nPktTotal += d_pktLen1;
            formatToModMu(&d_m, d_pktMcs0, 1, d_pktLen0, d_pktMcs1, 1, d_pktLen1);
            d_m.groupId = d_pktMuGroupId;
          }
          else
          {
            std::cout<<"ieee80211 modcu, su #"<<d_pktSeq<<", format:"<<d_pktFormat<<", mcs:"<<d_pktMcs0<<", nss:"<<d_pktNss0<<", len:"<<d_pktLen0<<std::endl;
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
          std::cout<<"ieee80211 modcu, get packet."<<std::endl;
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
        /* mod legacy sig and non-legacy sig, concatenate with training field*/
        if(d_pktFormat == C8P_F_VHT_BFQ_R)
        {
          memcpy(d_vhtBfQBR, d_pkt, 1024);
          d_sModcu = MODCU_S_RDTAG;
        }
        else if(d_pktFormat == C8P_F_VHT_BFQ_I)
        {
          memcpy(d_vhtBfQBI, d_pkt, 1024);
          float* tmpFloatPR = (float*)d_vhtBfQBR;
          float* tmpFloatPI = (float*)d_vhtBfQBI;
          for(int i=0;i<128;i++)
          {
            d_vhtBfQ[i+128] = gr_complex(*tmpFloatPR, *tmpFloatPI);
            tmpFloatPR += 1;
            tmpFloatPI += 1;
          }
          for(int i=0;i<128;i++)
          {
            d_vhtBfQ[i] = gr_complex(*tmpFloatPR, *tmpFloatPI);
            tmpFloatPR += 1;
            tmpFloatPI += 1;
          }
          d_modcu.cuModBfQCopy((cuFloatComplex*) d_vhtBfQ);
          d_sModcu = MODCU_S_RDTAG;
        }
        else if(d_pktFormat == C8P_F_VHT_MU)
        {
          d_pream.genVHTMuMimo(&d_m, d_vhtBfQ, d_sig0 + MODCU_GAP_HEAD, d_sig1 + MODCU_GAP_HEAD, d_pktVhtSigBCrc, d_pktVhtSigBCrc1);
          d_modcu.cuModPktCopy(0, d_nPktTotal, d_pkt);
          d_modcu.cuModVHTMuMimo(&d_m, (cuFloatComplex*) (d_sig0 + MODCU_GAP_HEAD + 720 + 80*d_m.nLTF), (cuFloatComplex*) (d_sig1 + MODCU_GAP_HEAD + 720 + 80*d_m.nLTF), d_pktVhtSigBCrc, d_pktVhtSigBCrc1);
          procWindowing(d_sig0 + MODCU_GAP_HEAD, d_m.nSym + 4 + d_m.nLTF);
          procWindowing(d_sig1 + MODCU_GAP_HEAD, d_m.nSym + 4 + d_m.nLTF);
          // memset((uint8_t*)(d_sig0 + MODCU_GAP_HEAD + 720 + d_m.nLTF * 80 + d_m.nSym * d_m.nSymSamp), 0, MODCU_GAP_TAIL * sizeof(gr_complex));
          // memset((uint8_t*)(d_sig1 + MODCU_GAP_HEAD + 720 + d_m.nLTF * 80 + d_m.nSym * d_m.nSymSamp), 0, MODCU_GAP_TAIL * sizeof(gr_complex));
          d_nSampTotal = 720 + d_m.nLTF * 80 + d_m.nSym * d_m.nSymSamp + MODCU_GAP_HEAD + MODCU_GAP_TAIL;
          d_sModcu = MODCU_S_COPY;
        }
        else
        {
          // SU
          d_modcu.cuModPktCopy(0, d_nPktTotal, d_pkt);
          formatToModSu(&d_m, d_pktFormat, d_pktMcs0, d_pktNss0, d_pktLen0);
          if(d_m.format == C8P_F_L)
          {
            d_pream.genLegacy(&d_m, d_sig0 + MODCU_GAP_HEAD);
            d_modcu.cuModLHTSiso(&d_m, (cuFloatComplex*) (d_sig0 + MODCU_GAP_HEAD + 400));
            procWindowing(d_sig0 + MODCU_GAP_HEAD, d_m.nSym + 1);
            d_nSampTotal = 400 + d_m.nSym * d_m.nSymSamp + MODCU_GAP_HEAD + MODCU_GAP_TAIL;
          }
          else if(d_m.nSS == 1)
          {
            if(d_m.format == C8P_F_HT)
            {
              d_pream.genHTSiso(&d_m, d_sig0 + MODCU_GAP_HEAD);
              d_modcu.cuModLHTSiso(&d_m, (cuFloatComplex*) (d_sig0 + MODCU_GAP_HEAD + 640 + 80*d_m.nLTF));
              procWindowing(d_sig0 + MODCU_GAP_HEAD, d_m.nSym + 4 + d_m.nLTF);
              d_nSampTotal = 640 + d_m.nLTF * 80 + d_m.nSym * d_m.nSymSamp + MODCU_GAP_HEAD + MODCU_GAP_TAIL;
            }
            else
            {
              d_pream.genVHTSiso(&d_m, d_sig0 + MODCU_GAP_HEAD, d_pktVhtSigBCrc);
              d_modcu.cuModVHTSiso(&d_m, (cuFloatComplex*) (d_sig0 + MODCU_GAP_HEAD + 720 + 80*d_m.nLTF), d_pktVhtSigBCrc);
              procWindowing(d_sig0 + MODCU_GAP_HEAD, d_m.nSym + 5 + d_m.nLTF);
              d_nSampTotal = 720 + d_m.nLTF * 80 + d_m.nSym * d_m.nSymSamp + MODCU_GAP_HEAD + MODCU_GAP_TAIL;
            }
          }
          else
          {
            if(d_m.format == C8P_F_HT)
            {
              d_pream.genHTSuMimo(&d_m, d_sig0 + MODCU_GAP_HEAD, d_sig1 + MODCU_GAP_HEAD);
              d_modcu.cuModHTMimo(&d_m, (cuFloatComplex*) (d_sig0 + MODCU_GAP_HEAD + 640 + 80*d_m.nLTF), (cuFloatComplex*) (d_sig1 + MODCU_GAP_HEAD + 640 + 80*d_m.nLTF));
              procWindowing(d_sig0 + MODCU_GAP_HEAD, d_m.nSym + 4 + d_m.nLTF);
              procWindowing(d_sig1 + MODCU_GAP_HEAD, d_m.nSym + 4 + d_m.nLTF);
              d_nSampTotal = 640 + d_m.nLTF * 80 + d_m.nSym * d_m.nSymSamp + MODCU_GAP_HEAD + MODCU_GAP_TAIL;
            }
            else
            {
              d_pream.genVHTSuMimo(&d_m, d_sig0 + MODCU_GAP_HEAD, d_sig1 + MODCU_GAP_HEAD, d_pktVhtSigBCrc);
              d_modcu.cuModVHTSuMimo(&d_m, (cuFloatComplex*) (d_sig0 + MODCU_GAP_HEAD + 720 + 80*d_m.nLTF), (cuFloatComplex*) (d_sig1 + MODCU_GAP_HEAD + 720 + 80*d_m.nLTF), d_pktVhtSigBCrc);
              procWindowing(d_sig0 + MODCU_GAP_HEAD, d_m.nSym + 4 + d_m.nLTF);
              procWindowing(d_sig1 + MODCU_GAP_HEAD, d_m.nSym + 4 + d_m.nLTF);
              d_nSampTotal = 720 + d_m.nLTF * 80 + d_m.nSym * d_m.nSymSamp + MODCU_GAP_HEAD + MODCU_GAP_TAIL;
            }
          }
          d_sModcu = MODCU_S_COPY;
        }
      }
      
      if(d_sModcu == MODCU_S_COPY)
      {
        d_sModcu = MODCU_S_RDTAG;
      }

      consume_each (d_nUsed);
      return d_nPassed;
    }

  } /* namespace ieee80211cu */
} /* namespace gr */
