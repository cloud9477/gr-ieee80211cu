

#include <gnuradio/io_signature.h>
#include "genpkt_impl.h"

namespace gr {
  namespace ieee80211cu {

    genpkt::sptr
    genpkt::make(const std::string& tsb_tag_key)
    {
      return gnuradio::make_block_sptr<genpkt_impl>(tsb_tag_key
        );
    }


    /*
     * The private constructor
     */
    genpkt_impl::genpkt_impl(const std::string& tsb_tag_key)
      : gr::tagged_stream_block("genpkt",
              gr::io_signature::make(0, 0, 0),
              gr::io_signature::make(1, 1, sizeof(uint8_t)), tsb_tag_key)
    {
      d_sEncode = GENPKT_S_IDLE;
      d_pktSeq = 0;

      message_port_register_in(pmt::mp("pdus"));
      set_msg_handler(pmt::mp("pdus"), boost::bind(&genpkt_impl::msgRead, this, _1));
    }

    /*
     * Our virtual destructor.
     */
    genpkt_impl::~genpkt_impl()
    {
    }

    void
    genpkt_impl::msgRead(pmt::pmt_t msg)
    {
      /* 1B format, 1B mcs, 1B nss, 2B len, total 5B, len is 0 then NDP*/
      pmt::pmt_t msgVec = pmt::cdr(msg);
      int pktLen = pmt::blob_length(msgVec);
      size_t tmpOffset(0);
      const uint8_t *tmpPkt = (const uint8_t *)pmt::uniform_vector_elements(msgVec, tmpOffset);
      std::cout<<"msg read packet len " << pktLen <<std::endl;
      if(pktLen < 5){
        return;
      }
      std::vector<uint8_t> pktVec(tmpPkt, tmpPkt + pktLen);
      d_pktQ.push(pktVec);
    }

    int
    genpkt_impl::calculate_output_stream_length(const gr_vector_int &ninput_items)
    {
      if(d_sEncode == GENPKT_S_SCEDULE)
      {
        d_sEncode = GENPKT_S_COPY;
        std::cout<<"calculate length " << d_nTotal <<std::endl;
        return d_nTotal;
      }
      return 0;
    }

    int
    genpkt_impl::work (int noutput_items,
                       gr_vector_int &ninput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items)
    {
      uint8_t* outPkt = static_cast<uint8_t*>(output_items[0]);
      d_nGen = noutput_items;

      if(d_sEncode == GENPKT_S_IDLE)
      {
        if(d_pktQ.size())
        {
          d_pktV = d_pktQ.front();
          d_pktQ.pop();
          d_pktFormat = (int)d_pktV[0];
          std::cout<<"idle packet format " << d_pktFormat <<std::endl;
          if(d_pktFormat == C8P_F_VHT_BFQ_R || d_pktFormat == C8P_F_VHT_BFQ_I)
          {
            d_pktMcs0 = 0;
            d_pktLen0 = 1024;
            d_headerShift = 1;
            d_nTotal = d_pktLen0;
          }
          else if(d_pktFormat == C8P_F_VHT_MU)
          {
            d_pktMcs0 = (int)d_pktV[1];
            d_pktNss0 = (int)d_pktV[2];
            d_pktLen0 = ((int)d_pktV[4] * 256  + (int)d_pktV[3]);
            d_pktMcs1 = (int)d_pktV[5];
            d_pktNss1 = (int)d_pktV[6];
            d_pktLen1 = ((int)d_pktV[8] * 256  + (int)d_pktV[7]);
            d_pktMuGroupId = (int)d_pktV[9];
            d_headerShift = 10;
            d_nTotal = d_pktLen0 + d_pktLen1;
          }
          else
          {
            d_pktMcs0 = (int)d_pktV[1];
            d_pktNss0 = (int)d_pktV[2];
            d_pktLen0 = ((int)d_pktV[4] * 256  + (int)d_pktV[3]);
            d_headerShift = 5;
            d_nTotal = d_pktLen0;
          }

          if(d_pktV.size() < (uint64_t)(d_nTotal + d_headerShift) || d_nTotal > CUDEMOD_B_MAX)
          {
            d_sEncode = GENPKT_S_IDLE;  // if error, keep in idle
          }
          else
          {
            d_sEncode = GENPKT_S_SCEDULE;
            // write tag
            pmt::pmt_t dict = pmt::make_dict();
            dict = pmt::dict_add(dict, pmt::mp("format"), pmt::from_long(d_pktFormat));
            dict = pmt::dict_add(dict, pmt::mp("mcs0"), pmt::from_long(d_pktMcs0));
            dict = pmt::dict_add(dict, pmt::mp("nss0"), pmt::from_long(d_pktNss0));
            dict = pmt::dict_add(dict, pmt::mp("len0"), pmt::from_long(d_pktLen0));
            if(d_pktFormat == C8P_F_VHT_MU)
            {
              dict = pmt::dict_add(dict, pmt::mp("mcs1"), pmt::from_long(d_pktMcs1));
              dict = pmt::dict_add(dict, pmt::mp("nss1"), pmt::from_long(d_pktNss1));
              dict = pmt::dict_add(dict, pmt::mp("len1"), pmt::from_long(d_pktLen1));
              dict = pmt::dict_add(dict, pmt::mp("gid"), pmt::from_long(d_pktMuGroupId));
            }
            dict = pmt::dict_add(dict, pmt::mp("seq"), pmt::from_long(d_pktSeq));
            
            pmt::pmt_t pairs = pmt::dict_items(dict);
            for (size_t i = 0; i < pmt::length(pairs); i++) {
                pmt::pmt_t pair = pmt::nth(i, pairs);
                add_item_tag(0,                   // output port index
                              nitems_written(0),  // output sample index
                              pmt::car(pair),     
                              pmt::cdr(pair),
                              alias_pmt());
            }
            std::cout<<"write tag with seq "<<d_pktSeq<<std::endl;
            d_pktSeq++;
          }
        }
        return 0;
      }

      else if(d_sEncode == GENPKT_S_SCEDULE)
      {
        return 0;
      }

      else
      {
        if(d_nGen >= (d_nTotal - d_nCopied))
        {
          memcpy(outPkt, d_pktV.data() + d_headerShift + d_nCopied, (d_nTotal - d_nCopied));
          d_sEncode = GENPKT_S_IDLE;
          return (d_nTotal - d_nCopied);
        }
        else
        {
          memcpy(outPkt, d_pktV.data() + d_headerShift + d_nCopied, d_nGen);
          d_nCopied += d_nGen;
          return d_nGen;
        }
      }

      return 0;
    }

  } /* namespace ieee80211cu */
} /* namespace gr */
