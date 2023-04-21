

#ifndef INCLUDED_IEEE80211CU_GENPKT_IMPL_H
#define INCLUDED_IEEE80211CU_GENPKT_IMPL_H

#include <gnuradio/ieee80211cu/genpkt.h>
#include <gnuradio/pdu.h>
#include <vector>
#include <queue>
#include "cloud80211phy.h"
#include "cloud80211phycu.cuh"

using namespace boost::placeholders;

#define GENPKT_S_IDLE 0
#define GENPKT_S_SCEDULE 1
#define GENPKT_S_COPY 2
#define GENPKT_S_PAD 3

#define GENPKT_GR_PAD 160

namespace gr {
  namespace ieee80211cu {

    class genpkt_impl : public genpkt
    {
    private:
      int d_sEncode;
      int d_nGen;
      int d_pktSeq;
      std::queue< std::vector<uint8_t> > d_pktQ;
      std::vector<uint8_t> d_pktV;
      void msgRead(pmt::pmt_t msg);

      int d_pktFormat;
      int d_pktMcs0;
      int d_pktNss0;
      int d_pktLen0;
      int d_pktMcs1;
      int d_pktNss1;
      int d_pktLen1;
      int d_pktMuGroupId;
      
      int d_headerShift;
      int d_nTotal;
      int d_nCopied;

    protected:
      int calculate_output_stream_length(const gr_vector_int &ninput_items);

    public:
      genpkt_impl(const std::string& lengthtagname = "packet_len");
      ~genpkt_impl();

      int work(
              int noutput_items,
              gr_vector_int &ninput_items,
              gr_vector_const_void_star &input_items,
              gr_vector_void_star &output_items
      );
    };

  } // namespace ieee80211cu
} // namespace gr

#endif /* INCLUDED_IEEE80211CU_GENPKT_IMPL_H */
