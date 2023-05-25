/*
 *
 *     GNU Radio IEEE 802.11a/g/n/ac 2x2
 *     Demodulation and decoding of 802.11a/g/n/ac 1x1 and 2x2 formats cuda ver
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

#ifndef INCLUDED_IEEE80211CU_DEMODCU2_IMPL_H
#define INCLUDED_IEEE80211CU_DEMODCU2_IMPL_H

#include <gnuradio/ieee80211cu/demodcu2.h>
#include <gnuradio/fft/fft.h>
#include <boost/crc.hpp>
#include "cloud80211phy.h"
#include "cloud80211phycu.cuh"

#define dout d_debug&&std::cout

#define DEMOD_S_SYNC 0
#define DEMOD_S_RDTAG 1
#define DEMOD_S_FORMAT 2
#define DEMOD_S_VHT 3
#define DEMOD_S_HT 4
#define DEMOD_S_DEMOD 7
#define DEMOD_S_CLEAN 8

namespace gr {
  namespace ieee80211cu {

    class demodcu2_impl : public demodcu2
    {
     private:
      // block
      bool d_debug;
      int d_nProc;
      int d_nUsed;
      int d_sDemod;
      cloud80211demodcu d_demodCu;
      // received info from tag
      std::vector<gr::tag_t> tags;
      int d_pktSeq;
      int d_nSigLMcs;
      int d_nSigLLen;
      std::vector<gr_complex> d_HL;
      int d_nSigLSamp;
      int d_nSampConsumed;
      float d_cfo;
      float d_snr;
      float d_rssi;
      float d_sssnr0;    // spatial stream snr only for vht
      float d_sssnr1;    // spatial stream snr only for vht
      // check format
      svSigDecoder d_decoder;
      gr_complex d_sig1[64];
      gr_complex d_sig2[64];
      float d_sigHtIntedLlr[96];
      float d_sigHtCodedLlr[96];
      float d_sigVhtAIntedLlr[96];
      float d_sigVhtACodedLlr[96];
      float d_sigVhtB20IntedLlr[52];
      float d_sigVhtB20CodedLlr[52];
      uint8_t d_sigHtBits[48];
      uint8_t d_sigVhtABits[48];
      uint8_t d_sigVhtB20Bits[26];
      gr_complex d_sigVhtBQam0[52];
      gr_complex d_sigVhtBQam1[52];
      uint8_t d_sigVhtB20BitsCoded[52];
      uint8_t d_sigVhtB20BitsInted[52];
      // fft
      fft::fft_complex_fwd d_ofdm_fft;
      gr_complex d_fftLtfOut1[64];
      gr_complex d_fftLtfOut2[64];
      gr_complex d_fftLtfOut12[64];
      gr_complex d_fftLtfOut22[64];
      // packet info
      c8p_mod d_m;
      c8p_sigHt d_sigHt;
      c8p_sigVhtA d_sigVhtA;
      gr_complex d_H_NL[64];
      gr_complex d_H_NL22[4][64];
      gr_complex d_H_NL22_INV[4][64];
      gr_complex d_mu2x1Chan[128];
      int d_nSampTotoal;
      int d_nSampCopied;
      int d_nTrellis;
      gr_complex d_pilotNlLtf[8];
      boost::crc_32_type d_crc32;
      uint8_t d_psduBytes[CUDEMOD_B_MAX];
      // packet counter
      uint64_t d_nPktCorrect;
      uint64_t d_legacyMcsCount[8];
      uint64_t d_vhtMcsCount[10];
      uint64_t d_htMcsCount[8];

     public:
      demodcu2_impl(bool ifdebug);
      ~demodcu2_impl();

      // Where all the action really happens
      void forecast (int noutput_items, gr_vector_int &ninput_items_required);

      int general_work(int noutput_items,
           gr_vector_int &ninput_items,
           gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);
      void packetAssemble();
      void fftDemod(const gr_complex* sig, gr_complex* res);
      void nonLegacyChanEstimate(const gr_complex* sig1, const gr_complex* sig2);
      void vhtSigBDemod(const gr_complex* sig1, const gr_complex* sig2);
    };

  } // namespace ieee80211cu
} // namespace gr

#endif /* INCLUDED_IEEE80211CU_DEMODCU2_IMPL_H */
