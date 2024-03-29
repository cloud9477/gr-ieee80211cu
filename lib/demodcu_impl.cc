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

#include <gnuradio/io_signature.h>
#include "demodcu_impl.h"

namespace gr {
  namespace ieee80211cu {

    demodcu::sptr
    demodcu::make(int mupos, int mugid, bool ifdebug)
    {
      return gnuradio::make_block_sptr<demodcu_impl>(mupos, mugid, ifdebug
        );
    }


    /*
     * The private constructor
     */
    demodcu_impl::demodcu_impl(int mupos, int mugid, bool ifdebug)
      : gr::block("demodcu",
              gr::io_signature::make(1, 1, sizeof(gr_complex)),
              gr::io_signature::make(0, 0, 0)),
              d_muPos(mupos),
              d_muGroupId(mugid),
              d_debug(ifdebug),
              d_ofdm_fft(64,1)
    {
      message_port_register_out(pmt::mp("out"));
      d_nProc = 0;
      d_nUsed = 0;
      // d_debug = true;
      d_sDemod = DEMOD_S_RDTAG;
      d_nPktCorrect = 0;
      memset(d_vhtMcsCount, 0, sizeof(uint64_t) * 10);
      memset(d_legacyMcsCount, 0, sizeof(uint64_t) * 8);
      memset(d_htMcsCount, 0, sizeof(uint64_t) * 8);
      set_tag_propagation_policy(block::TPP_DONT);
    }

    /*
     * Our virtual destructor.
     */
    demodcu_impl::~demodcu_impl()
    {
    }

    void
    demodcu_impl::forecast (int noutput_items, gr_vector_int &ninput_items_required)
    {
      int tmpRequired = noutput_items + 160;
      if(d_sDemod == DEMOD_S_DEMOD)
      {
        if((noutput_items + (d_nSampTotoal - d_nSampCopied)) <= 4096)
          tmpRequired = noutput_items + (d_nSampTotoal - d_nSampCopied);
        else
          tmpRequired = 4096;
      }
      ninput_items_required[0] = tmpRequired;
    }

    int
    demodcu_impl::general_work (int noutput_items,
                       gr_vector_int &ninput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items)
    {
      const gr_complex* inSig = static_cast<const gr_complex*>(input_items[0]);
      d_nProc = ninput_items[0];
      d_nUsed = 0;

      if(d_sDemod == DEMOD_S_RDTAG)
      {
        // tags, which input, start, end
        get_tags_in_range(tags, 0, nitems_read(0) , nitems_read(0) + 1);
        if (tags.size())
        {
          pmt::pmt_t d_meta = pmt::make_dict();
          for (auto tag : tags){
            d_meta = pmt::dict_add(d_meta, tag.key, tag.value);
          }
          d_cfo = pmt::to_float(pmt::dict_ref(d_meta, pmt::mp("cfo"), pmt::from_float(0.0f)));
          d_snr = pmt::to_float(pmt::dict_ref(d_meta, pmt::mp("snr"), pmt::from_float(0.0f)));
          d_rssi = pmt::to_float(pmt::dict_ref(d_meta, pmt::mp("rssi"), pmt::from_float(0.0f)));
          d_pktSeq = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("seq"), pmt::from_long(-1)));
          d_nSigLMcs = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("mcs"), pmt::from_long(-1)));
          d_nSigLLen = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("len"), pmt::from_long(-1)));
          d_nSigLSamp = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("nsamp"), pmt::from_long(-1)));
          d_HL = pmt::c32vector_elements(pmt::dict_ref(d_meta, pmt::mp("chan"), pmt::PMT_NIL));
          d_nSampConsumed = 0;
          d_nSigLSamp = d_nSigLSamp + 320;
          d_nSampCopied = 0;
          if(d_nSigLMcs > 0)
          {
            signalParserL(d_nSigLMcs, d_nSigLLen, &d_m);
            d_demodCu.cuDemodChanSiso((cuFloatComplex*) &d_HL[0]);
            d_nSampTotoal = d_m.nSymSamp * d_m.nSym;
            d_sDemod = DEMOD_S_DEMOD;
          }
          else
          {
            d_sDemod = DEMOD_S_FORMAT;
          }
        }
      }

      if(d_sDemod ==  DEMOD_S_FORMAT && (d_nProc - d_nUsed) >= 160)
      {
        fftDemod(&inSig[d_nUsed + 8], d_fftLtfOut1);
        fftDemod(&inSig[d_nUsed + 8+80], d_fftLtfOut2);
        procNLSigDemodDeint(d_fftLtfOut1, d_fftLtfOut2, d_HL, d_sigHtCodedLlr, d_sigVhtACodedLlr);
        d_decoder.decode(d_sigVhtACodedLlr, d_sigVhtABits, 48);
        if(signalCheckVhtA(d_sigVhtABits))
        {
          // go to vht
          signalParserVhtA(d_sigVhtABits, &d_m, &d_sigVhtA);
          d_sDemod = DEMOD_S_VHT;
          d_nSampConsumed += 160;
          d_nUsed += 160;
        }
        else
        {
          d_decoder.decode(d_sigHtCodedLlr, d_sigHtBits, 48);
          if(signalCheckHt(d_sigHtBits))
          {
            signalParserHt(d_sigHtBits, &d_m, &d_sigHt);
            d_sDemod = DEMOD_S_HT;
            d_nSampConsumed += 160;
            d_nUsed += 160;
          }
          else
          {
            // go to legacy
            signalParserL(d_nSigLMcs, d_nSigLLen, &d_m);
            d_demodCu.cuDemodChanSiso((cuFloatComplex*) &d_HL[0]);
            d_nSampTotoal = d_m.nSymSamp * d_m.nSym;
            d_sDemod = DEMOD_S_DEMOD;
          }
        }
      }

      if(d_sDemod == DEMOD_S_VHT && ((d_nProc - d_nUsed) >= (80 + d_m.nLTF*80 + 80)))
      {
        // get channel and signal b
        nonLegacyChanEstimate(&inSig[d_nUsed + 80]);
        vhtSigBDemod(&inSig[d_nUsed + 80 + d_m.nLTF*80]);
        signalParserVhtB(d_sigVhtB20Bits, &d_m);
        int tmpNLegacySym = (d_nSigLLen*8 + 22 + 23)/24;
        if(d_m.len > 0 && d_m.len <= 4095 && d_m.nSS <= 2 && (tmpNLegacySym * 80) >= (d_m.nSym * d_m.nSymSamp + 160 + 80 + d_m.nLTF * 80 + 80))  // =0 for NDP
        {
          d_demodCu.cuDemodChanSiso((cuFloatComplex*) d_H_NL);
          d_nSampTotoal = d_m.nSymSamp * d_m.nSym;
          d_sDemod = DEMOD_S_DEMOD;
          d_nSampConsumed += (80 + d_m.nLTF*80 + 80);
          d_nUsed += (80 + d_m.nLTF*80 + 80);
        }
        else if(d_m.len == 0)
        {
          // report the channel and go to clean
          int tmpLen = sizeof(float)*256 + 3;
          d_psduBytes[0] = C8P_F_VHT_NDP;
          d_psduBytes[1] = tmpLen%256;  // byte 1-2 packet len
          d_psduBytes[2] = tmpLen/256;
          float* tmpFloatPointer = (float*)&d_psduBytes[3];
          for(int i=0;i<128;i++)
          {
            tmpFloatPointer[i*2] = d_mu2x1Chan[i].real();
            tmpFloatPointer[i*2+1] = d_mu2x1Chan[i].imag();
          }
          pmt::pmt_t tmpMeta = pmt::make_dict();
          tmpMeta = pmt::dict_add(tmpMeta, pmt::mp("len"), pmt::from_long(tmpLen));
          pmt::pmt_t tmpPayload = pmt::make_blob((uint8_t*)d_psduBytes, tmpLen);
          message_port_pub(pmt::mp("out"), pmt::cons(tmpMeta, tmpPayload));

          d_sDemod = DEMOD_S_CLEAN;
        }
        else
        {
          d_sDemod = DEMOD_S_CLEAN;
        }
      }

      if(d_sDemod == DEMOD_S_HT && ((d_nProc - d_nUsed) >= (80 + d_m.nLTF*80)))
      {
        nonLegacyChanEstimate(&inSig[d_nUsed + 80]);
        int tmpNLegacySym = (d_nSigLLen*8 + 22 + 23)/24;
        if(d_m.len > 0 && d_m.len <= 4095 && d_m.nSS <= 2 && (tmpNLegacySym * 80) >= (d_m.nSym * d_m.nSymSamp + 160 + 80 + d_m.nLTF * 80))
        {
          d_demodCu.cuDemodChanSiso((cuFloatComplex*) d_H_NL);
          d_nSampTotoal = d_m.nSymSamp * d_m.nSym;
          d_sDemod = DEMOD_S_DEMOD;
          d_nSampConsumed += (80 + d_m.nLTF*80);
          d_nUsed += (80 + d_m.nLTF*80);
        }
        else
        {
          d_sDemod = DEMOD_S_CLEAN;
        }
      }

      if(d_sDemod == DEMOD_S_DEMOD)
      {
        if((d_nProc - d_nUsed) >= (d_nSampTotoal - d_nSampCopied))
        {
          // copy and decode
          d_demodCu.cuDemodSigCopy(d_nSampCopied, (d_nSampTotoal - d_nSampCopied), (const cuFloatComplex*) &inSig[d_nUsed]);
          d_demodCu.cuDemodSiso(&d_m, d_psduBytes);
          d_nSampConsumed += (d_nSampTotoal - d_nSampCopied);
          d_nUsed += (d_nSampTotoal - d_nSampCopied);          
          packetAssemble();
          d_sDemod = DEMOD_S_CLEAN;
        }
        else
        {
          // copy
          d_demodCu.cuDemodSigCopy(d_nSampCopied, (d_nProc - d_nUsed), (const cuFloatComplex*) &inSig[d_nUsed]);
          d_nSampCopied += (d_nProc - d_nUsed);
          d_nSampConsumed += (d_nProc - d_nUsed);
          d_nUsed = d_nProc;
        }
      }

      if(d_sDemod == DEMOD_S_CLEAN)
      {
        if((d_nProc - d_nUsed) >= (d_nSigLSamp - d_nSampConsumed))
        {
          d_nUsed += (d_nSigLSamp - d_nSampConsumed);
          d_sDemod = DEMOD_S_RDTAG;
        }
        else
        {
          d_nSampConsumed += (d_nProc - d_nUsed);
          d_nUsed = d_nProc;
        }
      }

      consume_each (d_nUsed);
      return 0;
    }

    void
    demodcu_impl::packetAssemble()
    {
      if(d_m.format == C8P_F_VHT)
      {
        int tmpDeliBits[24];
        int tmpEof = 0, tmpLen = 0, tmpProcP = 0;
        while(true)
        {
          if((d_m.len - tmpProcP) < 4)
          {
            break;
          }
          // get info from delimiter
          for(int i=0;i<3;i++)
          {
            for(int j=0;j<8;j++)
            {
              tmpDeliBits[i*8+j] = (d_psduBytes[tmpProcP+i] >> j) & 1;
            }
          }
          tmpEof = tmpDeliBits[0];
          tmpLen |= (((int)tmpDeliBits[2])<<12);
          tmpLen |= (((int)tmpDeliBits[3])<<13);
          for(int i=0;i<12;i++)
          {
            tmpLen |= (((int)tmpDeliBits[4+i])<<i);
          }
          if(d_m.len < (tmpProcP + 4 + tmpLen))
          {
            break;
          }
          // write info into delimiter part
          d_crc32.reset();
          d_crc32.process_bytes(&d_psduBytes[tmpProcP+4], tmpLen);
          if (d_crc32.checksum() != 558161692) {
            if(d_debug)
            {
              std::string dbgStr("ieee80211 demodcu, vht crc32 wrong, total:");
              dbgStr += std::to_string(d_nPktCorrect);
              for(int i=0;i<10;i++)
              {
                dbgStr += (std::string(",") + std::to_string(i) + std::string(":") + std::to_string(d_vhtMcsCount[i]));
              }
              dbgStr += (std::string(",cfo:") + std::to_string(d_cfo));
              dbgStr += (std::string(",snr:") + std::to_string(d_snr));
              dbgStr += (std::string(",rssi:") + std::to_string(d_rssi));
              dbgStr += (std::string(",sssnr0:") + std::to_string(d_sssnr));
              dout << dbgStr << std::endl;
            }
            tmpProcP = tmpProcP + 4 + tmpLen;
          }
          else
          {
            if(d_debug)
            {
              d_nPktCorrect++;
              if(d_m.mcs >= 0 && d_m.mcs < 10)
              {
                d_vhtMcsCount[d_m.mcs]++;
              }
              std::string dbgStr("ieee80211 demodcu, vht crc32 correct, total:");
              dbgStr += std::to_string(d_nPktCorrect);
              for(int i=0;i<10;i++)
              {
                dbgStr += (std::string(",") + std::to_string(i) + std::string(":") + std::to_string(d_vhtMcsCount[i]));
              }
              dbgStr += (std::string(",cfo:") + std::to_string(d_cfo));
              dbgStr += (std::string(",snr:") + std::to_string(d_snr));
              dbgStr += (std::string(",rssi:") + std::to_string(d_rssi));
              dbgStr += (std::string(",sssnr0:") + std::to_string(d_sssnr));
              dout << dbgStr << std::endl;
            }
            // 1 byte packet format, 2 byte len
            d_psduBytes[tmpProcP+1] = d_m.format;    // byte 0 format
            d_psduBytes[tmpProcP+2] = tmpLen%256;  // byte 1-2 packet len
            d_psduBytes[tmpProcP+3] = tmpLen/256;
            pmt::pmt_t tmpMeta = pmt::make_dict();
            tmpMeta = pmt::dict_add(tmpMeta, pmt::mp("len"), pmt::from_long(tmpLen+3));
            pmt::pmt_t tmpPayload = pmt::make_blob(&d_psduBytes[tmpProcP + 1], tmpLen+3);
            message_port_pub(pmt::mp("out"), pmt::cons(tmpMeta, tmpPayload));
            tmpProcP = tmpProcP + 4 + tmpLen;
          }
          if(tmpEof)
          {
            break;
          }
        }
      }
      else
      {
        // a and n general packet
        if(d_m.ampdu)
        {
          // n ampdu, to be added
        }
        else
        { 
          d_crc32.reset();
          d_crc32.process_bytes(d_psduBytes, d_m.len);
          if (d_crc32.checksum() != 558161692) {
            if(d_debug)
            {
              std::string dbgStr;
              dbgStr = (d_m.format == C8P_F_L) ? std::string("ieee80211 demodcu, legacy crc32 wrong, total:") : std::string("ieee80211 decode, ht crc32 wrong, total:");
              dbgStr += std::to_string(d_nPktCorrect);
              for(int i=0;i<8;i++)
              {
                dbgStr += (std::string(",") + std::to_string(i) + std::string(":"));
                dbgStr += (d_m.format == C8P_F_L) ? std::to_string(d_legacyMcsCount[i]) : std::to_string(d_htMcsCount[i]);
              }
              dbgStr += (std::string(",cfo:") + std::to_string(d_cfo));
              dbgStr += (std::string(",snr:") + std::to_string(d_snr));
              dbgStr += (std::string(",rssi:") + std::to_string(d_rssi));
              dout << dbgStr << std::endl;
            }
          }
          else
          {
            if(d_debug)
            {
              d_nPktCorrect++;
              std::string dbgStr;
              if(d_m.format == C8P_F_L){
                dbgStr = std::string("ieee80211 demodcu, legacy crc32 correct, total:");
                d_legacyMcsCount[d_m.mcs]++;
              }
              else{
                dbgStr = std::string("ieee80211 demodcu, ht crc32 correct, total:");
                d_htMcsCount[d_m.mcs%8]++;
              }
              dbgStr += std::to_string(d_nPktCorrect);
              for(int i=0;i<8;i++)
              {
                dbgStr += (std::string(",") + std::to_string(i) + std::string(":"));
                dbgStr += (d_m.format == C8P_F_L) ? std::to_string(d_legacyMcsCount[i]) : std::to_string(d_htMcsCount[i]);
              }
              dbgStr += (std::string(",cfo:") + std::to_string(d_cfo));
              dbgStr += (std::string(",snr:") + std::to_string(d_snr));
              dbgStr += (std::string(",rssi:") + std::to_string(d_rssi));
              dout << dbgStr << std::endl;
            }

            pmt::pmt_t tmpMeta = pmt::make_dict();
            tmpMeta = pmt::dict_add(tmpMeta, pmt::mp("len"), pmt::from_long(d_m.len+3));
            pmt::pmt_t tmpPayload = pmt::make_blob(d_psduBytes, d_m.len+3);
            message_port_pub(pmt::mp("out"), pmt::cons(tmpMeta, tmpPayload));
          }
        }
      }
    }

    void
    demodcu_impl::fftDemod(const gr_complex* sig, gr_complex* res)
    {
      memcpy(d_ofdm_fft.get_inbuf(), sig, sizeof(gr_complex)*64);
      d_ofdm_fft.execute();
      memcpy(res, d_ofdm_fft.get_outbuf(), sizeof(gr_complex)*64);
    }

    void
    demodcu_impl::nonLegacyChanEstimate(const gr_complex* sig1)
    {
      if(d_m.format == C8P_F_VHT && d_m.sumu)
      {
        // mu-mimo 2x2
        fftDemod(&sig1[8], d_fftLtfOut1);
        fftDemod(&sig1[8+80], d_fftLtfOut2);
        for(int i=0;i<64;i++)
        {
          if(i==0 || (i>=29 && i<=35))
          {}
          else
          {
            if(d_muPos == 0)
            {
              // ss0 LTF and LTF_N
              d_H_NL[i] = (d_fftLtfOut1[i] - d_fftLtfOut2[i]) / LTF_NL_28_F_FLOAT[i] / 2.0f;
            }
            else
            {
              // ss1 LTF and LTF
              //d_H_NL[i] = (d_fftLtfOut1[i] + d_fftLtfOut2[i]) / LTF_NL_28_F_FLOAT[i] / 2.0f;
              d_H_NL[i] = d_fftLtfOut1[i] / LTF_NL_28_F_FLOAT[i];
            }
          }
        }
      }
      else if(d_m.nSS == 1 && d_m.nLTF == 1)
      {
        fftDemod(&sig1[8], d_fftLtfOut1);
        for(int i=0;i<64;i++)
        {
          if(i==0 || (i>=29 && i<=35))
          {}
          else
          {
            d_H_NL[i] = d_fftLtfOut1[i] / LTF_NL_28_F_FLOAT[i];
          }
        }
      }
      else
      {
        // nSS larger than 1, only support 2x2 channel sounding now
        memcpy(&d_mu2x1Chan[0], &sig1[8], sizeof(gr_complex) * 64);
        memcpy(&d_mu2x1Chan[64], &sig1[8+80], sizeof(gr_complex) * 64);
        fftDemod(&sig1[8], d_fftLtfOut1);
        for(int i=0;i<64;i++)
        {
          if(i==0 || (i>=29 && i<=35))
          {
          }
          else
          {
            d_H_NL[i] = d_fftLtfOut1[i] / LTF_NL_28_F_FLOAT[i];
          }
        }
      }
    }

    void
    demodcu_impl::vhtSigBDemod(const gr_complex* sig1)
    {
      fftDemod(&sig1[8], d_fftLtfOut1);
      for(int i=0;i<64;i++)
      {
        if(i==0 || (i>=29 && i<=35))
        {}
        else
        {
          d_sig1[i] = d_fftLtfOut1[i] / d_H_NL[i];
        }
      }
      gr_complex tmpPilotSum = std::conj(d_sig1[7] - d_sig1[21] + d_sig1[43] + d_sig1[57]);
      float tmpPilotSumAbs = std::abs(tmpPilotSum);
      
      int j=26;
      for(int i=0;i<64;i++)
      {
        if(i==0 || (i>=29 && i<=35) || i==7 || i==21 || i==43 || i==57)
        {}
        else
        {
          d_sigVhtBQam[j] = d_sig1[i] * tmpPilotSum / tmpPilotSumAbs;
          d_sigVhtB20IntedLlr[j] = d_sigVhtBQam[j].real();
          j++;
          if(j >= 52){j = 0;}
        }
      }
      
      for(int i=0;i<52;i++)
      {
        d_sigVhtB20CodedLlr[mapDeintVhtSigB20[i]] = d_sigVhtB20IntedLlr[i];
      }
      d_decoder.decode(d_sigVhtB20CodedLlr, d_sigVhtB20Bits, 26);

      bccEncoder(d_sigVhtB20Bits, d_sigVhtB20BitsCoded, 26);
      procIntelVhtB20(d_sigVhtB20BitsCoded, d_sigVhtB20BitsInted);
      double tmpNoisePower = 0.0;
      for(int i=0;i<52;i++)
      {
        if(d_sigVhtB20BitsInted[i])
        {
          d_sigVhtBQam[i] -= gr_complex(1.0f, 0.0f);
        }
        else
        {
          d_sigVhtBQam[i] -= gr_complex(-1.0f, 0.0f);
        }
        tmpNoisePower += (double)(d_sigVhtBQam[i].real()*d_sigVhtBQam[i].real() + d_sigVhtBQam[i].imag() * d_sigVhtBQam[i].imag());
      }
      d_sssnr = (float)(log10(52.0/tmpNoisePower) * 10.0);
    }

  } /* namespace ieee80211cu */
} /* namespace gr */
