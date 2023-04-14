/*
 *
 *     GNU Radio IEEE 802.11a/g/n/ac 20M bw and upto 2x2
 *     PHY utilization functions and parameters CUDA Version
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

#ifndef INCLUDED_CLOUD80211PHYCU_H
#define INCLUDED_CLOUD80211PHYCU_H

#include <iostream>
#include <math.h>
#include <cuComplex.h>
#include <cufft.h>
#include "cloud80211phy.h"

#define PREPROC_MIN 1024
#define PREPROC_MAX 8192

#define CUDEMOD_B_MAX 4095     // max PSDU byte len
#define CUDEMOD_T_MAX 32782    // max trellis len, psdu * 8 + 22
#define CUDEMOD_L_MAX 65728    // max llr len, VHT CR1/2
#define CUDEMOD_S_MAX 1408     // max symbol number, legacy nDBPS 24 is 1366, round to multiple of fft batch 1408
#define CUDEMOD_FFT_BATCH 64    // each execution 64 symbols

#define CUDEMOD_VTB_LEN 80


class cloud80211preproccu
{
  private:
    cuFloatComplex* ppSig;
    cuFloatComplex* ppSigConj;
    cuFloatComplex* ppSigConjAvg;
    float* ppSigConjAvgMag;
    float* ppSigMag2;
    float* ppSigMag2Avg;
    float* ppOut;
    void cuMall();
    void cuFree();

  public:
    cloud80211preproccu();
    ~cloud80211preproccu();
    void preProc(int n, const cuFloatComplex *sig, float* ac, cuFloatComplex* conj);
};

class cloud80211demodcu
{
  private:
    cuFloatComplex* demodChanSiso;
    cuFloatComplex* demodChanMimo;
    cuFloatComplex* demodChanMimoInv;
    cuFloatComplex* demodSig;
    cuFloatComplex* demodSigFft;
    cufftHandle demodPlan;
    cuFloatComplex* pilotsLegacy;
    cuFloatComplex* pilotsHt;
    cuFloatComplex* pilotsHt2;
    cuFloatComplex* pilotsVht;
    cuFloatComplex* pilotsVht2;
    cuFloatComplex* pilotsNlLtf2;
    float* demodSigLlr;

    int* demodDemapFftL;
    int* demodDemapBpskL;
    int* demodDemapQpskL;
    int* demodDemap16QamL;
    int* demodDemap64QamL;
    int* demodDemapL[6] = {NULL, NULL, NULL, NULL, NULL, NULL};

    int* demodDemapFftNL;
    int* demodDemapBpskNL;
    int* demodDemapQpskNL;
    int* demodDemap16QamNL;
    int* demodDemap64QamNL;
    int* demodDemap256QamNL;
    int* demodDemapNL[6] = {NULL, NULL, NULL, NULL, NULL, NULL};

    int* demodDemapBpskNL2;
    int* demodDemapQpskNL2;
    int* demodDemap16QamNL2;
    int* demodDemap64QamNL2;
    int* demodDemap256QamNL2;
    int* demodDemapNL2[6] = {NULL, NULL, NULL, NULL, NULL, NULL};

    int* cuv_seq;
    int* cuv_seqtb;
    int* cuv_state_his;
    int* cuv_state_next;
    int* cuv_state_output;
    int* cuv_cr_punc;
    unsigned char* cuv_bits;
    unsigned char* cuv_descram_seq;
    unsigned char* cuv_bytes;
    void cuDemodMall();
    void cuDemodFree();
  
  public:
    cloud80211demodcu();
    ~cloud80211demodcu();
    void cuDemodChanSiso(cuFloatComplex *chan);
    void cuDemodChanMimo(cuFloatComplex *chan, cuFloatComplex *chaninv, cuFloatComplex *pilotsltf);
    void cuDemodSigCopy(int i, int n, const cuFloatComplex *sig);
    void cuDemodSigCopy2(int i, int j, int n, const cuFloatComplex *sig, const cuFloatComplex *sig2);
    void cuDemodSiso(c8p_mod* m, unsigned char* psduBytes);
    void cuDemodMimo(c8p_mod* m, unsigned char* psduBytes);
};

class cloud80211modcu
{
  private:
    float scaleFactorL;
    float scaleFactorNL;
    bool initSuccess;
    int scrambler;
    // constants
    unsigned char *scramSeq;
    int *interLutL;
    int *interLutLIdx[6];
    int *interLutNL;
    int *interLutNLIdx[6];
    int *interLutNL2;
    int *interLutNL2Idx[6];
    cuFloatComplex *qamLut;
    cuFloatComplex *qamLutIdx[6];
    int *qamScMapL;
    int *qamScMapNL;
    cuFloatComplex *pilotsL;
    cuFloatComplex *pilotsHT;
    cuFloatComplex *pilotsHT2;
    cuFloatComplex *pilotsVHT;
    cuFloatComplex *pilotsVHT2;
    cuFloatComplex *symCsdNL2;
    cufftHandle ifftModPlan;
    cudaStream_t modStream0;
    cudaStream_t modStream1;
    // packet
    unsigned char *pktBytes;
    unsigned char *pktBits;
    unsigned char *pktBitsCoded;
    unsigned char *pktBitsPuncd;
    unsigned char *pktBitsInted;
    unsigned char *pktBitsStream;
    cuFloatComplex *pktSymFreq;
    cuFloatComplex *pktSymTime;
    cuFloatComplex *pktSym;

  void cuModMall();
  void cuModFree();

  public:
    void cuModPktCopySu(int i, int n, const unsigned char *bytes);
    void cuModLHTSiso(c8p_mod *m, cuFloatComplex *sig);
    void cuModHTMimo(c8p_mod *m, cuFloatComplex *sig0, cuFloatComplex *sig1);
    void cuModVHTSiso(c8p_mod *m, cuFloatComplex *sig, unsigned char *vhtSigBCrc8Bits);
    void cuModVHTSuMimo(c8p_mod *m, cuFloatComplex *sig0, cuFloatComplex *sig1, unsigned char *vhtSigBCrc8Bits);
    void cuModVHTMuMimo(c8p_mod *m, cuFloatComplex *sig0, cuFloatComplex *sig1, unsigned char *vhtSigB0Crc8Bits, unsigned char *vhtSigB1Crc8Bits);
    cloud80211modcu();
    ~cloud80211modcu();
};

#endif /* INCLUDED_IEEE80211_SIGNAL_IMPL_H */