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

void preprocMall();
void preprocFree();
void cuPreProc(int n, const cuFloatComplex *sig, float* ac, cuFloatComplex* conj);

void cuDemodMall();
void cuDemodMall2();
void cuDemodFree();
void cuDemodFree2();
void cuDemodChanSiso(cuFloatComplex *chan);
void cuDemodChanMimo(cuFloatComplex *chan, cuFloatComplex *chaninv, cuFloatComplex *pilotsltf);
void cuDemodSigCopy(int i, int n, const cuFloatComplex *sig);
void cuDemodSigCopy2(int i, int j, int n, const cuFloatComplex *sig, const cuFloatComplex *sig2);
void cuDemodSiso(c8p_mod* m, unsigned char* psduBytes);
void cuDemodMimo(c8p_mod* m, unsigned char* psduBytes);

class cloud80211modcu
{
  private:
    bool initSuccess;
    int scrambler;
    // constants
    unsigned char *scramSeq;
    int *interBPSKL;
    int *interQPSKL;
    int *inter16QAML;
    int *inter64QAML;
    int *interSeqL[6];
    int *interBPSKNL;
    int *interQPSKNL;
    int *inter16QAMNL;
    int *inter64QAMNL;
    int *inter256QAMNL;
    int *interSeqNL[6];
    cuFloatComplex *modBpsk;
    cuFloatComplex *modQpsk;
    cuFloatComplex *mod16Qam;
    cuFloatComplex *mod64Qam;
    cuFloatComplex *mod256Qam;
    cuFloatComplex *modMap[6];
    // input
    unsigned char *pktBytes;
    unsigned char *pktBits;
    unsigned char *pktBitsCoded;
    unsigned char *pktBitsPuncd;
    unsigned char *pktBitsInted;

  void cuModMall();
  void cuModFree();

  public:
    void cuModPktCopySu(int i, int n, const unsigned char *bytes);
    void cuModSu(c8p_mod *m, cuFloatComplex *sig, unsigned char *vhtSigBCrc8Bits);
    cloud80211modcu();
    ~cloud80211modcu();
};

#endif /* INCLUDED_IEEE80211_SIGNAL_IMPL_H */