/*
 *
 *     GNU Radio IEEE 802.11a/g/n/ac 2x2, for SISO
 *     Legacy training field pre processing cuda
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

#ifndef INCLUDED_IEEE80211CU_PREPROCCU_IMPL_H
#define INCLUDED_IEEE80211CU_PREPROCCU_IMPL_H

#include <gnuradio/ieee80211cu/preproccu.h>
#include "cloud80211phycu.cuh"

namespace gr {
  namespace ieee80211cu {

    class preproccu_impl : public preproccu
    {
     private:
      // Nothing to declare in this block.

     public:
      preproccu_impl();
      ~preproccu_impl();

      // Where all the action really happens
      void forecast (int noutput_items, gr_vector_int &ninput_items_required);

      int general_work(int noutput_items,
           gr_vector_int &ninput_items,
           gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);

    };

  } // namespace ieee80211cu
} // namespace gr

#endif /* INCLUDED_IEEE80211CU_PREPROCCU_IMPL_H */
