#include "wavelets.h"
#include <cassert>
#include <numeric>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace wavelets {

DaubechyData const daubechy1{
  { 7.071067811865475244008443621048490392848359376884740365883398e-01,
      7.071067811865475244008443621048490392848359376884740365883398e-01 },
  { -7.071067811865475244008443621048490392848359376884740365883398e-01,
      7.071067811865475244008443621048490392848359376884740365883398e-01 }
};

DaubechyData const daubechy2{
  { 4.829629131445341433748715998644486838169524195042022752011715e-01,
      8.365163037378079055752937809168732034593703883484392934953414e-01,
      2.241438680420133810259727622404003554678835181842717613871683e-01,
      -1.294095225512603811744494188120241641745344506599652569070016e-01 },
  { 1.294095225512603811744494188120241641745344506599652569070016e-01,
      2.241438680420133810259727622404003554678835181842717613871683e-01,
      -8.365163037378079055752937809168732034593703883484392934953414e-01,
      4.829629131445341433748715998644486838169524195042022752011715e-01 }
};

Scalar apply_cyclical_filter(
    Signal const& signal, Signal::size_type location, Signal const& filter)
{
  return apply_cyclical_filter(
      signal.begin(), signal.begin() + location, signal.end(), filter);
}

Signal single_direct_transform(Signal const& signal, DaubechyData const& wavelet)
{
  Signal coefficients(number_of_coefficients(signal.size(), 1));
  single_direct_transform(signal.begin(), signal.end(), coefficients.begin(), wavelet);
  return coefficients;
}

Signal direct_transform(
    Signal const& signal, DaubechyData const& wavelet, unsigned int levels)
{
  // shifting the bit left is the same as multiplying by two
  Signal coefficients(number_of_coefficients(signal.size(), levels));
  direct_transform(signal.begin(), signal.end(), coefficients.begin(), wavelet, levels);
  return coefficients;
}

unsigned int number_of_coefficients(unsigned int signal_size, unsigned int levels)
{
  if (levels == 0)
    return signal_size;
  auto const even_size = signal_size + signal_size % 2;
  if (levels == 1)
    return even_size;
  return even_size / 2 + number_of_coefficients(even_size / 2, levels - 1);
}

Scalar apply_cyclical_filter(
    Signal::const_iterator const& start,
    Signal::const_iterator location,
    Signal::const_iterator const& end,
    Signal const& filter)
{
  if (start == end)
    return 0;

  assert(location >= start);
  assert(location < end);

  Signal::value_type result(0);
  auto i_filter = filter.begin();
  while (i_filter != filter.end()) {
    for (; i_filter != filter.end() and location != end; ++i_filter, ++location)
      result += (*i_filter) * (*location);
    location = start;
  }
  return result;
}

void single_direct_transform(
    Signal::const_iterator const& start, Signal::const_iterator const& end,
    Signal::iterator const &out, DaubechyData const& wavelet)
{
  assert(start <= end);
  int const half = ((end - start) + (end - start) % 2) / 2;
#pragma omp parallel
  {
#pragma omp for
    for (int i=0; i < half; ++i)
      *(out + i) = apply_cyclical_filter(start, start + 2 * i, end, wavelet.high_pass);
#pragma omp for
    for (int i=0; i < half; ++i)
      *(out + i + half) = apply_cyclical_filter(start, start + 2 * i, end, wavelet.low_pass);
  }
}

void direct_transform(
    Signal::const_iterator start, Signal::const_iterator end,
    Signal::iterator out, DaubechyData const& wavelet,
    unsigned int levels)
{
  if (start == end)
    return

        assert(start <= end);
  if (levels == 0) {
    std::copy(start, end, out);
    return;
  }

  single_direct_transform(start, end, out, wavelet); // first iteration

  auto const half = [](Signal::size_type n) { return (n + n % 2) / 2; };
  auto approx_size = half(end - start);
  Signal work_array(approx_size);
  auto const i_work = work_array.begin();
  for (unsigned int i(1); i < levels; ++i, approx_size = half(approx_size)) {
    out += approx_size;
    std::copy(out, out + approx_size, i_work);
    single_direct_transform(i_work, i_work + approx_size, out, wavelet);
  }
}
}
