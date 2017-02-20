#ifndef CPP_COURSE_WAVELETS_H

#include <cassert>
#include <vector>
#include <iostream>

namespace wavelets {

//! \brief Underlying type of all real numbers in the wavelets
//! \details Defining a "type hierarchy" makes it easy to modify all the basic
//! types underlying a coding project
typedef double Scalar;

//! \brief Represents 1-D signal of any size
typedef std::vector<Scalar> Signal;

//! \brief High and low-pass filter data
//! \details
//! Note
//! ----
//!
//! This structure declares it's attribute public. And later we declare global
//! variables as instances of this type. Both of these aspects are frowned
//! upon, *except* in this one case. The wavelet filters are constants for a
//! given Daubechy wavelet. For constants of nature and constants of math,
//! whether scalar or vectorial (and small), it's okay to use global variable
//! and public attributes. Otherwise, beware!
//!
//! Premature Optimization?
//! -----------------------
//!
//! We could have `std::array<Scalar, N> high_pass` and the same for
//! `low_pass`. However, `N`, the number of coeffients, depends on the actual
//! wavelet. And in `std::array<Scalar, N>`, `N` needs to be known at compile
//! time. We could make this change and the compiler might be able to make use
//! of this extra information to produce faster code. However, it would mean
//! that `DaubechyData` needs to be declared as `template<int N> struct
//! DaubechyData`, and all the functions taking `DaubechyData<N>` as input
//! would also need to be templates.
//!
//! Templating is cool, fun, but viral. So we might as well use the simpler
//! implementation and wait for benchmarks and profiling results to tell us
//! whether we really need the extra complexity.
struct DaubechyData {
  std::vector<Scalar> low_pass;
  std::vector<Scalar> high_pass;
};

//! \brief Data for the Daubechy wavelets of type 1
//! \details This data does not change from code to code. In fact, I got them
//! from wikipedia. Constants of nature and constants of math are the only
//! variable that should be declared global (and `const`).
extern DaubechyData const daubechy1;
//! Data for the Daubechy wavelets of type 2
extern DaubechyData const daubechy2;

//! \brief Applies a filter starting from a given location of the signal
//! \details The signal is defined by the range `start`, and `end`. `location`
//! must be a position inside that range. The filter is applied starting from
//! that location of the signal.
Scalar apply_cyclical_filter(
    Signal::const_iterator const& start,
    Signal::const_iterator location,
    Signal::const_iterator const& end,
    Signal const& filter);

//! \brief Applies a filter starting from a given location of the signal
//! \details Accumulates the result of `signal[location + i] * filter[i]` for i
//! in `[0, filter.size()[`. If `location + i` goes out of range, then it
//! cycles back to the begining of the signal. In practice, this means the
//! signal is periodic.
//!
//! This is a thin wrapper around the function that does the work. Its purpose
//! is to provide a user-friendly interface for end users and for testing.
Scalar apply_cyclical_filter(
    Signal const& signal, Signal::size_type location, Signal const& filter);

//! \brief Applies the wavelet transform once to the signal
//! \details The output iterator should point to a valid range of the same size
//! as the input signal *if the size of the signal is even*, and one larger than the signal
//! *if the size of the signal is odd*. The result is undefined when the signal
//! and output arrays overlap.
void single_direct_transform(
    Signal::const_iterator const& start, Signal::const_iterator const& end,
    Signal::iterator const &out, DaubechyData const& wavelet);

//! \brief Applies high and low pass once to the signal range
//! \details This wrapper is also to make testing somewhat simpler.
Signal single_direct_transform(Signal const& signal, DaubechyData const& wavelet);

//! \brief Applies the wavelet transform `levels` time to the signal
//! \details The number of coefficients is given by the function
//! `number_of_coefficients`.
void direct_transform(
    Signal::const_iterator start, Signal::const_iterator end,
    Signal::iterator out, DaubechyData const& wavelet,
    unsigned int levels = 1);

//! \brief Figures out number of coeffs
unsigned int number_of_coefficients(unsigned int signal_size, unsigned int levels);

//! \brief Transforms the input signal
Signal direct_transform(
    Signal const& signal, DaubechyData const& wavelet, unsigned int levels = 1);
}
#endif
