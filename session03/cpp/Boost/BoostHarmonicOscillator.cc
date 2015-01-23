#include <boost/numeric/odeint.hpp>
#include <vector>

const double gam = 0.15;
typedef std::vector< double > state_type;

// This is functor class
class harm_osc {
    double m_gam; // class can have member variables, state etc.
public:
    harm_osc( double gam ) : m_gam(gam) { }

    // odeint integrators normally call f(x, dxdt, t)
    void operator() ( const state_type &x , state_type &dxdt , const double /* t */ )
    {
        dxdt[0] = x[1];
        dxdt[1] = -x[0] - m_gam*x[1];
    }
};

// This is observer to record output, and is also a functor class
struct push_back_state_and_time
{
    std::vector< state_type >& m_states;
    std::vector< double >& m_times;

    push_back_state_and_time( std::vector< state_type > &states , std::vector< double > &times )
    : m_states( states ) , m_times( times ) { }

    void operator()( const state_type &x , double t )
    {
        m_states.push_back( x );
        m_times.push_back( t );
    }
};

int main(void) {

  state_type x(2);
  x[0] = 1.0; // start at x=1.0, p=0.0
  x[1] = 0.0;

  std::vector<state_type> x_vec; // vector of vectors
  std::vector<double> times;     // stores each time point

  harm_osc harmonic_oscillator(0.15);
  size_t steps = boost::numeric::odeint::integrate(
    harmonic_oscillator ,
    x , 0.0 , 10.0 , 0.1 ,
    push_back_state_and_time( x_vec , times ) );

  for( size_t i=0; i<=steps; i++ )
  {
    std::cout << times[i] << '\t' << x_vec[i][0] << '\t' << x_vec[i][1] << '\n';
  }

}