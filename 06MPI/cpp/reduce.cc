#include <mpi.h>
#define CATCH_CONFIG_RUNNER
#include "catch.hpp"
#include <cmath>
#include <iostream>

TEST_CASE("Collective communications reduce") {

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    SECTION("Collective reduce") {
      /// "independent calculation"
      const int TERMS_PER_PROCESS = 1000;
      double my_denominator = rank * TERMS_PER_PROCESS * 2 + 1;
      double sign = 1;
      double my_result = 0.0;
      for (int i(0); i<TERMS_PER_PROCESS; ++i)
      {
        my_result += sign / my_denominator;
        my_denominator += 2;
        sign = -sign;
      }
      /// "reduce"
      double result;
      int const error = MPI_Reduce(
        &my_result, &result, 1 /* size */, MPI_DOUBLE,
        MPI_SUM /* op */, 0 /* root */, MPI_COMM_WORLD);
      REQUIRE(error == MPI_SUCCESS);
      if (rank == 0) {
        double abs_err = fabs(result * 4.0 - M_PI);
        CHECK(abs_err < 0.001);
        std::cout << "With " << size << " processes, error = "
                  << abs_err << std::endl;
      }
      /// "end reduce"
    }
}

int main(int argc, char * argv[]) {
    MPI_Init(&argc, &argv);
    int result = Catch::Session().run(argc, argv);
    MPI_Finalize();
    return result;
}
