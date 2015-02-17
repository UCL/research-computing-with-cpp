#include <mpi.h>
#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

TEST_CASE("Collective communications") {

    int rank, size;
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);

    SECTION("Collective broadcast") {
      /// "broadcast"
      std::string const peace = "I come in peace!";
      std::string message = "";
      int error;
      if(rank == 0) {
          message = peace;
          error = MPI_Bcast(
             (void*) peace.c_str(), peace.size() + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
      } else {
          char buffer[256];
          int const error = MPI_Bcast(buffer, 256, MPI_CHAR, 0, MPI_COMM_WORLD);
          message = std::string(buffer);
      }

      /// "tests"
      for(int i(0); i < size; ++i) {
          if(rank == i) {
            INFO("Current rank is " << rank);
            REQUIRE(error == MPI_SUCCESS);
            CHECK(message == peace);
          }
          MPI_Barrier(MPI_COMM_WORLD);
      }
      /// "dummy"
    }
}

int main(int argc, char * argv[]) {
    MPI_Init (&argc, &argv);
    int result = Catch::Session().run(argc, argv);
    MPI_Finalize();
    return result;
}
