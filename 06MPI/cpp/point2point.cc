#include <mpi.h>
#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

TEST_CASE("Point to point communications") {

    int rank, size;
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);

    SECTION("Blocking send") {
      /// "send"
      std::string const peace = "I come in peace!";
      if(rank == 0) {
         int const error = MPI_Ssend(
           (void*) peace.c_str(), peace.size() + 1, MPI_CHAR, 1, 42, MPI_COMM_WORLD);
         REQUIRE(error ==  MPI_SUCCESS);
      }
      if(rank == 1) {
          char buffer[256];
          int const error = MPI_Recv(
            buffer, 256, MPI_CHAR, 0, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          REQUIRE(error ==  MPI_SUCCESS);
          CHECK(std::string(buffer) == peace);
      }
      /// "dummy"
    }
}


/// "main"
int main(int argc, char * argv[]) {
    MPI_Init (&argc, &argv);
    int result = Catch::Session().run(argc, argv);
    MPI_Finalize();
    return result;
}
