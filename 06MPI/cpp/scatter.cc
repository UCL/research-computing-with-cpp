#include <mpi.h>
#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

TEST_CASE("Collective communications") {

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    SECTION("Collective scatter") {
      /// "scatter"
      std::string const message = "This message is going to come out in separate channels";
      if (message.size() < (unsigned)size) return;
      int N = message.size() / size;

      char buffer[256];
      int const error = MPI_Scatter(
                (void*) message.c_str(), N, MPI_CHAR, // Only significant at root
                buffer, 256, MPI_CHAR, // Receiver side
                0, MPI_COMM_WORLD
      );
      REQUIRE(error == MPI_SUCCESS);
      CHECK(message.substr(rank*N, N) == std::string(buffer, N));
      /// "dummy"
    }
}

int main(int argc, char * argv[]) {
    MPI_Init(&argc, &argv);
    int result = Catch::Session().run(argc, argv);
    MPI_Finalize();
    return result;
}
