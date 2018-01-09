#include <mpi.h>
#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

TEST_CASE("Ring communications") {

    int rank, size;
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);

    REQUIRE(size >= 4);

    /// "Setup"
    int message = rank*rank;
    int received = -7;

    // Define the ring
    int left = rank-1;
    int right = rank+1;
    if (rank==0) {
      left = size-1;
    }
    if (rank == size-1){
      right = 0;
    }
    /// "End setup"

    SECTION("Blocking synchronous") {
      /// "ssend"

      if (rank%2 == 0) {
        int error = MPI_Ssend(
          &message, 1, MPI_INT, left, rank, MPI_COMM_WORLD);
        REQUIRE(error == MPI_SUCCESS);

        error = MPI_Recv(
          &received, 1, MPI_INT, right, right, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        REQUIRE(error == MPI_SUCCESS);
      }
      if (rank%2 == 1) {

        int error = MPI_Recv(
          &received, 1, MPI_INT, right, right, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        REQUIRE(error == MPI_SUCCESS);

        error = MPI_Ssend(
          &message, 1, MPI_INT, left, rank, MPI_COMM_WORLD);
        REQUIRE(error == MPI_SUCCESS);
      }
      REQUIRE( received == right*right );
      /// "Stub"
    }

    SECTION("Asynchronous") {
      /// "Isend"
      MPI_Request request;
      // Everyone sets up their messages to send
      int error = MPI_Isend(
        &message, 1, MPI_INT, left, rank, MPI_COMM_WORLD, &request);
      REQUIRE(error == MPI_SUCCESS);

      // Recv acts as our sync-barrier
      error = MPI_Recv(
        &received, 1, MPI_INT, right, right, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      REQUIRE(error == MPI_SUCCESS);

      // But let's check our send completed:
      error = MPI_Wait(&request, MPI_STATUS_IGNORE);
      REQUIRE(error ==  MPI_SUCCESS);

      REQUIRE( received == right*right );
      /// "Stub2"
    }

      SECTION("Sendreceive") {
        /// "sendrecv"
        int error = MPI_Sendrecv(
          &message, 1, MPI_INT, left, rank,
          &received, 1, MPI_INT, right, right,
          MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        REQUIRE(error ==  MPI_SUCCESS);

        REQUIRE( received == right*right );
        /// "Stub"
    }



}


/// "main"
int main(int argc, char * argv[]) {
    MPI_Init (&argc, &argv);
    int result = Catch::Session().run(argc, argv);
    MPI_Finalize();
    return result;
}
