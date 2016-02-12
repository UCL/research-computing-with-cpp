#include <mpi.h>
#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

TEST_CASE("Point to point communications") {

    int rank, size;
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);

    SECTION("Blocking synchronous send") {
      /// "ssend"
      std::string const peace = "I come in peace!";
      if(rank == 0) {
         int const error = MPI_Ssend(
           (void*) peace.c_str(), peace.size() + 1, MPI_CHAR, 1, 42, MPI_COMM_WORLD);
         // Here, we guarantee that Rank 1 has received the message.
         REQUIRE(error ==  MPI_SUCCESS);
      }
      if(rank == 1) {
          char buffer[256];
          int const error = MPI_Recv(
            buffer, 256, MPI_CHAR, 0, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          REQUIRE(error ==  MPI_SUCCESS);
          CHECK(std::string(buffer) == peace);
      }
      /// "stub"
    }

    SECTION("Blocking send") {
      /// "send"
      std::string peace = "I come in peace!";
      if(rank == 0) {
         int const error = MPI_Send(
           (void*) peace.c_str(), peace.size() + 1, MPI_CHAR, 1, 42, MPI_COMM_WORLD);
         // We do not guarantee that Rank 1 has received the message yet
         // But nor do we necessarily know it hasn't.
         // But we are definitely allowed to change the string, as MPI promises
         // it has been buffered
         peace = "Shoot to kill!"; // Safe to reuse the send buffer.
         REQUIRE(error ==  MPI_SUCCESS);
      }
      if(rank == 1) {
          char buffer[256];
          int const error = MPI_Recv(
            buffer, 256, MPI_CHAR, 0, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          REQUIRE(error ==  MPI_SUCCESS);
          CHECK(std::string(buffer) == peace);
      }
      /// "stub2"
    }

      SECTION("Nonblocking send") {
        /// "isend"
        std::string peace = "I come in peace!";
        if(rank == 0) {
          MPI_Request request;
           int error = MPI_Isend(
             (void*) peace.c_str(), peace.size() + 1, MPI_CHAR, 1, 42,
             MPI_COMM_WORLD, &request);
           // We do not guarantee that Rank 1 has received the message yet
           // We can carry on, and ANY WORK WE DO NOW WILL OVERLAP WITH THE
           // COMMUNICATION
           // BUT, we can't safely change the string.
           REQUIRE(error ==  MPI_SUCCESS);
           // Do some expensive work here
           for (int i=0; i<1000; i++) {}; // BUSYNESS FOR EXAMPLE
           MPI_Status status;
           error = MPI_Wait(&request, &status);
           REQUIRE(error ==  MPI_SUCCESS);
           // Here, we run code that requires the message to have been
           // successfully sent.
        }
        if(rank == 1) {
            char buffer[256];
            int const error = MPI_Recv(
              buffer, 256, MPI_CHAR, 0, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            REQUIRE(error ==  MPI_SUCCESS);
            CHECK(std::string(buffer) == peace);
        }
        /// "Stub3"
    }



}


/// "main"
int main(int argc, char * argv[]) {
    MPI_Init (&argc, &argv);
    int result = Catch::Session().run(argc, argv);
    MPI_Finalize();
    return result;
}
