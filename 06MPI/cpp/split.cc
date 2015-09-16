#include <mpi.h>
#include <iostream>
#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

TEST_CASE("Communicators can be split", "splitting") {

    int rank, size;
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);


    /// "main"
    bool const is_apple = rank % 3 == 0;
    SECTION("split 1:3 and keep same process order") {
        MPI_Comm apple_orange;
        MPI_Comm_split(MPI_COMM_WORLD, is_apple ? 0: 1, rank, &apple_orange);

        int nrank, nsize;
        MPI_Comm_rank(apple_orange, &nrank);
        MPI_Comm_size(apple_orange, &nsize);

        int const div = (size - 1) / 3, napples = 1 + div;
        if(is_apple) {
           CHECK(nsize == napples);
           CHECK(nrank == rank / 3);
        } else {
           CHECK(nsize == size - napples);
           CHECK(nrank == rank - 1 - (rank / 3));
        }
    }
    /// "dummy"

    SECTION("split 1:3 and reverse process order") {
        MPI_Comm apple_orange;
        MPI_Comm_split(MPI_COMM_WORLD, is_apple ? 0: 1, -rank, &apple_orange);

        int nrank, nsize;
        MPI_Comm_rank(apple_orange, &nrank);
        MPI_Comm_size(apple_orange, &nsize);

        int const div = (size - 1) / 3;
        int const napples = 1 + div;
        if(is_apple) {
           CHECK(nsize == napples);
           CHECK(nrank == napples - 1 - rank / 3);
        } else {
           CHECK(nsize == size - napples);
           CHECK(nrank == size - napples - 1 - (rank - 1 - (rank / 3)));
        }
    }

    SECTION("split 1:1:2 and keep process order") {
        MPI_Comm new_comm;
        int const nsubset = size / 4;
        int const color = rank < nsubset ? 0: (rank < nsubset * 2 ? 1: 2);
        MPI_Comm_split(MPI_COMM_WORLD, color, rank, &new_comm);

        int nrank, nsize;
        MPI_Comm_rank(new_comm, &nrank);
        MPI_Comm_size(new_comm, &nsize);

        if(color == 0) {
            CHECK(nsize == nsubset);
            CHECK(nrank == rank);
        } else if (color == 1) {
            CHECK(nsize == nsubset);
            CHECK(nrank == rank - nsubset);
        } else {
            CHECK(nsize == size - 2 * nsubset);
            CHECK(nrank == rank - 2 * nsubset);
        }
    }
}

int main(int argc, char * argv[]) {
    MPI_Init (&argc, &argv);
    int result = Catch::Session().run(argc, argv);
    MPI_Finalize();
    return 0;
}
