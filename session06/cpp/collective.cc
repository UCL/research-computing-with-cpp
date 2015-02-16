#include <mpi.h>
#include <iostream>

/// "functions"
void root_broadcasts(std::string const message) {
    int const error = MPI_Bcast(
            (void*) message.c_str(), message.size(), MPI_CHAR, 0,
            MPI_COMM_WORLD
    );
    if(error != MPI_SUCCESS) throw;
}

std::string others_get_it() {
    char buffer[256];
    int const error = MPI_Bcast(buffer, 256, MPI_CHAR, 0, MPI_COMM_WORLD);
    if(error != MPI_SUCCESS) throw;
    return buffer;
}

/// "dummy"
int main(int argc, char * argv[]) {
    MPI_Init (&argc, &argv);

    int rank, size;
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);

    /// "main"
    std::string message = "";
    if(rank == 0) root_broadcasts("Listen up folks.");
    else message = others_get_it();

    for(int i(1); i < size; ++i) {
        if(rank == i)
            std::cout << "I am " << rank
                << ", and I approve this message: " << message << "\n";
        MPI_Barrier(MPI_COMM_WORLD);
    }

    /// "dummy"
    MPI_Finalize();
    return 0;
}
