#include <mpi.h>

/// "Sends"
void root_sends_message(std::string const &message) {
  int const error = MPI_Ssend(
    (void*) message.c_str(), message.size(), MPI_CHAR, 0, 42, MPI_COMM_WORLD
  );
  if(error !=  MPI_SUCCESS) throw;
}

/// "Receive"
std::string uno_gets_it() {
  char message[256];
  int const error = MPI_Recv(
          message, 256, MPI_CHAR, 0, 42,
          MPI_COMM_WORLD, MPI_STATUS_IGNORE
  );
  if(error !=  MPI_SUCCESS) throw;
  return message;
}

/// "main"
int main(int argc, char * argv[]) {
    /// Must be first call
    MPI_Init (&argc, &argv);

    int rank, size;
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);

    /// "sendAndReceive"
    if(rank == 0) root_sends_message("I come in peace!");
    else if(rank == 1)
        std::cout << "I am " << rank << " of " << size
            << " and I approve the following message: "
            << uno_gets_it() << "\n";

    /// "dummy"
    MPI_Finalize();
    return 0;
}
