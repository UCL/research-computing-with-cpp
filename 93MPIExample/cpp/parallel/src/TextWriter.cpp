#include "TextWriter.h"
#include <fstream>

/// "Write"
void TextWriter::Write() {
  for(int x = smooth.Range(); x < smooth.LocalXSize() + smooth.Range(); x++) {
    for(int y = 0; y < smooth.Sizey(); y++) {
      *outfile << smooth.Field(x, y) << " , ";
    }
    *outfile << std::endl;
  }
  *outfile << std::endl;
}

/// "Delete"
TextWriter::~TextWriter() {
  delete outfile; // closes it
}

TextWriter::TextWriter(Smooth &smooth, int rank, int size) : SmoothWriter(smooth, rank, size) {
  /// "AddName"
  fname << "." << rank << std::flush;
  /// "Create"
  outfile = new std::ofstream(fname.str().c_str());
}

/// "Header"
void TextWriter::Header(int frames) {
  *outfile << smooth.LocalXSize() << ", " << smooth.Sizey() << ", " << rank << ", " << size
           << std::endl;
}
