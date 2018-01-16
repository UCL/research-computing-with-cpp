#include <itkImage.h>
#include <itkAddImageFilter.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

/// "typedefs"
int main(int argc, char** argv)
{

  const unsigned int Dimension = 2;
  typedef int PixelType;
  typedef itk::Image<PixelType, Dimension> ImageType;
  typedef itk::AddImageFilter<ImageType, ImageType> AddFilterType;
  typedef itk::ImageFileReader<ImageType> ImageReaderType;
  typedef itk::ImageFileWriter<ImageType> ImageWriterType;

  /// "construction"
  ImageReaderType::Pointer reader1 = ImageReaderType::New();
  ImageReaderType::Pointer reader2 = ImageReaderType::New();
  AddFilterType::Pointer addFilter = AddFilterType::New();
  ImageWriterType::Pointer writer = ImageWriterType::New();

  // eg. if not using typedefs
  //itk::ImageFileWriter< itk::Image<int, 2> >::Pointer writer
  //  = itk::ImageFileWriter< itk::Image<int, 2> >::New();

  /// "pipeline"
  reader1->SetFileName("inputFileName1.nii");
  reader2->SetFileName("inputFileName2.nii");
  addFilter->SetInput(0, reader1->GetOutput());
  addFilter->SetInput(1, reader2->GetOutput());
  writer->SetInput(addFilter->GetOutput());
  writer->SetFileName("outputFileName1.nii");
  //writer->Update(); // commented out, as filenames are fake.
                      // and build system for lecture notes
                      // tries to run the program.
  return 0;
}
