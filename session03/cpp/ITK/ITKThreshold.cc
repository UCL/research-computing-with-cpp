#include <itkImageToImageFilter.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>

/// "intro"
namespace itk
{
template< class TInputImage, class TOutputImage = TInputImage>
class MyThresholdFilter:public ImageToImageFilter< TInputImage, TOutputImage >
{
public:

  /// "boilerplate"
  /** Standard class typedefs. */
  typedef MyThresholdFilter                               Self;
  typedef ImageToImageFilter< TInputImage, TOutputImage > Superclass;
  typedef SmartPointer< Self >                            Pointer;
  typedef typename TInputImage::PixelType                 InputPixelType;
  typedef typename TOutputImage::PixelType                OutputPixelType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ImageFilter, ImageToImageFilter);

  /// "macro"
  itkSetMacro(Low, InputPixelType);
  itkGetMacro(Low, InputPixelType);
  itkSetMacro(High, InputPixelType);
  itkGetMacro(High, InputPixelType);

protected:
  MyThresholdFilter(){}
  ~MyThresholdFilter(){}

  /// "method"
  /** Does the real work. */
  virtual void GenerateData()
  {
    TInputImage  *inputImage  = static_cast< TInputImage  * >(this->ProcessObject::GetInput(0));
    TOutputImage *outputImage = static_cast< TOutputImage * >(this->ProcessObject::GetOutput(0));

    ImageRegionConstIterator<TInputImage> inputIterator = ImageRegionConstIterator<TInputImage>(inputImage, inputImage->GetLargestPossibleRegion());
    ImageRegionIterator<TOutputImage> outputIterator = ImageRegionIterator<TOutputImage>(outputImage, outputImage->GetLargestPossibleRegion());


    for (inputIterator.GoToBegin(),
         outputIterator.GoToBegin();
         !inputIterator.IsAtEnd() && !outputIterator.IsAtEnd();
         ++inputIterator,
         ++outputIterator)
    {
      if (*inputIterator >= m_Low && *inputIterator <= m_High)
      {
        *outputIterator = 1;
      }
      else
      {
        *outputIterator = 0;
      }
    }
  }

private:
  MyThresholdFilter(const Self &); //purposely not implemented
  void operator=(const Self &);  //purposely not implemented
  InputPixelType m_Low;
  InputPixelType m_High;
};
} // end namespace

int main(int argc, char** argv)
{
  // Not providing a real example,
  // as I dont know how to read/write images within the
  // dexy framework.
  return 0;
}