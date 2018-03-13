#include <stdio.h>
#include <stdlib.h>

int compare(const void* a, const void* b)
{
  return ( *(int*)a - *(int*)b );
}

int main()
{
  FILE* if1 = fopen("90stl/cpp/randomNumbers1.txt","r");
  FILE* if2 = fopen("90stl/cpp/randomNumbers2.txt","r");

  // First determine size of array needed
  int size1=0, size2=0;
  while (!feof(if1)) {
    fscanf(if1, "%*d");
    if (!feof(if1)) size1++;
  }
  rewind(if1);
  while (!feof(if2)) {
    fscanf(if2, "%*d");
    if (!feof(if2)) size2++;
  }
  rewind(if2);

  // Read in the data
  int theArray[size1+size2];
  for (int i=0;i<size1;++i) {
    fscanf(if1, "%d", &theArray[i]);
  }
  for (int i=size1;i<size1+size2;++i) {
    fscanf(if2, "%d", &theArray[i]);
  }
  fclose(if1);
  fclose(if2);

  // Sort and output
  qsort(theArray,size1+size2,sizeof(int),compare);
  for (int i=0;i<size1+size2;++i) {
    printf("%d ", theArray[i]);
  }      
}
