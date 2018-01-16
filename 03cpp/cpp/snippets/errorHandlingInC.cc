int foo(int a, int b)
{
  // stuff
  
  if(some error condition)
  {
    return 1;
  } else if (another error condition) {
    return 2;
  } else {
    return 0;
  }
}

void caller(int a, int b) 
{
 int result = foo(a, b);
 if (result == 1) // do something
 else if (result == 2) // do something difference
 else 
 {
   // All ok, continue as you wish
 }
} 
