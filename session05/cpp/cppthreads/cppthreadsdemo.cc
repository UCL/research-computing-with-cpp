#include <thread>
#include <iostream>

void f()
{
    std::cout << "Hello" << std::endl;
}; 

void g()
{
    std::cout << "world" << std::endl;
}; 


int main(int argc, char ** argv)
{
    std::thread t1 {f};
    std::thread t2 {g};

    t1.join();
    t2.join();
}