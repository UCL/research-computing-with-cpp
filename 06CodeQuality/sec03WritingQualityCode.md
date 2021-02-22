---
title: Writing Quality Code
---

> Programs are meant to be read by humans and only incidentally for computers to execute

If you leave this lesson with only the above quote from [Donald Knuth][wiki-dk] burned into your brain, I'll be happy. This section is about encouraging you to think about how you write C++ in a readable way. The more readable your code is, the easier it it to spot mistakes, add features, and generally maintain. Let's start with some truly terrible code (courtesy of [Katrina Ward][kat-example]):

``` cpp
// Kat                                                               

#include <iostream>
using namespace std;

int main()
{
  char mlevel;              // this is a valid char                   
  bool chalmers;
  string name;
  double r, g, c, peoplewhoareadults;                                            
                cout << "Hi" << endl;
				cin >> name >> endl;
		cin >> r >> endl;                                            
		cin >> g >> endl;
		cin >> c >> endl;
		cin >> peoplewhoareadults >> endl;
		cin >> chalmers >> endl;
  cout << "What is your motivation?\n"
       << " a. Amazingly Motivated\n"
	   << " b. Basicly a good worker\n"
	   << " c. Can't get good help no more\n"
	   << " d. Don't plan on work from me\n"
	   << " e. Elevated Slothfullness \n\n"
	   << "Enter the letter of your choice: ";
	   cin >> mlevel;
	   
	   if(mlevel=='a')   // this is an if statement
	   {
	   if(r>=0.5)
	   {
	   if(g<=5)    // checks if g <= 5
	   {
	   cout << "burn books" << endl;
	   }
	   else
	   {
	   cout << "clean the bathroom" << endl;
	   }
	   }
	   else
	   {
	   if(g<=5)
	   {
	   cout << "Go get more g" << endl;
	   }
	   else if(g>=5 && g<10)
	   {
	   cout << "Mow grass" << endl;
	   }
	   else
	   cout << "Do laps in the tractor" << endl;
	   }
	   // This is a comment right in the middle that says I stayed up too late and didn't do my homework
	   // and now my code looks horrible. Large paragraphs of comments makes your code harder to read
	   // try using short statements to briefly explain what a block of code is actually doing instead
	   // of paragraphs that state nothing really important.
	   }
	   else if(mlevel=='b')
	   {
	            if(chalmers==1||peoplewhoareadults>c)
		{
		    cout << "Scrub floors on hands and knees" << endl;
			}
			   else
			   {
			   cout << "Mop the floor." << endl;
			                    }
						}
		else if(mlevel=='c')
		{
		if(r<=1.5)
		{
		                         cout << "Lean on rake." << endl;
								 }
					else
					     cout << "Lean on broom inside" << end;
		}
		else if(mlevel=='e')
		{
		  cout << "Stay in bed" << endl;
		}
		
		return 0;
	}
			   
```

### Exercise

Read the above code and write down a list of **five** things you'd recommend to Kat to improve her code. This could include:
- how comments are used
- variable names
- formatting
- software design (or lack of!)

### Exercise

Read [J. B. Rainsberger's *The Four Elements of Simple Design*][four-elements] and answer the following questions. Keep your answers handy; we'll be discussing this on Friday.
- Do you agree with his 4 elements? What about the 2 elements he ends up with? Why?
- J codes in an iterative way, renaming functions to `foo` before he knows what they *really* do. How could this impact collaboration if J is writing functions called `foo`?
- What is the value in this kind of iterative programming?
- What are the potential downsides?
- How would you use J's iterative programming style to improve Kat's code from the previous exercise?


[wiki-dk]: https://en.wikipedia.org/wiki/Donald_Knuth
[kat-example]: https://web.mst.edu/~price/cs53/code_example.html
[four-elements]: https://blog.jbrains.ca/permalink/the-four-elements-of-simple-design
