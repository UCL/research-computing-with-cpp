#include <cmath>
#include <cstdlib>
#include <iostream>

#include "smooth.h"


Smooth::Smooth(int sizex,
        int sizey,
        distance inner,
        filling birth_1,
        filling birth_2 ,
        filling death_1,
        filling death_2,
        filling smoothing_disk,
        filling smoothing_ring)
    : sizex(sizex),
    sizey(sizey),
    inner(inner),
    birth_1(birth_1),
    birth_2(birth_2),
    death_1(death_1),
    death_2(death_2),
    smoothing_disk(smoothing_disk),
    smoothing_ring(smoothing_ring),
    outer(inner*3),
    smoothing(1.0),
    field1(sizex,std::vector<density>(sizey)),
    field2(sizex,std::vector<density>(sizey)),
    field(&field1),
    fieldNew(&field2),
    frame(0)
{
  normalisation_disk=NormalisationDisk();
  normalisation_ring=NormalisationRing();
}

int Smooth::Range(){
  return outer+smoothing/2;
}

int Smooth::Sizex(){
  return sizex;
}
int Smooth::Sizey(){
  return sizey;
}
int Smooth::Size(){
  return sizex*sizey;
}

/// "Disk_Smoothing"
double Smooth::Disk(distance radius) const {
  if (radius>inner+smoothing/2) {
    return 0.0;
  }
  if (radius<inner-smoothing/2) {
    return 1.0;
  }
  return (inner+smoothing/2-radius)/smoothing;
}

/// "Ring_Smoothing"
double Smooth::Ring(distance radius) const {
  if (radius<inner-smoothing/2) {
    return 0.0;
  }
  if (radius<inner+smoothing/2) {
    return (radius+smoothing/2-inner)/smoothing;
  }
  if (radius<outer-smoothing/2) {
    return 1.0;
  }
  if (radius<outer+smoothing/2) {
    return (outer+smoothing/2-radius)/smoothing;
  }
  return 0.0;
 }

double Smooth::Sigmoid(double variable, double center, double width){
  return 1.0/(1.0+std::exp(4.0*(center-variable)/width));
}

density Smooth::transition(filling disk, filling ring) const {
  double t1=birth_1*(1.0-Sigmoid(disk,0.5,smoothing_disk))+death_1*Sigmoid(disk,0.5,smoothing_disk);
  double t2=birth_2*(1.0-Sigmoid(disk,0.5,smoothing_disk))+death_2*Sigmoid(disk,0.5,smoothing_disk);
  return Sigmoid(ring,t1,smoothing_ring)*(1.0-Sigmoid(ring,t2,smoothing_ring));
}

const std::vector<std::vector<density> > & Smooth::Field() const {
  return *field;
};

/// "Torus_Difference"
int Smooth::TorusDifference(int x1, int x2, int size) const {
    int straight=std::abs(x2-x1);
    int wrapleft=std::abs(x2-x1+size);
    int wrapright=std::abs(x2-x1-size);
    if ((straight<wrapleft) && (straight<wrapright)) {
      return straight;
    } else {
      return (wrapleft < wrapright) ? wrapleft : wrapright;
    }
}

/// "Radius"
double Smooth::Radius(int x1,int y1,int x2,int y2) const {
  int xdiff=TorusDifference(x1,x2,sizex);
  int ydiff=TorusDifference(y1,y2,sizey);
  return std::sqrt(xdiff*xdiff+ydiff*ydiff);
}

double Smooth::NormalisationDisk() const {
  double total=0.0;
  for (int x=0;x<sizex;x++) {
     for (int y=0;y<sizey;y++) {
       total+=Disk(Radius(0,0,x,y));
    }
  };
  return total;
}

double Smooth::NormalisationRing() const {
  double total=0.0;
  for (int x=0;x<sizex;x++) {
     for (int y=0;y<sizey;y++) {
       total+=Ring(Radius(0,0,x,y));
    }
  };
  return total;
}

filling Smooth::FillingDisk(int x, int y) const {
  double total=0.0;
  for (int x1=0;x1<sizex;x1++) {
     for (int y1=0;y1<sizey;y1++) {
       total+=(*field)[x1][y1]*Disk(Radius(x,y,x1,y1));
    }
  };
  return total/normalisation_disk;
}

filling Smooth::FillingRing(int x, int y) const {
  double total=0.0;
  for (int x1=0;x1<sizex;x1++) {
     for (int y1=0;y1<sizey;y1++) {
       total+=(*field)[x1][y1]*Ring(Radius(x,y,x1,y1));
    }
  };
  return total/normalisation_ring;
}

density Smooth::NewState(int x, int y) const {
  return transition(FillingDisk(x,y),FillingRing(x,y));
}

void Smooth::Update() {
   for (int x=0;x<sizex;x++) {
     for (int y=0;y<sizey;y++) {
      (*fieldNew)[x][y]=NewState(x,y);
     }
   }

  std::vector<std::vector<density> > * fieldTemp;
  fieldTemp=field;
  field=fieldNew;
  fieldNew=fieldTemp;
  frame++;

}
/// "Main_Loop"
void Smooth::QuickUpdate() {
  for (int x=0;x<sizex;x++) {
    for (int y=0;y<sizey;y++) {
      double ring_total=0.0;
      double disk_total=0.0;

      for (int x1=0;x1<sizex;x1++) {
          int deltax=TorusDifference(x,x1,sizex);
          if (deltax>outer+smoothing/2) continue;

          for (int y1=0;y1<sizey;y1++) {
            int deltay=TorusDifference(y,y1,sizey);
            if (deltay>outer+smoothing/2) continue;

            double radius=std::sqrt(deltax*deltax+deltay*deltay);
            double fieldv=(*field)[x1][y1];
            ring_total+=fieldv*Ring(radius);
            disk_total+=fieldv*Disk(radius);
          }
      }

      (*fieldNew)[x][y]=transition(disk_total/normalisation_disk,ring_total/normalisation_ring);
    }
  }

/// "Swap_Fields"
  std::vector<std::vector<density> > * fieldTemp;
  fieldTemp=field;
  field=fieldNew;
  fieldNew=fieldTemp;
  frame++;
}
/// "Seed_Random"
void Smooth::SeedRandom() {
   for (int x=0;x<sizex;x++) {
     for (int y=0;y<sizey;y++) {
      (*field)[x][y]=(static_cast<double>(rand()) / static_cast<double>(RAND_MAX));
     }
   }
}

void Smooth::SeedDisk() {
   for (int x=0;x<sizex;x++) {
     for (int y=0;y<sizey;y++) {
      (*field)[x][y]=Disk(Radius(0,0,x,y));
     }
   }
}

void Smooth::SeedRing() {
   for (int x=0;x<sizex;x++) {
     for (int y=0;y<sizey;y++) {
      (*field)[x][y]=Ring(Radius(0,0,x,y));
     }
   }
}

void Smooth::Write(std::ostream &out) {
   for (int x=0;x<sizex;x++) {
     for (int y=0;y<sizey;y++) {
        out << (*field)[x][y] << " , ";
     }
     out << std::endl;
   }
   out << std::endl;
}

int Smooth::Frame() const {
  return frame;
}
