class Shape {
public:
  Shape();
  void setVisible(const bool &isVisible) { m_IsVisible = isVisible; }
  virtual void rotate(const double &degrees) = 0;
  virtual void scale(const double &factor) = 0;
  // + other methods
private:
  bool m_IsVisible;
  unsigned char m_Colour[3]; // RGB
  double m_CentreOfMass[2];
};

class Rectangle : public Shape {
public:
  Rectangle();
  virtual void rotate(const double &degrees);
  virtual void scale(const double &factor);
  // + other methods
private:
  double m_Corner1[2];
  double m_Corner2[2];
};

class Circle : public Shape {
public:
  Circle();
  virtual void rotate(const double &degrees);
  virtual void scale(const double &factor);
  // + other methods
private:
  float radius;
};