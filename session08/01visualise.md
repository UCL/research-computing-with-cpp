---
title: Visualisation
---

## Visualisation

### Visualisation is important

We've put a lot of emphasis onto using **Unit Tests** to verify your code.

However, it's just as important to *visualise* your results. 

"The graph looks OK" is not a good enough standard of truth.

But "the graph looks wrong" is a strong indication of problems!

### Visualisation is hard

Three-D slices through your data, contour surfaces, animations...

These are all really important for understanding scientific output.

But doing the visualisation in C++ can be very very difficult.

### Simulate remotely, analyse locally

Visualise locally if you can.

Use a tool like fabric to organise your data and make it easy to ship it back from the cluster.

Use dynamic languages like Python or Matlab to visualise your data, or tools like
[Paraview](http://www.paraview.org) or [VisIt](https://wci.llnl.gov/simulation/computer-codes/visit/).

### In-Situ visualisation

When working at the highest scales of parallel, saving out raw simulation data becomes impossible:
compute power scales faster than storage.

Under these circumstances, it is necessary to do **data reduction** on the cluster.

In-situ visualisation, where animations or graphs are constructed as part of the analysis, is one approach.

### Visualising with NumPy and Matplotlib

The Python toolchain for visualising quantitative data is powerful, fast and easy.

Here's how we turn a NumPy Matrix view of our data, with axes (frames, width, height), into an animation and
save it to disk as an mp4:

```python
{{d['session07/cpp/parallel/visualise.py|idio|t']['Animation']}}
```

Try doing that with C++ libraries!
