---
title: Overlap
---

## Overlapping computation and communication

### Wasteful blocking

We are calculating our whole field, then sharing all the halo.

This is wasteful: we don't need our neighbour's data to calculate the data in the *middle*
of our field.

We can start transmitting our halo and receiving our neighbour's, while calculating our
middle section. Thus, our communication will not take up extra time, as it is overlapped with calculation.

To do this, we need to use asynchronous communication.

### Asynchronous communication strategy

* Start sending/receiving data
* Do the part of calculation independent of needed data
* Wait until the communication is complete (hopefully instant, if it's already finished)
* Do the part of the calculation that needs communicated data

### Asynchronous communication for 2-d halo:

* Start clockwise send/receive
* Calculate middle
* Finish clockwise send
* Start anticlockwise send
* Calculate right
* Finish anticlockwise send
* Calculate left

### Asyncronous MPI


{% idio cpp/parallel/src/Smooth.cpp %}


{% fragment Start_Asynchronous_Left %}

{% fragment Resolve_Asynchronous_Left %}

### Implementation of Asynchronous Communication

{% fragment Update_Asynchronously %}

{% endidio %}
