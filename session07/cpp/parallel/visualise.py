import sys,os
import numpy
import matplotlib.animation
import matplotlib.pyplot
from StringIO import StringIO

folder=sys.argv[1]

with open(os.path.join(folder,'frames0.dat')) as data:
    header=numpy.genfromtxt(StringIO(data.readline()),delimiter=",",dtype=int)
    rows=header[0]
    
process_frames=[]
for process in range(header[3]):
    data=numpy.genfromtxt(os.path.join(folder,'frames'+str(process)+'.dat'),delimiter=",",skip_header=1)[:,:-1]
    lines, columns=data.shape
    process_frames.append(data.reshape((lines/rows,rows,columns)))
frames=numpy.concatenate(process_frames, 1)

figure=matplotlib.pyplot.figure()
def animate(frame_id):
    print "Processing frame", frame_id
    matplotlib.pyplot.imshow(frames[frame_id],vmin=0,vmax=1)

anim=matplotlib.animation.FuncAnimation(figure,animate,frames=len(frames),interval=100)
anim.save(os.path.join(folder,'smooth.mp4'))

