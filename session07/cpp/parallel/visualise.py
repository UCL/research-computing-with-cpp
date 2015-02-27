import sys,os
import numpy
import matplotlib.animation
import matplotlib.pyplot
from StringIO import StringIO

def plot(frames,outpath):
    figure = matplotlib.pyplot.figure()
    def _animate(frame_id):
        print "Processing frame", frame_id
        matplotlib.pyplot.imshow(frames[frame_id], vmin=0, vmax=1)

    anim = matplotlib.animation.FuncAnimation(figure, _animate, len(frames), interval=100)
    anim.save(outpath)

def append_process_suffix(prefix, process):
    return prefix + '.' + str(process)

def process_many_files(folder, prefix, size, header_type, bulk_type):
    process_frames=[]
    for process in range(size):
        path=os.path.join(folder,append_process_suffix(prefix,process))
        with open(path) as data:
            header=read_header(data, header_type)
            width, height, size, frame_count = header
            if header_type=='text':
                buffer=numpy.genfromtxt(data,delimiter=",")[:,:-1]
            else:
                buffer = numpy.fromfile(data, bulk_type, frame_count*height*width/size)
            frames_data=buffer.reshape([frame_count, width/size, height])
            process_frames.append(frames_data)
    return numpy.concatenate(process_frames, 1)

def process_single_file(folder, name, header_type, bulk_type):
    path=os.path.join(folder,prefix)
    with open(path) as data:
        header=read_header(data, header_type)
        width, height, size, frame_count = header
        buffer=numpy.fromfile(data, bulk_type, frame_count*height*width)
    return buffer.reshape([frame_count,width,height])

def read_header(data, header_type):
    if header_type=='text':
        line=data.readline()
        return numpy.genfromtxt(StringIO(line),delimiter=",",dtype=int)
    else:
        return numpy.fromfile(data,header_type,4)

def read_mpi_size(folder, prefix, header_type):
    with open(os.path.join(folder, append_process_suffix(prefix, 0))) as data:
        rows, columns, size, frame_count = read_header(data,header_type)
        return size

folder = sys.argv[1]
if len(sys.argv)>2:
    mode = sys.argv[2]
else:
    mode = 'xdr'

if len(sys.argv)>3:
    split = (sys.argv[3]=="split")
else:
    split = True

outpath = os.path.join(folder, 'smooth.mp4')
prefix = 'frames.dat'

header_types = {
    'xdr' : '>i4',
    'text': 'text',
    'native' : 'i4'
}

bulk_types = {
    'xdr' : '>f8',
    'text': 'text',
    'native' : '<f8'
}

header_type=header_types[mode]
bulk_type =bulk_types[mode]

size=read_mpi_size(folder, prefix, header_type)
if split:
    frames=process_many_files(folder, prefix, size, header_type, bulk_type)
else:
    frames=process_single_file(folder, prefix, header_type, bulk_type)
plot(frames, outpath)

