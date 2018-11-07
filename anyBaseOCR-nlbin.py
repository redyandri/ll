#!/usr/bin/env python

from __future__ import print_function
from pylab import *
from numpy.ctypeslib import ndpointer
import argparse,os,os.path
from scipy.ndimage import filters,interpolation,morphology,measurements
from scipy import stats
import multiprocessing
import ocrolib



parser = argparse.ArgumentParser("""
Image binarization using non-linear processing.

This is a compute-intensive binarization method that works on degraded
and historical book pages.
""")

parser.add_argument('-n','--nocheck',action="store_true",
                    help="disable error checking on inputs")
parser.add_argument('-t','--threshold',type=float,default=0.5,help='threshold, determines lightness, default: %(default)s')
parser.add_argument('-z','--zoom',type=float,default=0.5,help='zoom for page background estimation, smaller=faster, default: %(default)s')
parser.add_argument('-e','--escale',type=float,default=1.0,help='scale for estimating a mask over the text region, default: %(default)s')
parser.add_argument('-b','--bignore',type=float,default=0.1,help='ignore this much of the border for threshold estimation, default: %(default)s')
parser.add_argument('-p','--perc',type=float,default=80,help='percentage for filters, default: %(default)s')
parser.add_argument('-r','--range',type=int,default=20,help='range for filters, default: %(default)s')
parser.add_argument('-m','--maxskew',type=float,default=2,help='skew angle estimation parameters (degrees), default: %(default)s')
parser.add_argument('-g','--gray',action='store_true',help='force grayscale processing even if image seems binary')
parser.add_argument('--lo',type=float,default=5,help='percentile for black estimation, default: %(default)s')
parser.add_argument('--hi',type=float,default=90,help='percentile for white estimation, default: %(default)s')
parser.add_argument('--skewsteps',type=int,default=8,help='steps for skew angle estimation (per degree), default: %(default)s')
parser.add_argument('--debug',type=float,default=0,help='display intermediate results, default: %(default)s')
parser.add_argument('--show',action='store_true',help='display final result')
parser.add_argument('--rawcopy',action='store_true',help='also copy the raw image')
parser.add_argument('-o','--output',default=None,help="output directory")
parser.add_argument('files',nargs='+')
parser.add_argument('-Q','--parallel',type=int,default=0)
args = parser.parse_args()

args.files = ocrolib.glob_all(args.files)

if len(args.files)<1:
    parser.print_help()
    sys.exit(0)


def print_info(*objs):
    print("INFO: ", *objs, file=sys.stdout)

def print_error(*objs):
    print("ERROR: ", *objs, file=sys.stderr)

def check_page(image):
    if len(image.shape)==3: return "input image is color image %s"%(image.shape,)
    if mean(image)<median(image): return "image may be inverted"
    h,w = image.shape
    if h<600: return "image not tall enough for a page image %s"%(image.shape,)
    if h>10000: return "image too tall for a page image %s"%(image.shape,)
    if w<600: return "image too narrow for a page image %s"%(image.shape,)
    if w>10000: return "line too wide for a page image %s"%(image.shape,)
    return None

def estimate_scale(binary):
    objects = binary_objects(binary)
    bysize = sorted(objects,key=A)
    scalemap = zeros(binary.shape)
    for o in bysize:
        if amax(scalemap[o])>0: continue
        scalemap[o] = A(o)**0.5
    scale = median(scalemap[(scalemap>3)&(scalemap<100)])
    return scale

def estimate_skew_angle(image,angles):
    estimates = []
    for a in angles:
        v = mean(interpolation.rotate(image,a,order=0,mode='constant'),axis=1)
        v = var(v)
        estimates.append((v,a))
    if args.debug>0:
        plot([y for x,y in estimates],[x for x,y in estimates])
        ginput(1,args.debug)
    _,a = max(estimates)
    return a

def select_regions(binary,f,min=0,nbest=100000):
    labels,n = measurements.label(binary)
    objects = measurements.find_objects(labels)
    scores = [f(o) for o in objects]
    best = argsort(scores)
    keep = zeros(len(objects)+1,'B')
    for i in best[-nbest:]:
        if scores[i]<=min: continue
        keep[i+1] = 1
    return keep[labels]

def H(s): return s[0].stop-s[0].start
def W(s): return s[1].stop-s[1].start
def A(s): return W(s)*H(s)

def dshow(image,info):
    if args.debug<=0: return
    ion(); gray(); imshow(image); title(info); ginput(1,args.debug)

def process1(job):
    fname,i = job
    print_info("# %s" % (fname))
    if args.parallel<2: print_info("=== %s %-3d" % (fname, i))
    raw = ocrolib.read_image_gray(fname)
    dshow(raw,"input")
    # perform image normalization
    image = raw-amin(raw)
    if amax(image)==amin(image):
        print_info("# image is empty: %s" % (fname))
        return
    image /= amax(image)

    if not args.nocheck:
        check = check_page(amax(image)-image)
        if check is not None:
            print_error(fname+"SKIPPED"+check+"(use -n to disable this check)")
            return

    # check whether the image is already effectively binarized
    if args.gray:
        extreme = 0
    else:
        extreme = (sum(image<0.05)+sum(image>0.95))*1.0/prod(image.shape)
    if extreme>0.95:
        comment = "no-normalization"
        flat = image
    else:
        comment = ""
        # if not, we need to flatten it by estimating the local whitelevel
        if args.parallel<2: print_info("flattening")
        m = interpolation.zoom(image,args.zoom)
        m = filters.percentile_filter(m,args.perc,size=(args.range,2))
        m = filters.percentile_filter(m,args.perc,size=(2,args.range))
        m = interpolation.zoom(m,1.0/args.zoom)
        if args.debug>0: clf(); imshow(m,vmin=0,vmax=1); ginput(1,args.debug)
        w,h = minimum(array(image.shape),array(m.shape))
        flat = clip(image[:w,:h]-m[:w,:h]+1,0,1)
        if args.debug>0: clf(); imshow(flat,vmin=0,vmax=1); ginput(1,args.debug)

    # estimate skew angle and rotate
    if args.maxskew>0:
        if args.parallel<2: print_info("estimating skew angle")
        d0,d1 = flat.shape
        o0,o1 = int(args.bignore*d0),int(args.bignore*d1)
        flat = amax(flat)-flat
        flat -= amin(flat)
        est = flat[o0:d0-o0,o1:d1-o1]
        ma = args.maxskew
        ms = int(2*args.maxskew*args.skewsteps)
        angle = estimate_skew_angle(est,linspace(-ma,ma,ms+1))
        flat = interpolation.rotate(flat,angle,mode='constant',reshape=0)
        flat = amax(flat)-flat
    else:
        angle = 0

    # estimate low and high thresholds
    if args.parallel<2: print_info("estimating thresholds")
    d0,d1 = flat.shape
    o0,o1 = int(args.bignore*d0),int(args.bignore*d1)
    est = flat[o0:d0-o0,o1:d1-o1]
    if args.escale>0:
        # by default, we use only regions that contain
        # significant variance; this makes the percentile
        # based low and high estimates more reliable
        e = args.escale
        v = est-filters.gaussian_filter(est,e*20.0)
        v = filters.gaussian_filter(v**2,e*20.0)**0.5
        v = (v>0.3*amax(v))
        v = morphology.binary_dilation(v,structure=ones((int(e*50),1)))
        v = morphology.binary_dilation(v,structure=ones((1,int(e*50))))
        if args.debug>0: imshow(v); ginput(1,args.debug)
        est = est[v]
    lo = stats.scoreatpercentile(est.ravel(),args.lo)
    hi = stats.scoreatpercentile(est.ravel(),args.hi)
    # rescale the image to get the gray scale image
    if args.parallel<2: print_info("rescaling")
    flat -= lo
    flat /= (hi-lo)
    flat = clip(flat,0,1)
    if args.debug>0: imshow(flat,vmin=0,vmax=1); ginput(1,args.debug)
    bin = 1*(flat>args.threshold)

    # output the normalized grayscale and the thresholded images
    print_info("%s lo-hi (%.2f %.2f) angle %4.1f %s" % (fname, lo, hi, angle, comment))
    if args.parallel<2: print_info("writing")
    if args.debug>0 or args.show: clf(); gray();imshow(bin); ginput(1,max(0.1,args.debug))
    if args.output:
        if args.rawcopy: ocrolib.write_image_gray(args.output+"/%04d.raw.png"%i,raw)
        base,_ = os.path.splitext(os.path.basename(fname))
        ocrolib.write_image_binary(args.output+"/"+base+".bin.png",bin) #ocrolib.write_image_binary(args.output+"/%04d.bin.png"%i,bin)
        ocrolib.write_image_gray(args.output+"/"+base+".nrm.png",flat) #ocrolib.write_image_gray(args.output+"/%04d.nrm.png"%i,flat)
    else:
        base,_ = ocrolib.allsplitext(fname)
        ocrolib.write_image_binary(base+".bin.png",bin)
        ocrolib.write_image_gray(base+".nrm.png",flat)

if args.debug>0 or args.show>0: args.parallel = 0

if args.output:
    if not os.path.exists(args.output):
        os.mkdir(args.output)

if args.parallel<2:
    for i,f in enumerate(args.files):
        process1((f,i+1))
else:
    pool = multiprocessing.Pool(processes=args.parallel)
    jobs = []
    for i,f in enumerate(args.files): jobs += [(f,i+1)]
    result = pool.map(process1,jobs)
