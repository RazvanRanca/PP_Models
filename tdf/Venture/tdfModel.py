from venture.shortcuts import *
import sys
sys.path.append("../../")
import ppUtils as pu
from collections import Counter
import numpy as np
import os
import time
from matplotlib import pyplot as plt
import calcPost as cp
import random
import math
import cPickle
from itertools import chain
import scipy

def testPerf(ys):
  with open('rtRes', 'w') as f:
    for samples in [10000]:
      for dataPs in [10,50,100,500,1000]:
        for infType in range(4):
          startTime = time.time()
          v = make_church_prime_ripl()
          v.assume("d", "(uniform_continuous 2 100)")
          v.assume("y", "(lambda () (student_t d))")
          if dataPs > 0:
            [v.observe("(y)", str(ys[i])) for i in range(dataPs)]
          if infType == 0:
            v.infer(samples)
          elif infType == 1:
            for i in range(samples):
              v.infer(1)
          elif infType == 2:
            for i in range(samples):
              v.infer(1)
              label = "d" + str(i)
              v.predict("d",label)
          elif infType == 3:
            s = []
            for i in range(samples):
              v.infer(1)
              label = "d" + str(i)
              v.predict("d",label)
              s.append(v.report(label))
          rez = str(samples) + " " + str(dataPs) + " " + str(infType) + " " + str(time.time() - startTime)
          #f.write(rez + "\n")
          #f.flush()
          print rez


def genTdf(v):
  v.assume("y", "(student_t 4)")
  samples = pu.posterior_samples(v, "y", no_samples=1000, no_burns=0, int_mh=1)
  #pu.save_samples(samples, os.getcwd(), "gen")

def runModel(v, ys, mType, sample, burn, lag, timeTest = False, silentSamp = False):
  timeStart = time.time()
  if mType == "cont":
    v.assume("d", "(uniform_continuous 2 100)")
    v.assume("y", "(lambda () (student_t d))")
  elif mType == "cont5var":
    v.assume("d1", "(uniform_continuous 2 100)")
    v.assume("d2", "(uniform_continuous 0 0.001)")
    v.assume("d3", "(uniform_continuous 0 0.001)")
    v.assume("d4", "(uniform_continuous 0 0.0001)")
    v.assume("d5", "(uniform_continuous 0 0.0001)")
    v.assume("d", "(+ d1 d2 d3 d4 d5)") #(uniform_continuous 0 5) (uniform_continuous 0 1))")
    v.assume("y", "(lambda () (student_t d))")
  elif mType == "contBin":
    v.assume("f1", "(* 0.5 (categorical 0.5 0.5))")
    v.assume("r", "(uniform_continuous 0 0.5)")
    v.assume("d", "(+ f1 r)")
    v.assume("y", "(lambda () (student_t d))")
  elif mType == "contMix":
    v.assume("d1", "(uniform_continuous 2 97)")
    v.assume("d2", "(uniform_continuous 0 1)")
    v.assume("d3", "(uniform_continuous 0 2)") 
    v.assume("d", "(+ d1 d2 d3)") #(uniform_continuous 0 5) (uniform_continuous 0 1))")
    v.assume("y", "(lambda () (student_t d))")
  elif mType == "flip":
    v.assume("d0", "(uniform_continuous 0 1)")
    v.assume("f0", "(uniform_discrete 0 2)")
    v.assume("f1", "(* 2 (uniform_discrete 0 2))")
    v.assume("f2", "(* 4 (uniform_discrete 0 2))")
    v.assume("f3", "(* 8 (uniform_discrete 0 2))")
    v.assume("f4", "(* 16 (uniform_discrete 0 2))")
    v.assume("f5", "(* 32 (uniform_discrete 0 2))")
    v.assume("f6", "(* 64 (uniform_discrete 0 2))")
    v.assume("d", "(+ (+ (+ d0 f0) (+ f1 f2)) (+ (+ f3 f4) (+ f5 f6)))") #(uniform_continuous 0 5) (uniform_continuous 0 1))")
    v.assume("y", "(lambda () (student_t d))")
  elif mType == "disc1":
    v.assume("d", "(uniform_discrete 2 50)")
    v.assume("y", "(lambda () (student_t d))")
  elif mType == "disc2":
    v.assume("d", "(/ (uniform_discrete 20 60) 10)")
    v.assume("y", "(lambda () (student_t d))")
  elif mType == "disc21":
    v.assume("d", "(uniform_discrete 20 60)")
    v.assume("y", "(lambda () (student_t (/ d 10)))")
  else:
    raise Exception("Unknown model type: " + mType)

  [v.observe("(y)", str(ys[i])) for i in range(len(ys))]
  samples = pu.posterior_samples(v, "d", no_samples=sample, no_burns=burn, int_mh=lag, silent = silentSamp)

  if timeTest:
    return time.time() - timeStart
  print '\n'.join([str(samples[i][1]) for i in range(len(samples)) if i%10==0])
  #v.infer(11000000)
  vals = map(lambda x:x[1], samples)
  print "Sample mean: ", np.mean(vals), " Sample Stdev: ", np.std(vals)
  pu.save_samples(samples, os.getcwd(), mType)

def testCont(ys, sample, burn, lag):
  pals = [0,0.5,1,2,5,10,20]
  with open("contConvRT",'w') as f:
    for i1 in range(len(pals)):
      for i2 in range(len(pals))[i1:]:
        for i3 in range(len(pals))[i2:]:
          for i4 in range(len(pals))[i3:]:
            p1 = pals[i1]
            p2 = pals[i2]
            p3 = pals[i3]
            p4 = pals[i4]
            v = make_church_prime_ripl()
            timeStart = time.time()
            rest = 100.0 - (p1 + p2 + p3 + p4)
            if p1 > 0:
              v.assume("d1", "(uniform_continuous 0 " + str(p1) + ")")
            else:
              v.assume("d1", "0")
            if p2 > 0:
              v.assume("d2", "(uniform_continuous 0 " + str(p2) + ")")
            else:
              v.assume("d2", "0")
            if p3 > 0:
              v.assume("d3", "(uniform_continuous 0 " + str(p3) + ")")
            else:
              v.assume("d3", "0")
            if p4 > 0:
              v.assume("d4", "(uniform_continuous 0 " + str(p4) + ")")
            else:
              v.assume("d4", "0")
            v.assume("r", "(uniform_continuous 2 " + str(rest) + ")")
            v.assume("d", "(+ d1 d2 d3 d4 r)")#(uniform_continuous 0 5) (uniform_continuous 0 1))")
            v.assume("y", "(lambda () (student_t d))")

            [v.observe("(y)", str(ys[i])) for i in range(len(ys))]
            samples = pu.posterior_samples(v, "d", no_samples=sample, no_burns=burn, int_mh=lag)
            vals = map(lambda x:x[1], samples)
            print rest, p1, p2, p3, p4, np.mean(vals), np.var(vals), time.time() - timeStart
            f.write("==== " + str((rest,p1,p2,p3,p4,time.time() - timeStart)) + "\n" + str(vals))
            f.flush()

def dispContPerf():
  times = {}
  with open("contRT",'r') as f:
    for line in f:
      r, d1, d2, d3, d4, mean, time = map(float, line.strip()[1:-1].split(', '))
      times[time] = (r,d1,d2,d3,d4,mean)
  print '\n'.join(map(str,sorted(times.items())))

def repeatCont(ys, sample, burn, lag):
  with open("contRepeats10F",'w') as f:
    means = []
    times = []
    for j in range(10):
      v = make_church_prime_ripl()
      timeStart = time.time()
      v.assume("d1", "(uniform_continuous 0 99.9978)")
      v.assume("d2", "(uniform_continuous 0 0.001)")
      v.assume("d3", "(uniform_continuous 0 0.001)")
      v.assume("d4", "(uniform_continuous 0 0.0001)")
      v.assume("d5", "(uniform_continuous 0 0.0001)")
      v.assume("d", "(+ d1 d2 d3 d4 d5)")
      v.assume("y", "(lambda () (student_t d))")

      [v.observe("(y)", str(ys[i])) for i in range(len(ys))]
      samples = pu.posterior_samples(v, "d", no_samples=sample, no_burns=burn, int_mh=lag)
      vals = map(lambda x:x[1], samples)
      means.append(np.mean(vals))
      times.append(time.time() - timeStart)
      print j, means[-1], times[-1]
      f.write(str((j, means[-1], times[-1])) + "\n")
      f.flush()
    f.write(str(("Total: ", np.mean(means), np.mean(times))))

def checkDistRuntime(tp = 0):
  times = []
  xs = []
  for d in np.logspace(-7,2,100):
    v = make_church_prime_ripl()
    timeStart = time.time()
    if tp == 0:
      v.assume("y", "(student_t %f)" % d)
    elif tp == 1:
      v.assume("y", "(uniform_continuous 0 %f)" % d)
    samples = pu.posterior_samples(v, "y", no_samples=100, no_burns=0, int_mh=1, silent=True)
    vals = map(lambda x:x[1], samples)
    times.append(time.time() - timeStart)
    xs.append(d)
    print d, np.mean(vals), times[-1]

  print xs
  print times

  plt.plot(xs,times, '-x')

  if tp == 0:
    plt.xlabel("Degrees of freedom")
    plt.ylabel("Seconds")
    plt.title("Speed of generating 100 unconditioned student_t samples")
    plt.ylim([0,max(max(times),1)])
  elif tp == 1:
    plt.xlabel("Upper limit of interval (x)")
    plt.ylabel("Seconds")
    plt.title("Speed of generating 100 unconditioned (uniform_continuous 0 x) samples")
    plt.xscale("log")
    plt.ylim([0,max(max(times),1)])
  plt.show()

def runtimeVarObs(ys):
  conts = []
  cont5vars = []
  yrs = []
  for yr in range(0,1001,10):
    yrs.append(yr)
    v = make_church_prime_ripl()
    conts.append(runModel(v, ys[:yr], "cont", 100, 0, 1, True, True))
    v = make_church_prime_ripl()
    cont5vars.append(runModel(v, ys[:yr], "cont5var", 100, 0, 1, True, True))
    print yrs[-1], conts[-1], cont5vars[-1]

  print yrs
  print conts
  print cont5vars

  p1, = plt.plot(yrs, conts, '-xb')
  p2, = plt.plot(yrs, cont5vars, '-dr')
  plt.xlabel("Number of observations")
  plt.ylabel("Seconds")
  plt.title("Runtime vs. number of observations conditioned on")
  plt.legend([p1,p2],["cont","cont5var"])
  plt.show()

def testConv(ys):
  lens = []
  for x in range(100):
    v = make_church_prime_ripl()
    #timeStart = time.time()

    v.assume("f1", "(* 0.5 (uniform_discrete 0 2))")
    v.assume("f2", "(* 0.25 (uniform_discrete 0 2))")
    v.assume("f3", "(* 0.125 (uniform_discrete 0 2))")
    v.assume("f4", "(* 0.0625 (uniform_discrete 0 2))")
    v.assume("f5", "(* 0.03125 (uniform_discrete 0 2))")
    v.assume("f6", "(* 0.015625 (uniform_discrete 0 2))")
    v.assume("f7", "(* 0.0078125 (uniform_discrete 0 2))")
    v.assume("r", "(uniform_continuous 0 0.0078125)")
    v.assume("d", "(+ 2 (* 98 (+ f1 f2 f3 f4 f5 f6 f7 r)))")

    #v.assume("d1", "(uniform_continuous 0 9)")
    #v.assume("d2", "(uniform_continuous 2 89)")
    #v.assume("d", "(+ d1 d2)") #
    #v.assume("d", "(uniform_continuous 2 100)") 
    v.assume("y", "(lambda () (student_t d))")

    [v.observe("(y)", str(ys[i])) for i in range(len(ys))]
    samples = pu.posterior_samples_conv(v, "d", conv = 4.214, eps=0.5, silent=True)
    vals = map(lambda x:x[1], samples)
    #print vals
    lens.append(len(samples))
    print x, len(samples)
    #print len(vals), np.mean(vals), np.var(vals), time.time() - timeStart
    #print '\n'.join(map(str,vals))
  print lens
  print "5", np.mean(lens)

pd = None
fac=None
def getLL(ys,d,dof=4):
  global pd
  global fac
  if dof == 4:
    if pd == None:
      print "loading Dict"
      with open("posteriorDict",'r') as f:
        fac, pd = cPickle.load(f)
    return pd[round(d*fac)]
  elif dof == 21:
    if pd == None:
      print "loading Dict"
      with open("posteriorDict21",'r') as f:
        fac, pd = cPickle.load(f)
    return pd[round(d*fac)]
  else:
    return cp.logLiks(ys, d, base=2)

  
def simConv(ys, dof=4):
  lens = []
  eps = 0.1
  if dof == 4:
    mode = 4.214
  elif dof == 21:
    mode = 11.5
  ds = [98]
  
  for i in range(1000):
    ss = []
    for d in ds:
      ss.append(random.random()*d)

    samples = [sum(ss) + 2]
    curLL = getLL(ys,samples[-1],dof)

    count = 0
    while abs(samples[-1] - mode) > eps:
      ind = random.randrange(len(ss))
      ps = random.random()*ds[ind]

      prop = sum(ss) - ss[ind] + ps + 2
      propLL = getLL(ys,prop,dof) 

      if propLL >= curLL:
        samples.append(prop)
        curLL = propLL
        ss[ind] = ps
      else:
        accProb = 2**(propLL - curLL)
        if random.random() < accProb:
          samples.append(prop)
          curLL = propLL
          ss[ind] = ps
        else:
          samples.append(samples[-1])
      count += 1

    if i % 100 == 0:
      print i, len(samples)
    lens.append(len(samples))
  #print '\n'.join(map(str,samples))
  print np.mean(lens), np.var(lens)

def simConvSearch(ys):
  eps = 0.5
  mode = 4.214
  with open("posteriorDict",'r') as f:
    fac, pd = cPickle.load(f)

  pals = [0,0.5,1,2,5,7,10,15,20,25,30,35,40,45]
  with open("modeTimeSim05",'w') as f:
    for i1 in range(len(pals)):
      for i2 in range(len(pals))[i1:]:
        for i3 in range(len(pals))[i2:]:
          for i4 in range(len(pals))[i3:]:
            if pals[i1] + pals[i2] + pals[i3] + pals[i4] > 98:
              continue
            ds = [d for d in [pals[i1], pals[i2], pals[i3], pals[i4], 98 - pals[i1] - pals[i2] - pals[i3] - pals[i4]] if d != 0]
            lens = []
            for i in range(1000):
              ss = []
              for d in ds:
                ss.append(random.random()*d)

              samples = [sum(ss) + 2]
              curLL = pd[round(samples[-1]*fac)] # cp.logLiks(ys,samples[-1], base=2)

              count = 0
              while abs(samples[-1] - mode) > eps:
                ind = random.randrange(len(ss))
                ps = random.random()*ds[ind]
                prop = sum(ss) - ss[ind] + ps + 2
                propLL = pd[round(prop*fac)]
                if propLL >= curLL:
                  samples.append(prop)
                  curLL = propLL
                  ss[ind] = ps
                else:
                  accProb = 2**(propLL - curLL)
                  if random.random() < accProb:
                    samples.append(prop)
                    curLL = propLL
                    ss[ind] = ps
                  else:
                    samples.append(samples[-1])
                count += 1
                #if count % 100 == 0:
                #  print "Considered", count, "samples"

              #if i % 100 == 0:
              #  print i, len(samples)
              lens.append(len(samples))
            #print '\n'.join(map(str,samples))
            print ds, np.mean(lens), np.var(lens), len(lens)
            f.write(str((ds, np.mean(lens), np.var(lens))) + "\n")
            f.flush()
            #plt.hist(lens,100)
            #plt.show()

def estConvBin(eps):
  p = float(2 * eps) / 98
  negp = 1-p
  probs = []
  curNeg = 1
  for i in range(1,70000):
    probs.append(curNeg * p)
    curNeg *= negp

  norm = sum(probs)
  probs = [p / norm for p in probs]
    

  pMean = sum([i*probs[i] for i in range(len(probs))])
  stdDev = math.sqrt(sum([i*i*probs[i] for i in range(len(probs))]) - pMean*pMean)
  plot1, = plt.plot(probs)
  plt.legend([plot1, plot1],["Mean: " + str(pMean)[:5], "Std Dev: " + str(stdDev)[:5]])

  plt.show()

def irwinHall():
  probs = []
  for n in range(1,101):
    x = 4.214*n/98
    s = 0
    for k in range(n+1):
      s += (-1)**k * scipy.misc.comb(n,k) * (x-k)**(n-1) * math.copysign(1,x-k)
    s = s/(2 * math.factorial(n-1))
    probs.append(s)
  print '\n'.join(map(str, probs))

def procModeTimes(fn):
  lens = {}
  with open(fn,'r') as f:
    for line in f:
      tp, mean, var = line.strip()[1:-1].rsplit(',',2)
      mean, var = float(mean), float(var)
      lens[tp] = mean
  print '\n'.join(map(str, sorted(lens.items(), key=lambda x:x[1])[:10]))

def simMix(ys):
  mixs = []
  mode = 4.214
  iters = 1000
  ds = [98]
  with open("posteriorDict",'r') as f:
    fac, pd = cPickle.load(f)

  for i in range(1000):
    ss = []
    for d in ds:
      ss.append((mode-2)*(d/98.0))

    samples = [sum(ss) + 2]
    assert(samples[-1] == mode)
    curLL = pd[round(samples[-1]*fac)] # cp.logLiks(ys,samples[-1], base=2)

    count = 0
    for it in range(iters):
      ind = random.randrange(len(ss))
      ps = random.random()*ds[ind]

      prop = sum(ss) - ss[ind] + ps + 2

      propLL = pd[round(prop*fac)]
      if propLL >= curLL:
        samples.append(prop)
        curLL = propLL
        ss[ind] = ps
      else:
        accProb = 2**(propLL - curLL)
        if random.random() < accProb:
          samples.append(prop)
          curLL = propLL
          ss[ind] = ps
        else:
          samples.append(samples[-1])
      count += 1

    if i % 100 == 0:
      print i, len(samples)
    mixs.append(getMix(samples))
  #print '\n'.join(map(str,samples))
  print np.mean(mixs), np.var(mixs)

def simMixSearch(ys):
  mode = 4.214
  iters = 1000
  with open("posteriorDict",'r') as f:
    fac, pd = cPickle.load(f)

  pals = [0,0.5,1,2,5,7,10,15,20,25,30,35,40,45]
  with open("modeMixSim",'w') as f:
    for i1 in range(len(pals)):
      for i2 in range(len(pals))[i1:]:
        for i3 in range(len(pals))[i2:]:
          for i4 in range(len(pals))[i3:]:
            if pals[i1] + pals[i2] + pals[i3] + pals[i4] > 98:
              continue
            ds = [d for d in [pals[i1], pals[i2], pals[i3], pals[i4], 98 - pals[i1] - pals[i2] - pals[i3] - pals[i4]] if d != 0]
            mixs = []
            for i in range(100):
              ss = []
              for d in ds:
                ss.append((mode-2)*(d/98.0))

              samples = [sum(ss) + 2]
              assert(samples[-1] == mode)
              curLL = pd[round(samples[-1]*fac)] # cp.logLiks(ys,samples[-1], base=2)

              count = 0
              for it in range(iters):
                ind = random.randrange(len(ss))
                ps = random.random()*ds[ind]

                prop = sum(ss) - ss[ind] + ps + 2
                propLL = pd[round(prop*fac)]
                if propLL >= curLL:
                  samples.append(prop)
                  curLL = propLL
                  ss[ind] = ps
                else:
                  accProb = 2**(propLL - curLL)
                  if random.random() < accProb:
                    samples.append(prop)
                    curLL = propLL
                    ss[ind] = ps
                  else:
                    samples.append(samples[-1])
                count += 1
                #if count % 100 == 0:
                #  print "Considered", count, "samples"

              #if i % 100 == 0:
              #  print i, len(samples)
              mixs.append(getMix(samples))
            #print '\n'.join(map(str,samples))
            print ds, np.mean(mixs), np.var(mixs), len(mixs)
            f.write(str((ds, np.mean(mixs), np.var(mixs))) + "\n")
            f.flush()
            #plt.hist(lens,100)
            #plt.show()

def getMix(samples):
  return sum([abs(samples[i]-samples[i+1]) for i in range(len(samples)-1)])

def calcExpJump():
  mode = 4.214
  with open("posteriorDict",'r') as f:
    fac, pd = cPickle.load(f)

  modeLL = pd[int(mode*fac)]
  expJump = 0
  norm = 0
  for jump in np.arange(2,100.01,0.01):
    prob = 2**(pd[int(jump*fac)] - modeLL)
    expJump += prob*abs(jump-mode)
    norm += 1

  expJump /= norm
  print expJump

def simConvBin(ys, depth, eps, dof=4):
  lens = []
  if dof == 4:
    mode = 4.214
  elif dof == 21:
    mode = 11.5

  rest = 2**(-depth)
  ds = []
  for d in range(1, depth+1):
    ds.append(2**(-d))

  for i in range(1000):
    ss = []
    for d in ds:
      ss.append((random.random() >= 0.5)*d)

    ss.append(random.random()*rest)
    samples = [sum(ss)*98 + 2]
    curLL = getLL(ys,samples[-1],dof)
    count = 0
    while abs(samples[-1] - mode) > eps:
      ind = random.randrange(len(ss))
      if ind < len(ds):
        ps = (random.random() >= 0.5) *ds[ind]
        #ps = (not ss[ind]) *ds[ind]
      else:
        ps = random.random() * rest

      prop = (sum(ss) - ss[ind] + ps)*98 + 2
      propLL = getLL(ys,prop,dof) 

      if propLL >= curLL:
        samples.append(prop)
        curLL = propLL
        ss[ind] = ps
      else:
        accProb = 2**(propLL - curLL)
        if random.random() < accProb:
          samples.append(prop)
          curLL = propLL
          ss[ind] = ps
        else:
          samples.append(samples[-1])
      count += 1

    #if i % 1000 == 0:
    #  print i, len(samples)
    lens.append(len(samples))
  #print '\n'.join(map(str,samples))
  print eps, depth, np.mean(lens), np.var(lens)

def distToInt(sample, start, end):
  if sample > end:
    return sample - start
  elif sample < start:
    return start - sample
  else:
    return 0

def binConvInt(depths, start, end): # start,end in [0,1]
  for depth in depths:
    lens = []
    rest = 2**(-depth)
    ds = []
    for d in range(1, depth+1):
      ds.append(2**(-d))

    #print ds, rest
    for i in range(100):
      ss = []
      for d in ds:
        ss.append((random.random() >= 0.5)*d)

      ss.append(random.random()*rest)
      samples = [sum(ss)]
      partSamples = [[samples[-1]] + list(ss)]
      curDist = distToInt(samples[-1],start,end)
      while curDist > 0:
        ind = random.randrange(len(ss))
        #assert( len(ss) == len(ds) + 1)
        if ind < len(ds):
          ps = (random.random() >= 0.5) *ds[ind]
        else:
          ps = random.random() * rest

        prop = (sum(ss) - ss[ind] + ps)
        propDist = distToInt(prop,start,end) 

        if propDist < curDist:
          samples.append(prop)
          curDist = propDist
          ss[ind] = ps
        else:
          samples.append(samples[-1])
    
        partSamples.append([samples[-1]] + list(ss))
        if len(samples) > 1000:
          print '\n'.join(map(str,partSamples))
          assert(False)

      #if i % 1000 == 0:
      #  print i, len(samples)
      lens.append(len(samples))
    #print '\n'.join(map(str,samples))
    print start, end, depth, np.mean(lens), np.var(lens)

def binConvInt1(depths, start, end, stuckProb=0.5): # start,end in [0,1]
  for depth in depths:
    lens = []
    rest = 2**(-depth)
    ds = []
    minStuckBV = (end - start)
    intervals = []
    prevBV = 1
    for d in range(1, depth+1):
      bv = 2**(-d)
      ds.append(bv)
      if bv > minStuckBV:
        intervals.append([(bt+0.5*bv,bt+1.5*bv) for bt in np.arange(0, 1, prevBV)])
      prevBV = bv

    #print minStuckBV, stuckProb, intervals
    #print depth, intervals
    #print ds, rest
    for i in range(1000):
      ss = []
      for d in ds:
        ss.append((random.random() >= 0.5)*d)

      ss.append(random.random()*rest)
      samples = [sum(ss)]
      partSamples = [[samples[-1]] + list(ss)]
      curDist = distToInt(samples[-1],start,end)
      while curDist > 0:
        curPosStuck = inIntervals(samples[-1], intervals)
        #print samples[-1], curPosStuck
        if len(curPosStuck) == 0 or random.random() > stuckProb:
          ind = random.randrange(len(ss))
          #assert( len(ss) == len(ds) + 1)
          if ind < len(ds):
            ps = (random.random() >= 0.5) *ds[ind]
          else:
            ps = random.random() * rest
          prop = (sum(ss) - ss[ind] + ps)
          propDist = distToInt(prop,start,end) 

          if propDist < curDist:
            samples.append(prop)
            curDist = propDist
            ss[ind] = ps
          else:
            samples.append(samples[-1])
        else:
          ind = random.choice(curPosStuck)
          ps = random.random() * (ds[ind] / 2)
          if ss[ind] == 0:
            prop = sum(ss[:ind]) + ds[ind] + ps
          else:
            prop = sum(ss[:ind]) + ds[ind] - ps

          propDist = distToInt(prop,start,end) 
          if propDist < curDist:
            samples.append(prop)
            curDist = propDist
            ss = binaryExp(prop, depth, ds)
            #print prop, ss
          else:
            samples.append(samples[-1])

        partSamples.append([samples[-1]] + list(ss))
        if len(samples) > 100000:
          #print '\n'.join(map(str,partSamples))
          assert(False)

      #if i % 1000 == 0:
      #  print i, len(samples)
      lens.append(len(samples))
    #print '\n'.join(map(str,samples))
    print stuckProb, start, end, depth, np.mean(lens), np.var(lens)

def binaryExp(no, depth, ds = None):
  if ds == None:
    ds = []
    for d in range(1, depth+1):
      bv = 2**(-d)
      ds.append(bv)

  ss = []
  for d in ds:
    if d < no:
      ss.append(d)
      no -= d
    else:
      ss.append(0)

  ss.append(no)
  return ss

def inIntervals(no, intervals):
  ind = []
  for i in range(len(intervals)):
    for (s,e) in intervals[i]:
      if no >= s and no <= e:
        ind.append(i)
        break
  return ind

if __name__ == "__main__":
  ys = pu.readData("../tdfData")
  #calcExpJump()
  #procModeTimes("modeTimeSim05")
  #simMixSearch(ys)
  #testConv(ys)
  for prob in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    binConvInt1(range(20),0.049,0.05, prob)

  #for eps in [1]:#,0.25,0.1,0.01]:
  #  for d in range(0,10):
  #    simConvBin(ys, d, eps, 4)
  
  #estConvBin(0.01)
  #irwinHall()
  #simMix(ys)

  #runtimeVarObs(ys)

  #v = make_church_prime_ripl()
  #modelType = sys.argv[1]
  #runModel(v, ys, modelType, 10000, 0, 1, timeTest = False, silentSamp = False)
  
  #testCont(ys,1000,0,1)
  #dispContPerf()
  #repeatCont(ys,1000,200,10)
  """
  v.assume("d1", "(uniform_continuous 0 50)")
  v.assume("d2", "(uniform_continuous 0 25)")
  v.assume("d3", "(uniform_continuous 0 15)")
  v.assume("d4", "(uniform_continuous 0 8)")
  v.assume("d5", "(uniform_continuous 0 1.89)")
  v.assume("d6", "(uniform_continuous 0 0.1)")
  v.assume("d7", "(uniform_continuous 0 0.01)")
  v.assume("d", "(+ d1 d2 d3 d4 d5 d6 d7)")
  v.assume("y", "(lambda () (student_t d))")
  [v.observe("(y)", str(ys[i])) for i in range(len(ys))]
  samples = pu.posterior_samples(v, "d", no_samples=1000, no_burns=0, int_mh=1)
  print '\n'.join(map(lambda x: str(x[1]), samples))
  """
  #checkDistRuntime(1)
  #v.assume("y", "(student_t 20000000)")
  #samples = pu.posterior_samples(v, "y", no_samples=100, no_burns=0, int_mh=1, silent=True)
  #print samples

