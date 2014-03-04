from venture.shortcuts import *
import sys
sys.path.append("../../")
import ppUtils as pu
from collections import Counter
import numpy as np
import os
import time
from matplotlib import pyplot as plt

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
  #pu.save_samples(samples, os.getcwd(), mType)

def testCont(ys, sample, burn, lag):
  pals = [0,0.0001,0.001,0.01,0.1,1,10]
  with open("contRTsss",'w') as f:
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
            v.assume("r", "(uniform_continuous 0 " + str(rest) + ")")
            v.assume("d", "(+ d1 d2 d3 d4 r)")#(uniform_continuous 0 5) (uniform_continuous 0 1))")
            v.assume("y", "(lambda () (student_t d))")

            [v.observe("(y)", str(ys[i])) for i in range(len(ys))]
            samples = pu.posterior_samples(v, "d", no_samples=sample, no_burns=burn, int_mh=lag)
            vals = map(lambda x:x[1], samples)
            print rest, p1, p2, p3, p4, np.mean(vals), time.time() - timeStart
            f.write(str((rest,p1,p2,p3,p4,np.mean(vals),time.time() - timeStart)) + "\n")
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

if __name__ == "__main__":
  ys = pu.readData("tdf")
  #runtimeVarObs(ys)
  v = make_church_prime_ripl()
  #modelType = sys.argv[1]
  #print runModel(v, ys, modelType, 1000, 0, 1, True, True)
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
  v.assume("y", "(student_t 20000000)")
  samples = pu.posterior_samples(v, "y", no_samples=100, no_burns=0, int_mh=1, silent=True)
  #print samples

