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

def runModel(v, ys, mType, sample, burn, lag, timeTest = False):
  if mType == "cont":
    v.assume("d1", "(uniform_continuous 0 99.9978)")
    v.assume("d2", "(uniform_continuous 0 0.001)")
    v.assume("d3", "(uniform_continuous 0 0.001)")
    v.assume("d4", "(uniform_continuous 0 0.0001)")
    v.assume("d5", "(uniform_continuous 0 0.0001)")
    v.assume("d", "(+ d1 d2 d3 d4 d5)")#(uniform_continuous 0 5) (uniform_continuous 0 1))")
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
  samples = pu.posterior_samples(v, "d", no_samples=sample, no_burns=burn, int_mh=lag)
  print '\n'.join([str(samples[i][1]) for i in range(len(samples)) if i%10==0])
  #v.infer(11000000)
  if timeTest:
    return
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

if __name__ == "__main__":
  ys = pu.readData("tdf")
  v = make_church_prime_ripl()
  #modelType = sys.argv[1]
  #runModel(v, ys, modelType, 1000, 200, 1, False)
  #testCont(ys,1000,0,1)
  #dispContPerf()
  #repeatCont(ys,1000,200,10)
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


