from venture.shortcuts import *
import sys
sys.path.append("../../")
import ppUtils as pu
from collections import Counter
import numpy as np
import os

def genTdf(v):
  v.assume("y", "(student_t 4)")
  samples = pu.posterior_samples(v, "y", no_samples=1000, no_burns=0, int_mh=1)
  #pu.save_samples(samples, os.getcwd(), "gen")

def runModel(v, ys, mType, sample, burn, lag, timeTest = False):
  if mType == "cont":
    v.assume("d", "(uniform_continuous 2 100)")
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
  if timeTest:
    return
  vals = map(lambda x:x[1], samples)
  print "Sample mean: ", np.mean(vals), " Sample Stdev: ", np.std(vals)
  pu.save_samples(samples, os.getcwd(), mType)

if __name__ == "__main__":
  v = make_church_prime_ripl()
  modelType = sys.argv[1]
  ys = pu.readData("tdf")
  runModel(v, ys, modelType, 1000, 0, 1, True)


