from venture.shortcuts import *
import sys
sys.path.append("/home/haggis/Desktop/ModelTesting/Venture/")
from postSamp import posterior_samples
import numpy as np
from collections import Counter

def genTdf(v):
  v.assume("y", "(student_t 4)")
  samples = posterior_samples(v, "y", no_samples=1000, int_mh=100)
  print np.mean(samples), np.std(samples)

def readData():
  with open('../t-dfdata.txt','r') as f:
    raw = f.read()
    ys = map(float, raw.split('\n',2)[-1][:-2].translate(None,'\n\r').split(','))
  return ys

def contModel(v, ys):
  v.assume("d", "(uniform_continuous 2 100)")
  v.assume("y", "(lambda () (student_t d))")

  [v.observe("(y)", str(ys[i])) for i in range(len(ys))]

  samples = posterior_samples(v, "d", no_samples=1000, int_mh=1)
  print np.mean(samples), np.std(samples)

def disc1Model(v, ys):
  v.assume("d", "(uniform_discrete 2 50)")
  v.assume("y", "(lambda () (student_t d))")

  [v.observe("(y)", str(ys[i])) for i in range(len(ys))]

  samples = posterior_samples(v, "d", no_samples=1000, int_mh=1)
  print Counter(samples)
  print np.mean(samples), np.std(samples)

def disc2Model(v, ys):
  v.assume("d", "(uniform_discrete 20 60)")
  v.assume("y", "(lambda () (student_t (/ d 10)))")

  [v.observe("(y)", str(ys[i])) for i in range(len(ys))]

  samples = posterior_samples(v, "d", no_samples=1000, int_mh=1)
  print Counter(samples)
  print np.mean(samples), np.std(samples)


if __name__ == "__main__":
  v = make_church_prime_ripl()
  ys = readData()
  disc1Model(v, ys)


