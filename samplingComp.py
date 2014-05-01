import sys
sys.path.append('./tdf/Venture')

import calcPost as cp
import ppUtils as pu
import cPickle
import random
import numpy as np
import scipy.stats
from matplotlib import pyplot as plt

pd = None
fac=None
maxPd = None
def getLL(ys, d, dof=4, fn="tdf/Venture/posteriorDict"):
  global pd
  global fac
  global maxPd
  if dof == 4:
    if pd == None:
      #print "loading Dict"
      with open(fn,'r') as f:
        fac, pd = cPickle.load(f)
      maxPd = max(pd.values())
    try:
      return pd[round(d*fac)]
    except:
      return float("-inf")
  elif dof == 21:
    if pd == None:
      #print "loading Dict"
      with open(fn + "21",'r') as f:
        fac, pd = cPickle.load(f)
      maxPd = max(pd.values())
    try:
      return pd[round(d*fac)]
    except:
      return float("-inf")
  else:
    return cp.logLiks(ys, d, base=2)

def metropolis(ll, prop, stop): # assume proposal distribution is symmetric
  samples = [prop(None)]
  curLL = ll(samples[-1])

  while not stop(samples):
    curProp = prop(samples[-1])
    propLL = ll(curProp) 

    if propLL >= curLL:
      samples.append(curProp)
      curLL = propLL
    else:
      accProb = 2**(propLL - curLL)
      if random.random() < accProb:
        samples.append(curProp)
        curLL = propLL
      else:
        samples.append(samples[-1])

  return samples

def metropolisTdf(ys, iters, eps, dof=4):
  if dof == 4:
    mode = 4.214
  elif dof == 21:
    mode = 11.5

  ll = lambda sample: getLL(ys, sample, dof)
  prop = lambda _: 2 + random.random()*98
  stop = lambda samples: samples[-1] >= mode-eps and samples[-1] < mode+eps

  lens = []
  for i in range(iters):
    samples = metropolis(ll, prop, stop)
    lens.append(len(samples))
    #print samples
    
  return np.mean(lens), np.std(lens)

def metropolisMixTdf(ys, sampleNo, dof=4):
  if dof == 4:
    mode = 4.214
  elif dof == 21:
    mode = 11.5

  ll = lambda sample: getLL(ys, sample, dof)
  prop = lambda s: 2 + random.random()*98 if s else mode
  stop = lambda samples: len(samples) >= sampleNo

  return metropolis(ll, prop, stop)

def metropolisTdfSim(iters):
  mode = 4.214
  eps = 0.25
  ll = lambda sample: -1 * abs(sample - mode)
  prop = lambda _: random.random() * 100
  stop = lambda samples: samples[-1] >= mode-eps and samples[-1] < mode+eps

  lens = []
  for i in range(iters):
    lens.append(len(metropolis(ll, prop, stop)))
  print np.mean(lens), np.std(lens)

def sliceSampling(lik, x, w, stop): # assume proposal distribution is symmetric
  samples = [x]

  rejected = 0
  while not stop(samples, rejected):
    y = random.uniform(0, lik(samples[-1]))

    r = random.random()
    xl = samples[-1] - r*w
    xr = samples[-1] + (1-r)*w
    while lik(xl) > y:
      xl -= w
    while lik(xr) > y:
      xr += w

    #print samples[-1], y, xl, xr
    while True:
      prop = random.uniform(xl, xr)
      propLik = lik(prop)
      if propLik > y:
        samples.append(prop)
        break
      else:
        rejected += 1
        if prop > samples[-1]:
          xr = prop
        else:
          xl = prop
  
  return samples, rejected

def sliceSamplingTdf(ys, iters, eps, w = 1, dof=4):
  if dof == 4:
    mode = 4.214
  elif dof == 21:
    mode = 11.5

  ll = lambda sample: 2**(getLL(ys, sample, dof) - maxPd)
  stop = lambda samples, _: samples[-1] >= mode-eps and samples[-1] < mode+eps

  lens = []
  for i in range(iters):
    init = 2 + random.random()*98
    samples, _ = sliceSampling(ll, init, w, stop)
    lens.append(len(samples))
    #print samples

  return np.mean(lens), np.std(lens)

def sliceSamplingMixTdf(ys, sampleNo, w = 1, dof=4):
  if dof == 4:
    mode = 4.214
  elif dof == 21:
    mode = 11.5

  ll = lambda sample: 2**(getLL(ys, sample, dof) - maxPd)
  stop = lambda samples, rej: len(samples) + rej >= sampleNo

  return sliceSampling(ll, mode, w, stop)

def sliceSamplingGauss(iters, eps, gauss, w = 1):
  ll = lambda sample: gauss.pdf(sample)
  stop = lambda samples, _: samples[-1] >= gauss.mean()-eps and samples[-1] < gauss.mean()+eps

  lens = []
  rejs = []
  for i in range(iters):
    init = random.random() * 100
    samples, rej = sliceSampling(ll, init, w, stop)
    rejs.append(rej)
    lens.append(len(samples))
    #print i

  return np.mean(lens), np.std(lens), np.mean(rejs)

def plotLogs(fn = "samplingLogs"):
  dicLogs = {}
  with open("samplingTests/" + fn, 'r') as f:
    logs = f.read().strip().split("===")
    for log in logs:
      rows = log.strip().split('\n')
      tp = rows[0].strip()
      if len(tp) > 0:
        dicLogs[tp] = ([],[])
        for row in rows[1:]:
          eps, mode, var = map(float, row.strip().split())
          dicLogs[tp][0].append(eps * 2)
          dicLogs[tp][1].append(mode)

  daxs = {}
  for title,log in dicLogs.items():
    ax, = plt.plot(log[0],log[1])
    daxs[title] = ax

  axs = [ax for _, ax in sorted(daxs.items())]
  titles = [title for title, _ in sorted(daxs.items())]

  plt.legend(axs,titles)
  plt.yscale("log")
  plt.xlabel("Size of target neighbourhood")
  plt.ylabel("Number of samples")
  plt.title("Avg. no of samples to neighbourhood of mode") 
  plt.show()

def plotSimpleLog(fn, xl, yl, ti):
  xs = []
  ys = []
  zs = []
  with open("samplingTests/" + fn, 'r') as f:
    for row in f:
      key, mode, var, rej = map(float, row.strip().split())
      xs.append(key)
      ys.append(mode)
      zs.append(rej)

  plt.plot(xs,zs)
  #plt.ylim([0,20])
  plt.xscale("log")
  plt.xlabel(xl)
  plt.ylabel(yl)
  plt.title(ti)
  plt.show()

def plotGauss(mean, std):
  n = scipy.stats.norm(mean,std)
  xs = []
  ys = []
  for s in np.arange(0,100.001,0.1):
    xs.append(s)
    ys.append(n.pdf(s))
  plt.plot(xs,ys)
  plt.xlabel("x")
  plt.ylabel("likelihood")
  plt.title("Likelihood function with mean " + str(mean) + " and stDev " + str(std))
  plt.show()

def printSamples(samples):
  for s in range(len(samples)):
    print s, samples[s]

if __name__ == "__main__":
  ys = pu.readData("tdf/tdfData")
  iters = 100
  dof = 4

  #printSamples(sliceSamplingMixTdf(ys, 10000)[0])
  #printSamples(metropolisMixTdf(ys, 10000))
  #for eps in [0.05]: #np.arange(1,0,-0.01):
  #  print eps, metropolisTdf(ys, iters, eps, dof)
  #  print eps, sliceSamplingTdf(ys, iters, eps, dof)

  #for std in range(2,21):
  #  print std, sliceSamplingGauss(iters, 0.5, scipy.stats.norm(50,std))

  #for mean in range(1,100):
  #  print mean, sliceSamplingGauss(iters, 0.5, scipy.stats.norm(mean,3))

  #for w in np.logspace(-2,2,50):
  #  print w, sliceSamplingGauss(iters, 0.5, scipy.stats.norm(50,2), w)
  
  plotSimpleLog("slicePerfWidth", "Slice sampling initial width", "Number of rejected proposals", "Avg. rejected proposals when generating 100 samples for Normal(50,2)")

  #plotGauss(50,20)

  #sliceSamplingTdf(ys, iters, eps, dof)
