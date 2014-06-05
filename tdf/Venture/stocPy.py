import inspect
import math
import scipy.stats as ss
from matplotlib import pyplot as plt
import random
import copy
import time
import numpy as np
import cPickle
from scipy import interpolate
from scipy.interpolate import interp1d
import ppUtils as pu

startTime = time.time()
curNames = set()
ll = 0
llFresh = 0
llStale = 0
db = {}
erps = {"unifCont":0, "studentT":1, "poisson":2, "normal":3, "invGamma":4}
dists = [ss.uniform, ss.t, ss.poisson, ss.norm, ss.invgamma]
observ = set()
condition = set()
cols = ['b','r','g','k','m','c']

debugOn = False
def debugPrint(override, *mess):
  if debugOn or override:
    print ' '.join(map(str, mess))

def resetAll():
  global startTime
  global curNames
  global ll
  global llFresh
  global llStale
  global db
  global observ
  global condition

  startTime = time.time()
  curNames = set()
  ll = 0
  llFresh = 0
  llStale = 0
  db = {}
  observ = set()
  condition = set()

def initModel(model):
  model(True)
  while math.isnan(ll):
    resetAll()
    model(True)

def getSamples(model, noSamps, discAll=False, alg="met", thresh=0.1):
  initModel(model)
  if alg == "met":
    sampleList = [metropolisSampleTrace(model, no = n+1, discAll = discAll) for n in range(noSamps)]
  elif alg == "slice":
    sampleList = dict([(n+1, sliceSampleTrace(model, no = n+1, discAll = discAll)) for n in range(noSamps)])
  elif alg == "sliceNoTrans":
    sampleList = dict([(n+1, sliceSampleTrace(model, no = n+1, discAll = discAll, allowTransJumps = False)) for n in range(noSamps)])
  elif alg == "sliceMet":
    sampleList = [sliceMetMixSampleTrace(model, no = n+1, discAll = discAll) for n in range(noSamps)]
  else:
    raise Exception("Unknown inference algorithm: " + str(alg))

  """
  #print "sampleList", sampleList
  totTries = 0
  for k,l in sorted(tries.items()):
    totTries += len(l)
    print k, len(l), l[0]
  print "TotTries", totTries
  """
  resetAll()
  return aggSamples(sampleList)

def getSamplesByLL(model, noLLs, discAll=False, alg="met", thresh=0.1):
  initModel(model)
  totLLs = 0
  sampleDic = {}
  if alg == "met":
    while totLLs < noLLs:
      samp = metropolisSampleTrace(model, no = totLLs, discAll = discAll)
      totLLs += 1
      if totLLs < noLLs:
        sampleDic[totLLs] = samp
  elif alg == "slice":
    while totLLs < noLLs:
      llCount, samp = sliceSampleTrace(model, no = totLLs, discAll = discAll, countLLs = True)
      totLLs += llCount
      if totLLs < noLLs:
        sampleDic[totLLs] = samp
  elif alg == "sliceNoTrans":
    while totLLs < noLLs:
      llCount, samp = sliceSampleTrace(model, no = totLLs, discAll = discAll, allowTransJumps = False, countLLs = True)
      totLLs += llCount
      if totLLs < noLLs:
        sampleDic[totLLs] = samp
  elif alg == "sliceMet":
    while totLLs < noLLs:
      llCount, samp = sliceMetMixSampleTrace(model, no = totLLs, discAll = discAll, countLLs = True, thresh = thresh)
      totLLs += llCount
      if totLLs < noLLs:
        sampleDic[totLLs] = samp
  else:
    raise Exception("Unknown inference algorithm: " + str(alg))

  resetAll()
  print "rejectedTransJumps", rejTransJumps
  return aggSamples(sampleDic)

def getTimedSamples(model, maxTime, discAll=False):
  initModel(model)
  sampleList = []
  while time.time() - startTime < maxTime:
    sampleList.append(sliceMetMixSampleTrace(model, no = len(sampleList)+1, discAll = discAll))

  resetAll()
  return aggSamples(sampleList)

def aggSamples(samples):
  aggSamps = {}
  if isinstance(samples, list):
    for sample in samples:
      for k,v in sample.items():
        try:
          aggSamps[k].append(v)
        except:
          aggSamps[k] = [v]
  else:
    for count, sample in samples.items():
      for k,v in sample.items():
        try:
          aggSamps[k][count] = v
        except:
          aggSamps[k] = {count:v}

  return aggSamps

def sliceMetMixSampleTrace(model, no = None, discAll = False, thresh = 0.1, countLLs = False):
  if random.random() > thresh:
    return sliceSampleTrace(model, no = no, discAll = discAll, allowTransJumps = False, countLLs = countLLs)
  else:
    samp = metropolisSampleTrace(model, no = no, discAll = discAll)
    if countLLs:
      return 1, samp
    else:
      return samp

tries = {}
def metropolisSampleTrace(model, no = None, discAll = False):
  global db
  global ll
  global tries
  unCond = list(set(db.keys()).difference(condition))
  #print unCond, db
  n = random.choice(unCond)
  #print db
  otp, ox, ol, ops = db[n]

  x = dists[otp].rvs(*ops)
  try:
    l = dists[otp].logpdf(x, *ops)
  except:
    l = dists[otp].logpmf(x, *ops)

  odb = copy.copy(db)
  oll = ll
  recalcLL(model, n, x, l)
  #if llStale != 0:
  #  print "lls", ll, llFresh, llStale
  """
  if x == 0 and n == "branching-19-0":
    prob = math.e**(ll - oll)
    if math.isnan(prob):
      print n, x, ox, l, ll, oll, math.e**(ll - oll)
    else:
      p2 = db['branching-23-0'][1]
      try:
        tries[(ox,p2)].append((math.e**(ll - oll), ll, oll))
      except:
        tries[(ox,p2)] = [(math.e**(ll - oll), ll, oll)]
  """
    #print n, x, ox, l, ll, oll, math.e**(ll - oll),
  #print ox, x, math.e**(ll - oll)
  #print "before", db
  changed = True
  #print math.log(len(odb)), math.log(len(db)), llStale, llFresh
  acc = ll - oll + ol - l + math.log(len(unCond)) - math.log(len(set(db.keys()).difference(condition))) + llStale - llFresh
  if math.log(random.random()) < acc:
    pass
  else:
    changed = False
    db = odb
    ll = oll

  #print "after", db
  sample = {}
  #print observ
  for n in observ:
    sample[n] = db[n][1]
  #print changed, n, x, sample[n]
  #if x == 0 and changed:
  #  print changed, sample
  #print changed, db
  if no % 10000 == 0:
    print no, sample, time.time() - startTime
  return sample

rejTransJumps = 0
def sliceSampleTrace(model, width = 10, no = None, discAll=False, allowTransJumps = True, countLLs = False):
  global db
  global ll
  global rejTransJumps
  #print db
  llCount = 0
  unCond = list(set(db.keys()).difference(condition))
  n = random.choice(unCond)
  otp, ox, ol, ops = db[n]
  
  u = -1*ss.expon.rvs(-1*ll) #sample log(x), x~unif(0,likelihood)
  r = random.random()
  oll = ll

  xl = ox
  xr = ox
  debugPrint(False, ox, ll, u)
  curWidth = r*width

  assert(ll > u)
  llc = ll
  while llc > u:
    xl -= curWidth
    if discAll:
      xl = int(math.floor(xl))

    odb = copy.copy(db)
    recalcLL(model, n, xl)
    llCount += 1
    db = odb
    curWidth *= 2
    llc = ll + llStale - llFresh
    debugPrint(False, "l", xl, ll)

  ll = oll
  curWidth = r*width
  llc = ll
  while llc > u:
    xr += curWidth
    if discAll:
      xr = int(math.ceil(xr))

    odb = copy.copy(db)
    recalcLL(model, n, xr)
    llCount += 1
    db = odb
    curWidth *= 2
    llc = ll + llStale - llFresh
    debugPrint(False, "r", xr, ll)

  ll = oll
  first = True
  
  transJump = False
  llc = ll
  while first or llc < u or math.isnan(llc) or (transJump and (not allowTransJumps)):
    if first:
      first = False
    if discAll:
      x = random.randrange(xl,xr+1)
    else:
      x = random.uniform(xl, xr)

    #print xl, xr, x
    odb = copy.copy(db)
    recalcLL(model, n, x)
    llCount += 1
    transJump = (llStale != 0) or (llFresh != 0) 
    llc = ll + llStale - llFresh
    #u = -1*ss.expon.rvs(-1*(oll - llStale))# +math.log(len(db)) - math.log(len(odb))))
    debugPrint(False, "c", xl, xr, x, ll)
    if llc < u or math.isnan(llc) or (transJump and (not allowTransJumps)):
      debugPrint(False, "in")
      if transJump and (not allowTransJumps):
        rejTransJumps += 1
      db = odb
      if x > ox:
        xr = x
      else:
        xl = x

  sample = {}
  for o in observ:
    sample[o] = db[o][1]

  #if no%10000 == 0:
  #print no, n, xl, xr, sample[o]

  if countLLs:
    return llCount, sample
  else:
    return sample

def sliceMultSampleTrace(model, width = 100, no = None, discAll=False):
  global db
  global ll

  unCond = list(set(db.keys()).difference(condition))

  u = -1*ss.expon.rvs(-1*ll) #sample log(x), x~unif(0,likelihood)

  xls = {}
  xrs = {}
  oxs = {}
  for n in unCond:
    #print n
    r = random.random()
    oll = ll 
    _, ox, _, _ = db[n]
    oxs[n] = ox
    xl = ox
    xr = ox
    debugPrint(False, ox, ll, u)
    curWidth = r*width
    assert(ll > u)
    while ll > u:
      xl -= curWidth
      if discAll:
        xl = int(math.floor(xl))

      odb = copy.copy(db)
      recalcLL(model, n, xl)
      db = odb
      curWidth *= 2
      debugPrint(False, "l", xl, ll)

    ll = oll
    curWidth = r*width
    while ll > u:
      xr += curWidth
      if discAll:
        xr = int(math.ceil(xr))

      odb = copy.copy(db)
      recalcLL(model, n, xr)
      db = odb
      curWidth *= 2
      debugPrint(False, "r", xr, ll)
    xls[n] = xl
    xrs[n] = xr
    ll = oll

  ll = oll
  first = True
  while first or ll < u or math.isnan(ll):
    if first:
      first = False
    xs = {}
    if discAll:
      for n in unCond:
        xs[n] = random.randrange(xl,xr+1)
    else:
      for n in unCond:
        xs[n] = random.uniform(xl, xr)

    #print xl, xr, x
    odb = copy.copy(db)
    recalcMultLL(model, xs)
    debugPrint(False, "c", xl, xr, xs, ll)
    if ll < u or math.isnan(ll):
      debugPrint(False, "in")
      db = odb
      for n in unCond:
        if xs[n] > oxs[n]:
          xrs[n] = xs[n]
        else:
          xls[n] = xs[n]

  sample = {}
  for n in observ:
    sample[n] = db[n][1]

  if no%1000 == 0:
    print no, sample[n]
  return sample

def recalcLL(model, n, x, l = None):
  if l:
    ls = {n:l}
  else:
    ls = None
  recalcMultLL(model, {n:x}, ls)

def recalcMultLL(model, xs, ls = None):
  global db
  global ll
  global llStale
  global llFresh
  global curNames

  for n,x in xs.items():
    otp, ox, ol, ops = db[n]

    if not ls:
      try:
        l = dists[otp].logpdf(x, *ops)
      except:
        l = dists[otp].logpmf(x, *ops)
    else:
      l = ls[n]
    db[n] =  (otp, x, l, ops)
    if l == float("-inf"):    # TODO: Check handling of this is correct
      ll = float("-inf")
      llFresh = 0
      llStale = 0
      return
  #print x, db
  ll = 0
  llFresh = 0
  llStale = 0
  curNames = set()
  oldLen = len(db)
  model()

  newLen = len(db)
  assert(oldLen <= newLen)
  #if len(db) != len(curNames):
  #print len(db), len(curNames)

  for n in db.keys():
    if not n in curNames:
      llStale += db[n][2]
      db.pop(n)

"""
def getName(loopInd):
  global curNames
  _, _, lineNo, funcName, _, _ = inspect.stack()[1]
  name = funcName + "-" + str(lineNo) + "-" + str(loopInd)
  curNames.add(name)
  return name
"""

def getExplicitName(funcName, lineNo, loopInd):
  global curNames
  name = funcName + "-" + str(lineNo) + "-" + str(loopInd)
  curNames.add(name)
  return name

def getERP(n, c, tp, ps):
  global ll
  global llFresh
  global db
  global condition
  otp, ox, ol, ops = db.get(n, (None,None,None,None))
  if tp == otp:
    if ps == ops:
      ll += ol
      return ox
    else:
      try:
        l = dists[tp].logpdf(ox, *ps)
      except:
        l = dists[tp].logpmf(ox, *ps)

      #if math.isnan(l):
      #  print "jjjj", tp, ps, ops, ox
      db[n] = (tp, ox, l, ps)
      ll += l
      #print n, ops, ol, ps, l
      #if n in condition:
      #  print n, tp, otp, ps, ops
      #print db
      return ox
  else:
    if c:
      x = c
      condition.add(n)
    else:
      assert (not n in condition)
      x = dists[tp].rvs(*ps)
    try:
      l = dists[tp].logpdf(x, *ps)
    except:
      l = dists[tp].logpmf(x, *ps)

    db[n] = (tp, x, l, ps)
    ll += l
    if not c:
      llFresh += l
    return x
"""
def getERP1(n, c, tp, ps):
  global ll
  global llFresh
  global db
  global condition
  otp, ox, ol, ops = db.get(n, (None,None,None,None))

  if c:
    x = c
    condition.add(n)
  elif n in condition:
    x = ox
  else:
    assert (not n in condition)
    x = dists[tp].rvs(*ps)
  try:
    l = dists[tp].logpdf(x, *ps)
  except:
    l = dists[tp].logpmf(x, *ps)

  db[n] = (tp, x, l, ps)
  ll += l
  if not tp == otp and not c:
    llFresh += l
  return x
"""
def unifCont(start, end, name, cond = None, obs=False):
  global observ
  if obs:
    observ.add(name)
  return getERP(name, cond, erps["unifCont"], (start,end-start))

def studentT(dof, name, cond = None, obs=False):
  global observ
  if obs:
    observ.add(name)
  return getERP(name, cond, erps["studentT"], (dof,))

def poisson(shape, name, cond = None, obs=False):
  global observ
  if obs:
    observ.add(name)
  return getERP(name, cond, erps["poisson"], (shape,))

def normal(mean, stDev, name, cond = None, obs=False):
  global observ
  if obs:
    observ.add(name)
  return getERP(name, cond, erps["normal"], (mean, stDev))

def invGamma(shape, scale, name, cond = None, obs=False):
  global observ
  if obs:
    observ.add(name)
  return getERP(name, cond, erps["invGamma"], (shape, 0, scale))

def plotTestDist(b):
  samples = []
  for i in range(100000):
    samples.append(math.log(random.uniform(0,b)))
  print min(samples), max(samples)
  plt.hist(samples, 100)
  plt.show()

  samples = []
  for i in range(100000):
    samples.append(-1*ss.expon.rvs(-1*math.log(b)))
  plt.hist(samples, 100)
  plt.show()
  print min(samples), max(samples)

def readSamps(fn, start=0):
  with open(fn, 'r') as f:
    data = f.read().strip().split('\n')
    if len(data) == 1: # list format
      samps = map(float, data[0][1:-1].split(','))[start:]
    else: # rows of (count, sample) format
      samps = []
      for line in data:
        if len(line.strip().split()) == 2:
          count, samp = map(float, line.strip().split())
          samps.append(samp)
  return samps

def plotSampDist(fn, start=0):
  samps = readSamps(fn)
  plt.hist(samps, 100)
  plt.xlabel("DoFs")
  plt.ylabel("No. samples")
  plt.title("Metropolis ran for 10m - " + str(len(samps)) + " samples - mean: " + str(np.mean(samps))[:5] + ", stDev: " + str(np.std(samps))[:5])
  plt.show()

def plotCumSampDist(fn, plot=True, show=True):
  samps = readSamps(fn)
  hSamps, ds =  np.histogram(samps, 1000)
  cSamps = []
  curSum = 0
  norm = float(sum(hSamps))
  for samp in hSamps:
    curSum += samp / norm
    cSamps.append(curSum)

  locs = []
  for d in range(len(ds)-1):
    locs.append((ds[d] + ds[d+1]) / 2.0)

  cSamps = [0] + cSamps + [1]
  locs = [2] + locs + [40]

  if plot:
    plt.plot(locs, cSamps)
    plt.xlabel("DoFs")
    plt.ylabel("Number of samples with smaller DoF")
    if show:
      #plt.title("Venture ran for 10m - " + str(len(samps)) + " samples - mean: " + str(np.mean(samps))[:5] + ", stDev: " + str(np.std(samps))[:5])
      plt.show()
  return locs, cSamps

def plotCumPost(fn = "Posterior4", plot = True, show=True, zipRez=True):
  with open(fn,'r') as f:
    ds, ls = cPickle.load(f)
  cls = []
  curSum = 0
  for l in ls:
    curSum += l
    cls.append(curSum)

  if plot:
    plt.plot(ds, cls)
    plt.xlabel("DoFs")
    plt.ylabel("Number of samples with smaller DoF")

    if show:
      plt.show()
  if zipRez:
    return zip(ds,cls)
  else:
    return ds, cls

def plotCumSampDists(fns):
  for fn in fns:
    plotCumSampDist(fn, show=False)
  plt.show()

def calcKSTestDiff(fns, pfn):
  post = plotCumPost(pfn, plot=False)
  print post
  diffs = {}
  ps = []
  ns = []
  for fn in fns:
    diffs[fn] = []
    xs, ys =  plotCumSampDist(fn, plot=False)
    f = interpolate.interp1d(xs,ys)
    for x,y in post:
      diffs[fn].append(abs(f(x) - y))
    diffs[fn] = sorted(diffs[fn], reverse=True)

    p, = plt.plot(diffs[fn])
    ps.append(p)
    ns.append(fn[:-7])

  plt.legend(ps,ns)
  plt.xscale("log")
  #plt.yscale("log")
  plt.xlabel("Nth biggest difference")
  plt.ylabel("Difference from posterior")
  plt.title("Decreasing differences from posterior")
  plt.show()

def calcKSRun(run, postFun, aggFreq, burnIn = 0):
  samps = []
  xs = []
  ys = []
  curAgg = 0
  for k, samp in sorted(run.items()):
    if k < burnIn:
      continue
    samps.append(samp)
    ind = k-burnIn
    if ind > aggFreq[curAgg]:
      curAgg += 1
      ksDiff = calcKSDiff(postFun, samps)
      #print len(samps), len(run)
      xs.append(ind)
      ys.append(ksDiff)
      """
      if ind > 1000 and ksDiff > 0.9:
        print ksDiff
        print samps
        pu.plotSamples(samps)
      """

  xs.append(ind)
  ys.append(calcKSDiff(postFun, samps))
  return xs, ys

def calcKSTests(fns, pfn, aggFreq, burnIn = 0, plot=True, xlim = 200000, names=None):
  postFun = interpolate.interp1d(*plotCumPost(pfn, plot=False, zipRez=False))
  ps = []
  ns = []
  np.append(aggFreq, float("inf"))
  for i in range(len(fns)):
    fn = fns[i]
    p, = plt.plot([0],[0], cols[i])
    ps.append(p)
    if names:
      ns.append(names[i])
    else:
      ns.append(fn.split('/')[1])
    with open(fn, 'r') as f:
      runs = cPickle.load(f)
    for r in range(len(runs)):
      print fn, r
      xs, ys = calcKSRun(runs[r], postFun, aggFreq, burnIn)
      if plot:
        plt.plot(xs,ys, cols[i], alpha=0.25)

  if plot:
    plt.legend(ps,ns,loc=3)
    plt.xscale("log")
    plt.yscale("log", basey=2)
    plt.xlim([0, xlim])
    plt.xlabel("Number model simulations")
    plt.ylabel("KS difference from posterior")
    plt.title("Performance on Normal" + fns[0].split("PerLL")[0][-1] + " Model")
    plt.show()

def calcKSSumms(fns, pfn, aggFreq, burnIn = 0, xlim = 200000):
  postFun = interpolate.interp1d(*plotCumPost(pfn, plot=False, zipRez=False))
  ps = []
  ns = []
  np.append(aggFreq, float("inf"))
  funcs = []
  for i in range(len(fns)):
    fn = fns[i]
    p, = plt.plot([0],[0], cols[i])
    ps.append(p)
    ns.append(fn.split('/')[1])
    fs = []
    start = float("-inf")
    end = float("inf")
    with open(fn, 'r') as f:
      runs = cPickle.load(f)
    for r in range(len(runs)):
      print fn, r
      xs, ys = calcKSRun(runs[r], postFun, aggFreq, burnIn)
      if xs[0] > start:
        start = xs[0]
      if xs[-1] < end:
        end = xs[-1]
      fs.append(interp1d(xs,ys))

    end += 1

    top = []
    med = []
    bot = []

    for x in np.arange(start, end):
      if x % 1000 == 0:
        print "x", x
      vals = []
      for f in fs:
        vals.append(f(x))

      top.append(np.percentile(vals, 25))
      med.append(np.percentile(vals, 50))
      bot.append(np.percentile(vals, 75))

    plt.plot(range(start, end), med, cols[i])
    plt.plot(range(start, end), top, cols[i])
    plt.plot(range(start, end), bot, cols[i])

  plt.legend(ps,ns,loc=3)
  plt.xscale("log")
  plt.yscale("log", basey=2)
  plt.xlim([0, xlim])
  plt.xlabel("Number model simulations")
  plt.ylabel("KS difference from posterior")
  plt.title("Performance on Normal" + fns[0].split("PerLL")[0][-1] + " Model")
  plt.show()

cachedPost = {}
def calcKSDiff(postFun, samps):
  global cachedPost
  cumProb = 0
  inc = 1.0/len(samps)
  maxDiff = 0
  for samp in sorted(samps):
    try:
      post = cachedPost[samps]
    except:
      post = postFun(samp)
      cachedPost[samp] = post
    preDiff = abs(post - cumProb)
    cumProb += inc
    postDiff = abs(post - cumProb)
    maxDiff = max([maxDiff, preDiff, postDiff])
  return maxDiff

def calcKLCondTests(posts, sampList, test, freq = float("inf")):
  names = ["Posterior for r < 5", "Posterior for r >= 5"]
  axs = [None,None]
  for samps in sampList:
    #print samps[0], ind
    ind = test(samps)
    post = posts[ind]
    ax = calcKLTest(post, samps, freq, show=False, col=cols[ind])
    axs[ind] = ax

  #plt.xlim([0,10000])
  plt.ylabel("KL divergence")
  plt.xlabel("No. Model Simulations")
  plt.title("Conv. to Posteriors for Slice Sampling w/out Trans-Dimensional Jumps")
  plt.legend(axs, names)
  plt.show()

def calcKLTests(post, sampsLists, names, freq = float("inf"), burnIn = 0, xlim=None, alpha=0.5, show=True):
  axs = []
  for i in range(len(sampsLists)):  
    for samps in sampsLists[i]:
      ax = calcKLTest(post, samps, freq, show=False, col=cols[i], burnIn = burnIn, alpha=alpha)
    axs.append(ax)

  if xlim:
    plt.xlim([0,xlim])
  plt.ylabel("KL divergence", size=20)
  plt.xlabel("Number of MCMC Iterations", size=20)
  plt.title("Convergence to posterior", size=30)
  plt.legend(axs, names, prop={'size':20})
  if show:
    plt.show()

def calcKLTest(post, samps, freq = float("inf"), show=True, col=None, plot=True, cutOff = float("inf"), burnIn = 0, alpha=0.5):
  sampDic = {}
  if isinstance(samps, list):
    samps = dict([(i+1,samps[i]) for i in range(len(samps))])

  xs = []
  ys = []
  prevC = 0
  sampList = []
  for c, samp in sorted(samps.items()):
    if samp > cutOff or c < burnIn:
      continue
    c = c - burnIn
    sampList.append(samp)
    try:
      sampDic[samp] += 1
    except:
      sampDic[samp] = 1.0

    kl = getKLDiv(pu.norm(sampDic), post)
    xs.append(c)
    ys.append(kl)
    if (c+1) / freq > prevC:
      prevC += 1
      plt.hist(sampList, 100)
      plt.title(str(c) + " " + str(kl))
      #plt.show()
  if not plot:
    return xs, ys
  else:
    if col:
      ax, = plt.plot(xs,ys, col, alpha=alpha)
    else:
      ax, = plt.plot(xs,ys)
    plt.yscale("log", basey=2)
    plt.xscale("log")
    if show:
      plt.show()

    return ax

def getKLDiv(samp, post):
  eps = 0.0001
  assert (abs(sum(samp.values()) - 1) < eps)
  assert (abs(sum(post.values()) - 1) < eps)

  kl = 0
  for val, p in samp.items():
      kl += p * math.log(p / post[val])
  return kl

def calcKLSumms(post, sampsLists, names, burnIn = 0, xlim = None):
  axs = []
  for i in range(len(sampsLists)):
    axs.append(calcKLSumm(post, sampsLists[i], col=cols[i], show=False, burnIn = burnIn))

  if xlim:
    plt.xlim([0, xlim])
  plt.ylabel("KL divergence")
  plt.xlabel("No. Samples")
  plt.title("Convergence to posterior (quartiles)")
  plt.legend(axs, names)
  plt.show()

def calcKLSumm(post, sampsList, col = 'b', show=True, burnIn = 0):
  fs = []
  start = float("-inf")
  end = float("inf")

  for samps in sampsList:
    if len(fs) % 10 == 0:
      print "fs", len(fs)
    xs,ys = calcKLTest(post, samps, plot=False, burnIn = burnIn) #xs should be already sorted
    if xs[0] > start:
      start = xs[0]
    if xs[-1] < end:
      end = xs[-1]
    fs.append(interp1d(xs,ys))

  end += 1

  top = []
  med = []
  bot = []
  print start, end
  for x in range(start, end):
    if x % 1000 == 0:
      print "x", x
    vals = []
    for f in fs:
      vals.append(f(x))

    top.append(np.percentile(vals, 25))
    med.append(np.percentile(vals, 50))
    bot.append(np.percentile(vals, 75))

  plt.yscale("log", basey=2)
  plt.xscale("log")
  ax, = plt.plot(range(start, end), med, col)
  plt.plot(range(start, end), top, col)
  plt.plot(range(start, end), bot, col)
  
  if show:
    plt.show()

  return ax

if __name__ == "__main__":
  #data = [studentT(4,"p" + str(i)) for i in range(10000)]
  #print data, min(data), max(data)
  #plt.hist(data, 100)
  #plt.show()
  #plotTestDist(0.7)
  #plotSampDist("newMetTdfSamp600")
  #plotCumSampDists(["ventureTdfSamp600", "metTdfSamp600", "sliceTdfSamp600"])
  #calcKSTest(["tdf/ventureTdfSamp600"], "tdf/Posterior4")
  #calcKSTests(["normal/normal3PerLLSlicemet"], "normal/normal3Post", aggFreq=np.logspace(1,math.log(19998,10),10), burnIn=1000, plot=False)
  mi = "3"
  #calcKSTests(["normal/normal" + mi + "PerLLSliceV2"], "normal/normal" + mi + "Post", aggFreq=np.logspace(1,math.log(200000,10),10), burnIn=1000, xlim = 200000)
  calcKSSumms(["normal/normal" + mi + "PerLLMet","normal/normal" + mi + "PerLLSlice", "normal/normal" + mi + "PerLLSliceV1", "normal/normal" + mi + "PerLLSlicemet0.1", "normal/normal" + mi + "PerLLSlicemet0.5"], "normal/normal" + mi + "Post", aggFreq=np.logspace(1,math.log(200000,10),10), burnIn=1000, xlim = 200000)
