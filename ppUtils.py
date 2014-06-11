#from venture.shortcuts import *
import numpy as np
import sys
from matplotlib import pyplot as plt
from itertools import groupby
import matplotlib
import time

langs = ["Venture", "Bugs"]

def readData(path, sz = None):
  with open(path,'r') as f:
    raw = f.read()
    ys = map(float, raw.split('\n',2)[-1].strip()[:-2].translate(None,'\n\r').split(','))
  if sz:
    return reducePrec(ys, sz)
  return ys

def posterior_samples(v, var_name,no_samples,no_burns,int_mh, silent=False):
    s=[];
    v.infer(no_burns)
    if not silent:
      print "Burned", no_burns, "samples"

    counter = no_burns
    for sample in range(no_samples):
        v.infer(int_mh)
        counter += int_mh
        label = var_name + str(np.random.randint(10**5))+str(sample)
        v.predict(var_name,label)
        s.append((counter,v.report(label)))
        if sample % 100 == 0 and not silent:
          print "Collected", sample, "samples"
    return s

def posterior_samples_timed(v, var_name, maxTime, no_burns, int_mh, silent=False):
    startTime = time.time()
    s=[];
    v.infer(no_burns)
    if not silent:
      print "Burned", no_burns, "samples"

    counter = no_burns
    sample = 0
    while time.time() - startTime < maxTime:
        v.infer(int_mh)
        counter += int_mh
        label = var_name + str(np.random.randint(10**5))+str(sample)
        v.predict(var_name,label)
        s.append((counter,v.report(label)))
        if sample % 100 == 0 and not silent:
          print "Collected", sample, "samples", time.time() - startTime
        sample += 1
    return s

def posterior_samples_conv(v, var_name, conv, eps = 0.5, repeat=1, int_mh=1, silent=False):
    s=[];

    counter = 0
    inConv = 0
    sample = 0
    while inConv < repeat:
        v.infer(int_mh)
        counter += int_mh
        label = var_name + str(np.random.randint(10**5))+str(sample)
        v.predict(var_name,label)
        val = v.report(label)
        s.append((counter,val))
        if sample % 100 == 0 and not silent:
          print "Collected", sample, "samples"
        sample += 1
        if abs(val-conv) < eps:
          inConv += 1
        elif inConv > 0:
          inConv = 0
          print "+++++ Jumped out of conv range" 
    return s

def posterior_samples_from_conv(v, var_name, conv, eps = 0.5, extra=1000, int_mh=1, silent=False):
    s1=[]

    counter = 0
    sample = 0
    while True:
        v.infer(int_mh)
        counter += int_mh
        label = var_name + str(np.random.randint(10**5))+str(sample)
        v.predict(var_name,label)
        val = v.report(label)
        s1.append((counter,val))
        if sample % 100 == 0 and not silent:
          print "Collected", sample, "samples"
        sample += 1
        if abs(val-conv) < eps:
          break
    print "Reached mode"
    s2 = []
    counter = 0
    for sample in range(no_samples):
        v.infer(int_mh)
        counter += int_mh
        label = var_name + str(np.random.randint(10**5))+str(sample)
        v.predict(var_name,label)
        s2.append((counter,v.report(label)))
        if sample % 100 == 0 and not silent:
          print "Collected", sample, "samples"
    return s1,s2

def save_samples(samples, path, model):
  fn = path + "/" + model + "Samples"
  with open(fn, 'w') as f:
    f.write('\n'.join([str(s[0]) + " " + str(s[1]) for s in samples]))

def readSamples1(model, mType, getNaive = False):
  samples = {}
  times = {}
  lastInds = []
  facs = {"Venture":1, "Bugs":1}
  for lang in langs:
    fn = model + "/" + lang + "/" + mType + "Samples"
    if getNaive and lang == "Venture":
      fn += "Naive"
    try:
      with open(fn,'r') as f:
        vals = f.read().strip().split('\n')
        time = float(vals[-1])
        vals = map(lambda x: x.split(), vals[:-1])
        samples[lang] = [float(x[1])/ facs[lang] for x in vals]
        times[lang] = time
        inds = [float(x[0])  for x in vals]
        if not (lastInds == [] or lastInds == inds): 
          print "Warning: Indices do not match in all samples"
        lastInds = inds
    except:
      print "Error opening file", fn

  return times, samples

def norm(vals):
  if isinstance(vals, list):
    norm = sum(vals)
    return map(lambda x: x/norm, vals)
  elif isinstance(vals, dict):
    norm = sum(vals.values())
    nvals = {}
    for k,v in vals.items():
      nvals[k] = v/norm
    return nvals
  else:
    raise Exception("Unknown datatype given to norm:" + vals)

def getStats(samples):
  return dict([(k, np.mean(v)) for k,v in samples.items()]), dict([(k, np.std(v)) for k,v in samples.items()])

def showDists(samples, times, bins, sTitle):
  means, sds = getStats(samples)
  f, axs = plt.subplots(len(samples.keys()), sharex=True)
  count = 0
  for lang in samples.keys():
    ax = axs[count]
    ax.hist(samples[lang], bins, align = 'left', rwidth = 0.8)
    ax.set_title(lang + ";  m:" + str(means[lang])[:5] + ",  sd:" + str(sds[lang])[:5] + ",  rt:" + str(times[lang]) + "s", size=20)
    plt.setp(ax.get_xticklabels(), visible=True)
    count += 1
    ax.set_xlabel("Degrees of freedom", size=17)
    ax.set_ylabel("Sample frequency", size=17)
  f.suptitle(sTitle, size=30)
  f.tight_layout()
  plt.subplots_adjust(top=0.85)
  plt.show()

def showMixDists(samples, bins, sTitle):
  f, axs = plt.subplots(len(samples.keys()), sharex=True)
  count = 0
  for lang in samples.keys():
    mix = [len(list(g)) for k,g in groupby(samples[lang])]
    ax = axs[count]
    ax.hist(mix,bins, align='left', rwidth=0.8)
    ax.set_title(lang + " stretches of consecutive, identical, samples", size=20)
    plt.setp(ax.get_xticklabels(), visible=True)
    ax.set_xlabel("Run length", size=17)
    ax.set_ylabel("Run frequency", size=17)
    count += 1
  f.subplots_adjust(hspace=0.3)
  f.suptitle(sTitle, size=30)
  f.tight_layout()
  plt.subplots_adjust(top=0.85)
  plt.show()

def reducePrec(data, sz):
  return [float(str(x)[:sz]) for x in data]

def showPerfStats(fn):
  times = {}
  sInds = []
  pInds = []
  tInds = range(4)
  with open(fn,'r') as f:
    for line in f:
      samps, points, tp, time = map(float, line.strip().split())
      if samps not in sInds:
        sInds.append(samps)
      if (points + 0.1) not in pInds:
        pInds.append(points + 0.1)
      try:
        times[tp][samps].append(time)
      except:
        try:
          times[tp][samps] = [time]
        except:
            times[tp] = {samps : [time]}

  gCols = len(times.keys())
  gRows = len(times[times.keys()[0]].keys())
  fig, axs = plt.subplots(nrows=gRows, ncols=gCols, sharex=True, sharey=True)

  types = ["Direct", "Loop", "Loop+Pred", "Loop+Pred+Rep"]
  sInds.reverse()
  for gr in range(gRows):
    for gc in range(gCols):
      cax = axs[gr][gc]
      cax.plot(pInds, times[tInds[gc]][sInds[gr]], '-d')
      cax.set_yscale('log')
      cax.set_xscale('log')
      cax.set_ylim([0.01,1000])
      cax.set_xlim([1,1000])
      if gc == 0:
        cax.set_ylabel(str(int(sInds[gr])) + " Samples")
      if gr == gRows - 1:
        cax.set_xlabel("No. datapoints\n" + types[tInds[gc]])

  fig.subplots_adjust(left=0.05, bottom = 0.1, right = 0.95, top = 0.95, hspace=0.1, wspace=0.1)
  plt.show()
      
"""
  times = {}
  sInds = []
  pInds = []
  tInds = range(4)
  with open(fn,'r') as f:
    for line in f:
      samps, points, tp, time = map(float, line.strip().split())
      if samps not in sInds:
        sInds.append(samps)
      if points not in pInds:
        pInds.append(points)
      try:
        times[tp][points].append(time)
      except:
        try:
          times[tp][points] = [time]
        except:
            times[tp] = {points : [time]}

  gCols = len(times.keys())
  gRows = len(times[times.keys()[0]].keys())
  fig, axs = plt.subplots(nrows=gRows, ncols=gCols, sharex=True, sharey=True)

  types = ["Direct", "Loop", "Loop+Pred", "Loop+Pred+Rep"]
  pInds.reverse()
  for gr in range(gRows):
    for gc in range(gCols):
      cax = axs[gr][gc]
      cax.plot(sInds, times[tInds[gc]][pInds[gr]], '-d')
      cax.set_yscale('log')
      cax.set_xscale('log')
      cax.set_ylim([0.01,1000])
      if gc == 0:
        cax.set_ylabel(str(int(pInds[gr])) + " Datapts")
      if gr == gRows - 1:
        cax.set_xlabel("No. Samples\n" + types[tInds[gc]])

  fig.subplots_adjust(left=0.05, bottom = 0.1, right = 0.95, top = 0.95, hspace=0.1, wspace=0.1)
  plt.show()
"""

def readSamples(fn):
  samples = []
  jumps = []
  with open(fn,'r') as f:
    for line in f:
      if len(line.split()) == 1:
        s = float(line.strip())
      else:
        s = float(line.strip().split()[1])
      if not samples == []:
        jumps.append(s - samples[-1])
      samples.append(s)
  return samples, jumps

def dispSamples(fn, tp, burn=0):
  samples, jumps = readSamples(fn)

  samples = samples[burn:-1]
  fig, ax = plt.subplots()
  if "2" in fn:
    start, end = 5, 40
  else:
    start, end = 2.5, 6.5

  ax.hist(samples, bins = np.arange(start, end, 0.025))
  ax.set_title(tp + " sample distribution", size=30)
  #ax.set_xscale("log")
  #ax.set_xticks(range(1,10))
  #ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
  ax.set_xlim([start,end])
  #ax.set_ylim([0,250])
  ax.set_xlabel("Degrees of freedom", size=20)
  ax.set_ylabel("Sample Frequency", size=20)
  plt.show()

  print max(map(abs,jumps)), min([x for x in map(abs,jumps) if x > 0])
  plt.hist(map(lambda x: abs(x),jumps),bins=np.logspace(-9, 2, 300))
  plt.title(tp + " non-zero Jump dist", size=30)
  plt.xscale('log')
  #plt.yscale('log')
  plt.xlabel("Jump length", size=20)
  plt.ylabel("Number of jumps out of 10000", size=20)
  #plt.xlim(0.00000001,100)
  plt.show()

def plotConsSamps(fn, tp, burn=0):
  samps,_ = readSamples(fn)

  print samps
  plt.plot(samps[burn:-1])
  plt.title(tp + " sample evolution", size=30)
  plt.xlabel("Iteration", size=20)
  plt.ylabel("Sampled DoF", size=20)
  plt.yscale('log')
  plt.ylim([1,100])
  plt.show()

def autocorrSamps(fn, tp, burn=0):
  samples, _ = readSamples(fn)
  samples = samples[burn:-1]
  #ac = np.correlate(samples, samples, mode='same')
  n = len(samples)
  var = np.var(samples, ddof=0)
  samples = samples - np.mean(samples)
  ac = np.correlate(samples, samples, mode='full')[-n:]
  assert np.allclose(ac, np.array([(samples[:n-k]*samples[-(n-k):]).sum() for k in range(n)]))
  nac = ac / (var * n)
  plt.plot(nac)
  plt.title(tp + " autocorrelation", size=30)
  plt.ylabel("Autocorrelation", size=20)
  plt.xlabel("Sample distance", size=20)
  plt.show()

def dispModeTimes(fn):
  times = {}
  with open(fn, 'r') as f:
    for line in f:
      if len(line.strip()) == 0:
        continue
      name, dst, val = map(lambda x: x.strip(), line.split(" ",2))
      val = map(lambda x: int(x.strip()), val[1:-1].split(','))
      times[name] = (dst, val)

  for name,(dst, vals) in times.items():
    plt.hist(vals,200)
    plt.title(name, size=30)
    plt.xlabel("No. samples needed to reach mode +/- " + dst)
    plt.ylabel("Frequency of chain length")
    plot1, = plt.plot(0,0,"b")
    plt.legend([plot1, plot1],["Mean: " + str(np.mean(vals))[:5], "Std Dev: " + str(np.std(vals))[:5]])
    plt.show()

def dispExpStuck(fn):
  with open("tdf/Venture/" + fn, 'r') as f:
    data = map(lambda x: map(float, x.split()), f.read().split('\n'))[:-1]

  lines = {}
  targetInt = (data[0][1], data[0][2])
  for row in data:
    x =  row[3]
    y = row[4]
    try:
      lines[row[0]][0].append(x)
      lines[row[0]][1].append(y)
    except:
      lines[row[0]] = ([x],[y])

  legs = []
  acc = [0.1,0.3,0.5,0.7,0.9]
  for k in acc:
    (xs,ys) = lines[k]
    leg, = plt.plot(xs,ys)
    legs.append(leg)
  plt.legend(legs, map(str, acc))
  plt.xlabel("Binomial depth")
  plt.ylabel("Mean chain length")
  plt.title("Burn-in time to target interval " + str(targetInt))
  plt.show()

def dispShifted(fn):
  with open("tdf/Venture/" + fn, 'r') as f:
    data = map(lambda x: map(float, x.split()), f.read().split('\n'))[:-1]

  depths = {}
  for row in data:
    try:
      depths[row[2]].append(row[3])
    except:
      depths[row[2]] = [row[3]]

  bp = plt.boxplot(depths)
  print [line.get_xydata()[1] for line in bp["medians"]]
  plt.xlabel("Bit decomposition depth", size=20)
  plt.ylabel("Mean chain length", size=20)
  plt.title("Avg. burn-in time to 0.001 interval", size=30)
  plt.xticks(range(1, 21), range(20))
  plt.show()

if __name__ == "__main__":
  title = "Tdf21 - Bin3"
  #times, samples = readSamples1(sys.argv[1], sys.argv[2])
  #showDists(samples, times, np.arange(2.5, 6.5, 0.5), title) # np.arange(3.5, 6.5, 0.25)
  #showMixDists(samples, np.arange(1, 10, 1), title)
  #print readData("PP_Models/tdf/tdf")
  #print readData("PP_Models/tdf/tdf", 4)
  #showPerfStats("tdf/Venture/rtStats")
  #dispShifted("shiftMin0001")
  """
  fn = "custTdfSamps" #"tdf/Venture/flipSamples"
  dispSamples(fn)
  plotConsSamps(fn)
  autocorrSamps(fn)
  """

  #plotConsSamps("samplingTests/metropolisMix", "Metropolis")
  #plotConsSamps("samplingTests/sliceMixLik", "Slice Sampling")

  #autocorrSamps("samplingTests/metropolisMix", "Metropolis")
  #autocorrSamps("samplingTests/sliceMixLik", "Slice Sampling")

  #dispSamples("samplingTests/metropolisMix", "Metropolis")
  #dispSamples("samplingTests/sliceMixLik", "Slice Sampling")

  #fn = "tdf/Venture/cont21Samples"
  #fn = "tdf/Venture/contMix21Samples"
  fn = "tdf/Venture/21contBin3Samples"
  #plotConsSamps(fn, title)
  #autocorrSamps(fn, title)
  dispSamples(fn, title)
  #dispModeTimes("tdf/Venture/modeTime")
