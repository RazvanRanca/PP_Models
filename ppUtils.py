from venture.shortcuts import *
import numpy as np
import sys
from matplotlib import pyplot as plt
from itertools import groupby

langs = ["Venture", "Bugs"]

def readData(model, sz = None):
  with open("../" + model + "Data",'r') as f:
    raw = f.read()
    ys = map(float, raw.split('\n',2)[-1][:-2].translate(None,'\n\r').split(','))
  if sz:
    return reducePrec(ys, sz)
  return ys

def posterior_samples(v, var_name,no_samples,no_burns,int_mh):
    s=[];
    v.infer(no_burns)
    print "Burned", no_burns, "samples"

    counter = no_burns
    for sample in range(no_samples):
        v.infer(int_mh)
        counter += int_mh
        label = var_name + str(np.random.randint(10**5))+str(sample)
        v.predict(var_name,label)
        s.append((counter,v.report(label)))
        if sample % 100 == 0:
          print "Collected", sample, "samples"
    return s

def save_samples(samples, path, model):
  fn = path + "/" + model + "Samples"
  with open(fn, 'w') as f:
    f.write('\n'.join([str(s[0]) + " " + str(s[1]) for s in samples]))

def readSamples(model, mType):
  samples = {}
  times = {}
  lastInds = []
  facs = {"Venture":1, "Bugs":1}
  for lang in langs:
    fn = model + "/" + lang + "/" + mType + "Samples"
    #if lang == "Venture":
    #  fn += "Naive"
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

def getStats(samples):
  return dict([(k, np.mean(v)) for k,v in samples.items()]), dict([(k, np.std(v)) for k,v in samples.items()])

def showDists(samples, times, bins, sTitle):
  means, sds = getStats(samples)
  f, axs = plt.subplots(len(samples.keys()), sharex=True, sharey=True)
  count = 0
  for lang in samples.keys():
    ax = axs[count]
    ax.hist(samples[lang], bins, normed=True, align = 'left')
    ax.set_title(lang + ",  m:" + str(means[lang])[:5] + ",  sd:" + str(sds[lang])[:5] + ",  rt:" + str(times[lang]) + "s", size=20)
    plt.setp(ax.get_xticklabels(), visible=True)
    count += 1
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
    ax.hist(mix,bins, normed=True)
    ax.set_title(lang + " stretches of consecutive, identical, samples", size=20)
    plt.setp(ax.get_xticklabels(), visible=True)
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

def dispSamples(fn):
  samples = []
  jumps = []
  with open(fn,'r') as f:
    for line in f:
      s = float(line.strip())
      if not samples == []:
        jumps.append(s - samples[-1])
      samples.append(s)
  plt.hist(samples,400, normed=True)
  plt.title("tdf 5var Sample dist")
  plt.show()
  plt.hist(jumps,400, normed=True)
  plt.title("tdf 5var Jump dist")
  plt.show()

if __name__ == "__main__":
  #title = "Model: " + sys.argv[1].title() + "-" + sys.argv[2].title()
  #times, samples = readSamples(sys.argv[1], sys.argv[2])
  #showDists(samples, times, np.arange(3.5, 6.5, 0.05), title) # np.arange(3.5, 6.5, 0.25)
  #showMixDists(samples, np.arange(1, 30, 1), title)
  #print readData("PP_Models/tdf/tdf")
  #print readData("PP_Models/tdf/tdf", 4)
  #showPerfStats("tdf/Venture/rtStats")
  dispSamples("tdf/Venture/tdf5Samples")
