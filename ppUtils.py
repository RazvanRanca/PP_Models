from venture.shortcuts import *
import numpy as np
import sys
from matplotlib import pyplot as plt

langs = ["Venture", "Bugs"]

def posterior_samples(v, var_name,no_samples,no_burns,int_mh):
    s=[];
    for sample in range(no_burns):
        v.infer(1)
        if sample % 100 == 0:
          print "Burned", sample, "samples"

    counter = no_burns
    print "Burned", no_burns, "samples"
    for sample in range(no_samples): #v.restart()
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
  indices = []
  for lang in langs:
    fn = model + "/" + lang + "/" + mType + "Samples"
    with open(fn,'r') as f:
      vals = f.read().strip().split('\n')
      time = float(vals[-1])
      vals = map(lambda x: x.split(), vals[:-1])
      samples[lang] = [float(x[1]) for x in vals]
      times[lang] = time
      inds = [float(x[0]) for x in vals]
      assert (indices == [] or indices == inds) #sanity check
  return times, samples

def getStats(samples):
  return dict([(k, np.mean(v)) for k,v in samples.items()]), dict([(k, np.std(v)) for k,v in samples.items()])

def showDists(samples, times, bins):
  means, sds = getStats(samples)
  f, axs = plt.subplots(2, sharex=True, sharey=True)
  count = 0
  for lang in langs:
    ax = axs[count]
    ax.hist(samples[lang],bins, normed=True)
    ax.set_title(lang + ",  m:" + str(means[lang])[:5] + ",  sd:" + str(sds[lang])[:5] + ",  rt:" + str(times[lang]) + "s", size=20)
    plt.setp(ax.get_xticklabels(), visible=True)
    count += 1
  f.subplots_adjust(hspace=0.3)
  plt.show()

if __name__ == "__main__":
  times, samples = readSamples("tdf", "cont")
  showDists(samples, times, 50)
