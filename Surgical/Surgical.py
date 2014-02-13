from venture.shortcuts import *
import numpy as np

def posterior_samples(var_name,no_samples,int_mh):
    s=[];
    for sample in range(no_samples): #v.restart()
        v.infer(int_mh)
        label = var_name + str(np.random.randint(10**5))+str(sample)
        v.predict(var_name,label)
        s.append(v.report(label))
    return s

def SurgicalRand(v):
  v.assume("lgt", "(lambda (x) (/ 1 (+ 1 (exp (* -1 x)))))")
  v.assume("mu", "(normal 1 1000)")
  v.assume("sgm", "(inv_gamma 0.001 0.001)")

  v.assume("b", "(lambda () (normal mu sgm))")
  v.assume("probDeath", "(lambda () (lgt b))")
  v.assume("died","(lambda () (flip probDeath))")
  #v.assume("expDeaths", "(lambda (ops) (binomial probDeath ops))")
 
  [v.observe("(died)", "true") for i in range(10)]
  
  samples = posterior_samples("probDeath", no_samples=1, int_mh=300)
  print samples
  print np.mean(samples), np.std(samples)

def Surgical0(ops, deaths):
  binomStr = " (real(categorical (minus 1 pd) pd))" * ops

  v.assume("probDeath", "(lambda () (beta 1.0 1.0))")
  v.assume("expDeaths", "(lambda (pd) (plus%s))" % binomStr)

  v.observe("(expDeaths (probDeath))", str(deaths))

  #samples = posterior_samples("probDeath", no_samples=1, int_mh=300)
  #print samples
  #print np.mean(samples), np.std(samples)

def Surgical1(ops, deaths):
  binomStr = " (flip pd)" * ops

  v.assume("countTs", "(lambda (l) (branch (is_pair l) (lambda () (plus (branch (first l) (lambda () 1)  (lambda () 0)) (countTs (rest l)))) (lambda () 0)))")
  v.assume("probDeath", "(lambda () (beta 1.0 1.0))")
  v.assume("expDeaths", "(lambda (pd) (countTs (list %s)))" % binomStr)

  v.observe("(expDeaths (probDeath))", str(deaths))

  #samples = posterior_samples("probDeath", no_samples=1, int_mh=300)
  #print samples
  #print np.mean(samples), np.std(samples)

if __name__ == "__main__":
  v = make_church_prime_ripl()

  Surgical0(47, 0)
  """
(define opList '(47 148 119 810 211 196 148 215 207 97 256 360))
(define deathList '(0 18 8 46 8 13 9 31 14 8 29 24))

(define samples 
  (lambda (op death)
    (mh-query
     1000 20
     
     (define probDeath (beta 1.0 1.0))
     (define expDeaths  (round (binomial probDeath op)))
     
     probDeath
     
     (equal? expDeaths death ))))

(map (lambda (op death) (mean (samples op death))) opList deathList)


(define (lgt x) (/ 1 (+ 1 (exp (* -1 x)))))
(define opList '(47 148 119 810 211 196 148 215 207 97 256 360))
(define deathList '(0 18 8 46 8 13 9 31 14 8 29 24))

(define mu (gaussian 0 1000))
(define sgm (gamma 0.001 1000))
     
(define samples 
  (lambda (op death)
    (mh-query
     1000 20
     
     (define probDeath (lgt (gaussian mu sgm)))
     (define expDeaths  (round (binomial probDeath op)))
     
     probDeath
     
     (equal? expDeaths death ))))

(map (lambda (op death) (mean (samples op death))) opList deathList)
;(list mu sgm (lgt (gaussian mu sgm)))"""
