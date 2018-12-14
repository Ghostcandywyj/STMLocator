# -*- coding: utf-8 -*-
# Author: Yaojing Wang

import random
import time
from collections import Counter
import math
import argparse
import os

outf = None
sourceCom = {}

def load_data(args):
    global outf
    #pro = 'pde'
    outf = open(args.output,'w')
    #hit10 = open(pro.upper()+"hit10.txt","w")
    f_data = open(args.inputB, 'r')

    data = []
    for l in f_data:
        rec = {'ts': eval(l)[0], 'ws': eval(l)[1], 'ss':eval(l)[-1]}
        data.append(rec)
        global sourceCom
        for source in rec['ts']:
            if source not in sourceCom:
                sourceCom[source] = set()
            sourceCom[source].add(rec['ss'])
    f_data.close()

    k_vocab_data = []
    sourceLen = []
    k_vocab = open(args.inputS,'r')
    for l in k_vocab:
        rec = eval(l)
        k_vocab_data.append(rec[0])
        sourceLen.append(rec[1])
    k_vocab.close()
    
    return data, k_vocab_data, sourceLen


def split_data(data, fold, f):
    index = 0
    tr_data = []
    te_data = []

    for d in data:
        if (index % fold) != f:
            tr_data.append(d)
        else:
            te_data.append(d)
        index += 1
    
    print (len(tr_data), len(te_data))

    return tr_data, te_data


def llda_cvb0_init(data, k_vocab, k_vocab_total, beta_0, beta_1):
    t_vocab = []
    w_vocab = []
    for d in data:
        for t in d['ts']:
            t_vocab.append(t)
        for w in d['ws']:
            w_vocab.append(w)

    print "word count: %d" % len(w_vocab)
    t_vocab = list(set(t_vocab))
    w_vocab = list(set(w_vocab))

    print "TV: %d, WV: %d" % (len(t_vocab), len(w_vocab))

    # random gamma for words
    gamma = {}
    d_index = 1
    for d in data:
        g = []
        tags = d['ts']
        words = d['ws']

        for w in words:
            ts = {}
            g_sum = 0

            for t in tags:
                ts[t] = {}
                #if w == t:
                if w in k_vocab[t]:
                    r = random.random()
                    g_sum += r
                    ts[t][1] = r
                r = random.random()
                g_sum += r
                ts[t][0] = r

            for t in ts:
                for k in ts[t]:
                    ts[t][k] /= g_sum

            g.append((w, ts))

        gamma[d_index] = g
        d_index += 1

    return gamma, t_vocab, w_vocab


def calc_n0_n0all(gamma_d):
    sum_n0 = {}
    sum_n0_all = 0

    for w, ts in gamma_d:
        for t in ts:
            for k in ts[t]:
                sum_n0[t] = sum_n0.get(t, 0) + ts[t][k]
                sum_n0_all += ts[t][k]

    return sum_n0, sum_n0_all


def calc_n1_n1all_n2_n2all_n3_n3all(gamma, t_vocab, w_vocab, k_vocab, k_vocab_total):
    sum_n1 = {}
    sum_n1_all = {}
    sum_n2 = {}
    sum_n2_all = {}
    sum_n3 = {}
    sum_n3_all = {}

    for t in t_vocab:
        sum_n1[t] = {}
        sum_n2[t] = {}
        sum_n3[t] = {}
        sum_n1_all[t] = 0
        sum_n2_all[t] = 0
        sum_n3_all[t] = 0
        for w in w_vocab:
            sum_n1[t][w] = 0
        for k in xrange(2):
            sum_n2[t][k] = 0
        for w in k_vocab_total:
            sum_n3[t][w] = 0

    for d in gamma:
        for w, ts in gamma[d]:
            for t in ts:
                for k in ts[t]:
                    #if w == t:
                    if w in k_vocab[t]:
                        sum_n2[t][k] += ts[t][k]
                        sum_n2_all[t] += ts[t][k]
                    if k == 0:
                        sum_n1[t][w] += ts[t][k]
                        sum_n1_all[t] += ts[t][k]
                    if k == 1:
                        sum_n3[t][w] += ts[t][k]
                        sum_n3_all[t] += ts[t][k]

    return sum_n1, sum_n1_all, sum_n2, sum_n2_all, sum_n3, sum_n3_all


def llda_cvb0(gamma, t_vocab, w_vocab, k_vocab, k_vocab_total, beta_0, beta_1, alpha, sourceAplha, eta, delta, count):
    t_num = len(t_vocab)
    v_num = len(w_vocab)

    alpha_l = alpha/t_num
    veta = v_num*eta
    delta_all = delta*len(k_vocab_total)

    beta_all = beta_0 + beta_1

    theta = {}
    phi = {}
    omega = {}
    pl = {}

    for t in t_vocab:
        phi[t] = {}
        for w in w_vocab:
            phi[t][w] = 0

    for t in t_vocab:
        omega[t] = {}
        for w in k_vocab_total:
            omega[t][w] = 0

    # init theta, pl
    for d in gamma:
        if len(gamma[d]) == 0:
            print d
        w, ts = gamma[d][0]
        theta[d] = {}
        for t in ts:
            theta[d][t] = 0

    for t in t_vocab:
        pl[t] = 0

    # calc n1, n1_all, n2, n2_all
    n1, n1_all, n2, n2_all, n3, n3_all = calc_n1_n1all_n2_n2all_n3_n3all(gamma, t_vocab, w_vocab, k_vocab, k_vocab_total)

    for c in xrange(1, count+1):
        start_time = time.clock()
        for d in gamma:
            n0, n0_all = calc_n0_n0all(gamma[d])
            w, ts = gamma[d][0]
            for w, ts in gamma[d]:
                g_sum = 0

                # remove current word, so need re-calc n0, n1, n1_all, sometime re-calc n2, n2_all
                for t in ts:
                    #if w == t:
                    if w in k_vocab[t]:
                        n0[t] -= ts[t][1]
                        for k in ts[t]:
                            n2[t][k] -= ts[t][k]
                            n2_all[t] -= ts[t][k]

                        n3[t][w] -= ts[t][1]
                        n3_all[t] -= ts[t][1]

                    n0[t] -= ts[t][0]
                    n1[t][w] -= ts[t][0]
                    n1_all[t] -= ts[t][0]


                for t in ts:
                    theta_d = n0[t] + alpha_l * sourceAplha[t]
                    #if w == t:
                    if w in k_vocab[t]:
                        g1 = theta_d * (n2[t][1] + beta_1)/(n2_all[t] + beta_all) * (n3[t][w] + delta)/(n3_all[t] + delta_all)
                        ts[t][1] = g1
                        g_sum += g1

                        g0 = theta_d*(n1[t][w] + eta)*(n2[t][0] + beta_0)/((n1_all[t] + veta)*(n2_all[t] + beta_all))
                        ts[t][0] = g0
                        g_sum += g0
                    else:
                        g0 = theta_d*(n1[t][w] + eta)/(n1_all[t] + veta)
                        ts[t][0] = g0
                        g_sum += g0

                for t in ts:
                    for k in ts[t]:
                        ts[t][k] /= g_sum

                # add current word, so need re-calc n1, n1_all, n2, n2_all again
                for t in ts:
                    #if w == t:
                    if w in k_vocab[t]:
                        n0[t] += ts[t][1]
                        for k in ts[t]:
                            n2[t][k] += ts[t][k]
                            n2_all[t] += ts[t][k]

                        n3[t][w] += ts[t][1]
                        n3_all[t] += ts[t][1]

                    n0[t] += ts[t][0]
                    n1[t][w] += ts[t][0]
                    n1_all[t] += ts[t][0]

        print "%03d, elapse: %d" % (c, time.clock() - start_time)

    for d in gamma:
        n0, n0_all = calc_n0_n0all(gamma[d])
        for t in theta[d]:
            theta[d][t] = (n0[t] + alpha_l * sourceAplha[t])/(n0_all + len(theta[d])*alpha_l*sourceAplha[t])

    for d in theta:
        for t in theta[d]:
            pl[t] += theta[d][t]

    for t in pl:
        pl[t] /= len(theta)

    for t in phi:
        for w in w_vocab:
            phi[t][w] = (n1[t][w] + eta)/(n1_all[t] + veta)

    for t in omega:
        for w in k_vocab_total:
            #print w, t
            omega[t][w] = (n3[t][w] + delta)/(n3_all[t] + delta_all)

    ptw = {}
    for t in n2:
        ptw[t] = (n2[t][1] + beta_1)/(n2_all[t] + beta_all)

    return pl, phi, omega, ptw


def llda_cvb0_train(data, k_vocab, k_vocab_total, beta_0, beta_1, alpha, sourceAplha, eta, delta, count):
    gamma, t_vocab, w_vocab = llda_cvb0_init(data, k_vocab, k_vocab_total, beta_0, beta_1)
    pl, phi, omega, ptw = llda_cvb0(gamma, t_vocab, w_vocab, k_vocab, k_vocab_total, beta_0, beta_1, alpha, sourceAplha, eta, delta, count)
    #print ptw

    return pl, phi, omega, ptw


def calc_pws(ws, t_vocab, pz, phi, omega, ptw):
    pws = {}

    ws = list(set(ws))
    for w in ws:
        pws[w] = 0
        for t in t_vocab:
            if w == t:
                pws[w] += (ptw[w] + (1 - ptw[w])*phi[t][w])*pz[t]
            else:
                pws[w] += phi[t][w]*pz[t]

    return pws


def calc_pwds(ws):
    pwds = {}

    ws_sum = len(ws)
    cws = Counter(ws)
    for w in cws:
        pwds[w] = cws[w]*1.0/ws_sum

    return pwds

def main(args):
    data, k_vocab, sourcelen = load_data(args)
    fold = args.fold

    sourcelen = [math.log(ss+1) for ss in sourcelen]
    sourcesum = float(sum(sourcelen))/float(len(sourcelen))
    sourceAplha = [float(ss)/sourcesum for ss in sourcelen]

    k_vocab_total = set()
    for d in k_vocab:
        k_vocab_total = k_vocab_total | set(d)
    k_vocab_total = list(k_vocab_total)
    
    iters = args.iter

    beta_0 = args.beta
    beta_1 = args.beta
    alpha = args.alpha
    eta = args.eta
    mu = args.mu

    for f in xrange(fold):
        tr_data, te_data = split_data(data, fold, f)
        train_start = time.time()
        pl, phi, omega, ptw = llda_cvb0_train(tr_data, k_vocab, k_vocab_total, beta_0, beta_1, alpha, sourceAplha, eta, mu, iters)
        print "fold", f, ": train time is", time.time() - train_start
        
    global outf
    outf.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inputs')
    
    parser.add_argument('-inputB', nargs='?', required=True, help='Input bug reports and their corresponding source files.')
    parser.add_argument('-inputS', nargs='?', required=True, help='Input source files and their LOC.')
    parser.add_argument('-output', nargs='?', required=True, help='Output file.')
    parser.add_argument('-alpha', default=50.0, type=float, help='Super parameter for generating topics/source files.')
    parser.add_argument('-beta', default=0.01, type=float, help='Super parameter for generating common words distribution.')
    parser.add_argument('-mu', default=0.01, type=float, help='Super parameter for generating co-occurrence words distribution.')
    #parser.add_argument('-gamma', default=0.01, type=float, help='')
    
    parser.add_argument('-eta', default=0.01, type=float, help='Super parameter for generating Bernoulli distribution \Psi.')
    parser.add_argument('-lenfunc', default='lin', nargs='?', help='Length function of LOC. (Linear-[lin], Logarithmic-[log], Exponential-[exp], Square root-[srt])')
    parser.add_argument('-fold', default=10, type=int, help='Number of training folds.')
    parser.add_argument('-iter', default=20, type=int, help='Number of training iterations for each fold.')
    args = parser.parse_args()

    main(args)
