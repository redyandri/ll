from scipy.stats import entropy
from numpy.linalg import norm
import numpy as np


class JSD(object):
    def __init__(self):
        return

    def get_distance(self,P, Q):
        #print "test-white:%s--train_white:%s" %(str(P),str(Q))
        _P = P / norm(P, ord=1)
        _Q = Q / norm(Q, ord=1)
        _M = 0.5 * (_P + _Q)
        return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))
