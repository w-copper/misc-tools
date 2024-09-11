from scipy.optimize import leastsq
import numpy as np

def vod2agb(vod, a, b, c, d):
    pred_agb = a * ( np.arctan(b *(vod -c)) - np.arctan(-b * c) )/ ( np.arctan(b * (np.inf - c)) - np.arctan(-b * c)) + d
    return pred_agb

def vod2agb_error(params, vod, agb):

    a,b,c,d = params
    pred_agb = vod2agb(vod, a, b, c, d)
    return ((pred_agb - agb) ** 2)


def vod2agb_fit(vod, agb):
    """
    Fit the VOD to AGB relationship.
    :param vod: VOD
    :param agb: AGB
    :return: a,b,c,d
    """

    p0 = [1, 1, 1, 1]

    params = leastsq(vod2agb_error, p0, args=(vod, agb))[0]

    return params


if __name__ == '__main__':

    vod = np.array([0.05, 0.1, 0.2, 0.3,])
    agb = np.array([0.05, 0.1, 0.2, 0.3,])

    params = vod2agb_fit(vod, agb)

    print(params)