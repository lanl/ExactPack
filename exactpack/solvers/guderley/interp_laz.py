r"""This function calls the FMM routine "spline" to interpolate values
of "lambda" and "bhat" ( = ((gamma-1)/(gamma+1))*B ) for the
converging shock problem in cylindrical or spherical geometry
from the tabular values in Tables 6.4 (cyl) and 6.5 (sph) of: [Lazarus1981]_

Note, per the subsequent erratum: [Lazarus1982]_

The values in the third column of these tables, listed as "B" in
the first reference above, are actually:

.. math::
  \hat b = B * \frac{(\gamma - 1)}{\gamma + 1}

2007.07.03  Kamm    initial development -- seems to work!
2007.07.05  Kamm    clean up the code, which appears to work fine
2007.07.20  Ramsey  code converted to a function
2022.09.27  Thrussell Converted to Python
"""
import numpy as np
from scipy import interpolate


def interp_laz(n, gamma, lambda_):
    """Driver routines for the interpolation."""
    if n == 2:
        lcyl = True
    elif n == 3:
        lcyl = False
    else:
        raise ValueError('Invalid geometry')

    bhat = DO_INTERP(lcyl, gamma, lambda_)
    # Strictly using "bhat" as the upper bound for the more precise
    # value of bhat will in all likeihood produce a "endpoints do not
    # have opposite signs" error in the "zeroin" routine used in the
    # "guderley_1D driving program. Therefore, the interpolated value
    # of bhat is adjusted slightly to counter this error. The setting
    # at which it is currently placed may need to be adjusted for
    # various cases of gamma and n.
    return bhat


def DO_INTERP(lcyl, gamma, lambda_):
    """Driver routine for the interpolation of values from Lazarus's table of
    Guderley parameters.
    """
    if lcyl:
        gamval, lamval, bval = GET_CYL()
    else:
        gamval, lamval, bval = GET_SPH()
    # Orignal code has lamvals and lambda, we assume someone might want to
    # interpolate lambda?
    # tck = interpolate.splrep(lamval, bval)
    # bhat = interpolate.splev(lambda_, tck)
    tck = interpolate.splrep(gamval, bval)
    bhat = interpolate.splev(gamma, tck)

    return bhat


def GET_CYL():
    """This routine reads in the values from Lazarus Table 6.4 for the Guderley
    lambda and bhat = ((gamma-1)/(gamma+1))*B in the cylindrical geometry case.
    """
    nmax = 46
    gamval = np.zeros(nmax)
    lamval = np.zeros(nmax)
    bval = np.zeros(nmax)

    gamval[0] = 1.00001
    gamval[1] = 1.0001
    gamval[2] = 1.001
    gamval[3] = 1.005
    gamval[4] = 1.01
    gamval[5] = 1.03
    gamval[6] = 1.05
    gamval[7] = 1.07
    gamval[8] = 1.10
    gamval[9] = 1.15
    gamval[10] = 1.2
    gamval[11] = 1.3
    gamval[12] = 1.4
    gamval[13] = 1.5
    gamval[14] = 1.66667
    gamval[15] = 1.7
    gamval[16] = 1.8
    gamval[17] = 1.9
    gamval[18] = 1.92
    gamval[19] = 2.0
    gamval[20] = 2.0863
    gamval[21] = 2.0883
    gamval[22] = 2.125
    gamval[23] = 2.2
    gamval[24] = 2.3676
    gamval[25] = 2.3678
    gamval[26] = 2.4
    gamval[27] = 2.6
    gamval[28] = 2.8
    gamval[29] = 2.83920
    gamval[30] = 2.83929
    gamval[31] = 3.0
    gamval[32] = 3.4
    gamval[33] = 4.0
    gamval[34] = 5.0
    gamval[35] = 6.0
    gamval[36] = 7.0
    gamval[37] = 8.0
    gamval[38] = 10.0
    gamval[39] = 15.0
    gamval[40] = 20.0
    gamval[41] = 30.0
    gamval[42] = 50.0
    gamval[43] = 100.0
    gamval[44] = 1000.0
    gamval[45] = 9999.0

    lamval[0] = 1.0022073240
    lamval[1] = 1.0068195769
    lamval[2] = 1.0202846866
    lamval[3] = 1.0414733956
    lamval[4] = 1.0553973808
    lamval[5] = 1.0850737604
    lamval[6] = 1.1023892512
    lamval[7] = 1.1150692073
    lamval[8] = 1.1296268597
    lamval[9] = 1.1475773258
    lamval[10] = 1.1612203175
    lamval[11] = 1.1817213587
    lamval[12] = 1.1971414294
    lamval[13] = 1.2095591324
    lamval[14] = 1.2260537880
    lamval[15] = 1.2288931032
    lamval[16] = 1.2367055181
    lamval[17] = 1.2436278359
    lamval[18] = 1.2449208188
    lamval[19] = 1.2498244759
    lamval[20] = 1.2546830116
    lamval[21] = 1.2547907910
    lamval[22] = 1.2567323668
    lamval[23] = 1.2604989804
    lamval[24] = 1.2680643171
    lamval[25] = 1.2680727188
    lamval[26] = 1.2694076380
    lamval[27] = 1.2769816100
    lamval[28] = 1.2835139723
    lamval[29] = 1.2846912316
    lamval[30] = 1.2846938989
    lamval[31] = 1.2892136582
    lamval[32] = 1.2986950941
    lamval[33] = 1.3095267323
    lamval[34] = 1.3220499813
    lamval[35] = 1.3305627751
    lamval[36] = 1.3367301837
    lamval[37] = 1.3414054776
    lamval[38] = 1.3480251307
    lamval[39] = 1.3569909807
    lamval[40] = 1.3615356210
    lamval[41] = 1.3661223915
    lamval[42] = 1.3698225859
    lamval[43] = 1.3726158889
    lamval[44] = 1.3751432790
    lamval[45] = 1.3753967176

    bval[0] = 0.521740
    bval[1] = 0.554609
    bval[2] = 0.625514
    bval[3] = 0.697737
    bval[4] = 0.724429
    bval[5] = 0.731819
    bval[6] = 0.708880
    bval[7] = 0.682234
    bval[8] = 0.644590
    bval[9] = 0.593262
    bval[10] = 0.554542
    bval[11] = 0.502117
    bval[12] = 0.469268
    bval[13] = 0.447230
    bval[14] = 0.423698
    bval[15] = 0.420261
    bval[16] = 0.411663
    bval[17] = 0.405047
    bval[18] = 0.403911
    bval[19] = 0.399877
    bval[20] = 0.396295
    bval[21] = 0.396220
    bval[22] = 0.394904
    bval[23] = 0.392529
    bval[24] = 0.388444
    bval[25] = 0.388440
    bval[26] = 0.387812
    bval[27] = 0.384755
    bval[28] = 0.382794
    bval[29] = 0.382506
    bval[30] = 0.382505
    bval[31] = 0.381580
    bval[32] = 0.380564
    bval[33] = 0.380920
    bval[34] = 0.383355
    bval[35] = 0.386279
    bval[36] = 0.389064
    bval[37] = 0.391561
    bval[38] = 0.395687
    bval[39] = 0.402440
    bval[40] = 0.406405
    bval[41] = 0.410797
    bval[42] = 0.414640
    bval[43] = 0.417726
    bval[44] = 0.420658
    bval[45] = 0.420960

    return gamval, lamval, bval


def GET_SPH():
    """This routine reads in the values from Lazarus Table 6.5 for the Guderley
    lambda and bhat = ((gamma-1)/(gamma+1))*B in the spherical geometry case.
    """
    nmax = 52
    gamval = np.zeros(nmax)
    lamval = np.zeros(nmax)
    bval = np.zeros(nmax)

    gamval[0] = 1.00001
    gamval[1] = 1.0001
    gamval[2] = 1.001
    gamval[3] = 1.01
    gamval[4] = 1.03
    gamval[5] = 1.05
    gamval[6] = 1.07
    gamval[7] = 1.10
    gamval[8] = 1.15
    gamval[9] = 1.2
    gamval[10] = 1.3
    gamval[11] = 1.4
    gamval[12] = 1.5
    gamval[13] = 1.6
    gamval[14] = 1.66667
    gamval[15] = 1.7
    gamval[16] = 1.8
    gamval[17] = 1.86
    gamval[18] = 1.88
    gamval[19] = 1.9
    gamval[20] = 2.0
    gamval[21] = 2.010
    gamval[22] = 2.012
    gamval[23] = 2.2
    gamval[24] = 2.2215
    gamval[25] = 2.2217
    gamval[26] = 2.4
    gamval[27] = 2.5518
    gamval[28] = 2.55194
    gamval[29] = 2.6
    gamval[30] = 2.8
    gamval[31] = 3.0
    gamval[32] = 3.2
    gamval[33] = 3.4
    gamval[34] = 3.6
    gamval[35] = 3.8
    gamval[36] = 4.0
    gamval[37] = 4.5
    gamval[38] = 5.0
    gamval[39] = 5.5
    gamval[40] = 6.0
    gamval[41] = 6.5
    gamval[42] = 7.0
    gamval[43] = 8.0
    gamval[44] = 10.0
    gamval[45] = 15.0
    gamval[46] = 20.0
    gamval[47] = 30.0
    gamval[48] = 50.0
    gamval[49] = 100.0
    gamval[50] = 1000.0
    gamval[51] = 9999.0

    lamval[0] = 1.0044047883
    lamval[1] = 1.0135647885
    lamval[2] = 1.0401005736
    lamval[3] = 1.1088100742
    lamval[4] = 1.1671691602
    lamval[5] = 1.2015664277
    lamval[6] = 1.2269581432
    lamval[7] = 1.2563291060
    lamval[8] = 1.2928404943
    lamval[9] = 1.3207565353
    lamval[10] = 1.3628123548
    lamval[11] = 1.3943607838
    lamval[12] = 1.4195913539
    lamval[13] = 1.4405288149
    lamval[14] = 1.4526927211
    lamval[15] = 1.4583285785
    lamval[16] = 1.4737227445
    lamval[17] = 1.4820184714
    lamval[18] = 1.4846461951
    lamval[19] = 1.4872097129
    lamval[20] = 1.4991468274
    lamval[21] = 1.5002661592
    lamval[22] = 1.5004885113
    lamval[23] = 1.5193750470
    lamval[24] = 1.5213088378
    lamval[25] = 1.5213266323
    lamval[26] = 1.5358986669
    lamval[27] = 1.5465622206
    lamval[28] = 1.5465714207
    lamval[29] = 1.5496663736
    lamval[30] = 1.5613198923
    lamval[31] = 1.5713126233
    lamval[32] = 1.5799755842
    lamval[33] = 1.5875567751
    lamval[34] = 1.5942459679
    lamval[35] = 1.6001909794
    lamval[36] = 1.6055087137
    lamval[37] = 1.6166309698
    lamval[38] = 1.6254243269
    lamval[39] = 1.6325476141
    lamval[40] = 1.6384333257
    lamval[41] = 1.6433769444
    lamval[42] = 1.6475870992
    lamval[43] = 1.6543738548
    lamval[44] = 1.6637583967
    lamval[45] = 1.6760512867
    lamval[46] = 1.6821004429
    lamval[47] = 1.6880830534
    lamval[48] = 1.6928204564
    lamval[49] = 1.6963447551
    lamval[50] = 1.6994953607
    lamval[51] = 1.6998093041

    bval[0] = 0.541777
    bval[1] = 0.607335
    bval[2] = 0.758422
    bval[3] = 0.988008
    bval[4] = 0.996617
    bval[5] = 0.931071
    bval[6] = 0.860781
    bval[7] = 0.769242
    bval[8] = 0.658324
    bval[9] = 0.584657
    bval[10] = 0.496984
    bval[11] = 0.448082
    bval[12] = 0.417547
    bval[13] = 0.397073
    bval[14] = 0.386974
    bval[15] = 0.382711
    bval[16] = 0.372341
    bval[17] = 0.367499
    bval[18] = 0.366070
    bval[19] = 0.364725
    bval[20] = 0.359085
    bval[21] = 0.358608
    bval[22] = 0.358514
    bval[23] = 0.351834
    bval[24] = 0.351293
    bval[25] = 0.351288
    bval[26] = 0.348072
    bval[27] = 0.346707
    bval[28] = 0.346707
    bval[29] = 0.346472
    bval[30] = 0.346267
    bval[31] = 0.346985
    bval[32] = 0.348323
    bval[33] = 0.350078
    bval[34] = 0.352112
    bval[35] = 0.354327
    bval[36] = 0.356656
    bval[37] = 0.362682
    bval[38] = 0.368678
    bval[39] = 0.374437
    bval[40] = 0.379873
    bval[41] = 0.384959
    bval[42] = 0.389698
    bval[43] = 0.398201
    bval[44] = 0.411949
    bval[45] = 0.434177
    bval[46] = 0.447247
    bval[47] = 0.461834
    bval[48] = 0.474726
    bval[49] = 0.485184
    bval[50] = 0.495226
    bval[51] = 0.496265

    return gamval, lamval, bval
