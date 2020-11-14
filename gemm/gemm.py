import os, sys
import numpy as np
from scipy.io import mmread

def customGEMM(A, B, maxA=None, maxB=None, a_split=None, b_split=None, dtype=np.float32):
    INOUTPRES = dtype

    if INOUTPRES == np.float32:
        printFloatFormat = '{:.7e}'.format
        sa = 3
        FLOATFRACTION = 23
    elif INOUTPRES == np.float64:
        printFloatFormat = '{:.16e}'.format
        sa = 6
        FLOATFRACTION = 52
    else:
        sys.exit("INOUTPRES can only be np.float32 or np.float64")
        
    FLOATFRACTIONFP16 = 10

    m, k = A.shape
    n = m
    blk = 32
    C = np.zeros((m, n), dtype=INOUTPRES)

    # split A and B
    sb = sa
    sc = sa + sb - 1
    u = INOUTPRES(1 << (1+FLOATFRACTION-FLOATFRACTIONFP16))

    if a_split is None: # split A only if it is not provided as argument
        a_split = np.zeros((m//blk, k//blk, sa, blk, blk), dtype=np.float32)
        maxA = np.zeros((m//blk, k//blk, blk), dtype=np.float32)
        for row in range(m//blk):
            for col in range(k//blk):
                for row_in_blk in range(blk):
                    A_row_in_blk = A[row*blk+row_in_blk, col*blk:(col+1)*blk]
                    maxA[row, col, row_in_blk] = np.amax(np.abs(A_row_in_blk))
                    _, expo = np.frexp(maxA[row, col, row_in_blk])
                    maxA[row, col, row_in_blk] = np.ldexp(1.0, expo)
                    if np.abs(maxA[row, col, row_in_blk]) < 1e-19:
                        continue
                    A_row_in_blk /= maxA[row, col, row_in_blk]
                    for iSa in range(sa):
                        s = A_row_in_blk + u
                        s -= u
                        a_split[row, col, iSa, row_in_blk, :] = s.astype(np.float16)
                        A_row_in_blk -= a_split[row, col, iSa, row_in_blk, :]
                        A_row_in_blk *= INOUTPRES(1 << FLOATFRACTIONFP16)

    if b_split is None: # split B only if it is not provided as argument
        b_split = np.zeros((k//blk, n//blk, sb, blk, blk), dtype=np.float32)
        maxB = np.zeros((k//blk, n//blk, blk), dtype=np.float32)
        for row in range(k//blk):
            for col in range(n//blk):
                for col_in_blk in range(blk):
                    B_col_in_blk = B[row*blk:(row+1)*blk, col*blk+col_in_blk]
                    maxB[row, col, col_in_blk] = np.amax(np.abs(B_col_in_blk))
                    _, expo = np.frexp(maxB[row, col, col_in_blk])
                    maxB[row, col, col_in_blk] = np.ldexp(1.0, expo)
                    if np.abs(maxB[row, col, col_in_blk]) < 1e-19:
                        continue
                    B_col_in_blk /= maxB[row, col, col_in_blk]
                    for jSb in range(sb):
                        s = B_col_in_blk + u
                        s -= u
                        b_split[row, col, jSb, :, col_in_blk] = s.astype(np.float16)
                        B_col_in_blk -= b_split[row, col, jSb, :, col_in_blk]
                        B_col_in_blk *= INOUTPRES(1 << FLOATFRACTIONFP16)
    
    # multiply and combine; condition: sa >= sb
    c_split = np.zeros((m//blk, n//blk, sc, blk, blk), dtype=INOUTPRES)
    maxC = np.zeros((m//blk, n//blk), dtype=INOUTPRES)
    maxAB = np.zeros((blk, blk), dtype=np.float32)
    for row in range(m//blk):
        for col in range(n//blk):
            C_blk = C[row*blk:(row+1)*blk, col*blk:(col+1)*blk]
            for iSa_jSb_sum in range(sb):
                for jSb in range(iSa_jSb_sum+1):
                    iSa = iSa_jSb_sum - jSb;
                    for idx in range(k//blk):
                        for row_in_blk in range(blk):
                            maxAB[row_in_blk, :] = maxA[row, idx, row_in_blk] * maxB[idx, col, :]
                        c_split[row, col, iSa_jSb_sum, :, :] += maxAB * (a_split[row, idx, iSa, :, :] @ b_split[idx, col, jSb, :, :])
                C_blk += c_split[row, col, iSa_jSb_sum, :, :] / INOUTPRES(1 << FLOATFRACTIONFP16*iSa_jSb_sum)

            for iSa_jSb_sum in range(sb, sa):
                for jSb in range(sb):
                    iSa = iSa_jSb_sum - jSb;
                    for idx in range(k//blk):
                        for row_in_blk in range(blk):
                            maxAB[row_in_blk, :] = maxA[row, idx, row_in_blk] * maxB[idx, col, :]
                        c_split[row, col, iSa_jSb_sum, :, :] += maxAB * (a_split[row, idx, iSa, :, :] @ b_split[idx, col, jSb, :, :])
                C_blk += c_split[row, col, iSa_jSb_sum, :, :] / INOUTPRES(1 << FLOATFRACTIONFP16*iSa_jSb_sum)

            for iSa_jSb_sum in range(sa, sc):
                for jSb in range(iSa_jSb_sum-sa+1, sb):
                    iSa = iSa_jSb_sum - jSb;
                    for idx in range(k//blk):
                        for row_in_blk in range(blk):
                            maxAB[row_in_blk, :] = maxA[row, idx, row_in_blk] * maxB[idx, col, :]
                        c_split[row, col, iSa_jSb_sum, :, :] += maxAB * (a_split[row, idx, iSa, :, :] @ b_split[idx, col, jSb, :, :])
                C_blk += c_split[row, col, iSa_jSb_sum, :, :] / INOUTPRES(1 << FLOATFRACTIONFP16*iSa_jSb_sum)

    return maxC, c_split, C




INOUTPRES = np.float32 # set to np.float32 or np.float64
printFloatFormat = '{:.7e}'.format if INOUTPRES == np.float32 else '{:.16e}'.format

for fileName in os.listdir("./"):
    if not fileName.endswith(".mtx"): # cavity05_modified_1184
        continue

    # direct calculation result with float64 as the reference
    _A = mmread(fileName).A.astype(INOUTPRES)
    A = _A[:(_A.shape[0]//32)*32, :(_A.shape[1]//32)*32]
    B = A.T.copy()
    C_ref = (A.astype(np.float64) @ B.astype(np.float64)).astype(INOUTPRES)
    C_ref[np.abs(C_ref) < 1e-19] = 1e-19
    
    print("Matrix file: ", fileName)
    print("Matrix A shape: ", A.shape)
    print()
    
    # compare reference result with direct result
    C = A @ B
    maxC = C
    errorC = np.abs((C_ref - C) / C_ref)
    
    errorThreshold = 1e-9
    print("direct A @ B")
    print("max element in C: ", np.amax(np.abs(maxC)))
    errorC[np.abs(C_ref) <= errorThreshold*np.amax(np.abs(maxC))] = 0.0
    ind = np.unravel_index(np.argmax(errorC), errorC.shape)
    print("at max error C, index = {}, C_ref = {:.7e}, C = {:.7e}, errorC = {:.7e}".format(ind, C_ref[ind], C[ind], errorC[ind]))
    print("average relative error: ", np.mean(errorC[np.abs(C_ref) > errorThreshold*np.amax(np.abs(C_ref))]))
    print()
    
    # compare reference result with customGEMM block
    _1, _2, C = customGEMM(A.copy(), B.copy(), dtype=INOUTPRES)
    maxC = C
    errorC = np.abs((C_ref - C) / C_ref)
    
    print("customGEMM block, each row/col split")
    print("max element in C: ", np.amax(np.abs(maxC)))
    errorC[np.abs(C_ref) <= errorThreshold*np.amax(np.abs(maxC))] = 0.0
    ind = np.unravel_index(np.argmax(errorC), errorC.shape)
    print("at max error C, index = {}, C_ref = {:.7e}, C = {:.7e}, errorC = {:.7e}".format(ind, C_ref[ind], C[ind], errorC[ind]))
    print("average relative error: ", np.mean(errorC[np.abs(C_ref) > errorThreshold*np.amax(np.abs(C_ref))]))
    print()