import os, sys, codecs
import argparse
import numpy as np

def read_mat(filename):
    sin = codecs.open(filename, "r", "utf8")
    row, col = sin.readline().strip().split()
    row, col = int(row), int(col)

    mat = np.zeros((row, col), dtype=np.float32)

    for r , line in enumerate(sin):
        for c, v in enumerate(line.strip().split(" ")):
            mat[r, c] = float(v)

    return mat

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dict_file", type=str)
    parser.add_argument("--u_file", type=str)
    parser.add_argument("--layer", type=int)
    parser = parser.parse_args()

    id2w = [_.strip() for _ in open(parser.dict_file)]

    mat = read_mat(parser.u_file)

    for k in range(mat.shape[1]):
        info = []
        kk = k
        for i in range(parser.layer):
            info.append(str(parser.layer-i-1) + ":" + str(kk//2))
            kk = kk // 2
        print ("topic-%d, topic_tree [%s]" % (k, "->".join(info)))
        ww = dict([[w, wi] for w, wi in zip(id2w, mat[:, k])])
        ww = sorted(ww.items(), key=lambda x:x[1], reverse=True)

        for i in range(20):
            if ww[i][1] >= 0.:
                print ("\t%s : %.6f" % (ww[i][0], ww[i][1]))
        print ("")
