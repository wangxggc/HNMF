import tensorflow as tf
import numpy as np
from model import HOSCGrad
import codecs, math, sys, random, os
from collections import Counter


def load_data(data_file, id2w, valid_ids):
    docs = [_.strip().split() for _ in codecs.open(data_file, "r", "utf8")][1:]
    doc_ids = [i for i in range(len(docs))]
    num_docs = len(doc_ids)

    num_words = len(id2w)
    w2id = dict([[w, i] for i, w in enumerate(id2w)])
    wdf = Counter()
    for doc_id in doc_ids:
        wdf.update(set(docs[doc_id]))

    w2idf = {}
    for k, v in wdf.items():
        if k not in w2id:
            continue
        w2idf[k] = math.log(float(num_docs) / float(v))

    toy_docs_mat = np.zeros((num_words, 100), dtype=np.float32)
    docs_mat = [[], []]

    non_zeros = 0
    for id, doc_id in enumerate(valid_ids):
        doc = docs[doc_id]
        wtf = Counter()
        wtf.update([w for w in doc if w in w2idf])
        sum = 0.

        indices, values = [], []
        for w, v in wtf.items():
            if w not in w2idf:
                continue
            indices.append([w2id[w], id])
            values.append(float(v)/len(doc) * w2idf[w])
            # values.append(float(v)/len(doc) * w2idf[w] / scale)
            sum += values[-1] ** 2
        values = [v/math.sqrt(sum) for v in values]

        docs_mat[0] += indices
        docs_mat[1] += values
        non_zeros += len(values)
        if id < 100:
            for (wid, _), val in zip(indices, values):
                toy_docs_mat[wid, id] = val
    print ("none zero values in data %d" % non_zeros)
    return docs_mat, num_words, num_docs, w2id, id2w, valid_ids, toy_docs_mat


def norm_row(shape):
    mat = np.random.rand(shape[0], shape[1])
    for r in range(shape[0]):
        base = math.sqrt(sum(mat[r, :] ** 2))
        mat[r, :] /= base
    return mat


def norm_col(shape):
    mat = np.random.rand(shape[0], shape[1])
    for c in range(shape[1]):
        base = math.sqrt(sum(mat[:, c] ** 2))
        mat[:, c] /= base
    return mat

def norm_col_row(shape):
    mat = np.random.rand(shape[0], shape[1])
    for c in range(shape[1]):
        base = math.sqrt(sum(mat[:, c] ** 2))
        mat[:, c] /= base
    for r in range(shape[0]):
        base = math.sqrt(sum(mat[r, :] ** 2))
        if base > 1.:
            mat[r, :] /= base
    return mat


def scale_grad(U, gradU, lr):
    mu = np.sum(U) / float(np.nonzero(U)[0].shape[0])
    gu = np.mean(gradU)
    if gu == 0.:
        return 1.
    if lr * gu > mu:
        return mu / 10. / (lr * gu)
    return 1.


def updateU(session, model, layer, epoch, num_steps, lr, toyD, D, Us, Vs):
    log_format = "Layer {0: <5d}  epoch {1: <8}  update-{2}: step {3: <8}  loss {4:.6f}, scale {5:.6f}"
    for n in range(num_steps):
        feed_dict = {
            model.Ds: toyD,
            model.inputs[layer]["D"]: D
        }
        if layer == 0:
            feed_dict[model.inputs[layer]["U"]] = Us[-1]
            feed_dict[model.inputs[layer]["V"]] = Vs[-1]

        else:
            feed_dict[model.inputs[layer]["U-pre"]] = Us[-2]
            feed_dict[model.inputs[layer]["V-pre"]] = Vs[-2]
            feed_dict[model.inputs[layer]["U-cur"]] = Us[-1]
            feed_dict[model.inputs[layer]["V-cur"]] = Vs[-1]

        u_grad, loss = session.run([model.u_grads[layer], model.losses[layer]], feed_dict=feed_dict)
        scale = scale_grad(Us[-1], u_grad, lr)
        Us[-1] -= u_grad * lr * scale
        Us[-1][Us[-1] < 0.] = 0.
        print("\r" + log_format.format(layer, epoch, "u", n, loss, scale)),
        sys.stdout.flush()
    print ("")


def updateV(session, model, layer, epoch, num_steps, lr, toyD, D, Us, Vs):
    log_format = "Layer {0: <5d}  epoch {1: <8}  update-{2}: step {3: <8}  loss {4:.6f}, scale {5:.6f}"
    for n in range(num_steps):
        feed_dict = {
            model.Ds: toyD,
            model.inputs[layer]["D"]: D
        }
        if layer == 0:
            feed_dict[model.inputs[layer]["U"]] = Us[-1]
            feed_dict[model.inputs[layer]["V"]] = Vs[-1]

        else:
            feed_dict[model.inputs[layer]["U-pre"]] = Us[-2]
            feed_dict[model.inputs[layer]["V-pre"]] = Vs[-2]
            feed_dict[model.inputs[layer]["U-cur"]] = Us[-1]
            feed_dict[model.inputs[layer]["V-cur"]] = Vs[-1]

        v_grad, loss = session.run([model.v_grads[layer], model.losses[layer]], feed_dict=feed_dict)
        scale = scale_grad(Vs[-1], v_grad, lr)
        Vs[-1] -= v_grad * lr * scale
        
        sparse_v_grad = np.zeros_like(Vs[-1], dtype=np.float32)
        sparse_v_grad[Vs[-1] > 0] = 0.0001
        Vs[-1] -= sparse_v_grad * lr
        
        Vs[-1][Vs[-1] < 0.] = 0.
        print("\r" + log_format.format(layer, epoch, "v", n, loss, scale)),
        sys.stdout.flush()
    print ("")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, help="input datas, each doc pre line, seperate by ' '")
    parser.add_argument("--init_k", type=int, help="init topic number by HSOC")
    parser.add_argument("--num_layers", type=int, help="")
    parser.add_argument("--start_layer", type=int, help="")
    parser.add_argument("--num_split", type=int, help="")
    parser.add_argument("--a", type=float, help="parameter alpha")
    parser.add_argument("--a0", type=float, help="parameter alpha0")
    parser.add_argument("--a1", type=float, help="parameter alpha1")
    parser.add_argument("--b", type=float, help="parameter beta")
    parser.add_argument("--epoch", type=int, help="epoch per layer")
    parser.add_argument("--in_epoch", type=int, help="20")
    parser.add_argument("--min_hold", type=int, help="min hold for words")
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--save_dir", type=str, help="save models into here")
    parser.add_argument("--break_point_dir", type=str, help="old model re-run from")
    parser = parser.parse_args()

    os.system("cp -r %s/* %s/" % (parser.break_point_dir, parser.save_dir))
    print ("cp -r %s/* %s/" % (parser.break_point_dir, parser.save_dir))
    sys.stdout.flush()

    valid_ids = [int(data.strip()) for data in open(parser.save_dir + "/docids.txt")]
    id2w = [_.strip()  for _ in codecs.open(parser.save_dir + "wordmap.txt", "r", "utf8")]

    docs_mat, num_words, num_docs, w2id, id2w, valid_ids, toy_docs_mat = load_data(parser.data_file, id2w, valid_ids)
    docs_mat = tf.SparseTensorValue(indices=docs_mat[0], values=docs_mat[1],
                                    dense_shape=[num_words, num_docs])
                                    # dense_shape=tf.TensorShape([tf.Dimension(num_words), tf.Dimension(num_docs)]))

    print (docs_mat.dense_shape)
    print ("Load documents %d, words %d" % (num_docs, num_words))
    sys.stdout.flush()

    num_topics = parser.init_k
    num_layers = parser.num_layers
    num_split  = parser.num_split

    model = HOSCGrad(
        num_words, num_docs, parser.init_k, num_layers, num_split, parser.a, parser.a0, parser.a1
    )
    sess = tf.Session()

    def load_mat(filename):
        print ("load from %s" % filename)
        sys.stdout.flush()
        sin = open(filename)
        num_row, num_col = [int(x) for x in sin.readline().strip().split(" ")]
        mat = np.zeros((num_row, num_col), dtype=np.float32)
        print mat.shape
        for r, data in enumerate(sin):
            vals = [float(x) for x in data.strip().split(" ")]
            mat[r, 0:num_col] = vals[0:num_col]
        return mat

    def save_mat(filename, mat):
        sout = codecs.open(filename, "w", "utf8")
        sout.write("%d %d\n" % (mat.shape[0], mat.shape[1]))
        for r in range(mat.shape[0]):
            str = " ".join(["%.6f" % v for v in mat[r, :]])
            sout.write(str + "\n")
        sout.close()

    Us, Vs = [], []
    for i in range(0, parser.start_layer):
        Us.append(load_mat(parser.save_dir + "/U-%d.txt" % (i)))
        Vs.append(load_mat(parser.save_dir + "/V-%d.txt" % (i)))
        print ("load pre-training parameters from %s, U-%d.txt = %s, V-%d.txt = %s" % (parser.save_dir, i, str(Us[-1].shape), i, str(Vs[-1].shape)))
        num_topics *= num_split

    for layer in range(parser.start_layer, parser.num_layers):
        print("layer %d, topic num %d" % (layer, num_topics))
        # init
        Us.append(norm_col_row([num_words, num_topics]))
        Vs.append(norm_col_row([num_topics, num_docs]))
        for epoch in range(parser.epoch):
            # update
            updateU(sess, model, layer, epoch, parser.in_epoch, parser.lr, toy_docs_mat, docs_mat, Us, Vs)
            updateV(sess, model, layer, epoch, parser.in_epoch, parser.lr, toy_docs_mat, docs_mat, Us, Vs, parser.b)
            # saving
            save_mat(parser.save_dir + "U-%d.txt" % layer, Us[-1])
            save_mat(parser.save_dir + "V-%d.txt" % layer, Vs[-1])
        num_topics *= num_split

