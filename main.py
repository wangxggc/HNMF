import tensorflow as tf
import numpy as np
from model import HOSCGrad
import codecs, math, sys, random
from collections import Counter


def load_data(data_file, min_hold):
    docs = [_.strip().split() for _ in codecs.open(data_file, "r", "utf8")]
    print (docs[0])
    docs = docs[1:]
    # import random
    # random.shuffle(docs)
    # docs = docs[0:5000]
    doc_ids = [i for i in range(len(docs))]
    random.shuffle(doc_ids)
    # doc_ids = doc_ids[0:5000]
    num_docs = len(doc_ids)

    wdf = Counter()
    for doc_id in doc_ids:
        doc = docs[doc_id]
        wdf.update(set(doc))
    widf = {}
    for k, v in wdf.items():
        if v < min_hold:
            continue
        widf[k] = math.log(float(num_docs) / float(v))

    id2w = widf.keys()
    w2id = dict([[w, i] for i, w in enumerate(id2w)])
    num_words = len(id2w)

    toy_docs_mat = np.zeros((num_words, 100), dtype=np.float32)
    docs_mat = [[], []]
    valid_ids = []
    non_zeros = 0
    for id, doc_id in enumerate(doc_ids):
        doc = docs[doc_id]
        wtf = Counter()
        wtf.update([w for w in doc if w in widf])
        if len(wtf) == 0:
            continue
        valid_ids.append(doc_id)
        sum = 0.

        indices, values = [], []
        for w, v in wtf.items():
            if w not in widf:
                continue
            indices.append([w2id[w], id])
            values.append(float(v)/len(doc) * widf[w])
            # values.append(float(v)/len(doc) * widf[w] / scale)
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
    parser.add_argument("--num_split", type=int, help="")
    parser.add_argument("--a", type=float, help="parameter alpha")
    parser.add_argument("--a0", type=float, help="parameter alpha0")
    parser.add_argument("--a1", type=float, help="parameter alpha1")
    parser.add_argument("--epoch", type=int, help="epoch per layer")
    parser.add_argument("--in_epoch", type=int, help="20")
    parser.add_argument("--min_hold", type=int, help="min hold for words")
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--save_dir", type=str, help="save models into here")
    parser = parser.parse_args()

    docs_mat, num_words, num_docs, w2id, id2w, valid_ids, toy_docs_mat = load_data(parser.data_file, min_hold=parser.min_hold)
    docs_mat = tf.SparseTensorValue(indices=docs_mat[0], values=docs_mat[1],
                                    dense_shape=[num_words, num_docs])
                                    # dense_shape=tf.TensorShape([tf.Dimension(num_words), tf.Dimension(num_docs)]))

    sout = codecs.open(parser.save_dir + "/wordmap.txt", "w", "utf8")
    sout.write("\n".join(id2w))
    sout.close()

    sout = codecs.open(parser.save_dir + "/docids.txt", "w", "utf8")
    sout.write("\n".join([str(i) for i in valid_ids]))
    sout.close()

    print (docs_mat.dense_shape)
    print ("Load documents %d, words %d" % (num_docs, num_words))

    num_topics = parser.init_k / parser.num_split
    num_layers = parser.num_layers
    num_split  = parser.num_split

    model = HOSCGrad(
        num_words, num_docs, parser.init_k, num_layers, num_split, parser.a, parser.a0, parser.a1
    )

    sess = tf.Session()

    Us, Vs = [norm_col([num_words, num_topics])], [norm_row([num_topics, num_docs])]

    def save_mat(filename, mat):
        sout = codecs.open(filename, "w", "utf8")
        sout.write("%d %d\n" % (mat.shape[0], mat.shape[1]))
        for r in range(mat.shape[0]):
            str = " ".join(["%.6f" % v for v in mat[r, :]])
            sout.write(str + "\n")
        sout.close()

    for layer in range(parser.num_layers):
        num_topics *= num_split
        print("layer %d, topic num %d" % (layer, num_topics))
        # init
        Us.append(norm_col_row([num_words, num_topics]))
        Vs.append(norm_col_row([num_topics, num_docs]))
        for epoch in range(parser.epoch):
            # update
            updateU(sess, model, layer, epoch, parser.in_epoch, parser.lr, toy_docs_mat, docs_mat, Us, Vs)
            updateV(sess, model, layer, epoch, parser.in_epoch, parser.lr, toy_docs_mat, docs_mat, Us, Vs)
            # saving
            save_mat(parser.save_dir + "U-%d.txt" % layer, Us[-1])
            save_mat(parser.save_dir + "V-%d.txt" % layer, Vs[-1])
