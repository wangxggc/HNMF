import tensorflow as tf
import numpy as np

def get_initializer(name):
    if name == "xavier":
        print("get xavier initialzer")
        return tf.contrib.layers.xavier_initializer(uniform=True, seed=None)
    elif name == "he":
        print("get he initializer")
        return tf.keras.initializers.he_normal(seed=None)
    else:
        print("get random normal initizlier")
        return tf.random_normal_initializer(stddev=0.1)


class HOSCGrad(object):
    def __init__(self, num_words, num_docs, num_topics, num_layers, num_split, alpha, alpha0, alpha1):
        self.u_grads = []
        self.v_grads = []
        self.inputs = []
        self.losses = []

        self.D = tf.sparse_placeholder(dtype=tf.float32, name="D")
        # used for print loss, top-100 docs in D
        self.Ds = tf.placeholder(dtype=tf.float32, shape=[num_words, 100], name="Ds")

        u_grad, v_grad, loss, inputs = self.get_init_grad("init", num_topics, num_words, num_docs, alpha)
        self.u_grads.append(u_grad)
        self.v_grads.append(v_grad)
        self.inputs.append({
            "D":self.D,
            "U":inputs[0],
            "V":inputs[1]
        })
        self.losses.append(loss)
        print ("init layer 0, topic num %d" % num_topics)

        for l in range(1, num_layers):
            num_topics = num_topics * num_split
            print("init layer %d, topic num %d" % (l, num_topics))

            u_grad, v_grad, loss, inputs = self.get_next_layer_grad("layer-%d" % l, num_topics, num_words, num_docs, num_split, alpha, alpha0, alpha1)

            self.u_grads.append(u_grad)
            self.v_grads.append(v_grad)
            self.inputs.append({
                "D":self.D,
                "U-pre":inputs[0],
                "V-pre":inputs[1],
                "U-cur":inputs[2],
                "V-cur":inputs[3]
            })
            self.losses.append(loss)
        print ("Init self.Done, layers = %d, topics = %s" % (num_layers, [str(num_topics) for l in range(1, num_layers+1)]))

    def D_mul_U(self, D, U, num_topics, num_parts=10):
        size_part = num_topics // num_parts
        miniUs = [U[:, size_part*i:size_part*(i+1)] for i in range(num_parts)]
        if size_part * num_parts < num_topics:
            miniUs.append(U[:, size_part*num_parts:])
        return tf.concat(
            [tf.sparse_tensor_dense_matmul(D, miniU) for miniU in miniUs], axis=1
        )

    def get_init_grad(self, name, num_topics, num_words, num_docs, alpha):
        U = tf.placeholder(dtype=tf.float32, shape=[num_words, num_topics], name="U-%s" % (name))
        V = tf.placeholder(dtype=tf.float32, shape=[num_topics, num_docs],  name="V-%s" % (name))
        D = tf.sparse_reorder(self.D)

        VVT = tf.matmul(V, V, transpose_b=True)
        UTU = tf.matmul(U, U, transpose_a=True)
        DVT = self.D_mul_U(D, tf.transpose(V), num_topics)
        eye = tf.eye(num_topics, dtype=tf.float32)
        u_grads = [
            # 2.0 * tf.matmul(UV - D, V, transpose_b=True),
            2.0 * (tf.matmul(U, VVT) - DVT),
            4.0 * alpha * tf.matmul(U, UTU - eye)
        ]

        UTD = tf.transpose(self.D_mul_U(tf.sparse_transpose(D), U, num_topics))
        v_grads = [2.0 * (tf.matmul(UTU, V) - UTD)]

        u_grads = tf.reduce_sum(tf.stack(u_grads, axis=0), axis=0)
        v_grads = tf.reduce_sum(tf.stack(v_grads, axis=0), axis=0)

        # loss = tf.reduce_sum([
        #     tf.reduce_sum(tf.norm(D - UV, ord=2)),
        #     tf.reduce_sum(tf.norm(tf.matmul(U, U, transpose_a=True) - eye, ord=2))
        # ])
        UV = tf.matmul(U, V[:, 0:100])
        loss = tf.reduce_mean(tf.norm(self.Ds-UV, ord=2))
        return u_grads, v_grads, loss, [U, V]

    def get_next_layer_grad(self, name, num_topics, num_words, num_docs, num_split, alpha, alpha0, alpha1):
        Up = tf.placeholder(dtype=tf.float32, shape=[num_words, num_topics / num_split], name="Up_%s" % (name))
        Vp = tf.placeholder(dtype=tf.float32, shape=[num_topics / num_split, num_docs],  name="Vp_%s" % (name))
        U = tf.placeholder(dtype=tf.float32, shape=[num_words, num_topics], name="U_%s" % (name))
        V = tf.placeholder(dtype=tf.float32, shape=[num_topics, num_docs], name="U_%s" % (name))
        D = tf.sparse_reorder(self.D)

        VVT = tf.matmul(V, V, transpose_b=True)
        DVT = self.D_mul_U(D, tf.transpose(V), num_topics)
        u_grads = [2.0 * (tf.matmul(U, VVT) - DVT)]

        UTU = tf.matmul(U, U, transpose_a=True)
        UTD = tf.transpose(self.D_mul_U(tf.sparse_transpose(D), U, num_topics))
        eye = tf.eye(num_topics, dtype=tf.float32)
        
        u_grads.append(4.0 * alpha * tf.matmul(U, UTU - eye))
        v_grads = [2.0 * (tf.matmul(UTU, V) - UTD)]

        u_grads_pieces_c = []
        v_grads_pieces_c = []
        u_grads_pieces_o = []

        eye = tf.eye(num_split, dtype=tf.float32)

        loss_piece = []
        for i in range(num_topics/num_split):
            Up_piece = tf.expand_dims(Up[:, i], axis=1)
            Vp_piece = tf.expand_dims(Vp[i, :], axis=0)
            U_piece = U[:, i*num_split:(i+1)*num_split]
            V_piece = V[i*num_split:(i+1)*num_split, :]

            UVVT = tf.matmul(U_piece, tf.matmul(V_piece, V_piece, transpose_b=True))
            UpVpVT = tf.matmul(Up_piece, tf.matmul(Vp_piece, V_piece, transpose_b=True))
            u_grads_pieces_c.append(2.0 * alpha0 * (UVVT - UpVpVT))

            UTU = tf.matmul(U_piece, U_piece, transpose_a=True)
            UTUV = tf.matmul(UTU, V_piece)
            UTUpVp = tf.matmul(tf.matmul(U_piece, Up_piece, transpose_a=True), Vp_piece)
            v_grads_pieces_c.append(2.0 * alpha0 * (UTUV - UTUpVp))

            u_grads_pieces_o.append(4.0 * alpha1 * tf.matmul(U_piece, UTU - eye))

        u_grads.append(tf.concat(u_grads_pieces_c, axis=1))
        u_grads.append(tf.concat(u_grads_pieces_o, axis=1))
        v_grads.append(tf.concat(v_grads_pieces_c, axis=0))

        u_grads = tf.reduce_sum(tf.stack(u_grads, axis=0), axis=0)
        v_grads = tf.reduce_sum(tf.stack(v_grads, axis=0), axis=0)

        # loss = tf.reduce_sum([
        #     tf.reduce_sum(tf.norm(D - UV, ord=2)),
        # ])
        UV = tf.matmul(U, V[:, 0:100])
        loss = tf.reduce_mean(tf.norm(self.Ds-UV, ord=2))

        return u_grads, v_grads, loss, [Up, Vp, U, V]
