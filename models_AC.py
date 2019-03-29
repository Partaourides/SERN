import os
# Restrict the script to run on CPU
os.environ ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Import Keras Tensoflow Backend
# from keras import backend as K
import tensorflow as tf
# Configure it to use only specific CPU Cores
config = tf.ConfigProto(intra_op_parallelism_threads=4,
                        inter_op_parallelism_threads=4,
                        device_count={"CPU": 1, "GPU": 0},
                        allow_soft_placement=True)
tf.Session(config=config)
# import tensorflow as tf

# Formulation of the baseline emotion recognition model
# The input is a set of sentences padded on the word level
def SentenceModel(vocab_size, embedding_size, first_rnn_size, num_classes, dropout, embedding, num_speakers):
    # Sanity check
    tf.reset_default_graph()

    ####################################################################################################################
    # Placeholders and other needed variables
    ####################################################################################################################
    x = tf.placeholder(tf.int32, [None, None])
    speaker = tf.placeholder(tf.int32, [None, 2])
    seqlen = tf.placeholder(tf.int32, [None])
    y = tf.placeholder(tf.int32, [None])
    batch_size = tf.shape(x)[0]
    keep_prob = tf.constant(dropout)

    ########################################### MODEL STRUCTURE ########################################################

    ####################################################################################################################
    # Embedding layer
    ####################################################################################################################
    if embedding == 'glove':
        filename = '../glove.6B.' + str(embedding_size) + 'd.txt'

        def loadGloVe(filename):
            embd = []
            file = open(filename, 'r', encoding="utf-8")
            for line in file.readlines():
                row = line.strip().split(' ')
                embd.append([float(i) for i in row[1:]])
            print('Loaded GloVe Weights!')
            file.close()
            return embd

        glove_embd = loadGloVe(filename)
        glove_weights_initializer = tf.constant_initializer(glove_embd)
        embeddings = tf.get_variable(
            name='embeddings',
            shape=(len(glove_embd), embedding_size),
            initializer=glove_weights_initializer,
            trainable=False)

    if embedding == 'random':
        embeddings = tf.get_variable('embedding_matrix', [vocab_size, embedding_size])

    rnn_inputs = tf.nn.embedding_lookup(embeddings, x)
    rnn_inputs = tf.nn.dropout(rnn_inputs, keep_prob)

    ####################################################################################################################
    # Bidirectional RNN
    ####################################################################################################################
    # They say forget bias helps if its 1.0
    lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(first_rnn_size, forget_bias=1.0)
    lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(first_rnn_size, forget_bias=1.0)
    (fw, bw), final_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, rnn_inputs, dtype=tf.float32)
    rnn_outputs = tf.concat([fw, bw], axis=2)
    # Get the last output of the variable length sequence
    last_rnn_output = tf.gather_nd(rnn_outputs, tf.stack([tf.range(batch_size), seqlen - 1], axis=1))
    if num_speakers:
        last_rnn_output = tf.concat([tf.cast(speaker, tf.float32), last_rnn_output], 1)

    ####################################################################################################################
    # Final Dense layer to produce outputs
    ####################################################################################################################
    last_rnn_output = tf.nn.dropout(last_rnn_output, keep_prob)
    logits = tf.layers.dense(last_rnn_output, num_classes)

    ##################################### END OF MODEL STRUCTURE #######################################################

    ####################################################################################################################
    # Training Function
    ####################################################################################################################
    preds = tf.nn.softmax(logits)
    predictions = tf.cast(tf.argmax(preds, 1), tf.int32)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
    train_step = tf.train.AdamOptimizer(5e-3).minimize(loss)

    return {
        'x': x,
        'seqlen': seqlen,
        'speaker': speaker,
        'y': y,
        'ts': train_step,
        'preds': predictions
    }

# Formulation of the sentence dependent emotion recognition model
# The input is a set of sentences padded on word and sentence level
def DialogModel(vocab_size, embedding_size, first_rnn_size, second_rnn_size, num_classes, dropout, score_type, model_type,
                window_size, embedding, num_speakers):
    # Sanity check
    tf.reset_default_graph()

    ####################################################################################################################
    # Placeholders and other needed variables
    ####################################################################################################################
    x = tf.placeholder(tf.int32, [None, None])
    speaker = tf.placeholder(tf.int32, [None, 2])
    seqlen = tf.placeholder(tf.int32, [None])
    num_dialogs = tf.placeholder(tf.int32, [])
    y = tf.placeholder(tf.int32, [None, None])
    batch_size = tf.shape(x)[0]
    keep_prob = tf.constant(dropout)

    ###################################### MODEL STRUCTURE #############################################################

    ####################################################################################################################
    # Embedding layer
    ####################################################################################################################
    if embedding == 'glove':
        filename = '../glove.6B.' + str(embedding_size) + 'd.txt'

        def loadGloVe(filename):
            embd = []
            file = open(filename, 'r', encoding="utf-8")
            for line in file.readlines():
                row = line.strip().split(' ')
                embd.append([float(i) for i in row[1:]])
            print('Loaded GloVe Weights!')
            file.close()
            return embd

        glove_embd = loadGloVe(filename)
        glove_weights_initializer = tf.constant_initializer(glove_embd)
        embeddings = tf.get_variable(
            name='embeddings',
            shape=(len(glove_embd), embedding_size),
            initializer=glove_weights_initializer,
            trainable=False)

    if embedding == 'random':
        embeddings = tf.get_variable('embedding_matrix', [vocab_size, embedding_size])

    rnn_inputs = tf.nn.embedding_lookup(embeddings, x)
    rnn_inputs = tf.nn.dropout(rnn_inputs, keep_prob)

    ####################################################################################################################
    # Bidirectional RNN
    ####################################################################################################################
    # They say forget bias helps if its 1.0
    lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(first_rnn_size / 2, forget_bias=1.0)
    lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(first_rnn_size / 2, forget_bias=1.0)
    (fw, bw), final_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, rnn_inputs, dtype=tf.float32)
    rnn_outputs = tf.concat([fw, bw], axis=2)
    # Get the last output of the variable length sequence
    last_rnn_output = tf.gather_nd(rnn_outputs, tf.stack([tf.range(batch_size), seqlen - 1], axis=1))
    if num_speakers:
        last_rnn_output = tf.concat([tf.cast(speaker, tf.float32), last_rnn_output], 1)
        first_rnn_size += 2
    # Restructure the dialog format
    last_rnn_output = tf.reshape(last_rnn_output, (num_dialogs, -1, first_rnn_size))

    ####################################################################################################################
    # Sentence level RNN
    ####################################################################################################################
    cell = tf.nn.rnn_cell.GRUCell(second_rnn_size)
    # trainable initial state of RNN
    init_state = tf.Variable(tf.zeros([1, second_rnn_size]))
    init_state = tf.tile(init_state, [num_dialogs, 1])
    rnn_outputs2, final_state = tf.nn.dynamic_rnn(cell, last_rnn_output,
                                                  initial_state=init_state)  # sequence_length=seqlen

    ####################################################################################################################
    # Custom differential memory component for the incremental sentence length
    ####################################################################################################################

    # Dot score function
    if score_type == 'dot':
        window_size = int(window_size)
        def score_func(j, context, rnn_outputs2):
            if window_size != -1:
                start = tf.cond(j > window_size, lambda: j-window_size, lambda: 0)
                att = tf.einsum('ijm,im->ij', rnn_outputs2[:, start:j + 1, :], rnn_outputs2[:, j, :])
                att = tf.nn.softmax(att)
                context_temp = tf.einsum('ij,ijk->ik', att, rnn_outputs2[:, start:j + 1, :])
            else:
                att = tf.einsum('ijm,im->ij', rnn_outputs2[:, :j + 1, :], rnn_outputs2[:, j, :])
                att = tf.nn.softmax(att)
                context_temp = tf.einsum('ij,ijk->ik', att, rnn_outputs2[:, :j + 1, :])
            context_temp = tf.expand_dims(context_temp, axis=1)
            context = tf.cond(j > 0, lambda: tf.concat([context, context_temp], axis=1), lambda: context_temp)
            j = tf.add(j, tf.constant(1))
            return j, context, rnn_outputs2

    # General score function
    if score_type == 'general':
        W = tf.get_variable('score_W', [second_rnn_size, second_rnn_size])

        def score_func(j, context, rnn_outputs2):
            att = tf.einsum('ijm,im->ij', tf.tanh(tf.einsum('ijk,km->ijm', rnn_outputs2[:, :j + 1, :], W)),
                            rnn_outputs2[:, j, :])
            att = tf.nn.softmax(att)
            # att = tf.Print(att, [att], summarize=25)
            context_temp = tf.einsum('ij,ijk->ik', att, rnn_outputs2[:, :j + 1, :])
            context_temp = tf.expand_dims(context_temp, axis=1)
            context = tf.cond(j > 0, lambda: tf.concat([context, context_temp], axis=1), lambda: context_temp)
            j = tf.add(j, tf.constant(1))
            return j, context, rnn_outputs2

    # Concat score function
    if score_type == 'concat':
        W = tf.get_variable('score_W', [second_rnn_size, second_rnn_size / 2])
        B = tf.get_variable('score_B', [second_rnn_size / 2])
        U = tf.get_variable('score_U', [second_rnn_size / 2])

        def score_func(j, context, rnn_outputs2):
            att = tf.tanh(tf.add(tf.einsum('ijk,km->ijm', rnn_outputs2[:, :j + 1, :], W), B))
            att = tf.einsum('ijm,m->ij', att, U)
            att = tf.nn.softmax(att)
            #att = tf.Print(att, [att], summarize=25)
            context_temp = tf.einsum('ij,ijk->ik', att, rnn_outputs2[:, :j + 1, :])
            context_temp = tf.expand_dims(context_temp, axis=1)
            context = tf.cond(j > 0, lambda: tf.concat([context, context_temp], axis=1), lambda: context_temp)
            j = tf.add(j, tf.constant(1))
            return j, context, rnn_outputs2

    # Differential memory loop
    if model_type == 'Double_RNN_with_memory':
        j0 = tf.constant(0)
        context0 = tf.zeros([10, 10, second_rnn_size])
        cond = lambda j, m, rnn_outputs2: j < tf.shape(rnn_outputs2)[1]
        _, context, _ = tf.while_loop(cond, score_func, [j0, context0, rnn_outputs2],
                                      shape_invariants=[j0.get_shape(),
                                                        tf.TensorShape([None, None, second_rnn_size]),
                                                        rnn_outputs2.get_shape()],
                                      parallel_iterations=1)

    if model_type == 'RNN_with_memory':
        j0 = tf.constant(0)
        context0 = tf.zeros([10, 10, first_rnn_size])
        cond = lambda j, m, last_rnn_output: j < tf.shape(last_rnn_output)[1]
        _, context, _ = tf.while_loop(cond, score_func, [j0, context0, last_rnn_output],
                                      shape_invariants=[j0.get_shape(),
                                                        tf.TensorShape([None, None, first_rnn_size]),
                                                        last_rnn_output.get_shape()],
                                      parallel_iterations=1)

    ####################################################################################################################
    # Final Dense layer to produce outputs
    ####################################################################################################################
    if model_type == 'Double_RNN_with_memory' or model_type == 'RNN_with_memory':
        # TODO dropout option
        # Dropout before final layer
        context = tf.nn.dropout(context, keep_prob)
        logits = tf.layers.dense(context, num_classes)

    ##################################### END OF MODEL STRUCTURE #######################################################

    ####################################################################################################################
    # Training Function
    ####################################################################################################################
    preds = tf.nn.softmax(logits)
    predictions = tf.cast(tf.argmax(preds, 2), tf.int32)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
    train_step = tf.train.AdamOptimizer(5e-3).minimize(loss)

    return {
        'x': x,
        'seqlen': seqlen,
        'speaker': speaker,
        'y': y,
        'ts': train_step,
        'preds': predictions,
        'preds_': preds,
        'num_dialogs': num_dialogs
    }
