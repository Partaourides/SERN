
# import tensorflow as tf

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
# tf.Session(config=config)


import numpy as np
from IEOMAP_dataset_AC import dataset, IeomapDialogIterator
from sklearn.metrics import confusion_matrix
from models_AC import DialogModel
import json
import os


def model_with_dialog_dependence(n_run, epochs, batch_size, embedding_size, first_rnn_size, second_rnn_size,
                                 dropout, score_type, model_type, window_size, embedding, num_speakers, primal_emotion, num_desired_classes):
    ########################################################################################################################
    # Hyper-parameters
    ########################################################################################################################
    split_size = 0.8

    log_dir = './logs_AC/' + model_type + '_' + str(num_speakers) + '_' + str(num_desired_classes) + '/' + str(n_run) + '/'

    if window_size != '-1':
        log_dir = './logs_AC/window_size/' + model_type + '_' + str(num_speakers) + '/' + 'window_size_' + str(window_size) + '/' + str(n_run) + '/'

    if primal_emotion != 'off':
        log_dir = './logs_AC/' + model_type + '_' + str(num_speakers) + '_' + primal_emotion + '/' + str(n_run) + '/'
    train_log_dir = log_dir + 'train'
    val_log_dir = log_dir + 'val'

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    ########################################################################################################################
    # Initialize the Data set
    ########################################################################################################################
    sentences, targets, data_info, speakers = dataset(mode='dialog',
                                                      embedding=embedding,
                                                      embedding_size=embedding_size,
                                                      primal_emotion=primal_emotion,
                                                      num_desired_classes=num_desired_classes)

    train_data = IeomapDialogIterator(sentences[0], targets[0], data_info['sentences_length'][0], speakers[0])
    val_data = IeomapDialogIterator(sentences[1], targets[1], data_info['sentences_length'][1], speakers[1])
    test_data = IeomapDialogIterator(sentences[2], targets[2], data_info['sentences_length'][2], speakers[2])

    ########################################################################################################################
    # Initialize the model
    ########################################################################################################################
    g = DialogModel(vocab_size=(data_info['vocabulary_size'] + 1),
                    embedding_size=embedding_size,
                    first_rnn_size=first_rnn_size,
                    second_rnn_size=second_rnn_size,
                    num_classes=data_info['num_classes'],
                    dropout=dropout,
                    score_type=score_type,
                    model_type=model_type,
                    window_size=window_size,
                    embedding=embedding,
                    num_speakers=num_speakers)

    # Store model setup
    model_setup = {'vocab_size': (data_info['vocabulary_size'] + 1),
                    'embedding_size': embedding_size,
                    'first_rnn_size': first_rnn_size,
                    'second_rnn_size': second_rnn_size,
                    'num_classes': data_info['num_classes'],
                    'dropout': dropout,
                    'score_type': score_type,
                    'model_type': model_type,
                    'window_size': window_size,
                    'embedding': embedding,
                    'num_speakers': num_speakers}
    with open(log_dir + 'model_setup.p', 'w') as file:
        json.dump(model_setup, file, indent=4)

    ########################################################################################################################
    # Initialize the parameters
    ########################################################################################################################
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    saver = tf.train.Saver(max_to_keep=0)
    epoch = 0
    best_epoch = 0
    train_conf_matrix = 0
    val_conf_matrix = 0
    test_conf_matrix = 0
    best_acc = 0
    best_precision = 0
    best_recall = 0
    store_info = {}

    ########################################################################################################################
    # Performance Indicators
    ########################################################################################################################
    writer_train = tf.summary.FileWriter(train_log_dir, sess.graph)
    writer_val = tf.summary.FileWriter(val_log_dir)

    accuracy_tf = tf.placeholder(tf.float32, [])
    precision_tf = tf.placeholder(tf.float32, [])
    recall_tf = tf.placeholder(tf.float32, [])

    summary_op = tf.summary.scalar('accuracy', accuracy_tf)
    summary_op = tf.summary.scalar('precision', precision_tf)
    summary_op = tf.summary.scalar('recall', recall_tf)

    ########################################################################################################################
    # Model training procedure
    ########################################################################################################################
    while train_data.epoch < epochs: #  and train_data.epoch < best_epoch + 20:
        sentences_batch, sentences_length_batch, targets_batch, speakers_batch = train_data.next_batch(batch_size)
        preds, _ = sess.run([g['preds'],
                             g['ts']],
                            feed_dict={g['x']: np.array(sentences_batch),
                                       g['y']: np.array(targets_batch),
                                       g['seqlen']: sentences_length_batch[1],
                                       g['speaker']: np.array(speakers_batch),
                                       g['num_dialogs']: len(sentences_length_batch[0])})

        ####################################################################################################################
        # Calculate the Train data Confusion Matrix
        ####################################################################################################################
        predictions = [preds[i, :j] for i, j in enumerate(sentences_length_batch[0])]
        targets_batch = [targets_batch[i][:j] for i, j in enumerate(sentences_length_batch[0])]
        temp = 0
        for y, y_ in zip(targets_batch, predictions):
            temp += confusion_matrix(y, y_, labels=range(data_info['num_classes']))
        train_conf_matrix += temp

        ####################################################################################################################
        # Add the end of each training epoch compute the test results and store the relevant information
        ####################################################################################################################
        if train_data.epoch != epoch:
            while val_data.epoch == epoch:
                sentences_batch, sentences_length_batch, targets_batch, speakers_batch = val_data.next_batch(
                    batch_size)
                preds = sess.run([g['preds']],
                                 feed_dict={g['x']: np.array(sentences_batch),
                                            g['y']: np.array(targets_batch),
                                            g['seqlen']: sentences_length_batch[1],
                                            g['speaker']: np.array(speakers_batch),
                                            g['num_dialogs']: len(sentences_length_batch[0])})

                ############################################################################################################
                # Calculate the Test data Confusion Matrix
                ############################################################################################################
                predictions = [preds[0][i, :j] for i, j in enumerate(sentences_length_batch[0])]
                targets_batch = [targets_batch[i][:j] for i, j in enumerate(sentences_length_batch[0])]

                temp = 0
                for y, y_ in zip(targets_batch, predictions):
                    temp += confusion_matrix(y, y_, labels=range(data_info['num_classes']))

                val_conf_matrix += temp
            ################################################################################################################
            # Compute Accuracy, Precision and Recall
            ################################################################################################################
            train_CM_size = len(train_conf_matrix)
            total_train = sum(sum(train_conf_matrix))
            # TODO: fix TN they are all wrong but fortunately unused (replace size with total)
            train_TP = np.diagonal(train_conf_matrix)
            train_FP = [sum(train_conf_matrix[:, i]) - train_TP[i] for i in range(train_CM_size)]
            train_FN = [sum(train_conf_matrix[i, :]) - train_TP[i] for i in range(train_CM_size)]
            train_TN = train_CM_size - train_TP - train_FP - train_FN

            train_precision = train_TP / (train_TP + train_FP)  # aka True Positive Rate
            train_recall = train_TP / (train_TP + train_FN)

            total_train_correct = sum(train_TP)
            total_train_accuracy = total_train_correct / total_train
            total_train_precision = sum(train_precision) / train_CM_size
            total_train_recall = sum(train_recall) / train_CM_size

            val_CM_size = len(val_conf_matrix)
            total_val = sum(sum(val_conf_matrix))

            val_TP = np.diagonal(val_conf_matrix)
            val_FP = [sum(val_conf_matrix[:, i]) - val_TP[i] for i in range(val_CM_size)]
            val_FN = [sum(val_conf_matrix[i, :]) - val_TP[i] for i in range(val_CM_size)]
            val_TN = val_CM_size - val_TP - val_FP - val_FN

            val_precision = val_TP / (val_TP + val_FP)
            val_recall = val_TP / (val_TP + val_FN)

            total_val_correct = sum(val_TP)
            total_val_accuracy = total_val_correct / total_val
            total_val_precision = sum(val_precision) / val_CM_size
            total_val_recall = sum(val_recall) / val_CM_size

            ################################################################################################################
            # Store Accuracy Precision Recall
            ################################################################################################################
            train_acc_summary = tf.Summary(
                value=[tf.Summary.Value(tag="accuracy", simple_value=total_train_accuracy), ])
            train_prec_summary = tf.Summary(
                value=[tf.Summary.Value(tag="precision", simple_value=total_train_precision), ])
            train_rec_summary = tf.Summary(
                value=[tf.Summary.Value(tag="recall", simple_value=total_train_recall), ])

            val_acc_summary = tf.Summary(
                value=[tf.Summary.Value(tag="accuracy", simple_value=total_val_accuracy), ])
            val_prec_summary = tf.Summary(
                value=[tf.Summary.Value(tag="precision", simple_value=total_val_precision), ])
            val_rec_summary = tf.Summary(value=[tf.Summary.Value(tag="recall", simple_value=total_val_recall), ])

            writer_train.add_summary(train_acc_summary, epoch)
            writer_train.add_summary(train_prec_summary, epoch)
            writer_train.add_summary(train_rec_summary, epoch)

            writer_val.add_summary(val_acc_summary, epoch)
            writer_val.add_summary(val_prec_summary, epoch)
            writer_val.add_summary(val_rec_summary, epoch)

            writer_val.flush()

            ################################################################################################################
            # Print the confusion matrix and store important information
            ################################################################################################################
            print(train_conf_matrix)
            print(val_conf_matrix)

            for best, current, name in zip([best_acc], [total_val_accuracy], ['acc']):
                if best < current:
                    saver.save(sess, log_dir + name + "_best_validation_model.ckpt")
                    if name == 'acc':
                        best_acc = current
                    if name == 'pre':
                        best_precision = current
                    if name == 'rec':
                        best_recall = current
                    best_epoch = val_data.epoch
                    store_info[name] = {'epoch': best_epoch,
                                  'train_conf_matrix': list([list(x) for x in train_conf_matrix]),
                                  'train_accuracy': total_train_accuracy,
                                  'train_precision': list(train_precision),
                                  'total_train_precision': total_train_precision,
                                  'train_recall': list(train_recall),
                                  'total_train_recall': total_train_recall,
                                  'val_conf_matrix': list([list(x) for x in val_conf_matrix]),
                                  'val_accuracy': total_val_accuracy,
                                  'val_precision': list(val_precision),
                                  'total_val_precision': total_val_precision,
                                  'val_recall': list(val_recall),
                                  'total_val_recall': total_val_recall}

            store_convergence_info = {'epoch': train_data.epoch,
                                      'train_conf_matrix': list([list(x) for x in train_conf_matrix]),
                                      'train_accuracy': total_train_accuracy,
                                      'train_precision': list(train_precision),
                                      'total_train_precision': total_train_precision,
                                      'train_recall': list(train_recall),
                                      'total_train_recall': total_train_recall,
                                      'val_conf_matrix': list([list(x) for x in val_conf_matrix]),
                                      'val_accuracy': total_val_accuracy,
                                      'val_precision': list(val_precision),
                                      'total_val_precision': total_val_precision,
                                      'val_recall': list(val_recall),
                                      'total_val_recall': total_val_recall}

            ################################################################################################################
            # Get ready for the next epoch
            ################################################################################################################
            epoch += 1
            train_conf_matrix = 0
            val_conf_matrix = 0
            ################################################################################################################

    ####################################################################################################################
    # Add the end of training compute the test results and store the relevant information
    ####################################################################################################################
    while test_data.epoch == 0:
        sentences_batch, sentences_length_batch, targets_batch, speakers_batch = test_data.next_batch(
            batch_size)
        preds = sess.run([g['preds']],
                         feed_dict={g['x']: np.array(sentences_batch),
                                    g['y']: np.array(targets_batch),
                                    g['seqlen']: sentences_length_batch[1],
                                    g['speaker']: np.array(speakers_batch),
                                    g['num_dialogs']: len(sentences_length_batch[0])})

        ############################################################################################################
        # Calculate the Test data Confusion Matrix
        ############################################################################################################
        predictions = [preds[0][i, :j] for i, j in enumerate(sentences_length_batch[0])]
        targets_batch = [targets_batch[i][:j] for i, j in enumerate(sentences_length_batch[0])]

        temp = 0
        for y, y_ in zip(targets_batch, predictions):
            temp += confusion_matrix(y, y_, labels=range(data_info['num_classes']))

        test_conf_matrix += temp

    ################################################################################################################
    # Compute Accuracy, Precision and Recall
    ################################################################################################################
    test_CM_size = len(test_conf_matrix)
    total_test = sum(sum(test_conf_matrix))

    test_TP = np.diagonal(test_conf_matrix)
    test_FP = [sum(test_conf_matrix[:, i]) - test_TP[i] for i in range(test_CM_size)]
    test_FN = [sum(test_conf_matrix[i, :]) - test_TP[i] for i in range(test_CM_size)]
    test_TN = test_CM_size - test_TP - test_FP - test_FN

    test_precision = test_TP / (test_TP + test_FP)
    test_recall = test_TP / (test_TP + test_FN)

    total_test_correct = sum(test_TP)
    total_test_accuracy = total_test_correct / total_test
    total_test_precision = sum(test_precision) / test_CM_size
    total_test_recall = sum(test_recall) / test_CM_size

    ################################################################################################################
    # Print the confusion matrix and store important information
    ################################################################################################################
    print(test_conf_matrix)

    store_convergence_info['test_conf_matrix'] = list([list(x) for x in test_conf_matrix])
    store_convergence_info['test_accuracy'] = total_test_accuracy
    store_convergence_info['test_precision'] = list(test_precision)
    store_convergence_info['total_test_precision'] = total_test_precision
    store_convergence_info['test_recall'] = list(test_recall)
    store_convergence_info['total_test_recall'] = total_test_recall

    # trick to be able to save numpy.int64 into json
    def default(o):
        if isinstance(o, np.int64): return int(o)
        raise TypeError

    with open(log_dir + 'convergence_results.p', 'w') as file:
        json.dump(store_convergence_info, file, default=default, indent=4)


    saver.save(sess, log_dir + "convergence_model.ckpt")

    ####################################################################################################################
    # Add the end of training compute the test results of the best validation model and store the relevant information
    ####################################################################################################################
    # for name in ['acc', 'pre', 'rec']:
    # for name in ['acc']:
    name = 'acc'
    saver.restore(sess, log_dir + name + "_best_validation_model.ckpt")
    test_conf_matrix = 0 # FIXME
    epoch_ = test_data.epoch
    while epoch_ == test_data.epoch:
        sentences_batch, sentences_length_batch, targets_batch, speakers_batch = test_data.next_batch(
            batch_size)
        preds = sess.run([g['preds']],
                         feed_dict={g['x']: np.array(sentences_batch),
                                    g['y']: np.array(targets_batch),
                                    g['seqlen']: sentences_length_batch[1],
                                    g['speaker']: np.array(speakers_batch),
                                    g['num_dialogs']: len(sentences_length_batch[0])})

        ############################################################################################################
        # Calculate the Test data Confusion Matrix
        ############################################################################################################
        predictions = [preds[0][i, :j] for i, j in enumerate(sentences_length_batch[0])]
        targets_batch = [targets_batch[i][:j] for i, j in enumerate(sentences_length_batch[0])]

        temp = 0
        for y, y_ in zip(targets_batch, predictions):
            temp += confusion_matrix(y, y_, labels=range(data_info['num_classes']))

        test_conf_matrix += temp

    ################################################################################################################
    # Compute Accuracy, Precision and Recall
    ################################################################################################################
    test_CM_size = len(test_conf_matrix)
    total_test = sum(sum(test_conf_matrix))

    test_TP = np.diagonal(test_conf_matrix)
    test_FP = [sum(test_conf_matrix[:, i]) - test_TP[i] for i in range(test_CM_size)]
    test_FN = [sum(test_conf_matrix[i, :]) - test_TP[i] for i in range(test_CM_size)]
    test_TN = test_CM_size - test_TP - test_FP - test_FN

    test_precision = test_TP / (test_TP + test_FP)
    test_recall = test_TP / (test_TP + test_FN)

    total_test_correct = sum(test_TP)
    total_test_accuracy = total_test_correct / total_test
    total_test_precision = sum(test_precision) / test_CM_size
    total_test_recall = sum(test_recall) / test_CM_size

    ################################################################################################################
    # Print the confusion matrix and store important information
    ################################################################################################################
    print(test_conf_matrix)

    store_info[name]['test_conf_matrix'] = list([list(x) for x in test_conf_matrix])
    store_info[name]['test_accuracy'] = total_test_accuracy
    store_info[name]['test_precision'] = list(test_precision)
    store_info[name]['total_test_precision'] = total_test_precision
    store_info[name]['test_recall'] = list(test_recall)
    store_info[name]['total_test_recall'] = total_test_recall

    # trick to be able to save numpy.int64 into json
    def default(o):
        if isinstance(o, np.int64): return int(o)
        raise TypeError

    with open(log_dir + name + '_best_validation_results.p', 'w') as file:
        json.dump(store_info[name], file, default=default, indent=4)

