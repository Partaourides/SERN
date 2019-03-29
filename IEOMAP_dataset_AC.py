import csv
import glob
import gensim
import nltk
import numpy as np
from typing import Dict, List, TextIO
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import shuffle
from collections import Counter
import json

# nltk.download('punkt')

# When calling the function the user must decide if he/she would like to have the speaker information
# speaker_info = False

# The function that prepares the dataset for emotion recognition.
# mode = 'dialog' returns the dataset in chat format (DEFAULT)
# mode = 'sentence' returns the dataset in sentences format

def dataset(mode='dialog', embedding='random', embedding_size='50', primal_emotion='off', num_desired_classes=6, return_dialogs=False): # ang sad fru
    # Get the dataset files
    dialogs_files = glob.glob("IEOMAP_dataset/Sentences/*.txt")
    dialogs_emotions_files = glob.glob("IEOMAP_dataset/Classification/*.txt")

    dialogs_normalized = []
    dialogs: List[List[Dict[str, str]]] = []
    words: List[str] = []
    emotions_corpus: List[str] = []

    for i, j in zip(dialogs_files, dialogs_emotions_files):
        with open(i) as utterances_file, open(j) as emotions_file:
            utterances_iter: object = csv.reader(utterances_file, delimiter="&")
            emotions_iter: object = csv.reader(emotions_file, delimiter="&")

            dialog_normalized = []
            dialog: List[Dict[str, str]] = []
            dialog_emotions: Dict[str, str] = {}

            # Parser for extracting the emotion from the dataset
            for emotion_gross in emotions_iter:
                if emotion_gross != [] and emotion_gross[0][0] == '[':
                    utterance_id: str = emotion_gross[0].split('\t')[1].split('_')[-1]
                    emotion: str = emotion_gross[0].split('\t')[2]

                    dialog_emotions[utterance_id] = emotion

            # Parser for extracting the needed information for the dataset
            for count, utterance_gross in enumerate(utterances_iter):
                # The following 'if' is to deal with a type of inconsistency in the files
                if utterance_gross[0][0] != 'M' and utterance_gross[0][0] != 'F':
                    line_information: List[str] = utterance_gross[0].split(' ', 2)
                    utterance_id = line_information[0].split('_')[-1]
                    speaker: str = utterance_id[0]
                    # Replaced gensim with ntlk that has better tokenizers
                    # TODO Still need better preprocessing
                    # TODO Check http://www.nltk.org/api/nltk.tokenize.html#module-nltk.tokenize.casual
                    utterance: List[str] = nltk.word_tokenize(line_information[2])
                    utterance = [word.lower() for word in utterance]

                    # The following 'if' is to deal with another type of inconsistency in the files
                    if num_desired_classes == 4 or num_desired_classes == 5:
                        removed_classes = ['fru', 'xxx', 'fea', 'oth', 'dis', 'sur']
                    else:
                        removed_classes = ['xxx', 'fea', 'oth', 'dis', 'sur']

                    if 'X' not in utterance_id and dialog_emotions[utterance_id] not in removed_classes: # FIXME for 5-class and 4-class classification
                        dialog.append(
                            {'speaker': speaker, 'sentence': utterance, 'emotion': dialog_emotions[utterance_id]})
                        dialog_normalized.append(line_information[2])
                        words += utterance
                        emotions_corpus.append(dialog_emotions[utterance_id])

        dialogs.append(dialog)
        dialogs_normalized.append(dialog_normalized)

    # Create the words ditionary for the word2int transformation
    if embedding == 'random':
        # Remove words with <5 frequency (4276 --> 1669)
        words_count = Counter(words)
        words = set(words)
        for i in words_count:
            if words_count[i] < 5:
                words.remove(i)
        dictionary = {word: id + 1 for id, word in enumerate(words)}
        unknown = 0
        with open('dataset_dictionary.json', 'w') as fp:
            json.dump(dictionary, fp)
    if embedding == 'glove':
        # initialize glove dictionary
        filename = '../glove.6B.' + str(embedding_size) +'d.txt'

        def loadGloVe(filename):
            vocab = []
            file = open(filename, 'r', encoding="utf-8")
            for line in file.readlines():
                row = line.strip().split(' ')
                vocab.append(row[0])
            print('Loaded GloVe!')
            file.close()
            return vocab

        glove_vocab = loadGloVe(filename)
        dictionary = {glove_vocab[count]: count for count in range(len(glove_vocab))}
        unknown = dictionary['unk']

    emotions = np.unique(emotions_corpus)
    emotions_count = Counter(emotions_corpus)
    emotions_count = [emotions_count[x] for x in emotions]

    unwanted_emotions = []
    for count, emotion in zip(emotions_count, emotions):
        if count < 500:
            unwanted_emotions.append(emotion)

    # emotions_corpus = ['xxx' if d in unwanted_emotions else d for d in emotions_corpus]
    if num_desired_classes == 4:
        emotions_corpus = ['hap' if d == 'exc' else d for d in emotions_corpus] # FIXME for 4-class classification

    if primal_emotion != 'off':
        emotions_corpus = ['xxx' if d != primal_emotion else d for d in emotions_corpus]

    # Transform the target (emotions) to their integer ID
    enc = OrdinalEncoder()
    sentences_targets: List[int] = enc.fit_transform(np.array(emotions_corpus).reshape(-1, 1))
    print(enc.categories_)

    sentences_sources: List[List[int]] = []
    sentences_speakers: List[List[int]] = []

    dialogs_sources_: List[List[List[int]]] = []
    dialogs_speakers_: List[List[List[int]]] = []
    dialogs_targets_: List[List[int]] = []

    # Prepare both the sentences and dialogs dataset
    for dialog in dialogs:
        utterances_sources: List[List[int]] = []
        utterances_speakers: List[List[int]] = []
        utterances_targets: List[str] = []
        temp_sentences_length: List[int] = []

        for utterance in dialog:
            # Transform the sentence into a list of integers based on the dictionary (word corpus)
            sentence = [dictionary[word] if word in dictionary else unknown for word in utterance['sentence']]
            utterances_sources.append(sentence)
            utterances_speakers.append([0,1] if utterance['speaker'] == 'F' else [1,0])
            # sentences_length_1D.append(len(sentence))
            temp_sentences_length.append(len(sentence))
            temp_append = utterance['emotion'] # if utterance['emotion'] not in unwanted_emotions else 'xxx'
            if num_desired_classes == 4:
                temp_append = temp_append if temp_append != 'exc' else 'hap' # FIXME for 4-class classification
            if primal_emotion != 'off':
                temp_append = temp_append if temp_append == primal_emotion else 'xxx'
            utterances_targets.append(temp_append)

        dialogs_sources_.append(utterances_sources)
        dialogs_speakers_.append(utterances_speakers)
        dialogs_targets_.append(enc.transform(np.array(utterances_targets).reshape(-1, 1)))
        sentences_sources += utterances_sources
        sentences_speakers += utterances_speakers

    # Default Train-Validation-Test split (112-8-31)
    train_list = []
    validation_list = []
    test_list = []
    for dialog_id, dialog_name in enumerate(dialogs_files):
        if 'Ses05' in dialog_name:
            test_list.append(dialog_id)
        elif 'F_impro02' in dialog_name or 'F_script02_1' in dialog_name:
            validation_list.append(dialog_id)
        else:
            train_list.append(dialog_id)

    train_dialogs_sources = [dialogs_sources_[i] for i in train_list]
    train_sentences_length_2D = [[len(sentence_) for sentence_ in dialog_] for dialog_ in train_dialogs_sources]
    train_sentences_sources = [sentence for dialog in train_dialogs_sources for sentence in dialog]
    train_sentences_length_1D = [len(sentence_) for dialog_ in train_dialogs_sources for sentence_ in dialog_]

    validation_dialogs_sources = [dialogs_sources_[i] for i in validation_list]
    validation_sentences_length_2D = [[len(sentence_) for sentence_ in dialog_] for dialog_ in validation_dialogs_sources]
    validation_sentences_sources = [sentence for dialog in validation_dialogs_sources for sentence in dialog]
    validation_sentences_length_1D = [len(sentence_) for dialog_ in validation_dialogs_sources for sentence_ in dialog_]

    test_dialogs_sources = [dialogs_sources_[i] for i in test_list]
    test_sentences_length_2D = [[len(sentence_) for sentence_ in dialog_] for dialog_ in test_dialogs_sources]
    test_sentences_sources = [sentence for dialog in test_dialogs_sources for sentence in dialog]
    test_sentences_length_1D = [len(sentence_) for dialog_ in test_dialogs_sources for sentence_ in dialog_]

    train_dialogs_speakers = [dialogs_speakers_[i] for i in train_list]
    train_sentences_speakers = [speaker for dialog in train_dialogs_speakers for speaker in dialog]

    validation_dialogs_speakers = [dialogs_speakers_[i] for i in validation_list]
    validation_sentences_speakers = [speaker for dialog in validation_dialogs_speakers for speaker in dialog]

    test_dialogs_speakers = [dialogs_speakers_[i] for i in test_list]
    test_sentences_speakers = [speaker for dialog in test_dialogs_speakers for speaker in dialog]

    train_dialogs_targets = [dialogs_targets_[i] for i in train_list]
    train_sentences_targets = [target for dialog in train_dialogs_targets for target in dialog]

    validation_dialogs_targets = [dialogs_targets_[i] for i in validation_list]
    validation_sentences_targets = [target for dialog in validation_dialogs_targets for target in dialog]

    test_dialogs_targets = [dialogs_targets_[i] for i in test_list]
    test_sentences_targets = [target for dialog in test_dialogs_targets for target in dialog]

    dialogs_sources = [train_dialogs_sources, validation_dialogs_sources, test_dialogs_sources]
    sentences_length_2D = [train_sentences_length_2D, validation_sentences_length_2D, test_sentences_length_2D]
    dialogs_speakers = [train_dialogs_speakers, validation_dialogs_speakers, test_dialogs_speakers]
    dialogs_targets = [train_dialogs_targets, validation_dialogs_targets, test_dialogs_targets]

    sentences_sources = [train_sentences_sources, validation_sentences_sources, test_sentences_sources]
    sentences_length_1D = [train_sentences_length_1D, validation_sentences_length_1D, test_sentences_length_1D]
    sentences_speakers = [train_sentences_speakers, validation_sentences_speakers, test_sentences_speakers]
    sentences_targets = [train_sentences_targets, validation_sentences_targets, test_sentences_targets]

    # Relevant information for the the sentences dataset
    if mode == 'sentences':
        data_info = {
            'vocabulary_size': len(dictionary),
            'num_classes': len(enc.categories_[0]),
            'sentences_length': sentences_length_1D}
        return sentences_sources, sentences_targets, data_info, sentences_speakers

    # Relevant information for the dialogs dataset
    data_info = {
        'sentences_length': sentences_length_2D,
        'vocabulary_size': len(dictionary),
        'num_classes': len(enc.categories_[0])}

    if return_dialogs:
        return dialogs_sources, dialogs_targets, data_info, dialogs_speakers, dialogs_normalized
    else:
        return dialogs_sources, dialogs_targets, data_info, dialogs_speakers


class IeomapSentenceIterator():
    # Initialize the batch relevant information
    def __init__(self, sentences, targets, sentences_length, speakers):
        self.sentences = sentences
        self.targets = targets
        self.sentences_length = sentences_length
        self.speakers = speakers
        self.size = len(sentences)
        self.epoch = 0
        self.shuffle()

    # Shuffle the train/test sentences
    def shuffle(self):
        self.sentences, self.targets, self.sentences_length, self.speakers = shuffle(self.sentences, self.targets,
                                                                      self.sentences_length, self.speakers)
        self.pointer = 0

    # Sentence padding
    def pad(self, sentence, length):
        return np.pad(sentence,
                      (0, length - len(sentence)),
                      'constant',
                      constant_values=0)

    # Next batch creator
    def next_batch(self, n):
        if self.pointer + n >= self.size:
            self.epoch += 1
            sentences_batch = self.sentences[self.pointer:]
            targets_batch = self.targets[self.pointer:]
            sentences_length_batch = self.sentences_length[self.pointer:]
            speakers_batch = self.speakers[self.pointer:]
            self.shuffle()
        else:
            sentences_batch = self.sentences[self.pointer:self.pointer + n]
            targets_batch = self.targets[self.pointer:self.pointer + n]
            sentences_length_batch = self.sentences_length[self.pointer:self.pointer + n]
            speakers_batch = self.speakers[self.pointer:self.pointer + n]
            self.pointer += n

        # Pad the sentences batch to have the same length as the longest sentence
        length = max(sentences_length_batch)
        sentences_batch = [self.pad(sentence, length) for sentence in sentences_batch]

        return sentences_batch, sentences_length_batch, targets_batch, speakers_batch


class IeomapDialogIterator():
    # Initialize the batch relevant information
    def __init__(self, dialogs, targets, sentences_length, speakers):
        self.dialogs = dialogs
        self.targets = targets
        self.sentences_length = sentences_length
        self.speakers = speakers
        self.size = len(dialogs)
        self.epoch = 0
        flat = []
        targets_flat = [flat.extend(d) for d in self.targets]
        emotions_count = Counter(targets_flat)
        self.emotions_count = [emotions_count[x] for x in set(targets_flat)]
        self.shuffle()

    # Shuffle the train/test dialogs but keep the sentence's order
    def shuffle(self):
        self.dialogs, self.targets, self.sentences_length, self.speakers = shuffle(self.dialogs, self.targets,
                                                                    self.sentences_length, self.speakers)
        self.pointer = 0

    # Sentence padding
    def pad1D(self, sentence, length, constant_values=0):
            return np.pad(sentence,
                          (0, length - len(sentence)),
                          'constant',
                          constant_values=constant_values)

    # Dialog padding
    def pad2D(self, dialog, length):
        return np.pad(dialog,
                      ((0, length - len(dialog)), (0, 0)),
                      'constant',
                      constant_values=0)

    # Next batch creator
    def next_batch(self, n):
        if self.pointer + n >= self.size:
            self.epoch += 1
            dialogs_batch = self.dialogs[self.pointer:]
            targets_batch = self.targets[self.pointer:]
            sentences_length_batch = self.sentences_length[self.pointer:]
            speakers_batch = self.speakers[self.pointer:]
            self.shuffle()
        else:
            dialogs_batch = self.dialogs[self.pointer:self.pointer + n]
            targets_batch = self.targets[self.pointer:self.pointer + n]
            sentences_length_batch = self.sentences_length[self.pointer:self.pointer + n]
            speakers_batch = self.speakers[self.pointer:self.pointer + n]
            self.pointer += n

        # Pad the dataset to have the same length in all sentences and dialogs
        max_sentence_length = max([max(x) for x in sentences_length_batch])
        max_dialog_length = max([len(x) for x in sentences_length_batch])
        dialog_length_batch = [len(x) for x in sentences_length_batch]
        # The empty sentences are giving length of 1
        sentences_length_batch = [self.pad1D(x, max_dialog_length, 1) for x in sentences_length_batch]
        # Use concatenation to avoid 4D dataset
        sentences_length_batch = np.concatenate(np.array(sentences_length_batch))
        # Pad the targets to have the same length as the longest dialog
        targets_batch = [self.pad1D(x.reshape(-1), max_dialog_length) for x in targets_batch]
        # Pad the speakers to have the same length as the longest dialog
        speakers_batch = [self.pad2D(dialog, max_dialog_length) for dialog in speakers_batch]
        # Use concatenation to avoid 4D dataset, Not needed for the double RNN
        # targets_batch = np.concatenate(targets_batch)
        # Pad the sentences to have the same length as the longest sentence in all the dialogs batch
        dialogs_batch = [[self.pad1D(sentence, max_sentence_length) for sentence in dialog] for dialog in dialogs_batch]
        # Pad the dialogs to have the same length as the longest dialog
        dialogs_batch = [self.pad2D(dialog, max_dialog_length) for dialog in dialogs_batch]
        # Use concatenation to avoid 4D dataset
        dialogs_batch = np.concatenate(np.array(dialogs_batch))
        speakers_batch = np.concatenate(np.array(speakers_batch))

        return dialogs_batch, [np.array(dialog_length_batch), sentences_length_batch], targets_batch, speakers_batch
