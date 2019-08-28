import tensorflow as tf
import random as ran
import numpy as np
import pickle
import copy
from collections import Counter
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_data(path):
    input_file = os.path.join(path)
    print(os.path.join(path))
    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.read()

    return data
source_path = 'neural-network\data\small_vocab_en'
target_path = 'neural-network\data\small_vocab_fr'
source_text = load_data(source_path)
target_text = load_data(target_path)

batch_size = 128

# input english sentences
english_sentences = source_text.split('\n')

# target output french sentences
french_sentences = target_text.split('\n')

sample_sentence_range = (0, 5)
side_by_side_sentences = list(zip(english_sentences, french_sentences))[sample_sentence_range[0]:sample_sentence_range[1]]

for index, sentence in enumerate(side_by_side_sentences):
    en_sent, fr_sent = sentence
    print('[{}-th] sentence'.format(index+1))
    print('\tEN: {}'.format(en_sent))
    print('\tFR: {}'.format(fr_sent))
    print()

CODES = {'<PAD>': 0, '<EOS>': 1, '<UNK>': 2, '<GO>': 3 }

def load_preprocess():
    with open('neural-network/preprocess.p', mode='rb') as in_file:
        return pickle.load(in_file)

def load_params():
    with open('neural-network/params.p', mode='rb') as in_file:
        return pickle.load(in_file)

def sentence_to_seq(sentence, vocab_to_int):
    """breaking up a sentence into an array of words

    Returns:
        results: an array of words with values not in vocab_to_int replaced with <UNK>
    """
    results = []
    for word in sentence.split(" "):
        if word in vocab_to_int:
            results.append(vocab_to_int[word])
        else:
            results.append(vocab_to_int['<UNK>'])
            
    return results

_, (source_vocab_to_int, target_vocab_to_int), (source_int_to_vocab, target_int_to_vocab) = load_preprocess()
load_path = load_params()

# displays 10 comparisons (input english to output french)
for x in range (10):
    translate_sentence = english_sentences[ran.randint(0, 1000)]

    translate_sentence = sentence_to_seq(translate_sentence, source_vocab_to_int)

    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # Load saved model
        loader = tf.train.import_meta_graph(load_path + '.meta')
        loader.restore(sess, load_path)

        input_data = loaded_graph.get_tensor_by_name('input:0')
        logits = loaded_graph.get_tensor_by_name('predictions:0')
        target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
        keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

        translate_logits = sess.run(logits, {input_data: [translate_sentence]*batch_size, target_sequence_length: [len(translate_sentence)*2]*batch_size, keep_prob: 1.0})[0]

    print('Input')
    print('  Word Ids:      {}'.format([i for i in translate_sentence]))
    print('  English Words: {}'.format([source_int_to_vocab[i] for i in translate_sentence]))

    print('\nPrediction')
    print('  Word Ids:      {}'.format([i for i in translate_logits]))
    print('  French Words: {}'.format(" ".join([target_int_to_vocab[i] for i in translate_logits])))