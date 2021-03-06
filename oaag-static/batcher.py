# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains code to process data into batches"""

import queue as Queue
from random import shuffle
from threading import Thread
import time
import numpy as np
import tensorflow as tf
import data


class Example(object):
  """Class representing a train/val/test example for text summarization."""

  def __init__(self, reviews, ratings, answer_sentences, question, label, vocab, hps):
    """Initializes the Example, performing tokenization and truncation to produce the encoder, decoder and target sequences, which are stored in self.

    Args:
      reviews: review text; a list.
      ratings: alist.
      answer_sentences: list of strings, one per abstract sentence. In each sentence, each token is separated by a single space.
      vocab: Vocabulary object
      hps: hyperparameters
    """
    self.hps = hps

    # Get ids of special tokens
    start_decoding = vocab.word2id(data.START_DECODING)
    stop_decoding = vocab.word2id(data.STOP_DECODING)



    # Process the reviews
    self.r_lens = []
    self.r_batch = []
    self.rating_batch = ratings
    reviews_words = []
    for review in reviews:
      review_words = review.split()
      if len(review_words) > hps.max_enc_steps:
        review_words = review_words[:hps.max_enc_steps]
      reviews_words.append(review_words)
      self.r_lens.append(len(review_words)) # store the length after truncation but before padding
      self.r_batch.append([vocab.word2id(w) for w in review_words]) # list of word ids; OOVs are represented by the id for UNK token

    # Process the abstract
    answer = ' '.join(answer_sentences) # string
    answer_words = answer.split() # list of strings
    ans_ids = [vocab.word2id(w) for w in answer_words] # list of word ids; OOVs are represented by the id for UNK token

    # Process the question
    question_words = question.split()
    self.q_lens = len(question_words)
    self.q_batch = [vocab.word2id(w) for w in question_words]

    # Process the label
    self.y_target = label

    # Get the decoder input sequence and target sequence
    self.dec_input, self.target = self.get_dec_inp_targ_seqs(ans_ids, hps.max_dec_steps, start_decoding, stop_decoding)
    self.dec_len = len(self.dec_input)
    assert self.dec_len > 0

    # If using pointer-generator mode, we need to store some extra info
    if hps.pointer_gen:
      self.oovs = []
      # Store a version of the enc_input where in-article OOVs are represented by their temporary OOV id; also store the in-article OOVs words themselves
      self.reviews_extend_vocab = []
      for review_words in reviews_words:
        review_extend_vocab, self.oovs = data.article2ids(review_words, vocab, self.oovs)
        self.reviews_extend_vocab.append(review_extend_vocab)

      # question OOV id.
      self.question_extend_vocab, self.oovs = data.article2ids(question_words, vocab, self.oovs)

      # Get a verison of the reference summary where in-article OOVs are represented by their temporary article OOV id
      ans_ids_extend_vocab = data.abstract2ids(answer_words, vocab, self.oovs)

      # Overwrite decoder target sequence so it uses the temp article OOV ids
      _, self.target = self.get_dec_inp_targ_seqs(ans_ids_extend_vocab, hps.max_dec_steps, start_decoding, stop_decoding)

    # Store the original strings
    self.original_reviews = reviews
    self.original_answer = answer
    self.original_answer_sents = answer_sentences
    self.original_question = question


  def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
    """Given the reference summary as a sequence of tokens, return the input sequence for the decoder, and the target sequence which we will use to calculate loss. The sequence will be truncated if it is longer than max_len. The input sequence must start with the start_id and the target sequence must end with the stop_id (but not if it's been truncated).

    Args:
      sequence: List of ids (integers)
      max_len: integer
      start_id: integer
      stop_id: integer

    Returns:
      inp: sequence length <=max_len starting with start_id
      target: sequence same length as input, ending with stop_id only if there was no truncation
    """
    inp = [start_id] + sequence[:]
    target = sequence[:]
    if len(inp) > max_len: # truncate
      inp = inp[:max_len]
      target = target[:max_len] # no end_token
    else: # no truncation
      target.append(stop_id) # end token
    assert len(inp) == len(target)
    return inp, target


  def pad_decoder_inp_targ(self, max_len, pad_id):
    """Pad decoder input and target sequences with pad_id up to max_len."""
    while len(self.dec_input) < max_len:
      self.dec_input.append(pad_id)
    while len(self.target) < max_len:
      self.target.append(pad_id)


  def pad_reviews(self, max_len, pad_id):
    """Pad the encoder input sequence with pad_id up to max_len."""
    for i in range(len(self.r_batch)):
      while len(self.r_batch[i]) < max_len:
        self.r_batch[i].append(pad_id)
    if self.hps.pointer_gen:
      for i in range(len(self.reviews_extend_vocab)):
        while len(self.reviews_extend_vocab[i]) < max_len:
          self.reviews_extend_vocab[i].append(pad_id)


  def pad_question(self, max_len, pad_id):
    """Pad the encoder input sequence with pad_id up to max_len."""
    while len(self.q_batch) < max_len:
      self.q_batch.append(pad_id)
    if self.hps.pointer_gen:
      while len(self.question_extend_vocab) < max_len:
        self.question_extend_vocab.append(pad_id)


class Batch(object):
  """Class representing a minibatch of train/val/test examples for text summarization."""

  def __init__(self, example_list, hps, vocab):
    """Turns the example_list into a Batch object.

    # Pad the encoder input sequences up to the length of the longest sequence
    for ex in example_list:
      ex.pad_reviews(max_r_seq_len, self.pad_id)
      ex.pad_question(max_q_seq_len, self.pad_id)


    Args:
       example_list: List of Example objects
       hps: hyperparameters
       vocab: Vocabulary object
    """
    self.pad_id = vocab.word2id(data.PAD_TOKEN) # id of the PAD token used to pad sequences
    self.init_encoder_seq(example_list, hps) # initialize the input to the encoder
    self.init_decoder_seq(example_list, hps) # initialize the input and targets for the decoder
    self.store_orig_strings(example_list) # store the original strings

  def init_encoder_seq(self, example_list, hps):
    """Initializes the following:
        self.enc_batch:
          numpy array of shape (batch_size, <=max_enc_steps) containing integer ids (all OOVs represented by UNK id), padded to length of longest sequence in the batch
        self.enc_lens:
          numpy array of shape (batch_size) containing integers. The (truncated) length of each encoder input sequence (pre-padding).
        self.enc_padding_mask:
          numpy array of shape (batch_size, <=max_enc_steps), containing 1s and 0s. 1s correspond to real tokens in enc_batch and target_batch; 0s correspond to padding.

      If hps.pointer_gen, additionally initializes the following:
        self.max_art_oovs:
          maximum number of in-article OOVs in the batch
        self.art_oovs:
          list of list of in-article OOVs (strings), for each example in the batch
        self.enc_batch_extend_vocab:
          Same as self.enc_batch, but in-article OOVs are represented by their temporary article OOV number.
    """
    # Determine the maximum length of the encoder input sequence in this batch
    max_r_seq_len = max([max(ex.r_lens) for ex in example_list])
    max_q_seq_len = max([ex.q_lens for ex in example_list])

    # Pad the encoder input sequences up to the length of the longest sequence
    for ex in example_list:
      ex.pad_reviews(max_r_seq_len, self.pad_id)
      ex.pad_question(max_q_seq_len, self.pad_id)

    # Initialize the numpy arrays
    # Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.
    self.r_batch = np.zeros((hps.batch_size, hps.review_num, max_r_seq_len), dtype=np.int32)
    self.q_batch = np.zeros((hps.batch_size, max_q_seq_len), dtype=np.int32)
    self.r_lens = np.zeros((hps.batch_size, hps.review_num), dtype=np.int32)
    self.q_lens = np.zeros((hps.batch_size), dtype=np.int32)
    self.r_padding_mask = np.zeros((hps.batch_size, hps.review_num, max_r_seq_len), dtype=np.float32)
    self.q_padding_mask = np.zeros((hps.batch_size, max_q_seq_len), dtype=np.float32)
    self.rating_batch = np.zeros((hps.batch_size, hps.review_num), dtype=np.int32)

    # Fill in the numpy arrays
    for i, ex in enumerate(example_list):
      self.r_batch[i, :] = ex.r_batch[:]
      self.r_lens[i, :] = ex.r_lens[:]
      self.q_batch[i, :] = ex.q_batch[:]
      self.q_lens[i] = ex.q_lens
      self.rating_batch[i, :] = ex.rating_batch[:]
      for j in range(len(ex.r_lens)):
        for k in range(ex.r_lens[j]):
          self.r_padding_mask[i][j][k] = 1
      for j in range(ex.q_lens):
        self.q_padding_mask[i][j] = 1

    # For pointer-generator mode, need to store some extra info
    if hps.pointer_gen:
      # Determine the max number of in-article OOVs in this batch
      self.max_oovs = max([len(ex.oovs) for ex in example_list])
      # Store the in-article OOVs themselves
      self.oovs = [ex.oovs for ex in example_list]
      # Store the version of the enc_batch that uses the article OOV ids
      self.r_batch_extend_vocab = np.zeros((hps.batch_size, hps.review_num, max_r_seq_len), dtype=np.int32)
      self.q_batch_extend_vocab = np.zeros((hps.batch_size, max_q_seq_len), dtype=np.int32)
      for i, ex in enumerate(example_list):
        self.r_batch_extend_vocab[i, :] = ex.reviews_extend_vocab[:]
      for i, ex in enumerate(example_list):
        self.q_batch_extend_vocab[i, :] = ex.question_extend_vocab[:]

  def init_decoder_seq(self, example_list, hps):
    """Initializes the following:
        self.dec_batch:
          numpy array of shape (batch_size, max_dec_steps), containing integer ids as input for the decoder, padded to max_dec_steps length.
        self.target_batch:
          numpy array of shape (batch_size, max_dec_steps), containing integer ids for the target sequence, padded to max_dec_steps length.
        self.dec_padding_mask:
          numpy array of shape (batch_size, max_dec_steps), containing 1s and 0s. 1s correspond to real tokens in dec_batch and target_batch; 0s correspond to padding.
        """
    # Pad the inputs and targets
    for ex in example_list:
      ex.pad_decoder_inp_targ(hps.max_dec_steps, self.pad_id)

    # Initialize the numpy arrays.
    # Note: our decoder inputs and targets must be the same length for each batch (second dimension = max_dec_steps) because we do not use a dynamic_rnn for decoding. However I believe this is possible, or will soon be possible, with Tensorflow 1.0, in which case it may be best to upgrade to that.
    self.dec_batch = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.int32)
    self.target_batch = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.int32)
    self.y_target_batch = np.zeros((hps.batch_size), dtype=np.int32)
    self.dec_padding_mask = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.float32)

    # Fill in the numpy arrays
    for i, ex in enumerate(example_list):
      self.dec_batch[i, :] = ex.dec_input[:]
      self.target_batch[i, :] = ex.target[:]
      self.y_target_batch[i] = ex.y_target
      for j in range(ex.dec_len):
        self.dec_padding_mask[i][j] = 1

  def store_orig_strings(self, example_list):
    """Store the original article and abstract strings in the Batch object"""
    self.original_reviews = [ex.original_reviews for ex in example_list] # list of lists
    self.original_answers = [ex.original_answer for ex in example_list] # list of lists
    self.original_questions = [ex.original_question for ex in example_list] # list of lists
    self.original_answers_sents = [ex.original_answer_sents for ex in example_list] # list of list of lists


class Batcher(object):
  """A class to generate minibatches of data. Buckets examples together based on length of the encoder sequence."""

  BATCH_QUEUE_MAX = 100 # max number of batches the batch_queue can hold

  def __init__(self, data_path, vocab, hps, single_pass):
    """Initialize the batcher. Start threads that process the data into batches.

    Args:
      data_path: tf.Example filepattern.
      vocab: Vocabulary object
      hps: hyperparameters
      single_pass: If True, run through the dataset exactly once (useful for when you want to run evaluation on the dev or test set). Otherwise generate random batches indefinitely (useful for training).
    """
    self._data_path = data_path
    self._vocab = vocab
    self._hps = hps
    self._single_pass = single_pass

    # Initialize a queue of Batches waiting to be used, and a queue of Examples waiting to be batched
    self._batch_queue = Queue.Queue(self.BATCH_QUEUE_MAX)
    self._example_queue = Queue.Queue(self.BATCH_QUEUE_MAX * self._hps.batch_size)

    # Different settings depending on whether we're in single_pass mode or not
    if single_pass:
      self._num_example_q_threads = 1 # just one thread, so we read through the dataset just once
      self._num_batch_q_threads = 1  # just one thread to batch examples
      self._bucketing_cache_size = 1 # only load one batch's worth of examples before bucketing; this essentially means no bucketing
      self._finished_reading = False # this will tell us when we're finished reading the dataset
    else:
      self._num_example_q_threads = 16 # num threads to fill example queue
      self._num_batch_q_threads = 4  # num threads to fill batch queue
      self._bucketing_cache_size = 100 # how many batches-worth of examples to load into cache before bucketing

    # Start the threads that load the queues
    self._example_q_threads = []
    for _ in range(self._num_example_q_threads):
      self._example_q_threads.append(Thread(target=self.fill_example_queue))
      self._example_q_threads[-1].daemon = True
      self._example_q_threads[-1].start()
    self._batch_q_threads = []
    for _ in range(self._num_batch_q_threads):
      self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
      self._batch_q_threads[-1].daemon = True
      self._batch_q_threads[-1].start()

    # Start a thread that watches the other threads and restarts them if they're dead
    if not single_pass: # We don't want a watcher in single_pass mode because the threads shouldn't run forever
      self._watch_thread = Thread(target=self.watch_threads)
      self._watch_thread.daemon = True
      self._watch_thread.start()


  def next_batch(self):
    """Return a Batch from the batch queue.

    If mode='decode' then each batch contains a single example repeated beam_size-many times; this is necessary for beam search.

    Returns:
      batch: a Batch object, or None if we're in single_pass mode and we've exhausted the dataset.
    """
    # If the batch queue is empty, print a warning
    if self._batch_queue.qsize() == 0:
      tf.compat.v1.logging.warning('Bucket input queue is empty when calling next_batch. Bucket queue size: %i, Input queue size: %i', self._batch_queue.qsize(), self._example_queue.qsize())
      print(self._single_pass,self._finished_reading)
      if self._single_pass and self._finished_reading:
        tf.logging.info("Finished reading dataset in single_pass mode.")
        return None

    batch = self._batch_queue.get() # get the next Batch
    return batch

  def fill_example_queue(self):
    """Reads data from file and processes into Examples which are then placed into the example queue."""

    input_gen = self.text_generator(data.example_generator(self._data_path, self._single_pass))

    while True:
      try:
        (reviews, ratings, answer, question, label) = next(input_gen) # read the next example from file. article and abstract are both strings.
      #except StopIteration: # if there are no more examples:
      except Exception as e:
        tf.logging.info("The example generator for this example queue filling thread has exhausted data.")
        if self._single_pass:
          tf.logging.info("single_pass mode is on, so we've finished reading dataset. This thread is stopping.")
          self._finished_reading = True
          break
        else:
          raise Exception("single_pass mode is off but the example generator is out of data; error.")

      answer_sentences = [sent.strip() for sent in data.abstract2sents(answer)] # Use the <s> and </s> tags in abstract to get a list of sentences.
      example = Example(reviews, ratings, answer_sentences, question, label, self._vocab, self._hps) # Process into an Example.
      self._example_queue.put(example) # place the Example in the example queue.


  def fill_batch_queue(self):
    """Takes Examples out of example queue, sorts them by encoder sequence length, processes into Batches and places them in the batch queue.

    In decode mode, makes batches that each contain a single example repeated.
    """
    while True:
      if self._hps.mode != 'decode':
        # Get bucketing_cache_size-many batches of Examples into a list, then sort
        inputs = []
        for _ in range(self._hps.batch_size * self._bucketing_cache_size):
          inputs.append(self._example_queue.get())
        inputs = sorted(inputs, key=lambda inp: inp.q_lens) # sort by length of encoder sequence

        # Group the sorted Examples into batches, optionally shuffle the batches, and place in the batch queue.
        batches = []
        for i in range(0, len(inputs), self._hps.batch_size):
          batches.append(inputs[i:i + self._hps.batch_size])
        if not self._single_pass:
          shuffle(batches)
        for b in batches:  # each b is a list of Example objects
          self._batch_queue.put(Batch(b, self._hps, self._vocab))

      else: # beam search decode mode
        ex = self._example_queue.get()
        b = [ex for _ in range(self._hps.batch_size)]
        self._batch_queue.put(Batch(b, self._hps, self._vocab))


  def watch_threads(self):
    """Watch example queue and batch queue threads and restart if dead."""
    while True:
      time.sleep(60)
      for idx,t in enumerate(self._example_q_threads):
        if not t.is_alive(): # if the thread is dead
          tf.logging.error('Found example queue thread dead. Restarting.')
          new_t = Thread(target=self.fill_example_queue)
          self._example_q_threads[idx] = new_t
          new_t.daemon = True
          new_t.start()
      for idx,t in enumerate(self._batch_q_threads):
        if not t.is_alive(): # if the thread is dead
          tf.logging.error('Found batch queue thread dead. Restarting.')
          new_t = Thread(target=self.fill_batch_queue)
          self._batch_q_threads[idx] = new_t
          new_t.daemon = True
          new_t.start()


  def text_generator(self, example_generator):
    """Generates article and abstract text from tf.Example.

    Args:
      example_generator: a generator of tf.Examples from file. See data.example_generator"""
    while True:
      e = next(example_generator) # e is a tf.Example
      try:
        reviews = eval(e.features.feature['reviews'].bytes_list.value[0].decode()) # the article text was saved under the key 'article' in the data files
        ratings = eval(e.features.feature['ratings'].bytes_list.value[0].decode())
        answer_text = e.features.feature['answer'].bytes_list.value[0].decode() # the abstract text was saved under the key 'abstract' in the data files
        question_text = e.features.feature['question'].bytes_list.value[0].decode() # the question text was saved under the key 'question' in the data files
        answer_label = e.features.feature['label'].bytes_list.value[0].decode() # the answer label was saved under the key 'label' in the data files
      except ValueError:
        tf.logging.error('Failed to get article or abstract from example')
        continue
      if len(question_text)==0:
        tf.logging.warning('Found an example with empty article text. Skipping it.')
      else:
        yield (reviews, ratings, answer_text, question_text, answer_label)
