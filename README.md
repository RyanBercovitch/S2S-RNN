# Introduction

The neural network that we use is a Recurrent Neural Network (RNN) that takes in an input sequence, in this case a sentence (or sequence of words) in English, and outputs its recommendation (what it believes to be the sentence in french). It is made up of LSTM (Long Short Term Memory) cells which each process one element in the output sequence and then perdicts the next element.

## Requirements for Neural Network

since `pip install` does not save to the `package.json`, you need to use `pip install` for these packages:

- `tensorflow`
- `numpy`
- `pickle`
- `requests`
- `dotenv`

### Running the neural network

Before running `request-test.py`, make sure that you start `cf_backend` using `npm start` in its terminal.

If you have not built the neural network, run `python -u "(YOUR PATH TO WHERE THE CF FILES ARE)\cf_backend\neural-network\rnn-build.py"` to build it.
After you have built it, run `python -u "(YOUR PATH TO WHERE THE CF FILES ARE)\cf_backend\neural-network\rnn-load.py"` to load it.

### `rnn-build.py` process

1. Imports what it needs inclideing tensorflow which is vital for a lot of the advanced calculations involved within neural networks and pickle which is useful in saving and loading the neural network.
2. `load_data`: Loads the data to be used as inputs and outputs for training the neural network.
3. Defines these global variables:
    - source_text: holds the data that will be used as inputs.
    - target_text: holds the data that will be used as target outputs.
    - display_step: shows in the console the epoch, batch, accuracy, loss... after performing this many steps.
    - epochs: the number of total itterations that the data will go though to train the neural network.
    - batch_size: the number of training data that will be compared at a time before altering the neural network.
    - rnn_size: the maximum number of LSTM cells in a sequence.
    - num_layers: the total number of layers (including input, hidden, and output) that the neural network will work with.
    - encoding_embedding_size: size of the value associated with converting each word into something that is readable by the computer for computations (a vector)
    - decoding_embedding_size: size of the value that needs to be converted back into a word after the computer is done with its computations
    - learning_rate: the extent to what newly computed information overrides previous information to determine new relationships between inputs and outputs (weights).
    - keep_probability: the probability that a cell in the hidden layer(s) will be used in determining training results.
    - english_sentences: the data that will be used as inputs put into a readable form.
    - french_sentences: the data that will be used as outputs put into a readable form.
    - CODES: holds an enum object with strings that relate to different numbers.
        - `<PAD>`: Padding
        - `<EOS>`: End Of Sentence
        - `<UNK>`: Unknown
        - `<GO>`: Go

4. Defines functions (I will get to each when we need them).
5. `preprocess_and_save_data`: processes the data from the source and target files and saves them to another file.
    1. reloads the data in local variables.
    2. sets the data to consist of only lower case words.
    3. `create_lookup_tables`: creates the lookup tables for vocabulary to ints and ints to vocabulary using the text passed to it.
        1. creates an array of unique words called `vocab`.
        2. starts with the codes from CODES, storing their respective enum elements to an array that will hold every word's enum element.
        3. converts each unique value in `vocab` to an enum object's element and stores it to the array that already has the CODES.
        4. makes an array which stores the the string values of each previous enum in the proper order so you can plug in a int as an index and get it's respective string.
        5. returns both of these arrays.

    4. `text_to_ids`: redefines source and target text as int arrays.
        1. defines source and target text ids arrays to be used to hold the converted sequences of ids.
        2. defines the sentences to be used in the conversions to int ids.
        3. finds number of words in the sentence with the most words for both the source and target
        4. loops through each of the sentences (the number of source and target sentences should be equal) at each loop it:
            1. defines the current sentence for both source and target.
            2. creates an array of each word in those sentence.
            3. gets each id of those words and appends these ids to new arrays.
            4. appends the id of the `<EOS>` code to these new arrays.
            5. appends these new arrays to the text ids arrays.

        5. returns the text ids arrays.

    5. uses pickel to store these values in a file (will be created if not found).

6. defines the current checkpoint's path to save to.
7. `load_prprocess`: loads the variables that were saved during `preprocess_and_save_data` using pickle.
8. defines the tensorflow graph to use its funcitons.
9. `enc_dev_model_inputs`: defines placeholders for inputs, outputs, and the lengths/weights connecting them and also finds the max of these lengths.
10. `hyperparam_inputs`: defines placeholders for learning rate and keep probability.
11. `seq2seq_model`:
    1. `encoding_layer`: creates the layers of the RNN using weighted connections between these layers.
        1. maps a sequence of inputs to a sequence of embeddings of 'encoding_embedding_size' size
        2. uses LSTM cells (with a dropout rate of `keep_prob`) to create a RNN cell composed with a sequence of some simple cells.
        3. creates the RNN using the created RNN cells.
        4. returns the outputs and state of the created RNN.

    2. `process_decoder_input`: processes the input data for the training phase.
        1. finds the id of the `<GO>` code.
        2. extracts one part of the target data from index 0 to the batch's size
        3. returns the `<GO>` id (as a cell) and the extracted part of the target data as an array after adding 1 to each.

    3. `decoding_layer`: creates the decoding layer.
        1. gets the size of the target's vocab to int array.
        2. defines a tesnorflow variable (cell) that stores an array of random values and is the size of the `decoding_embedding_size` variable.
        3. looks up the ids in the list of embedded cells (what was returned by the previous function).
        4. uses LSTM cells (without a dropout rate this time) to create a RNN cell composed with a sequence of some simple cells.
        5. creates a "densely-connected" layer that is the same size of the target's vocab to int array.
        6. `decoding_layer_train`: creates a training process for the decoding layer.
            1. drops some of the decoding cells.
            2. defines something to help the computer read the inputs.
            3. performs basic decoding on the remaining decoding cells.
            4. performs dynamic decoding on the decoded cells.
            5. returns the outputs of the dynamic decoding.

        7. `decoding_layer_infer`: creates an inference process in the decoding layer
            1. drops some of the decoding cells.
            2. defines a helper that uses the argmax of the output and passes the result through an embedding layer to get the next input.
            3. does the same last three steps of the previous function.

        8. returns the results of `decoding_layer_train` and `decoding_layer_infer`.

    4. returns the results `decoding_layer`.

12. defines the logits (functions which map probabilities) for training and inferencing based on what was returned by `seq2seq_model`.
13. defines a mask function (ignores the padding on some values).
14. defines a function that will find the cost/loss of training logits.
15. defines a function that uses the "Adam" algorithm (more at `https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer`) to help find the gradient later on.
16. defines a function that will compute the gradient decent using the "Adam" algorithm and cost.
17. splits the data into training (from `batch_size` and on) and validation sets (from 0 to `batch_size`).
18. initializes the placeholder variables.
19. loops through the training data `epochs` number of times and at each loop it:
    1. `get_batches`:(defines the batch contents and sizes using source and target data).
        1. loops for each batch that needs to be made:
            1. defines the start of the data that will be put into the current batch
            2. defines the source and target batchs that go from the previously defined start to the batch's size (passed to this function)
            3. padds these batches so they are the same length as the others
            4. creates two arrays and stores in them the lengths of each element in the source and target padded batches.
            5. yeilds the padded batches and the arrays of their lengths. (for what    `yield` is visit `https://pythontips.com/2013/09/29/the-python-yield-keyword-explained/`).

    2. loops though the training data for each batch in `get_batches`.
        1. gets the current loss using the previously defined function for calculating cost/loss by running the session with it and giving it all the current inputs, targets, learning rate, and the keep probability.
        2. `get_accuracy`: gets the accuracy comparing the outputs and targets.
            1. pads the smaller of either target or loggits (both are passed to the function) with the same values at the start of the other until they are the same length.
            2. returns the average of the elements that are not equal.

        3. after `display_step` number of batches, gets batch train and vailid logits using the inference logits to use in `get_accuracy` and then prints.

20. saves the model in the current session to `save_path`.
21. `save_params`: saves to a file anything that was passes as a parameter using pickel.
22. `load_preprocess`: loads the preprocesses from a file using pickel.
23. `load_params`: loads the params from a file using pickel.
24. defines a english sentence for testing purposes.
25. `sentence_to_seq`: creating a sequence of words from a sentence.
26. again defines the tensorflow graph.
27. recreates a graph saved in meta data and reloads the previous session.
28. redefines the input data, logits, target sequence lengths array, and keep probability from the loaded session.
29. tests the results by printing the input and output sentences in the console.

### `rnn-load.py` process

This code is made up almost entirely of functions that were also found in `rnn-build.py` which main functions were to load and test data. The only unique functionality is testing multiple output sentences in the console using sentences randomly taken from the input data.
