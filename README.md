# seq2seq-attention

# Introduction

This code implements RNN/LSTM/GRU seq2seq and seq2seq+attention models for training and sampling in word-level. You can apply it in Bot, Auto Text Summarization, Machine Translation, Question Answer System etc. Here, we show you a bot demo.

# Requirements

- <code>senna</code> 

This interface supports Part-of-speech tagging, Chunking, Name Entity Recognition and Semantic Role Labeling. It is used in sampling.

You can find how to install `senna` [here](https://github.com/torch/senna)

- <code>hdf5</code>

It is a file format, the format is fast, flexible, and supported by a wide range of other software - including MATLAB, Python, and R.

You can find how to install `hdf5` [here](https://github.com/deepmind/torch-hdf5/blob/master/doc/usage.md)

- <code>cutorch/cunn</code>

If you want to run the code in GRU, you need to install cutorch and cunn.

`[sudo] luarocks install cutorch`

`[sudo] luarocks install cunn`


# Dataset

We use the [Wikipedia Talkpages Conversations Dataset]((http://pan.baidu.com/s/1kVHCxwj)) as our corpus, to implement a conversation bot with it. After the download finished, ensure the data file in directory `data/`.


# Run

<b>Step 1</b> run the data preprocessing code, to generate the dataset file and vocabulary file.

`python bot.py`

If you want to do research with any other datasets or tasks, you may need to implement your preprocessing python script, then write the result data into the hdf5 file.

<b>Step 2</b> run the training code.

`th train.lua`

<b>Step 3</b> run the sampling code.

`th test.lua`

You can change the parameters or choose the model you need in the `CmdLine()` part of the code.

<b>Step 4</b> run the bot server code.

`th server.lua`

<b>Step 5</b> test the bot response through bot server.

`curl http://localhost:8080/bot -d "input=<your input words>"`

# Acknowledge

Thanks for [oxford deep learning course code](https://github.com/oxford-cs-ml-2015/practical6) and Karpathy's [char-rnn code](https://github.com/karpathy/char-rnn).

