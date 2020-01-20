from Prediction import model2
from model.processing import Process
import tensorflow as tf
#model , architecture , pre_process , model_inputs = tweet_classification.get_model_instance (model_name , embed_model)


def get_model_instance(model_name, embedding_model,  keep_prob=1.0,
                       label_size=2, n_word=40, n_char=33 ):
    """
    Get model instance, preprocessing, and input format

    :param model_name:
    :param embedding_model:
    :param float learning_rate: learning rate
    :param float gradient_clip: (option) max norm for gradient clipping
    :param float keep_prob: (option) keep probability for dropout.
    :param float batch_norm: (option) decaying parameter for batch normalization
    :param int label_size: output label size
    :param int n_word: word size (word sequence is pad by this value)
    :param int n_char: character size (character sequence is pad by this value)
    :return:
    """
    # set up pre processing
    _pre_process = Process("embed", {"length_word": n_word, "dim": embedding_model.vector_size, "model": embedding_model , "path": "./data/random_dict.json"})
    if model_name == "cnn_char":
        _model = model2.CharCNN
        _pre_process = [Process("onehot", {"length_word": n_word, "length_char": n_char}), _pre_process]
        _net = {"input_char": [n_word, n_char, _pre_process[0].char_dict_size],
                "input_word": [n_word, embedding_model.vector_size],
                "label_size": label_size,
                "char_embed_dim": 5, "char_cnn_unit": 10, "char_cnn_kernel": 3,  # character embedding
                "word_embed_dim": 30, "cnn_unit": 300, "cnn_kernel": 5, "hidden_unit": 300}
        _model_inputs = model2.InputFormat.char_word

    elif model_name == "lstm_char":
        _model = model2.CharLSTM
        _pre_process = [Process("onehot", {"length_word": n_word, "length_char": n_char}), _pre_process]
        _net = {"input_char": [n_word, n_char, _pre_process[0].char_dict_size],
                "input_word": [n_word, embedding_model.vector_size],
                "label_size": label_size,
                "char_embed_dim": 5, "char_cnn_unit": 10, "char_cnn_kernel": 3,  # character embedding
                "n_hidden_1": 64, "n_hidden_2": 128, "n_hidden_3": 256}
        _model_inputs = model2.InputFormat.char_word
    elif model_name == "cnn_gap":
        _net = {"input_word": [n_word, embedding_model.vector_size, 1], "label_size": label_size}
        _model = model2.GapCNN
        _model_inputs = model2.InputFormat.word_3d
    elif model_name == "lstm":
        _net = {"input_word": [n_word, embedding_model.vector_size], "label_size": label_size,
                "n_hidden_1": 64, "n_hidden_2": 128, "n_hidden_3": 256}
        _model = model2.LSTM
        _model_inputs = model2.InputFormat.basic


    else:
        raise ValueError("unknown model!")
    #model_instance = _model(network_architecture=_net , keep_prob=keep_prob)
    return  _net, _pre_process,  _model_inputs
