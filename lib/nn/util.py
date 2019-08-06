import torch
import torch.nn.functional as F


def masked_divide(tensor_a, tensor_b, eps=1e-10):
    return tensor_b.ne(0).float()*(tensor_a / (tensor_b+1e-10))


def sentences2paragraph_tensor(sentencewise_tensor, sentences_mask):
    """
    # Input:
    # sentencewise_tensor: (batch_size, num_sentences, sentence_max_seq_len, ...)
    # sentences_mask: (batch_size, num_sentences, sentence_max_seq_len)

    # Output:
    # paragraphwise_tensor: (batch_size, paragraph_max_seq_len, ...)
    """
    num_sentences = sentencewise_tensor.shape[1]
    trailing_shape = list(sentencewise_tensor.shape[3:])

    sentences_mask = sentences_mask.byte()
    # keep unsqueezing instance_sentences_mask at -1 to make it same shape as sentencewise_tensor and then .byte()
    while len(sentences_mask.shape) < len(sentencewise_tensor.shape):
        sentences_mask = sentences_mask.unsqueeze(-1)

    paragraphwise_tensor = []
    for instance_sentencewise_tensor, instance_sentences_mask in zip(torch.unbind(sentencewise_tensor, dim=0),
                                                                     torch.unbind(sentences_mask, dim=0)):
        instance_paragraphwise_tensor = instance_sentencewise_tensor.masked_select(instance_sentences_mask)
        instance_paragraphwise_tensor = instance_paragraphwise_tensor.reshape([-1]+trailing_shape)
        paragraphwise_tensor.append(instance_paragraphwise_tensor)

    paragraphwise_tensor = torch.nn.utils.rnn.pad_sequence(paragraphwise_tensor, batch_first=True)
    return paragraphwise_tensor


def paragraph2sentences_tensor(paragraphwise_tensor, sentence_lengths):
    """
    # # Input:
    # paragraphwise_tensor: (batch_size, paragraph_max_seq_len, ...)
    # sentence_lengths: (batch_size, premises_count)

    # Output:
    # sentencewise_tensor: (batch_size, num_sentences, sentence_max_seq_len, ...)

    # rough eg. for one instance of batch:
    # paragraphwise_tensor = torch.tensor(45, 10)
    # sentence_lengths = torch.tensor([10, 10, 10, 15])
    # cumulated_sentence_lengths = (10, 20, 30, 45)
    # shifted_cumulated_sentence_lengths = (0, 10, 20, 30)
    # range_indices = zip(shifted_cumulated_sentence_lengths, cumulated_sentence_lengths)
    # sentencewise_tensor = ([paragraphwise_tensor[start:end] for start, end in range_indices])
    # sentencewise_tensor = torch.nn.utils.rnn.pad_sequence(sentencewise_tensor, batch_first=True)
    # return sentencewise_tensor
    """
    sentencewise_tensors = []
    # max_sentence_length across all paragraphs (ie. any batch instance)
    max_sentence_length = sentence_lengths.max()
    for instance_paragraphwise_tensor, instance_sentence_lengths in zip(torch.unbind(paragraphwise_tensor, dim=0),
                                                                        torch.unbind(sentence_lengths, dim=0)):

        instance_cumulated_sentence_lengths = instance_sentence_lengths.cumsum(dim=0)
        instance_shifted_cumulated_sentence_lengths = instance_cumulated_sentence_lengths - instance_sentence_lengths
        range_indices = zip(instance_shifted_cumulated_sentence_lengths, instance_cumulated_sentence_lengths)

        sentencewise_tensor = [instance_paragraphwise_tensor[start.int():end.int()] for start, end in range_indices]
        sentencewise_tensor = torch.nn.utils.rnn.pad_sequence(sentencewise_tensor, batch_first=True)

        # sentencewise_tensor: (num_sentences, sentence_max_seq_len, ...)
        # adjust first dim by max sentence length across all the batch instances.
        padding = max_sentence_length - sentencewise_tensor.shape[1]
        padding_tuple = ([0]*(len(sentencewise_tensor.shape)-2)*2) + [0, padding.int(), 0, 0]
        sentencewise_tensor = F.pad(sentencewise_tensor, pad=padding_tuple)
        sentencewise_tensors.append(sentencewise_tensor)

    sentencewise_tensors = torch.nn.utils.rnn.pad_sequence(sentencewise_tensors, batch_first=True)
    return sentencewise_tensors


def sentencewise_scores2paragraph_tokenwise_scores(sentences_scores, sentences_mask):
    """
    # Input:
    # sentences_mask: (batch_size X num_sentences X sent_seq_len)
    # sentences_scores: (batch_size X num_sentences)

    # Output:
    # paragraph_tokenwise_scores: (batch_size X max_para_seq_len)
    """
    paragraph_tokenwise_scores = []
    for instance_sentences_scores, instance_sentences_mask in zip(torch.unbind(sentences_scores, dim=0),
                                                                  torch.unbind(sentences_mask, dim=0)):
        instance_paragraph_tokenwise_scores = torch.masked_select(instance_sentences_scores.unsqueeze(-1),
                                                                    instance_sentences_mask.byte())
        paragraph_tokenwise_scores.append(instance_paragraph_tokenwise_scores)
    paragraph_tokenwise_scores = torch.nn.utils.rnn.pad_sequence(paragraph_tokenwise_scores, batch_first=True)
    return paragraph_tokenwise_scores


def unbind_tensor_dict(dict_tensors, dim):
    """
    Unbinds each tensor dict as returned by text_field.as_tensor in forward method
    on a certain dimension and returns a list of such tensor dicts
    """
    intermediate_dict = {}
    for key, tensor in dict_tensors.items():
        intermediate_dict[key] = torch.unbind(tensor, dim=dim)
        items_count = len(intermediate_dict[key])
    dict_tensor_list = [{} for _ in range(items_count)]
    for key, tensors in intermediate_dict.items():
        for index, tensor in enumerate(tensors):
            dict_tensor_list[index][key] = tensor
    return dict_tensor_list
