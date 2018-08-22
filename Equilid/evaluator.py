from __future__ import print_function, division

import torch
import torchtext

from Seq2seq import seq2seq
from loss import NLLLoss

class Evaluator(object):
    """ Class to evaluate models with given datasets.

    Args:
        loss (seq2seq.loss, optional): loss for evaluator (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for evaluator (default: 64)
    """

    def __init__(self, loss=NLLLoss(), batch_size=64):
        self.loss = loss
        self.batch_size = batch_size

    def evaluate(self, model, data):
        """ Evaluate a model on given dataset and return performance.

        Args:
            model (seq2seq.models): model to evaluate
            data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against

        Returns:
            loss (float): loss of the given model on the given dataset
        """
        model.eval()

        loss = self.loss
        loss.reset()
        match = 0
        total = 0

        device = None if torch.cuda.is_available() else -1
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=True, sort_key=lambda x: len(x.src),
            device=device, train=False)
        tgt_vocab = data.fields['tgt'].vocab
        pad = tgt_vocab.stoi[data.fields['tgt'].pad_token]

        for batch in batch_iterator:
            input_variables, input_lengths  = getattr(batch, 'src')
            target_variables = getattr(batch, 'tgt')

            decoder_outputs, decoder_hidden, other = model(input_variables, input_lengths.tolist(), target_variables)

            # Evaluation
            seqlist = other['sequence']
            for step, step_output in enumerate(decoder_outputs):
                target = target_variables[:, step + 1]
                loss.eval_batch(step_output.view(target_variables.size(0), -1), target)

                non_padding = target.ne(pad)
                correct = seqlist[step].view(-1).eq(target).masked_select(non_padding).sum()
                match += correct
                total += non_padding.sum()

        # if (total == 0).data.cpu().numpy():
        if total == 0:
            accuracy = float('nan')
        else:
            accuracy = match / total

        return loss.get_loss(), accuracy
