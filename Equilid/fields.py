import logging

import torchtext

class SourceField(torchtext.data.Field):
    """ Wrapper class of torchtext.sampleData.Field that forces batch_first and include_lengths to be True. """

    def __init__(self, **kwargs):
        logger = logging.getLogger(__name__)

        if kwargs.get('batch_first') is False:
            logger.warning("Option batch_first has to be set to use pytorch-seq2seq.  Changed to True.")
        kwargs['batch_first'] = True
        if kwargs.get('include_lengths') is False:
            logger.warning("Option include_lengths has to be set to use pytorch-seq2seq.  Changed to True.")
        kwargs['include_lengths'] = True

        super(SourceField, self).__init__(**kwargs)

    def process(self, batch, device=None):
        """ Process a list of examples to create a torch.Tensor.
        Pad, numericalize, and postprocess a batch and create a tensor.
        Args:
            batch (list(object)): A list of object from a batch of examples.
        Returns:
            torch.autograd.Variable: Processed object given the input
                and custom postprocessing Pipeline.
        """
        super(SourceField, self).process(batch,device)

    def set_specials(self, specials):
        '''
        use this when you don't want to build the whole vocab, but just wish to have special constants
        :param specials = dict['spl_var':spl_string]
        :return: None
        '''
        for spl_var,spl_symbol in specials.items():
            setattr(self,spl_var,spl_symbol)

class TargetField(torchtext.data.Field):
    """ Wrapper class of torchtext.sampleData.Field that forces batch_first to be True and prepend <sos> and append <eos> to sequences in preprocessing step.

    Attributes:
        sos_id: index of the start of sentence symbol
        eos_id: index of the end of sentence symbol
    """

    SYM_SOS = '<sos>'
    SYM_EOS = '<eos>'

    def __init__(self, **kwargs):
        logger = logging.getLogger(__name__)

        if kwargs.get('batch_first') == False:
            logger.warning("Option batch_first has to be set to use pytorch-seq2seq.  Changed to True.")
        kwargs['batch_first'] = True
        if kwargs.get('preprocessing') is None:
            kwargs['preprocessing'] = lambda seq: [self.SYM_SOS] + seq + [self.SYM_EOS]
        else:
            func = kwargs['preprocessing']
            kwargs['preprocessing'] = lambda seq: [self.SYM_SOS] + func(seq) + [self.SYM_EOS]

        self.sos_id = None
        self.eos_id = None
        super(TargetField, self).__init__(**kwargs)

    def process(self, batch, device=None):
        """ Process a list of examples to create a torch.Tensor.
        Pad, numericalize, and postprocess a batch and create a tensor.
        Args:
            batch (list(object)): A list of object from a batch of examples.
        Returns:
            torch.autograd.Variable: Processed object given the input
                and custom postprocessing Pipeline.
        """
        super(TargetField, self).process(batch,device)

    def set_specials(self, specials):
        '''
        use this when you don't want to build the whole vocab, but just wish to have special constants
        :param specials = dict['spl_var':spl_string]
        :return: None
        '''
        for spl_var,spl_symbol in specials.items():
            setattr(self, spl_var, spl_symbol)

    def build_vocab(self, *args, **kwargs):
        super(TargetField, self).build_vocab(*args, **kwargs)
        self.sos_id = self.vocab.stoi[self.SYM_SOS]
        self.eos_id = self.vocab.stoi[self.SYM_EOS]



