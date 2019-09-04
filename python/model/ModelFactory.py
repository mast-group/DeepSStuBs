import json
import os

import python.model.AbstractModel
# from AbstractModel import *
import python.model.Word2VecModel
import python.model.FastTextModel
import python.model.ELMoModel
import python.model.ELMoBPEModel

from Word2VecModel import *
from FastTextModel import *
from ELMoBPEModel import *
from ELMoModel import *
from BPEModel import *


# Supported models 
supported_models = ['w2v', 'FastText', 'BPE', 'ELMo', 'ELMoBPE']


class ModelFactory:

    def __init__(self, config_file, sess=None):
        """[summary]
        
        Arguments:
            config_file {[type]} -- [description]
        """
        assert(config_file.endswith('.json'))
        with open(config_file, 'r') as f:
            self.settings = json.load(f)
        assert(self.settings['model'] in supported_models)
        self._sess = sess


    def get_model(self):
        """[summary]
        
        Raises:
            KeyError: [description]
        
        Returns:
            [type] -- [description]
        """
        model = self.settings['model']
        assert(model in supported_models)
        
        if model == 'w2v':
            return Word2VecModel(self.settings['model_file'])
        elif model == 'FastText':
            return FastTextModel(self.settings['model_file'])
        elif model == 'BPE':
            assert self._sess is not None
            
            # data_dir = self.settings['data_dir']
            model_dir = self.settings['model_dir'], 
            vocab_file = os.path.join(model_dir, self.settings['vocab_file'])
            
            nlmBPEModel = BPEModel(model_dir, vocab_file, self._sess)
            self._sess.run(tf.global_variables_initializer())
            return nlmBPEModel
        elif model == 'ELMo':
            assert self._sess is not None
            
            data_dir = '/disk/scratch/mpatsis/eddie/data/phog/js/'
            data_dir = self.settings['data_dir']
            model_dir = self.settings['model_dir']
            vocab_file = os.path.join(data_dir, self.settings['vocab_file'])
            weight_file = os.path.join(model_dir, self.settings['weight_file'])
            options_file = os.path.join(model_dir, self.settings['options_file'])
        
            elmoModel = ELMoModel(model_dir, vocab_file, weight_file, options_file, self._sess)
            self._sess.run(tf.global_variables_initializer())
            # if 'warm_up' in self.settings:
            #     elmoModel.warm_up(self.settings['warm_up'])
            return elmoModel
        elif model == 'ELMoBPE':
            assert self._sess is not None
            
            data_dir = self.settings['data_dir']
            model_dir = self.settings['model_dir'], 
            vocab_file = os.path.join(data_dir, self.settings['vocab_file'])
            weight_file = os.path.join(model_dir, self.settings['weight_file'])
            options_file = os.path.join(model_dir, self.settings['options_file'])
            
            elmoBPEModel = ELMoBPEModel(model_dir, vocab_file, weight_file, options_file, self._sess)
            self._sess.run(tf.global_variables_initializer())
            return elmoBPEModel
    
    def get_model_type(self):
        """[summary]
        """
        return self.settings['model']
