# -*- coding: utf-8 -*-
import pandas as pd

class read_source_files():
    '''Read source files using config file configuration

        Parameters
        ----------
        None
        

        Raises
        ------
        RuntimeError
            None

        Returns
        -------
        source dataframe
            reading source datafile using config parameters
        RFM dataframe
            Calculate RFTM from transactions
        Clusters
            Calculate Frequency clusters
            
        '''
    def __init__(self):
        """
        docstring
        """  
   
    def read_source_files_all(self, config):
        """
        docstring
        """
        self.config = config
        self.sales_train_validation_file = self.config['parameters']['sales_train_validation']
        self.sales_train_validation = pd.read_csv(self.sales_train_validation_file)
        
        self.config = config
        self.calendar_file = self.config['parameters']['calendar']
        self.calendar = pd.read_csv(self.calendar_file)
        
        self.config = config
        self.sell_prices_file = self.config['parameters']['sell_prices']
        self.sell_prices = pd.read_csv(self.sell_prices_file)
        
        self.config = config
        self.sample_submission_file = self.config['parameters']['sample_submission']
        self.sample_submission = pd.read_csv(self.sample_submission_file)
        
        return self.sales_train_validation, self.calendar, self.sell_prices, self.sample_submission
