import pandas as pd
import numpy as np
import lightgbm as lgbm
import xgboost as xgb

"""
Decision point: 
    Multiple options   
    1. more generic - complex pipeline but easy to extend
        We create one pipeline manager for all model types XGB, LGBM, sklearn etc.
        1. This requires pipeline specific config and blocks
        2. Similar to class calling dynamic methods
        3. Every method has own parameters
        4. Simple to extend for adding new models
    2. model generic - simple pipeline but hard to extend and requires lot of maintenance
        We create dedicated pipeline manager for every model type XGB, LGBM, sklearn etc.
        1. This requires model dedicated functions
        2. Simplifies pipeline process
        3. Complex to extend
Prav - decision notes
    Taking more generic approach for easy to extend and less maintenance
    There is no right or wrong approach here, it depends on context
"""

class PipelineManager:
    """
    PipelineManager class to call model
    train, predict, evaluate steps
    
    Args:
        trainingSet: pandas dataframe to train model
        validationSet: pandas dataframe to validate model at run time
        feature_names: list of feature names to subset training 
            and validation sets
        target_feature_name: target feature name to subset training 
            and validation sets at run time
        config: configuration dic to read settings to construct parameters
        pipeline_name: model selection name from config file settings
    
    Returns:
        train: trained model, depends on config pipeline name setting
        predict: prediction results array
        evaluate: dic - metric results
    
    Raises:
        None
    """
    def train(self 
              ,trainingSet
              ,validationSet
              ,feature_names
              ,target_feature_name
              ,config 
              ,pipeline_name
              ):
        return train( trainingSet
                      ,validationSet
                      ,feature_names
                      ,target_feature_name
                      ,config
                      ,pipeline_name)
    def train_full(self 
              ,trainingSet
              ,validationSet
              ,feature_names
              ,target_feature_name
              ,config 
              ,pipeline_name
              ):
        return train( trainingSet
                      ,validationSet
                      ,feature_names
                      ,target_feature_name
                      ,config
                      ,pipeline_name)
        
    def predict(self 
                ,model
                ,testingSet
                ,feature_names
                ,pipeline_name
                ):
        return predict( model
                        ,testingSet
                        ,feature_names
                        ,pipeline_name
                      )
              
    
    def evaluate(self ,y, y_pred):
        return evaluate(y, 
                        y_pred
                       )

    def train_evaluate(self ,pipeline_name):
        train_evaluate(pipeline_name)

    def train_evaluate_predict(self ,pipeline_name):
        train_evaluate_predict(pipeline_name)

def train(trainingSet, 
          validationSet, 
          feature_names,
          target_feature_name,
          config,
          pipeline_name
          ):
    """
    model training
        
    Args:
        trainingSet: pandas dataframe to train model
        validationSet: pandas dataframe to validate model at run time
        feature_names: list of feature names to subset training 
            and validation sets
        target_feature_name: target feature name to subset training 
            and validation sets at run time
        config: configuration dic to read settings to construct parameters
        pipeline_name: model selection name from config file settings
    
    Returns:
        train: trained model, depends on config pipeline name setting
    
    Raises:
        None
    """
    if(pipeline_name == "lightgbm"):
        model_parameters = {
                               'boosting_type': config['parameters']['lgbm__boosting_type']
                              ,'objective': config['parameters']['lgbm__objective']
                              ,'metric': config['parameters']['lgbm__metric']
                              ,'min_child_weight': config['parameters']['lgbm__min_child_samples']
                              ,'num_leaves': config['parameters']['lgbm__num_leaves']
                              ,'feature_fraction': config['parameters']['lgbm__feature_fraction']
                              ,'bagging_fraction': config['parameters']['lgbm__bagging_fraction']
                              ,'bagging_freq': config['parameters']['lgbm__bagging_frequency']
                              ,'learning_rate': config['parameters']['lgbm__learning_rate']                              
                              ,'min_data_in_leaf': config['parameters']['lgbm__min_data_in_leaf']                                 
                              ,'nthread': config['parameters']['lgbm__n_thread']
                              ,'seed': config['parameters']['lgbm__seed']
                           }
        
        lgbmbuild = lgbm.Dataset(trainingSet[feature_names], trainingSet[target_feature_name],categorical_feature=config['parameters']['categorical_features'])
        lgbmval   = lgbm.Dataset(validationSet[feature_names], validationSet[target_feature_name],categorical_feature=config['parameters']['categorical_features'])
        
        model = lgbm.train(model_parameters, 
                           lgbmbuild,
                           config['parameters']['lgbm__number_boosting_rounds'],
                           valid_sets=[lgbmval,lgbmbuild],
                           valid_names=['valid','train'],
                           verbose_eval = config['parameters']['lgbm__verbose_eval'],
                           early_stopping_rounds= config['parameters']['lgbm__early_stopping_rounds']
                          )
        return model
    
    if(pipeline_name == "xgboost"):
        model_parameters = {                               
                               'objective': config['parameters']['xgb__objective']
                              ,'metric': config['parameters']['xgb__eval_metric']
                              ,'eta': config['parameters']['xgb__eta']
                              ,'max_depth': config['parameters']['xgb__max_depth']
                              ,'min_child_weight': config['parameters']['xgb__min_child_weight']                              
                              ,'subsample': config['parameters']['xgb__subsample']
                              ,'colsample_bytree': config['parameters']['xgb__colsample_bytree']  
                              ,'nthread': config['parameters']['xgb__n_thread']
                              ,'seed': config['parameters']['xgb__seed']
                           }
        
        xgbbuild = xgb.DMatrix(trainingSet[feature_names], label=trainingSet[target_feature_name])
        xgbval   = xgb.DMatrix(validationSet[feature_names], label=validationSet[target_feature_name])
        watchlist = [ (xgbbuild,'build'), (xgbval, 'valid') ]
    
        model = xgb.train(model_parameters, 
                          xgbbuild, 
                          config['parameters']['xgb__number_boosting_rounds'], 
                          watchlist,
                          verbose_eval = config['parameters']['xgb__print_every_n']                 
                          )      
        
        return model

def train_full(trainingSet, 
          validationSet, 
          feature_names,
          target_feature_name,
          config,
          pipeline_name
          ):
    """
    model training
        
    Args:
        trainingSet: pandas dataframe to train model
        validationSet: pandas dataframe to validate model at run time
        feature_names: list of feature names to subset training 
            and validation sets
        target_feature_name: target feature name to subset training 
            and validation sets at run time
        config: configuration dic to read settings to construct parameters
        pipeline_name: model selection name from config file settings
    
    Returns:
        train: trained model, depends on config pipeline name setting
    
    Raises:
        None
    """
    if(pipeline_name == "lightgbm"):
        model_parameters = {
                               'boosting_type': config['parameters']['lgbm__boosting_type']
                              ,'objective': config['parameters']['lgbm__objective']
                              ,'metric': config['parameters']['lgbm__metric']
                              ,'min_child_weight': config['parameters']['lgbm__min_child_samples']
                              ,'num_leaves': config['parameters']['lgbm__num_leaves']
                              ,'feature_fraction': config['parameters']['lgbm__feature_fraction']
                              ,'bagging_fraction': config['parameters']['lgbm__bagging_fraction']
                              ,'bagging_freq': config['parameters']['lgbm__bagging_frequency']
                              ,'learning_rate': config['parameters']['lgbm__learning_rate']                              
                              ,'min_data_in_leaf': config['parameters']['lgbm__min_data_in_leaf']                                 
                              ,'nthread': config['parameters']['lgbm__n_thread']
                              ,'seed': config['parameters']['lgbm__seed']
                           }
        
        lgbmbuild = lgbm.Dataset(trainingSet[feature_names], trainingSet[target_feature_name],categorical_feature=config['parameters']['categorical_features'])
#        lgbmval   = lgbm.Dataset(validationSet[feature_names], validationSet[target_feature_name],categorical_feature=config['parameters']['categorical_features'])
        
        model = lgbm.train(model_parameters, 
                           lgbmbuild,
                           config['parameters']['lgbm__number_boosting_rounds'],
                           valid_sets=[lgbmbuild],
                           valid_names=['train'],
                           verbose_eval = config['parameters']['lgbm__verbose_eval'],
                           early_stopping_rounds= config['parameters']['lgbm__early_stopping_rounds']
                          )
        return model
    
    if(pipeline_name == "xgboost"):
        model_parameters = {                               
                               'objective': config['parameters']['xgb__objective']
                              ,'metric': config['parameters']['xgb__eval_metric']
                              ,'eta': config['parameters']['xgb__eta']
                              ,'max_depth': config['parameters']['xgb__max_depth']
                              ,'min_child_weight': config['parameters']['xgb__min_child_weight']                              
                              ,'subsample': config['parameters']['xgb__subsample']
                              ,'colsample_bytree': config['parameters']['xgb__colsample_bytree']  
                              ,'nthread': config['parameters']['xgb__n_thread']
                              ,'seed': config['parameters']['xgb__seed']
                           }
        
        xgbbuild = xgb.DMatrix(trainingSet[feature_names], label=trainingSet[target_feature_name])
#        xgbval   = xgb.DMatrix(validationSet[feature_names], label=validationSet[target_feature_name])
#        watchlist = [ (xgbbuild,'build'), (xgbval, 'valid') ]
    
        model = xgb.train(model_parameters, 
                          xgbbuild, 
                          config['parameters']['xgb__number_boosting_rounds'], 
#                          watchlist,
                          verbose_eval = config['parameters']['xgb__print_every_n']                 
                          )      
        
        return model
    
def predict(model,
            testingSet,
            feature_names,
            pipeline_name):
    """
    predict results for test data using trained model
    
    Args:
        model: trained model
        testingSet: pandas dataframe to score predictions
        feature_names: list of feature names to subset training 
            and validation sets
        pipeline_name: model selection name from config file settings
    
    Returns:
        predictions: prediction resultset
    
    Raises:
        None
    """
    if(pipeline_name == "lightgbm"):
        predictions = model.predict(testingSet[feature_names])
        return predictions
    if(pipeline_name == "xgboost"):
        xgtest = xgb.DMatrix(testingSet[feature_names])
        predictions = model.predict(xgtest)
        return predictions

def evaluate(y, y_pred):
    """
    model evaluate 
    
    Args:
        y: actual values
        y_pred: predicted values
            
    Returns:
        evaluation_metrics: dict metric score results
    
    Raises:
        None
    """
    evaluation_metrics = dict()
    evaluation_metrics['rmse'] = np.sqrt(np.mean(np.square(y - y_pred)))
    evaluation_metrics['mape'] = np.mean(np.abs((y - y_pred) / y)) * 100
    return evaluation_metrics
    
def train_evaluate():
    """
    Prav : future use
    train()  & evaluate()
    """
    
def train_evaluate_predict():
    """
    Prav : future use
    train(), evaluate() & predict()
    """

