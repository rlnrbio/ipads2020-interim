# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 23:11:07 2020

@author: rapha
"""
import numpy as np


def accuracy(y_true, y_pred):
    
    """ 
    Function to calculate the accuracy of a binary classifier 
    
    Parameters:
    ---
    y_true: numpy array,
            ground truth values to be predicted
        
    y_pred: numpy array, 
            values predicted from model
    
    Returns:
    ---
    The accuracy of the binary model
    """
    
    #True Positives
    tp = sum(np.logical_and(y_true == True, y_pred == True))
    
    #True Negatives
    tn = sum(np.logical_and(y_true == False, y_pred == False))
    
    #False Positives
    fp = sum(np.logical_and(y_true == False, y_pred == True))
    
    #False Negatives
    fn = sum(np.logical_and(y_true == True, y_pred == False))
    
    acc = (tp + tn)/(tp + tn + fp + fn)
    return acc


def evaluation(y_true, y_pred):
    
    """ 
    Function to calculate the accuracy of a binary classifier 
    
    Parameters:
    ---
    y_true: numpy array,
            ground truth values to be predicted
        
    y_pred: numpy array, 
            values predicted from model
    
    Returns:
    ---
    A tuple with the following values:
    (percent correctly classified,  sensitivity, specificity, accuracy)
    """
    #True Positives
    tp = sum(np.logical_and(y_true == True, y_pred == True))
    
    #True Negatives
    tn = sum(np.logical_and(y_true == False, y_pred == False))
    
    #False Positives
    fp = sum(np.logical_and(y_true == False, y_pred == True))
    
    #False Negatives
    fn = sum(np.logical_and(y_true == True, y_pred == False))
    
    acc = (tp + tn)/(tp + tn + fp + fn)
    
    pc_correct = sum(y_true == y_pred)/len(y_true)
    
    sens = tp/(fn+tp)
    
    spec = tn/(fp+tn)
    
    return (pc_correct, sens, spec, acc)