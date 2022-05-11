#!/usr/bin/env python3

"""
Example class to read quark-gluon dataset
"""

import os
import sys
import argparse
import yaml
import h5py
import pickle
import subprocess
import functools
import shutil
import time

# Data analysis and plotting
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_context('paper', rc={'font.size':18,'axes.titlesize':18,'axes.labelsize':18})

# Energy flow package
import energyflow
import energyflow.archs

# sklearn
import sklearn
import sklearn.linear_model
import sklearn.ensemble
import sklearn.model_selection
import sklearn.pipeline

# Tensorflow and Keras
from tensorflow import keras
import keras_tuner

# Base class
sys.path.append('.')
from base import common_base

################################################################
class AnalyzeQG(common_base.CommonBase):

    #---------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------
    def __init__(self, config_file='', output_dir='', **kwargs):
        super(common_base.CommonBase, self).__init__(**kwargs)
        
        self.config_file = config_file
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Initialize config file
        self.initialize_config()

        self.filename = 'subjets_unshuffled.h5'
        with h5py.File(os.path.join(self.output_dir, self.filename), 'r') as hf:
            self.N_list = hf['N_list'][:]
            self.beta_list = hf['beta_list'][:]
            self.r_list = hf['r_list'][:]
            self.N_max_hf = hf['N_max'] 
            self.n_total_hf = hf['n_total']
            self.Clustering_Alg = hf['Clustering_Alg']
            self.Laman_construction = hf['Laman_construction']


        self.qa_observables = ['jet_pt', 'jet_angularity', 'thrust', 'LHA', 'pTD', 'jet_mass', 'jet_theta_g', 'zg', 'multiplicity_0000', 'multiplicity_0150', 'multiplicity_0500', 'multiplicity_1000']
            
        # Remove keras-tuner folder, if it exists
        if os.path.exists('keras_tuner'):
            shutil.rmtree('keras_tuner')

        print(self)
        print()
        
    #---------------------------------------------------------------
    # Initialize config file into class members
    #---------------------------------------------------------------
    def initialize_config(self):
    
        # Read config file
        with open(self.config_file, 'r') as stream:
          config = yaml.safe_load(stream)
          
        self.R = config['R']
        self.pt = config['pt']
        self.y_max = config['y_max']
          
        self.K_list = config['K']
        self.N_max = config['N_max']
        self.r_list = config['r'] #this is not necessary since the r_list is read from the output dir in the __init__

        self.q_label = config['q_label']
        self.g_label = config['g_label']

        self.n_train = config['n_train']
        self.n_val = config['n_val']
        self.n_test = config['n_test']
        self.n_total = self.n_train + self.n_val + self.n_test
        self.test_frac = 1. * self.n_test / self.n_total
        self.val_frac = 1. * self.n_val / (self.n_train + self.n_val)

        self.random_state = None  # seed for shuffling data (set to an int to have reproducible results)
        
        # Clustering Algorithm 
        self.Clustering_Alg = config['Clustering_Alg']

        #Laman Construction
        self.Laman_construction = config['Laman_construction']


        # Initialize model-specific settings
        self.config = config
        self.models = config['models']
        self.model_settings = {}
        for model in self.models:
            self.model_settings[model] = {}
            
            if 'dnn' in model:
                self.model_settings[model]['loss'] = config[model]['loss']
                self.model_settings[model]['learning_rate'] = config[model]['learning_rate']
                self.model_settings[model]['epochs'] = config[model]['epochs']
                self.model_settings[model]['batch_size'] = config[model]['batch_size']
                self.model_settings[model]['metrics'] = config[model]['metrics']
            
            if 'linear' in model:
                self.model_settings[model]['sgd_loss'] = config[model]['sgd_loss']
                self.model_settings[model]['sgd_penalty'] = config[model]['sgd_penalty']
                self.model_settings[model]['sgd_alpha'] = [float(x) for x in config[model]['sgd_alpha']]
                self.model_settings[model]['sgd_max_iter'] = config[model]['sgd_max_iter']
                self.model_settings[model]['sgd_tol'] = [float(x) for x in config[model]['sgd_tol']]
                self.model_settings[model]['sgd_learning_rate'] = config[model]['sgd_learning_rate']
                self.model_settings[model]['sgd_early_stopping'] = config[model]['sgd_early_stopping']
                self.model_settings[model]['n_iter'] = config[model]['n_iter']
                self.model_settings[model]['cv'] = config[model]['cv']
                self.model_settings[model]['lda_tol'] = [float(x) for x in config[model]['lda_tol']]

            if model == 'pfn':
                self.model_settings[model]['Phi_sizes'] = tuple(config[model]['Phi_sizes'])
                self.model_settings[model]['F_sizes'] = tuple(config[model]['F_sizes'])
                self.model_settings[model]['epochs'] = config[model]['epochs']
                self.model_settings[model]['batch_size'] = config[model]['batch_size']
                self.model_settings[model]['use_pids'] = config[model]['use_pids']
            
            if model == 'sub_pfn':
                self.model_settings[model]['Phi_sizes'] = tuple(config[model]['Phi_sizes'])
                self.model_settings[model]['F_sizes'] = tuple(config[model]['F_sizes'])
                self.model_settings[model]['epochs'] = config[model]['epochs']
                self.model_settings[model]['batch_size'] = config[model]['batch_size']
                self.model_settings[model]['use_pids'] = config[model]['use_pids']
                
            if model == 'efn':
                self.model_settings[model]['Phi_sizes'] = tuple(config[model]['Phi_sizes'])
                self.model_settings[model]['F_sizes'] = tuple(config[model]['F_sizes'])
                self.model_settings[model]['epochs'] = config[model]['epochs']
                self.model_settings[model]['batch_size'] = config[model]['batch_size']
                self.model_settings[model]['learning_rate'] = config[model]['learning_rate']

    #---------------------------------------------------------------
    # Main processing function
    #---------------------------------------------------------------
    def analyze_qg(self):
    
        # Clear variables
        self.AUC = {}
        self.units = {} # To save the hidden layers' size that keras.tuner calculated
        self.epochs = {} # If we use early stopping we need to know how many epochs were run
        self.y = None
        self.X_Nsub = None

        # Read in dataset
        with h5py.File(os.path.join(self.output_dir, self.filename), 'r') as hf:

            self.y_total = hf[f'y'][:]
            X_Nsub_total = hf[f'nsub'][:] 


            self.subjet_input_total={}
            for r in self.r_list:
                self.subjet_input_total[f'subjet_r{r}_edges'] = hf[f'subjet_r{r}_edges'][:]
                self.subjet_input_total[f'subjet_r{r}_angles'] = hf[f'subjet_r{r}_angles'][:]
                self.subjet_input_total[f'subjet_r{r}_z'] = hf[f'subjet_r{r}_z'][:]
                self.subjet_input_total[f'subjet_r{r}_sub_phi'] = hf[f'subjet_r{r}_sub_phi'][:]
                self.subjet_input_total[f'subjet_r{r}_sub_rap'] = hf[f'subjet_r{r}_sub_rap'][:]
                


            # Check whether any training entries are empty
            [print(f'WARNING: input entry {i} is empty') for i,x in enumerate(X_Nsub_total) if not x.any()]                                          

            # Determine total number of jets
            total_jets = int(self.y_total.size)
            total_jets_q = int(np.sum(self.y_total))
            total_jets_g = total_jets - total_jets_q
            print(f'Total number of jets available: {total_jets_q} (q), {total_jets_g} (g)')

            # If there is an imbalance, remove excess jets
            if total_jets_q > total_jets_g:
                indices_to_remove = np.where( np.isclose(self.y_total,1) )[0][total_jets_g:]
            elif total_jets_q <=  total_jets_g:
                indices_to_remove = np.where( np.isclose(self.y_total,0) )[0][total_jets_q:]
            y_balanced = np.delete(self.y_total, indices_to_remove)
            X_Nsub_balanced = np.delete(X_Nsub_total, indices_to_remove, axis=0)

            self.subjet_input_balanced={}
            for r in self.r_list:
                self.subjet_input_balanced[f'subjet_r{r}_edges'] = np.delete(self.subjet_input_total[f'subjet_r{r}_edges'], indices_to_remove, axis=0) 
                self.subjet_input_balanced[f'subjet_r{r}_angles'] = np.delete(self.subjet_input_total[f'subjet_r{r}_angles'], indices_to_remove, axis=0) 
                self.subjet_input_balanced[f'subjet_r{r}_z'] = np.delete(self.subjet_input_total[f'subjet_r{r}_z'], indices_to_remove, axis=0) 
                self.subjet_input_balanced[f'subjet_r{r}_sub_phi'] = np.delete(self.subjet_input_total[f'subjet_r{r}_sub_phi'], indices_to_remove, axis=0) 
                self.subjet_input_balanced[f'subjet_r{r}_sub_rap'] = np.delete(self.subjet_input_total[f'subjet_r{r}_sub_rap'], indices_to_remove, axis=0) 


            total_jets = int(y_balanced.size)
            total_jets_q = int(np.sum(y_balanced))
            total_jets_g = total_jets - total_jets_q
            print(f'Total number of jets available after balancing: {total_jets_q} (q), {total_jets_g} (g)')

            # Shuffle dataset 
            idx = np.random.permutation(len(y_balanced))
            if y_balanced.shape[0] == idx.shape[0]:
                y_shuffled = y_balanced[idx]
                X_Nsub_shuffled = X_Nsub_balanced[idx]


                self.subjet_input_shuffled={}
                for r in self.r_list:
                    self.subjet_input_shuffled[f'subjet_r{r}_edges'] = self.subjet_input_balanced[f'subjet_r{r}_edges'][idx]
                    self.subjet_input_shuffled[f'subjet_r{r}_angles'] = self.subjet_input_balanced[f'subjet_r{r}_angles'][idx]
                    self.subjet_input_shuffled[f'subjet_r{r}_z'] = self.subjet_input_balanced[f'subjet_r{r}_z'][idx]
                    self.subjet_input_shuffled[f'subjet_r{r}_sub_phi'] = self.subjet_input_balanced[f'subjet_r{r}_sub_phi'][idx]
                    self.subjet_input_shuffled[f'subjet_r{r}_sub_rap'] = self.subjet_input_balanced[f'subjet_r{r}_sub_rap'][idx]


            else:
                print(f'MISMATCH of shape: {y_balanced.shape} vs. {idx.shape}')

            # Truncate the input arrays to the requested size
            self.y = y_shuffled[:self.n_total]
            self.X_Nsub = X_Nsub_shuffled[:self.n_total]
            

            self.subjet_input={}
            for r in self.r_list:
                self.subjet_input[f'subjet_r{r}_edges'] = self.subjet_input_shuffled[f'subjet_r{r}_edges'][:self.n_total]
                self.subjet_input[f'subjet_r{r}_angles'] = self.subjet_input_shuffled[f'subjet_r{r}_angles'][:self.n_total]
                self.subjet_input[f'subjet_r{r}_z'] = self.subjet_input_shuffled[f'subjet_r{r}_z'][:self.n_total]
                self.subjet_input[f'subjet_r{r}_sub_phi'] = self.subjet_input_shuffled[f'subjet_r{r}_sub_phi'][:self.n_total]
                self.subjet_input[f'subjet_r{r}_sub_rap'] = self.subjet_input_shuffled[f'subjet_r{r}_sub_rap'][:self.n_total]
 

            print(f'y_shuffled sum: {np.sum(self.y)}')
            print(f'y_shuffled shape: {self.y.shape}')

            # Also get some QA info
            self.qa_results = {}
            for qa_observable in self.qa_observables:
                qa_result = hf[qa_observable][:self.n_total]
                if qa_result.shape[0] == 0:
                    continue
                self.qa_results[qa_observable] = qa_result
                self.qa_results[qa_observable][np.isnan(self.qa_results[qa_observable])] = -1.

        # Define formatted labels for features
        self.feature_labels = []
        for i,N in enumerate(self.N_list):
            beta = self.beta_list[i]
            self.feature_labels.append(r'$\tau_{}^{{{}}}$'.format(N,beta))

        # Split into training and test sets
        # We will split into validatation sets (for tuning hyperparameters) separately for each model
        X_Nsub_train, X_Nsub_test, self.y_train, self.y_test = sklearn.model_selection.train_test_split(self.X_Nsub, self.y, test_size=self.test_frac)
        test_jets = int(self.y_test.size)
        test_jets_q = int(np.sum(self.y_test))
        test_jets_g = test_jets - test_jets_q
        print(f'Total number of test jets: {test_jets_g} (g), {test_jets_q} (q)')

        self.X_laman_train={}
        self.X_laman_test={}
        self.y_laman_train={}
        self.y_laman_test={}
        self.X_subjet_train={}
        self.X_subjet_test={}
        self.y_subjet_train={}
        self.y_subjet_test={}


        # Train Laman DNN and Subjet DNN for different r
        for r in self.r_list:

            # Split into Train and Test sets for Laman DNN
            self.X_laman_train[f'r{r}'],self.X_laman_test[f'r{r}'],self.y_laman_train[f'r{r}'],self.y_laman_test[f'r{r}'] =  sklearn.model_selection.train_test_split(np.concatenate((self.subjet_input[f'subjet_r{r}_z'], self.subjet_input[f'subjet_r{r}_angles']),axis=1), self.y, test_size=self.test_frac)
            
            # Use as input to the Sub DNN (pt_1, η_1, φ_1,...,pt_n, η_n, φ_n) instead of (pt_1,...,pt_n, η_1,...η_n, φ_1, ... , φ_n)
            self.x_subjet_combined = []
            for n in range(self.N_max):
                self.x_subjet_combined.append(np.array(self.subjet_input[f'subjet_r{r}_z'][:,n]))
                self.x_subjet_combined.append(np.array(self.subjet_input[f'subjet_r{r}_sub_phi'][:,n]))
                self.x_subjet_combined.append(np.array(self.subjet_input[f'subjet_r{r}_sub_rap'][:,n]))

            # Τhe shape of self.x_subjet_combined is (3*N_max, n_total) instead of (n_tot , 3*N_max)
            self.x_subjet_combined = np.array(self.x_subjet_combined).T
            
            # Split into Train and Test sets for Subjet DNN
            self.X_subjet_train[f'r{r}'],self.X_subjet_test[f'r{r}'],self.y_subjet_train[f'r{r}'],self.y_subjet_test[f'r{r}'] = sklearn.model_selection.train_test_split(self.x_subjet_combined, self.y, test_size=self.test_frac)


        # Construct training/test sets for each K
        self.training_data = {}
        for K in self.K_list:
            n = 3*K-4
            self.training_data[K] = {}
            self.training_data[K]['X_Nsub_train'] = X_Nsub_train[:,:n]
            self.training_data[K]['X_Nsub_test'] = X_Nsub_test[:,:n]
            self.training_data[K]['N_list'] = self.N_list[:n]
            self.training_data[K]['beta_list'] = self.beta_list[:n]
            self.training_data[K]['feature_labels'] = self.feature_labels[:n]

        # Set up dict to store roc curves
        self.roc_curve_dict = {}
        for model in self.models:
            self.roc_curve_dict[model] = {}
                        
        # Plot the input data
        self.plot_QA()

        # Plot first few K (before and after scaling)
        if 'dnn' in self.models and K == 3:
            self.plot_nsubjettiness_distributions(K, self.training_data[K]['X_Nsub_train'], self.y_train, self.training_data[K]['feature_labels'], 'before_scaling')
            self.plot_nsubjettiness_distributions(K, sklearn.preprocessing.scale(self.training_data[K]['X_Nsub_train']), self.y_train, self.training_data[K]['feature_labels'], 'after_scaling')

        # Train models
        self.train_models()

        self.diff = np.array(self.AUC['laman_dnn']) - np.array(self.AUC['sub_dnn']) 

        print(f'Dataset size : {self.n_total}')
        print(f'r_list : {self.r_list}')
        print(f'K_list : {self.K_list}')
        print(f'N max : {self.N_max}')
        print(f'Layers: {self.units}' )
        print(f'Epochs: {self.epochs}' )
        print(f'Clustering Algorithm : {self.Clustering_Alg}')
        print(f'Laman Construction : {self.Laman_construction}')
        print(f'AUC : {self.AUC}' )
        print(f'AUC Difference Laman - Sub : {self.diff}')
        
        # Run plotting script
        print('Run plotting script...')
        cmd = f'python analysis/plot_qg.py -c {self.config_file} -o {self.output_dir}'
        subprocess.run(cmd, check=True, shell=True)

    #---------------------------------------------------------------
    # Train models
    #---------------------------------------------------------------
    def train_models(self):

        # Train ML models
        for model in self.models:
            print()
        
            # Dict to store AUC
            self.AUC[model] = []
        
            model_settings = self.model_settings[model]

            # Nsubjettiness  
            for K in self.K_list:
                if model == 'nsub_linear':
                    self.fit_nsub_linear(model, model_settings, K)
                if model == 'nsub_dnn':
                    self.fit_nsub_dnn(model, model_settings, K)

            # Deep sets
            if model in ['pfn', 'efn']:
                if model == 'pfn':
                    self.fit_pfn(model, model_settings)
                if model == 'efn':
                    self.fit_efn(model, model_settings)
            

            for r in self.r_list:
                # Laman Graphs DNN
                if model == 'laman_dnn':
                    self.fit_laman_dnn(model, model_settings,r)
                # Subjet DNN based on pt ordering 
                if model == 'sub_dnn':
                    self.fit_subjet_dnn(model, model_settings,r)
                # Subjet PFN 
                if model == 'sub_pfn':
                    self.fit_sub_pfn(model, model_settings,r)
            


        # Plot traditional observables
        for observable in self.qa_observables:
            self.roc_curve_dict[observable] = sklearn.metrics.roc_curve(self.y_total[:self.n_total], -self.qa_results[observable])

        # Save ROC curves to file
        if 'nsub_dnn' in self.models or 'sub_dnn' in self.models or 'laman_dnn' in self.models  or 'nsub_linear' in self.models  or 'pfn' in self.models or 'sub_pfn' in self.models or 'efn' in self.models:
            output_filename = os.path.join(self.output_dir, f'ROC.pkl')
            with open(output_filename, 'wb') as f:
                pickle.dump(self.roc_curve_dict, f)
                pickle.dump(self.AUC, f)

    #---------------------------------------------------------------
    # Fit linear model for Nsubjettiness
    #---------------------------------------------------------------
    def fit_nsub_linear(self, model, model_settings, K):

        X_train = self.training_data[K]['X_Nsub_train']
        X_test = self.training_data[K]['X_Nsub_test']
        y_train = self.y_train
        y_test = self.y_test

        self.fit_linear_model(X_train, y_train, X_test, y_test, model, model_settings, dim_label='K', dim=K, type='LDA_search')

    #---------------------------------------------------------------
    # Fit Dense Neural Network for Nsubjettiness
    #---------------------------------------------------------------
    def fit_nsub_dnn(self, model, model_settings, K):

        # Preprocessing: zero mean unit variance
        X_Nsub_train = sklearn.preprocessing.scale(self.training_data[K]['X_Nsub_train'])
        X_Nsub_test = sklearn.preprocessing.scale(self.training_data[K]['X_Nsub_test'])

        self.fit_dnn(X_Nsub_train, self.y_train, X_Nsub_test, self.y_test, model, model_settings, dim_label='K', dim=K)

    #---------------------------------------------------------------
    # Fit Dense Neural Network for Laman Graphs
    #---------------------------------------------------------------
    def fit_laman_dnn(self, model, model_settings,r):

        # Preprocessing: zero mean unit variance
        X_laman_train = sklearn.preprocessing.scale(self.X_laman_train[f'r{r}'])
        X_laman_test = sklearn.preprocessing.scale(self.X_laman_test[f'r{r}'])

        self.fit_dnn_subjet(X_laman_train, self.y_laman_train[f'r{r}'], X_laman_test, self.y_laman_test[f'r{r}'], model, model_settings,r) 
  

    #---------------------------------------------------------------
    # Fit Dense Neural Network for Subjets (not Deep Sets)
    #---------------------------------------------------------------
    def fit_subjet_dnn(self, model, model_settings,r):

        # Preprocessing:
        # There are two types of preprocessing to consider:
        #   (1) Center each jet around eta,phi=0
        #   (2) Normalize each feature to zero mean and unit variance (across the full set of jets)
        #
        # Note that for nsubjettiness/laman, there is no need to do (1), since the features are independent of coordinates.
        # 
        # We should treat the zero-pads carefully during the preprocessing. There are several possibilities:
        #   (A) Center all the non-zero-pad values, then keep the zero-pad values at 0
        #   (B) Center all the non-zero-pad values, and also shift the zero-pad values to a constant at the lower edge of the non-zero-pad range
        #   (C) Don't center, and normalize all the values including the zero-pad values
        # It seems to me that option (B) should be the best...
        # 
        # Note that for nsubjettiness, 0 is at the lower edge of the allowed range.
        #
        # For subjet/laman cases, we probably want to avoid (C) since 0 is not the lower edge of the angular variables. 
        # Moreover, the features depend on the jet angular coordinates, so we want to center the jets themselves:
        #  - Using sklearn.preprocessing.scale, this would normalize each feature (i.e. {phi1,i}, {phi2,i}) independently,
        #    and would result in nonzero mean for each jet (since eta_avg,phi_avg=0) but on-average unit variance.
        #  - Instead, we can manually center each jet's eta-phi at 0 -- i.e. loop through subjets and 
        #    normalize {phi1, ..., phi_nmax} to zero mean and range [-R, R]. We could further scale the variance to 1.
        preprocessing_option = 'B'

        if preprocessing_option == 'A':

            X_subjet_train = self.center_eta_phi(self.X_subjet_train[f'r{r}'], shift_zero_pads=False)
            X_subjet_test = self.center_eta_phi(self.X_subjet_test[f'r{r}'], shift_zero_pads=False)

        elif preprocessing_option == 'B':

            X_subjet_train = self.center_eta_phi(self.X_subjet_train[f'r{r}'], shift_zero_pads=True)
            X_subjet_test = self.center_eta_phi(self.X_subjet_test[f'r{r}'], shift_zero_pads=True)

            # TODO: should we also fix the orientation, e.g. rotate to align leading and subleading subjets?

            X_subjet_train = sklearn.preprocessing.scale(X_subjet_train)
            X_subjet_test = sklearn.preprocessing.scale(X_subjet_test)

        elif preprocessing_option == 'C':
            X_subjet_train = sklearn.preprocessing.scale(self.X_subjet_train[f'r{r}'])
            X_subjet_test = sklearn.preprocessing.scale(self.X_subjet_test[f'r{r}'])

        self.fit_dnn_subjet(X_subjet_train, self.y_subjet_train[f'r{r}'], X_subjet_test, self.y_subjet_test[f'r{r}'], model, model_settings,r) 

    #---------------------------------------------------------------
    # Center eta-phi of subjets 
    #---------------------------------------------------------------
    def center_eta_phi(self, X, shift_zero_pads=True):

        # Since there are a different number of non-empty subjets per jet, we loop through jets and normalize pt/eta/phi for each set of subjets.
        for i,jet in enumerate(X):

            # Compute eta-phi avg
            # For convenience, reshape (z_1, η_1, φ_1,...,z_n, η_n, φ_n) --> [[z_1, η_1, φ_1], ...]
            jet_2d = np.reshape(jet, (-1, 3))
            mask = jet_2d[:,0] > 0
            yphi_avg = np.average(jet_2d[mask,1:], weights=jet_2d[mask,0], axis=0)
            jet_2d[mask,1:] -= yphi_avg
            if shift_zero_pads:
                jet_2d[~mask,1:] -= self.R
            X[i] = jet_2d.ravel()

        return X

    #---------------------------------------------------------------
    # Fit ML model -- SGDClassifier or LinearDiscriminant
    #   - SGDClassifier: Linear model (SVM by default, w/o kernel) with SGD training
    #   - For best performance, data should have zero mean and unit variance
    #---------------------------------------------------------------
    def fit_linear_model(self, X_train, y_train, X_test, y_test, model, model_settings, dim_label='', dim=None, type='SGD'):
        print(f'Training {model} ({type}), {dim_label}={dim}...')
        
        if type == 'SGD':
        
            # Define model
            clf = sklearn.linear_model.SGDClassifier(loss=model_settings['sgd_loss'],
                                                        max_iter=model_settings['sgd_max_iter'],
                                                        learning_rate=model_settings['sgd_learning_rate'],
                                                        early_stopping=model_settings['sgd_early_stopping'],
                                                        random_state=self.random_state)

            # Optimize hyperparameters with random search, using cross-validation to determine best set
            # Here we just search over discrete values, although can also easily specify a distribution
            param_distributions = {'penalty': model_settings['sgd_penalty'],
                                'alpha': model_settings['sgd_alpha'],
                                'tol': model_settings['sgd_tol']}

            randomized_search = sklearn.model_selection.RandomizedSearchCV(clf, param_distributions,
                                                                           n_iter=model_settings['n_iter'],
                                                                           cv=model_settings['cv'],
                                                                           random_state=self.random_state)
            search_result = randomized_search.fit(X_train, y_train)
            final_model = search_result.best_estimator_
            result_info = search_result.cv_results_
            print(f'Best params: {search_result.best_params_}')

            # Get predictions for the test set
            #y_predict_train = final_model.predict(X_train)
            #y_predict_test = final_model.predict(X_test)
            
            y_predict_train = sklearn.model_selection.cross_val_predict(clf, X_train, y_train, cv=3, method="decision_function")
            
            # Compare AUC on train set and test set
            AUC_train = sklearn.metrics.roc_auc_score(y_train, y_predict_train)
            print(f'AUC = {AUC_train} (cross-val train set)')
            print()
            
            # Compute ROC curve: the roc_curve() function expects labels and scores
            self.roc_curve_dict[model][dim] = sklearn.metrics.roc_curve(y_train, y_predict_train)

            # Store AUC
            self.AUC[f'{model}'].append(AUC_train)
            # Check number of thresholds used for ROC curve
            # print('thresholds: {}'.format(self.roc_curve_dict[model][K][2]))
            
            # Plot confusion matrix
            #self.plot_confusion_matrix(self.y_train, y_predict_train, f'{model}_K{K}')

        elif type == 'LDA':

            # energyflow implementation
            clf = energyflow.archs.LinearClassifier(linclass_type='lda')
            history = clf.fit(X_train, y_train)
            preds_EFP = clf.predict(X_test)        
            auc_EFP = sklearn.metrics.roc_auc_score(y_test,preds_EFP[:,1])
            print(f'  AUC = {auc_EFP} (test set)')
            self.AUC[f'{model}'].append(auc_EFP)
            self.roc_curve_dict[model][dim] = sklearn.metrics.roc_curve(y_test, preds_EFP[:,1])

        elif type == 'LDA_search':

            # Define model
            clf = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()

            # Optimize hyperparameters
            param_distributions = {'tol': model_settings['lda_tol']}

            randomized_search = sklearn.model_selection.GridSearchCV(clf, param_distributions)
            search_result = randomized_search.fit(X_train, y_train)
            final_model = search_result.best_estimator_
            result_info = search_result.cv_results_
            print(f'Best params: {search_result.best_params_}')

            y_predict_train = sklearn.model_selection.cross_val_predict(clf, X_train, y_train, cv=3, method="decision_function")
            
            # Compare AUC on train set and test set
            AUC_train = sklearn.metrics.roc_auc_score(y_train, y_predict_train)
            print(f'AUC = {AUC_train} (cross-val train set)')
            print()

            # Compute ROC curve: the roc_curve() function expects labels and scores
            self.roc_curve_dict[model][dim] = sklearn.metrics.roc_curve(y_train, y_predict_train)

    #---------------------------------------------------------------
    # Train DNN, using hyperparameter optimization with keras tuner
    #---------------------------------------------------------------
    def fit_dnn(self, X_train, Y_train, X_test, Y_test, model, model_settings, dim_label='', dim=None):
        print()
        print(f'Training {model}, {dim_label}={dim}...')

        tuner = keras_tuner.Hyperband(functools.partial(self.dnn_builder, input_shape=[X_train.shape[1]], model_settings=model_settings),
                                        objective='val_accuracy',
                                        max_epochs=10,
                                        factor=3,
                                        directory='keras_tuner',
                                        project_name=f'{model}{dim}')

        tuner.search(X_train, Y_train, 
                        batch_size=model_settings['batch_size'],
                        epochs=model_settings['epochs'], 
                        validation_split=self.val_frac)
        
        # Get the optimal hyperparameters
        best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
        units1 = best_hps.get('units1')
        units2 = best_hps.get('units2')
        units3 = best_hps.get('units3')
        learning_rate = best_hps.get('learning_rate')
        print()
        print(f'Best hyperparameters:')
        print(f'   units: ({units1}, {units2}, {units3})')
        print(f'   learning_rate: {learning_rate}')
        print()

        # Retrain the model with best number of epochs
        hypermodel = tuner.hypermodel.build(best_hps)
        history = hypermodel.fit(X_train, Y_train, epochs=model_settings['epochs'], validation_split=self.val_frac)

        # Plot metrics as a function of epochs
        self.plot_NN_epochs(model_settings['epochs'], history, model, dim_label=dim_label, dim=dim) 

        # Get predictions for test data set
        preds_DNN = hypermodel.predict(X_test).reshape(-1)
        
        # Get AUC
        auc_DNN = sklearn.metrics.roc_auc_score(Y_test, preds_DNN)
        print(f'  AUC = {auc_DNN} (test set)')
        
        # Store AUC
        self.AUC[f'{model}'].append(auc_DNN)
        
        # Get & store ROC curve
        self.roc_curve_dict[model][dim] = sklearn.metrics.roc_curve(Y_test, preds_DNN)


    #---------------------------------------------------------------
    # Train Subjet and Laman DNN, using hyperparameter optimization with keras tuner
    #---------------------------------------------------------------
    def fit_dnn_subjet(self, X_train, Y_train, X_test, Y_test, model, model_settings, r):
        print()
        print(f'Training {model} with r = {r}...')

        tuner = keras_tuner.Hyperband(functools.partial(self.dnn_builder, input_shape=[X_train.shape[1]], model_settings=model_settings),
                                        objective='val_accuracy',
                                        max_epochs=15,
                                        factor=3,
                                        directory='keras_tuner',
                                        project_name=f'{model} with r = {r}')

        tuner.search(X_train, Y_train, 
                        batch_size=model_settings['batch_size'],
                        epochs=model_settings['epochs'], 
                        validation_split=self.val_frac)
        
        # Get the optimal hyperparameters
        best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
        units1 = best_hps.get('units1')
        units2 = best_hps.get('units2')
        units3 = best_hps.get('units3')
        learning_rate = best_hps.get('learning_rate')
        print()
        print(f'Best hyperparameters:')
        print(f'   units: ({units1}, {units2}, {units3})')
        print(f'   learning_rate: {learning_rate}')
        print()

        self.units[f'{model}, r ={r}'] = [units1, units2, units3]
        # Retrain the model with best number of epochs
        hypermodel = tuner.hypermodel.build(best_hps)

        # Early Stopping
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=6)
        
        # Training
        history = hypermodel.fit(X_train, Y_train, epochs=model_settings['epochs'], validation_split=self.val_frac, callbacks =[early_stopping]) # We employ early stopping
        
        history_epochs = len(history.history['loss']) # For the x axis of the plot
        self.epochs[f'{model}, r = {r}'] = history_epochs

        # Plot metrics as a function of epochs
        self.plot_NN_epochs(history_epochs, history, model, dim_label= f'{model} DNN with r = ', dim=r)  
 
        # Get predictions for test data set
        preds_DNN = hypermodel.predict(X_test).reshape(-1)
        
        # Get AUC
        auc_DNN= sklearn.metrics.roc_auc_score(Y_test, preds_DNN)
        print(f'{model},  AUC = {auc_DNN} (test set)')
        
        # Store AUC
        self.AUC[f'{model}'].append(auc_DNN)
          
        # Get & store ROC curve
        self.roc_curve_dict[model][r] = sklearn.metrics.roc_curve(Y_test, preds_DNN)


    #---------------------------------------------------------------
    # Construct model for hyperparameter tuning with keras tuner
    #---------------------------------------------------------------
    def dnn_builder(self, hp, input_shape, model_settings):

        model = keras.models.Sequential()
        model.add(keras.layers.Flatten(input_shape=input_shape))

        # Tune size of first dense layer
        hp_units1 = hp.Int('units1', min_value=128, max_value=512, step=64)
        hp_units2 = hp.Int('units2', min_value=128, max_value=512, step=64)
        hp_units3 = hp.Int('units3', min_value=128, max_value=512, step=64)
        model.add(keras.layers.Dense(units=hp_units1, activation='relu'))
        model.add(keras.layers.Dense(units=hp_units2, activation='relu'))
        model.add(keras.layers.Dense(units=hp_units3, activation='relu'))
        model.add(keras.layers.Dense(1,activation='sigmoid'))  # For a binary classification softmax = sigmoid

        # Print DNN summary
        model.summary()

        # Tune the learning rate for the optimizer
        hp_learning_rate = hp.Choice('learning_rate', values=model_settings['learning_rate']) # if error, change name to lr or learning_rate

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),  # For Stochastic gradient descent use: SGD
                      loss=model_settings['loss'],
                      metrics=model_settings['metrics'])

        return model

    #---------------------------------------------------------------
    # Fit ML model -- Deep Set/Particle Flow Networks
    #---------------------------------------------------------------
    def fit_pfn(self, model, model_settings):
    
        # Load the four-vectors directly from the quark vs gluon data set
        X_PFN, y_PFN = energyflow.datasets.qg_jets.load(self.n_total)
        Y_PFN = energyflow.utils.to_categorical(y_PFN, num_classes=2)
        print(f'(n_jets, n_particles per jet, n_variables): {X_PFN.shape}')

        # Preprocess by centering jets and normalizing pts
        for x_PFN in X_PFN:
            mask = x_PFN[:,0] > 0
            yphi_avg = np.average(x_PFN[mask,1:3], weights=x_PFN[mask,0], axis=0)
            x_PFN[mask,1:3] -= yphi_avg
            x_PFN[mask,0] /= x_PFN[:,0].sum()
        
        # Handle particle id channel !! Note: If changed to pT,y,phi,m the 4th component is not PID but m .. fix later
        #if model_settings['use_pids']:
        #    self.my_remap_pids(X_PFN)
        #else:
        X_PFN = X_PFN[:,:,:3]
        
        # Check shape
        if y_PFN.shape[0] != X_PFN.shape[0]:
            print(f'Number of labels {y_PFN.shape} does not match number of jets {X_PFN.shape} ! ')

        # Split data into train, val and test sets
        (X_PFN_train, X_PFN_val, X_PFN_test, Y_PFN_train, Y_PFN_val, Y_PFN_test) = energyflow.utils.data_split(X_PFN, Y_PFN,
                                                                                             val=self.n_val, test=self.n_test)
        # Build architecture
        pfn = energyflow.archs.PFN(input_dim=X_PFN.shape[-1],
                                   Phi_sizes=model_settings['Phi_sizes'],
                                   F_sizes=model_settings['F_sizes'])
        # Early Stopping
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', patience=8)

        # Train model
        history = pfn.fit(X_PFN_train,
                          Y_PFN_train,
                          epochs=model_settings['epochs'],
                          batch_size=model_settings['batch_size'],
                          validation_data=(X_PFN_val, Y_PFN_val),
                          verbose=1, callbacks =[early_stopping])

        history_epochs = len(history.history['loss']) # For the x axis of the plot    
        self.epochs[f'{model}']= history_epochs  
        # Plot metrics are a function of epochs
        self.plot_NN_epochs(history_epochs, history, model)
        
        # Get predictions on test data
        preds_PFN = pfn.predict(X_PFN_test, batch_size=1000)

        # Get AUC and ROC curve + make plot
        auc_PFN = sklearn.metrics.roc_auc_score(Y_PFN_test[:,1], preds_PFN[:,1])
        print('Particle Flow Networks/Deep Sets: AUC = {} (test set)'.format(auc_PFN))
        self.AUC[f'{model}'].append(auc_PFN)
        
        self.roc_curve_dict[model] = sklearn.metrics.roc_curve(Y_PFN_test[:,1], preds_PFN[:,1])

    #---------------------------------------------------------------
    # Fit PFN for subjets
    #---------------------------------------------------------------
    def fit_sub_pfn(self, model, model_settings, r):
    
        x_subjet_input_sub_pfn = []
        for n in range(self.N_max):
            x_subjet_input_sub_pfn.append(np.array(self.subjet_input_total[f'subjet_r{r}_z'][:,n]))
            x_subjet_input_sub_pfn.append(np.array(self.subjet_input_total[f'subjet_r{r}_sub_rap'][:,n]))
            x_subjet_input_sub_pfn.append(np.array(self.subjet_input_total[f'subjet_r{r}_sub_phi'][:,n]))
        x_subjet_input_sub_pfn = np.array(x_subjet_input_sub_pfn).T
        x_subjet_input_sub_pfn = x_subjet_input_sub_pfn.reshape(self.n_total,self.N_max,3)  # To bring it to the form (n_total, n_particles, dof of each particle)

        # Preprocess by centering jets and normalizing pts
        for x_PFN in x_subjet_input_sub_pfn:
            mask = x_PFN[:,0] > 0
            yphi_avg = np.average(x_PFN[mask,1:3], weights=x_PFN[mask,0], axis=0)
            x_PFN[mask,1:3] -= yphi_avg
            x_PFN[mask,0] /= x_PFN[:,0].sum()

        self.y_total_sub_pfn = energyflow.utils.to_categorical(self.y_total, num_classes=2)

        

        # Split data into train, val and test sets
        (X_PFN_train, X_PFN_val, X_PFN_test, Y_PFN_train, Y_PFN_val, Y_PFN_test) = energyflow.utils.data_split(x_subjet_input_sub_pfn, self.y_total_sub_pfn,
                                                                                             val=self.n_val, test=self.n_test)
        # Build architecture
        pfn = energyflow.archs.PFN(input_dim=x_subjet_input_sub_pfn.shape[-1],
                                   Phi_sizes=model_settings['Phi_sizes'],
                                   F_sizes=model_settings['F_sizes'])

        # Early Stopping
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', patience=8)

        # Train model
        history = pfn.fit(X_PFN_train,
                          Y_PFN_train,
                          epochs=model_settings['epochs'],
                          batch_size=model_settings['batch_size'],
                          validation_data=(X_PFN_val, Y_PFN_val),
                          verbose=1, callbacks =[early_stopping])

        history_epochs = len(history.history['loss']) # For the x axis of the plot    
        self.epochs[f'{model}, r = {r}'] = history_epochs             
        # Plot metrics are a function of epochs
        self.plot_NN_epochs(history_epochs, history, model)
        
        # Get predictions on test data
        preds_PFN = pfn.predict(X_PFN_test, batch_size=1000)

        # Get AUC and ROC curve + make plot
        auc_PFN = sklearn.metrics.roc_auc_score(Y_PFN_test[:,1], preds_PFN[:,1])
        print('Particle Flow Networks/Deep Sets: AUC = {} (test set)'.format(auc_PFN))
        self.AUC[f'{model}'].append(auc_PFN)
        
        self.roc_curve_dict[model] = sklearn.metrics.roc_curve(Y_PFN_test[:,1], preds_PFN[:,1])


    #---------------------------------------------------------------
    # Fit ML model -- (IRC safe) Energy Flow Networks
    #---------------------------------------------------------------
    def fit_efn(self, model, model_settings):
    
        # Load the four-vectors directly from the quark vs gluon data set
        X_EFN, y_EFN = energyflow.datasets.qg_jets.load(self.n_total)
        Y_EFN = energyflow.utils.to_categorical(y_EFN, num_classes=2)
        print('(n_jets, n_particles per jet, n_variables): {}'.format(X_EFN.shape))

        # Preprocess data set by centering jets and normalizing pts
        # Note: this step is somewhat different for pp/AA compared to the quark/gluon data set -- check
        for x_EFN in X_EFN:
            mask = x_EFN[:,0] > 0
            
            # Compute y,phi averages
            yphi_avg = np.average(x_EFN[mask,1:3], weights=x_EFN[mask,0], axis=0)

            # Adjust phi range: Initially it is [0,2Pi], now allow for negative values and >2Pi 
            # so there are no gaps for a given jet.
            # Mask particles that are far away from the average phi & cross the 2Pi<->0 boundary
            mask_phi_1 = ((x_EFN[:,2] - yphi_avg[1] >  np.pi) & (x_EFN[:,2] != 0.))
            mask_phi_2 = ((x_EFN[:,2] - yphi_avg[1] < -np.pi) & (x_EFN[:,2] != 0.))
            
            x_EFN[mask_phi_1,2] -= 2*np.pi
            x_EFN[mask_phi_2,2] += 2*np.pi            
            
            # Now recompute y,phi averages after adjusting the phi range
            yphi_avg1 = np.average(x_EFN[mask,1:3], weights=x_EFN[mask,0], axis=0)            
            
            # And center jets in the y,phi plane
            x_EFN[mask,1:3] -= yphi_avg1

            # Normalize transverse momenta p_Ti -> z_i
            x_EFN[mask,0] /= x_EFN[:,0].sum()
            
            # Set particle four-vectors to zero if the z value is below a certain threshold.
            mask2 = x_EFN[:,0]<0.00001
            x_EFN[mask2,:]=0
        
        # Do not use PID for EFNs
        X_EFN = X_EFN[:,:,:3]
        
        # Make 800 four-vector array smaller, e.g. only 150. Ok w/o background
        X_EFN = X_EFN[:,:150]
        
        # Check shape
        if self.y.shape[0] != X_EFN.shape[0]:
            print(f'Number of labels {self.y.shape} does not match number of jets {X_EFN.shape} ! ')
            
        # Split data into train, val and test sets 
        # and separate momentum fraction z and angles (y,phi)
        (z_EFN_train, z_EFN_val, z_EFN_test, 
         p_EFN_train, p_EFN_val, p_EFN_test,
         Y_EFN_train, Y_EFN_val, Y_EFN_test) = energyflow.utils.data_split(X_EFN[:,:,0], X_EFN[:,:,1:], Y_EFN, 
                                                                           val=self.n_val, test=self.n_test)
        
        # Build architecture
        opt = keras.optimizers.Adam(learning_rate=model_settings['learning_rate']) # if error, change name to learning_rate
        efn = energyflow.archs.EFN(input_dim=2,
                                   Phi_sizes=model_settings['Phi_sizes'],
                                   F_sizes=model_settings['F_sizes'],
                                   optimizer=opt)
        
        # Early Stopping
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', patience=8)

        # Train model
        history = efn.fit([z_EFN_train,p_EFN_train],
                          Y_EFN_train,
                          epochs=model_settings['epochs'],
                          batch_size=model_settings['batch_size'],
                          validation_data=([z_EFN_val,p_EFN_val], Y_EFN_val),
                          verbose=1, callbacks =[early_stopping])

        history_epochs = len(history.history['loss']) # For the x axis of the plot
        self.epochs[f'{model}'] = history_epochs

        # Plot metrics are a function of epochs
        self.plot_NN_epochs(history_epochs, history, model)
        
        # Get predictions on test data
        preds_EFN = efn.predict([z_EFN_test,p_EFN_test], batch_size=1000)     

        # Get AUC and ROC curve + make plot
        auc_EFN = sklearn.metrics.roc_auc_score(Y_EFN_test[:,1], preds_EFN[:,1])
        print('(IRC safe) Energy Flow Networks: AUC = {} (test set)'.format(auc_EFN))
        self.AUC[f'{model}'].append(auc_EFN)
        
        self.roc_curve_dict[model] = sklearn.metrics.roc_curve(Y_EFN_test[:,1], preds_EFN[:,1])

    #--------------------------------------------------------------- 
    # My own remap PID routine (similar to remap_pids from energyflow)
    #---------------------------------------------------------------         
    def my_remap_pids(self,events, pid_i=3, error_on_unknown=True):
        # PDGid to small float dictionary
        PID2FLOAT_MAP = {0: 0.0, 22: 1.4,
                         211: .1, -211: .2,
                         321: .3, -321: .4,
                         130: .5,
                         2112: .6, -2112: .7,
                         2212: .8, -2212: .9,
                         11: 1.0, -11: 1.1,
                         13: 1.2, -13: 1.3}
        
        """Remaps PDG id numbers to small floats for use in a neural network.
        `events` are modified in place and nothing is returned.
    
        **Arguments**
    
        - **events** : _numpy.ndarray_
            - The events as an array of arrays of particles.
        - **pid_i** : _int_
            - The column index corresponding to pid information in an event.
        - **error_on_unknown** : _bool_
            - Controls whether a `KeyError` is raised if an unknown PDG ID is
            encountered. If `False`, unknown PDG IDs will map to zero.
        """
    
        if events.ndim == 3:
            pids = events[:,:,pid_i].astype(int).reshape((events.shape[0]*events.shape[1]))
            if error_on_unknown:
                events[:,:,pid_i] = np.asarray([PID2FLOAT_MAP[pid]
                                                for pid in pids]).reshape(events.shape[:2])
            else:
                events[:,:,pid_i] = np.asarray([PID2FLOAT_MAP.get(pid, 0)
                                                for pid in pids]).reshape(events.shape[:2])
        else:
            if error_on_unknown:
                for event in events:
                    event[:,pid_i] = np.asarray([PID2FLOAT_MAP[pid]
                                                 for pid in event[:,pid_i].astype(int)])
            else:
                for event in events:
                    event[:,pid_i] = np.asarray([PID2FLOAT_MAP.get(pid, 0)
                                                 for pid in event[:,pid_i].astype(int)])        

    #---------------------------------------------------------------
    # Plot NN metrics are a function of epochs
    #---------------------------------------------------------------
    def plot_NN_epochs(self, n_epochs, history, label, dim_label='', dim=None):
    
        epoch_list = range(1, n_epochs+1)
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        if 'acc' in history.history:
            acc = history.history['acc']
            val_acc = history.history['val_acc']
        else:
            acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']
        
        plt.axis([0, n_epochs, 0, 1])
        plt.xlabel('epochs', fontsize=16)
        plt.plot(epoch_list, loss, linewidth=2,
                 linestyle='solid', alpha=0.9, color=sns.xkcd_rgb['dark sky blue'],
                 label='loss')
        plt.plot(epoch_list, val_loss, linewidth=2,
                 linestyle='solid', alpha=0.9, color=sns.xkcd_rgb['faded purple'],
                 label='val_loss')
        plt.plot(epoch_list, acc, linewidth=2,
                 linestyle='dotted', alpha=0.9, color=sns.xkcd_rgb['watermelon'],
                 label='acc')
        plt.plot(epoch_list, val_acc, linewidth=2,
                 linestyle='dotted', alpha=0.9, color=sns.xkcd_rgb['medium green'],
                 label='val_acc')
        
        plt.legend(loc='best', fontsize=12)
        plt.tight_layout()
        if dim:
            plt.savefig(os.path.join(self.output_dir, f'DNN_epoch_{label}_{dim_label}{dim}.pdf'))
        else:
            plt.savefig(os.path.join(self.output_dir, f'PFN_epoch_{label}.pdf'))
        plt.close()

    #---------------------------------------------------------------
    # Plot confusion matrix
    # Note: not normalized to relative error
    #---------------------------------------------------------------
    def plot_confusion_matrix(self, y_train, y_predict_train, label):
    
        confusion_matrix = sklearn.metrics.confusion_matrix(y_train, y_predict_train)
        sns.heatmap(confusion_matrix)
        plt.savefig(os.path.join(self.output_dir, f'confusion_matrix_{label}.pdf'))
        plt.close()
        
    #---------------------------------------------------------------
    # Plot QA
    #---------------------------------------------------------------
    def plot_QA(self):
    
        for qa_observable in self.qa_observables:
            if qa_observable not in self.qa_results:
                continue
            qa_observable_shape = self.qa_results[qa_observable].shape
            if qa_observable_shape[0] == 0:
                continue
            
            if  self.y_total[:self.n_total].shape[0] != qa_observable_shape[0]:
                sys.exit(f'ERROR: {qa_observable}: {qa_observable_shape}, y shape: {self.y_total[self.n_total].shape}')
               
            q_indices = self.y_total[:self.n_total]
            g_indices = 1 - self.y_total[:self.n_total]
            result_q = self.qa_results[qa_observable][q_indices.astype(bool)]
            result_g = self.qa_results[qa_observable][g_indices.astype(bool)]

            # Set some labels
            if qa_observable == 'jet_theta_g':
                xlabel = rf'$\theta_{{\rm{{g}} }}$'
                ylabel = rf'$\frac{{1}}{{\sigma}} \frac{{d\sigma}}{{d\theta_{{\rm{{g}} }} }}$'
                bins = np.linspace(0, 1., 50)
            elif qa_observable == 'thrust':
                xlabel = rf'$\lambda_2$'
                ylabel = rf'$\frac{{1}}{{\sigma}} \frac{{d\sigma}}{{d\lambda_2 }} }}$'
                bins = np.linspace(0, 0.3, 50)
            elif qa_observable == 'zg':
                xlabel = rf'$z_{{\rm{{g}} }}$'
                ylabel = rf'$\frac{{1}}{{\sigma}} \frac{{d\sigma}}{{dz_{{\rm{{g}} }} }}$'
                bins = np.linspace(0.2, 0.5, 15)
            else:
                ylabel = ''
                xlabel = rf'{qa_observable}'
                bins = np.linspace(0, np.amax(result_g), 20)
            plt.xlabel(xlabel, fontsize=14)
            plt.ylabel(ylabel, fontsize=16)

            if qa_observable == 'jet_pt':
                stat='count'
            else:
                stat='density'

            # Construct dataframes for histplot
            df_q = pd.DataFrame(result_q, columns=[xlabel])
            df_g = pd.DataFrame(result_g, columns=[xlabel])
            
            # Add label columns to each df to differentiate them for plotting
            df_q['generator'] = np.repeat(self.q_label, result_q.shape[0])
            df_g['generator'] = np.repeat(self.g_label, result_g.shape[0])
            df = df_q.append(df_g, ignore_index=True)

            # Histplot
            h = sns.histplot(df, x=xlabel, hue='generator', stat=stat, bins=bins, element='step', common_norm=False)
            h.legend_.set_title(None)
            plt.setp(h.get_legend().get_texts(), fontsize='14') # for legend text

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{qa_observable}.pdf'))
            plt.close()

    #---------------------------------------------------------------
    # Plot q vs. g
    #---------------------------------------------------------------
    def plot_observable(self, X, y_train, xlabel='', ylabel='', filename='', xfontsize=12, yfontsize=16, logx=False, logy=False):

        q_indices = y_train
        g_indices = 1 - y_train

        observable_q = X[q_indices.astype(bool)]
        observable_g = X[g_indices.astype(bool)]

        df_q = pd.DataFrame(observable_q, columns=[xlabel])
        df_g = pd.DataFrame(observable_g, columns=[xlabel])

        df_q['generator'] = np.repeat(self.q_label, observable_q.shape[0])
        df_g['generator'] = np.repeat(self.g_label, observable_g.shape[0])
        df = df_q.append(df_g, ignore_index=True)

        bins = np.linspace(np.amin(X), np.amax(X), 50)
        stat='density'
        h = sns.histplot(df, x=xlabel, hue='generator', stat=stat, bins=bins, element='step', common_norm=False, log_scale=[False, logy])
        if h.legend_:
            #h.legend_.set_bbox_to_anchor((0.85, 0.85))
            h.legend_.set_title(None)
            plt.setp(h.get_legend().get_texts(), fontsize='14') # for legend text

        plt.xlabel(xlabel, fontsize=xfontsize)
        plt.ylabel(ylabel, fontsize=yfontsize)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{filename}'))
        plt.close()
            
##################################################################
if __name__ == '__main__':

    # Define arguments
    parser = argparse.ArgumentParser(description='Analyze qg')
    parser.add_argument('-c', '--configFile', action='store',
                        type=str, metavar='configFile',
                        default='config/qg.yaml',
                        help='Path of config file for analysis')
    parser.add_argument('-o', '--outputDir', action='store',
                        type=str, metavar='outputDir',
                        default='./TestOutput',
                        help='Output directory for output to be written to')

    # Parse the arguments
    args = parser.parse_args()

    print('Configuring...')
    print('configFile: \'{0}\''.format(args.configFile))
    print('ouputDir: \'{0}\"'.format(args.outputDir))

    # If invalid configFile is given, exit
    if not os.path.exists(args.configFile):
        print('File \"{0}\" does not exist! Exiting!'.format(args.configFile))
        sys.exit(0)

    analysis = AnalyzeQG(config_file=args.configFile, output_dir=args.outputDir)
    analysis.analyze_qg()