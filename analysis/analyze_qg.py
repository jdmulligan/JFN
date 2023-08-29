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

# Pytorch
import torch
import torch_geometric
import networkx
#import GCN
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
import torch.nn.functional as F
from torch.nn import Linear

# sklearn
import sklearn
import sklearn.linear_model
import sklearn.ensemble
import sklearn.model_selection
import sklearn.pipeline

# Tensorflow and Keras
from tensorflow import keras
import keras_tuner

# Energy flow package
import energyflow
import energyflow.archs

# Data analysis and plotting
import pandas as pd
import numpy as np
import statistics
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_context('paper', rc={'font.size':18,'axes.titlesize':18,'axes.labelsize':18})

# Particle Net Model 
import ParticleNet

# Pytorch implementation of PFN 
from PFN_pytorch import ParticleFlowNetwork
# Base class
sys.path.append('.')
from base import common_base

##################################################################
    
class GCN_class(torch.nn.Module):
    def __init__(self, graph_batch, hidden_channels):
        super(GCN_class,self).__init__()
        self.conv1 = GCNConv(graph_batch.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 2)

    def forward(self, x, edge_index, batch):

        #x = F.dropout(x, p=0.5, training = self.training)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)

        x = global_mean_pool(x, batch)

        #x = F.dropout(x, p=0.5, training = self.training)
        x = self.lin(x)

        return x 

##################################################################
    
class GAT_class(torch.nn.Module):
    def __init__(self, graph_batch, hidden_channels, heads = 2, edge_dimension = 1):
        super(GAT_class,self).__init__()
        self.conv1 = GATConv(graph_batch.num_features, hidden_channels, heads)
        self.lin1 = Linear(hidden_channels*heads, hidden_channels*4)
        self.lin2 = Linear(hidden_channels*4, hidden_channels*8)
        self.conv2 = GATConv(hidden_channels*8, hidden_channels*2, heads)
        self.lin3 = Linear(hidden_channels*2*heads, hidden_channels*4) 
        self.lin4 = Linear(hidden_channels*4, hidden_channels*8)
        self.conv3 = GATConv(hidden_channels*8, hidden_channels*4, heads)
        #self.conv3 = GATConv(hidden_channels*heads, hidden_channels, heads, edge_dim = edge_dimension)
        self.lin_final1 = Linear(hidden_channels*4*heads, hidden_channels*4)
        self.lin_final2 = Linear(hidden_channels*4, hidden_channels*4)
        self.lin_final3 = Linear(hidden_channels*4, hidden_channels*2)
        self.lin_final4 = Linear(hidden_channels*2, 2)
        
    def forward(self, x, edge_index, batch):

        x = F.dropout(x, p=0.1, training = self.training)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.lin1(x)
        x = x.relu()
        x = self.lin2(x)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.lin3(x)
        x = x.relu()
        x = self.lin4(x)
        x = x.relu()
        #x = x.relu()
        x = self.conv3(x,edge_index)

        x = global_mean_pool(x, batch)

        #x = F.dropout(x, p=0.4, training = self.training)
        x = self.lin_final1(x)
        x = x.relu()
        x = self.lin_final2(x)
        x = x.relu()
        x = self.lin_final3(x)
        x = x.relu()
        x = self.lin_final4(x)
        # Do I need to do a softmax here? 
        

        return x 



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
            self.N_cluster_list = hf['N_clustering'][:]
        
        # We require njet and Nmax are a list 
        if type(self.njet_list) != list:
            print(f'ERROR: njet must be a list')
            print(f'Changing njet into a list')
            self.njet_list = list([self.njet_list])
        if type(self.N_max_list) != list:
            print(f'ERROR: N_max must be a list')
            print(f'Changing N_max into a list')
            self.N_max_list = list([self.N_max_list])

        # Based on the subjet basis choose the appropriate max number of subjets
        if self.subjet_basis == 'inclusive':
            self.N_cluster_list_config = self.N_max_list
        elif self.subjet_basis == 'exclusive':
            self.N_cluster_list_config = self.njet_list
        else:
            sys.exit(f'ERROR: Invalid choice for subjet_basis')

        
        # For 'exclusive' we need to make sure we don't lose information so we need r=0.4
        if self.subjet_basis == 'exclusive':
            if self.r_list != [self.R]:
                    print(f'ERROR: Wrong subjet radius r. For exlusive basis we need r = {self.R}')
                    print()
                    print(f'Changing radius to r = {self.R}')
                    self.r_list = [self.R]

        self.qa_observables = ['jet_pt', 'jet_angularity', 'thrust', 'LHA', 'pTD', 'jet_mass', 'jet_theta_g', 'zg', 'multiplicity_0000', 'multiplicity_0150', 'multiplicity_0500', 'multiplicity_1000']
            
        # Set torch device
        os.environ['TORCH'] = torch.__version__
        print()
        print(f'pytorch version: {torch.__version__}')
        self.torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', self.torch_device)
        if self.torch_device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
        print()

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
        self.laman_load = config['laman']
        self.Laman_construction = config['Laman_construction']

        # Fully connected gnn on hadrons
        self.fully_con_hadrons = config['fully_con_hadrons']

        # Load Herwig Dataset: Boolean variable
        self.Herwig_dataset = config['Herwig_dataset']
        
        # Subjet Basis
        self.r_list = config['r'] # This is not necessary since the r_list is read from the output dir in the __init__
        self.subjet_basis = config['subjet_basis']
        self.njet_list = config['njet']
        self.N_max_list= config['N_max']

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
            
            if model == 'sub_efn':
                self.model_settings[model]['Phi_sizes'] = tuple(config[model]['Phi_sizes'])
                self.model_settings[model]['F_sizes'] = tuple(config[model]['F_sizes'])
                self.model_settings[model]['epochs'] = config[model]['epochs']
                self.model_settings[model]['batch_size'] = config[model]['batch_size']
                self.model_settings[model]['learning_rate'] = config[model]['learning_rate']

            if 'gnn' in model:
                self.model_settings[model]['epochs'] = config[model]['epochs']
                self.model_settings[model]['batch_size'] = config[model]['batch_size']
            
            if model == "particle_net" :
                self.model_settings[model]['epochs'] = config[model]['epochs']
                self.model_settings[model]['batch_size'] = config[model]['batch_size']

    #---------------------------------------------------------------
    # Main processing function
    #---------------------------------------------------------------
    def analyze_qg(self):
    
        # Clear variables
        self.AUC, self.AUC_av, self.AUC_std = {}, {}, {}
        self.units = {} # To save the hidden layers' size that keras.tuner calculated
        self.epochs = {} # If we use early stopping we need to know how many epochs were run
        self.y = None
        self.X_Nsub = None

        # Read in dataset
        with h5py.File(os.path.join(self.output_dir, self.filename), 'r') as hf:

            self.y_total = hf[f'y'][:]
            X_Nsub_total = hf[f'nsub'][:] 
            #X_Nsub_total_herwig = hf[f'nsub_herwig'][:] 

            self.subjet_input_total={}
            for r in self.r_list:
                for N_cluster in self.N_cluster_list:
                    
                    if self.laman_load == 'True':
                        self.subjet_input_total[f'subjet_r{r}_N{N_cluster}_edges'] = hf[f'subjet_r{r}_N{N_cluster}_edges'][:]
                        self.subjet_input_total[f'subjet_r{r}_N{N_cluster}_angles'] = hf[f'subjet_r{r}_N{N_cluster}_angles'][:]
                    #print("here")
                    #print(hf[f'subjet_r{r}_N{N_cluster}_z'].shape)
                    N = N_cluster # Change this if you want to load N<N_cluster particles. e.g. N=2 and not all the subjets
                    self.subjet_input_total[f'subjet_r{r}_N{N_cluster}_z'] = hf[f'subjet_r{r}_N{N_cluster}_z'][:,:N]
                    self.subjet_input_total[f'subjet_r{r}_N{N_cluster}_sub_phi'] = hf[f'subjet_r{r}_N{N_cluster}_sub_phi'][:,:N]
                    self.subjet_input_total[f'subjet_r{r}_N{N_cluster}_sub_rap'] = hf[f'subjet_r{r}_N{N_cluster}_sub_rap'][:,:N]
                    #print(self.subjet_input_total[f'subjet_r{r}_N{N_cluster}_z'].shape)
                    #print(self.subjet_input_total[f'subjet_r{r}_N{N_cluster}_z'][0])
                    self.subjet_input_total[f'subjet_r{r}_N{N_cluster}_z'] = np.true_divide(self.subjet_input_total[f'subjet_r{r}_N{N_cluster}_z'], np.sum(self.subjet_input_total[f'subjet_r{r}_N{N_cluster}_z'], axis=1)[:,None])
                    #print(self.subjet_input_total[f'subjet_r{r}_N{N_cluster}_z'][0])
                    
                    
                    if self.Herwig_dataset == 'True':
                        self.y_herwig_total = hf[f'y_herwig'][:]
                        if self.laman_load == 'True':
                            self.subjet_input_total[f'subjet_herwig_r{r}_N{N_cluster}_angles'] = hf[f'subjet_herwig_r{r}_N{N_cluster}_angles'][:]
                        self.subjet_input_total[f'subjet_herwig_r{r}_N{N_cluster}_z'] = hf[f'subjet_herwig_r{r}_N{N_cluster}_z'][:]
                        self.subjet_input_total[f'subjet_herwig_r{r}_N{N_cluster}_sub_phi'] = hf[f'subjet_herwig_r{r}_N{N_cluster}_sub_phi'][:]
                        self.subjet_input_total[f'subjet_herwig_r{r}_N{N_cluster}_sub_rap'] = hf[f'subjet_herwig_r{r}_N{N_cluster}_sub_rap'][:]

           
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

            total_jets = int(y_balanced.size)
            total_jets_q = int(np.sum(y_balanced))
            total_jets_g = total_jets - total_jets_q
            print(f'Total number of jets available after balancing: {total_jets_q} (q), {total_jets_g} (g)')



            self.subjet_input_balanced={}
            for r in self.r_list:
                for N_cluster in self.N_cluster_list:
                    if self.laman_load == 'True':
                        self.subjet_input_balanced[f'subjet_r{r}_N{N_cluster}_edges'] = np.delete(self.subjet_input_total[f'subjet_r{r}_N{N_cluster}_edges'], indices_to_remove, axis=0) 
                        self.subjet_input_balanced[f'subjet_r{r}_N{N_cluster}_angles'] = np.delete(self.subjet_input_total[f'subjet_r{r}_N{N_cluster}_angles'], indices_to_remove, axis=0) 
                    self.subjet_input_balanced[f'subjet_r{r}_N{N_cluster}_z'] = np.delete(self.subjet_input_total[f'subjet_r{r}_N{N_cluster}_z'], indices_to_remove, axis=0) 
                    self.subjet_input_balanced[f'subjet_r{r}_N{N_cluster}_sub_phi'] = np.delete(self.subjet_input_total[f'subjet_r{r}_N{N_cluster}_sub_phi'], indices_to_remove, axis=0) 
                    self.subjet_input_balanced[f'subjet_r{r}_N{N_cluster}_sub_rap'] = np.delete(self.subjet_input_total[f'subjet_r{r}_N{N_cluster}_sub_rap'], indices_to_remove, axis=0) 



            # Shuffle dataset 
            idx = np.random.permutation(len(y_balanced))
            if y_balanced.shape[0] == idx.shape[0]:
                y_shuffled = y_balanced[idx]
                X_Nsub_shuffled = X_Nsub_balanced[idx]


                self.subjet_input_shuffled={}
                for r in self.r_list:
                    for N_cluster in self.N_cluster_list:
                        if self.laman_load == 'True':
                            self.subjet_input_shuffled[f'subjet_r{r}_N{N_cluster}_edges'] = self.subjet_input_balanced[f'subjet_r{r}_N{N_cluster}_edges'][idx]
                            self.subjet_input_shuffled[f'subjet_r{r}_N{N_cluster}_angles'] = self.subjet_input_balanced[f'subjet_r{r}_N{N_cluster}_angles'][idx]
                        self.subjet_input_shuffled[f'subjet_r{r}_N{N_cluster}_z'] = self.subjet_input_balanced[f'subjet_r{r}_N{N_cluster}_z'][idx]
                        self.subjet_input_shuffled[f'subjet_r{r}_N{N_cluster}_sub_phi'] = self.subjet_input_balanced[f'subjet_r{r}_N{N_cluster}_sub_phi'][idx]
                        self.subjet_input_shuffled[f'subjet_r{r}_N{N_cluster}_sub_rap'] = self.subjet_input_balanced[f'subjet_r{r}_N{N_cluster}_sub_rap'][idx]


            else:
                print(f'MISMATCH of shape: {y_balanced.shape} vs. {idx.shape}')

            # Truncate the input arrays to the requested size
            self.y = y_shuffled[:self.n_total]
            self.X_Nsub = X_Nsub_shuffled[:self.n_total]

            self.subjet_input={}
            for r in self.r_list:
                for N_cluster in self.N_cluster_list:
                    if self.laman_load == 'True':
                        self.subjet_input[f'subjet_r{r}_N{N_cluster}_edges'] = self.subjet_input_shuffled[f'subjet_r{r}_N{N_cluster}_edges'][:self.n_total]
                        self.subjet_input[f'subjet_r{r}_N{N_cluster}_angles'] = self.subjet_input_shuffled[f'subjet_r{r}_N{N_cluster}_angles'][:self.n_total]
                    self.subjet_input[f'subjet_r{r}_N{N_cluster}_z'] = self.subjet_input_shuffled[f'subjet_r{r}_N{N_cluster}_z'][:self.n_total]
                    self.subjet_input[f'subjet_r{r}_N{N_cluster}_sub_phi'] = self.subjet_input_shuffled[f'subjet_r{r}_N{N_cluster}_sub_phi'][:self.n_total]
                    self.subjet_input[f'subjet_r{r}_N{N_cluster}_sub_rap'] = self.subjet_input_shuffled[f'subjet_r{r}_N{N_cluster}_sub_rap'][:self.n_total]
  

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
        # We will split into validation sets (for tuning hyperparameters) separately for each model
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
            for N_cluster in self.N_cluster_list:
                
                # Split into Train and Test sets for Laman DNN
                if self.laman_load == 'True':
                    self.X_laman_train[f'r{r}_N{N_cluster}'],self.X_laman_test[f'r{r}_N{N_cluster}'],self.y_laman_train[f'r{r}_N{N_cluster}'],self.y_laman_test[f'r{r}_N{N_cluster}'] =  sklearn.model_selection.train_test_split(np.concatenate((self.subjet_input[f'subjet_r{r}_N{N_cluster}_z'], self.subjet_input[f'subjet_r{r}_N{N_cluster}_angles']),axis=1), self.y, test_size=self.test_frac)
            
                # Use as input to the Sub DNN (pt_1, η_1, φ_1,...,pt_n, η_n, φ_n) instead of (pt_1,...,pt_n, η_1,...η_n, φ_1, ... , φ_n)
                self.x_subjet_combined = []

                N = N_cluster # Change this if you want to load N<N_cluster particles. e.g. N=2 and not all the subjets
                for n in range(N):
                    self.x_subjet_combined.append(np.array(self.subjet_input[f'subjet_r{r}_N{N_cluster}_z'][:,n]))
                    self.x_subjet_combined.append(np.array(self.subjet_input[f'subjet_r{r}_N{N_cluster}_sub_phi'][:,n]))
                    self.x_subjet_combined.append(np.array(self.subjet_input[f'subjet_r{r}_N{N_cluster}_sub_rap'][:,n]))

                # Τhe shape of self.x_subjet_combined is (3*N_max, n_total) instead of (n_tot , 3*N_max)
                self.x_subjet_combined = np.array(self.x_subjet_combined).T
            
                # Split into Train and Test sets for Subjet DNN
                self.X_subjet_train[f'r{r}_N{N_cluster}'],self.X_subjet_test[f'r{r}_N{N_cluster}'],self.y_subjet_train[f'r{r}_N{N_cluster}'],self.y_subjet_test[f'r{r}_N{N_cluster}'] = sklearn.model_selection.train_test_split(self.x_subjet_combined, self.y, test_size=self.test_frac)


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
            self.roc_curve_dict[f'{model}_herwig'] = {}
                        
        # Plot the input data
        self.plot_QA()

        # Plot first few K (before and after scaling)
        if 'nsub_dnn' in self.models and K == 3:
            self.plot_nsubjettiness_distributions(K, self.training_data[K]['X_Nsub_train'], self.y_train, self.training_data[K]['feature_labels'], 'before_scaling')
            self.plot_nsubjettiness_distributions(K, sklearn.preprocessing.scale(self.training_data[K]['X_Nsub_train']), self.y_train, self.training_data[K]['feature_labels'], 'after_scaling')

        # Train models
        self.train_models()


        print(f'Dataset size : {self.n_total}')
        print(f'Herwig Dataset : {self.Herwig_dataset}')
        print(f'r_list : {self.r_list}')
        print(f'K_list : {self.K_list}')
        print(f'N max : {self.N_max_list}')
        print(f'N cluster : {self.N_cluster_list}')
        print(f'Inclusive or Exclusive: {self.subjet_basis }')
        print(f'Layers: {self.units}' )
        print(f'Epochs: {self.epochs}' )
        print(f'Clustering Algorithm : {self.Clustering_Alg}')
        print(f'Laman Construction : {self.Laman_construction}')
        print(f'AUC : {self.AUC}' )
        print(f'AUC_av : {self.AUC_av}' )
        print(f'AUC_std : {self.AUC_std}' )
        if 'laman_dnn' in self.models and 'sub_dnn' in self.models:
            self.diff = np.array(self.AUC['laman_dnn']) - np.array(self.AUC['sub_dnn']) 
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
            self.AUC[model], self.AUC_av[model], self.AUC_std[model] = [], [], []
        
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
            if model == 'pfn_pytorch':
                self.fit_pfn_pytorch(model, model_settings)

            # Particle Net 
            if model == 'particle_net':
                self.fit_particle_net(model, model_settings)
            # GNN
            if model == 'particle_gnn':
               self.fit_particle_gnn(model, model_settings)

            for r in self.r_list:
                for N_cluster in self.N_cluster_list:
                    # Laman Graphs DNN
                    if model == 'laman_dnn':
                        self.fit_laman_dnn(model, model_settings, r, N_cluster)
                    # Subjet DNN based on pt ordering 
                    if model == 'sub_dnn':
                        self.fit_subjet_dnn(model, model_settings, r, N_cluster)
                    #Subjet PFN 
                    if model == 'sub_pfn':
                        self.fit_sub_pfn(model, model_settings, r, N_cluster)
                    if model == 'sub_efn':
                        self.fit_sub_efn(model, model_settings, r, N_cluster)

        # Plot traditional observables
        for observable in self.qa_observables:
            self.roc_curve_dict[observable] = sklearn.metrics.roc_curve(self.y_total[:self.n_total], -self.qa_results[observable])

        # Save ROC curves to file
        if self.models:
            output_filename = os.path.join(self.output_dir, f'ROC.pkl')
            with open(output_filename, 'wb') as f:
                pickle.dump(self.roc_curve_dict, f)
                pickle.dump(self.AUC, f)

    #---------------------------------------------------------------
    # Fit Graph Neural Network with particle four-vectors
    #---------------------------------------------------------------
    def fit_particle_gnn(self, model, model_settings):
        print('fit_particle_gnn...')
        print()
        start_time = time.time()

        # load data
        #self.n_total = 50000 # Need to change this
        X, y = energyflow.qg_jets.load(self.n_total)

        # ignore pid information for now
        X = X[:,:,:3]

        # preprocess by centering jets and normalizing pts
        for x in X:
            mask = x[:,0] > 0
            yphi_avg = np.average(x[mask,1:3], weights=x[mask,0], axis=0)
            x[mask,1:3] -= yphi_avg
            x[mask,0] /= x[:,0].sum()

        # convert labels to categorical # Need this??
        #Y = to_categorical(y, num_classes=2)
        Y = y

        # Initialize list of graphs
        
        graph_list = {}
        graph_list_fullycon = []
        coo_laman = {}
        coo_laman_del = {}
        edge_indices_laman = {}
        edge_indices_laman_long = {} 
        graph_batch = {}
        node_features_laman = {}
        edge_attr_laman = {}
        edge_attr_laman_del = {}
        for r in self.r_list:
                for N in self.N_cluster_list:
                    graph_list[f'subjet_r{r}_N{N}'] = []
        
        
        # Loop over all jets
        for i, xp in enumerate(X):
            
            if self.fully_con_hadrons :
                # 1. Get node feature vector 
                #    First need to remove zero padding
                xp = xp[~np.all(xp == 0, axis=1)]
                # print info on xp 
                
                node_features = torch.tensor(xp,dtype=torch.float)

                # 2. Get adjacency matrix / edge indices
                
                # Fully connected graph 
                adj_matrix = np.ones((xp.shape[0],xp.shape[0])) - np.identity((xp.shape[0]))
                row, col = np.where(adj_matrix)
                
                # Use sparse COO format
                coo = np.array(list(zip(row,col)))

                #    Switch format
                edge_indices = torch.tensor(coo)
                edge_indices_long = edge_indices.t().to(torch.long).view(2, -1) #long .. ?!

                #    or can use this directly: edge_indices_full_conn = torch.tensor([row,col],dtype=torch.long) 

                # 3. Can add edge features later on ...

                # 4. Get graph label
                graph_label = torch.tensor(Y[i],dtype=torch.int64)

                # 5. Create PyG data object
                graph = torch_geometric.data.Data(x=node_features, edge_index=edge_indices_long, edge_attr=None, y=graph_label).to(self.torch_device)

                # 6. Add to list of graphs
                graph_list_fullycon.append(graph)

            # Laman Graph
            for r in self.r_list:
                break # TO DO: Remove this break
                for N in self.N_cluster_list:
                    coo_laman[f'subjet_r{r}_N{N}'] = self.subjet_input_total[f'subjet_r{r}_N{N}_edges'][i,:,:]
                    row, col = np.where(coo_laman[f'subjet_r{r}_N{N}'] ==-1)
                    
                    # Number of subjets 
                    if len(row)!=0:
                        n = int((row[0] + 3)/2)
                    else:  
                        n=N
                    
                    # Removing the zero-padding 
                    coo_laman_del[f'subjet_r{r}_N{N}'] = np.delete(coo_laman[f'subjet_r{r}_N{N}'],row,0)
                    node_features_laman[f'subjet_r{r}_N{N}'] = torch.tensor(self.subjet_input_total[f'subjet_r{r}_N{N}_z'][i,:n],dtype=torch.float)
                    node_features_laman[f'subjet_r{r}_N{N}'] = node_features_laman[f'subjet_r{r}_N{N}'].reshape(-1,1)
                    

                  #  if np.array(coo_laman_del[f'subjet_r{r}_N{N}']).shape != np.array(coo_laman[f'subjet_r{r}_N{N}']).shape:
                   #     print(i)

                    edge_indices_laman[f'subjet_r{r}_N{N}'] = torch.tensor(coo_laman_del[f'subjet_r{r}_N{N}'])
                    edge_indices_laman_long[f'subjet_r{r}_N{N}'] = edge_indices_laman[f'subjet_r{r}_N{N}'].t().to(torch.long).view(2, -1) #long .. ?!

                    edge_attr_laman[f'subjet_r{r}_N{N}'] = torch.tensor(self.subjet_input_total[f'subjet_r{r}_N{N}_angles'][i,:],dtype=torch.float).reshape(-1,1)  

                    edge_attr_laman_del[f'subjet_r{r}_N{N}'] = np.delete(edge_attr_laman[f'subjet_r{r}_N{N}'],row,0)


                    graph_label = torch.tensor(Y[i],dtype=torch.int64)

                    graph = torch_geometric.data.Data(x=node_features_laman[f'subjet_r{r}_N{N}'], edge_index=edge_indices_laman_long[f'subjet_r{r}_N{N}'], edge_attr=edge_attr_laman_del[f'subjet_r{r}_N{N}'], y=graph_label).to(self.torch_device) 


                    graph_list[f'subjet_r{r}_N{N}'].append(graph)
        # end of loop over jets 
        # Now we have a list of graphs for each subjet radius and number of subjets for both the fully connected and the Laman graphs
        # Run the fully connected graphs through the GNN classifier
        
        if self.fully_con_hadrons:
            # 7. Create PyG batch object that contains all the graphs and labels
            graph_batch = torch_geometric.data.Batch().from_data_list(graph_list_fullycon)

            # Print
            print()
            print(f"Training the fully connected graphs (hadrons)...")
            print(f'Number of graphs in PyG batch object: {graph_batch.num_graphs}')
            print(f'Graph batch structure: {graph_batch}') # It says "DataDataBatch" .. correct?

            # [Check the format that is required by the GNN classifier!!]
            # [Fully connected edge index, ok: N*(N-1) = 18 * 17 = 306 ]

            # Visualize one of the jet graphs as an example ...
            # Are the positions adjusted if we include edge features?
            vis = torch_geometric.utils.convert.to_networkx(graph_batch[3],to_undirected=True) #... undirected graph?
            plt.figure(1,figsize=(10,10))
            networkx.draw(vis,cmap=plt.get_cmap('Set2'),node_size=10,linewidths=6)
            plt.savefig(os.path.join(self.output_dir, 'jet_graph.pdf'))
            plt.close()

            # Check adjacency of the first jet
            print()
            print(f'adjacency of first jet: {graph_batch[0].edge_index}')
            print()

            # Check a few things .. 
            # 1. Graph batch
            print(f'Number of graphs: {graph_batch.num_graphs}') # correct number ...??
            #print(f'Number of features: {graph_batch.num_features}') # ok
            print(f'Number of node features: {graph_batch.num_node_features}') # ok
            #print(f'Number of classes: {graph_batch.num_classes}') # .. labels?? print(graph_batch.y)

            # 2. A particular graph
            print(f"For a particular graph: ")
            print(f'Number of nodes: {graph_batch[1].num_nodes}') # ok
            print(f'Number of edges: {graph_batch[1].num_edges}') # ok, this jet has 17 hadrons in it -> # edges = 17*16=272
            print(f'Has self-loops: {graph_batch[1].has_self_loops()}') # ok
            print(f'Is undirected: {graph_batch[1].is_undirected()}') # ok

            # Shuffle and split into training and test set
            #graph_batch = graph_batch.shuffle() # seems like I have the wrong format ...

            #self.n_train = 40000 # TO DO: erase
            train_dataset = graph_batch[:self.n_train] # ok ...
            test_dataset = graph_batch[self.n_train:]

            print(f'Number of training graphs: {len(train_dataset)}') # now ok, doesn't work for graph_batch ...?
            print(f'Number of test graphs: {len(test_dataset)}')

            # Group graphs into mini-batches for parallelization (..?)
            train_loader = torch_geometric.loader.DataLoader(train_dataset, batch_size=model_settings['batch_size'], shuffle=True)
            test_loader = torch_geometric.loader.DataLoader(test_dataset, batch_size=model_settings['batch_size'], shuffle=False)

            # Set up GNN structure
            # 1. Embed each node by performing multiple rounds of message passing
            # 2. Aggregate node embeddings into a unified graph embedding (readout layer)
            # 3. Train a final classifier on the graph embedding
            
            #gnn_model = GCN_class(graph_batch, hidden_channels = 64)  # TO DO: In order to train this model with the Laman w/ edge_attr we need to add a new train_gnn -> 
            gnn_model = GAT_class(graph_batch, hidden_channels = 64)
            
            gnn_model = gnn_model.to(self.torch_device)
            
            print(f"Running the fully connected GNN classifier on hadrons ...")
            print(f"torch.cuda.is_available = {torch.cuda.is_available()}")
            print(f"gnn_model: {gnn_model}")
            print(f'gnn_model is_cuda: {next(gnn_model.parameters()).is_cuda}')
            
            # test the feature space 
            #batch_size = 1
            #num_nodes = 4
            #in_features = 3
            
            #x = torch.randn(batch_size, num_nodes, in_features)  # Example feature matrix with `batch_size` samples
            
            # Fully connected graph 
            #adj_matrix = np.ones((num_nodes, num_nodes)) - np.identity((num_nodes))
            #row, col = np.where(adj_matrix)

            #    Switch format
            #edge_index = torch.tensor([row,col],dtype=torch.long)

            # Move the model and input data to CUDA if available
            #if torch.cuda.is_available():
             #   gnn_model = gnn_model.to('cuda')
              #  x = x.to('cuda')
               # edge_index = edge_index.to('cuda')

            # Forward pass to compute feature vectors at each layer
            #layer_outputs = []
            #x_original = x.clone()  # Keep a copy of the original input features

            # Store the output of each layer (including the input features as the first layer)
            #layer_outputs.append(x_original)
            #index = 0 
            #for layer in gnn_model.children():
             #   if index == 2:
              #      continue
               # print(f"layer: {layer}")
                #x = layer(x, edge_index)
                #layer_outputs.append(x.clone())
                #index += 1

            #for i, output in enumerate(layer_outputs):
             #   if i == 2:
              #      continue
               # print(f"Layer {i}:")
                #print(output)
                #print("---------")

            # Now train the fully connected GNN
            optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.0001)
            criterion = torch.nn.CrossEntropyLoss()

            for epoch in range(1, model_settings['epochs']): # -> 171
                self.train_gnn(train_loader, gnn_model, optimizer, criterion, False)
                train_acc = self.test_gnn(train_loader, gnn_model, False)
                test_acc = self.test_gnn(test_loader, gnn_model, False)
                print(f'Epoch: {epoch:02d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

            # Get AUC & ROC curve for the fully connected GNN
            for i, datatest in enumerate(test_loader):  # Iterate in batches over the test dataset.
                pred_graph  = gnn_model(datatest.x, datatest.edge_index, datatest.batch).data # '.data' removes the 'grad..' from the torch tensor
                pred_graph  = pred_graph.cpu().data.numpy() # Convert predictions to np.array. Not: values not in [0,1]
                label_graph = datatest.y.cpu().data.numpy() # Get labels

                if i==0:
                    pred_graphs = pred_graph
                    label_graphs = label_graph
                else:
                    pred_graphs = np.concatenate((pred_graphs,pred_graph),axis=0)
                    label_graphs = np.concatenate((label_graphs,label_graph),axis=0)

            # get AUC
            gnn_auc = sklearn.metrics.roc_auc_score(label_graphs, pred_graphs[:,1])
            print(f'Fully connected GNN AUC based on particle four-vectors is: {gnn_auc}')

            # get ROC curve for the fully connected GNN
            self.roc_curve_dict[model] = sklearn.metrics.roc_curve(label_graphs, pred_graphs[:,1])

        # Run the GNN for the Laman graphs
        for r in self.r_list:
            break # TO DO: Remove this break
            for N in self.N_cluster_list:
                    
                graph_batch[f'subjet_r{r}_N{N}'] = torch_geometric.data.Batch().from_data_list(graph_list[f'subjet_r{r}_N{N}'])
                graph_batch = graph_batch[f'subjet_r{r}_N{N}']
                print(f'graph_batch: {graph_batch.is_cuda}')

                # Print
                print(f'Number of graphs in PyG batch object: {graph_batch.num_graphs}')
                print(f'Graph batch structure: {graph_batch}') # It says "DataDataBatch" .. correct?
           
                # Visualize one of the jet graphs as an example ...

                vis = torch_geometric.utils.convert.to_networkx(graph_batch[3],to_undirected=True) #... undirected graph?
                plt.figure(1,figsize=(10,10))
                networkx.draw(vis,cmap=plt.get_cmap('Set2'),node_size=10,linewidths=6)
                plt.savefig(os.path.join(self.output_dir, 'jet_graph.pdf'))
                plt.close()

                # Check adjacency of the first jet
                print()
                print(f'adjacency of first jet: {graph_batch[0].edge_index}')
                print()

                # Check a few things .. 
                # 1. Graph batch
                print(f'Number of graphs: {graph_batch.num_graphs}') # ok
                print(f'Number of features: {graph_batch.num_features}') # ok
                print(f'Number of node features: {graph_batch.num_node_features}') # ok
                print(f'Number of edge features: {graph_batch.num_edge_features}') # ok

                #print(f'Number of classes: {graph_batch.num_classes}') # .. labels?? print(graph_batch.y)

                # Shuffle and split into training and test set
                #graph_batch = graph_batch.shuffle() # seems like I have the wrong format ...

                train_dataset = graph_batch[:self.n_train] # ok ...
                test_dataset = graph_batch[self.n_train:]

                print(f'Number of training graphs: {len(train_dataset)}') # now ok, doesn't work for graph_batch ...?
                print(f'Number of test graphs: {len(test_dataset)}')

                # Group graphs into mini-batches for parallelization (..?)
                train_loader = torch_geometric.loader.DataLoader(train_dataset, batch_size=model_settings['batch_size'], shuffle=True)
                test_loader = torch_geometric.loader.DataLoader(test_dataset, batch_size=model_settings['batch_size'], shuffle=False)

       
                # Set up GNN structure
                # 1. Embed each node by performing multiple rounds of message passing
                # 2. Aggregate node embeddings into a unified graph embedding (readout layer)
                # 3. Train a final classifier on the graph embedding
                gnn_model = GAT_class(graph_batch, hidden_channels = 8, heads = 8, edge_dimension = 1, edge_attributes = graph_batch.edge_attr)
                    
                gnn_model = gnn_model.to(self.torch_device)
            
                print(f"Running the Laman GNN for r={r} and N={N}")
                print(f"gnn_model: {gnn_model}")

                #print(f'gnn_model is_cuda: {gnn_model.is_cuda}')

                # Now train the GNN
                optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)
                criterion = torch.nn.CrossEntropyLoss()

                self.best_valid_acc = 0
                last_epochs = 0
                self.patience_gnn = 4
                for epoch in range(1, model_settings['epochs'] + 1): # -> 171
                    time_start = time.time()
                    loss_train = self.train_gnn(train_loader, gnn_model, optimizer, criterion, True)
                    auc_train, train_acc = self.test_gnn(train_loader, gnn_model, True)
                    auc_test,  test_acc = self.test_gnn(test_loader, gnn_model, True)
                    time_end = time.time()
                    print(f'Epoch: {epoch:02d}, Train Loss: {loss_train:.4f},  Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, AUC: {auc_test:4f}, Duration: {time_end-time_start}')

                    if test_acc > self.best_valid_acc:
                        last_epochs = 0
                        self.best_valid_acc = test_acc
                        # Save the best model
                        torch.save(gnn_model.state_dict(), 'best-model-parameters.pt')

                    if last_epochs >= self.patience_gnn:
                        print(f"Ending training after {epoch} epochs due to performance saturation with a patience parameter of {self.patience_gnn} epochs")
                        break
                    last_epochs += 1
                    
                # Use the best model
                gnn_model.load_state_dict(torch.load('best-model-parameters.pt'))

                # Get AUC & ROC curve
                for i, datatest in enumerate(test_loader):  # Iterate in batches over the test dataset.
                    pred_graph = gnn_model(datatest.x, datatest.edge_index, datatest.batch, datatest.edge_attr).data # '.data' removes the 'grad..' from the torch tensor
                    pred_graph = pred_graph.cpu().data.numpy() # Convert predictions to np.array. Not: values not in [0,1]
                    label_graph = datatest.y.cpu().data.numpy() # Get labels

                    if i==0:
                        pred_graphs = pred_graph
                        label_graphs = label_graph
                    else:
                        pred_graphs = np.concatenate((pred_graphs,pred_graph),axis=0)
                        label_graphs = np.concatenate((label_graphs,label_graph),axis=0)

                # get AUC
                gnn_auc = sklearn.metrics.roc_auc_score(label_graphs, pred_graphs[:,1])
                print()
                print(f'Laman GNN with r={r}, N={N} : AUC based on particle four-vectors is: {gnn_auc}')

                self.AUC[f'{model}'].append(gnn_auc)

        print(f'--- runtime: {time.time() - start_time} seconds ---')
        print()

    #---------------------------------------------------------------
    def train_gnn(self, train_loader, gnn_model, optimizer, criterion, edge_attr_boolean = False):
        gnn_model.train()

        loss_cum=0

        for data in train_loader:  # Iterate in batches over the training dataset.

            data = data.to(self.torch_device)
            if edge_attr_boolean:
                out = gnn_model(data.x, data.edge_index, data.batch, data.edge_attr)  # Perform a single forward pass.
            else: 
                out = gnn_model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)  # Compute the loss.
            loss_cum += loss.item() #Cumulative loss
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.

        return loss_cum/len(train_loader)

    #---------------------------------------------------------------
    def test_gnn(self, loader, gnn_model, edge_attr_boolean):
        gnn_model.eval()

        correct = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            data = data.to(self.torch_device)
            if edge_attr_boolean:
                out = gnn_model(data.x, data.edge_index, data.batch, data.edge_attr)  # Perform a single forward pass.
            else: 
                out = gnn_model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        return (auc, correct / len(loader.dataset) ) # Derive ratio of correct predictions.

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
    def fit_laman_dnn(self, model, model_settings, r, N):

        # Preprocessing: zero mean unit variance
        X_laman_train = sklearn.preprocessing.scale(self.X_laman_train[f'r{r}_N{N}'])
        X_laman_test = sklearn.preprocessing.scale(self.X_laman_test[f'r{r}_N{N}'])

        self.fit_dnn_subjet(X_laman_train, self.y_laman_train[f'r{r}_N{N}'], X_laman_test, self.y_laman_test[f'r{r}_N{N}'], model, model_settings, r, N) 
  

    #---------------------------------------------------------------
    # Fit Dense Neural Network for Subjets (not Deep Sets)
    #---------------------------------------------------------------
    def fit_subjet_dnn(self, model, model_settings, r, N):

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

            X_subjet_train = self.center_eta_phi(self.X_subjet_train[f'r{r}_N{N}'], shift_zero_pads=False)
            X_subjet_test = self.center_eta_phi(self.X_subjet_test[f'r{r}_N{N}'], shift_zero_pads=False)

        elif preprocessing_option == 'B':

            X_subjet_train = self.center_eta_phi(self.X_subjet_train[f'r{r}_N{N}'], shift_zero_pads=True)
            X_subjet_test = self.center_eta_phi(self.X_subjet_test[f'r{r}_N{N}'], shift_zero_pads=True)

            # TODO: should we also fix the orientation, e.g. rotate to align leading and subleading subjets?

            X_subjet_train = sklearn.preprocessing.scale(X_subjet_train)
            X_subjet_test = sklearn.preprocessing.scale(X_subjet_test)

        elif preprocessing_option == 'C':
            X_subjet_train = sklearn.preprocessing.scale(self.X_subjet_train[f'r{r}_N{N}'])
            X_subjet_test = sklearn.preprocessing.scale(self.X_subjet_test[f'r{r}_N{N}'])

        self.fit_dnn_subjet(X_subjet_train, self.y_subjet_train[f'r{r}_N{N}'], X_subjet_test, self.y_subjet_test[f'r{r}_N{N}'], model, model_settings,r) 

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
    def fit_dnn_subjet(self, X_train, Y_train, X_test, Y_test, model, model_settings, r, N):
        print()
        print(f'Training {model} with r = {r}...')

        tuner = keras_tuner.Hyperband(functools.partial(self.dnn_builder, input_shape=[X_train.shape[1]], model_settings=model_settings),
                                        objective='val_accuracy',
                                        max_epochs=3,
                                        factor=3,
                                        directory='keras_tuner',
                                        project_name=f'{model} with r = {r}, N = {N}')

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

        self.units[f'{model}, r = {r}, N = {N}'] = [units1, units2, units3]
        # Retrain the model with best number of epochs
        hypermodel = tuner.hypermodel.build(best_hps)

        # Early Stopping
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
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
        
        if self.subjet_basis=='exclusive':
            dim = N
        elif self.subjet_basis=='inclusive':
            dim = r 
        # Get & store ROC curve
        self.roc_curve_dict[model][dim] = sklearn.metrics.roc_curve(Y_test, preds_DNN)


    #---------------------------------------------------------------
    # Construct model for hyperparameter tuning with keras tuner
    #---------------------------------------------------------------
    def dnn_builder(self, hp, input_shape, model_settings):

        model = keras.models.Sequential()
        model.add(keras.layers.Flatten(input_shape=input_shape))

        # Tune size of first dense layer
        hp_units1 = hp.Int('units1', min_value=300, max_value=300, step=64)
        hp_units2 = hp.Int('units2', min_value=300, max_value=300, step=64)
        hp_units3 = hp.Int('units3', min_value=300, max_value=512, step=64)
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
    # Fit ML model -- Particle Net 
    #---------------------------------------------------------------
    def fit_particle_net(self, model, model_settings):
        print()
        print('fit_particle_net...')

        # Manually set the data size for now
        # TODO: Move this to the model settings
        n_total = 13000
        n_val = 1000   
        n_test = 1000

        # Load the four-vectors directly from the quark vs gluon data set
        X_PFN, Y_PFN = energyflow.datasets.qg_jets.load(num_data=n_total, pad=True, 
                                                        generator='pythia',  # Herwig is also available
                                                        with_bc=False        # Turn on to enable heavy quarks
                                                       )                     # X_PFN.shape = (n_jets, n_particles per jet, n_variables)  
               
        print(f'(n_jets, n_particles per jet, n_variables): {X_PFN.shape}')

        # Preprocess by centering jets and normalizing pts
        for x_PFN in X_PFN:
            mask = x_PFN[:,0] > 0
            yphi_avg = np.average(x_PFN[mask,1:3], weights=x_PFN[mask,0], axis=0)
            x_PFN[mask,1:3] -= yphi_avg
            x_PFN[mask,0] /= x_PFN[:,0].sum()

        X_PFN = X_PFN[:,:,:3]
        
        # Split data into train, val and test sets
        (X_PFN_train, X_PFN_val, X_PFN_test, Y_PFN_train, Y_PFN_val, Y_PFN_test) = energyflow.utils.data_split(X_PFN, Y_PFN,
                                                                                                               val=n_val, test=n_test)

        # Define the model 
        
        particlenet_model = ParticleNet.ParticleNet(input_dims = 3, num_classes = 2)
        particlenet_model = particlenet_model.to(self.torch_device)
        
        print(f"torch.cuda.is_available = {torch.cuda.is_available()}")
        print()
        print(f"particle_net model: {particlenet_model}")
        print()

        # Now train particle net
        learning_rate = 0.001
        #optimizer = torch.optim.Adam(particlenet_model.parameters(), lr = learning_rate)
        optimizer = torch.optim.SGD(particlenet_model.parameters(), lr = learning_rate, momentum = 0.8) # SGD seems to work better than Adam 
        criterion = torch.nn.CrossEntropyLoss()

        # For debugging in case of a gradient computation error
        torch.autograd.set_detect_anomaly(True)

        # Train the model with a specific batch size 
        batch_size = 128
        epochs = 10

        # How long it takes to train the model for one epoch
        time_start = time.time()
        for epoch in range(0, epochs): # -> 171
            print("--------------------------------")
            indices = torch.randperm(len(X_PFN_train))
            X_PFN_train_shuffled = X_PFN_train[indices]
            Y_PFN_train_shuffled = Y_PFN_train[indices]

            for batch_start in range(0, len(X_PFN_train), batch_size):
                # Get the current batch
                batch_end = min(batch_start + batch_size, len(X_PFN_train))
                inputs_batch = X_PFN_train_shuffled[batch_start:batch_end]
                labels_batch = Y_PFN_train_shuffled[batch_start:batch_end]
                self.train_particlenet(inputs_batch, labels_batch, particlenet_model, optimizer, criterion)
            
            auc_train, train_acc = self.test_particlenet(X_PFN_train, Y_PFN_train, particlenet_model)
            auc_test,  test_acc =  self.test_particlenet(X_PFN_test,  Y_PFN_test,  particlenet_model)
            print(f'Epoch: {epoch+1:02d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, AUC: {auc_test:.4f}')
            torch.cuda.empty_cache()

        time_end = time.time()
        print()
        print(f"Time to train model for 1 epoch = {(time_end - time_start)/epochs} seconds")
        # For debugging. Ignore for now 

        #for epoch in range(1, 150): # -> 171
        #    print("particlenet_model.state_dict() = ", particlenet_model.state_dict())
        #    model_weights_before_training = [param.data.clone() for param in particlenet_model.parameters()]
        #    initial_state_dict = particlenet_model.state_dict()
        #    self.train_particlenet(X_PFN_train, Y_PFN_train, particlenet_model, optimizer, criterion)
            
        #    print("--------------------------------")
        #    model_weights_after_training = [param.data.clone() for param in particlenet_model.parameters()]
        #    trained_state_dict = particlenet_model.state_dict()

            #for name, param in initial_state_dict.items():
            #    if torch.any(param != trained_state_dict[name]):
            #        print(f"Parameter {name} changed after training.")
            

        #    train_acc = self.test_particlenet(X_PFN_train, Y_PFN_train, particlenet_model)
        #    test_acc =  self.test_particlenet(X_PFN_test,  Y_PFN_test,  particlenet_model)
        #    print(f'Epoch: {epoch:02d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')


    #---------------------------------------------------------------
    def train_particlenet(self, X_PFN, Y_PFN, particlenet_model, optimizer, criterion):
        particlenet_model.train()

        X_PFN = torch.tensor(X_PFN, dtype=torch.float32).to(self.torch_device)

        points = X_PFN[:, :, 1:3]     # the eta-phi points of the particles to use as the points for the k-NN algorithm
        loss_cum = 0

        # The prediction is the output of the network\
        out = particlenet_model(points = points, features = X_PFN, mask = None) 

        Y_PFN = energyflow.utils.to_categorical(Y_PFN, num_classes=2)           # Convert to one-hot encoding to calculate the loss
        Y_PFN = torch.tensor(Y_PFN, dtype=torch.float32).to(self.torch_device)  # Convert to tensor and move to GPU
        
        loss = criterion(out, Y_PFN)  # Compute the loss.
        loss_cum += loss.item()       # Cumulative loss
        loss.backward()               # Derive gradients.
        optimizer.step()              # Update parameters based on gradients.
        optimizer.zero_grad()         # Clear gradients.

        return loss_cum/len(X_PFN)

    #---------------------------------------------------------------
    def test_particlenet(self, X_PFN, Y_PFN, particlenet_model):
        particlenet_model.eval()
        
        X_PFN = torch.tensor(X_PFN, dtype=torch.float32).to(self.torch_device)
        y_PFN = energyflow.utils.to_categorical(Y_PFN, num_classes=2) # Convert to one-hot encoding (y_PFN.shape = (njets, 2)) to calculate the AUC. This needs to remain as a numpy array in the cpu

        points = X_PFN[:, :, 1:3] 

        correct = 0
        out = particlenet_model(points = points, features = X_PFN, mask = None) 
        
        out_softmax = torch.nn.functional.softmax(out, dim=1).cpu().detach().numpy()  # Convert to numpy array in the cpu to calculate the AUC
        
        auc_particlenet = sklearn.metrics.roc_auc_score(y_PFN[:,1], out_softmax[:,1]) # Calculate the AUC
        
        Y_PFN = torch.tensor(Y_PFN, dtype=torch.float32).to(self.torch_device)        # Convert to tensor and move to GPU to compare with the output of the network for the accuracy 
        pred = out.argmax(dim=1)                                                      # Use the class with highest probability.
        correct += int((pred == Y_PFN).sum())                                         # Check against ground-truth labels.

        return (auc_particlenet, correct / len(Y_PFN))                                # Derive ratio of correct predictions.

    #---------------------------------------------------------------
    # Fit ML model -- PFN Pytorch Implementation 
    #---------------------------------------------------------------
    def fit_pfn_pytorch(self, model, model_settings):
        print()
        print('fit_pfn_pytorch...')

        # Manually set the data size for now
        # TODO: Move this to the model settings
        n_total = 40000
        n_val = 5000   
        n_test = 5000

        # Load the four-vectors directly from the quark vs gluon data set
        X_PFN, Y_PFN = energyflow.datasets.qg_jets.load(num_data=n_total, pad=True, 
                                                        generator='pythia',  # Herwig is also available
                                                        with_bc=False        # Turn on to enable heavy quarks
                                                       )                     # X_PFN.shape = (n_jets, n_particles per jet, n_variables)  
               
        print(f'(n_jets, n_particles per jet, n_variables): {X_PFN.shape}')

        # Preprocess by centering jets and normalizing pts
        for x_PFN in X_PFN:
            mask = x_PFN[:,0] > 0
            yphi_avg = np.average(x_PFN[mask,1:3], weights=x_PFN[mask,0], axis=0)
            x_PFN[mask,1:3] -= yphi_avg
            x_PFN[mask,0] /= x_PFN[:,0].sum()

        X_PFN = X_PFN[:,:,:3]
        
        # Split data into train, val and test sets
        (X_PFN_train, X_PFN_val, X_PFN_test, Y_PFN_train, Y_PFN_val, Y_PFN_test) = energyflow.utils.data_split(X_PFN, Y_PFN,
                                                                                                               val=n_val, test=n_test)

        # Define the model and move it to the GPU if available
        pfnpyt_model = ParticleFlowNetwork(input_dims = 3, num_classes = 2)
        pfnpyt_model = pfnpyt_model.to(self.torch_device)
        
        print(f"torch.cuda.is_available = {torch.cuda.is_available()}")
        print()
        print(f"particle_net model: {pfnpyt_model}")
        print()

        # Now train particle net
        learning_rate = 0.001
        optimizer = torch.optim.Adam(pfnpyt_model.parameters(), lr = learning_rate)
        #optimizer = torch.optim.SGD(pfnpyt_model.parameters(), lr = learning_rate, momentum = 0.8)
        criterion = torch.nn.CrossEntropyLoss()

        # For debugging in case of a gradient computation error
        torch.autograd.set_detect_anomaly(True)

        # Train the model with a batch size of 256
        batch_size = 256
        epochs = 10

        # How long it takes to train the model for one epoch
        time_start = time.time()

        for epoch in range(0, epochs): # -> 171
            print("--------------------------------")
            indices = torch.randperm(len(X_PFN_train))
            X_PFN_train_shuffled = X_PFN_train[indices]
            Y_PFN_train_shuffled = Y_PFN_train[indices]

            for batch_start in range(0, len(X_PFN_train), batch_size):
                # Get the current batch
                batch_end = min(batch_start + batch_size, len(X_PFN_train))
                inputs_batch = X_PFN_train_shuffled[batch_start:batch_end]
                labels_batch = Y_PFN_train_shuffled[batch_start:batch_end]
                self.train_pfnpyt(inputs_batch, labels_batch, pfnpyt_model, optimizer, criterion)
            
            auc_train, train_acc = self.test_pfnpyt(X_PFN_train, Y_PFN_train, pfnpyt_model)
            auc_test,  test_acc =  self.test_pfnpyt(X_PFN_test,  Y_PFN_test,  pfnpyt_model)
            print(f'Epoch: {epoch+1:02d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, AUC: {auc_test:.4f}')
            torch.cuda.empty_cache()

        time_end = time.time()
        print()
        print(f"Time to train model for 1 epoch = {(time_end - time_start)/epochs} seconds")

    #---------------------------------------------------------------
    def train_pfnpyt(self, X_PFN, Y_PFN, pfnpyt_model, optimizer, criterion):
        pfnpyt_model.train()

        X_PFN = torch.tensor(X_PFN, dtype=torch.float32).to(self.torch_device)

        loss_cum = 0

        # The prediction is the output of the network
        out = pfnpyt_model(features = X_PFN) 

        Y_PFN = energyflow.utils.to_categorical(Y_PFN, num_classes=2)           # Convert to one-hot encoding to calculate the loss
        Y_PFN = torch.tensor(Y_PFN, dtype=torch.float32).to(self.torch_device)  # Convert to tensor and move to GPU
        
        loss = criterion(out, Y_PFN)  # Compute the loss.
        loss_cum += loss.item()       # Cumulative loss
        loss.backward()               # Derive gradients.
        optimizer.step()              # Update parameters based on gradients.
        optimizer.zero_grad()         # Clear gradients.

        return loss_cum/len(X_PFN)

    #---------------------------------------------------------------
    def test_pfnpyt(self, X_PFN, Y_PFN, pfnpyt_model):
        pfnpyt_model.eval()
        
        X_PFN = torch.tensor(X_PFN, dtype=torch.float32).to(self.torch_device)
        y_PFN = energyflow.utils.to_categorical(Y_PFN, num_classes=2) # Convert to one-hot encoding (y_PFN.shape = (njets, 2)) to calculate the AUC. This needs to remain as a numpy array in the cpu

        correct = 0
        out = pfnpyt_model(features = X_PFN) 
        
        out_softmax = torch.nn.functional.softmax(out, dim=1).cpu().detach().numpy()  # Convert to numpy array in the cpu to calculate the AUC
        
        auc_particlenet = sklearn.metrics.roc_auc_score(y_PFN[:,1], out_softmax[:,1]) # Calculate the AUC
        
        Y_PFN = torch.tensor(Y_PFN, dtype=torch.float32).to(self.torch_device)        # Convert to tensor and move to GPU to compare with the output of the network for the accuracy 
        pred = out.argmax(dim=1)                                                      # Use the class with highest probability.
        correct += int((pred == Y_PFN).sum())                                         # Check against ground-truth labels.

        return (auc_particlenet, correct / len(Y_PFN))                                # Derive ratio of correct predictions.


    #---------------------------------------------------------------
    # Fit ML model -- Deep Set/Particle Flow Networks
    #---------------------------------------------------------------
    def fit_pfn(self, model, model_settings):
        print()
        print('fit_pfn...')
        start_time = time.time()
    
        # Load the four-vectors directly from the quark vs gluon data set
        # This is here just to compare with the ParticleNet results that have a maximum data size of 15k jets 
        self.n_total = 80000
        self.n_val = 10000
        self.n_test = 10000

        X_PFN, y_PFN = energyflow.datasets.qg_jets.load(num_data=self.n_total, pad=True, 
                                                     generator='pythia',  # Herwig is also available
                                                     with_bc=False        # Turn on to enable heavy quarks
                                                     )
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
        
        # Herwig
        if self.Herwig_dataset:

            X_PFN_herwig, y_PFN_herwig = energyflow.datasets.qg_jets.load(num_data=self.n_test + self.n_val, pad=True, 
                                                     generator='herwig',  # Herwig is also available
                                                     with_bc=False        # Turn on to enable heavy quarks
                                                     )
            Y_PFN_herwig = energyflow.utils.to_categorical(y_PFN_herwig, num_classes=2)

            for x_PFN in X_PFN_herwig:
                mask = x_PFN[:,0] > 0
                yphi_avg = np.average(x_PFN[mask,1:3], weights=x_PFN[mask,0], axis=0)
                x_PFN[mask,1:3] -= yphi_avg
                x_PFN[mask,0] /= x_PFN[:,0].sum()
            
            X_PFN_herwig = X_PFN_herwig[:,:,:3]
            
            (X_PFN_herwig_train, X_PFN_herwig_val, X_PFN_herwig_test, Y_PFN_herwig_train, Y_PFN_herwig_val, Y_PFN_herwig_test) = energyflow.utils.data_split(X_PFN_herwig, Y_PFN_herwig,
                                                                                             val=self.n_val, test=self.n_test)
        # Build architecture
        pfn = energyflow.archs.PFN(input_dim=X_PFN.shape[-1],
                                   Phi_sizes=model_settings['Phi_sizes'],
                                   F_sizes=model_settings['F_sizes'])
    
        # Early Stopping
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', patience=10, restore_best_weights=True)
        # Train model
        if not self.Herwig_dataset:
            # How long to train for one epoch
            start_time = time.time()
            history = pfn.fit(X_PFN_train,
                          Y_PFN_train,
                          epochs=model_settings['epochs'],
                          batch_size=model_settings['batch_size'],
                          validation_data=(X_PFN_val, Y_PFN_val),
                          verbose=1, callbacks =[early_stopping])
            end_time = time.time()
            history_epochs = len(history.history['loss']) # For the x axis of the plot    
            self.epochs[f'{model}']= history_epochs  

            # Plot metrics are a function of epochs
            self.plot_NN_epochs(history_epochs, history, model)
        
            # Get predictions on test data
            preds_PFN = pfn.predict(X_PFN_test, batch_size=1000)

            # Get AUC and ROC curve + make plot
            auc_PFN = sklearn.metrics.roc_auc_score(Y_PFN_test[:,1], preds_PFN[:,1])
            print('Particle Flow Networks/Deep Sets: AUC = {} (test set)'.format(auc_PFN))
            print(f"Time to train PFN for 1 epoch: {(end_time - start_time)/model_settings['epochs']} seconds")
            self.AUC[f'{model}'].append(auc_PFN)
        
            self.roc_curve_dict[model] = sklearn.metrics.roc_curve(Y_PFN_test[:,1], preds_PFN[:,1])

        elif self.Herwig_dataset:
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
        
            # Get predictions on Herwig test data
            preds_PFN = pfn.predict(X_PFN_herwig_test, batch_size=1000)

            # Get AUC and ROC curve + make plot
            auc_PFN = sklearn.metrics.roc_auc_score(Y_PFN_herwig_test[:,1], preds_PFN[:,1])
            print('Particle Flow Networks/Deep Sets: AUC = {} (test set Herwig)'.format(auc_PFN))
            self.AUC[f'{model}'].append(auc_PFN)
        
            self.roc_curve_dict[f'{model}_herwig'] = sklearn.metrics.roc_curve(Y_PFN_herwig_test[:,1], preds_PFN[:,1])


            # Get predictions on test data using PYTHIA
            preds_PFN = pfn.predict(X_PFN_test, batch_size=1000)

            # Get AUC and ROC curve + make plot
            auc_PFN = sklearn.metrics.roc_auc_score(Y_PFN_test[:,1], preds_PFN[:,1])
            print('Particle Flow Networks/Deep Sets: AUC = {} (test set Pythia)'.format(auc_PFN))
            self.AUC[f'{model}'].append(auc_PFN)
        
            self.roc_curve_dict[model] = sklearn.metrics.roc_curve(Y_PFN_test[:,1], preds_PFN[:,1])
        else:
            sys.exit(f'ERROR: Wrong Herwig_dataset choice')

        print(f'--- runtime: {time.time() - start_time} seconds ---')
        print()

    #---------------------------------------------------------------
    # Fit PFN for subjets
    #---------------------------------------------------------------
    def fit_sub_pfn(self, model, model_settings, r, N):
        print()
        print('fit_sub_pfn...')
        start_time = time.time()
    
        x_subjet_input_sub_pfn = []
        N_temp = N  # change this if you want to load N_temp < N_cluster particles
        for n in range(N_temp):
            x_subjet_input_sub_pfn.append(np.array(self.subjet_input_total[f'subjet_r{r}_N{N}_z'][:,n]))
            x_subjet_input_sub_pfn.append(np.array(self.subjet_input_total[f'subjet_r{r}_N{N}_sub_rap'][:,n]))
            x_subjet_input_sub_pfn.append(np.array(self.subjet_input_total[f'subjet_r{r}_N{N}_sub_phi'][:,n]))
        x_subjet_input_sub_pfn = np.array(x_subjet_input_sub_pfn).T
        x_subjet_input_sub_pfn = x_subjet_input_sub_pfn.reshape(self.n_total, N_temp, 3)  # To bring it to the form (n_total, n_particles, dof of each particle)

        # Preprocess by centering jets and normalizing pts
        for x_PFN in x_subjet_input_sub_pfn:
            mask = x_PFN[:,0] > 0
            yphi_avg = np.average(x_PFN[mask,1:3], weights=x_PFN[mask,0], axis=0)
            x_PFN[mask,1:3] -= yphi_avg
            x_PFN[mask,0] /= x_PFN[:,0].sum()

        self.y_total_sub_pfn = energyflow.utils.to_categorical(self.y_total, num_classes=2)

        
        # Herwig Dataset
        if self.Herwig_dataset == 'True':
            x_subjet_herwig_input_sub_pfn = []
            for n in range(N):
                x_subjet_herwig_input_sub_pfn.append(np.array(self.subjet_input_total[f'subjet_herwig_r{r}_N{N}_z'][:,n]))
                x_subjet_herwig_input_sub_pfn.append(np.array(self.subjet_input_total[f'subjet_herwig_r{r}_N{N}_sub_rap'][:,n]))
                x_subjet_herwig_input_sub_pfn.append(np.array(self.subjet_input_total[f'subjet_herwig_r{r}_N{N}_sub_phi'][:,n]))
            x_subjet_herwig_input_sub_pfn = np.array(x_subjet_herwig_input_sub_pfn).T
            x_subjet_herwig_input_sub_pfn = x_subjet_herwig_input_sub_pfn.reshape(self.n_test + self.n_val, N, 3)  # To bring it to the form (n_total, n_particles, dof of each particle)


            # Preprocess by centering jets and normalizing pts
            for x_PFN in x_subjet_herwig_input_sub_pfn:
                mask = x_PFN[:,0] > 0
                yphi_avg = np.average(x_PFN[mask,1:3], weights=x_PFN[mask,0], axis=0)
                x_PFN[mask,1:3] -= yphi_avg
                x_PFN[mask,0] /= x_PFN[:,0].sum()


            self.y_total_herwig_sub_pfn = energyflow.utils.to_categorical(self.y_herwig_total, num_classes=2)


            # Split data into train, val and test sets
            (X_PFN_herwig_train, X_PFN_herwig_val, X_PFN_herwig_test, Y_PFN_herwig_train, Y_PFN_herwig_val, Y_PFN_herwig_test) = energyflow.utils.data_split(x_subjet_herwig_input_sub_pfn, 
                                                                                            self.y_total_herwig_sub_pfn, val=self.n_val, test=self.n_test)


        auc_scores = []
        for i in range(4):

            # Split data into train, val and test sets
            (X_PFN_train, X_PFN_val, X_PFN_test, Y_PFN_train, Y_PFN_val, Y_PFN_test) = energyflow.utils.data_split(x_subjet_input_sub_pfn, self.y_total_sub_pfn,
                                                                                             val=int(self.n_val*self.n_total/self.n_total), test=int(self.n_test*self.n_total/self.n_total))
            # Build architecture
            opt = keras.optimizers.Adam(learning_rate=0.0003) # if error, change name to learning_rate
        
            pfn = energyflow.archs.PFN(input_dim=x_subjet_input_sub_pfn.shape[-1],
                                   Phi_sizes=model_settings['Phi_sizes'],
                                   F_sizes=model_settings['F_sizes'],
                                   optimizer=opt)


            # Early Stopping
            early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', patience=10, restore_best_weights=True)

            if self.Herwig_dataset == 'False':
                # Train model
                history = pfn.fit(X_PFN_train,
                                Y_PFN_train,
                                epochs=model_settings['epochs'],
                                batch_size=model_settings['batch_size'],
                                validation_data=(X_PFN_val, Y_PFN_val),
                                verbose=1, callbacks =[early_stopping])

                history_epochs = len(history.history['loss']) # For the x axis of the plot    
                self.epochs[f'{model}, r = {r}, N = {N}'] = history_epochs

                # Plot metrics are a function of epochs
                self.plot_NN_epochs(history_epochs, history, model)
            
                # Get predictions on test data
                preds_PFN = pfn.predict(X_PFN_test, batch_size=1000)

                # Get AUC and ROC curve + make plot
                auc_PFN = sklearn.metrics.roc_auc_score(Y_PFN_test[:,1], preds_PFN[:,1])
                print('Particle Flow Networks/Deep Sets: AUC = {} (test set)'.format(round(auc_PFN,4)))
                self.AUC[f'{model}'].append(auc_PFN)
                auc_scores.append(auc_PFN)
                if self.subjet_basis=='exclusive':
                    dim = N
                elif self.subjet_basis=='inclusive':
                    dim = r 

                self.roc_curve_dict[model][dim] = sklearn.metrics.roc_curve(Y_PFN_test[:,1], preds_PFN[:,1])

        self.AUC_av[f'{model}'].append(round(np.mean(auc_scores),4))
        self.AUC_std[f'{model}'].append(round(statistics.stdev(auc_scores),4))
        print()
        print(f'For r={r}, N={N}, AUC = {auc_scores}')
        print(f'For r={r}, N={N}, AUC = {np.mean(auc_scores)}, Error = {statistics.stdev(auc_scores)}')
        print()
        
        if self.Herwig_dataset == 'True':
            # Train model
            history = pfn.fit(X_PFN_train,
                            Y_PFN_train,
                            epochs=model_settings['epochs'],
                            batch_size=model_settings['batch_size'],
                            validation_data=(X_PFN_val, Y_PFN_val),
                            verbose=1, callbacks =[early_stopping])

            history_epochs = len(history.history['loss']) # For the x axis of the plot    
            self.epochs[f'{model}, r = {r}, N = {N}'] = history_epochs 

            # Plot metrics are a function of epochs
            self.plot_NN_epochs(history_epochs, history, model)
        
            # Get predictions on Herwig test data
            preds_PFN = pfn.predict(X_PFN_herwig_test, batch_size=1000)

            # Get AUC and ROC curve + make plot
            auc_PFN = sklearn.metrics.roc_auc_score(Y_PFN_herwig_test[:,1], preds_PFN[:,1])
            print('Particle Flow Networks/Deep Sets: AUC = {} (test set Herwig)'.format(auc_PFN))
            self.AUC[f'{model}'].append(auc_PFN)
        
            if self.subjet_basis=='exclusive':
                dim = N
            elif self.subjet_basis=='inclusive':
                dim = r 

            self.roc_curve_dict[f'{model}_herwig'][dim] = sklearn.metrics.roc_curve(Y_PFN_herwig_test[:,1], preds_PFN[:,1])

            # Get predictions on test data using PYTHIA

            preds_PFN = pfn.predict(X_PFN_test, batch_size=1000)

            # Get AUC and ROC curve + make plot
            auc_PFN = sklearn.metrics.roc_auc_score(Y_PFN_test[:,1], preds_PFN[:,1])
            print('Particle Flow Networks/Deep Sets: AUC = {} (test set Pythia)'.format(auc_PFN))
            self.AUC[f'{model}'].append(auc_PFN)

            self.roc_curve_dict[model][dim] = sklearn.metrics.roc_curve(Y_PFN_test[:,1], preds_PFN[:,1])

        print(f'--- runtime: {time.time() - start_time} seconds ---')
        print()

    #---------------------------------------------------------------
    # Fit ML model -- (IRC safe) Energy Flow Networks
    #---------------------------------------------------------------
    def fit_efn(self, model, model_settings):
    
        # Load the four-vectors directly from the quark vs gluon data set
        X_EFN, y_EFN = energyflow.datasets.qg_jets.load(num_data=self.n_total, pad=True, 
                                                     generator='pythia',  # Herwig is also available
                                                     with_bc=False        # Turn on to enable heavy quarks
                                                     )

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
        

        if self.Herwig_dataset == 'True':
            
            # Load the four-vectors directly from the quark vs gluon data set
            X_EFN_herwig, y_EFN_herwig = energyflow.datasets.qg_jets.load(num_data=self.n_val + self.n_test, pad=True, 
                                                     generator='herwig', 
                                                     with_bc=False        
                                                     )
            Y_EFN_herwig = energyflow.utils.to_categorical(y_EFN_herwig, num_classes=2)
       
            # Preprocess data set by centering jets and normalizing pts
            # Note: this step is somewhat different for pp/AA compared to the quark/gluon data set -- check
            for x_EFN in X_EFN_herwig:
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
            X_EFN_herwig = X_EFN_herwig[:,:,:3]
        
            # Make 800 four-vector array smaller, e.g. only 150. Ok w/o background
            X_EFN_herwig = X_EFN_herwig[:,:150]
        
       
            
            # Split data into train, val and test sets 
            # and separate momentum fraction z and angles (y,phi)
            (z_EFN_herwig_train, z_EFN_herwig_val, z_EFN_herwig_test, 
            p_EFN_herwig_train, p_EFN_herwig_val, p_EFN_herwig_test,
            Y_EFN_herwig_train, Y_EFN_herwig_val, Y_EFN_herwig_test) = energyflow.utils.data_split(X_EFN_herwig[:,:,0], X_EFN_herwig[:,:,1:], Y_EFN_herwig, 
                                                                           val=self.n_val, test=self.n_test)



        # Build architecture
        opt = keras.optimizers.Adam(learning_rate=model_settings['learning_rate']) # if error, change name to learning_rate
        efn = energyflow.archs.EFN(input_dim=2,
                                   Phi_sizes=model_settings['Phi_sizes'],
                                   F_sizes=model_settings['F_sizes'],
                                   optimizer=opt)
        
        # Early Stopping
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', patience=5, restore_best_weights=True)


        if self.Herwig_dataset == 'False':
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

        elif self.Herwig_dataset == 'True':
            # Train model
            history = efn.fit([z_EFN_train,p_EFN_train],
                            Y_EFN_train,
                            epochs=model_settings['epochs'],
                            batch_size=model_settings['batch_size'],
                            validation_data=([z_EFN_herwig_val,p_EFN_herwig_val], Y_EFN_herwig_val),
                            verbose=1, callbacks =[early_stopping])

            history_epochs = len(history.history['loss']) # For the x axis of the plot
            self.epochs[f'{model}'] = history_epochs

            # Plot metrics are a function of epochs
            self.plot_NN_epochs(history_epochs, history, model)
        
            # Get predictions on Herwig test data
            preds_EFN = efn.predict([z_EFN_herwig_test,p_EFN_herwig_test], batch_size=1000)     

            # Get AUC and ROC curve + make plot
            auc_EFN = sklearn.metrics.roc_auc_score(Y_EFN_herwig_test[:,1], preds_EFN[:,1])
            print('(IRC safe) Energy Flow Networks: AUC = {} (test set Herwig) '.format(auc_EFN))
            self.AUC[f'{model}'].append(auc_EFN)
        
            self.roc_curve_dict[model] = sklearn.metrics.roc_curve(Y_EFN_herwig_test[:,1], preds_EFN[:,1])

            # Get predictions on test data using PYTHIA
            preds_EFN = efn.predict([z_EFN_test,p_EFN_test], batch_size=1000)     

            # Get AUC and ROC curve + make plot
            auc_EFN = sklearn.metrics.roc_auc_score(Y_EFN_test[:,1], preds_EFN[:,1])
            print('(IRC safe) Energy Flow Networks: AUC = {} (test set Pythia)'.format(auc_EFN))
            self.AUC[f'{model}'].append(auc_EFN)
        
            self.roc_curve_dict[model] = sklearn.metrics.roc_curve(Y_EFN_test[:,1], preds_EFN[:,1])

        else:
            sys.exit(f'ERROR: Wrong Herwig Dataset choice')
        

    #---------------------------------------------------------------
    # Fit EFN for subjets
    #---------------------------------------------------------------
    def fit_sub_efn(self, model, model_settings, r, N):
        print()
        print('fit_sub_efn...')
        start_time = time.time()
        print('YES')
        x_subjet_input_sub_efn = []
        for n in range(N):
            x_subjet_input_sub_efn.append(np.array(self.subjet_input_total[f'subjet_r{r}_N{N}_z'][:,n]))
            x_subjet_input_sub_efn.append(np.array(self.subjet_input_total[f'subjet_r{r}_N{N}_sub_rap'][:,n]))
            x_subjet_input_sub_efn.append(np.array(self.subjet_input_total[f'subjet_r{r}_N{N}_sub_phi'][:,n]))
        x_subjet_input_sub_efn = np.array(x_subjet_input_sub_efn).T
        x_subjet_input_sub_efn = x_subjet_input_sub_efn.reshape(self.n_total, N, 3)  # To bring it to the form (n_total, n_particles, dof of each particle)

        self.y_total_sub_efn = energyflow.utils.to_categorical(self.y_total, num_classes=2)
        
        for x_EFN in x_subjet_input_sub_efn:
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
        x_subjet_input_sub_efn = x_subjet_input_sub_efn[:,:,:3]
        
        # Make 800 four-vector array smaller, e.g. only 150. Ok w/o background
        x_subjet_input_sub_efn = x_subjet_input_sub_efn[:,:150]
        
        # Check shape
        if self.y_total_sub_efn.shape[0] != x_subjet_input_sub_efn.shape[0]:
            print(f'Number of labels {self.y_total_sub_efn.shape} does not match number of jets {x_subjet_input_sub_efn.shape} ! ')
            
        # Split data into train, val and test sets 
        # and separate momentum fraction z and angles (y,phi)
        (z_EFN_train, z_EFN_val, z_EFN_test, 
         p_EFN_train, p_EFN_val, p_EFN_test,
         Y_EFN_train, Y_EFN_val, Y_EFN_test) = energyflow.utils.data_split(x_subjet_input_sub_efn[:,:,0], x_subjet_input_sub_efn[:,:,1:], self.y_total_sub_efn, 
                                                                           val=self.n_val, test=self.n_test)
        

        # Build architecture
        opt = keras.optimizers.Adam(learning_rate=model_settings['learning_rate']) # if error, change name to learning_rate
        efn = energyflow.archs.EFN(input_dim=2,
                                   Phi_sizes=model_settings['Phi_sizes'],
                                   F_sizes=model_settings['F_sizes'],
                                   optimizer=opt)
        
        # Early Stopping
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', patience=5, restore_best_weights=True)


        if self.Herwig_dataset == 'False':
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
                bins = np.linspace(0, np.amax(result_g), 60)
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
            df = pd.concat([df_q,df_g], ignore_index=True)

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
        df = pd.concat([df_q,df_g], ignore_index=True)

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



