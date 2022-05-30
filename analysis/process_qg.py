#!/usr/bin/env python3

"""
Class to read q-g data set, do jet finding, and compute subjet basis
"""

import os
import sys
import argparse
import yaml
import h5py
import time
from collections import defaultdict
import math

# Data analysis and plotting
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Fastjet via python (from external library heppy)
import fastjet as fj
import fjcontrib
import fjext

# Energy flow package
import energyflow

# Base class
sys.path.append('.')
from base import common_base

################################################################
class ProcessQG(common_base.CommonBase):

    #---------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------
    def __init__(self, config_file='', output_dir='', **kwargs):
        super(common_base.CommonBase, self).__init__(**kwargs)
       
        self.start_time = time.time()
        
        self.config_file = config_file
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Initialize config file
        self.initialize_config()

        # Initialize data structures to store results
        self.initialize_data_structures()

        # Load quark-gluon dataset
        self.load_qg_dataset()
     
        #print(self)
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
        self.n_total = config['n_total']
        self.n_val = config['n_val']
        self.n_test = config['n_test']
        self.event_index = 0
          
        # Nsubjettiness basis
        self.K = config['K_max']
        self.N_list = []
        self.beta_list = []
        for i in range(self.K-2):
            self.N_list += [i+1] * 3
            self.beta_list += [0.5,1,2]
        self.N_list += [self.K-1] * 2  
        self.beta_list += [1,2]

        # Subjet basis
        self.subjet_basis = config['subjet_basis']
        self.njet_list = config['njet']
        self.N_max_list= config['N_max']
        self.r_list = config['r']

        if type(self.njet_list) != list:
            print(f'ERROR: njet must be a list')
            print(f'Changing njet into a list')
            self.njet_list = list([self.njet_list])
        if type(self.N_max_list) != list:
            print(f'ERROR: N_max must be a list')
            print(f'Changing N_max into a list')
            self.N_max_list = list([self.N_max_list])

         if self.subjet_basis == 'exclusive':
            if self.r_list != [0.4]:
                    print('ERROR: Wrong subjet radius r. For exlusive basis we need r = 0.4')
                    print('Changing radius to r = 0.4')
                    self.r_list = [0.4]
                    time.sleep(2)
            if self.Clustering_Alg == 'antikt_algorithm':
                sys.exit(f'ERROR: For the exclusive algorithm we can not use antikt_algorithm (fastjet throws a warning)')

        # Clustering Algorithm 
        self.Clustering_Alg = config['Clustering_Alg']

        # Laman Construction
        self.Laman_construction = config['Laman_construction']


        # Load Herwig Dataset:
        self.Herwig_dataset = config['Herwig_dataset']    


    #---------------------------------------------------------------        
    # Initialize empty data structures to store results
    #---------------------------------------------------------------
    def initialize_data_structures(self):

        # Create two-layer nested defaultdict of lists to store jet observables
        self.output = defaultdict(lambda: defaultdict(list))

    #---------------------------------------------------------------
    # Load qg data set 
    #---------------------------------------------------------------
    def load_qg_dataset(self):
        
        # https://energyflow.network/docs/datasets/#quark-and-gluon-jets
        # X : a three-dimensional numpy array of jets:
        #     list of jets with list of particles for each jet, with (pt,y,phi,pid) values for each particle
        # y : a numpy array of quark/gluon jet labels (quark=1 and gluon=0).
        # The jets are padded with zero-particles in order to make a contiguous array.
        print()
        print('Loading qg dataset:')
        X, self.y = energyflow.datasets.qg_jets.load(num_data=self.n_total, pad=True, 
                                                     generator='pythia',  # Herwig is also available
                                                     with_bc=False        # Turn on to enable heavy quarks
                                                     )
        print('(n_jets, n_particles per jet, n_variables): {}'.format(X.shape))
        print()

        # Next, we will transform these into fastjet::PseudoJet objects.
        # This allows us to use the fastjet contrib to compute our custom basis (Nsubjettiness, subjets, etc).

        # Translate 3D numpy array (100,000 x 556 particles x 4 vars) into a dataframe
        # Define a unique index for each jet
        columns = ['pt', 'y', 'phi', 'pid']
        df_particles = pd.DataFrame(X.reshape(-1, 4), columns=columns)
        df_particles.index = np.repeat(np.arange(X.shape[0]), X.shape[1]) + 1
        df_particles.index.name = 'jet_id'
        
        # (i) Group the particle dataframe by jet id
        #     df_particles_grouped is a DataFrameGroupBy object with one particle dataframe per jet
        df_fjparticles_grouped = df_particles.groupby('jet_id')
        
        # (ii) Transform the DataFrameGroupBy object to a SeriesGroupBy of fastjet::PseudoJets
        # NOTE: for now we neglect the mass -- and assume y=eta
        # TO DO: Add y to https://github.com/matplo/heppy/blob/master/cpptools/src/fjext/fjtools.cxx
        # TO DO: Add mass vector using pdg
        print('Converting particle dataframe to fastjet::PseudoJets...')
        self.df_fjparticles = df_fjparticles_grouped.apply(self.get_fjparticles)
        print('Done.')
        print()

         # Load the Herwig Dataset for testing (For now only pfn and sub_pfn are supported)
        if self.Herwig_dataset == 'True':

            X_herwig, self.y_herwig = energyflow.datasets.qg_jets.load(num_data=self.n_val + self.n_test, pad=True, 
                                                     generator='herwig',  # Herwig is also available
                                                     with_bc=False        # Turn on to enable heavy quarks
                                                     )
          
            columns = ['pt', 'y', 'phi', 'pid']
            df_particles_herwig = pd.DataFrame(X_herwig.reshape(-1, 4), columns=columns)
            df_particles_herwig.index = np.repeat(np.arange(X_herwig.shape[0]), X_herwig.shape[1]) + 1
            df_particles_herwig.index.name = 'jet_id'

            df_fjparticles_herwig_grouped = df_particles_herwig.groupby('jet_id')

            self.df_fjparticles_herwig = df_fjparticles_herwig_grouped.apply(self.get_fjparticles)

    #---------------------------------------------------------------
    # Main processing function
    #---------------------------------------------------------------
    def process_qg(self):

        # Loop over events and do jet finding
        # Fill each of the jet_variables into a list
        fj.ClusterSequence.print_banner()
        print('Finding jets and computing N-subjettiness and subjets...')
        result = [self.analyze_event(fj_particles, 'pythia') for fj_particles in self.df_fjparticles]
        
        if self.Herwig_dataset == 'True':
           result = [self.analyze_event(fj_particles, 'herwig') for fj_particles in self.df_fjparticles_herwig]
        
        # Transform the dictionary of lists into a dictionary of numpy arrays
        self.output_numpy = {}
        for key,value in self.output.items():
            self.output_numpy[key] = self.transform_to_numpy(value)
        
        # Reformat output for ML algorithms (array with 1 array per jet which contain all N-subjettiness values)
        self.output_final = {}
        self.output_final['nsub'] = np.array([list(self.output_numpy['nsub'].values())])[0].T
        for key,val in self.output_numpy['subjet'].items():
            self.output_final[f'subjet_{key}'] = val
            print(key)


        # Write jet arrays to file
        with h5py.File(os.path.join(self.output_dir, 'subjets_unshuffled.h5'), 'w') as hf:
            print('-------------------------------------')

            # Write labels: gluon 0, quark 1
            hf.create_dataset(f'y', data=self.y)
            print(f'labels: {self.y.shape}')
            
            if self.Herwig_dataset == 'True':
                hf.create_dataset(f'y_herwig', data=self.y_herwig)


            # Write numpy arrays
            for key,val in self.output_final.items():
                hf.create_dataset(key, data=val)
                print(f'{key}: {val.shape}')

                # Check whether any training entries are empty
                [print(f'WARNING: input entry {i} is empty') for i,x in enumerate(val) if not x.any()]

            for qa_observable in self.output['qa']:
                hf.create_dataset(f'{qa_observable}', data=self.output_numpy['qa'][qa_observable])
                print(f'{qa_observable}')

            # Make some QA plots
            self.plot_QA()

            # Store some info on the output directory so that we don't need to remember the exact details of the config when we run the code at a later date

            hf.create_dataset('N_list', data=self.N_list)
            hf.create_dataset('beta_list', data=self.beta_list)
            hf.create_dataset('r_list', data=self.r_list)
            hf.create_dataset('N_max', data=self.N_max_list)
            hf.create_dataset('n_total', data = self.n_total)
            hf.create_dataset('Clustering_Alg', data = self.Clustering_Alg)
            hf.create_dataset('Laman_construction', data = self.Laman_construction)
            hf.create_dataset('N_clustering', data = self.N_cluster_list)   
            hf.create_dataset('Herwig_dataset', data = self.Herwig_dataset)        

            

    #---------------------------------------------------------------
    # Process an event
    #---------------------------------------------------------------
    def analyze_event(self, fj_particles, dataset_choice):
    
        # Check that the entries exist appropriately
        if fj_particles and type(fj_particles) != fj.vectorPJ:
            print('fj_particles type mismatch -- skipping event')
            return

        # Find jets -- one jet per "event".  We only use antikt for the Jet Clustering
        jet_def = fj.JetDefinition(fj.antikt_algorithm, fj.JetDefinition.max_allowable_R)

        cs = fj.ClusterSequence(fj_particles, jet_def)
        jet_selected = fj.sorted_by_pt(cs.inclusive_jets())[0]
        jet_pt = jet_selected.pt()
        jet_y = jet_selected.rap()
        if jet_pt < self.pt[0] or jet_pt > self.pt[1] or np.abs(jet_y) > self.y_max:
            sys.exit(f'ERROR: jet found with pt={jet_pt}, y={jet_y} outside of expected range.')

        # Compute jet quantities and store in our data structures
        self.analyze_jets(jet_selected, dataset_choice)

        self.event_index += 1
        if self.event_index%1000 == 0:
            print(f'event: {self.event_index}  --  {int(time.time() - self.start_time)}s')

    #---------------------------------------------------------------
    # Analyze jets of a given event.
    #---------------------------------------------------------------
    def analyze_jets(self, jet_selected, dataset_choice):

        self.fill_nsubjettiness(jet_selected)
        self.fill_subjets(jet_selected, dataset_choice)
        self.fill_qa(jet_selected)

    #---------------------------------------------------------------
    # Compute Nsubjettiness of jet
    #---------------------------------------------------------------
    def fill_nsubjettiness(self, jet):

        axis_definition = fjcontrib.KT_Axes()
        for i,N in enumerate(self.N_list):
            beta = self.beta_list[i]
        
            measure_definition = fjcontrib.UnnormalizedMeasure(beta)
            n_subjettiness_calculator = fjcontrib.Nsubjettiness(N, axis_definition, measure_definition)
            n_subjettiness = n_subjettiness_calculator.result(jet)/jet.pt()
            self.output['nsub'][f'N{N}_beta{beta}'].append(n_subjettiness)

    #---------------------------------------------------------------
    # Compute subjet kinematics...
    #---------------------------------------------------------------
    def fill_subjets(self, jet, dataset_choice):
        
        if self.subjet_basis == 'inclusive':
            self.N_cluster_list = self.N_max_list
        elif self.subjet_basis == 'exclusive':
            self.N_cluster_list = self.njet_list
        else:
            sys.exit(f'ERROR: Invalid choice for subjet_basis')

        for r in self.r_list:

            if self.Clustering_Alg == 'kt_algorithm':
                subjet_def = fj.JetDefinition(fj.kt_algorithm, r)
            elif self.Clustering_Alg == 'antikt_algorithm':
                subjet_def = fj.JetDefinition(fj.antikt_algorithm, r)
            elif self.Clustering_Alg == 'cambridge_algorithm':
                subjet_def = fj.JetDefinition(fj.cambridge_algorithm, r)
            else:
                sys.exit(f'ERROR: Wrong Clustering_Algorithm.')

            cs_subjet = fj.ClusterSequence(jet.constituents(), subjet_def)



            # Construct a Laman graph for each jet, and save the edges (node connections) and angles
            for N_cluster in self.N_cluster_list:

                edges_list = []
                angles_list = []
                subjet_phi_list=[]
                subjet_rap_list=[]
                z_list = []

                if self.subjet_basis == 'inclusive':
                    subjets = fj.sorted_by_pt(cs_subjet.inclusive_jets())
                elif self.subjet_basis == 'exclusive':
                    subjets = fj.sorted_by_pt(cs_subjet.exclusive_jets_up_to(N_cluster))

                for N in range(N_cluster): #the max number of N is N_max-1 because we start from 0
                    # First, fill the z values of the node + (η,φ) for the subjets
                    if N < len(subjets):
                        z = subjets[N].pt()/jet.pt()
                        z_list.append(z)
                        subjet_phi = subjets[N].phi()
                        subjet_rap = subjets[N].rap()
                        subjet_phi_list.append(subjet_phi)
                        subjet_rap_list.append(subjet_rap)
                    else:
                        z_list.append(0)
                        subjet_phi_list.append(0)
                        subjet_rap_list.append(0)

            

                    # Henneberg construction using Type 1 connections
                    # To start, let's just build based on pt ordering
                    # A simple construction is to have each node N connected to nodes N+1,N+2
                    # We will also zero-pad for now (edges denoted [-1,-1]) to keep fixed-size arrays

                    if self.Laman_construction == 'naive':
                        if N < N_cluster-1: 
                            if N < len(subjets)-1: 
                                angle = subjets[N].delta_R(subjets[N+1])
                                edges_list.append(np.array([N, N+1])) #in order to know to which pair the angle's list entry corresponds to 
                                angles_list.append(angle)
                            else:
                                edges_list.append(np.array([-1, -1]))
                                angles_list.append(0)

                        if N < N_cluster-2:
                            if N < len(subjets)-2:
                                angle = subjets[N].delta_R(subjets[N+2])
                                edges_list.append(np.array([N, N+2]))
                                angles_list.append(angle) 
                            else:
                                edges_list.append(np.array([-1, -1]))
                                angles_list.append(0)

                    elif self.Laman_construction == '1N':
                        if N == 0:
                            for i in range(N_cluster-1): # Because we want to start from i=1 
                                if i < len(subjets)-1:
                                    angle = subjets[0].delta_R(subjets[i+1])
                                    edges_list.append(np.array([0, i+1])) #in order to know to which pair the angle's list entry corresponds to 
                                    angles_list.append(angle)
                                else:
                                    edges_list.append(np.array([0, -1]))
                                    angles_list.append(0)
                               
                        elif N < N_cluster-1:
                            if N < len(subjets)-1:
                                angle = subjets[N].delta_R(subjets[N+1])
                                edges_list.append(np.array([N, N+1]))
                                angles_list.append(angle) 
                            else:
                                edges_list.append(np.array([-1, -1]))
                                angles_list.append(0)

                    elif self.Laman_construction == '1N2N':
                        if N == 0:
                            for i in range(len(subjets)-1): # Because we want to start from i=1
                                if i < len(subjets)-1:
                                    angle = subjets[0].delta_R(subjets[i+1])
                                    edges_list.append(np.array([0, i+1])) #in order to know to which pair the angle's list entry corresponds to 
                                    angles_list.append(angle)
                                else:
                                    edges_list.append(np.array([0, -1]))
                                    angles_list.append(0)
                        elif N == 1:
                            for i in range(len(subjets)-2): # Because we want to start from i=2
                                if i < len(subjets)-2:
                                    angle = subjets[1].delta_R(subjets[i+2])
                                    edges_list.append(np.array([1, i+2])) #in order to know to which pair the angle's list entry corresponds to 
                                    angles_list.append(angle)
                                else:
                                    edges_list.append(np.array([1, -1]))
                                    angles_list.append(0)         
                    else:
                        sys.exit(f'Wrong Laman Construction Algorithm.')



            if dataset_choice == 'pythia':
                    self.output[f'subjet'][f'r{r}_N{N_cluster}_edges'].append(np.array(edges_list))
                    self.output[f'subjet'][f'r{r}_N{N_cluster}_angles'].append(np.array(angles_list))
                    self.output[f'subjet'][f'r{r}_N{N_cluster}_z'].append(np.array(z_list))
                    self.output[f'subjet'][f'r{r}_N{N_cluster}_sub_phi'].append(np.array(subjet_phi_list))
                    self.output[f'subjet'][f'r{r}_N{N_cluster}_sub_rap'].append(np.array(subjet_rap_list))
                elif dataset_choice == 'herwig':
                    self.output[f'subjet'][f'herwig_r{r}_N{N_cluster}_edges'].append(np.array(edges_list))
                    self.output[f'subjet'][f'herwig_r{r}_N{N_cluster}_angles'].append(np.array(angles_list))
                    self.output[f'subjet'][f'herwig_r{r}_N{N_cluster}_z'].append(np.array(z_list))
                    self.output[f'subjet'][f'herwig_r{r}_N{N_cluster}_sub_phi'].append(np.array(subjet_phi_list))
                    self.output[f'subjet'][f'herwig_r{r}_N{N_cluster}_sub_rap'].append(np.array(subjet_rap_list))
                else: 
                    sys.exit(f'Error: Wrong dataset choice.')
                
        
        
    #---------------------------------------------------------------
    # Analyze jets of a given event.
    #---------------------------------------------------------------
    def fill_qa(self, jet):

        # Fill some jet QA
        self.output['qa']['jet_pt'].append(jet.pt())
        
        # angularity
        alpha = 1
        kappa = 1
        angularity = fjext.lambda_beta_kappa(jet, alpha, kappa, self.R)
        self.output['qa']['jet_angularity'].append(angularity)

        # thrust
        alpha = 2
        kappa = 1
        angularity = fjext.lambda_beta_kappa(jet, alpha, kappa, self.R)
        self.output['qa']['thrust'].append(angularity)

        # LHA
        alpha = 0.5
        kappa = 1
        angularity = fjext.lambda_beta_kappa(jet, alpha, kappa, self.R)
        self.output['qa']['LHA'].append(angularity)

        # pTD
        alpha = 0
        kappa = 2
        angularity = fjext.lambda_beta_kappa(jet, alpha, kappa, self.R)
        self.output['qa']['pTD'].append(angularity)
        
        # mass
        self.output['qa']['jet_mass'].append(jet.m())
        
        # theta_g
        beta = 0
        zcut = 0.2
        gshop = fjcontrib.GroomerShop(jet, self.R, fj.cambridge_algorithm)
        jet_groomed_lund = gshop.soft_drop(beta, zcut, self.R)
        theta_g = jet_groomed_lund.Delta() / self.R
        self.output['qa']['jet_theta_g'].append(theta_g)

        # zg
        zg = jet_groomed_lund.z()
        self.output['qa']['zg'].append(zg)
        
        # multiplicity
        n_constituents = len(jet.constituents())
        self.output['qa']['multiplicity_0000'].append(n_constituents)
        multiplicity_0150 = 0
        multiplicity_0500 = 0
        multiplicity_1000 = 0
        for constituent in jet.constituents():
            if constituent.pt() > 0.15:
                multiplicity_0150 += 1
            if constituent.pt() > 0.5:
                multiplicity_0500 += 1
            if constituent.pt() > 1.:
                multiplicity_1000 += 1
        self.output['qa']['multiplicity_0150'].append(multiplicity_0150)
        self.output['qa']['multiplicity_0500'].append(multiplicity_0500)
        self.output['qa']['multiplicity_1000'].append(multiplicity_1000)
        
    #---------------------------------------------------------------
    # Transform dictionary of lists into a dictionary of numpy arrays
    #---------------------------------------------------------------
    def transform_to_numpy(self, jet_variables_list):

        jet_variables_numpy = {}
        for key,val in jet_variables_list.items():
            jet_variables_numpy[key] = np.array(val)
        
        return jet_variables_numpy

    #---------------------------------------------------------------
    # Plot QA
    #---------------------------------------------------------------
    def plot_QA(self):
    
        for qa_observable in self.output_numpy['qa'].keys():
            
            qa_result = self.output_numpy['qa'][qa_observable]
            qa_observable_shape = qa_result.shape
            if qa_observable_shape[0] == 0:
                continue

            # Plot distributions
            plt.xlabel(rf'{qa_observable}', fontsize=14)
            max = np.amax(qa_result)*1.2
            bins = np.linspace(0, max, 20)
            plt.hist(qa_result,
                     bins,
                     histtype='step',
                     density=True,
                     label = 'q or g',
                     linewidth=2,
                     linestyle='-',
                     alpha=0.5)
            plt.legend(loc='best', fontsize=14, frameon=False)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{qa_observable}.pdf'))
            plt.close()   
                                    
    #---------------------------------------------------------------
    # Transform particles to fastjet::PseudoJets
    #---------------------------------------------------------------
    def get_fjparticles(self, df_particles_grouped):

        user_index_offset = 0
        return fjext.vectorize_pt_eta_phi(df_particles_grouped['pt'].values,
                                          df_particles_grouped['y'].values,
                                          df_particles_grouped['phi'].values,
                                          user_index_offset)

##################################################################
if __name__ == '__main__':

    # Define arguments
    parser = argparse.ArgumentParser(description='Process qg')
    parser.add_argument('-c', '--configFile', action='store',
                        type=str, metavar='configFile',
                        default='./config/qg.yaml',
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

    analysis = ProcessQG(config_file=args.configFile, output_dir=args.outputDir)
    analysis.process_qg()