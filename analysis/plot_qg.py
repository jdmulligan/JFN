#!/usr/bin/env python3

"""
Plot q vs. g classification performance
"""

import os
import sys
import argparse
import yaml
import re
import pickle

# Data analysis and plotting
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_context('paper', rc={'font.size':18,'axes.titlesize':18,'axes.labelsize':18})

# Base class
sys.path.append('.')
from base import common_base

################################################################
class PlotQG(common_base.CommonBase):

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

        # Suffix for plot outputfile names
        self.roc_plot_index = 0
        self.significance_plot_index = 0
        self.auc_plot_index = 0

        self.plot_title = False
                
    #---------------------------------------------------------------
    # Initialize config file into class members
    #---------------------------------------------------------------
    def initialize_config(self):
    
        # Read config file
        with open(self.config_file, 'r') as stream:
          config = yaml.safe_load(stream)

        self.models = config['models']
        self.K_list = config['K']
        self.r_list = config['r']
        
    #---------------------------------------------------------------
    # Main processing function
    #---------------------------------------------------------------
    def plot_qg(self):
    
        # Load ML results from file
        roc_filename = os.path.join(self.output_dir, f'ROC.pkl')
        with open(roc_filename, 'rb') as f:
            self.roc_curve_dict = pickle.load(f)
            self.AUC = pickle.load(f)

        # Plot models for a single setting
        self.plot_models()

    #---------------------------------------------------------------
    # Plot several versions of ROC curves and significance improvement
    #---------------------------------------------------------------
    def plot_models(self):

        if 'particle_gnn' in self.models:
            roc_list = {}
            roc_list['particle_gnn'] = self.roc_curve_dict['particle_gnn']
            self.plot_roc_curves(roc_list)

        if 'pfn' in self.models and 'efn' in self.models:
            roc_list = {}
            roc_list['PFN'] = self.roc_curve_dict['pfn']
            roc_list['EFN'] = self.roc_curve_dict['efn']
            self.plot_roc_curves(roc_list)

        if 'pfn' in self.models and 'efn' in self.models and 'nsub_dnn' in self.models:
            roc_list = {}
            roc_list['PFN'] = self.roc_curve_dict['pfn']
            roc_list['EFN'] = self.roc_curve_dict['efn']
            for K in self.K_list:
                roc_list[f'Nsub (M = {K}), DNN'] = self.roc_curve_dict['nsub_dnn'][K]
            self.plot_roc_curves(roc_list)

        if 'sub_dnn' in self.models and 'laman_dnn' in self.models and 'nsub_dnn' in self.models:
            roc_list = {}
            for r in self.r_list:
                roc_list[f'sub (r = {r}), DNN'] = self.roc_curve_dict['sub_dnn'][r]
                roc_list[f'laman (r = {r}), DNN'] = self.roc_curve_dict['laman_dnn'][r]
            for K in self.K_list:
                roc_list[f'Nsub (M = {K}), DNN'] = self.roc_curve_dict['nsub_dnn'][K]
            self.plot_roc_curves(roc_list)

        if 'nsub_linear' in self.models:
            roc_list = {}
            for K in self.K_list:
                roc_list[f'Nsub (M = {K}), Linear'] = self.roc_curve_dict['nsub_linear'][K]
            roc_list['theta_g'] = self.roc_curve_dict['jet_theta_g']
            self.plot_roc_curves(roc_list)

    #--------------------------------------------------------------- 
    # Plot ROC curves
    #--------------------------------------------------------------- 
    def plot_roc_curves(self, roc_list):
    
        plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
        plt.axis([0, 1, 0, 1])
        plt.title('q vs. g', fontsize=14)
        plt.xlabel('False q Rate', fontsize=16)
        plt.ylabel('True q Rate', fontsize=16)
        plt.grid(True)
    
        for label,value in roc_list.items():
            if label in ['PFN', 'EFN', 'jet_mass', 'jet_angularity', 'LHA', 'thrust', 'pTD', 'hadron_z', 'zg', 'jet_theta_g'] or 'multiplicity' in label:
                linewidth = 4
                alpha = 0.5
                linestyle = self.linestyle(label)
                color=self.color(label)
                legend_fontsize = 12
                if label == 'jet_mass':
                    label = r'$m_{\mathrm{jet}}$'
                if label == 'jet_angularity':
                    label = r'$\lambda_1$ (girth)'
                    linewidth = 2
                    alpha = 0.6
                if label == 'thrust':
                    label = r'$\lambda_2$ (thrust)'
                    linewidth = 2
                    alpha = 0.6
                if label == 'jet_theta_g':
                    label = r'$\theta_{\mathrm{g}}$'
                    linewidth = 2
                    alpha = 0.6
                if label == 'zg':
                    label = r'$z_{\mathrm{g}}$'
                    linewidth = 2
                    alpha = 0.6
                if label == 'PFN':
                    label = 'Particle Flow Network'
                if label == 'EFN':
                    label = 'Energy Flow Network'

            elif 'Nsub' in label or 'subjet' in label or 'sub' in label or 'laman' in label:
                linewidth = 2
                alpha = 0.9
                linestyle = self.linestyle(label)
                color=self.color(label)
                legend_fontsize = 12
            else:
                linewidth = 2
                linestyle = 'solid'
                alpha = 0.9
                color = sns.xkcd_rgb['almost black']
                legend_fontsize = 12
  
            FPR = value[0]
            TPR = value[1]
            plt.plot(FPR, TPR, linewidth=linewidth, label=label,
                     linestyle=linestyle, alpha=alpha, color=color)
                    
        plt.legend(loc='lower right', fontsize=legend_fontsize)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'ROC_{self.roc_plot_index}.pdf'))
        plt.close()

        self.roc_plot_index += 1
 
    #--------------------------------------------------------------- 
    # Plot Significance improvement
    #--------------------------------------------------------------- 
    def plot_significance_improvement(self, roc_list):
    
        plt.axis([0, 1, 0, 3])
        plt.title('q vs. g', fontsize=14)
        plt.xlabel('True q Rate', fontsize=16)
        plt.ylabel('Significance improvement', fontsize=16)
        plt.grid(True)
            
        for label,value in roc_list.items():
            if label in ['PFN', 'EFN', 'jet_mass', 'jet_angularity', 'LHA', 'thrust', 'pTD', 'hadron_z', 'zg', 'jet_theta_g'] or 'multiplicity' in label:
                linewidth = 4
                alpha = 0.5
                linestyle = self.linestyle(label)
                color=self.color(label)
                if label == 'jet_angularity':
                    label = r'$\lambda_1$ (girth)'
                    linewidth = 2
                    alpha = 0.6
                if label == 'PFN':
                    label = 'Particle Flow Network'
                if label == 'EFN':
                    label = 'Energy Flow Network'
            elif 'Nsub' in label or 'subjet' in label:
                linewidth = 2
                alpha = 0.9
                linestyle = 'solid'
                color=self.color(label)
            else:
                linewidth = 2
                linestyle = 'solid'
                alpha = 0.9
                color = sns.xkcd_rgb['almost black']
                
            FPR = value[0]
            TPR = value[1]
            plt.plot(TPR, TPR/np.sqrt(FPR+0.001), linewidth=linewidth, label=label,
                     linestyle=linestyle, alpha=alpha, color=color)
         
        plt.legend(loc='best', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'Significance_improvement_{self.significance_plot_index}.pdf'))
        plt.close()

        self.significance_plot_index += 1

    #---------------------------------------------------------------
    # Get color for a given label
    #---------------------------------------------------------------
    def color(self, label):

        color = None
        if label in ['PFN']:
            color = sns.xkcd_rgb['faded purple'] 
        elif label in ['EFN']:
            color = sns.xkcd_rgb['faded red']  
            #color = sns.xkcd_rgb['medium green'] 
        elif label in [f'sub (r = {self.r_list[1]}), DNN']:
            color = sns.xkcd_rgb['light lavendar']    
        elif label in [f'sub (r = {self.r_list[0]}), DNN']:
            color = sns.xkcd_rgb['dark sky blue']    
        elif label in [f'laman (r = {self.r_list[1]}), DNN']:
            color = sns.xkcd_rgb['light brown']  
        elif label in [rf'Nsub (M = {self.K_list[0]}), DNN', f'laman (r = {self.r_list[0]}), DNN']:
            color = sns.xkcd_rgb['watermelon'] 
        elif label in ['jet_theta_g']:
            color = sns.xkcd_rgb['medium brown']
        else:
            color = sns.xkcd_rgb['almost black']

        return color

    #---------------------------------------------------------------
    # Get linestyle for a given label
    #---------------------------------------------------------------
    def linestyle(self, label):
 
        linestyle = None
        if 'PFN' in label and 'min_pt' in label or 'Nsub' in label:
            linestyle = 'dotted'
        elif 'PFN' in label or 'EFN' in label or 'DNN' in label or 'pfn' in label:
            linestyle = 'solid'
        else:
            linestyle = 'dotted'

        return linestyle
            
##################################################################
if __name__ == '__main__':

    # Define arguments
    parser = argparse.ArgumentParser(description='Plot ROC curves')
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

    analysis = PlotQG(config_file=args.configFile, output_dir=args.outputDir)
    analysis.plot_qg()