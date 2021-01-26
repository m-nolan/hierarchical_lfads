# load packages
import torch
import torch.nn as nn
import os
import wandb
import argparse

# set hyperparameters with argparse
parser = argparse.ArgumentParser()
parser.add_argument("-p","--project",type=str,default="lfads_ecog")
parser.add_argument("-h","--hparam_file",type=str,default=None)
parser.add_argument("-d","--data_file",type=str,default=os.curdir)
parser.add_argument("-o","--output_dir",type=str,default="/tmp/lfads_ecog/")


def main():
    # create wandb project link

    # get data
    
    # create model

    # create objective

    # create optimizer

    # create plotter/reporter

    # pass to training script

    # get performance metrics, save to table

if __name__ == "__main__":
    main()