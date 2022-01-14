# JFN

## Setup software environment -- on hiccup cluster

First, request an interactive node from the slurm batch system:
   ```
   srun -N 1 -n 20 -t 24:00:00 -p std --pty bash
   ``` 
   which requests 1 full node (20 cores) for 24 hours in the std partition. You can choose the time, queue, and number of cores to suite your needs. When you’re done with your session, just type `exit`.
Depending how busy the squeue is, you may get the node instantly, or you may have to wait awhile.
Please do not run anything bust the lightest tests on the login node.

Then initialize a python environment with the packages listed in Pipfile:
```
cd JFN
pipenv install                                # (this step is only needed the first time)
pipenv shell
```

Finally, load some modules that we will need
```
module use /software/users/james/heppy/modules
module load heppy/1.0
module list
```

Now we are ready to run our scripts.

Note that the setup above is designed to run on hiccup -- some modifications to the python setup might be needed if running elsewhere. 

## Quark-gluon analysis

There are two steps: compute subjet observables from the quark-gluon data set, and then fit the ML models.

These steps use a common config file at `JFN/config/qg.yaml`.

1. Compute Nsubjettiness arrays from input events, and write them to file, along with labels and four-vectors: 
   ```
   cd JFN
   python analysis/process_qg.py -c config/qg.yaml -o /rstorage/jfn/<your_name>/<your_run_id>
   ```
   where you should set the output directory to your liking (but keep it somewhere in `/rstorage/jfn`).

2. Fit model and make plots:
   ```
   cd JFN/analysis
   python analyze_qg.py -c <config> -o <output_dir>
   ```
   The `-o` path should point to the directory containing `subjets.h5` from Step 3. This is the location that the output plots will be written to. 

