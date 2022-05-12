# JFN

## Setup software environment – on hiccup cluster

### Logon and allocate a node
  
Logon to hiccup:
```
ssh <user>@hic.lbl.gov
```

First, request an interactive node from the slurm batch system:
   ```
   srun -N 1 -n 20 -t 2:00:00 -p quick --pty bash
   ``` 
   which requests 1 full node (20 cores) for 2 hours in the `quick` queue. You can choose the time and queue: you can use the `quick` partition for up to a 2 hour session, `std` for a 24 hour session, or `long` for a 72 hour session – but you will wait longer for the longer queues). 
Depending how busy the squeue is, you may get the node instantly, or you may have to wait awhile.
When you’re done with your session, just type `exit`.
Please do not run anything but the lightest tests on the login node. If you are finding that you have to wait a long time, let us know and we can take a node out of the slurm queue and logon to it directly.

### Initialize environment
  
Now we need to initialize the environment: load heppy, set the python version, and create a virtual environment for python packages.
Since various ML packages require higher python versions than installed system-wide, we have set up an initialization script to take care of this. 
The first time you set up, you can do:
```
cd JFN
./init.sh --install
```
  
On subsequent times, you don't need to pass the `install` flag:
```
cd JFN
./init.sh
```

Now we are ready to run our scripts.

## Quark-gluon analysis

There are two steps: compute subjet observables from the quark-gluon data set, and then fit the ML models.

These steps use a common config file at `JFN/config/qg.yaml`.

1. Compute subjet/Nsubjettiness arrays from input events, and write them to file, along with training labels: 
   ```
   cd JFN
   python analysis/process_qg.py -c config/qg.yaml -o /rstorage/jfn/<your_name>/<your_run_id>
   ```
   where you should set the output directory to your liking (but keep it somewhere in `/rstorage/jfn`).

2. Fit model and make plots:
   ```
   cd JFN
   python analysis/analyze_qg.py -c <config> -o <output_dir>
   ```
   The `-o` path should point to the directory containing `subjets_unshuffled.h5` from Step 1. This is the location that the output plots will be written to. 

3. You can then re-run the plotting script, if you like:
   ```
   cd JFN
   python analysis/plot_qg.py -c <config> -o <output_dir>
   ```

