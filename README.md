# JFN

## Setup software environment -- on hiccup cluster

Initialize a python environment with the packages listed in Pipfile. For example:
```
cd JFN
pipenv install                                # (this step is only needed the first time)
pipenv shell
```

The setup was designed to run on hiccup -- some modifications to the python setup might be needed if running elsewhere. 
Please note that in order to run the code on the hiccup cluster, all but the lightest code tests should use the slurm batch system. This means that every code you run should either:
1. Be called through `sbatch`, which will launch your jobs across the 8 hiccup nodes (see step 1 below for examples)
2. Be run on an interactive node requested via slurm (this should always be done for steps 2-3 below):
   ```
   srun -N 1 -n 20 -t 24:00:00 -p std --pty bash
   ``` 
   which requests 1 full node (20 cores) for 24 hours in the std partition. You can choose the time, queue, and number of cores to suite your needs. When you’re   done with your session, just type `exit`.

## Quark-gluon analysis

There are three steps: compute subjet observables from the quark-gluon data, aggregate those results over many files, and then fit the ML models.

These steps use a common config file at `JFN/config/qg.yaml`.

1. Compute Nsubjettiness arrays from input events, and write them to file, along with labels and four-vectors: 
   ```
   cd JFN/process/slurm
   sbatch slurm_compute_subjets.sh
   ```

   This writes output to `/rstorage/jfn/<job_id>/files.txt`
   
   Shuffle the file list: `shuf --output=files_randomized.txt files.txt`

2. Aggregate the results from each file's processing output
   ```
   cd JFN/process
   python aggregate_subjets.py -o /rstorage/jfn/<process_job_id>
   ```
   The `-o` path should point to the directory containing `files.txt` from Step 2. This is the location that the output file, `subjets.h5`, will be written to. 
   
3. Fit model and make plots:
   ```
   cd JFN/analysis
   python analyze_qg.py -c <config> -o <output_dir>
   ```
   The `-o` path should point to the directory containing `subjets.h5` from Step 3. This is the location that the output plots will be written to. 

