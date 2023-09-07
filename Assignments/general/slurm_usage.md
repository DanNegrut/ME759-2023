# Slurm Usage for ME759

### Ground Rules
* All programs that _you_ write need to be run using Slurm with a Slurm script. Do not run them on the head node (you are on the head node if you see `@euler` in your command prompt).
* Always submit your script to Slurm for running on a compute node with `sbatch myscript.sh`. Do **_not_** run `bash myscript.sh`. This will run your script directly on the head node.
* Include all _required_ flags listed below.
* Don't allocate resources that you aren't explicitly programming for. For example, don't ask for more than one CPU core if you aren't writing multithreaded code. This wastes resources that others could be using. This is particularly important with GPUs.
* Follow the appropriate sections below when making your Slurm script header.
* Compiling on the head node is fine. Running programs that you have written on the head node is not allowed and is grounds for account suspension (regardless of how inconvenient the timing is).
---
### Shebang
*This should always be the first line.*
```
#!/usr/bin/env bash
```
---
### Required Flags
* Select partition (section of the cluster)
```
#SBATCH -p wacc
```
---
### General Flags
* Time limit d-hh:mm:ss (larger will lower your queue priority)
```
#SBATCH -t 0-00:30:00
```
* Job name
```
#SBATCH -J MyJob
```
* Output and Error files
```
#SBATCH -o output-%j.out -e output-%j.err
```
---
### GPU Jobs
*Allocate a gpu and a single CPU core*
```
#SBATCH --gres=gpu:1 -c 1
```
---
### Multi-Core Jobs
*Allocate `n` CPU cores*
```
#SBATCH -c n
```
---
### Multi-Node Jobs
*Allocate `n` compute nodes with `m` CPU cores on each*
```
#SBATCH -N n -c m
```
---
### More Memory
*Allocate at least `n` MB of RAM (or GB with suffix G)*
```
#SBATCH --mem=n
```
_or_
```
#SBATCH --mem=nG
```

### Full Documentation
You should stick to the above configurations for these types of jobs, but for more inforation on the other features of Slurm see [here](https://slurm.schedmd.com/sbatch.html).
