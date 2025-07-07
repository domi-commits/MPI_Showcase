# MPI Monte Carlo PI Estimation Showcase

A short example for the usage of MPI. This app estimates PI by applying the Monte Carlo Simulation.

![Example](./example.gif)

## Build

```
mpic++ o mpi_monte monteCarlorPi_MPI.cpp -lncurses
```

## Execute

```
mpirun -np <number of processes> ./mpi_monte_pi
```
