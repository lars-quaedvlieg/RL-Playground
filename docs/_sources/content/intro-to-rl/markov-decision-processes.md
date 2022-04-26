# Markov Decision Processes

```{warning}
Please note that this notebook is still **work in progress**, hence some sections could behave unexpectedly, or provide incorrect information.
```

As said in chapter 1, fully observable environments in Reinforcement Learning have the Markov property. This means the environment can be represented by a \textbf{Markov Decision Process} (MDP). This means that the current state completely characterizes the process. MDP's are very important in RL, since they can represent almost every problem.\\

## From Markov Chains to Markov Decision Processes

```{figure} ../../images/intro-to-rl/markov-chain-example.png
:name: figure:markov-chain-example
:width: 14cm
Example of a Markov Chain
```