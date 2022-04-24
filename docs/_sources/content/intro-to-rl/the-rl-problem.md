# The Reinforcement Learning Problem

```{warning}
Please note that this notebook is still **work in progress**, hence some sections could behave unexpectedly, or provide
incorrect information.
```

Reinforcement Learning (RL) is an area of Machine Learning concerned with deciding on a sequence of actions in an 
unknown environment in order to maximize cumulative reward.

To give an idea of this, imagine you are somewhere in a 2d-maze. At each point, you can either move to the left, right, up or down. The goal is to find your way out of the maze.
This corresponds to obtaining positive reward when completing the maze. Using Reinforcement Learning, you can figure out the optimal way to behave in this environment.

## The Reinforcement Learning Problem

What makes reinforcement learning different from other machine learning paradigms?
- There is no supervisor, only a **reward** signal
- Feedback is delayed, not instantaneous
- Time really matters (sequential, non i.i.d. data)
- Agent's actions affect the subsequent data it receives

```{prf:definition}
:label: definiton:reward-hypothesis

The *Reward Hypothesis* states that all imaginable goals can be described by the maximization of an expected cumulative
reward function.

```

Rewards function as scalar feedback signals. Reinforcement Learning is based on the **Reward Hypothesis**, which has 
been assumed to be true. Some problems can be difficult to solve, since actions can have long-term consequences and 
reward can be delayed.


````{prf:example}
:class: dropdown
:label: example:chess-reward-function

Imagine our goal is to train the best computer program at chess. What could a good reward function be defined as?

```{dropdown} Reveal answer
There could be multiple correct answers (where some definitions of the reward function may lead to solving the problem
more quickly than others). However, one example is giving a positive reward whenever the program wins a game. It will 
aim to maximize the cumulative reward, so over time it should get better at the game. 
```
````

````{prf:example}
:class: dropdown
:label: example:non-trivial-reward-function

Imagine you have built an agent that controls the way electricity is shipped to houses from a provider. Your goal is to 
save as much money as you can, by modifying the shipment procedure. As an AI engineer, you decide to define the reward 
function as punishing shipped electricity, hoping the agent would make the procedure more efficient by not shipping
unwanted electricity.

1. Why may the design of this reward function not be the best idea?

```{dropdown} Reveal answer
By punishing shipped electricity, there is a good chance that the agent will learn not to ship anything at all.
This would maximize it's cumulative reward.
```

2. How would you improve the reward function?

```{dropdown} Reveal answer
There can be multiple correct answers, but one might define a reward function directly as the profit that has been
earned by the procedure. This should then be maximize to the best of the agent's abilities.

This was just a toy example, but the point here is that reward functions may have undesirable outcomes (that could, in 
some cases, have bad consequences). The definition of a good reward function may not be trivial in some situations.
```
````

A state is **Markov**, if and only if it has the **Markov Property**, meaning $\mathbb{P}(S_t+1 | S_t) = 
\mathbb{P}(S_t+1 | S_1, ..., S_t)$. This means the probability of future states solely depends on the current state, and
not on any previous states. I.e. the history is a sufficient statistic of the future.

Let $S_t^a$ be the state of the agent at any time t and $S_t^e$ be the state of the environment on any time t. If the 
environment is **fully observable**, then $S_t^a = S_t^e$. This means that the Markov property holds, so formally it is
a Markov Decision Process.

However, when the environment is **partially observable**, the agent indirectly observes the environment. Now, 
$S_t^a \neq S_t^e$. Formally, this is called a partially observable Markov decision process (POMDP). The agent must 
construct it's own state representation $S_t^a$. For example:

- Complete history: $S_t^a = H_t$
- Beliefs of environment state: $S_t^a = (\mathbb{P}[S_t^e = s^1], ...,\mathbb{P}[S_t^e = s^n])$
- Recurrent Neural Network: $S_t^a = \sigma(S_{t-1}^a W_s + O_t W_o))$

## Components of a Reinforcement Learning Agent

An RL agent may include one or more of these components:
- **Policy**: agent's behaviour function
- **Value function**: how good is each state and/or action
- **Model**: agent's representation of the environment

A policy describes the agent's behavior. It maps states to actions. You can have deterministic ($a = \pi(s)$) and 
stochastic policies ($\pi(a | s) = \mathbb{P}(A_t = a | S_t = s)$). Often, $\pi$ is used to denote a policy.

A **value function** is a prediction of future reward of a given state. You can use it to determine if a state is good 
or bad. This means you can use it to select actions. It can be computed by $v_\pi(s) = \mathbb{E}_\pi(G_t | S_t = s)$, 
where $G_t$ is the **return** (or discounted cumulative reward). The return is defined as 
$G_t = R_1 + \gamma R_2 + \gamma^2 R_3 + ... = \sum_{i=t+1}^\infty\gamma^{i-t-1}R_i$ for some $\gamma \in [0, 1]$. 
This gamma is the **discount factor**, and it influences how much the future impacts return. This is useful, since 
it is not known if the representation of the environment is perfect. If it is not, it is not good to let the future 
influence the return as much as more local states. So, it is discounted.

Finally, a **model** predicts what the environment will do next. We let 
$P_{ss'}^a = \mathbb{P}(S_t+1 = s' | S_t = s, A_t = a)$ and $R_{s}^a = \mathbb{P}(R_t+1 | S_t = s, A_t = a)$. 
$P$ (**Transition model**) is the probability of transitioning to a next state given an action, while R is the 
reward when taking an action in some state.

```{list-table} Types of Reinforcement Learning agents
:header-rows: 1
:name: table:rl-agent-categories

* - Category
  - Properties
* - Value based
  - No Policy (implicit), Value function
* - Policy based
  - Policy, No Value function
* - Actor Critic
  - Policy, Value function
* - Model Free
  - No Model of the environment
* - Model based
  - Model of the environment
```

RL Agents can be categorized into the categories that are listed in {numref}`table:rl-agent-categories`. These can require 
different approaches that will be discussed throughout the book.

There are two fundamental problems in **sequential decision making**.
- Reinforcement Learning
  - The environment is initially unknown
  - The agent interacts with the environment
  - The agent improves its policy
- Planning (e.g. deliberation, reasoning, introspection, pondering, thought, search)
  - A model of the environment is known
  - The agent performs computations with its model (without any external interaction)
  - The agent improves its policy

It is important for an agent to make a trade-off between exploration and exploitation as well. Depending on the choice 
in this trade-off, agents will be more or less flexible and may or may not find better actions to perform.

- **Exploration** finds more information about the environment
- **Exploitation** exploits known information to maximize reward

Finally, it is possible to differentiate between prediction and control. **Prediction** is about evaluating the future 
given a certain policy, while **control** is about finding the best policy to optimize the future.