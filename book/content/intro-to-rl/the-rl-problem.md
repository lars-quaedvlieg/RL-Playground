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

- Why may the design of this reward function not be the best idea?

```{dropdown} Reveal answer
By punishing shipped electricity, there is a good chance that the agent will learn not to ship anything at all.
This would maximize it's cumulative reward.
```

- How would you improve the reward function?

```{dropdown} Reveal answer
There can be multiple correct answers, but one might define a reward function directly as the profit that has been
earned by the procedure. This should then be maximize to the best of the agent's abilities.

This was just a toy example, but the point here is that reward functions may have undesirable outcomes (that could, in 
some cases, have bad consequences). The definition of a good reward function may not be trivial in some situations.
```
````

% Explain MDP briefly before markov property (how does it relate to RL)

```{prf:definition}
:label: definiton:markov-property

A stochastic process $\{X_t\}_{t \in T}$, with values $X_0, X_1, \dotsc, X_T$ from $t = 0$ until $t = T$ is said to have 
the *Markov Property*, if and only if

$$
\mathbb{P}(X_{t+1} | X_t) = \mathbb{P}(X_{t+1} | X_0, ..., X_t)
$$

```

Let $S_t^a$ and $S_t^e$ be the state of the agent and the environment at any time $t$. If the environment is 
**fully-observable**, then $S_t^a = S_t^e$. Reinforcement learning environments can be seen as stochastic processes 
$\{S_t\}_{t \in T}$.

For an environment, having the Markov property means that, for all possible next states, the probability of obtaining 
that future state ($S_{t+1}$) solely depends on the current state ($S_t$), and not on any previous states 
($S_{t-1}, \dotsc, S_0$). This property describes that the current state is a sufficient statistic of the future, and 
history does not matter.

This will prove to be a key property in Reinforcement Learning, and it will be useful in many scenarios in the book.
But what happens when the markov property does not hold?

The class of **partially-observable** environments do not have the Markov property. Here, the agent indirectly observes 
the environment, meaning it may not have all information that is needed to know what happens next. Now, 
$S_t^a \neq S_t^e$. Since history is now important to predict the future, the agent must construct its own state 
representation $S_t^a$. For example:

- Complete history: $S_t^a = H_t = (O_0, O_1, \dotsc, O_t)$
- Beliefs of environment state: $S_t^a = (\mathbb{P}[S_t^e = s^{(1)}], \dotsc,\mathbb{P}[S_t^e = s^{(n)}])$
- Recurrent Neural Network: $S_t^a = \sigma(S_{t-1}^a W_s + O_t W_o))$

Here, $O_t$ is the observed state at time $t$. These problems are typically much more difficult to solve, since the 
Markov property cannot be exploited.

`````{prf:example}
:class: dropdown
:label: example:vase-non-markovian

Imagine we have the following process. There exists a dark vase with three balls. One is red, one is green, and one is 
blue. Every $t$, we take a ball out **without replacement**. Does this environment have the Markov property? Why or why 
not? 

````{dropdown} Reveal answer
This is a non-Markovian stochastic process. Recall that the markov property comes down to 

$$
\mathbb{P}(X_{t+1} | X_t) = \mathbb{P}(X_{t+1} | X_0, ..., X_t)
$$

Here, $X_t$ is the colour of the ball at time $t$. Imagine we have the order $X_0 = $ red, $X_1 = $ green, and $X_2 = $ 
blue.

Let's first think about $\mathbb{P}(X_2 | X_1)$. If we only know that $X_1$ is green, what do we know about $X_{2}$? 
We know the $X_2$ must be *either* blue or green.

However, $\mathbb{P}(X_{t+1} | X_0, ..., X_t)$ means we know that both $X_0$ is red and $X_1$ is green. Hence, $X_2$
*must* be blue.

We now see that here $\mathbb{P}(X_{t+1} | X_t) \neq \mathbb{P}(X_{t+1} | X_0, ..., X_t)$. Hence, the Markov property
does not hold for this stochastic process.

```{note}
This type of environment would actually be partially-observable, since we agent cannot directly observe the underlying 
state. Instead, we must maintain a probability distribution of different observations given the underlying state.
```
````
`````

````{prf:example}
:class: dropdown
:label: example:poker-partially-observable

Is the game of Poker a fully- or partially-observable environment?

```{dropdown} Reveal answer
Poker is partially-observable, since each player can only see their own cards. This means that we don't really know 
which state of the game we are in. Full observability would require knowledge about every player's cards.

A large part of the game is trying to estimate the quality of other players' cards. You can see partial observability
makes the game a lot more difficult to play. A good agent would thus need to create good estimates of the environment
states (for example by predicting other players' card qualities) to perform well.
```
````

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
$P_{ss'}^a = \mathbb{P}(S_{t+1} = s' | S_t = s, A_t = a)$ and $R_{s}^a = \mathbb{P}(R_{t+1} | S_t = s, A_t = a)$. 
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