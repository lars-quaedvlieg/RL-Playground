# Value Function Approximation

```{warning}
Please note that this notebook is still **work in progress**, hence some sections could behave unexpectedly, or provide incorrect information.
```

Before this lecture, the value function was always defined per state. But what if the problems are really big? The game 
of Go for example has $10^{170}$ states. Some problems even have a *continuous* state space. How is it possible 
to scale up the previously discussed model-free methods for prediction and control to handle these situations?

The whole time, (action-)value functions were represented using a *lookup table*, where each state had a 
corresponding value. The problem with large MDP's is that it is too slow to learn the value for each state individually,
or even impossible to store all states or actions in memory.

The solution that will be proposed is to estimate the value function using **function approximation**. This means 
we use parameters $w$ estimate the value functions 
$\hat{v}(s, w) \approx v_\pi(s)$ or $\hat{q}(s, a, w) \approx q_\pi(s, a)$. There is a really nice benefit to this. 
After learning $w$ (using for example MC or TD learning), it will also be able to *generalize* to unseen states.

This chapter will focus on *differentiable* function approximators, since they can be updated using 
gradient-based optimization. Furthermore, the method must be able to handle *non-stationary* and 
*non-iid* data.

Two different methods of learning will be discussed, incremental methods and batch methods.

## Incremental Methods

This book assumes you are already familiar with the concept of **Gradient Descent**. If you are not, here is a 
quick explanation. If $J(w)$ is a differentiable function of parameter vector $w$. The idea is to update the parameters
in the direction of the gradient. We obtain $w = w - \alpha \nabla_w J(w)$ where $\alpha$ is a step-size parameter.

The goal of *Value-function approximation* using Stochastic Gradient Descent (SGD), is to find the parameter 
vector $w$ in order to minimize the mean-squared error between the approximate value $\hat{v}(s, w)$ and $v_\pi(s)$. 
The cost function to minimize is then $J(w) = \mathbb{E}_\pi \left[(v_\pi(S) - \hat{v}(S, w))^2\right]$.

Gradient descent converges to a *local* minimum. SGD will do this by sampling the gradient based on one single 
sample. The expected update is then equal to the full gradient update.

### Prediction algorithms

The state of the problem can be represented by a **feature vector** $x(S)$, where each element is a feature that 
described the state (preferably independently). The value function is equal to $\hat{v}(S, w) = x(S)^\intercal w$.

Table-lookup is actually a special case of linear function approximation. If a state is a one-hot-encoding of all states
, then the dot product with parameter vector $w$ is just $v(S_i) = w_i$.

Since it is impossible to use the true value function $v_\pi$ in the cost function, lets substitute in the targets 
instead. These are $G_t$ for MC and $G^\lambda_t$ for TD($\lambda$). The following are more detailed descriptions for 
these methods

- **MC**
	
    Return $G_t$ is an unbiased, noisy sample of $v_\pi(S_t)$. For that reason, it is possible to apply supervised 
    learning to the data $(S_1, G_1), (S_2, G_2), ..., (S_T, G_T)$. 
    $\Delta w = \alpha (G_t - \hat{v}(S_t, w)) \nabla_w \hat{v}(S_t, w) = \alpha (G_t - \hat{v}(S_t, w)) x(S_t)$. 
    Monte-carlo evaluation converges to a local optimum, even when using non-linear function approximation.
	
- **TD(0)**
		
    Return $R_{t+1} + \gamma \hat{v}(S_{t+1}, w)$ is a biased sample of $v_\pi(S_t)$. It is possible to apply supervised
    learning to the data $(S_1, R_2 + \gamma \hat{v}(S_2, w)), (S_2, R_3 + \gamma \hat{v}(S_3, w)), ..., (S_{T-1}, R_T)$
    . $\Delta w = \alpha (R_{t+1} + \gamma \hat{v}(S_{t+1}, w) - \hat{v}(S_t, w)) x(S_t)$. Linear TD(0) converges close 
    to the global optimum.
	
- **TD($\lambda$)**
	
    Return $G^\lambda_t$ is also a biased sample of $v_\pi(S_t)$. It is again possible to apply supervised learning to 
    the data $(S_1, G^\lambda_1), (S_2, G^\lambda_2), ..., (S_{T-1}, G^\lambda_{T-1})$. 
	
    - Forward view linear TD($\lambda$)
		
        $$
            \Delta w = \alpha (G^\lambda_t - \hat{v}(S_t, w)) x(S_t)
        $$
			
    - Backward view linear TD($\lambda$)
		
        ```{math}
                \delta_t & = R_{t+1} + \gamma \hat{v}(S_{t+1}, w) - \hat{v}(S_t, w)\\
                E_t 	 & = \gamma \lambda E_{t-1} + \hat{v}(S_t, w)\\
                \Delta w & = \alpha \delta_t E_t
        ```
		
        Here, the eligibility traces are defined for every parameter in the function approximator.
				
    The forward and backward view of linear TD($\lambda$) are again equivalent.

### Control algorithms

Control, just like prediction, preserves the same intuition from its tabular case. First, *approximate* policy 
evaluation ($\hat{q}(., ., w) \approx q_\pi$) will be performed, followed by an $\epsilon$-greedy policy improvement. 
The goal becomes to minimize the following cost function: 
$J(w) = \mathbb{E}_\pi \left[(q_\pi(S_t, A_t) - \hat{q}(S_t, A_t, w))^2\right]$. Then, 
$\Delta w = \alpha (q_\pi(S_t, A_t) - \hat{q}(S_t, A_t, w)) \nabla_w \hat{q}(S_t, A_t, w)$.

Now, the state and action are represented by a feature vector $x(S_t, A_t)$. The action-value approximation becomes 
$\hat{q}(S_t, A_t, w) = x(S, A)^\intercal w$. In this case, $\nabla_w \hat{q}(S_t, A_t, w) = x(S_t, A_t)$.

All algorithms work the exact same way as in the previous section about prediction. All that is different is the swap of
$v$ with $q$. The eligibility traces of backwards TD($\lambda$) are still defined for all parameters.

### Algorithm convergence

By using function approximators, the algorithms might not always converge. In some cases they diverge, but in other 
cases they also chatter around the near-optimal value function. In the case of control, this is because you are not sure
if each step is actually improving the policy anymore. The following tables describe the convergence properties of 
prediction and control algorithms.

```{list-table} Guarantee of convergence properties of Prediction algorithms
:header-rows: 1

* - On/Off-Policy 
  - Algorithm
  - Table Lookup
  - Linear
  - Non-Linear
* - On-Policy 
  - MC
  - Yes
  - Yes
  - Yes
* - On-Policy
  - TD 
  - Yes
  - Yes
  - No
* - On-Policy
  - Gradient TD
  - Yes
  - Yes
  - Yes
* - Off-Policy
  - MC
  - Yes
  - Yes
  - Yes
* - Off-Policy
  - MC
  - Yes
  - No
  - No
* - Off-Policy
  - Gradient TD
  - Yes
  - Yes
  - Yes 
```

As seen before, TD does not follow the gradient of any objective function. For this reason, TD might diverge when 
off-policy or using non-linear function approximation. **Gradient TD** is an algorithm that exists, which follows 
the true gradient of the projected Bellman error.

```{list-table} Guarantee of convergence properties of Control algorithms
:header-rows: 1

* - Algorithm
  - Table Lookup
  - Linear
  - Non-Linear
* - MC Control
  - Yes
  - Chatter
  - No
* - SARSA
  - Yes
  - Chatter
  - No
* - Q-Learning
  - Yes
  - No
  - No
* - Gradient Q-Learning
  - Yes 
  - Yes 
  - No
```

Control algorithms are even worse than prediction algorithms. The next section in this chapter will discuss how to 
address these issues when using Neural Networks.

## Batch Methods

The problem using gradient descent is that it is not sample efficient. It just experiences something once, and then 
removes it and moves on the next experience. This means you don't make maximum use of the data to update the model.
**Batch methods** seek to find the best fitting value function given the agent's experience.

### Least-Squares Prediction

Given $\hat{v}(s, w) \approx v_\pi(s)$ and experience $D = \{(s_1, v_1^\pi), (s_2, v_2^\pi), ..., (s_T, v_T^\pi)\}$, 
the aim is to find which parameters $w$ give the best fitting value $\hat{v}(s, w)$. **Least-Squares** algorithms 
aim to find parameter vector $w$ that minimizes the squared error between $\hat{v}(s, w)$ and $v_t^\pi$. 
$LS(w) = \sum^T_{t = 1} (v_t^\pi - \hat{v}(s_t, w))^2 = \mathbb{E}_D \left[(v^\pi - \hat{v}(s_t, w))^2\right]$.

This can be done by using **Experience Replay**. This method saves dataset $D$ of experience, and repeatedly 
samples $(s, v^\pi) \sim D$. Then, it applies SGD to the network 
($\Delta w = \alpha (v^\pi - \hat{v}(s, w)) \nabla_w \hat{v}(s, w))$. Applying this algorithm will find 
$w^\pi = \arg\min_w LS(w)$.

```{note}
TO SELF, REMOVE OR MOVE THIS?
```

**Deep Q-Networks** use experience replay together with **fixed Q-targets**. The algorithm consists of the 
following steps

- Take an action $a_t$ according to some $\epsilon$-greedy policy
- Store transition $(s_t, a_t, r_{t+1}, s_{t+1})$ in replay memory $D$
- Sample a random mini-batch of size $n$ of transitions $(s, a, r, s') \sim D$
- Compute Q-learning targets with respect to old, fixed parameters $w^-$ (these are the fixed Q-targets)
- optimize MSE between Q-network and Q-learning targets using some variant of SGD
	
	$$
		LS_i(w_i) = \mathbb{E}_{s, a, r, s' \sim D_i} \left[\left(r + \gamma \max_{a'} Q(s', a'; w_i^-) - Q(s, a; w_i)\right)^2\right]
	$$

*Experience replay* de-correlates the sequence in actions by randomly sampling actions at every step, leading to 
a better convergence. The *fixed Q-targets* are calculated from $w_i^-$, which are some old parameters of the 
network (they are frozen for a defined number of steps before being updated). This is used, because that way we don't 
bootstrap using our current network. If you do that, it could end up being unstable.

### Linear Least-Squares Prediction

When using a linear value function approximation, it is possible to solve the cost function directly. This might be 
more efficient, since experience replay might take many iterations to find the optimal parameters. It is fairly simple.

```{math}
		\mathbb{E}_D \left[\Delta w\right] & = 0\\
		\alpha \sum_{t = 1}^T x(S_t)(v_t^\pi - x(S_t)^\intercal w) & = 0\\
		\sum_{t = 1}^T x(S_t)v_t^\pi & = \sum_{t = 1}^T x(S_t)x(S_t)^\intercal w\\
		w & = \left(\sum_{t = 1}^T x(S_t)x(S_t)^\intercal\right)^{-1} \sum_{t = 1}^T x(S_t)v_t^\pi
```

For $n$ features, the time complexity of solving this equation is $O(n^3)$. Since we do not know $v^\pi_t$, we can again
use the estimates $G_t$, $R_{t+1} + \gamma \hat{v}(S_{t+1}, w)$, or $G_t^\lambda$. Respectively, these algorithms are 
called **LSMC**, **LSTD** and **LSTD($\lambda$)**. For *off-policy* LSTD with a linear function, 
it will now also converge to the optimal value function, whereas standard TD does not guarantee this.

### Least-Squares Control

An idea for control would be to do policy evaluation using *least-squares Q-Learning*, and then follow that with 
a greedy policy improvement. Approximate $q_\pi(s, a) \approx \hat{q}(s, a, w) = x(s, a)^\intercal w$. We want to 
minimize the squared error between those from experience generated by $\pi$. The data would consist of 
$D = \{(s_1, a_1, q_1^\pi), (s_2, a_2, q_2^\pi), ..., (s_T, a_T, q_T^\pi)\}$.

For control the policy aims to be improved. However, the experience has been generated by many different policies. So, 
to evaluate $q_\pi(s, a)$, learning must happen off-policy. The same idea as Q-learning can be used. First, use 
experience generated by the old policy $S_t, A_t, R_{t+1}, S_{t+1} \sim \pi_{old}$. Then, consider a successor action 
$A' = \pi_{new}(S_{t+1})$. $\hat{q}(S_t, A_t, w)$ should then be updated towards the alternative action 
$R_{t+1} + \gamma \hat{q}(S_t, A', w)$.

The **LSTDQ** algorithm does this. The calculation of the parameters can be derived in a similar way as linear 
least-squares prediction (Using linear Q-target $R_{t+1} + \gamma \hat{q}(S_{t+1}, \pi(S_{t+1}), w))$. After doing this 
derivation, the result becomes

$$
	w = \left(\sum^T_{t = 1} x(S_t, A_t)(x(S_t, A_t) - \gamma x(S_{t+1}, \pi(S_{t+1})))^\intercal\right)^{-1} \sum^T_{t = 1} x(S_t, A_t)R_{t+1}
$$

Now, it is possible to use LSTDQ for evaluation in Least-Squares Policy Iteration (**LSPI-TD**).

(algorithm:least-squared-policy-iteration-TD)
```{pcode}
\begin{algorithm}
	\caption{LSPI-TD}
	\begin{algorithmic}
		\REQUIRE $D$, $\pi_0$
		\STATE $\pi' \Leftarrow \pi_0$
		\WHILE{$ \pi' \not\approx \pi$}
			\STATE $\pi \Leftarrow \pi'$
			\STATE $Q \Leftarrow LSTDQ(\pi, D)$
			\FORALL{$s \in S$}
				\STATE $\pi' \Leftarrow \arg\max_{a \in A} Q(s, a)$
			\ENDFOR
		\ENDWHILE
		\RETURN $\pi$
	\end{algorithmic}
\end{algorithm}
```

The algorithm uses LSTDQ for policy evaluation. Then, it repeatedly re-evaluates experience $D$ with different policies.
The LSPI algorithm always converges when doing table lookup, and chatters around the near-optimal value function when 
using linear approximation. Non-linear approximation is not possible, since LSTDQ uses a linear target.