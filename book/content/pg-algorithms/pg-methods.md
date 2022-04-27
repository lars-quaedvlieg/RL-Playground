# Policy Gradient Methods

```{warning}
Please note that this notebook is still **work in progress**, hence some sections could behave unexpectedly, or provide incorrect information.
```

The goal of this chapter is to discuss methods that optimize for policy directly, and not by acting greedily to a value 
function. The goal is to sample experience, and from that learn in which direction to move the policy to improve it. 
This will be done by considering the *gradient* of the policy.

For this, the policy will be parametrised by $\theta$, so $\pi_\theta(s, a) = \mathbb{P}\left[a |s, \theta\right]$. This 
could be, for example, a neural network that predicts the probability of actions given a certain state.

Now, it is possible to talk about more approaches to Reinforcement Learning. 

- Value Based

    The value function is learned, and there is an explicit policy

- Policy Based

    The policy is learned, but there is no value function

- Actor-Critic

    Both the value function and the policy are learned

The *advantages* over Policy-Based RL are the following: Better convergence properties, effective in 
high-dimensional/continuous action spaces, and can learn stochastic policies. However, there are also 
*disadvantages*. Evaluating a policy is typically inefficient and has high variance. Also, it usually converges 
to a local rather than a global optimum.

In POMDPs, a stochastic policy may be optimal in certain problems. But how to measure the quality of a policy? There are
different options.

1. Start value (Episodic environments, start state)

    $J_1(\theta) = V^{\pi_\theta}(s_1) = \mathbb{E}_{\pi_\theta}\left[v_1\right]$. This means it is a measure of reward 
    until the end of an episode with current parameters $\theta$.

2. Average value (Continuous environments)

    $J_{avV}(\theta) = \sum_s d^{\pi_\theta}(s) V^{\pi_\theta}(s_1)$ where $d^{\pi_\theta}(s)$ is the stationary 
    distribution (probability of being in a state) of the Markov chain for $\pi_\theta$. It is basically an averaged value over all states.

3. Average reward per time-step

    $J_{avR}(\theta) = \sum_s d^{\pi_\theta}(s) \sum_a \pi_\theta(s, a) R^a_s$. This is a weighted average over the 
    possible immediate rewards for each state and action pair.

However, the policy gradient of all of these are the same. Policy-based RL is an optimization problem to find the 
$\theta$ that maximizes $J(\theta)$. Methods that do not use a gradient (e.g. Hill climbing) can be used. However, it is
usually much more efficient to use gradients (e.g. Gradient Descent, Conjugate Gradient, Quasi-Newton).

## Finite Difference Policy Gradient

Policy gradient algorithms search for a local maximum in $J(\theta$) by ascending the gradient with respect to $\theta$.
This can be represented by $\Delta \theta = \alpha \nabla_\theta J(\theta)$.

**Computing gradients by finite differences** means to numerically compute them. For each dimension 
$k \in \left[1, n\right]$, estimate the $k$-th partial derivative by numerical approximation. 
$\frac{\delta J(\theta)}{\delta \theta_k} \approx \frac{J(\theta + \epsilon u_k) - J(\theta)}{\epsilon}$, where $u_k$ is
a unit vector with a 1 in the $k$-th component.

It is noisy and inefficient, but can sometimes be effective. An advantage is that it even works with non-differentiable
policies.

## Monte-Carlo Policy Gradient

The goal now is to compute the policy gradient *analytically*. Assume that $\pi_\theta$ is differentiable when it
is non-zero and gradient $\nabla_\theta \pi_\theta(s, a)$ is known.

**Likelihood Ratios** exploit the identity 

$$
	\nabla_\theta \pi_\theta(s, a) = \pi_\theta(s, a) \frac{\nabla_\theta \pi_\theta(s, a)}{\pi_\theta(s, a)} = \pi_\theta(s, a) \nabla_\theta \log \pi_\theta(s, a) 
$$

Here, the **score function** is defined by $\nabla_\theta \log \pi_\theta(s, a)$. This is useful, since it will 
make computing expectations much easier.

An example is the *Softmax Policy* $\pi_\theta(s, a) \propto e^{x(s, a)^\intercal \theta}$. For this, the score 
function is $\nabla_\theta \log \pi_\theta(s, a) = x(s, a) - \mathbb{E}_{\pi_\theta} \left[x(s, .)\right]$.

Another one is a *Gaussian Policy*. These are common in continuous action spaces. Here, the mean is a linear 
combination of state features $\mu(s) = x(s)^\intercal \theta$. The same can be done for the variance, with different 
parameters. Then, $a \sim N(\mu(s), \sigma^2)$. The score function is 
$\nabla_\theta \log \pi_\theta(s, a) = \frac{(a - \mu(s))x(s)}{\sigma^2}$.

In terms of MDPs, the policy gradient can be derived in the following manner.

```{math}
		J(\theta) & = \mathbb{E}_{\pi_\theta} \left[Q^{\pi_\theta}(s, a)\right]\\
				  & = \sum_{s \in S} d(s) \sum_{a \in A} \pi_\theta(s, a) Q^{\pi_\theta}(s, a)\\
		\nabla_\theta J(\theta) & = \sum_{s \in S} d(s) \sum_{a \in A} \pi_\theta (s, a) \nabla_\theta \log \pi_\theta(s, a) Q^{\pi_\theta}(s, a)\\
						 & = \mathbb{E}_{\pi_\theta} \left[\nabla_\theta \log \pi_\theta(s, a) Q^{\pi_\theta}(s, a)\right]
```

This is called the **Policy Gradient Theorem**. The theorem applies to all 3 of the previously discussed objective 
functions.

The algorithm for **Monte-Carlo Policy Gradient (REINFORCE)** uses the return $G_t$ as an unbiased sample of 
$Q^{\pi_\theta}(s_t, a_t)$ and this theorem to update the parameters using stochastic gradient descent.

(algorithm:monte-carlo-policy-gradient)=
```{pcode}
\begin{algorithm}
	\caption{Monte-Carlo Policy Gradient (REINFORCE)}
	\begin{algorithmic}
		\REQUIRE $\theta$
		\FORALL{episodes $\{s_1, a_1, r_2, ..., s_{T-1}, a_{T-1}, r_T\} \sim \pi_\theta$}
			\FOR{$t \Leftarrow 1, ..., T-1$}
				\STATE $\theta \Leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(s_t, a_t) G_t$
			\ENDFOR
		\ENDFOR
		\RETURN $\theta$
	\end{algorithmic}
\end{algorithm}
```

## Actor-Critic Policy Gradient

The algorithms presented before are sadly slow and have a high variance. The following ideas are to speed of these 
algorithms and use a similar idea as before. To reduce this variance (but introduce some bias), the idea of function 
approximation can be re-used on the estimate of $Q_w(s, a) \approx Q^{\pi_\theta}(s, a)$.

The idea consists of two components

- **Actor**: Update action-value function parameters $w$
- **Critic**: Update policy parameters $\theta$, in the direction of suggestion of the critic.

These algorithms follow an *approximate* policy gradient 
$\nabla_\theta J(\theta) \approx \mathbb{E}_{\pi_\theta} \left[\nabla_\theta \log \pi_\theta(s, a) Q_w(s, a)\right]$.

The critic is solving the problem of policy evaluation. So, previously discussed methods can be applied for solving it. 
An example algorithm below is an *Action-Value Actor-Critic* with the critic using TD(0) for policy evaluation, 
and the actor using policy gradient.

(algorithm:q-value-actor-critic)=
```{pcode}
\begin{algorithm}
	\caption{QAC}
	\begin{algorithmic}
		\REQUIRE $s$, $\theta$
		\STATE $a \sim \pi_\theta$
		\FORALL{steps}
			\STATE $r \Leftarrow R^a_s, s' \sim P^a_s$
			\STATE $a' \sim \pi_\theta(s', a')$
			\STATE $\delta \Leftarrow r + \gamma Q_w(s', a') - Q_w(s, a)$
			\STATE $\theta \Leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(s, a) Q_w(s, a)$
			\STATE $w \Leftarrow w + \beta \delta x(s, a)$
			\STATE $s \Leftarrow s', a \Leftarrow a'$
		\ENDFOR
	\end{algorithmic}
\end{algorithm}
```

Since there is bias, usually the algorithms will end up in a local optimum. However, if the approximation of the value 
function is chosen carefully, the policy gradient is exact. **Compatible Function Approximation Theorem** says 
that the policy gradient is *exact*
($\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[\nabla_\theta \log \pi_\theta(s, a) Q_w(s, a)\right])$, when two 
conditions are satisfied.

1. $\nabla_w Q_w(s, a) = \nabla_\theta \log \pi_\theta(s, a)$ (value function approximator is *compatible* to the 
policy)
2. $\epsilon = \mathbb{E}_{\pi_\theta} \left[(Q^{\pi_\theta}(s, a) - Q_w(s, a)^2)\right]$ (value function parameters minimize 
the MSE)

To reduce the variance on the method, it is possible to subtract a **baseline function** $B(s)$ from the policy 
gradient. This can reduce variance without affecting the expectation. The proof is as follows

```{math}
		\mathbb{E}_{\pi_\theta} \left[\nabla_\theta \log \pi_\theta(s, a)B(s)\right] & = \sum_{s \in S} d^{\pi_\theta}(s) \sum_a \nabla_\theta \pi_\theta(s, a)B(s)\\
		& = \sum_{s \in S} d^{\pi_\theta}(s) B(s) \nabla_\theta \sum_a \pi_\theta(s, a)\\
		& = \sum_{s \in S} d^{\pi_\theta}(s) B(s) \nabla_\theta 1\\
		& = 0
```

So, as long as the baseline does not include the action, it will not modify the expectation. An example of a good 
baseline function is $B(s) = V^{\pi_\theta}(s)$. The policy gradient can then be rewritten as the 
**advantage function** $A^{\pi_\theta}(s, a) = Q^{\pi_\theta}(s, a) - V^{\pi_\theta}(s)$. $\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[\nabla_\theta \log \pi_\theta(s, a)A^{\pi_\theta}(s, a)\right]$. 
This can significantly reduce variance, since the advantage function is now the distance from the mean value of a state.

The critic is in charge of estimating both of these value functions. There are multiple ways to do this.

- Using two separate parameter vectors:
    
    $V_v(s) \approx V^{\pi_\theta}(s)$ and $Q_w(s) \approx Q^{\pi_\theta}(s, a)$. Then, $A(s, a) = Q_w(s, a) - V_v(s)$.

- Using the TD error

    The TD error for $V^{\pi_\theta}(s)$ equals $\delta^{\pi_\theta} = r + \gamma V^{\pi_\theta}(s') - V^{\pi_\theta}(s)$. 
    Then,

    ```{math}
            \mathbb{E}_{\pi_\theta} \left[\delta^{\pi_\theta} |s, a\right] & = \mathbb{E}_{\pi_\theta} \left[r + \gamma V^{\pi_\theta}(s') |s, a\right] - V^{\pi_\theta}(s)\\
            & = Q^{\pi_\theta}(s, a) - V^{\pi_\theta}(s)\\
            & = A^{\pi_\theta}(s, a)
    ```

    So, the TD error can be used to compute the policy gradient. 
    $\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[\nabla_\theta \log \pi_\theta(s, a)\delta^{\pi_\theta}\right]$. 
    In practice, $\delta_v = r + \gamma V_v(s') - V_v(s)$ is used instead. This approach only requires one set of critic
    parameters $v$.

The value function for the critic can be estimated at different time-scales, as seen in previous lectures. Ideas like MC
, TD(0), and TD($\lambda$) still work. The same thing holds for actors. Monte-Carlo policy gradient can be applied, as 
well as one-step TD for Actor-critic. In summary, this means the actor and critic's choices of algorithms do not 
influence each other.

It is also possible to apply policy gradient with eligibility traces for TD($\lambda$). 
$\Delta \theta = \alpha (v_t^\lambda - V_v(s_t)) \nabla_\theta \log \pi_\theta(s_t, a_t)$, where 
$v_t^\lambda - V_v(s_t)$ is a biased estimate of the *advantage function*.

```{math}
		\delta & = r_{t+1} + \gamma V_v(s_{t+1}) - V_v(s_t)\\
		e_{t+1} & = \lambda e_t + \nabla_\theta \log \pi_\theta(s, a)\\
		\Delta \theta & = \alpha \delta e_t
```

Similar to prediction using TD($\lambda$) with a value function approximation, the eligibility traces are defined for 
each parameter and updated using the scores it encounters. The gradient with respect to the parameters becomes 
$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[\nabla_\theta \log \pi_\theta(s, a) \delta e\right]$.

Gradient ascent algorithms can follow any ascent direction. Following a good ascent direction can massively speed up 
convergence. Another problem is that a policy can often be re-parametrised without changing action probabilities. The 
vanilla gradient is sensitive to these re-parametrisation.

The benefit of the **Natural Policy Gradient** is that it is parametrisation independent. It finds the ascent 
direction that is closest to the vanilla gradient when changing the policy by a small fixed amount.

$$
	\nabla^{nat}_\theta \pi_\theta(s, a) = G^{-1}_\theta \nabla_\theta \pi_\theta(s, a)
$$

Here, $G_\theta$ is the *Fisher information matrix*. 
$G_\theta = \mathbb{E}_{\pi_\theta} \left[\nabla_\theta \log \pi_\theta(s, a) \nabla_\theta \log \pi_\theta(s, a)^\intercal\right]$. 
Using compatible function approximation, $\nabla_\theta^{nat} J(\theta) = w$. The actor parameters are updated in the 
direction of the critic parameters.