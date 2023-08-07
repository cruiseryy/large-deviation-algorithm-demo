# Importance Sampling

Given some random variable $X$ s.t. $X \sim P_0(X)$ and we want to estimate the expectation of some function of $X$ (i.e., $F(X)$ ). A Monte Carlo way is to generate an ensemble of realizations of $(x^i_n, F(x^i_n))$ for $i = 1, 2, ..., N$ according to the distribution $P_0$ and the approximate expectation is then given by

$$E\_{P_0}\[F(X)\] = \int_x F(x) P_0(x) dx \approx \frac{1}{N} \sum_n^N F(x^0_n) \qquad -EQ(1)$$

For situations where we have special interests in rare events (which can be described by a subset $B$ of all possible outcomes of $X$). Assume the $P(X \in B) = \gamma_B$ over some time horizon $T$, the relative error using Eq 1 is of the order of $\sqrt{(N \gamma_B (1 - \gamma_B))}/(N \gamma_B) \approx 1/ \sqrt{N \gamma_B}$ which can be pretty large when $\gamma_B$ is small.

The importance sampling can let one draw more frequently from $B$ (or near $B$) using some well-designed, altered pdf $P_k$ based on the Eq 2.

$$E\_{P_0}\[F(X)\] = \int_x F(x) P_0(x) dx = \int_x F(x) \frac{P_0(x)}{P_k(x)} P_k(x) dx = E\_{P_k}\[F(X)\frac{P_0(X)}{P_k(X)}\] \approx \frac{1}{N} \sum_n^N  F(x^k_n) \frac{P_0(x^k_n)}{P_k(x^k_n)} \qquad -EQ(2)$$

In our context, importance sampling has to be performed at the level of trajectories. Trajectories generated by the model are distributed according to some unknown pdf *P*<sub>0</sub>({*X*(*t*)}<sub>0 ≤ *t* ≤ *T*</sub>={*x*(*t*)}<sub>0 ≤ *t* ≤ *T*</sub>). Suppose given the model trajectory, we can evaluate some quantity of interest as given by $A(x(t))$. If we want to estimate the return period/probability of the time integral of $A(x(t))$ dropping below some threshold $a$, this can be transformed to an expectation estimation problem as given by the Eq 3, where **1**<sub>*b*</sub>(*x*) equals 1 if
*x* \< *b* and 0 otherwise. In our case, *A*(*x*<sub>*t*</sub>) can the precipitation at time step *t* and this can be used to estimate probability of total DJF precipitation dropping below some level *b*.

$$ P(\int_0^T A(x(t)) dt < b) = \int_x 1_b(\int_0^T A(x(t))dt)P_0(\{x(t)\}_{0\leq t \leq T})dx = E[1_b(\int_0^T A(x(t))dt)] \qquad -EQ(3)$$

Suppose from Eq 2 and 3 we can estimate the probability of total DJF precipitation being smaller than *b* as denoted by $\gamma_b$, then the return period is simply $r(b) = 1/ \gamma_b$ years.

# Importance Sampling + Monte Carlo

Suppose the simulation horizon is $T$ (e.g., DJF or ∼<!-- -->90 days in our case) and it can be divided into multiple sub-intervals (i.e., $T = m \tau$). A total of *N* trajectories are initiated using independent initial conditions and the same set of climatological boundary conditions is used. A workflow to implement the algorithm is given below for *i* = 1, 2, ..., *m*

1\. Iterate each trajectory from time *t*<sub>*i* − 1</sub> = (*i*−1)*τ* to time *t*<sub>*i*</sub> = *iτ*,

2\. At time *t*<sub>*i*</sub>, stop the simulation and estimate the weight associated with each trajectory *n* as given by

$$W^i_n = \frac{e^{k\int^{t_i}\_{t\_{i-1}}A(X_n(t))dt}}{R_i}$$

where

$$R_i = \frac{1}{N} \sum_n e^{k\int^{t_i}\_{t\_{i-1}}A(X_n(t))dt},$$

3\. Randomly sample *N* new trajectories (with replacement) according to the probability mass function $P(x=n) \propto W_n^i$,

4\. Add small random perturbations to the states of the $N$ new trajectories at time $t_i$,

5\. Increment $i$ by 1 and repeat 1-5.

Then more weights we put on the trajectories distributed near the domain of interets ($B$) must be inverted before plotting the intensity vs frequency/return period relationship. And the inversion is given by

$$ E[F(X)]  \approx \frac{1}{N} \sum_{n=1}^N F(X_n(t)_{0 \leq t \leq T} ) \cdot e^{-k \int_0^T A(X_n(t)) dt} \cdot e^{T \lambda(k,T)} \qquad -EQ(4)$$

where $\lambda(k,T) = \frac{1}{T} \sum_i log R_i$

# Toy Problem

I used a simple Markov chain to simulate daily precipitation over a 200-day interval. The probabilities of rain and no rain for today are only conditioned on the state of yeasterday and if we roll 'rain' for today, the intensity is drawn from a log normal distribution. The parameter values are arbitrarily assigned. Run ```control_run.m``` to do a control run of 100,000 realizations (i.e., 100,000 years) and run ```alter_run.m``` to repeat for 20 times the altered simulation using the abovementioned algorithm (and each altered run consists of 128 realizations/years). **Note that step 4 was not used in this toy problem since I used a stochastic process already.**

# A Brief Derivation
<img width="1144" alt="image" src="https://github.com/cruiseryy/large-deviation-algorithm-demo/assets/66930258/7f189982-5b69-437a-b595-ce472b32bcac">

## Reference
Ragone, F., Wouters, J., & Bouchet, F. (2018). Computation of extreme heat waves in climate models using a large deviation algorithm. Proceedings of the National Academy of Sciences, 115(1), 24-29.

Giardina, C., Kurchan, J., Lecomte, V., & Tailleur, J. (2011). Simulating rare events in dynamical processes. Journal of statistical physics, 145, 787-811.
