Assume that we have a verry simple model as follows:

$$p(y \mid \mu)=\prod_{n=1}^{N} \operatorname{Normal}\left(y_{n} \mid \mu, 1\right)$$

The normal distrution is defined as follows:

$$\operatorname{Normal}(y, \sigma^2) = \frac{1}{\sigma \sqrt{2 \pi}} e^{-\frac{1}{2}\left(\frac{y-\mu}{\sigma}\right)^{2}}$$

Admitedly, this eqution looks dauntingly complex. However, once we assume the **standard normal** distribution, the eqution becomes much more simple:

$$\operatorname{Normal}(y, 1) = \frac{1}{\sqrt{2 \pi}} e^{-\frac{1}{2}\left(y-\mu\right)^{2}}$$

When we transform both sides of the equation to the log domain, we get an even simpler equation:

$$\log p(y) =  -\frac{1}{2} (y-\mu)^2 + -\frac{1}{2}\log{2\pi}$$

The first derivative of our function (w.r.t. $\mu$) is:

$$\log p'(\mu) = \mu - y$$

As we want to maximize $p(\mu)$, we need to find the root of the first derivative.

$$ 
\mu - y = 0 \\ 
y=\mu
$$

This is clearly a (global) maximum, as the second derivative is never positive:
$$\log p''(\mu) = -1$$

It is also easy to see that maximizing $p(\mu)$ is equivalent to minimizing $-\log p(\mu)$, or in other words, minimizing the negative log-likelihood is the same as minimizing the squared distance between the given data point $y$ and the mean $\mu$:
$$ \operatorname{argmax}\left(-\frac{1}{2} (y-\mu)^2 + -\frac{1}{2}\log{2\pi}\right) = \operatorname{argmax}\left(  (y-\mu)^2 \right) $$


