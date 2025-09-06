Title: How to Analyse Algorithms (Part 1)
Date: 2025-09-06 18:22
Category: Algorithms
Tags: algorithms, education, analysis

# Introduction

When talking about an algorithms complexity, we are typically referring to an estimate of the time and space required to execute the algorithm. For this article, we will focus on the time requirements, but the space requirements are quite analogous. Normally the time and space required are functions of the size of the algorithms input.

# Complexity

## Building intuition

A question that is naturally arising is, how do we standardise the measurement of the time required for an algorithm to run? We'll first introduce a barometer instruction.

Barometer Instruction
: The instruction that is executed the most number of times in an algorithm

While it is not a perfect measure of the number of operations in an algorithm, it is often proportional to the algorithm's running time.

The following code block is a simple linear search, that takes in an array and a target that is guaranteed to be in the array, and returns the index of the target

```
def linear_search(array, target):
    for i < array.length:
        if array[i] == target:
            return i
```

A barometer instruction for this algorithm is the expression `array[i] == target`, and there are others, but we start to get lost in nuance if we talk talk too much about that, so for now, we will take this expression as our barometer instruction.

How often is our barometer running though? There are three cases:

1. Best case: The target is found in first element of the array and we make one comparison
2. Worst case: The target is in the last position of the array and we do $n$ comparisons
3. Average case: The average case is more complex. We need to make an assumption, that the target has equal probability to be in any of the locations in the array. In that case, we have to do 1 comparison if it is in the first location, two if it is in the second, and so on, all the way up to $n$ comparisons if it is in the last location. The expected number of comparisons we need to make is then the sum of the probability it is in location $i$, multiplied by the number of comparisons required if it is there. That is, $$\sum_{i=1}^ni\times\frac{1}{n}=\frac{n+1}{2}$$ so the average case the algorithm does $\frac{n+1}{2}$ operations.

## Complexity classes

So now that we have seen a simple example, how do we actually compare algorithms time complexity? Well first, in general we'll talk about the worst case because that's what every other introduction to complexity works too, but there are examples where the average case is actually used in practice (for example, quicksort is in the worst case worse than merge sort).

When we're talking about the worst case behaviour of an algorithm, it's typically in the context of a large input. The nuance of smaller inputs is often overlooked, and maybe an algorithm that has poor performance for big inputs is actually really good for your use case because you know what your inputs can be so you can perform more specific analysis rather than a random input analysis.

So if all we care about is how it grows as the input grows, we really only care about the highest order term when calculating the number of times a barometer instruction is called. For example, if we have the function $$t(n)=2n^2+5n+1$$ when $n$ is small, say $n=1$, then $t(1)=2\times1^2+5\times1+1=8,$ and the linear term actually matters. But when $n$ is even just slightly larger, say $n=10,$ then we get $t(10)=2\times10^2+5\times10+1=251,$ and the linear term is contributing a lot less. As $n$ gets even bigger, the linear term matters less and less. And we haven't even talked about how little that poor 1 at the end matters.

Making a small mathematical leap, we can see that the ratio between $t(n)$ and the function $2n^2$ goes to 1 as $n\rightarrow\infty$, that is $$\lim_{n\rightarrow\infty}\frac{2n^2+5n+1}{2n^2}=1.$$

When we're talking about the complexity of an algorithm we barely even care about the coefficient of the highest order term, and often just throw that away too. But we can formalise that idea, which in my experience is where a lot of students start to struggle. Hopefully building up to the formal idea, rather than starting at the formal idea and working our way through examples gives some insight into how we've arrived at the formal idea and it helps someone.

Asymptotic Upper Bound, $O(g(n))$
: Let $f(n)$ and $g(n)$ be eventually nonnegative functions on the positive integers. Then we write $f(n)=O(g(n))$ if there exists a positive real number $M$ and a real number $x_0$ such that $|f(x)|\leq Mg(x)$ for all $x\geq x_0$.

So yeah, that's a hard to parse definition if you're not used to reading maths. To try to get a feel for it we'll start at the end. For every $x$ bigger than some value, which we're labelling $x_0$, which we can interpret as every $x$ that is to the right of some point on a plot, the function $f$ is smaller than the function $g$, and we can multiply $g$ by some number if we really want to make it true, but it has to be the same number for every possible $x$ to the right of our magic point.

So circling back to our example from earlier, we had $t(n)=2n^2+5n+1$. We need to find a function $g$ that is no smaller than $t(n)$ after some value of $n$, but we're only allowed to user the $n^2$ term. It's easy if $t(n)=2n^2$, we can just use $g(n)=2n^2$. Unfortunately it is not that easy, but we can start building up our function $g$ from here. The next term that we need to account for is the $5n$ term, and at some point $5n^2\geq 5n$, so we can tack that onto our $g$ function to get $g(n)=2n^2+5n^2$. We do still have to worry about the $1$ term, but hopefully it's easy to see that $n^2$ will also be bigger than $1$ at some point, so we can update our $g$ function be $$g(n)=2n^2+5n^2+n^2=8n^2$$ and because each of the terms within this function is at least as big as the corresponding terms in $t(n)$, then we know that the identity $$t(n)\leq g(n)=8n^2$$ holds, so we know that $g$ is an upper bound for $t$. We even know what a possible constant (the $M$) the definition asks for can be, $8$ in this case. So we can now write that $$t(n)=O(n^2),$$ which is read as "t of n is big O of $n^2$."

So this is great, we can approximate the run time of algorithms we design using big O notation, and that can give us an upper bound on just how bad it can be. There are plenty of cases where we want to know that we're going to have to wait at least a certain amount of time too. There is an analogous concept called the asymptotic lower bound.

Asymptotic Lower Bound, $\Omega(g(n))$
: Let $f(n)$ and $g(n)$ be eventually nonnegative functions on the positive integers. Then we write $f(n)=\Omega(g(n))$ if there exists a positive real number $M$ and a real number $x_0$ such that $|f(x)|\geq Mg(x)$ for all $x\geq x_0$.

This is often less cumbersome to calculate than the upper bound, although care does need to be taken in the case of negative coefficients in our function. In our case we have the chain of inequalities $$2n^2\leq2n^2+5n\leq2n^2+5n+1$$ and so trivially we get the value for our constant being $2$ in this case. So we can write that $$t(n)=\Omega(n^2),$$ which reads as "t of n is omega of $n^2$."

Naturally the question arises about the upper and lower bound being equal, and yes, there is a tight bound, and it's definition is a combination of the lower and upper bound at the same time.

Asymptotic Tight Bound, $\Theta(g(n))$
: $f(n)$ and $g(n)$ be eventually nonnegative functions on the positive integers. Then we write $f(n)=\Theta(g(n))$ if there exists positive real numbers $M_1,M_2$ and a real number $x_0$ such that $M_1g(x)\leq f(x)\leq M_2g(x)$ for all $x\geq x_0$.

This definition on the surface seems more complex than the previous two, there are two constants after all, but it really just says that if $f(n)=\Theta(g(n))$ if both $f(n)=O(g(n))$ and $f(n)=\Omega(n)$ at the same time.

# Combining Intuition and Formalism

Now that we've seen the formal definition, we can apply it back to where we were building intuition. We know there are $n$ comparisons that need to be done in the worst case for our linear search function. So the function is $O(n)$ because $n\leq 1\times n$ so our constant can just be $1$ in this case. Likewise, it is also $\Omega(n)$, and the constant can again be $1$. Because it is both $O(n)$ and $\Omega(n)$, then we automatically get that it is $\Theta(n)$ as well.

I mentioned earlier that if we know nothing about our input data, then there's not really much we can do other than assume the input data is random. What if it is not random though, what if the array we were passing in was sorted? Can we do better than $\Theta(n)$? The answer is yes, we can do as well as $\Theta(\log(n))$, but that is for another post.

# Conclusion
Do people actually talk like this? All these big O's, $\Omega$'s, and $\Theta$'s? If we have to be precise, sure, but the everyday parlance for talking about runtime complexity is to use "big O". In most situations where it is coming up everyone will understand you to mean it's the smallest upper bound for the run time of the function. Don't get too hung up on that though, when you're talking to a group of people the agreed upon vocabulary will quickly become apparent even without directly communicating about what it will be.

Do not forget, this analysis is for "large" inputs (where large may actually be really small still). Think about the input your functions are actually going to receive. Maybe the standard choice of function isn't actually the best choice for your particular use case.