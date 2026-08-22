Title: Project Summaries
Date: 2026-08-22 13:00
Category: Personal
Tags: projects

I've had a few people ask me about what I've worked on over the years, so writing a short article I can point to every time is easier than giving the run down every time. It's also far more digestable than handing over a resume.

I'll roughly try to keep it in order of what I find most interesting and I will indicate if I worked on the project on personal time, or if it was paid work. This is also not an exhaustive list, so I've tried to pull out a good mix of problems I've touched.

# Text to Image Generation

### Hobby

By far the hardest and deepest project I've worked on. I had to train a language model, an autoencoder, and a flow matching model (which I colliquially call a diffusion model when talking to people who aren't deep in the space). From inception to completion of version 1 took about 9 months of work, with long training runs. [This post goes into more detail about the process and components, with links to even earlier posts](../../../articles/2026/06/the-complexity-of-building-a-flow-matching-model-for-fun.html)

Each step had different data requirements. The language model needed a nice balance of sentence structure to make the connections in english required for some form of semantic clarity, but at the same time it needed a large enough vocabulary that it could relate niche objects to common objects and settings. It also needed to not be surprised by the types of phrases in the training data for the text conditioning on images. It ended up being a combination of short stories, simple wikipedia, and the captions themselves that were used to train the language model.

The autoencoder was probably the easiest for data setup, but the most complex training pipeline. It had a patchGAN discriminator to ensure crisp images on the decoding stage, and it had a discreet bottlneck. Tuning all the hyper parameters took days of starting the training, evaluating, and stopping, before finally giving the model time to train when I was happy with different aspects such as loss function coefficient schedules, learning rate schedules, and model capacities.

The actual flow matching model took the hard parts from both the earlier models and softened them slightly. It has a lot of tweaking of hyper parameters, even throughout training I was tweaking the parameter related to how I was sampling time. I got quite good at telling if a training run is worth continuing, or if it should be stopped early because the model was stuck somewhere.

# Molecular Metrics

### Hobby, maybe paid?

I've recently been working on a project with an academic friend who proposed working on the COMPAS1d dataset and attempt to produce a model that can predict various metrics associated with it. While I don't understand much of the chemistry and physics involved, the dataset is well structured, with good augmentations available to the dataset as well, which has made the project go relatively smoothly.

The model I've developed isn't particularly complex or big by today's standards, and trains quite quickly to a high quality result. We're still exploring options for publication and if we can make the model even better, either with higher quality results, or more efficient, in either the size of the model, or the sample efficiency of the model. Given it's still in the early days though, I expect we'll make a lot more progress on this.

# TSP Solver

### Paid turned into hobby

My first major bit of work in the ML space that was paid was related to trying to quickly produce high quality solutions to the TSP. I did a lot of exploration, both because I was really learning ML at the time, and there were a lot of techniques to try. I ended up settling on using a model inspired by a pointer network with an LSTM backbone that was trained with RL, and eventually in a MARL setup with a model that generates training instances. Given that it is paid work, I won't go into too many details.

Where I have gone into details though was a hobby extension of this project, where I use a different backbone for my pointer network. I was curious if similar tricks could be used to make the model perform well. [This can be read more about here](../../../articles/2025/11/reinforcement-learning-for-combinatorial-optimisation.html)

# Various Accoustic Classification Problems

### Paid

I don't overly enjoy these types of problems, mostly becasue the data tends to not be particularly good on the problems I've worked on up until this point, so really good results aren't typically feasible. But they also suffer from mostly being the same problem over and over, so you apply the same techniques, and tweak hyperparamters for that particular dataset.

In general though, these problems give you some form of wave form, maybe it's an animal, a boat, an engine, and you have to figure out what class it is from some predetermined set of classes.

# Anomaly detection

### Paid

Anomaly detection problems on the other hand are much more interesting to me. They also often suffer from the data being not particularly good though, which makes them challenging. But even a perfect dataset to work over for a problem of this type tends to be interesting to me. There is typically a large inbalance between the number of samples of normal operations and the number of samples of anomalies. It is typically some form of time series again, but most of the time it is normal. If an expert hasn't spent weeks anotating the dataset, then it's often an unsupervised learning task as well, needing to understand what normal operations is, so it is possible to detect what not normal is. Downstream from this task is then trying to classify the type of anomaly. 

My first glimpse into the world of ML was on this type of problem. It was before the days where deep learning was popular and applied without thought to everything, so we had to come up with interesting ways to solve this problem. I still find problems like this fascinating, and it is a widely applicable problem type. It can be applied to machines, demand forecasting like in electricity grids, demand forecasting in retail, stock markets, or you can even go a bit meta and train a model to determine if a model trained to solve a type of problem is going to perform well on a particular instance, or poorly, which you can then use to build a dataset which can be good for sample efficiency at train time of yet another model.

I occasionally look around for datasets where anomaly detection could be used for fun, and while I haven't dedicate any time at the hobby side yet, I do have a few ideas for what I Want to do in this space to learn more about it.

# Other Notes

## Non-Deep Learning

Deep learning isn't everything. I find it very fun though. There's something that really tickles my brain when I'm tweaking an architecture that is failing, and getting it to work. Then tweaking the hyperparameters and training loop to make it work even better. And eventually seeing the end product.

But I also really enjoy metaheuristics, I think everyone should understand simulated annealing for example, it's history, what lead to it being such a good metaheuristic, why it's still used regularly today despite being so old. 

Mathematical programming is also incredibly useful. I use Mixed Integer Linear Programming far more than I probably should, [some might say I optimise the fun out of games](../../../articles/2025/09/linear-programming-for-fun.html#linear-programming-for-fun). I've never touched Eve Online, and I've never touched Factorio. Both of those games would consume my life. I'm also pretty sure both of those games will have many mathematical programming applications.

Then there's just good old algorithms to solve particular problems. Variations on A* for path finding, k-opt for combinatorial optimisation, binary search, the list can just go on and on.

And in reality, a combination of these techniques often perform better than any one on their own. The TSP solver I mentioned earlier actually employs a pointer network followed by some simple 2-opt. Initially though when I was trying to just get a solution to the TSP I was using simulated annealing. Which transitioned into using a worse version of the pointer network, and feeding that as the initial condition into a shorter run of simulated annealing. Eventually I made the pointer network good enough that I didn't need the power of that simulated annealing provided though, so it got dropped and simplified to 2-opt.

## PhD

I don't have a PhD. This is both good in that I haven't bogged myself down with all of the academic bloat that comes along with a PhD. But a PhD does look good on a resume still as well. There is definitely a different perception of someone who has done a PhD compared with someone who hasn't in this space. It can be hard to even get a conversation with someone without a PhD in your back pocket. My attempt to fix this issue was the text to image model, it's complex, requires deep understanding of a variety of models, and it worked, all these aspects look great and hopefully make up for large chunks of missing out on a PhD.

