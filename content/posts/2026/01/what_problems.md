Title: What Problems I Enjoy
Date: 2026-01-26 18:50
Category: Personal
Tags: machine learning, maths, personal philosophy

# Motivation

A colleague recently asked me what problems I enjoy working on, so I've been thinking more and more about where I draw enjoyment from, especially in my diffusion model project, but also from other previous projects and what I have worked on for others. This post is to organise the type of problems I enjoy

From what I've observed, most people aren't going to their 9-5 and then feel compelled to work midnight regularly. This isn't a "hustle" type of feeling either, it's literally just, that's what I want to be doing with my time. That's been my life for the last six months or so. Initially there was an extrinsic motivation for this project, but that side has become second to the intrinsic motivation for most of the time now.

I have worked on personal projects plenty of times over the course of my life, but never for this long contiguously. Almost every night unless I've got a good reason not to I'm writing some code, tweaking some configs, and seeing what changes. I'm using earlier parts of the project as test beds for current components. I'm introducing more and more complications. I'm trying to simplify what I already have. I'm trying to optimise the code. I'm trying to make the spaces that models have to learn simpler for them. It's all fantastic fun to me.

I think I've come to the conclusion that two different categories really matter to me, is the problem interesting, and do I think I can solve the problem. There are two options for each of these, so there are four major categories of problems from the perspective of how much I want to work on them. I'll call the categories

1. Interesting and likely solvable
2. Interesting and likely not solvable
3. Not interesting and likely solvable
4. Not interesting and likely not solvable

It's important to note that I'm using "solvable" and "not solvable" to have an implicit "by me with my current skills", not that I don't think these problems are solvable in general, or that they can't be solved by someone currently even. So in this sense they are in a category I'm going to call "subjectively solvable"

I'll start with what I think is the most boring of these categories, not interesting or solvable, and end with potentially the most interesting from a moral perspective, not interesting but likely solvable.

## Not Interesting and Likely Not Solvable

Problems that fall into this category are not suitable for me to even think about really. There's no intrinsic motivation for me to touch them, and even if I do, I don't have faith that I can actually have an outcome that is even promising. An example of a problem like this is hard to come up with, I try not to let problems that fall into this category occupy too much of my thoughts. Maybe a problem from physics such as resolving the conflicts between general relatively and quantum mechanics. I'm aware there are problems that have been grappled with for some time, but I don't even have enough interest to delve deeper than that really.

## Interesting and Likely Not Solvable

These problems are the most dangerous problems, and to contrast the previous category, I have many examples of these problems. These problems are dangerous because they will consume me, I have an addictive personality, and as an undergrad, and even into post grad, these types of problems consumed a lot of my thoughts, and I have nothing really to show for it in the way of a solution to any of the problems. Problems like the Collatz Conjecture and calculating Ramsey Numbers are classics of this category for a lot of people who have gone through mathematics degrees I suspect. Collatz in particular was the graph for me. A slightly less classic is thinking about how to find (or show it doesn't exist) a fixed point in hashing functions such as md5.

I think problems in this category are incredibly interesting, but due to not being likely to get a solution to them, barely ever thinking about them is a good choice. Terrence Tao in one of his blogs once suggested giving yourself permission to think about these problems for a couple of days a year to see if you can make progress, and scratch the itch, but otherwise try to avoid them.

These problems are definitely a siren song to me, and I have to resist them sometimes.

## Interesting and Likely Solvable

These are the problems I actually work on. I don't need extrinsic motivation like money to work on these, and if anything, working on these problems cost me money. If you look at my post history so far, I've been delving deeper and deeper into diffusion models. As I'm writing this post I'm training a VAE on a dataset with roughly 6m images in it (up from roughly 1.2m of my last model) so I can train another diffusion model, this time instead of being conditioned on a class label, it'll be my first attempt at a text to image diffusion model. Despite having trained a handful of VAEs for diffusion models at this point, there are still challenges involved, and solving those challenges is frustrating and rewarding at the same time. Not once have I contemplated giving in and stopping. It's felt like every step of the way I've got the skill to solve the problem I'm facing and I just need to keep working at it. It pushes me, but not so far that I don't see the path to a working model.

I mentioned earlier that the diffusion model in particular changed from extrinsic to intrinsic. This happened when after my first model started working and I knew I understood enough to talk about how these models worked to someone else in a professional setting, I wanted to keep going. The initial model was fine, it even generated MNIST digits quite well. But if I stopped there I wouldn't have a model that gives decent 256x256 images, I wouldn't understand the interactions between the VAE and diffusion model, I wouldn't understand the multiple types of VAEs or multiple types of diffusion models, I wouldn't have trained a language model, I wouldn't know the difficulties of working with massive datasets. And every step of the way has just sucked me in more and more, without ever giving me a feeling of "I can't do this".

There are a lot of problems in this category, both in the maths/tech world, and outside it. I worked in restaurants for a couple of years, and saw that generating rosters is a major bottleneck in the weekly tasks for a manager, so I have a MILP model for how to solve that problem. I don't have a pretty interface for it though, so it's virtually impossible to use unless you're me. I occasionally think about coming back to this though and working with someone to make it usable by the hospitality industry.

Even more foundational, I love cooking (hence I worked in restaurants for a bit), and over the years I've been working on focaccia in particular. It's a handful of ingredients, but the process to making it better and better is a lot of fun. I don't think I'll ever be "done" with the focaccia, but it is great to see people who haven't had it for a while and say that it's better than they remember. I'm definitely at the point of diminishing returns on my current method, so taking a note from all my ML work, I think I need to nudge myself out of the local minimum I'm currently in and change a major component of it.

One problem that arises with problems that I'm near the end of solving is they get really boring to me once I see the solution. A lot of my 9-5 work is like that, where the initial part of the project is to figure out how to develop a model or algorithm to solve a problem, and I'm pretty confident I can do it. But as soon as I see how everything links together, I need the extrinsic motivation to finish it. This emphasis that "solvable" does NOT mean EASY to me. Easy is the opposite of interesting for me, if it's easy, then the puzzle has gone, so the interesting aspect has disappeared too.

In an ideal world I get extrinsic motivation, such as someone paying me, to solve problems in this category. I always find it very hard to price services and items in this category, so I typically charge just enough to cover costs. For example, I enjoy woodworking. The last table I sold was sold at just below the cost of the wood I used in the project, so I lost money on it.

## Not interesting and Likely Solvable

This set of problems are where the money really comes from in my life. Problems that I can solve and provide value to someone else, but I don't really want to be working on them. In my academic life there was a bunch of these problems, especially in undergrad. I didn't really want to be working on differential equations, I didn't really want to be working on topology. I enjoy discrete spaces much more than continuous. I look around at the job market lately, and a lot of position descriptions seem to be falling into this category. A lot of fine tuning of LLMs, a lot of RAG. These don't have interest for me. I want to work on models from scratch. I don't really mind if they are language models, diffusion models, agents that are used in control situations. For example, I've written a post about solving combinatorial optimisation problems with a Pointer Network both in my professional life and as a personal project, they are fun, and combine my knowledge of maths and reinforcement learning.

But there is also a moral conundrum with problems in this category. What if I can do some social good, maybe I can optimise how resources for fighting fires are distributed, and I have a good chance of improving the techniques involved, but I also don't find the problem intrinsically interesting. If I could drastically reduce the amount of resources required to fight a bushfire though, that could have massive second and third order implications. Definitely more so than working on a diffusion model that isn't going to be close to SotA. Or maybe I should learn some frontend web development and finish wrapping the solution for rostering in restaurants for others to use, I can probably even charge a modest amount of money for it if I provide hosting and support, or just release it open source. But it's really not interesting enough for me to want to solve.

# Conclusion

Over the years different problems have drifted into and out of various categories. When I was a younger undergrad I thought I had a great chance of solving Collatz Conjecture, but that didn't turn out to be the case. But I've grown and it's still interesting, but I have lost some of the ego associated with thinking I can solve these types of problems (I have other ego problems now though). The real challenge is to decide how to allocate my time. As I'm getting older and most of the jobs I look at are sufficient to pay my bills, the questions start to become which of these jobs do I find most interesting, and do I think I can actually provide value to the people who are willing to pay me to work on them.

There is a moral problem with taking a job just for the money too, without that intrinsic motivation, the work will not be my best. I won't be staying up until the early hours of the morning thinking about it. I won't be dreaming about it, or thinking about it in the shower. I'll go through the motions, I'll work on the problem when I'm being paid to work on problems, but as soon as my mind is allowed to freely think, it'll be back to the intrinsically interesting problems.