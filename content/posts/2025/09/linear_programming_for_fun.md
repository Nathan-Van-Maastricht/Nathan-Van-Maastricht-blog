Title: Linear Programming for Fun
Date: 2025-09-13 21:46
Category: Maths
Tags: maths, optimisation, hobbies

# Introduction

There's a game I play called Farmer Against Potatoes Idle. Much like is implied by the name, it is an idle game. For those not familiar with this genre of games, they are designed to require only occasional input. Cookie Clicker is another idle game, and was an instrumental game in popularising the category.

Because of the idle nature though, there is a lot of room within these games to gain or lose efficiencies. To combat some of these efficiency problems, I decided to write a little optimiser for one aspect of the game, the pet system.

The story of the game is you are a farmer who woke up in the middle of a field and you are being attacked by a horde of mutated potatoes. One goal you might have is to complete the last world in the game, and to do that you need to make your farmer stronger. Once you get far enough through the game the pet system becomes available to you, and you can start using a party of pets.

# Pets

Pets are an integral part to the game, providing bonuses to stats such as how good items are, experience, and proxy stats that give you more health, more damage, and proxies for other proxies. The entire game is a large, non linear combination of proxies for strength and health essentially, and pets are just one of these aspects, but is also one of the more interesting mechanics in my opinion.

All pets are classified in two classes, either Ground or Air. Each pet has at least one bonus it gives to your farmer, but most pets have more than one. Each pet has a level, but I have picked to ignore this aspect as typically you would maximise the level of the pet if you're going to use it. Each party can have up to six pets, and within a party you are only allowed up to three Ground pets and three Air pets.

# Optimising Pets

If an appropriate team of pets is not selected, then parts of the game can take up to days longer at each stage, which could mean instead of playing for 5000 hours, you could be sitting at 7000 hours. This process isn't necessary, but it does save a lot of time and make the sense of progress feel faster! So lets talk about how I wrote an optimiser for my "base" party of pets.

The optimiser is pretty simple. I've used Binary Integer Linear Programming approach for it as it is a problem that naturally maps to this tool. What I'm going to describe in this section is actually the first implementation I did, which worked great. I later refined the linear program though to make it easier to select exactly what types of bonuses I wanted, although I did not make the workflow for using that program particularly great (it is actually bad)

## High Level Linear Program

This section will list of the linear program at a high level, leaving out any maths, this is essentially the documentation.

Objective function
: Maxmisie the number of bonuses

We do have a couple of constraints too.

The first type of constraints is on the number of pets, and there are three.

Maximum number of pets
: Total pets used needs to be less than or equal to six.

Maximum number of Air pets
: Total number of Air pets needs to be less than or equal to three.

Maximum number of Ground pets
: Total number of Ground pets needs to be less than or equal to three.

Conveniently the two specific constraints on the types implies the more general less than six, so we don't actually need that constraint. In some cases it is better to over do it on the constraints and let the solver work it out, but in this case our problem space is so small I didn't actually waste the time writing the associated code.

We also have more auxiliary constraints related to the relationship between the pets and the pets bonuses.

Pet $\Rightarrow$ Bonus
: If a pet is selected, then we need to indicate that all the bonuses associated with that pet is also selected.

Bonus $\Rightarrow$ Pet
: Similarly, if a bonus is selected, then at least one pet with that bonus needs to be selected too.

These two constraints ensure that no pet is selected without recording the bonus is counted, and at the same time, a bonus can't be counted unless there is a pet with that bonus selected.

This set of constraints is sufficient to capture our entire problem, so we can formalise it, and then write some code to automate finding the optimal solution.

## Mathematical Definition

### Variables

First we'll introduce our variables, with potentially some abuse of notation. We have two types of decision variables that are both binary.

For each pet, $p$ it has an identifier, $i$, it has a type $t\in{\text{Air}, \text{Ground}}$, and an associated set of bonuses $B$. We will denote the binary variable $$p_{i,t}^B.$$

For each bonus, $b$ it is much simpler, we just have an indentifier, $j$, and we will denote the binary variable $$b_j.$$

### Objective

Due to the decision to include variables for the bonuses, this makes the objective trivial to write out, we just add up all of them which tells us how many bonuses have been selected, and we want to maximise the number of bonuses.

$$\max\sum_{j}b_j.$$

### Constraints

There are four constraints in total, given that we don't need a constraint for the total number of pets as it is implied by the two constraints for the type of pet.

Maximum number of Air pets
: Total number of Air pets needs to be less than or equal to three, can be formalised as $$\sum_{i}p_{i,t}^B\leq3\text{ where } t=\text{Air}$$

Maximum number of Ground pets
: Total number of Ground pets needs to be less than or equal to three, can be formalised as $$\sum_{i}p_{i,t}^B\leq3\text{ where } t=\text{Ground}$$

These two constraints are hopefully easy enough to grasp. We are adding up the value of the binary variable for each of the Air or Ground pets, and only three of them are allowed to be set to a 1, the rest need to be 0.

Pet $\Rightarrow$ Bonus
: If a pet is selected, then we need to indicate that all the bonuses associated with that pet is also selected. This is slightly more complex to understand the formalism, for every element in the set $B$ associated with a pet $i$, we need to make sure the corresponding variable for that bonus is at least as big as the variable for the pet's variable. That is, if the pet variable is set to 0, then the variable for the bonus can be either 0 or 1. But if the pet is selected, then the variable has to be exactly 1.

$$p_{i,t}^{B}\leq b_j \text{ for all } b_j\in B.$$

Because of the unconstrained nature of the bonus variables when a pet isn't selected, we do need to make sure we can't indicate a variable is selected when there is no pet with that bonus selected. Otherwise the objective function would just select every bonus, which maximises the objective function.

Bonus $\Rightarrow$ Pet
: Similarly, if a bonus is selected, then at least one pet with that bonus needs to be selected too. Unfortunately, this is the hardest constraint to understand, although in the world of binary constraints, it could be worse. For each of the bonus we want to sum the variables for the pets that contain those bonuses, and make sure the variable for the bonus is less than that sum. So if 0 pets with that bonus is selected, the sum will be 0 and we have fixed the problem of counting bonuses that don't actually have a pet with that bonus. As a sanity check to make sure we haven't messed up the counting in other ways, if at least one pet is selected, then the sum will be at least 1, and the variable will be allowed to be 0 or 1. Combining that with the previous constraint though, and we get that it is not allowed to be 0 if a pet is selected with it, so it is forced to be 1. This is where the biggest abuse of notation is going to come in, our sum is going to be over the pets that have the bonus in their bonus set $$b_j\leq\sum_{b_j\in B}p_{i,t}^B.$$

### Solving

There are many libraries for solving binary integer linear programming problems, both open source and otherwise. In this case I used [HiGHS](https://highs.dev/), which I have been exploring more recently. I have also used [OR-Tools](https://developers.google.com/optimization) in the past.

You do need to find the appropriate data, or collect it on your own while playing the game. I haven't found a full dataset yet, but the wiki for the game links to an out of date subset of the data, that has not been patched, and is the underlying data used by another [useful tool](https://erik434.github.io/fapi-pets/) for tracking pets and pet combos (which we have completely ignored in this post). The list that I currently have is on my [github](https://github.com/Nathan-Van-Maastricht/fapi_optimiser), and is being updated as I get access to more pets. There is probably a way to find this information more effectively, but on a personal note, I enjoy the anticipation of finding out if my team is going to change when I find a new pet or not.

## Summary

Was doing any of this work essential to playing an idle game that I probably shouldn't actually care too much about to begin with? Absolutely not. Was it fun to do this little bit of work to optimise just a little bit of my life? Absolutely yes. The pet system isn't the only aspect of the game that can be optimised, but like I said earlier, I think it's the most interesting. Almost every other optimisation can be done on the fly with heuristics that can be played out in your head. At least with the pet system though, there is optimal solutions which are hard to just stumble across.

As for limitations with this model, I alluded to a couple throughout this post, and this isn't even the optimiser I use anymore. I now have a further constraint that says exactly how many bonuses I want, and then my objective function maximises the number of times particular bonuses are selected given the total number of bonuses is fixed.

There is also the aspect of pet levels and how that impacts the bonuses. The higher the level the more of the bonus you get. I swept this under the rug by saying you maximise the level as fast as possible so it doesn't really matter, but this is not entirely true, for some pets the requirements to level it up are restrictive, so it takes time. I haven't incorporated this into the solver at all, but it is something I occasionally think about doing, but I don't belive the return will be worth it.

And finally, this is not the only aspect of my life I have applied linear programming to, obviously in my professional life I have used it numerous times, but in my day to day it will occasionally make an appearance too. Linear programming is a very useful tool, especially if the constraints are relatively natural to write. With the massive amount of progress over the last few years on the solvers too, more and more problems can actually be solved in practice too, which is great! Some will argue I'm optimising away the fun, but the process of doing the optimisation is a lot of fun to me too.