# Multi-agent deep reinforcement learning

This repository implements an agent that solves the
[Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis)
Unity Agents environment.

![Trained Agent][image1]

## Introduction

In this environment, two agents control rackets to bounce a ball over a net. If
an agent hits the ball over the net, it receives a reward of +0.1.  If an agent
lets a ball hit the ground or hits the ball out of bounds, it receives a reward
of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and
velocity of the ball and racket. Each agent receives its own, local observation.
Two continuous actions are available, corresponding to movement toward (or away
from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must
get an average score of +0.5 (over 100 consecutive episodes, after taking the
maximum over both agents). Specifically,

 - After each episode, we add up the rewards that each agent received (without
   discounting), to get a score for each agent. This yields 2 (potentially
   different) scores. We then take the maximum of these 2 scores.
 - This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of
those **scores** is at least +0.5.

## Getting started

This project requires Python 3.6+. To install all dependencies, ensure you are
on an activated virtualenv.

Once that's the case, performing the command `pip install -r requirements.txt`
will install all software dependencies to run the code.

Users of macOS and Linux should be good after installing requirements. If
execution fails for some reason, unzipping the files Tennis.app.zip or
Tennis_Linux.zip should fix any issues found.

## Running the code

### Training the agent

With all requirements installed, the preferred method for training the agent is
by executing the train.py module. The easiest way to run it is by calling

```bash
python -m deeprl.train
```

In the command line. For additional arguments, please run `python -m
deeprl.train --help`.

### Evaluating the agent

Similarly, to evaluate the agent, please call `python -m deeprl.eval`. The code
will search for files called `checkpoint-%d.pth` in the current working
directory.  You can specify other paths by using the command line interface.
Failure to load a checkpoint file will imply in evaluating a random agent.

## Jupyter Notebook

As an alternative, you can call the code interactively in the
[Report](Report.ipynb) notebook.
