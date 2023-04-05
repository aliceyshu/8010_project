# Empirical Analysis of Built-In POMDPs.jl Algorithms: A Comparative Study

In this study, we aim to perform an empirical analysis of multiple algorithms using the package [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl) and examine the performance of each algorithm.

This study is part of DS8010 -Interactive Learning in Decision Process course project, Winter 2023 term. Details about the course can be found [here](https://www.torontomu.ca/graduate/datascience/courses/).

# POMDP Problems

1. Crying baby
2. Tiger problem
3. Paint problem
4. Query problem
5. Mini hallway problem
6. Rock problem
7. Simple grid world problem
8. Maze problem

# Algorithm

We solve the above POMDP problem using the algorithms listed below:

1. Incremental Pruning (IP)
2. QMDP
3. Successive Approx-imations of the Reachable Space under Optimal Policies(SARSOP)
4. Partially Observable Monte CarloPlanning (POMCP)
5. POMCP with observation widening (POMCPOW)
6. Fast Informed Bound (FIB)
7. Point-Based Value Iteration (PBVI)

# Outcomes

Results of the experiments can be found in result folder

# Report

- Project Proposal can be found here:
- Intermediate report can be found here:
- Final complete report can be found here:

# Timeline

- [ ] Timeline•Week 1-2 (March 13-26)
  - [ ] Review relevant literature on POMDPs and their algorithms
  - [ ] Prepare the POMDPs problems (crying baby, paint problem, query problem)for evaluationand evaluate the performance of each algorithm on the selected POMDPs problems
  - [ ] diagrams to explain the problem, architecture of the algorithm
- [ ] Week 3-4 (March 27-April 9)
  - [ ] Review relevant literature on POMDPs and their algorithms, and complete intermediatereport (introduction, literature review, methods section) which is due **April 10**
  - [ ] Prepare the POMDPs problems (simple grid world problem/ maze problem) and evaluatethe performance of each algorithm on the selected POMDPs problems
- [ ] Week 5-6 (April 10-23)
  - [ ] Prepare the remaining POMDPs problems (simple grid world problem/ maze problem) andevaluate the performance of each algorithm on the selected POMDPs problems
  - [ ] Draft the research report (experimental setup, results, conclusions)
- [ ] Week 7 (April 24-26)
  - [ ] Review and finalize the research report, submit the research report by thedue date of **April 26**

# Reference

- Edward Balaban Tim A. Wheeler Jayesh K. Gupta Maxim Egorov, Zachary N. Sunberg andMykel J. Kochenderfer.  POMDPs.jl: A Framework for Sequential Decision Making under Un-certainty.Journal of Machine Learning Research, 18, April 2017

# License

[LICENSE.md](https://github.com/aliceyshu/8010_project/LICENSE.md)
