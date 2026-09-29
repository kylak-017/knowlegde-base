---
title: Welcome to My Knowledge Base
---
---
Kyla's Knowledge Base
---

Welcome! This is my digital garden for physics and chemistry research notes.

## 📂 Topics

### 1. Computational Chemistry
* [[Computational-Chemistry/Initial Learnings]]
	* This is a hand-written note compiled with the following information:
		* Purpose of DFT/DMFT
		* DFT Theory/Assumptions
		* DFT Workflow
		* DMFT Theory/Assumptions
		* DMFT Workflow
		* Wannier Functions
		* NQS Impurity Solvers
* [[Computational-Chemistry/DFT-Code/DFT-Code]]
	* This page introduces the self-written SCF loop using Python.
	* It does not use any external libraries and rather uses native libraries such as **numpy.**
	* Additionally, it uses Fermi-Dirac smearing for the degauss function.
	* ![[Fermi-Dirac.png]]
		* This is a brief comparison of other smearing techniques. 
- [[Computational-Chemistry/DMFT-Code/DMFT-Code]]
	- This page explains DMFT implementation.
	- It compares accuracy to prediction model for single-band Hubbard Hamiltonian.
	- It also elaborates on the derivation of the prediction model for said Hamiltonian.
	- It explains the iteration number that is recommended for the accuracy for different types of systems (Mott Insulator, Doped Metallic State).
- [[Computational-Chemistry/Impurity-Solver-NNs/Impurity-Solver-NNs]]
	- This page explores in-depth annotations of:
		- impurity solver-related research papers
		- neural network research papers
		- NQS research papers
	- It also explores self-implementation of the code of impurity solver NNs.
- [[Computational-Chemistry/Parameter-Automate/Parameter-Automate]]
	- This page explores explanations of what each parameter in DMFT does and how it affects the predicted results.
	- It compares this effect with DFPT, which has reported observations of parameters causing shifts in predicted results.
	- It also explores the development of an algorithm that predicts and optimizes the parameters needed to analyze the properties of a random material given.