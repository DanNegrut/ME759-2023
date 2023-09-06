# Linear Solvers

Solving linear systems with high accuracy and time efficiency is an important aspect of many computational engineering applications. Below are some commonly used numerical methods to solve systems of linear equations that you can implement, and more importantly, *optimize* and *accelerate* by taking advantage of the parallel computing frameworks you have learned in this class.

## Iterative methods

An iterative method uses an initial guess as an estimation of the solution and improves the estimation at every iteration until reaching a maximum number of iterations or satisfying an accuracy test. 

### Jacobi method

This is an easy entry point to understand iterative algorithms. It will provide an opportunity to understand how properties of the coefficient matrices that come into play impact the speed of convergence and the accuracy of the solution (https://en.wikipedia.org/wiki/Jacobi_method). 

### GMRES

Another type of iterative method that can solve a larger class of linear systems (https://en.wikipedia.org/wiki/Generalized_minimal_residual_method). 

## Direct methods

A direct method usually produces a numerical solution by a finite sequence of operations, with no reliance on an iterative method (although sometimes direct methods are employing an iterative step as well – not discussed here).

### LU decomposition

Some sequential programming examples can be found in the wiki page (https://en.wikipedia.org/wiki/LU_decomposition). 

---


## HPC Considerations
As this is a high-performance computing project, you should make sure that HPC is a prominent part of your project. As such, do not just implement the above processes in serial, or in the simplest HPC programming paradigm. Consider doing some of the following.
- Implement some of the processes in an HPC framework that demonstrates that you know the strengths and weaknesses of that hardware.
- Implement the process on multiple different HPC frameworks and compare things like the final performance, ease of implementation, and reusability/readability.
- Try to use libraries such as CUB or thrust
---

## Final Notes

This is a project, _not_ a homework. We do not have specific and strict rules for how you should go about your work. You still need to synthesize and augment the ideas presented in this document to produce your project proposal. Aspects that you might want to keep in mind:

1. You need to take the ideas presented here and produce a final project proposal
2. You need to demonstrate HPC skills learned in the class.
3. You should show the ability to apply class concepts to topics not covered in the class and/or take class concepts to a higher level of detail.
4. You should choose an amount of work appropriate to the number of people in your group. Recall that you have roughly a month to work on this, so it should be a good deal more commitment than a homework.
