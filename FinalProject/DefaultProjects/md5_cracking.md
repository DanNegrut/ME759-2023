# MD5 Hash Attack

## Background

Human-generated computer passwords are known to have a number of flaws that make them vulnerable to attack by malicious hackers. To avoid this problem, many domains enact requirements designed to increase how complex, and therefore decrease how guessable, user passwords are.

With such requirements in place, the only viable methods left to deduce these passwords become brute-force attacks: computational _guess-and-check_. Attackers must calculate the hashes of each possible password until they find one that matches a target hash.

This project outlines one such attack, which can be run in a reasonable amount of time on a GPU, even on relatively inexpensive consumer-grade hardware.


## Starting Point

This project is based on a CUDA kernel, in which you are expected to implement an MD5 hash calculator.

The algorithm for MD5 hashing is well documented and available on common reference sources such as [Wikipedia](https://en.wikipedia.org/wiki/MD5). Likewise, a reference implementation in C is publicly available as a part of the IETF documentation on [RFC 1321](https://tools.ietf.org/html/rfc1321). Please note that while there may be some necessary similarities between implementations, your implementation should be original to you. It will be compared to several existing implementations by means of MOSS and human verification.

Your grader will test your implementation against a 7-character alphanumeric (a-z, A-Z, 0-9) password, so a minimal implementation will be able to handle at least that much.

---

## Important Points

### HPC Methodology
Your implementation should make effective use of the features of the CUDA ecosystem. The primary focus should be to achieve efficient data parallelism by leveraging features of the GPU architecture.

### Discussion
This is an HPC project, so you should be able to provide critical reasoning as to why you chose the implementation that you did. Aside from that, the following topics might help you in drafting a proposal:

Consider the strengths and weaknesses of your implementation, especially in terms of how it performs against passwords of various length and complexity.

A well-thought-out analysis might also include some discussion of how the mixing of HPC technology and hash attacks such as this one impacts industries on a larger scale.

Consider how one might mitigate against this type of attack.

### Further Considerations

This document is **not** a homework assignment, and it is **not** a complete project. As such, there is quite a lot of flexibility in what you can do with it. When drafting your project proposal, think about ways in which you can adapt or extend what is presented here to create an original and interesting project. 

NOTE: It is important for the scope of your project to be commensurate with the size of your group. If your project is too limited or underdeveloped, your proposal may not be accepted.

---

### Disclaimer
MD5 has been superseded in cryptographic use by other algorithms due to its vulnerability to brute-force attacks and relatively high collision rate. Additionally, practices such as [salting](https://en.wikipedia.org/wiki/Salt_\(cryptography\)) and the use of more complex passwords have made the relevant attack a bit less effective. Still, your implementation should only be used for academic purposes and should _never_ be used to engage in any illegal or otherwise dubious activity of a nefarious nature.
