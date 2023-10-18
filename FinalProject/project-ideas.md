Collection of ideas that might become 759 Final Projects.

- investigate how AMD support to program their GPUs ([see HIP](https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP-GUIDE.html), which stands for Heterogeneous-Computing Interface for Portability), compares to the support provided by NVIDIA (through CUDA) to program their GPUs.
- explore and learn more about CUB
- explore any analog of CUB for AMD GPUs, and how it compares with the NVIDIA CUB
- explore how easy it is to program AMD or Intel GPUs
- look into timing several algortihms that solve the same problem (like the exercise done for prefix scan, when two implementations were discussed). That is, start with a naive implementation for the solution of a problem and continue with more and more sophsiticated algorithms and/or implementations
- look into the concept of roofline-analysis for the CPU; it should be done in the context of a project related to your research work
- look into the concept of roofline-analysis for the GPU; it should be done in the context of a project related to your research work
- explore efficient sensor simulation with ML augmention on the GPU
- install [preferably] on your home machine and assess the latest release of the Thrust library ([see here](https://github.com/NVIDIA/thrust/releases)). Go through all the Thrust examples discussed in class and explain how the new Thrust implementation compares with the old version of Thrust. Run scaling analysis to compare performance. Summarize API changes and how ease of use has changed relative to what was discussed in class
- learn more about just-in-time compiling, which attempts to convert your CUDA code to a best set of instructions on the fly, to leverage the underlying compute capability (that is, hardware generation you run on)
- look into using Tensor Core for some computing in FEA or CFD; compare the performance of your solution method on the CPU and then on the Tensor Core-accelerated GPU

