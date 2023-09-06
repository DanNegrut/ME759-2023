Collection of ideas that might become 759 Final Projects.

- investigate how AMD support to program their GPUs ([see HIP](https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP-GUIDE.html), which stands for Heterogeneous-Computing Interface for Portability), compares to the support provided by NVIDIA (through CUDA) to program their GPUs.
- explore any analog of CUB for AMD GPUs, and how it compares with the NVIDIA CUB
- explore how easy it is to program AMD or Intel GPUs
- look into timing several algortihms that solve the same problem (like the exercise done for prefix scan, when two implementations were discussed). That is, start with a naive implementation for the solution of a problem and continue with more and more sophsiticated algorithms and/or implementations
- look into the concept of roofline-analysis for the CPU; it should be done in the context of a project related to your research work
- look into the concept of roofline-analysis for the GPU; it should be done in the context of a project related to your research work
- explore efficient sensor simulation with ML augmention on the GPU
- install and assess the release candidate for Thrust 2.0 ([see here](https://github.com/NVIDIA/thrust/releases)), the first major release probably in more than six years. Go through all the Thrust examples discussed in class and explain how the new Thrust implementation compares with the olf version of Thrust. Run scaling analysis to compare performance. Summarize API changes and how ease of use changed