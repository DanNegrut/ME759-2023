# Image Processing

Processing high-resolution images is a compute-heavy operation, often requiring high performance desktops to do professional editing. Below are some common operations in image processing which you may consider using in a project proposal.

## Topics
### Applying a Filter
Filters (i.e., 2D convolutions) sweep across an image and act on each pixel combining their values with those of their neighbors in order to produce a desired effect. [This page](https://en.wikipedia.org/wiki/Kernel_(image_processing)) has some common filters.

### Expressing Edges
Given an image which has already undergone an edge detection process to identify edge pixels (e. g., [Canny](https://en.wikipedia.org/wiki/Canny_edge_detector) and [Sobel](https://en.wikipedia.org/wiki/Sobel_operator)), it is often useful to have an algebraic expression of the lines which define edges. In order to get equations for these lines, one can apply a [Hough Transform](http://web.ipac.caltech.edu/staff/fmasci/home/astro_refs/HoughTrans_lines_09.pdf). Essentially, for each identified edge pixel, the transform takes account for each possible line that could go through that point. Once each pixel has been covered, lines that have occurred the most are identified. From there, these lines can be
- reported and used in further image processing algorithms
- drawn over the original image
- bounded between two pixels to produce a line segment instead of the full line

### Image Stitching
It can also be useful to stitch multiple images that depict overlapping portions of the same scene into a single image. This requires steps to detect features common between each image and to correctly align them together. See [here](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform#Panorama_stitching) for a starting point.

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
