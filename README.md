# Where-Comes-The-Name-Softmax-
Why it is named 'Softmax'?
When I was learning multiclass classifiers such as SVM and Neural Networks, "Softmax" came across to my mind with some mystery in its name. I was wondering why it was named so, and whether there was "Hardmax" function being its brother or even ancestor. I checked it out on Wikipedia but failed to find any useful information there (https://en.wikipedia.org/wiki/Softmax_function). There was the same question about Softmax's name on Quora and one of the answers, to my memory, revealed some part of the mystery. However, I cannot find that Quora link anymore.

Like most people, I continue to use it natually in my work without thinking about its origin. Engineers and researchers are mostly pragmatic, aren't they?

But recently when I was preparing for a few job interviews and Softmax drew my attention again (because I was afraid that an interviewer would ask me to explain Softmax function or Softmax classifier in details). I decided to figure out why it's got this funny name and whether it really stems from another related function nick-named "Hardmax" (or not, no chance to find it on Wikipedia, anyway).
Hardmax

Before elaborating Softmax, I just jump to the conclusion that there is "Hardmax" function, which is usually called Hinge Loss function used in linear classifiers such as SVM:

Li=∑j≠imax(0,sj−si+Δ)
Li=∑j≠imax(0,sj−si+Δ)

where sjsj and sisi are classification scores of the j-th and i-th element of the output vector of the model. And LiLi is the loss for classifying the input xixi as the i-th class.

Stanford CS class CS231n: Convolutional Neural Networks for Visual Recognition has a very good explanation of the above loss function. Please check it out here http://cs231n.github.io/linear-classify/#softmax.

And here is an example from it: %E5%9B%BE%E7%89%87.png

Wikipedia has Hinge Loss as well https://en.wikipedia.org/wiki/Hinge_loss.

Basically, hinge loss has a threshold ΔΔ below which the loss is perceived as zero. The threshold ΔΔ, which functions as a margin between the classification boundary (a.k.a. decision boundary) and the nearest samples, is applied to sjsj for all j≠ij≠i so that the loss of sjsj is added up to the overall loss of sisi only when sjsj has a difference from sisi smaller than the threshold.

Thus the hinge loss function has the form of max function max(0,x)max(0,x) and it is the threshold that makes the max "hard". We'll see this later on when we draw the graph of max function. Now here is an example of how hinge loss is calculated (from Stanford CS231n), in which i=0i=0, i.e., the ground truth label of the input pitcure is "cat", and Δ=10Δ=10:

Li=max(0,437.9−(−96.8)+10)+max(0,61.95−(−96.8)+10)
Li=max(0,437.9−(−96.8)+10)+max(0,61.95−(−96.8)+10)

Then we want to see how max function looks like if we draw a graph of it. I simplify the graph by using only integers for sjsj while fixing sisi to 0 and Δ=0Δ=0.
