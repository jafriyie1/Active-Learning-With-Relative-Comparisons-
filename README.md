# Active-Learning-With-Relative-Comparisons-
This repository contains the code, data, and models for this research.

Labeling new instances when lables are not provided for all data points is difficult. In order to circumvent this problem an algorithm 
was created by Fern et al. [1] that provides a means of assigning labels onto data points by using relative comparisons. Relative 
comparisons, given a triple of data points xi, xj, xk are way of finding similarities amongst data points; moreover, this gives 
us the ability to utilize the constraint of xi is more similar to xj than xk. This can be done by using a Support Vector Machine using 
and Euclidean distance, clustering, and a Random Forest for building a distribution of predicted labels [2,3].
We then use this to find a means of maximizing the surprisal,or information of a triple to use to label new instances. This is a powerful method that will aid in our research efforts for creating better
search algorithms.

The following papers are used in this research (more will be added)

[1] Xiong, S., Rosales, R., Pei, Y., & Fern, X. Z. (2014). 
Active Metric Learning from Relative Comparisons. arXiv preprint arXiv:1409.4155.

[2] Schultz, M., & Joachims, T. (2004). 
Learning a distance metric from relative comparisons. 
In Advances in neural information processing systems (pp. 41-48).

[3]Tsang, I. W., Kwok, J. T., Bay, C., & Kong, H. (2003, June). Distance metric learning with kernels. 
In Proceedings of the International Conference on Artificial Neural Networks (pp. 126-129).

[4] A. Frank and A. Asuncion. UCI machine learning repository,

