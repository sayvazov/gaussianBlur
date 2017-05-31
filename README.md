# gaussianBlur

Code is broken into two working pieces. The first, built from gaussianBlur.cu into blur, is a 2-dimensional tiled convolution. The second, from gaussian1dBlur.cu into blur1, makes use of the separability of the Gaussian kernel to achieve much better performance. Essentially, it does a horizontal GaussianBlur, a transpose, and then another horizontal blur, followed by another transpose. However, it is more memory intensive, as these two pieces are optimized by different block structures. 
