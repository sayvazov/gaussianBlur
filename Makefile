makeall: 
	nvcc gaussian1dBlur.cu `pkg-config --libs opencv` -o blur1
 
	nvcc gaussianBlur.cu `pkg-config --libs opencv` -o blur
 