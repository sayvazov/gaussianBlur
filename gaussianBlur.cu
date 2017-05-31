#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cuda.h>
#include <typeinfo>

using namespace std;
using namespace cv;


//Functions to print out char arrays for testing
void disp(char* data, int cols, int rows)
{
	for ( int i = 0 ; i < rows; i++)
	{
		for (int j = 0 ; j < cols; j++)
		{
			cout << (int)data[i*cols + j] << " ";
		}
		cout << endl;
	}
}

void disp(float* data, int cols, int rows)
{
	for ( int i = 0 ; i < rows; i++)
	{
		for (int j = 0 ; j < cols; j++)
		{
			cout << data[i*cols + j] << " ";
		}
		cout << endl;
	}
}

//Use OpenCV Mat class to save char array to file as greyscale image
void saveImage(string name, char* A, int rows, int cols)
{
	std::cout << "Printing to " << name << endl;
	cv::Mat greyImage (rows, cols, cv::DataType<uchar>::type);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			greyImage.at<uchar>(i, j) = min(A[i*cols + j]*2, 255);
		}
	}
	cv::imwrite(name, greyImage);
}

//Convert data from OpenCV Mat class to char array
void readImage(Mat image, char* A, int rows, int cols)
{
	std::cout << "Initializing array" << endl;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			A[i*cols + j] = image.at<uchar>(i, j)/2;
		}
	}
}

// The convolution matrix is identical, so we generate it once and pass through
// cudamemcpy to each vertex
void gaussianMatrix2d ( float* array, int radius, float sigSquare)
{
	int diameter = radius*2 + 1;
	float normalizationFactor = 0.0; // To make up for not convolving the whole matrix
	for (int i = -1* radius; i <= radius ; i ++)
	{
		for (int j = -1* radius ; j <= radius; j++)
		{
			array[(i + radius)* diameter + j + radius] = exp(-1.0*(i*i + j*j) / sigSquare);
			normalizationFactor += array[(i + radius)* diameter + j + radius];
		}
	}
	//Normalize, since the Gaussian is truncated, and we would like to integrate to 1
	for (int i = -1* radius; i <= radius ; i ++)
	{
		for (int j = -1* radius ; j <= radius; j++)
		{
			array[(i + radius)* diameter + j + radius] /= normalizationFactor;
		}
	}
}


//Main convolution function
__global__ void convolve2d(char* input, char* output, float* convolving, int width, int height, int radius, int diameter )
{
	//Which block are we in
	int blockX = blockIdx.x;
	int blockY = blockIdx.y;

	//which thread, of all threads, is this
	int threadX = blockX * blockDim.x + threadIdx.x;
	int threadY = blockY * blockDim.y + threadIdx.y;

	// How big an area does each thread have to cover?
	int sizeOfXSection = width  / (gridDim.x * blockDim.x); 
	int sizeOfYSection = height / (gridDim.y * blockDim.y);

	// if threadX < leftoverX, it takes on an extra thread
	int leftoverX = width  % (gridDim.x * blockDim.x); 
	int leftoverY = height % (gridDim.y * blockDim.y); 

	int startX = ( threadX) * sizeOfXSection + min (leftoverX, threadX );
	int startY = ( threadY) * sizeOfYSection + min (leftoverY, threadY );
	
	// if threadX < leftoverX, this thread should do an extra element
	int endX = startX + sizeOfXSection + (leftoverX > threadX );
	int endY = startY + sizeOfYSection + (leftoverY > threadY );

	float convolved = 0.f; // temp variable for result
	int locX, locY; // Local x and y
	
	// Main loop. Iterates over all pixels in its domain, and then convolves th
	// Gaussian matrix with the submatrix around the pixel.
	for (int curY = startY; curY < endY; curY++)
	{
		for (int curX = startX; curX < endX ; curX ++)
		{
			// Restart summing the convolved, iterate over all elements of submatrix
			convolved = 0.f;
			for (int ofsetY = -1* radius ; ofsetY <= radius; ofsetY++)
			{
				for (int ofsetX = -1* radius ; ofsetX <= radius ; ofsetX++)
				{
					locX = min ( width-1, max(0, curX + ofsetX));
					locY = min ( height-1,max(0, curY + ofsetY));
					convolved += input[locY* width + locX] * convolving[(ofsetY+radius)*diameter + radius + ofsetX];
				}
			}
			output[curY*width + curX] = (int)(convolved);
		}
	}
	return;

}

int main (int argc, char** argv)
{
	//parameters 
	int blurRadius = 5;
	float sigma = 1.0;
	string filename;
	if ( argc < 2){
		cout << "Please specify image file to work with"<<endl;
		return -1;
	}
	else {
		filename = argv[1];
	}
	if (argc > 2){
		blurRadius = atoi(argv[2]);
	}
	if (argc > 3){
		sigma = atoi(argv[3]);
	}

	//load image data to OpenCV matrix
	Mat image;
	image = imread(filename, 0);
	
	int imgWidth = image.cols;
	int imgHeight =image.rows;

	//Initialize variables and cuda arrays
	float sigSquare = 2 * sigma * sigma;
	int blurDiameter = 2* blurRadius + 1;
	int blurSize = blurDiameter* blurDiameter;
	int imgSize = imgWidth * imgHeight;

	char input[imgSize], output[imgSize];
	float gaussian[blurSize];
	char* deviceInput, *deviceOutput;
	float* deviceGaussian;

	cudaMalloc( (void**) &deviceInput, imgSize * sizeof(char) );
	cudaMalloc( (void**) &deviceOutput, imgSize * sizeof(char) );
	cudaMalloc( (void**) &deviceGaussian, blurSize *sizeof(float) );

	// Copy from OpenCV matrix to regular old char array
	readImage(image, input, imgHeight, imgWidth);

	//Start timing
	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventRecord(start,0);

	//Generate Gaussian matrix
	gaussianMatrix2d(gaussian, blurRadius, sigSquare);

	//Copy input and gaussian to device. Device does not need all of the input matrix
	// but some testing shows that copying the matrix accounts for less than 1%
	// of runtime
	cudaMemcpy(deviceInput, input, imgSize*sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceGaussian, gaussian, blurSize*sizeof(float), cudaMemcpyHostToDevice);
	// Mostly arbitrary numbers.
	dim3 threads(16, 16); 
	dim3 grid(8, 8);

	//Actually do the convolution
	convolve2d <<<grid, threads>>>(deviceInput, deviceOutput, deviceGaussian, imgWidth, imgHeight, blurRadius, blurDiameter);
	
	//Wait for all threads to end, then push output back to Host
	cudaDeviceSynchronize(); 
	cudaMemcpy(output, deviceOutput, imgSize*sizeof(char), cudaMemcpyDeviceToHost);
	
	//Stop timing and report
	cudaEventCreate(&stop);
 	cudaEventRecord(stop,0);
 	cudaEventSynchronize(stop);
 	cudaEventElapsedTime(&elapsedTime, start,stop);
 	printf("Elapsed time : %f ms\n" ,elapsedTime);
	
	//Export images 
	saveImage("input.png", input, imgHeight, imgWidth);
	saveImage("output.png", output, imgHeight, imgWidth);
	return 1;
}