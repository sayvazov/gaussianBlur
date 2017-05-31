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

//Use OpenCV Mat class to save char array to file
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

//Use OpenCV Mat class to save char array to file
void saveImage2(string name, char* A, int rows, int cols)
{
	std::cout << "Printing to " << name << endl;
	cv::Mat greyImage (cols, rows, cv::DataType<uchar>::type);
	for (int i = 0; i < cols; i++)
	{
		for (int j = 0; j < rows; j++)
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

// This is a Gaussian Vector, rather than matrix. Because the Gaussian 
// kernel is seperable, we can convolve in one direction independantly
// of the other. So we convolve, transpose, and convolve again. 
void gaussianVector ( float* array, int radius, float sigSquare)
{
	float normalizationFactor = 0.0; // To make up for not convolving the whole matrix
	for (int i = -1* radius; i <= radius ; i ++)
	{
        array[i+radius] = exp(-1*i*i/sigSquare);
        normalizationFactor += array[i+radius];
	}
	//Normalize, since the Gaussian is truncated, and we would like to integrate to 1
	for (int i = -1* radius; i <= radius ; i ++)
	{
		array[i+radius] /= normalizationFactor;
	}
}


//Convolve only horizontally, which means splitting the image into vertical 'Y' blocks
__global__ void convolveY(char* input, char* output, float* convolving, int width, int height, int radius, int diameter )
{
	//which thread, of all threads, is this
	int threadY = blockIdx.y * blockDim.y +threadIdx.y;
	// How big an area does each thread have to cover?
	int sizeOfYSection = height / (gridDim.y * blockDim.y);
	// if threadX < leftoverX, it takes on an extra thread
	int leftoverY = height % (gridDim.y * blockDim.y); 
	int startY = ( threadY) * sizeOfYSection + min (leftoverY, threadY );	
	// if threadX < leftoverX, this thread should do an extra element
	int endY = startY + sizeOfYSection + (leftoverY > threadY );

	float convolved = 0.f; //temp variable for result
	int locX, yPosition; //local X, global Y
	// Main Loop. Iterates over all its rows, convolving each with the 
    // Gaussian vector.
	for (int curY = startY; curY < endY; curY++)
	{
		for (int curX = 0; curX < width ; curX ++)
		{
            yPosition = curY* width;
			//Restart convolved
			convolved = 0.f;
            //Convolution loop
            for (int ofsetX = -1* radius ; ofsetX <= radius ; ofsetX++)
            {
                locX = min ( width-1, max(0, curX + ofsetX));
                convolved += input[yPosition + locX] * convolving[radius + ofsetX];
            }
			output[yPosition + curX] = (int)(convolved);
		}
	}
	return;
}


// CUDA transpose in blocks, with shared memory access.
// This is not my original code - it is a fairly ubiquotous 
// solution from an academic paper.  
__global__ void transpose(char* output, char* input)
{
  __shared__ char tile[16][16+1];
    
  int x = blockIdx.x * 16 + threadIdx.x;
  int y = blockIdx.y * 16 + threadIdx.y;
  int width = gridDim.x * 16;

  for (int j = 0; j < 16; j += 8){
      tile[threadIdx.y+j][threadIdx.x] = input[(y+j)*width + x];
  }

  __syncthreads(); 

  x = blockIdx.y * 16 + threadIdx.x;  // transpose block offset
  y = blockIdx.x * 16 + threadIdx.y;

  for (int j = 0; j < 16; j += 8){
      output[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
  }
}

__global__ void transposeNaive(char*output, char*input, int width, int height)
{	//which thread, of all threads, is this
	int threadX =  blockIdx.x * blockDim.x + threadIdx.x;
	int threadY = blockIdx.y * blockDim.y + threadIdx.y;

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

    // Actually do transpose
    for (int i = startY ; i < endY ; i ++)
    {
        for (int j = startX; j < endX ; j++)
        {
            output[i* width + j] = input[j*width + i];
            //printf("Copied %d, %d to %d, %d moving %d\n", i, j, j, i, input[j*width + i]);
        }
    }
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

	//load image data to matrix
	Mat image;
	image = imread(filename, 0);
	
	int imgWidth = image.cols;
	int imgHeight =image.rows;

	//Initialize variables and cuda arrays
	float sigSquare = 2 * sigma * sigma;
	int blurDiameter = 2* blurRadius + 1;
	int imgSize = imgWidth * imgHeight;
	char input[imgSize], output[imgSize];
	float gaussian[blurDiameter];
	char *deviceInput, *deviceTranspose,
    *deviceOutput, *deviceOutpose;
	float* deviceGaussian;

    //This is a lot of memory. Essentially, because the shapes 
    // of the blocks/threads are different, they need to have separate
    // memory structures, so transposes and convolutions have to 
    // pass through global memory. 
	cudaMalloc( (void**) &deviceInput, imgSize * sizeof(char) );
	cudaMalloc( (void**) &deviceOutput, imgSize * sizeof(char) );
	cudaMalloc( (void**) &deviceGaussian, blurDiameter *sizeof(float) );
    cudaMalloc( (void**) &deviceTranspose, imgSize * sizeof(char) );
	cudaMalloc( (void**) &deviceOutpose, imgSize * sizeof(char) );

	// Copy from OpenCV matrix to regular old char array
	readImage(image, input, imgHeight, imgWidth);

    //This algorithm is destructive to the input vector, so we save the 
    //input grayscal image before starting, rather than after. 
    saveImage("input1d.png", input, imgHeight, imgWidth);

    //Bbegin timing
	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventRecord(start,0);


	// Generate Gaussian Vector. Since we use square convolutions, 
    // this is the same for both stages
	gaussianVector(gaussian, blurRadius, sigSquare);
    //Copy input and gaussian to device. Device does not need all of the input matrix
	// but some testing shows that copying the matrix accounts for less than 1%
	// of runtime
	cudaMemcpy(deviceInput, input, imgSize*sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceGaussian, gaussian, blurDiameter*sizeof(float), cudaMemcpyHostToDevice);

    //Convolution uses only Y blocks
	dim3 threads(1, 128); 
	dim3 grid(1, 32);
    //Transposition uses both
    dim3 tThreads(16, 16);
    dim3 tGrid ( 8, 8);

	convolveY <<<grid, threads>>>(deviceInput, deviceOutput, deviceGaussian, imgWidth, imgHeight, blurRadius, blurDiameter);
	cudaDeviceSynchronize(); 

    cudaMemcpy(input, deviceOutput, imgSize*sizeof(char), cudaMemcpyDeviceToHost);
    cudaMemcpy(deviceTranspose, input, imgSize*sizeof(char), cudaMemcpyHostToDevice);

    transposeNaive <<<tGrid, tThreads>>>( deviceOutpose, deviceTranspose, imgWidth, imgHeight);
    cudaMemcpy(output, deviceOutpose, imgSize*sizeof(char), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaMemcpy(deviceInput, output, imgSize*sizeof(char), cudaMemcpyHostToDevice);
    convolveY <<<grid, threads>>>(deviceInput, deviceOutput, deviceGaussian, imgHeight, imgWidth, blurRadius, blurDiameter);
    cudaDeviceSynchronize();

    cudaMemcpy(output, deviceOutput, imgSize*sizeof(char), cudaMemcpyDeviceToHost);
    cudaMemcpy(deviceTranspose, output, imgSize*sizeof(char), cudaMemcpyHostToDevice);

    transposeNaive <<<tGrid, tThreads>>>( deviceOutpose, deviceTranspose, imgWidth, imgHeight);
    cudaMemcpy(output, deviceOutpose, imgSize*sizeof(char), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    

	cudaEventCreate(&stop);
 	cudaEventRecord(stop,0);
 	//cudaEventSynchronize(stop);
        cout << "Got here" << endl;

 	cudaEventElapsedTime(&elapsedTime, start,stop);
 	printf("Elapsed time : %f ms\n" ,elapsedTime);
	
	saveImage("output1d.png", output, imgHeight, imgWidth);

	//cout << "Result " << endl;
	//disp (output, imgWidth, imgHeight);

	


}