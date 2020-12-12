#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/random.h>
#include "device_launch_parameters.h"
#include <vector>
#include <random>
#include <time.h>
#include <Windows.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb\stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_MSC_SECURE_CRT
#include <stb\stb_image_write.h>

using namespace std;

#define HOST_DEVICE __host__ __device__

// Utility
float Clamp(float V, float Min, float Max)
{
	if (V < Min) V = Min;
	if (V > Max) V = Max;
	return V;
}

bool SavePng(const char* Filename, int Width, int Height, float* Input){
	unsigned char* Transform = new unsigned char[Width*Height * 3];
	for (int i = 0; i < Height; ++i){
		for (int j = 0; j < Width; ++j){
			unsigned Pixel = i*Width + j;
			unsigned C = unsigned(Clamp(Input[Pixel], 0.f, 1.f)*255.f);
			Transform[3 * Pixel] = C;
			Transform[3 * Pixel + 1] = C;
			Transform[3 * Pixel + 2] = C;
		}
	}

	stbi_write_png(Filename, Width, Height, 3, Transform, 0);

	delete[] Transform;

	return true;
}

class Timer {
private:
	LARGE_INTEGER start;
	LARGE_INTEGER end;

public:
	Timer()
		:start()
		, end(){}

	void Start()
	{
		QueryPerformanceCounter(&start);
	}

	void End()
	{
		QueryPerformanceCounter(&end);
	}

	float GetElapsed() const
	{
		LARGE_INTEGER freq;
		//frequency per second
		QueryPerformanceFrequency(&freq);

		return ((float)(end.QuadPart - start.QuadPart)) / freq.QuadPart;
	}
};
//

// ---------------------White Noise----------------------
static __forceinline HOST_DEVICE unsigned int WangHash(unsigned int seed)
{
	seed = (seed ^ 61) ^ (seed >> 16);
	seed = seed + (seed << 3);
	seed = seed ^ (seed >> 4);
	seed = seed * 0x27d4eb2d;
	seed = seed ^ (seed >> 15);

	return seed;
}

__global__ void WhiteNoiseGenerator(float* Image)
{
	int X = blockIdx.x*blockDim.x + threadIdx.x;
	int Y = blockIdx.y*blockDim.y + threadIdx.y;
	int Pixel = Y*gridDim.x*blockDim.x + X;

	thrust::default_random_engine Rng(WangHash(Pixel));
	thrust::uniform_real_distribution<float> Uniform(0.f, 1.f);

	Image[Pixel] = Uniform(Rng);
}

void WhiteNoiseGenerator(int Width, int Height, float* Output)
{
	float* Image;
	cudaMalloc(&Image, Width*Height*sizeof(float));

	dim3 BlockSize(32, 32);
	dim3 GridSize(Width / BlockSize.x, Height / BlockSize.y);
	WhiteNoiseGenerator << <GridSize, BlockSize >> >(Image);

	cudaMemcpy(Output, Image, Width*Height*sizeof(float), cudaMemcpyDeviceToHost);
}
// ---------------------White Noise----------------------

// ----------------------Blue Noise----------------------
HOST_DEVICE float ToroidalDistanceSq(float X1, float Y1, float X2, float Y2, float Width, float Height)
{
	float Dx = abs(X2 - X1);
	float Dy = abs(Y2 - Y1);

	if (Dx > Width * 0.5f)
		Dx = Width - Dx;

	if (Dy > Height * 0.5f)
		Dy = Height - Dy;

	return Dx*Dx + Dy*Dy;
}

__global__ void Init(int Width, int Height, float* EnergyLut)
{
	int X = blockIdx.x*blockDim.x + threadIdx.x;
	int Y = blockIdx.y*blockDim.y + threadIdx.y;
	int Pixel = Y*Width + X;

	EnergyLut[Pixel] = 0.f;
}

__global__ void Normalize(int Width, int Height, float* Image)
{
	int X = blockIdx.x*blockDim.x + threadIdx.x;
	int Y = blockIdx.y*blockDim.y + threadIdx.y;
	int Pixel = Y*Width + X;

	Image[Pixel] /= (Width*Height);
}

__global__ void UpdateEnergyLut(int SX, int SY, int Width, int Height, float* EnergyLut)
{
	int X = blockIdx.x*blockDim.x + threadIdx.x;
	int Y = blockIdx.y*blockDim.y + threadIdx.y;
	int Pixel = Y*Width + X;

	//Calc Distance
	float DistanceSq = ToroidalDistanceSq(X, Y, SX, SY, Width, Height);

	float SigmaSquareTimesTwo = 2.f*1.9f*1.9f;
	EnergyLut[Pixel] += exp(-DistanceSq / SigmaSquareTimesTwo);
}

const int BlockWidth = 32;
__device__ int BestXY[2];
__shared__ float MinVec[BlockWidth*BlockWidth];
__shared__ int PixelIndex[BlockWidth*BlockWidth];
__global__ void UpdateEnergyLut(int Width, int Height, float* EnergyLut)
{
	int X = blockIdx.x*blockDim.x + threadIdx.x;
	int Y = blockIdx.y*blockDim.y + threadIdx.y;
	int Pixel = Y*Width + X;

	//Calc Distance
	float DistanceSq = ToroidalDistanceSq(X, Y, BestXY[0], BestXY[1], Width, Height);

	float SigmaSquareTimesTwo = 2.f*1.9f*1.9f;
	EnergyLut[Pixel] += exp(-DistanceSq / SigmaSquareTimesTwo);
}

__global__ void FindLargestVoid(int Width, int Height, int* BinaryPattern, float* EnergyLut, float* Image, int CurrentIndex)
{
	const int NumProcesses = Width*Height / (BlockWidth*BlockWidth);
	const int ThreadIdx = threadIdx.y*blockDim.x + threadIdx.x;

	float MinF = INFINITY;
	for (int Index = 0; Index < NumProcesses; ++Index)
	{
		float V = EnergyLut[ThreadIdx*NumProcesses + Index];
		if (V < MinF && !BinaryPattern[ThreadIdx*NumProcesses + Index])
		{
			PixelIndex[ThreadIdx] = ThreadIdx*NumProcesses + Index;
			MinF = V;
		}
	}
	MinVec[ThreadIdx] = MinF;
	__syncthreads();

	// Parallel Reduction Min Begin
	auto MaySwapVolatile = [&](volatile int* Indices, volatile float* Vec, int Gap)
	{
		if (Vec[ThreadIdx] > Vec[ThreadIdx + Gap] && !BinaryPattern[Indices[ThreadIdx + Gap]])
		{
			Vec[ThreadIdx] = Vec[ThreadIdx + Gap];
			Indices[ThreadIdx] = Indices[ThreadIdx + Gap];
		}
	};

	auto MaySwap = [&](int* Indices, float* Vec, int Gap)
	{
		if (Vec[ThreadIdx] > Vec[ThreadIdx + Gap] && !BinaryPattern[Indices[ThreadIdx + Gap]])
		{
			Vec[ThreadIdx] = Vec[ThreadIdx + Gap];
			Indices[ThreadIdx] = Indices[ThreadIdx + Gap];
		}
	};

	auto WarpReductionMin = [&](volatile int* Indices, volatile float* Vec)
	{
		MaySwapVolatile(Indices, Vec, 32);
		if (ThreadIdx < 16)	MaySwapVolatile(Indices, Vec, 16);
		if (ThreadIdx < 8)	MaySwapVolatile(Indices, Vec, 8);
		if (ThreadIdx < 4)	MaySwapVolatile(Indices, Vec, 4);
		if (ThreadIdx < 2)	MaySwapVolatile(Indices, Vec, 2);
		if (ThreadIdx < 1)	MaySwapVolatile(Indices, Vec, 1);
	};

	auto ParallelReductionMin = [&](int* Indices, float* Vec)
	{
		if (ThreadIdx < 512) MaySwap(Indices, Vec, 512);
		__syncthreads();
		if (ThreadIdx < 256) MaySwap(Indices, Vec, 256);
		__syncthreads();
		if (ThreadIdx < 128) MaySwap(Indices, Vec, 128);
		__syncthreads();
		if (ThreadIdx < 64) MaySwap(Indices, Vec, 64);
		__syncthreads();
		if (ThreadIdx < 32) WarpReductionMin(Indices, Vec);
	};
	// Parallel Reduction Min End
	ParallelReductionMin(PixelIndex, MinVec);
	if (ThreadIdx == 0)
	{
		BestXY[0] = PixelIndex[0] % Width;
		BestXY[1] = PixelIndex[0] / Width;
		BinaryPattern[PixelIndex[0]] = 1;

		Image[PixelIndex[0]] = CurrentIndex;
	}
}

//Mitchell Best Candiate Algorithm
void MitchellBestCandiate(const int N, const int Width, const int Height, int* BinaryPattern, float* Out)
{
	struct Vec2f
	{
		float X, Y;

		Vec2f(float X, float Y) :X(X), Y(Y){}
	};
	srand(time(nullptr));
	auto Generate2DNoise = [&]()->Vec2f
	{
		float X = float(rand()) / RAND_MAX;
		float Y = float(rand()) / RAND_MAX;
		return Vec2f(X, Y);
	};

	int CurrentIndex = 0;
	vector<Vec2f> Samples;
	if (CurrentIndex == 0)
	{
		Samples.push_back(Generate2DNoise());
		CurrentIndex = 1;
	}

	while (CurrentIndex < N)
	{
		vector<Vec2f> Candidates;
		int NumCandidates = Samples.size() + 1;
		for (int Index = 0; Index < NumCandidates; ++Index)
		{
			Candidates.push_back(Generate2DNoise());
		}

		float MaxDistance = 0.f;
		int BestIndex = -1;
		for (int Index = 0; Index < NumCandidates; ++Index)
		{
			Vec2f Candiate = Candidates[Index];
			float Distance = INFINITY;
			for (int S = 0; S < Samples.size(); ++S)
			{
				Vec2f Sample = Samples[S];

				float D = ToroidalDistanceSq(Sample.X, Sample.Y, Candiate.X, Candiate.Y, 1.f, 1.f);
				if (D < Distance) Distance = D;
			}

			if (Distance > MaxDistance)
			{
				MaxDistance = Distance;
				BestIndex = Index;
			}
		}

		Samples.push_back(Candidates[BestIndex]);

		CurrentIndex++;

		printf("Mitchells Best Candiates : %f%%\r", 100.f * float(CurrentIndex) / N);
	}
	printf("\n");

	for (int Index = 0; Index < Samples.size(); ++Index)
	{
		Vec2f Sample = Samples[Index];
		int X = floor(Sample.X*Width); X = X == Width ? X - 1 : X;
		int Y = floor(Sample.Y*Height); Y = Y == Height ? Y - 1 : Y;
		Out[Y*Height + X] = Index + 1;
		BinaryPattern[Y*Height + X] = 1;
	}
}

//Void And Cluster Algorithm
//http://cv.ulichney.com/papers/1993-void-cluster.pdf
void BlueNoiseGenerator(int Width, int Height, float* Image)
{
	vector<int> BinaryPattern;
	BinaryPattern.resize(Width*Height);
	memset(&BinaryPattern[0], 0, Width*Height*sizeof(int));

	int nSamples = 256;
	MitchellBestCandiate(nSamples, Width, Height, &BinaryPattern[0], Image);
	int CurrentIndex = nSamples;

	float* EnergyLut, *ImageDevice;
	cudaMalloc(&EnergyLut, Width*Height*sizeof(float));
	int* BinaryPatternDevice;
	cudaMalloc(&BinaryPatternDevice, Width*Height*sizeof(int));
	cudaMemcpy(BinaryPatternDevice, &BinaryPattern[0], Width*Height*sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc(&ImageDevice, Width*Height*sizeof(float));
	cudaMemcpy(ImageDevice, Image, Width*Height*sizeof(float), cudaMemcpyHostToDevice);

	dim3 BlockSize(BlockWidth, BlockWidth);
	dim3 GridSize(Width / BlockSize.x, Height / BlockSize.y);
	Init << <GridSize, BlockSize >> >(Width, Height, EnergyLut);
	for (int Index = 0; Index < Width*Height; ++Index)
	{
		if (!BinaryPattern[Index]) continue;

		int X = Index % Width;
		int Y = Index / Width;
		UpdateEnergyLut << <GridSize, BlockSize >> >(X, Y, Width, Height, EnergyLut);
	}

	while (CurrentIndex < Width*Height)
	{
		CurrentIndex++;

		dim3 SpecialGridSize(1, 1);
		FindLargestVoid << <SpecialGridSize, BlockSize >> >(Width, Height, BinaryPatternDevice, EnergyLut, ImageDevice, CurrentIndex);

		UpdateEnergyLut << <GridSize, BlockSize >> >(Width, Height, EnergyLut);

		if (CurrentIndex % 10 == 0)
			printf("Phase2 And Phase3 : %f%%\r", CurrentIndex * 100.f / (Width*Height));
	}

	printf("Phase2 And Phase3 : %f%%\r", 100.f);

	// normalize
	Normalize << <GridSize, BlockSize >> >(Width, Height, ImageDevice);
	cudaMemcpy(Image, ImageDevice, Width*Height*sizeof(float), cudaMemcpyDeviceToHost);

	// free resources
	cudaFree(EnergyLut);
	cudaFree(ImageDevice);
	cudaFree(BinaryPatternDevice);
}
// ----------------------Blue Noise----------------------

void PrintHelp()
{
	printf("******************************************************************\n");
	printf(R"(Options:
--size <num>          Size of blue or white noise texture need be generated
--output <filename>   Output filename of blue or white noise texture
--type <num>          Which noise do you want to generate (0: white, 1: blue)
--help                Print help
)");
	printf("******************************************************************\n");
}

int main(int argc, char** argv)
{
	PrintHelp();

	string Filename = "D:/Noise.png";
	int Width = 64, Height = 64;
	int Type = 1;
	for (int Index = 1; Index < argc; ++Index)
	{
		if (!strcmp(argv[Index], "--size"))
		{
			Width = Height = atoi(argv[++Index]);
		}
		else if (!strcmp(argv[Index], "--output"))
		{
			Filename = argv[++Index];
		}
		else if (!strcmp(argv[Index], "--type"))
		{
			Type = atoi(argv[++Index]);
		}
		else if (!strcmp(argv[Index], "--help"))
		{
			PrintHelp();
			return 0;
		}
		else
		{
			printf("Unexpected parameters\n");
			return 1;
		}
	}	

	printf("Will generate %snoise with size[%dx%d] and output file[%s]\n", Type == 0 ? "white" : "blue", Width, Height, Filename.c_str());

	float *Output;
	Output = new float[Width*Height];
	for (int Index = 0; Index < Width*Height; ++Index) Output[Index] = 0.f;

	Timer Record;
	Record.Start();
	if (Type == 0)
	{
		WhiteNoiseGenerator(Width, Height, Output);
	}
	else
	{
		BlueNoiseGenerator(Width, Height, Output);
	}
	Record.End();

	SavePng(Filename.c_str() , Width, Width, Output);

	printf("\nElapsed Time : %f\n", Record.GetElapsed());
	delete[] Output;

	return 0;
}