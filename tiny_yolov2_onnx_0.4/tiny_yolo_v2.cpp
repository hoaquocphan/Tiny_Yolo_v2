#include <assert.h>
#include <onnxruntime_c_api.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <map>
#include <fstream>
#include <sys/time.h>
#include <iostream>
#include <numeric>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

/*****************************************
* Includes
******************************************/
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <errno.h>
#include <jpeglib.h>
#include <termios.h>
#include <math.h>
#include <iomanip>
#include <sstream>

#include "box.h"
#include "define.h"
#include "image.h"

/*****************************************
* Global Variables
******************************************/
std::map<int,std::string> label_file_map;
char inference_mode = DETECTION;
char* save_filename = "output.jpg";
int model=0;

double anchors[] = {
	1.08, 	1.19,
	3.42, 	4.41,
	6.63, 	11.38,
	9.42, 	5.11,
	16.62, 	10.52
};


/*****************************************
* Function Name	:  loadLabelFile
* Description		: Load txt file
* Arguments			:
* Return value	:
******************************************/
int loadLabelFile(std::string label_file_name)
{
    int counter = 0;
    std::ifstream infile(label_file_name);

    if (!infile.is_open())
    {
        perror("error while opening file");
        return -1;
    }

    std::string line;
    while(std::getline(infile,line))
    {
        label_file_map[counter++] = line;
    }

    if (infile.bad())
    {
        perror("error while reading file");
        return -1;
    }

    return 0;
}

/*****************************************
* Function Name	: sigmoid
* Description	: helper function for YOLO Post Processing
* Arguments	:
* Return value	:
******************************************/
double sigmoid(double x){
	return 1.0/(1.0+exp(-x));
}

/*****************************************
* Function Name	: softmax
* Description	: helper function for YOLO Post Processing
* Arguments	:
* Return value	:
******************************************/
void softmax(float val[]){
	float max = -INT_MAX;
	float sum = 0;

	for (int i = 0;i<20;i++){
		max = std::max(max, val[i]);
	}

	for (int i = 0;i<20;i++){
		val[i]= (float) exp(val[i]-max);
		sum+= val[i];
	}

	for (int i = 0;i<20;i++){
		val[i]= val[i]/sum;
	}

	// printf("Softmax: max %f sum %f\n", max, sum);
}

/*****************************************
* Function Name	: timedifference_msec
* Description	:
* Arguments	:
* Return value	:
******************************************/
static double timedifference_msec(struct timeval t0, struct timeval t1)
{
    return (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_usec - t0.tv_usec) / 1000.0;
}


/*****************************************
* Function Name	:offset
* Description	: c
* Arguments	:
* Return value	:
******************************************/
int offset(int o,int channel){
	return 	o+ channel*YOLO_GRID_X*YOLO_GRID_Y;
}

/*****************************************
* Function Name	:offset
* Description		: c
* Arguments			:
* Return value	:
******************************************/
int offset_(int b, int y, int x){
	return b*(20+5)* YOLO_GRID_X * YOLO_GRID_Y + y * YOLO_GRID_X + x;
}

/*****************************************
* Function Name	: print_box
* Description	: Function to printout details of single bounding box to standard output
* Arguments	:	detection d: detected box details
*             int i : Result number
* Return value	:
******************************************/
void print_box(detection d, int i){
	printf("\x1b[4m"); //Change colour
	printf("\nResult %d\n", i);
	printf("\x1b[0m"); //Change the top first colour
	printf("\x1b[1m"); //Change the top first colour
	printf("Detected        : %s\n",label_file_map[d.c].c_str());//, detected
	printf("\x1b[0m"); //Change the colour to default
	printf("Bounding Box    : (X, Y, W, H) = (%.2f, %.2f, %.2f, %.2f)\n", d.bbox.x, d.bbox.y, d.bbox.w, d.bbox.h);
	printf("Confidence (IoU): %.1f %%\n", d.conf*100);
	printf("Probability     : %.1f %%\n",  d.prob*100);
	printf("Score           : %.1f %%\n", d.prob * d.conf*100);
}


int main(int argc, char* argv[])
{

	//Config : inference mode
	inference_mode = DETECTION;
	//Config : model
	std::string model_name = "Model.onnx";
	std::string model_out= "grid";
	std::string model_path= "/usr/bin/onnxruntime/examples/unitest/tiny_yolov2/Model.onnx";

	printf("Start Loading Model %s\n", model_name.c_str());

	int img_sizex, img_sizey, img_channels;
	img_sizex = 416;
	img_sizey = 416;

	//Postprocessing Variables
	float th_conf = 0.6;
	float th_prob = 0.5;
	int count		  = 0;
	std::vector<detection> det;

	//Timing Variables
	struct timeval start_time, stop_time;
	double diff, diff_capture;

	//UNCOMMENT to use dog image as an input
	stbi_uc * img_data = stbi_load("/usr/bin/onnxruntime/examples/inference/yolo004.jpg", &img_sizex, &img_sizey, &img_channels, STBI_default);
	struct S_Pixel
  {
      unsigned char RGBA[3];
  };
  const S_Pixel * imgPixels(reinterpret_cast<const S_Pixel *>(img_data));

	//Config: label txt
	std::string filename("/usr/bin/onnxruntime/examples/inference/VOC_pascal_classes.txt");

	//ONNX runtime: Necessary
  OrtEnv* env;
  OrtCreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env);

	//ONNX runtime: Necessary
	OrtSession* session;
	OrtSessionOptions* session_options=OrtCreateSessionOptions();
	OrtSetSessionThreadPoolSize(session_options, 2); //Multi-core
	OrtCreateSession(env, model_path.c_str(), session_options, &session);

  size_t num_input_nodes;
  OrtStatus* status;
  OrtAllocator* allocator;
  OrtCreateDefaultAllocator(&allocator);

  status = OrtSessionGetInputCount(session, &num_input_nodes);

  std::vector<const char*> input_node_names(num_input_nodes);
  std::vector<int64_t> input_node_dims;

  printf("\nCurrent Model is %s\n",model_name.c_str());
  printf("Number of inputs = %zu\n", num_input_nodes);

	/* Print Out Input details */
  // iterate over all input nodes
  for (size_t i = 0; i < num_input_nodes; i++) {
    // print input node names
    char* input_name;
    status = OrtSessionGetInputName(session, i, allocator, &input_name);
    printf("Input %zu : name=%s\n", i, input_name);
    input_node_names[i] = input_name;

    // print input node types
    OrtTypeInfo* typeinfo;
    status = OrtSessionGetInputTypeInfo(session, i, &typeinfo);
    const OrtTensorTypeAndShapeInfo* tensor_info = OrtCastTypeInfoToTensorInfo(typeinfo);
    ONNXTensorElementDataType type = OrtGetTensorElementType(tensor_info);
    printf("Input %zu : type=%d\n", i, type);

    size_t num_dims = 4;
    printf("Input %zu : num_dims=%zu\n", i, num_dims);
    input_node_dims.resize(num_dims);
    OrtGetDimensions(tensor_info, (int64_t*)input_node_dims.data(), num_dims);

		if(input_node_dims[0]<0){//Necessary for  Tiny YOLO v2
			input_node_dims[0]=1;	//Change the first dimension from -1 to 1
		}

    for (size_t j = 0; j < num_dims; j++) printf("Input %zu : dim %zu=%jd\n", i, j, input_node_dims[j]);

    OrtReleaseTypeInfo(typeinfo);
  }

  OrtReleaseAllocator(allocator);

	//ONNX: Prepare input container
  size_t input_tensor_size = img_sizex * img_sizey * 3;
  std::vector<float> input_tensor_values(input_tensor_size);

	//ONNX: Prepare output type
  std::vector<const char*> output_node_names;
	output_node_names.push_back(model_out.c_str());

	int frame_count = 0;
	size_t offs, c, y, x;
	std::map<float,int> result; //Output for classification


		Image *img = new Image(img_sizex, img_sizey, 3);
		//Transpose
		offs = 0;
		for ( c = 0; c < 3; c++){
			for ( y = 0; y < img_sizey; y++){
				for ( x = 0; x < img_sizex; x++, offs++){
					const int val(imgPixels[y * img_sizex + x].RGBA[c]);
					img->set((y*img_sizex+x)*3+c, val);
					input_tensor_values[offs] = val;
				}
			}
		}

		// create input tensor object from data values
		OrtAllocatorInfo* allocator_info;
		OrtCreateCpuAllocatorInfo(OrtArenaAllocator, OrtMemTypeDefault, &allocator_info);
		OrtValue* input_tensor = NULL;
		OrtCreateTensorWithDataAsOrtValue(allocator_info, input_tensor_values.data(), input_tensor_size * sizeof(float), input_node_dims.data(), 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);
		assert(OrtIsTensor(input_tensor));
	  OrtReleaseAllocatorInfo(allocator_info);

		// RUN: score model & input tensor, get back output tensor
		OrtValue* output_tensor = NULL;
		gettimeofday(&start_time, nullptr);
		OrtRun(session, NULL, input_node_names.data(), (const OrtValue* const*)&input_tensor, 1, output_node_names.data(), 1, &output_tensor);
		gettimeofday(&stop_time, nullptr);
		assert(OrtIsTensor(output_tensor));
		diff = timedifference_msec(start_time,stop_time);

		// Get pointer to output tensor float values
		float* floatarr;
		OrtGetTensorMutableData(output_tensor, (void**)&floatarr);

		if(loadLabelFile(filename) != 0)
		{
				fprintf(stderr,"Fail to open or process file %s\n",filename.c_str());
				delete img;
				return -1;
		}

		//Postprocessing

		gettimeofday(&start_time, nullptr); //Start postproc timer

		assert(OrtIsTensor(output_tensor));
		int b;

		for(b = 0;b<YOLO_NUM_BB;b++){
			for(y = 0;y<YOLO_GRID_Y;y++){
				for(x = 0;x<YOLO_GRID_X;x++){
					int offs = offset_(b, y, x);
					double tc = floatarr[offset(offs, 4)];
					double conf = sigmoid(tc);

					if (conf > th_conf){
						float tx = floatarr[offs];
						float ty = floatarr[offset(offs, 1)];
						float tw = floatarr[offset(offs, 2)];
						float th = floatarr[offset(offs, 3)];

						float xPos = ((float) x + sigmoid(tx))*32;
						float yPos = ((float) y + sigmoid(ty))*32;
						float wBox = (float) exp(tw)*anchors[2*b+0]*32;
						float hBox = (float) exp(th)*anchors[2*b+1]*32;

						Box bb = float_to_box(xPos, yPos, wBox, hBox);

						float classes[20];
						for (int c = 0;c<20;c++){
							classes[c] = floatarr[offset(offs, 5+c)];
						}
						softmax(classes);
						float max_pd = 0;
						int detected = -1;
						for (int c = 0;c<20;c++){
							if (classes[c]>max_pd){
								detected = c;
								max_pd = classes[c];
							}
						}
						float score = max_pd * conf;
						if (score>th_prob){
							detection d = { bb, conf , detected,max_pd };
							det.push_back(d);
							count++;
						}
					}
				}
			}
		}

		//NMS filter
		filter_boxes_nms(det, count, 0.6);

		int i, j=0;
		//Render boxes on image and print their details
		for (i =0;i<count;i++){
			if (det[i].prob == 0) continue;
			j++;
			print_box(det[i], j);
			std::stringstream stream;
			stream << std::fixed << std::setprecision(2) << det[i].conf*det[i].prob;
			std::string result_str = label_file_map[det[i].c]+ " "+ stream.str();
	  	img->drawRect((int)det[i].bbox.x, (int)det[i].bbox.y, (int)det[i].bbox.w, (int)det[i].bbox.h, (int)det[i].c, result_str.c_str());
		}
		gettimeofday(&stop_time, nullptr);//Stop postproc timer
		size_t time_post = timedifference_msec(start_time,stop_time);
		printf("Postprocessing Time: %.3f msec\n", time_post);

		//Save Image
		img->save(save_filename);

		OrtReleaseValue(output_tensor);
		OrtReleaseValue(input_tensor);
		printf("\x1b[36;1m");
		printf("Prediction Time: %.3f msec\n\n", diff);
		printf("\x1b[0m");

		delete img;


  OrtReleaseSession(session);
  OrtReleaseSessionOptions(session_options);
  OrtReleaseEnv(env);
  printf("Done!\n");

  return 0;
}
