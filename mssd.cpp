
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <sys/ioctl.h>
#include <errno.h>
#include <linux/videodev2.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <unistd.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "tengine_c_api.h"
#include "common.hpp"
#include <linux/fb.h>
#include "opencv2/imgcodecs/imgcodecs.hpp"
#define DEF_PROTO "models/MobileNetSSD_deploy.prototxt"
#define DEF_MODEL "models/MobileNetSSD_deploy.caffemodel"


#define TRUE            (1)
#define FALSE           (0)

#define FILE_VIDEO      "/dev/video10"
#define FBDEVICE		"/dev/fb0"


#define IMAGEWIDTH      640
#define IMAGEHEIGHT     480

#define FRAME_NUM       4

void draw_back(unsigned int* pfb, unsigned int width, unsigned int height, unsigned int color);

void draw_line(unsigned int* pfb, unsigned int width, unsigned int height);


int fd;
struct v4l2_buffer buf;
int ret = -1;
const char* class_names[] = { "background", "aeroplane", "bicycle",   "bird",   "boat",        "bottle",
							 "bus",        "car",       "cat",       "chair",  "cow",         "diningtable",
							 "dog",        "horse",     "motorbike", "person", "pottedplant", "sheep",
							 "sofa",       "train",     "tvmonitor" };

struct Box
{
	float x0;
	float y0;
	float x1;
	float y1;
	int class_idx;
	float score;
};

struct buffer
{
	void* start;
	unsigned int length;
	long long int timestamp;
} *buffers;


int v4l2_init()
{
	struct v4l2_capability cap;
	struct v4l2_fmtdesc fmtdesc;
	struct v4l2_format fmt;
	struct v4l2_streamparm stream_para;

	//打开摄像头设备
	if ((fd = open(FILE_VIDEO, O_RDWR)) == -1)
	{
		printf("Error opening V4L interface\n");
		return FALSE;
	}

	//查询设备属性
	if (ioctl(fd, VIDIOC_QUERYCAP, &cap) == -1)
	{
		printf("Error opening device %s: unable to query device.\n", FILE_VIDEO);
		return FALSE;
	}
	else
	{
		printf("driver:\t\t%s\n", cap.driver);
		printf("card:\t\t%s\n", cap.card);
		printf("bus_info:\t%s\n", cap.bus_info);
		printf("version:\t%d\n", cap.version);
		printf("capabilities:\t%x\n", cap.capabilities);

		if ((cap.capabilities & V4L2_CAP_VIDEO_CAPTURE) == V4L2_CAP_VIDEO_CAPTURE)
		{
			printf("Device %s: supports capture.\n", FILE_VIDEO);
		}

		if ((cap.capabilities & V4L2_CAP_STREAMING) == V4L2_CAP_STREAMING)
		{
			printf("Device %s: supports streaming.\n", FILE_VIDEO);
		}
	}


	//显示所有支持帧格式
	fmtdesc.index = 0;
	fmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	printf("Support format:\n");
	while (ioctl(fd, VIDIOC_ENUM_FMT, &fmtdesc) != -1)
	{
		printf("\t%d.%s\n", fmtdesc.index + 1, fmtdesc.description);
		fmtdesc.index++;
	}

	//检查是否支持某帧格式
	struct v4l2_format fmt_test;
	fmt_test.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	fmt_test.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB24;
	if (ioctl(fd, VIDIOC_TRY_FMT, &fmt_test) == -1)
	{
		printf("not support format RGB32!\n");
	}
	else
	{
		printf("support format RGB32\n");
	}


	//查看及设置当前格式
	printf("set fmt...\n");
	fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	//fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB24; //jpg格式
	fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;//yuv格式

	fmt.fmt.pix.height = IMAGEHEIGHT;
	fmt.fmt.pix.width = IMAGEWIDTH;
	fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;
	printf("fmt.type:\t\t%d\n", fmt.type);
	printf("pix.pixelformat:\t%c%c%c%c\n", fmt.fmt.pix.pixelformat & 0xFF, (fmt.fmt.pix.pixelformat >> 8) & 0xFF, (fmt.fmt.pix.pixelformat >> 16) & 0xFF, (fmt.fmt.pix.pixelformat >> 24) & 0xFF);
	printf("pix.height:\t\t%d\n", fmt.fmt.pix.height);
	printf("pix.width:\t\t%d\n", fmt.fmt.pix.width);
	printf("pix.field:\t\t%d\n", fmt.fmt.pix.field);
	if (ioctl(fd, VIDIOC_S_FMT, &fmt) == -1)
	{
		printf("Unable to set format\n");
		return FALSE;
	}

	printf("get fmt...\n");
	if (ioctl(fd, VIDIOC_G_FMT, &fmt) == -1)
	{
		printf("Unable to get format\n");
		return FALSE;
	}
	{
		printf("fmt.type:\t\t%d\n", fmt.type);
		printf("pix.pixelformat:\t%c%c%c%c\n", fmt.fmt.pix.pixelformat & 0xFF, (fmt.fmt.pix.pixelformat >> 8) & 0xFF, (fmt.fmt.pix.pixelformat >> 16) & 0xFF, (fmt.fmt.pix.pixelformat >> 24) & 0xFF);
		printf("pix.height:\t\t%d\n", fmt.fmt.pix.height);
		printf("pix.width:\t\t%d\n", fmt.fmt.pix.width);
		printf("pix.field:\t\t%d\n", fmt.fmt.pix.field);
	}

	//设置及查看帧速率，这里只能是30帧，就是1秒采集30张图
	memset(&stream_para, 0, sizeof(struct v4l2_streamparm));
	stream_para.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	stream_para.parm.capture.timeperframe.denominator = 30;
	stream_para.parm.capture.timeperframe.numerator = 1;

	if (ioctl(fd, VIDIOC_S_PARM, &stream_para) == -1)
	{
		printf("Unable to set frame rate\n");
		return FALSE;
	}
	if (ioctl(fd, VIDIOC_G_PARM, &stream_para) == -1)
	{
		printf("Unable to get frame rate\n");
		return FALSE;
	}
	{
		printf("numerator:%d\ndenominator:%d\n", stream_para.parm.capture.timeperframe.numerator, stream_para.parm.capture.timeperframe.denominator);
	}
	return TRUE;
}



int v4l2_mem_ops()
{
	unsigned int n_buffers;
	struct v4l2_requestbuffers req;

	//申请帧缓冲
	req.count = FRAME_NUM;
	req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	req.memory = V4L2_MEMORY_MMAP;
	if (ioctl(fd, VIDIOC_REQBUFS, &req) == -1)
	{
		printf("request for buffers error\n");
		return FALSE;
	}

	// 申请用户空间的地址列
	buffers = (buffer*)malloc(req.count * sizeof(*buffers));
	if (!buffers)
	{
		printf("out of memory!\n");
		return FALSE;
	}

	// 进行内存映射
	for (n_buffers = 0; n_buffers < FRAME_NUM; n_buffers++)
	{
		buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		buf.memory = V4L2_MEMORY_MMAP;
		buf.index = n_buffers;
		//查询
		if (ioctl(fd, VIDIOC_QUERYBUF, &buf) == -1)
		{
			printf("query buffer error\n");
			return FALSE;
		}

		//映射
		buffers[n_buffers].length = buf.length;
		buffers[n_buffers].start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
		if (buffers[n_buffers].start == MAP_FAILED)
		{
			printf("buffer map error\n");
			return FALSE;
		}
	}
	return TRUE;
}


/*
int v4l2_frame_process()
{
	unsigned int n_buffers;
	enum v4l2_buf_type type;
	char file_name[100];
	char index_str[10];
	long long int extra_time = 0;
	long long int cur_time = 0;
	long long int last_time = 0;

	//入队和开启采集
	for (n_buffers = 0; n_buffers < FRAME_NUM; n_buffers++)
	{
		buf.index = n_buffers;
		ioctl(fd, VIDIOC_QBUF, &buf);
	}
	type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	ioctl(fd, VIDIOC_STREAMON, &type);



	//出队，处理，写入yuv文件，入队，循环进行
	int loop = 0;
	while (loop < 15)
	{
		for (n_buffers = 0; n_buffers < FRAME_NUM; n_buffers++)
		{
			//出队
			buf.index = n_buffers;
			ioctl(fd, VIDIOC_DQBUF, &buf);

			//查看采集数据的时间戳之差，单位为微妙
			buffers[n_buffers].timestamp = buf.timestamp.tv_sec * 1000000 + buf.timestamp.tv_usec;
			cur_time = buffers[n_buffers].timestamp;
			extra_time = cur_time - last_time;
			last_time = cur_time;
			printf("time_deta:%lld\n\n", extra_time);
			printf("buf_len:%d\n", buffers[n_buffers].length);

			//此处完成处理数据的函数


			//处理数据只是简单写入文件，名字以loop的次数和帧缓冲数目有关
			printf("grab image data OK\n");
			memset(file_name, 0, sizeof(file_name));
			memset(index_str, 0, sizeof(index_str));
			sprintf(index_str, "%d", loop * 4 + n_buffers);
			strcpy(file_name, IMAGE);
			strcat(file_name, index_str);
			strcat(file_name, ".jpg");
			strcat(file_name,".yuv");
			FILE* fp2 = fopen(file_name, "wb");
			if (!fp2)
			{
				printf("open %s error\n", file_name);
				return(FALSE);
			}
			fwrite(buffers[n_buffers].start, IMAGEHEIGHT * IMAGEWIDTH * 2, 1, fp2);
			fclose(fp2);
			printf("save %s OK\n", file_name);

			//入队循环
			ioctl(fd, VIDIOC_QBUF, &buf);
		}

		loop++;
	}
	return TRUE;
}
*/



int v4l2_release()
{
	unsigned int n_buffers;
	enum v4l2_buf_type type;

	//关闭流
	type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	ioctl(fd, VIDIOC_STREAMON, &type);

	//关闭内存映射
	for (n_buffers = 0; n_buffers < FRAME_NUM; n_buffers++)
	{
		munmap(buffers[n_buffers].start, buffers[n_buffers].length);
	}

	//释放自己申请的内存
	free(buffers);

	//关闭设备
	close(fd);
	return TRUE;
}




/*int v4l2_video_input_output()
{
	struct v4l2_input input;
	struct v4l2_standard standard;

	//首先获得当前输入的index,注意只是index，要获得具体的信息，就的调用列举操作
	memset (&input,0,sizeof(input));
	if (-1 == ioctl (fd, VIDIOC_G_INPUT, &input.index)) {
		printf("VIDIOC_G_INPUT\n");
		return FALSE;
	}
	//调用列举操作，获得 input.index 对应的输入的具体信息
	if (-1 == ioctl (fd, VIDIOC_ENUMINPUT, &input)) {
		printf("VIDIOC_ENUM_INPUT \n");
		return FALSE;
	}
	printf("Current input %s supports:\n", input.name);


	//列举所有的所支持的 standard，如果 standard.id 与当前 input 的 input.std 有共同的
	//bit flag，意味着当前的输入支持这个 standard,这样将所有驱动所支持的 standard 列举一个
	//遍，就可以找到该输入所支持的所有 standard 了。

	memset(&standard,0,sizeof (standard));
	standard.index = 0;
	while(0 == ioctl(fd, VIDIOC_ENUMSTD, &standard)) {
		if (standard.id & input.std){
			printf ("%s\n", standard.name);
		}
		standard.index++;
	}
	// EINVAL indicates the end of the enumeration, which cannot be empty unless this device falls under the USB exception.

	if (errno != EINVAL || standard.index == 0) {
		printf("VIDIOC_ENUMSTD\n");
		return FALSE;
	}

}*/



void draw_line(unsigned int* pfb, unsigned int width, unsigned int height)
{
	unsigned int x, y;
	for (x = 50; x < width - 50; x++)
	{
		*(pfb + 50 * width + x) = 0xffffff00;
	}
	for (y = 50; y < height - 50; y++)
	{
		*(pfb + y * width + 50) = 0xffffff00;
	}
}

void yuyv_to_bgr(unsigned char* yuv,unsigned char* rgb,int width, int height )
{
    unsigned int i;
    unsigned char* y0 = yuv + 0;   
    unsigned char* u0 = yuv + 1;
    unsigned char* y1 = yuv + 2;
    unsigned char* v0 = yuv + 3;
 
    unsigned  char* b0 = rgb + 0;
    unsigned  char* g0 = rgb + 1;
    unsigned  char* r0 = rgb + 2;
    unsigned  char* b1 = rgb + 3;
    unsigned  char* g1 = rgb + 4;
    unsigned  char* r1 = rgb + 5;
   
    float rt0 = 0, gt0 = 0, bt0 = 0, rt1 = 0, gt1 = 0, bt1 = 0;
 
    for(i = 0; i <= (width * height) / 2 ;i++)
    {
        bt0 = 1.164 * (*y0 - 16) + 2.018 * (*u0 - 128); 
        gt0 = 1.164 * (*y0 - 16) - 0.813 * (*v0 - 128) - 0.394 * (*u0 - 128); 
        rt0 = 1.164 * (*y0 - 16) + 1.596 * (*v0 - 128); 
   
        bt1 = 1.164 * (*y1 - 16) + 2.018 * (*u0 - 128); 
        gt1 = 1.164 * (*y1 - 16) - 0.813 * (*v0 - 128) - 0.394 * (*u0 - 128); 
        rt1 = 1.164 * (*y1 - 16) + 1.596 * (*v0 - 128); 
    
      
        if(rt0 > 250)   rt0 = 255;
        if(rt0< 0)      rt0 = 0;    
 
        if(gt0 > 250)   gt0 = 255;
        if(gt0 < 0) gt0 = 0;    
 
        if(bt0 > 250)   bt0 = 255;
        if(bt0 < 0) bt0 = 0;    
 
        if(rt1 > 250)   rt1 = 255;
        if(rt1 < 0) rt1 = 0;    
 
        if(gt1 > 250)   gt1 = 255;
        if(gt1 < 0) gt1 = 0;    
 
        if(bt1 > 250)   bt1 = 255;
        if(bt1 < 0) bt1 = 0;    
                    
        *r0 = (unsigned char)rt0;
        *g0 = (unsigned char)gt0;
        *b0 = (unsigned char)bt0;
    
        *r1 = (unsigned char)rt1;
        *g1 = (unsigned char)gt1;
        *b1 = (unsigned char)bt1;
 
        yuv = yuv + 4;
        rgb = rgb + 6;
        if(yuv == NULL)
          break;
 
        y0 = yuv;
        u0 = yuv + 1;
        y1 = yuv + 2;
        v0 = yuv + 3;
  
        b0 = rgb + 0;
        g0 = rgb + 1;
        r0 = rgb + 2;
        b1 = rgb + 3;
        g1 = rgb + 4;
        r1 = rgb + 5;
    }   
}

int main(int argc, char const* argv[])
{

	int fb = -1;
	int ret = -1;
	unsigned int * pfb = NULL;
	struct fb_fix_screeninfo finfo;
	struct fb_var_screeninfo vinfo;

	fb = open(FBDEVICE, O_RDWR);
	if (fb < 0)
	{
		perror("open");
		return -1;
	}
	printf("open %s success \n", FBDEVICE);

	ret = ioctl(fb, FBIOGET_FSCREENINFO, &finfo);

	if (ret < 0)
	{
	perror("ioctl");
		return -1;
	}

	ret = ioctl(fb, FBIOGET_VSCREENINFO, &vinfo);
	if (ret < 0)
	{
		perror("ioctl");
		return -1;
	}

	pfb = (unsigned int *)mmap(NULL, finfo.smem_len, PROT_READ | PROT_WRITE, MAP_SHARED, fb, 0);
	//unsigned int fbLineSize = finfo.line_length;
	printf("smem_len: %ld", finfo.smem_len);
	if (NULL == pfb)
	{
		perror("mmap");
		return -1;
	}

	printf("pfb :0x%x \n", *pfb);
	std::cout << "height: " << vinfo.yres << "weight: "<< vinfo.xres << std::endl;

	const std::string root_path = get_root_path();
	std::string proto_file;
	std::string model_file;
	const char* device = nullptr;

	proto_file = root_path + DEF_PROTO;
	model_file = root_path + DEF_MODEL;

	// init tengine
	if (init_tengine() < 0)
	{
		std::cout << " init tengine failed\n";
		return 1;
	}
	if (request_tengine_version("0.9") != 1)
	{
		std::cout << " request tengine version failed\n";
		return 1;
	}

	// check file
	if (!check_file_exist(proto_file) or !check_file_exist(model_file))
	{
		return 1;
	}

	// create graph
	graph_t graph = create_graph(nullptr, "caffe", proto_file.c_str(), model_file.c_str());
	if (graph == nullptr)
	{
		std::cout << "Create graph failed\n";
		std::cout << " ,errno: " << get_tengine_errno() << "\n";
		return 1;
	}


	if (device != nullptr)
	{
		set_graph_device(graph, device);
	}

	// input
	int img_h = 300;
	int img_w = 300;
	int img_size = img_h * img_w * 3;
	float* input_data = (float*)malloc(sizeof(float) * img_size);

	int node_idx = 0;
	int tensor_idx = 0;
	tensor_t input_tensor = get_graph_input_tensor(graph, node_idx, tensor_idx);
	if (input_tensor == nullptr)
	{
		std::printf("Cannot find input tensor,node_idx: %d,tensor_idx: %d\n", node_idx, tensor_idx);
		return -1;
	}

	int dims[] = { 1, 3, img_h, img_w };
	set_tensor_shape(input_tensor, dims, 4);
	ret = prerun_graph(graph);
	if (ret != 0)
	{
		std::cout << "Prerun graph failed, errno: " << get_tengine_errno() << "\n";
		return 1;
	}

	printf("begin....\n");
	sleep(10);

	v4l2_init();
	printf("init....\n");
	sleep(10);

	v4l2_mem_ops();
	printf("malloc....\n");
	sleep(10);

	unsigned int n_buffers;
	enum v4l2_buf_type type;
	char file_name[100];
	char index_str[10];
	long long int extra_time = 0;
	long long int cur_time = 0;
	long long int last_time = 0;

	//入队和开启采集
	for (n_buffers = 0; n_buffers < FRAME_NUM; n_buffers++)
	{
		buf.index = n_buffers;
		ioctl(fd, VIDIOC_QBUF, &buf);
	}
	type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	ioctl(fd, VIDIOC_STREAMON, &type);



	//出队，处理，入队，循环进行
	int loop = 0;
	
	while (TRUE)
	{
		for (n_buffers = 0; n_buffers < FRAME_NUM; n_buffers++)
		{
			//出队
			buf.index = n_buffers;
			ioctl(fd, VIDIOC_DQBUF, &buf);

			//查看采集数据的时间戳之差，单位为微妙
			//buffers[n_buffers].timestamp = buf.timestamp.tv_sec * 1000000 + buf.timestamp.tv_usec;
			//cur_time = buffers[n_buffers].timestamp;
			//extra_time = cur_time - last_time;
			//last_time = cur_time;
			//printf("time_deta:%lld\n\n", extra_time);
			//printf("buf_len:%d\n", buffers[n_buffers].length);

			//此处完成处理数据的函数

			cv::Mat frame = cv::Mat(IMAGEHEIGHT, IMAGEWIDTH, CV_8UC3);
			yuyv_to_bgr((unsigned char*)buffers[n_buffers].start,frame.data,640, 480);
			if (frame.empty()){
				ioctl(fd, VIDIOC_QBUF, &buf);
				continue;
			}
			if(loop > 100){
				cv::imwrite("out.jpg", frame);
				return 0;}
			
			cv::Mat outFrame;
			//frame = cv::imdecode(frame, cv::IMREAD_COLOR);
			outFrame = frame;
			//cv::cvtColor(frame, outFrame, cv::COLOR_YUV2BGR_YUYV);
			//cv::cvtColor(frame, frame, cv::COLOR_YUV2BGR_YUYV);
			//cv::resize(outFrame, outFrame, cv::Size(1080,1920));
			cv::resize(frame, frame, cv::Size(img_h, img_w));

			frame.convertTo(frame, CV_32FC3);

			float* img_data = (float*)frame.data;
			int hw = img_h * img_w;

			float mean[3] = { 127.5, 127.5, 127.5 };
			for (int h = 0; h < img_h; h++)
			{
				for (int w = 0; w < img_w; w++)
				{
					for (int c = 0; c < 3; c++)
					{
						input_data[c * hw + h * img_w + w] = 0.007843 * (*img_data - mean[c]);
						img_data++;
					}
				}
			}


			set_tensor_buffer(input_tensor, input_data, img_size * 4);
			run_graph(graph, 1);

			tensor_t out_tensor = get_graph_output_tensor(graph, 0, 0);    //"detection_out");

			int out_dim[4];
			ret = get_tensor_shape(out_tensor, out_dim, 4);
			if (ret <= 0)
			{
				std::cout << "get tensor shape failed, errno: " << get_tengine_errno() << "\n";
				return 1;
			}

			float* outdata = (float*)get_tensor_buffer(out_tensor);
			int num = out_dim[1];
			float show_threshold = 0.5;

			int raw_h = IMAGEHEIGHT;
			int raw_w = IMAGEWIDTH;
			std::vector<Box> boxes;
			int line_width = raw_w * 0.005;
			printf("detect result num: %d \n", num);
			for (int i = 0; i < num; i++)
			{
				if (outdata[1] >= show_threshold)
				{
					Box box;
					box.class_idx = outdata[0];
					box.score = outdata[1];
					box.x0 = outdata[2] * raw_w;
					box.y0 = outdata[3] * raw_h;
					box.x1 = outdata[4] * raw_w;
					box.y1 = outdata[5] * raw_h;
					boxes.push_back(box);
					printf("%s\t:%.0f%%\n", class_names[box.class_idx], box.score * 100);
					printf("BOX:( %g , %g ),( %g , %g )\n", box.x0, box.y0, box.x1, box.y1);
				}
				outdata += 6;
			}
			
			for (int i = 0; i < (int)boxes.size(); i++)
			{
				Box box = boxes[i];
				//cv::rectangle(img, cv::Rect(box.x0, box.y0, (box.x1 - box.x0), (box.y1 - box.y0)), cv::Scalar(255, 255, 0),
				//	line_width);
				std::ostringstream score_str;
				score_str << box.score;
				std::string label = std::string(class_names[box.class_idx]) + ": " + score_str.str();
				//int baseLine = 0;
				//cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
				//cv::rectangle(img,
				//	cv::Rect(cv::Point(box.x0, box.y0 - label_size.height),
				//		cv::Size(label_size.width, label_size.height + baseLine)),
				//	cv::Scalar(255, 255, 0), CV_FILLED);
				//cv::putText(img, label, cv::Point(box.x0, box.y0), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
			}
			release_graph_tensor(out_tensor);
			//处理数据只是简单写入文件，名字以loop的次数和帧缓冲数目有关
			//printf("grab image data OK\n");
			//memset(file_name, 0, sizeof(file_name));
			//memset(index_str, 0, sizeof(index_str));
			//sprintf(index_str, "%d", loop * 4 + n_buffers);
			//strcpy(file_name, IMAGE);
			//strcat(file_name, index_str);
			//strcat(file_name, ".jpg");
			//strcat(file_name,".yuv");
			//FILE* fp2 = fopen(file_name, "wb");
			//if (!fp2)
			//{
			//	printf("open %s error\n", file_name);
			//	return(FALSE);
			//}
			//fwrite(buffers[n_buffers].start, IMAGEHEIGHT * IMAGEWIDTH * 2, 1, fp2);
			//fclose(fp2);
			//printf("save %s OK\n", file_name);
			//cv::imshow("demo", frame);
				
			uint32_t color = 0;
			for (int h=0; h < 480; h++){
				for(int w=0;w <640; w++){
					color = (0xff000000) | ((outFrame.data[h*640+w+640*480*2] << 16) & 0x00ff0000) | ((outFrame.data[h*640+w+640*480] << 8) & 0x0000ff00) | ((outFrame.data[h*640+w]&0x000000ff));
					*(pfb+h*vinfo.xres_virtual+w+640)  = color;
					//*(pfb+h*vinfo.xres_virtual+w*32+640+8) = outFrame.data[h*640+w];
					//*(pfb+h*vinfo.xres_virtual+w*32+640+16) = outFrame.data[h*640+w+640*480];
					//*(pfb+h*vinfo.xres_virtual+w*32+640+24) = outFrame.data[h*640+w+2*640*480];
				}
			}
			
			
			ioctl(fd, VIDIOC_QBUF, &buf);
			loop++;
		}
		
		if(cv::waitKey(10)==27)
			break;
		loop++;
	}
	printf("process....\n");
	sleep(10);
	release_graph_tensor(input_tensor);

	ret = postrun_graph(graph);
	if (ret != 0)
	{
		std::cout << "Postrun graph failed, errno: " << get_tengine_errno() << "\n";
		return 1;
	}
	free(input_data);
	destroy_graph(graph);
	release_tengine();

	v4l2_release();
	printf("release\n");


	close(fd);

	sleep(20);
	
	return TRUE;
}
