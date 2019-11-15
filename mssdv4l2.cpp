/*v4l2_example.c*/
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


#define DEF_PROTO "models/MobileNetSSD_deploy.prototxt"
#define DEF_MODEL "models/MobileNetSSD_deploy.caffemodel"


#define TRUE            (1)
#define FALSE           (0)

#define FILE_VIDEO      "/dev/video10"
#define IMAGE           "./imgdemo"

#define IMAGEWIDTH      640
#define IMAGEHEIGHT     480

#define FRAME_NUM       4

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

	//������ͷ�豸
	if ((fd = open(FILE_VIDEO, O_RDWR)) == -1)
	{
		printf("Error opening V4L interface\n");
		return FALSE;
	}

	//��ѯ�豸����
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


	//��ʾ����֧��֡��ʽ
	fmtdesc.index = 0;
	fmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	printf("Support format:\n");
	while (ioctl(fd, VIDIOC_ENUM_FMT, &fmtdesc) != -1)
	{
		printf("\t%d.%s\n", fmtdesc.index + 1, fmtdesc.description);
		fmtdesc.index++;
	}

	//����Ƿ�֧��ĳ֡��ʽ
	struct v4l2_format fmt_test;
	fmt_test.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	fmt_test.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB32;
	if (ioctl(fd, VIDIOC_TRY_FMT, &fmt_test) == -1)
	{
		printf("not support format RGB32!\n");
	}
	else
	{
		printf("support format RGB32\n");
	}


	//�鿴�����õ�ǰ��ʽ
	printf("set fmt...\n");
	fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB32; //jpg��ʽ
	//fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;//yuv��ʽ

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

	//���ü��鿴֡���ʣ�����ֻ����30֡������1��ɼ�30��ͼ
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

	//����֡����
	req.count = FRAME_NUM;
	req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	req.memory = V4L2_MEMORY_MMAP;
	if (ioctl(fd, VIDIOC_REQBUFS, &req) == -1)
	{
		printf("request for buffers error\n");
		return FALSE;
	}

	// �����û��ռ�ĵ�ַ��
	buffers = (buffer*)malloc(req.count * sizeof(*buffers));
	if (!buffers)
	{
		printf("out of memory!\n");
		return FALSE;
	}

	// �����ڴ�ӳ��
	for (n_buffers = 0; n_buffers < FRAME_NUM; n_buffers++)
	{
		buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		buf.memory = V4L2_MEMORY_MMAP;
		buf.index = n_buffers;
		//��ѯ
		if (ioctl(fd, VIDIOC_QUERYBUF, &buf) == -1)
		{
			printf("query buffer error\n");
			return FALSE;
		}

		//ӳ��
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



int v4l2_frame_process()
{
	unsigned int n_buffers;
	enum v4l2_buf_type type;
	char file_name[100];
	char index_str[10];
	long long int extra_time = 0;
	long long int cur_time = 0;
	long long int last_time = 0;

	//��ӺͿ����ɼ�
	for (n_buffers = 0; n_buffers < FRAME_NUM; n_buffers++)
	{
		buf.index = n_buffers;
		ioctl(fd, VIDIOC_QBUF, &buf);
	}
	type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	ioctl(fd, VIDIOC_STREAMON, &type);



	//���ӣ�����д��yuv�ļ�����ӣ�ѭ������
	int loop = 0;
	while (loop < 15)
	{
		for (n_buffers = 0; n_buffers < FRAME_NUM; n_buffers++)
		{
			//����
			buf.index = n_buffers;
			ioctl(fd, VIDIOC_DQBUF, &buf);

			//�鿴�ɼ����ݵ�ʱ���֮���λΪ΢��
			//buffers[n_buffers].timestamp = buf.timestamp.tv_sec * 1000000 + buf.timestamp.tv_usec;
			//cur_time = buffers[n_buffers].timestamp;
			//extra_time = cur_time - last_time;
			//last_time = cur_time;
			//printf("time_deta:%lld\n\n", extra_time);
			//printf("buf_len:%d\n", buffers[n_buffers].length);

			//�˴���ɴ������ݵĺ���


			//��������ֻ�Ǽ�д���ļ���������loop�Ĵ�����֡������Ŀ�й�
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

			//���ѭ��
			ioctl(fd, VIDIOC_QBUF, &buf);
		}

		loop++;
	}
	return TRUE;
}




int v4l2_release()
{
	unsigned int n_buffers;
	enum v4l2_buf_type type;

	//�ر���
	type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	ioctl(fd, VIDIOC_STREAMON, &type);

	//�ر��ڴ�ӳ��
	for (n_buffers = 0; n_buffers < FRAME_NUM; n_buffers++)
	{
		munmap(buffers[n_buffers].start, buffers[n_buffers].length);
	}

	//�ͷ��Լ�������ڴ�
	free(buffers);

	//�ر��豸
	close(fd);
	return TRUE;
}




/*int v4l2_video_input_output()
{
	struct v4l2_input input;
	struct v4l2_standard standard;

	//���Ȼ�õ�ǰ�����index,ע��ֻ��index��Ҫ��þ������Ϣ���͵ĵ����оٲ���
	memset (&input,0,sizeof(input));
	if (-1 == ioctl (fd, VIDIOC_G_INPUT, &input.index)) {
		printf("VIDIOC_G_INPUT\n");
		return FALSE;
	}
	//�����оٲ�������� input.index ��Ӧ������ľ�����Ϣ
	if (-1 == ioctl (fd, VIDIOC_ENUMINPUT, &input)) {
		printf("VIDIOC_ENUM_INPUT \n");
		return FALSE;
	}
	printf("Current input %s supports:\n", input.name);


	//�о����е���֧�ֵ� standard����� standard.id �뵱ǰ input �� input.std �й�ͬ��
	//bit flag����ζ�ŵ�ǰ������֧����� standard,����������������֧�ֵ� standard �о�һ��
	//�飬�Ϳ����ҵ���������֧�ֵ����� standard �ˡ�

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






int main(int argc, char const* argv[])
{


	int ret = -1;
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

	//��ӺͿ����ɼ�
	for (n_buffers = 0; n_buffers < FRAME_NUM; n_buffers++)
	{
		buf.index = n_buffers;
		ioctl(fd, VIDIOC_QBUF, &buf);
	}
	type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	ioctl(fd, VIDIOC_STREAMON, &type);



	//���ӣ�������ӣ�ѭ������
	//int loop = 0;
	
	while (TRUE)
	{
		for (n_buffers = 0; n_buffers < FRAME_NUM; n_buffers++)
		{
			//����
			buf.index = n_buffers;
			ioctl(fd, VIDIOC_DQBUF, &buf);

			//�鿴�ɼ����ݵ�ʱ���֮���λΪ΢��
			//buffers[n_buffers].timestamp = buf.timestamp.tv_sec * 1000000 + buf.timestamp.tv_usec;
			//cur_time = buffers[n_buffers].timestamp;
			//extra_time = cur_time - last_time;
			//last_time = cur_time;
			//printf("time_deta:%lld\n\n", extra_time);
			//printf("buf_len:%d\n", buffers[n_buffers].length);

			//�˴���ɴ������ݵĺ���
			std::cout << "frame create" << std::endl;
			cv::Mat frame = cv::Mat(IMAGEHEIGHT, IMAGEWIDTH, CV_8UC3, (void*)buffers[n_buffers].start);
			std::cout << "frame end" << std::endl;
			cv::resize(frame, frame, cv::Size(img_h, img_w));
			std::cout << "resize end" << std::endl;
			frame.convertTo(frame, CV_32FC3);
			std::cout << "convert end " << std::endl;
			float* img_data = (float*)frame.data;
			int hw = img_h * img_w;
			std::cout << "data end" << std::endl;	
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
			std::cout << "copy end" << std::endl;

			set_tensor_buffer(input_tensor, input_data, img_size * 4);
			run_graph(graph, 1);
			std::cout << "run end" << std::endl;
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
				std::cout << "label:" << label << std::endl;
				//int baseLine = 0;
				//cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
				//cv::rectangle(img,
				//	cv::Rect(cv::Point(box.x0, box.y0 - label_size.height),
				//		cv::Size(label_size.width, label_size.height + baseLine)),
				//	cv::Scalar(255, 255, 0), CV_FILLED);
				//cv::putText(img, label, cv::Point(box.x0, box.y0), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
			}
			release_graph_tensor(out_tensor);
			//��������ֻ�Ǽ�д���ļ���������loop�Ĵ�����֡������Ŀ�й�
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
			cv::imshow("demo", frame);
			if(cv::waitKey(10)==27)
				break;
			//���ѭ��
			ioctl(fd, VIDIOC_QBUF, &buf);
		}
		if(cv::waitKey(10)==27)
			break;
		//loop++;
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
	sleep(20);

	return TRUE;
}
