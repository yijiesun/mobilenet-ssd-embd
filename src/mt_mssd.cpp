/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2018, Open AI Lab
 * Author: chunyinglv@openailab.com
 */

#include <unistd.h>
#include <sys/time.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <atomic>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "tengine_c_api.h"
#include "cpu_device.h"
#include "common.hpp"
#include "config.h"
#include "v4l2/v4l2.h"  
#include "screen/screen.h"
#define DEF_PROTO "models/MobileNetSSD_deploy.prototxt"
#define DEF_MODEL "models/MobileNetSSD_deploy.caffemodel"
#define CPU_THREAD_CNT 3 //a53-012         a72-4          a72-5
#define GPU_THREAD_CNT 2 
using namespace cv;
using namespace std;

int cpu_num_[] ={0,1,2,3,4};
std::mutex  mutex_;
V4L2 v4l2_;
vector<Box>	boxes; 
vector<Box>	boxes_all; 
cv::Mat rgb;
SCREEN screen_;
unsigned int * pfb;
cv::Mat frame;
cv::Mat frame_[CPU_THREAD_CNT+(GPU_THREAD_CNT+1)/2];
tensor_t input_tensor[CPU_THREAD_CNT+(GPU_THREAD_CNT+1)/2];
int IMG_WID;
int IMG_HGT;
int img_h;
int img_w;
int img_size;
float* input_data[CPU_THREAD_CNT+(GPU_THREAD_CNT+1)/2];
tensor_t out_tensor[CPU_THREAD_CNT+(GPU_THREAD_CNT+1)/2];
int thread_num = 0;
std::string image_file;
graph_t graph[CPU_THREAD_CNT+(GPU_THREAD_CNT+1)/2]; //GPU-0;CPU-123

const char* class_names[] = {"background", "aeroplane", "bicycle",   "bird",   "boat",        "bottle",
                                "bus",        "car",       "cat",       "chair",  "cow",         "diningtable",
                                "dog",        "horse",     "motorbike", "person", "pottedplant", "sheep",
                                "sofa",       "train",     "tvmonitor"};
inline int set_cpu(int i)  
{  
    cpu_set_t mask;  
    CPU_ZERO(&mask);  
  
    CPU_SET(i,&mask);  

    if(-1 == pthread_setaffinity_np(pthread_self() ,sizeof(mask),&mask))  
    {  
        fprintf(stderr, "pthread_setaffinity_np erro\n");  
        return -1;  
    }  
    return 0;  
} 

void get_input_data_ssd(Mat& img, float* input_data, int img_h, int img_w)
{
    cv::resize(img, img, cv::Size(img_h, img_w));
    img.convertTo(img, CV_32FC3);
    float* img_data = ( float* )img.data;
    int hw = img_h * img_w;

    float mean[3] = {127.5, 127.5, 127.5};
    for(int h = 0; h < img_h; h++)
    {
        for(int w = 0; w < img_w; w++)
        {
            for(int c = 0; c < 3; c++)
            {
                input_data[c * hw + h * img_w + w] = 0.007843 * (*img_data - mean[c]);
                img_data++;
            }
        }
    }
}

void post_process_ssd(Mat& img, float threshold, float* outdata, int num)
{
    int raw_h = frame.size().height;
    int raw_w = frame.size().width;
    boxes.clear();
    int line_width = raw_w * 0.005;
    printf("detect result num: %d \n", num);
    for(int i = 0; i < num; i++)
    {
        if(outdata[1] >= threshold)
        {
            Box box;
            box.class_idx = outdata[0];
            box.score = outdata[1];
            box.x0 = outdata[2] * raw_w;
            box.y0 = outdata[3] * raw_h;
            box.x1 = outdata[4] * raw_w;
            box.y1 = outdata[5] * raw_h;
            boxes.push_back(box);
        }
        outdata += 6;
    }

}

void mssd_core(Mat &img,graph_t &graph, int thread_num, float* input_data,tensor_t &input_tensor, tensor_t &out_tensor )
{
    struct timeval t0, t1;
    float total_time = 0.f;
    gettimeofday(&t0, NULL);
    get_input_data_ssd(img, input_data, img_h, img_w);
    set_tensor_buffer(input_tensor, input_data, img_size * 4);
    run_graph(graph, 1);
    
    out_tensor = get_graph_output_tensor(graph, 0, 0); 
    int out_dim[4];
    get_tensor_shape(out_tensor, out_dim, 4);
    float* outdata = ( float* )get_tensor_buffer(out_tensor);
    int num = out_dim[1];
    float show_threshold = 0.5;

    post_process_ssd(img, show_threshold, outdata, num);

    gettimeofday(&t1, NULL);
    float mytime = ( float )((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;

    std::cout <<"thread " << thread_num << " times  " << mytime << "ms\n";
}

void *cpu_pthread(void *threadarg)
{
    int cpu_num = (*(int*)threadarg )+(GPU_THREAD_CNT+1)/2;
    mssd_core(frame_[cpu_num],graph[cpu_num], cpu_num,input_data[cpu_num],input_tensor[cpu_num],out_tensor[cpu_num]);
}
void *gpu_pthread(void *threadarg)
{
    mssd_core(frame_[0],graph[0], 0,input_data[0],input_tensor[0],out_tensor[0]);
#if (GPU_THREAD_CNT>=2)
    mssd_core(frame_[0],graph[0], 4,input_data[0],input_tensor[0],out_tensor[0]);
#endif
}

void *v4l2_thread(void *threadarg)
{
    set_cpu(3);
	while (1)
	{
        struct timeval t0, t1;
         float total_time = 0.f;
         gettimeofday(&t0, NULL);
        v4l2_.read_frame_argb(pfb,frame,screen_.vinfo.xres_virtual,0,0);
        // pthread_mutex_lock(&mutex_);
        screen_.refresh_draw_box(pfb,0,0);
        // pthread_mutex_unlock(&mutex_);
        memcpy(screen_.pfb,pfb,screen_.finfo.smem_len);
        sleep(0.01); 
        gettimeofday(&t1, NULL);
        float mytime = ( float )((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
        //std::cout <<"v4l2_thread "  << " times  " << mytime << "ms\n";
    }
}
void togetherAllBox(double zoom_value,int x0,int y0 )
{
	for (int i = 0; i<boxes.size(); i++) {
		float		bx0 = boxes[i].x0, by0 = boxes[i].y0, bx1= boxes[i].x1, by1 = boxes[i].y1;
			boxes[i].x0= bx0 / zoom_value + x0;
			boxes[i].y0 = by0 / zoom_value + y0;
			boxes[i].x1 = bx1 / zoom_value + x0;
			boxes[i].y1 = by1/ zoom_value + y0;
		   boxes_all.push_back(boxes[i]);
	}
}

int main(int argc, char* argv[])
{
    cout<<"CPU_THREAD_CNT+(GPU_THREAD_CNT+1)/2:"<<CPU_THREAD_CNT+(GPU_THREAD_CNT+1)/2<<endl;
    std::string in_video_file;
    std::string out_video_file;
    get_param_mssd_video_knn(in_video_file,out_video_file);
    std::cout<<"input video: "<<in_video_file<<"\noutput video: "<<out_video_file<<std::endl;
    std::string dev_num;
    get_param_mms_V4L2(dev_num);
    std::cout<<"open "<<dev_num<<std::endl;

    cv::VideoCapture capture;
    capture.open(in_video_file.c_str());
    capture.set(CV_CAP_PROP_FOURCC, cv::VideoWriter::fourcc ('M', 'J', 'P', 'G'));
    IMG_WID = capture.get(CV_CAP_PROP_FRAME_WIDTH);
    IMG_HGT = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
    frame.create(IMG_HGT,IMG_WID,CV_8UC3);
    frame = Mat::zeros(IMG_HGT,IMG_WID,CV_8UC3);
    const std::string root_path = get_root_path();
    std::string proto_file;
    std::string model_file;
    const char* pproto_file;
    const char* pmodel_file;
    proto_file = root_path + DEF_PROTO;
    model_file = root_path + DEF_MODEL;
    screen_.init((char *)"/dev/fb0",640,480);
    pfb = (unsigned int *)malloc(screen_.finfo.smem_len);
    v4l2_.init(dev_num.c_str(),640,480);
    v4l2_.open_device();
	v4l2_.init_device();
	v4l2_.start_capturing();
    /* do not let GPU run concat */
    setenv("GPU_CONCAT", "0", 1);
    /* using GPU fp16 */
    setenv("ACL_FP16", "1", 1);
    /* default CPU device using 0,1,2,3 */
    setenv("TENGINE_CPU_LIST", "2", 1);
    /* using fp32 or int8 */
    setenv("KERNEL_MODE", "2", 1);

    // init tengine
    init_tengine();
    if(request_tengine_version("0.9") < 0)
        return -1;

    img_h = 300;
    img_w = 300;
    img_size = img_h * img_w * 3;
    pproto_file = proto_file.c_str();
    pmodel_file = model_file.c_str();
    int node_idx = 0;
    int tensor_idx = 0;
    int dims[] = {1, 3, img_h, img_w};
    // thread 0 for cpu 2A72
    const struct cpu_info* p_info = get_predefined_cpu("rk3399");
    int a72_list[] = {4};
    set_online_cpu(( struct cpu_info* )p_info, a72_list, sizeof(a72_list) / sizeof(int));
    create_cpu_device("a72", p_info);
    int a72_list01[] = {5};
    set_online_cpu(( struct cpu_info* )p_info, a72_list01, sizeof(a72_list01) / sizeof(int));
    create_cpu_device("a7201", p_info);
    // thread 3 for cpu 4A53
    int a53_list[] = {0,1,2};
    set_online_cpu(( struct cpu_info* )p_info, a53_list, sizeof(a53_list) / sizeof(int));
    create_cpu_device("a53", p_info);

    for(int i=0;i<CPU_THREAD_CNT+(GPU_THREAD_CNT+1)/2;i++)
    {
        graph[i] = create_graph(NULL, "caffe", pproto_file, pmodel_file);
        input_data[i] = ( float* )malloc(sizeof(float) * img_size);
        input_tensor[i] = get_graph_input_tensor(graph[i], node_idx, tensor_idx);
        if(input_tensor[i] == nullptr)
            printf("Get input node failed : node_idx: %d, tensor_idx: %d\n", node_idx, tensor_idx);   
        set_tensor_shape(input_tensor[i], dims, 4);
    }
#if GPU_THREAD_CNT>=1
    set_graph_device(graph[0], "acl_opencl");
#endif
#if (CPU_THREAD_CNT>=1)
    if(set_graph_device(graph[(GPU_THREAD_CNT+1)/2], "a72") < 0)
        std::cerr << "set device a72 failed\n";
#endif
#if (CPU_THREAD_CNT>=2)
    if(set_graph_device(graph[(GPU_THREAD_CNT+1)/2+1], "a7201") < 0)
         std::cerr << "set device a7201 failed\n";
#endif
#if (CPU_THREAD_CNT>=3)
    if(set_graph_device(graph[(GPU_THREAD_CNT+1)/2+2], "a53") < 0)
        std::cerr << "set device a53 failed\n";
#endif

	pthread_t threads_v4l2;
	int rc = pthread_create(&threads_v4l2, NULL, v4l2_thread, NULL);
    pthread_detach(threads_v4l2);
    for(int i=0;i<CPU_THREAD_CNT+(GPU_THREAD_CNT+1)/2;i++)
    {
        int ret_prerun = prerun_graph(graph[i]);
        if(ret_prerun < 0)
            std::printf("prerun failed\n"); 
    }

    while(1){

        for(int i=0;i<CPU_THREAD_CNT+(GPU_THREAD_CNT+1)/2;i++)
            frame_[i] = frame.clone();
      
 
        struct timeval t0_, t1_;
        float total_time = 0.f;
  
        pthread_t threads_c[CPU_THREAD_CNT];      
        gettimeofday(&t0_, NULL);
        boxes_all.clear();

 #if GPU_THREAD_CNT>=1
        pthread_t threads_gpu;
        pthread_create(&threads_gpu, NULL, gpu_pthread, NULL);
#endif

       for(int i=0;i<CPU_THREAD_CNT;i++)
            pthread_create(&threads_c[i], NULL, cpu_pthread, (void*)& cpu_num_[i]);

        for(int i=0;i<CPU_THREAD_CNT;i++)
           pthread_join(threads_c[i],NULL);
 #if GPU_THREAD_CNT>=1

       pthread_join(threads_gpu,NULL);
#endif
        togetherAllBox(1,0,0);
        screen_.v_draw.clear();
        for (int i = 0; i<boxes_all.size(); i++) {
            draw_box box_tmp{Point(boxes_all[i].x0,boxes_all[i].y0),Point(boxes_all[i].x1,boxes_all[i].y1),0};
        screen_.v_draw.push_back(box_tmp);
        }

        gettimeofday(&t1_, NULL);
        float mytime = ( float )((t1_.tv_sec * 1000000 + t1_.tv_usec) - (t0_.tv_sec * 1000000 + t0_.tv_usec)) / 1000;
        std::cout <<"thread_done"  << " times  " << mytime << "ms\n";
        std::cout <<" --------------------------------------------------------------------------\n";
        //cv::imshow("MSSD", frame);
        //cv::waitKey(10) ;

    }
    for(int i=0;i<CPU_THREAD_CNT+(GPU_THREAD_CNT+1)/2;i++)
    {
        release_graph_tensor(out_tensor[i]);
        release_graph_tensor(input_tensor[i]);
        postrun_graph(graph[i]);
        destroy_graph(graph[i]);
        free(input_data[i]);
    }
    release_tengine();

    return 0;
}
