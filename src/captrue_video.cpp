
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <sys/time.h>
#include <stdio.h>
#include "config.h"
#include "v4l2/v4l2.h"  
#include <unistd.h> 

V4L2 v4l2_;
cv::Mat rgb;
bool quit;
 pthread_mutex_t mutex_;
void *v4l2_thread(void *threadarg);
void my_handler(int s);

int main(int argc, char *argv[])
{
    VideoWriter outputVideo;
   quit = false;
    pthread_mutex_init(&mutex_, NULL);

    struct sigaction sigIntHandler;
 
   sigIntHandler.sa_handler = my_handler;
   sigemptyset(&sigIntHandler.sa_mask);
   sigIntHandler.sa_flags = 0;
 
   sigaction(SIGINT, &sigIntHandler, NULL);

    Mat frame;
    bool is_show_img;
    get_show_img(is_show_img);
    std::cout<<"is_show_img "<<is_show_img<<std::endl;
    std::string dev_num,imgfld,video_fld;
    get_param_mms_V4L2(dev_num);
    get_captrue_save_data_floder(imgfld,video_fld);
    std::cout<<"open "<<dev_num<<std::endl;

     mkdir(video_fld.c_str(), 0775);
    Size sWH = Size( 640,480);
    
    char tmp_buf[200];
    getTimesSecf(tmp_buf);
    string video_name = video_fld+tmp_buf+".avi";
    cout<<"save video: "<<video_name<<endl;
	bool ret = outputVideo.open(video_name, cv::VideoWriter::fourcc ('M', 'P', '4', '2'), 25, sWH);
    v4l2_.init(dev_num.c_str(),640,480);
    v4l2_.open_device();
	v4l2_.init_device();
	v4l2_.start_capturing();

    rgb.create(480,640,CV_8UC3);
	pthread_t threads_v4l2;
	int rc = pthread_create(&threads_v4l2, NULL, v4l2_thread, NULL);

    while(1){
 
        pthread_mutex_lock(&mutex_);
         frame = rgb.clone();
        pthread_mutex_unlock(&mutex_);
        outputVideo.write(frame);
        if(is_show_img)
        {
            cv::imshow("MSSD", frame);
            cv::waitKey(10) ;
        }
        if (quit)
             break;
    }
    pthread_join(threads_v4l2,NULL);
    outputVideo.release();
	v4l2_.stop_capturing();
	v4l2_.uninit_device();
	v4l2_.close_device();
    cout<<"save video success!:  "<<video_name<<endl;
    return 0;
}


void *v4l2_thread(void *threadarg)
{
	while (1)
	{
        pthread_mutex_lock(&mutex_);
        v4l2_.read_frame(rgb);
        pthread_mutex_unlock(&mutex_);
        sleep(0.01); 
        if (quit)
            pthread_exit(NULL);
    }
}


void my_handler(int s)
{
            quit = true;
            cout<<"Caught signal "<<s<<" quit="<<quit<<endl;
}
 