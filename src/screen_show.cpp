#include <stdint.h>
#include <fcntl.h>
#include <string.h>
#include <sys/ioctl.h>
#include <errno.h>
#include <sys/mman.h>
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
#include "screen/screen.h"
#include "v4l2/v4l2.h" 
#include <unistd.h>  
#include <linux/fb.h>
#include <linux/videodev2.h>


#define TRUE            (1)
#define FALSE           (0)
#define FILE_VIDEO      "/dev/video10"
#define FBDEVICE		"/dev/fb0"
#define IMAGEWIDTH      640
#define IMAGEHEIGHT     480
#define FRAME_NUM       4

V4L2 v4l2_;
SCREEN screen_;
unsigned int screen_pos_x,screen_pos_y;
unsigned int * pfb;
cv::Mat rgb;
draw_box box_tmp{Point(150,150),Point(400,400),0};
bool quit;
 pthread_mutex_t mutex_;
void *v4l2_thread(void *threadarg);
void my_handler(int s);


int main(int argc, char *argv[])
{
    
	screen_pos_x = 0;
	screen_pos_y = 0;
	screen_.init((char *)"/dev/fb0",640,480);
 
   quit = false;
    pthread_mutex_init(&mutex_, NULL);

    struct sigaction sigIntHandler;
 
   sigIntHandler.sa_handler = my_handler;
   sigemptyset(&sigIntHandler.sa_mask);
   sigIntHandler.sa_flags = 0;
 
   sigaction(SIGINT, &sigIntHandler, NULL);

    rgb.create(480,640,CV_8UC3);
    Mat frame;
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

    v4l2_.init(dev_num.c_str(),640,480);
    v4l2_.open_device();
	v4l2_.init_device();
	v4l2_.start_capturing();
    pfb = (unsigned int *)malloc(screen_.finfo.smem_len);

	pthread_t threads_v4l2;
	int rc = pthread_create(&threads_v4l2, NULL, v4l2_thread, NULL);
#if 0
    while(1){
 
        //pthread_mutex_lock(&mutex_);
		bgr_frame = bgr.clone();
       // pthread_mutex_unlock(&mutex_);

		screen_.show_bgr_mat_at_screen(bgr_frame,50,50);

        if (quit)
             break;
    }
#endif
    pthread_join(threads_v4l2,NULL);
    screen_.uninit();
	v4l2_.stop_capturing();
	v4l2_.uninit_device();
	v4l2_.close_device();
    cout<<"save video success!:  "<<video_name<<endl;
    return 0;
}


void *v4l2_thread(void *threadarg)
{
    
    screen_.v_draw.push_back(box_tmp);
	while (1)
	{
            v4l2_.read_frame_argb(pfb,rgb,screen_.vinfo.xres_virtual,screen_pos_x,screen_pos_y);
            screen_.refresh_draw_box(pfb,screen_pos_x,screen_pos_y);
            memcpy(screen_.pfb,pfb,screen_.finfo.smem_len);
       // screen_.draw_line(200,200,600,200);

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
 