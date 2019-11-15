
#ifndef SCREEN_H
#define SCREEN_H
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <getopt.h> /* getopt_long() */
#include <fcntl.h> /* low-level i/o */
#include <unistd.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <linux/fb.h>
#include <linux/videodev2.h>
#include <opencv2/video.hpp>
#include "opencv2/opencv.hpp"  
using namespace std;
using namespace cv;
class SCREEN {
    public :
        SCREEN();
        ~SCREEN();
        int fb = -1;
        int ret = -1;
        char dev_name[200];
        unsigned int * pfb;
        struct fb_fix_screeninfo finfo;
        struct fb_var_screeninfo vinfo;
        int init(char *dev);
        void show_bgr_mat_at_screen(Mat &in,int pos_x,int pos_y);
};

#endif