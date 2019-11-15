#include "screen.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <iomanip>

using namespace std;
using namespace cv;

SCREEN::SCREEN()
{
        fb = -1;
        ret = -1;
        pfb = NULL;
}
SCREEN::~SCREEN()
{

}

int SCREEN::init(char *dev)
{
        strncpy(dev_name,dev,200);
        fb = open(dev_name, O_RDWR);
        if (fb < 0)
        {
            perror("open");
            return -1;
        }
        printf("open %s success \n", dev_name);

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

}

void SCREEN::show_bgr_mat_at_screen(Mat &in,int pos_x,int pos_y)
{
    uint32_t color = 0;
    for (int h=0; h < in.rows; h++){
        for(int w=0;w <in.cols; w++){
            color = (0xff000000) | ((in.data[h*in.cols*3+w*3+2] << 16) & 0x00ff0000) | ((in.data[h*in.cols*3+w*3+1] << 8) & 0x0000ff00) | ((in.data[h*in.cols*3+w*3]&0x000000ff));
            *(pfb+(h+pos_y)*vinfo.xres_virtual+w+pos_x)  = color;
   
        }
    }
}