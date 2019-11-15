#include "v4l2.h"  
using namespace std;
using namespace cv;
#define CLEAR(x) memset(&(x), 0, sizeof(x))
 V4L2:: V4L2()
 {
 }
  V4L2:: ~V4L2()
 {
	stop_capturing();
	uninit_device();
	close_device();
 }

 void  V4L2:: init(const char *dev,int wid,int hgt)
 {
	strncpy(dev_name,dev,200);
	width = wid;
	height = hgt;
	fd = -1; 
	n_buffers = 4;
 }

void V4L2::errno_exit(const char *s)
{
	fprintf(stderr, "%s error %d, %s\n", s, errno, strerror(errno));
}

int V4L2::xioctl(int fh, int request, void *arg)
{
	int r;
	do {
		r = ioctl(fh, request, arg);
	} while (-1 == r && EINTR == errno);
	return r;
}

void  V4L2::process_image(const void *p, int size,Mat &rgb)
{
	//int width = 640;
	//int height = 480;
	//Mat bgr;
	//bgr.create(height,width,CV_8UC3);
	yuyv_to_bgr((unsigned char*) p,rgb.data,width, height);
}


int V4L2::read_frame(Mat &out)
{
	struct v4l2_buffer buf;
	CLEAR(buf);
	
	buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	buf.memory = V4L2_MEMORY_MMAP;

	if (-1 == xioctl(fd, VIDIOC_DQBUF, &buf))
		errno_exit("VIDIOC_DQBUF");
	
	process_image(buffers[buf.index].start, buf.bytesused,out);

	if (-1 == xioctl(fd, VIDIOC_QBUF, &buf))
		errno_exit("VIDIOC_QBUF");

	return 1;
}

void V4L2::stop_capturing(void)
{
	enum v4l2_buf_type type;

	type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	if (-1 == xioctl(fd, VIDIOC_STREAMOFF, &type))
		errno_exit("VIDIOC_STREAMOFF");
}

void V4L2::start_capturing(void)
{
	unsigned int i;
	enum v4l2_buf_type type;

	for (i = 0; i < n_buffers; ++i) {
		struct v4l2_buffer buf;

		CLEAR(buf);
		buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		buf.memory = V4L2_MEMORY_MMAP;
		buf.index = i;

		if (-1 == xioctl(fd, VIDIOC_QBUF, &buf))
			errno_exit("VIDIOC_QBUF");
	}
	type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	if (-1 == xioctl(fd, VIDIOC_STREAMON, &type))
		errno_exit("VIDIOC_STREAMON");
}

void V4L2::uninit_device(void)
{
	unsigned int i;

	for (i = 0; i < n_buffers; ++i)
		if (-1 == munmap(buffers[i].start, buffers[i].length))
			errno_exit("munmap");

	free(buffers);
}



void V4L2::init_mmap(void)
{
	struct v4l2_requestbuffers req;

	CLEAR(req);

	req.count = 4;
	req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	req.memory = V4L2_MEMORY_MMAP;

	if (-1 == xioctl(fd, VIDIOC_REQBUFS, &req)) {
		if (EINVAL == errno) {
			fprintf(stderr, "%s does not support "
				"memory mapping\n", dev_name);
			exit(EXIT_FAILURE);
		}
		else {
			errno_exit("VIDIOC_REQBUFS");
		}
	}

	if (req.count < 2) {
		fprintf(stderr, "Insufficient buffer memory on %s\n",
			dev_name);
		exit(EXIT_FAILURE);
	}

	buffers = (struct buffer *)calloc(req.count, sizeof(*buffers));

	if (!buffers) {
		fprintf(stderr, "Out of memory\n");
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < req.count; ++i) {
		struct v4l2_buffer buf;

		CLEAR(buf);

		buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		buf.memory = V4L2_MEMORY_MMAP;
		buf.index = i;

		if (-1 == xioctl(fd, VIDIOC_QUERYBUF, &buf))
			errno_exit("VIDIOC_QUERYBUF");

		buffers[i].length = buf.length;
		buffers[i].start =
			mmap(NULL /* start anywhere */,
				buf.length,
				PROT_READ | PROT_WRITE /* required */,
				MAP_SHARED /* recommended */,
				fd, buf.m.offset);

		if (MAP_FAILED == buffers[i].start)
			errno_exit("mmap");
	}
}

void V4L2::init_device(void)
{
	struct v4l2_capability cap;
	struct v4l2_format fmt;

	if (-1 == xioctl(fd, VIDIOC_QUERYCAP, &cap)) { 
		if (EINVAL == errno) {
			fprintf(stderr, "%s is no V4L2 device\n",
				dev_name);
			exit(EXIT_FAILURE);
		}
		else {
			errno_exit("VIDIOC_QUERYCAP");
		}
	}

	if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
		fprintf(stderr, "%s is no video capture device\n",
			dev_name);
		exit(EXIT_FAILURE);
	}

	if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
		fprintf(stderr, "%s does not support streaming i/o\n",
			dev_name);
		exit(EXIT_FAILURE);
	}

	CLEAR(fmt);
	fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	fmt.fmt.pix.width = width;
	fmt.fmt.pix.height = height;
	fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
	//fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
	fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;

	if (-1 == xioctl(fd, VIDIOC_S_FMT, &fmt)) 
		errno_exit("VIDIOC_S_FMT");
	init_mmap();
}

void V4L2::open_device(void)
{
	fd = open(dev_name, O_RDWR /* required */ /*| O_NONBLOCK*/, 0);

	if (-1 == fd) {
		fprintf(stderr, "Cannot open '%s': %d, %s\n",
			dev_name, errno, strerror(errno));
		exit(EXIT_FAILURE);
	}
}

void V4L2::close_device(void)
{
	if (-1 == close(fd))
		errno_exit("close");

	fd = -1;
}

void V4L2::yuyv_to_bgr(unsigned char* yuv,unsigned char* rgb,int width, int height )
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
    
      
       	        if(rt0 > 250)  	rt0 = 255;
		if(rt0< 0)    	rt0 = 0;	
 
		if(gt0 > 250) 	gt0 = 255;
		if(gt0 < 0)	gt0 = 0;	
 
		if(bt0 > 250)	bt0 = 255;
		if(bt0 < 0)	bt0 = 0;	
 
		if(rt1 > 250)	rt1 = 255;
		if(rt1 < 0)	rt1 = 0;	
 
		if(gt1 > 250)	gt1 = 255;
		if(gt1 < 0)	gt1 = 0;	
 
		if(bt1 > 250)	bt1 = 255;
		if(bt1 < 0)	bt1 = 0;	
					
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

#if 0
int main(int argc, char *argv[])
{
	
	strcpy(dev_name, argv[1]);
	cout<<dev_name<<endl;
	strcpy(fileDir, "");

	open_device();
	init_device();
	start_capturing();
	mainloop();
	stop_capturing();
	uninit_device();
	close_device();

	return 0;
}
#endif