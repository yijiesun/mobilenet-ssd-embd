#include "config.h"
#include "v4l2/v4l2.h"  
#include <signal.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <unistd.h> 

using namespace std;
using namespace cv;
bool quit;
cv::Mat rgb;
V4L2 v4l2_;
pthread_mutex_t mutex_;
void my_handler(int s);
void *v4l2_thread(void *threadarg);
double moveDetect(Mat &cur, Mat &last, int wid, int hgt);

int main(int argc, char *argv[])
{
    int pushToAVICntRear = 0;
    bool first = true;
    int width;
    int height;
   quit = false;
   struct sigaction sigIntHandler;
   sigIntHandler.sa_handler = my_handler;
   sigemptyset(&sigIntHandler.sa_mask);
   sigIntHandler.sa_flags = 0;
   sigaction(SIGINT, &sigIntHandler, NULL);

    std::string dev_num,save_img_floder,save_video_floder;
    bool is_show_img;
    int save_video_mode,save_img_mode,buff_cnt;
    double move_percent;
    get_param_mms_V4L2(dev_num);
    get_camera_size(width,height);
    get_show_img(is_show_img);
    get_move_percent(move_percent);
    get_move_buff_cnt(buff_cnt);
    get_captrue_save_data_floder(save_img_floder,save_video_floder);
    get_captrue_data_save_video_mode(save_video_mode);
    get_captrue_data_save_img_mode(save_img_mode);
    std::cout<<"open "<<dev_num<<std::endl;
    std::cout<<"width "<<width<<" height "<<height<<std::endl;
    std::cout<<"save_img_floder "<<save_img_floder<<" save_video_floder "<<save_video_floder<<std::endl;
    std::cout<<"is_show_img "<<is_show_img<<std::endl;
    std::cout<<"save_img_mode "<<save_img_mode<<std::endl;
    std::cout<<"move_percent "<<move_percent<<std::endl;
    std::cout<<"save_video_mode "<<save_video_mode<<std::endl;
    std::cout<<"buff_cnt "<<buff_cnt<<std::endl;
    //waitKey(5000);
    mkdir(save_img_floder.c_str(), 0775);
    mkdir(save_video_floder.c_str(), 0775);

    v4l2_.init(dev_num.c_str(),width,height);
    v4l2_.open_device();
	v4l2_.init_device();
	v4l2_.start_capturing();

    cv::Mat frame,frame_last;
    frame.create(height,width,CV_8UC3);
    rgb.create(height,width,CV_8UC3);
	pthread_t threads_v4l2;
	int rc = pthread_create(&threads_v4l2, NULL, v4l2_thread, NULL);
    string dir_son;
	while(1)
	{
        bool isCatchMove = false;
        pthread_mutex_lock(&mutex_);
         frame = rgb.clone();
        pthread_mutex_unlock(&mutex_);
        if (first)
        {
            first = false;
            frame_last = frame.clone();
        }
        if(is_show_img)
            cv::imshow("MSSD", frame);


        double whitePercent = moveDetect(frame, frame_last, width, height);
        if (whitePercent >= move_percent)
        {
            cout<<"catch move --" << whitePercent << endl;
            isCatchMove = true;
        }
         if (pushToAVICntRear == 0 && isCatchMove)
		{
            char tmp_buf[200];
            getTimesSec(tmp_buf);
            dir_son.clear();
            dir_son = save_img_floder+tmp_buf+"/";
            int ret = mkdir(dir_son.c_str(), 0775);
            cout<<"mkdir: "<<dir_son<<" ret="<<ret<<endl;
         }
        if (isCatchMove)
        {
            pushToAVICntRear = buff_cnt;
            cout << "move find ! record next : " << pushToAVICntRear << " frames" << endl;
        }

        if (pushToAVICntRear != 0)
        {
            char time_[200];
            getTimesSecf(time_);
            string img_name = dir_son+time_+".jpg";
            imwrite(img_name.c_str(),frame);
            
            pushToAVICntRear--;
            if (pushToAVICntRear == 0)
            {
                cout  << "save move frames finish!" << endl;
            }	
            else
                cout<<"save img_name "<<img_name<<endl;
        }

        frame_last = frame.clone();
        if(is_show_img)
            cv::waitKey(10);
        if(quit)
            break;
	}


    pthread_join(threads_v4l2,NULL);
    v4l2_.stop_capturing();
    v4l2_.uninit_device();
    v4l2_.close_device();
     cout<<"QUIT COMPLETE!"<<endl;

	return 0;
}

void my_handler(int s)
{
            quit = true;
            cout<<"Caught signal "<<s<<" quit="<<quit<<endl;
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
            break;
        
    }
}
double moveDetect(Mat &cur, Mat &last, int wid, int hgt)
{
	cv::Mat cur_, last_;
	cur_ = cur.clone();
	last_ = last.clone();
	cv::Mat frame_cur_gray, frame_last_gray;
	cv::Mat diff(cv::Size(wid, hgt), CV_8UC1);
	cvtColor(cur, frame_cur_gray, 6);
	cvtColor(last, frame_last_gray, 6);
	absdiff(frame_cur_gray, frame_last_gray, diff);
	int blockSize = 25;
	int constValue = 10;
	cv::Mat diff_thresh;
	cv::adaptiveThreshold(diff, diff_thresh, 255, 0, 1, blockSize, constValue);
	Mat element = getStructuringElement(MORPH_RECT, Size(2, 2));
	//medianBlur(src, dst, 7);
	//GaussianBlur(dst, dst, Size(3, 1), 0.0);
	erode(diff_thresh, diff_thresh, element);
	dilate(diff_thresh, diff_thresh, element);
	//calc white percent
	Mat diff01;
	normalize(diff_thresh, diff01, 0, 1, cv::NORM_MINMAX);
	double sum_diff_roi = countNonZero(diff01);
	double percent = sum_diff_roi / ((double)wid * hgt);

	cout << "move percent: " << percent << endl;

	return percent;

}