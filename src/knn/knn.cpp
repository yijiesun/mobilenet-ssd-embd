#include "knn.h"

#include<algorithm>
using namespace cv;
using namespace std;

KNN_BGS::KNN_BGS()
{
	framePixelHistory = NULL;
	frameCnt = 0;
};

KNN_BGS::~KNN_BGS(void)
{
	
}

void KNN_BGS::init(/*VideoCapture &v_capture*/)
{
	int gray = 0;
	FGMask.create(IMG_HGT, IMG_WID, CV_8UC1);// ���ͼ���ʼ��
										// framePixelHistory����ռ�
	framePixelHistory = (PixelHistory*)malloc(IMG_WID*IMG_HGT * sizeof(PixelHistory));

	for (int i = 0; i < IMG_WID*IMG_HGT; i++)
	{
		framePixelHistory[i].gray = (unsigned char*)malloc((history_num + solid_frame) * sizeof(unsigned char));
		framePixelHistory[i].IsBG = (unsigned char*)malloc((history_num + solid_frame) * sizeof(unsigned char));
		memset(framePixelHistory[i].gray, 0, (history_num + solid_frame) * sizeof(unsigned char));
		memset(framePixelHistory[i].IsBG, 0, (history_num + solid_frame) * sizeof(unsigned char));
	}
	/*
	for (int f = 0; f < solid_frame; f++)
	{
		v_capture >> bk;
		cv::cvtColor(bk, bk, CV_BGR2GRAY);
		//bk֡�̻��ڵ�һ��λ��
		for (int i = 0; i < IMG_HGT; i++)
		{
			for (int j = 0; j < IMG_WID; j++)
			{
				gray = bk.at<unsigned char>(i, j);
				// ������ʷֵ
				framePixelHistory[i*IMG_WID + j].gray[f] = gray;
				framePixelHistory[i*IMG_WID + j].IsBG[f] = 1;// ��ǰ����Ϊ�����������ʷ��Ϣ
			}
		}
	}
*/
}

void KNN_BGS::knn_core()
{
	cv::cvtColor(frame, fgray, 6);
	FGMask.setTo(Scalar(255));
	int gray = 0;
	for (int i = 0; i < IMG_HGT; i++)
	{
		for (int j = 0; j < IMG_WID; j++)
		{
			gray = fgray.at<unsigned char>(i, j);
			int fit = 0;
			int fit_bg = 0;
			// �Ƚ�ȷ��ǰ��/����
			for (int n = 0; n < history_num + solid_frame; n++)
			{
				if (fabs((float)gray - framePixelHistory[i*IMG_WID + j].gray[n]) < defaultDist2Threshold)// �ҶȲ���Ƿ�λ���趨��ֵ��
				{
					fit++;
					if (framePixelHistory[i*IMG_WID + j].IsBG[n])// ��ʷ��Ϣ��Ӧ��֮ǰ���ж�Ϊ����
					{
						fit_bg++;
					}
				}
			}
			if (fit_bg >= knnv)// ��ǰ���ж�Ϊ����
			{
				FGMask.at<unsigned char>(i, j) = 0;
			}
			// ������ʷֵ
			int index = frameCnt % history_num + solid_frame;
			framePixelHistory[i*IMG_WID + j].gray[index] = gray;
			framePixelHistory[i*IMG_WID + j].IsBG[index] = fit >= knnv ? 1 : 0;// ��ǰ����Ϊ�����������ʷ��Ϣ
		}
	}
	frameCnt++;
}

void KNN_BGS::postTreatment()
{
	cv::medianBlur(FGMask, FGMask, 3);
	threshold(FGMask, FGMask, 2, 255, 0);
	Mat element = getStructuringElement(MORPH_RECT, Size(dilateRatio, dilateRatio));
	//erode(FGMask, FGMask, element);
	dilate(FGMask, FGMask, element);
	//dilate(FGMask, FGMask, element);
	//if (insideDilate_win_size != 0)
	//{
	//	insideDilate(FGMask, FGMask, insideDilate_win_size, insideDilate_scale);
		//insideDilate(FGMask, FGMask, insideDilate_win_size, insideDilate_scale);
	//}
}

bool sortFun(const cv::Rect &p1, const cv::Rect &p2)
{
	return p1.width * p1.height > p2.width * p2.height;//��������  
}

void KNN_BGS::getTopRects(vector<Rect> &rects0, vector<Rect> &rects)
{
	for (int i = 0; i<rects0.size(); i++)
	{
		if (i >= useTopRect)
			break;
		rects.push_back(rects0[i]);
	}
}

void KNN_BGS::processRects()
{
#if FIND_ROI
	std::vector<cv::Rect> boundRectTmp;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarcy;
	findContours(FGMask.clone(), contours, hierarcy, 0, 1); //��������
	int rec_nums = buildAndClearSmallContors(contours, boundRectTmp, minContorSize);
	std::sort(boundRectTmp.begin(), boundRectTmp.end(), sortFun);
	getTopRects(boundRectTmp, boundRect);
	mergeRecs(boundRect, knn_over_percent);
	mergeRecs(boundRect, knn_over_percent);
	paddingRecs(boundRect, padSize);
	drawRecs(showImg, boundRect, Scalar(0, 255, 0, 0));
	boundRectTmp.shrink_to_fit();
	contours.shrink_to_fit();
	hierarcy.shrink_to_fit();
#else
	Rect rect_tmp(0,0, IMG_WID, IMG_HGT);
	boundRect.push_back(rect_tmp);
#endif
}

int KNN_BGS::buildAndClearSmallContors(vector<vector<Point>> &contours, vector<Rect> &rects, int size)
{
	int x0 = 0, y0 = 0, w0 = 0, h0 = 0;
	int  cnt = 0;
	for (int i = 0; i<contours.size(); i++)
	{
		Rect rect_tmp = boundingRect((Mat)contours[i]);
		x0 = rect_tmp.x;
		y0 = rect_tmp.y;
		w0 = rect_tmp.width;
		h0 = rect_tmp.height;
		if (w0 * h0 >= size)
		{
			rects.push_back(rect_tmp);
			cnt++;
		}
	}
	return cnt;
}

void KNN_BGS::mergeRecs(vector<Rect> &rects, float percent)
{
	int len = rects.size();
	int new_len = 0;
	int ptr = 0;
	Rect tmp;

	for (;;)
	{
		if (ptr >= len)
			break;

		for (int i = 0; i < len; i++)
		{
			if (ptr < 0 || ptr >= rects.size() || i < 0 || i >= rects.size())
				break;
			if (i == ptr)
				continue;
			if (DecideOverlap(rects[ptr], rects[i], tmp) >= percent)
			{
				rects[ptr] = tmp;
				if (rects.begin() + i <= rects.end())
				{
					rects.erase(rects.begin() + i);
					i--;
				}
				else
					break;
			}

		}
		ptr++;
		len = rects.size();
	}

}

void KNN_BGS::drawRecs(Mat & img, vector<Rect> &rects, const Scalar& color)
{
	int x0 = 0, y0 = 0, w0 = 0, h0 = 0;
	for (int i = 0; i< rects.size(); i++)
	{
		x0 = rects[i].x;  //��õ�i����Ӿ��ε����Ͻǵ�x����
		y0 = rects[i].y; //��õ�i����Ӿ��ε����Ͻǵ�y����
		w0 = rects[i].width; //��õ�i����Ӿ��εĿ���
		h0 = rects[i].height; //��õ�i����Ӿ��εĸ߶�

		if (w0 <= tooSmalltoDrop || h0 <= tooSmalltoDrop)
			continue;
		rectangle(img, Point(x0, y0), Point(x0 + w0, y0 + h0), color, 2, 8); //���Ƶ�i����Ӿ���
	}
}

float KNN_BGS::DecideOverlap(const Rect &r1, const Rect &r2, Rect &r3)
{
	//r3 = r1;
	int x1 = r1.x;
	int y1 = r1.y;
	int width1 = r1.width;
	int height1 = r1.height;

	int x2 = r2.x;
	int y2 = r2.y;
	int width2 = r2.width;
	int height2 = r2.height;

	int endx = max(x1 + width1, x2 + width2);
	int startx = min(x1, x2);
	int width = width1 + width2 - (endx - startx);

	int endy = max(y1 + height1, y2 + height2);
	int starty = min(y1, y2);
	int height = height1 + height2 - (endy - starty);

	float ratio = 0.0f;
	float Area, Area1, Area2;

	if (width <= 0 || height <= 0)
		return 0.0f;
	else
	{
		Area = width*height;
		Area1 = width1*height1;
		Area2 = width2*height2;
		ratio = max(Area / (float)Area1, Area / (float)Area2);
		r3.x = startx;
		r3.y = starty;
		r3.width = endx - startx;
		r3.height = endy - starty;
	}

	return ratio;
}

void KNN_BGS::paddingRecs(vector<Rect> &rects, int size)
{
	for (int i = 0; i<rects.size(); i++)
	{
		rects[i].x = min(max(rects[i].x - size, 0), IMG_WID);
		rects[i].y = min(max(rects[i].y - size, 0), IMG_HGT);
		rects[i].width = rects[i].x + rects[i].width + 2 * size > IMG_WID ? IMG_WID - rects[i].x : rects[i].width + 2 * size;
		rects[i].height = rects[i].y + rects[i].height + 2 * size > IMG_HGT ? IMG_HGT - rects[i].y : rects[i].height + 2 * size;
	}
}

/*insideDilate:�����ͺ���
*���ص����»�����������ͬʱ�а����ص�б�ʱ��������
*bimg:��ֵͼ��
*win_size:��������
*scale:�б���Ч��������
*/
void KNN_BGS::insideDilate(Mat & bimg, Mat & bout, int win_size, int scale)
{
	for (int w = 0 + win_size; w < IMG_WID - win_size; w++)
	{
		for (int h = 0 + win_size; h < IMG_HGT - win_size; h++)
		{
			Point curr(w, h);
			Point refer;
			Mat tmp;
			int l = 0, r = 0, u = 0, d = 0;
			Mat roi_u(bimg, Rect(w, h - win_size, 1, win_size));
			Mat roi_d(bimg, Rect(w, h, 1, win_size));
			Mat roi_l(bimg, Rect(w - win_size, h, win_size, 1));
			Mat roi_r(bimg, Rect(w, h, win_size, 1));
			u = countNonZero(roi_u);
			d = countNonZero(roi_d);
			l = countNonZero(roi_l);
			r = countNonZero(roi_r);
			uchar* data = bout.ptr<uchar>(h);

			if ((u >= scale && d >= scale) || (l >= scale && r >= scale))
				data[w] = 255;
			else
			{
				if(u+d+r+l <= win_size)
					data[w] = 0;
			}
				
		}
	}
}

void KNN_BGS::set(int *conf)
{
	solid_frame = conf[0];
	history_num = conf[1];
	knnv = conf[2];
	padSize = conf[3];
	minContorSize = conf[4];
	insideDilate_win_size = conf[6];
	insideDilate_scale = conf[7];
}

void KNN_BGS::saveROI()
{
	int x0 = 0, y0 = 0, w0 = 0, h0 = 0;
	for (int i = 0; i< boundRect.size(); i++)
	{
		x0 = boundRect[i].x;  //��õ�i����Ӿ��ε����Ͻǵ�x����
		y0 = boundRect[i].y; //��õ�i����Ӿ��ε����Ͻǵ�y����
		w0 = boundRect[i].width; //��õ�i����Ӿ��εĿ���
		h0 = boundRect[i].height; //��õ�i����Ӿ��εĸ߶�

		if (w0 <= tooSmalltoDrop || h0 <= tooSmalltoDrop)
			continue;
		cv::Mat src_roi = frame(cv::Rect(x0, y0, w0, h0));
		std::stringstream ss;
		
		ss << saveAdress << pos << "_"<< i << ".bmp" ;
		std::string s = ss.str();
		imwrite(s, src_roi);
	}
}