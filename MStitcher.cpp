#include "MStitcher.h"

Corner::Corner(int width, int height)
{
	leftUp.x = 0;
	leftUp.y = 0;

	leftLow.x = 0;
	leftLow.y = height - 1;

	rigthUp.x = width - 1;
	rigthUp.y = 0;

	rightLow.x = width - 1;
	rightLow.y = height - 1;;
}


void Corner::move(int x, int y)
{
	leftUp.x += x;
	leftUp.y += y;


	leftLow.x += x;
	leftLow.y += y;



	rigthUp.x += x;
	rigthUp.y += y;


	rightLow.x += x;
	rightLow.y += y;


}


void Corner::transPerspective(cv::Mat M)
{
	cv::Mat pt, trans;
	pt = (cv::Mat_<double>(3, 1) << leftUp.x, leftUp.y, 1);
	trans = M * pt;

	leftUp.x = trans.at<double>(0, 0) / trans.at<double>(2, 0);
	leftUp.y = trans.at<double>(1, 0) / trans.at<double>(2, 0);


	pt = (cv::Mat_<double>(3, 1) << leftLow.x, leftLow.y, 1);
	trans = M * pt;

	leftLow.x = trans.at<double>(0, 0) / trans.at<double>(2, 0);
	leftLow.y = trans.at<double>(1, 0) / trans.at<double>(2, 0);

	pt = (cv::Mat_<double>(3, 1) << rigthUp.x, rigthUp.y, 1);
	trans = M * pt;

	rigthUp.x = trans.at<double>(0, 0) / trans.at<double>(2, 0);
	rigthUp.y = trans.at<double>(1, 0) / trans.at<double>(2, 0);

	pt = (cv::Mat_<double>(3, 1) << rightLow.x, rightLow.y, 1);
	trans = M * pt;

	rightLow.x = trans.at<double>(0, 0) / trans.at<double>(2, 0);
	rightLow.y = trans.at<double>(1, 0) / trans.at<double>(2, 0);
}


void MStitcher::drawBoundry(cv::Mat& image, int thick)
{
	
	
	for (int i = 0;i < corners.size();i++)
	{
		Corner cn = corners[i];
		cv::Scalar color(rand() % 255, rand() % 255, rand() % 255);

		std::stringstream ss;
		ss << "Picture:" <<i + 1;
		cv::putText(image, ss.str(), cn.leftLow + cv::Point2f(30, -30), CV_FONT_HERSHEY_SIMPLEX, 1, color, thick);
		cv::line(image, cn.leftUp, cn.leftLow, color, thick);
		cv::line(image, cn.leftLow, cn.rightLow, color, thick);
		cv::line(image, cn.rightLow, cn.rigthUp, color, thick);
		cv::line(image, cn.rigthUp, cn.leftUp, color, thick);
	}
}


void MStitcher::clear()
{
	corners.clear();
	stds.clear();
	SSIMscores.clear();
}


std::vector<double> MStitcher::getSSIM()
{
	if (SSIMscores.size() > 0)
		return SSIMscores;
}


double MStitcher::getMeanSSIM()
{
	double sum = 0;
	for (int i = 0;i < SSIMscores.size();i++)
	{
		sum += SSIMscores[i];
	}

	return sum / double(SSIMscores.size());
}


double MStitcher::calcDeformation()
{
	double sum = 0;
	for (int i = 1;i < corners.size();i++)
	{
		Corner cn = corners[i];
		
		double lenL = sqrt((cn.leftUp.x - cn.leftLow.x) * (cn.leftUp.x - cn.leftLow.x)
			+ (cn.leftUp.y - cn.leftLow.y) * (cn.leftUp.y - cn.leftLow.y));
		double lenR = sqrt((cn.rigthUp.x - cn.rightLow.x) * (cn.rigthUp.x - cn.rightLow.x)
			+ (cn.rigthUp.y - cn.rightLow.y) * (cn.rigthUp.y - cn.rightLow.y));


		double ratio = lenL / lenR;
		sum += log(ratio)*log(ratio);
	}

	return sum / double(corners.size());
	
}


std::vector<double> MStitcher::getDev()
{
	if (stds.size() > 0)
		return stds;
}


std::vector<Corner> MStitcher::getCorners()
{
	if (corners.size() > 0)
	{
		return corners;
	}
	else
	{
		return std::vector<Corner>();
	}
}

int FindGoodMatches(const std::vector<cv::KeyPoint>& keyPointQue,
	const std::vector<cv::KeyPoint>& keyPointsTrain,
	const std::vector<cv::DMatch>& matches,
	std::vector<cv::Point2f>& goodPointsQuery,
	std::vector<cv::Point2f>& goodPointsTrain,
	double ratio)
{
	double minDist = matches[0].distance;
	for (int i = 0;i < matches.size();i++)
	{
		cv::DMatch m = matches[i];
		if (m.distance < minDist)
		{
			minDist = m.distance;
		}
	}

	// 找出优秀匹配点
	double threshold = minDist * ratio + 200;
	for (int i = 0;i < matches.size();i++)
	{
		cv::DMatch m = matches[i];
		if (m.distance < threshold)
		{
			goodPointsQuery.push_back(keyPointQue[m.queryIdx].pt);
			goodPointsTrain.push_back(keyPointsTrain[m.trainIdx].pt);
		}
	}

	return goodPointsQuery.size();
}


void MStitcher::setFocal(float f)
{
	if (f > 0.6 && f<1.2)
	{
		ratio = f;
		clear();
	}
}


void MStitcher::setSeamMode(Seam mode)
{
	mode_seam = mode;
}


void MStitcher::CylinderProject(const cv::Mat& Input, cv::Mat& Output)
{
	int width = Input.cols;
	int height = Input.rows;
	
	float f = ratio * width;

	double angle = atan(Input.cols / 2 / f);//视场角的一半
	int trans_width = f * angle * 2;//投影后宽度
	int trans_height = Input.rows;//投影后高度


	cv::Mat imgBorder;

	
	cv::Mat out = cv::Mat::zeros(trans_height, trans_width, CV_8UC3);

	// 双线性插值
	for (int y = 0;y < trans_height;y++)
	{
		cv::Vec3b* ptr = out.ptr<cv::Vec3b>(y);
		for (int x = 0;x < trans_width;x++, ptr++)
		{

			double angle_h = (x - double(trans_width) / 2) / f;
			double angle_v = atan((y - double(trans_height) / 2) / f);

			double r = f / cos(angle_h);
			double x0 = width / 2 + tan(angle_h) * f;
			double y0 = height / 2 + tan(angle_v) * r;

			if (y0 < 1 || y0 >= height - 2)
			{
				continue;
			}

			int x1 = int(x0), x2 = int(x0 + 1);
			int y1 = int(y0), y2 = int(y0 + 1);

			//if (y1 < 0)
			//	system("pause");
			cv::Vec3b left_up = Input.at<cv::Vec3b>(y1, x1);
			cv::Vec3b left_down = Input.at<cv::Vec3b>(y1, x2);
			cv::Vec3b right_up = Input.at<cv::Vec3b>(y2, x1);
			cv::Vec3b right_down = Input.at<cv::Vec3b>(y2, x2);

			uchar b_lu = left_up[0], b_ld = left_down[0];
			uchar g_lu = left_up[1], g_ld = left_down[1];
			uchar r_lu = left_up[2], r_ld = left_down[2];

			double dertaY = y0 - y1;
			uchar b1 = b_lu + (b_ld - b_lu) * dertaY;
			uchar g1 = g_lu + (g_ld - g_lu) * dertaY;
			uchar r1 = r_lu + (r_ld - r_lu) * dertaY;

			uchar b_ru = right_up[0], b_rd = right_down[0];
			uchar g_ru = right_up[1], g_rd = right_down[1];
			uchar r_ru = right_up[2], r_rd = right_down[2];

			uchar b2 = b_ru + (b_rd - b_ru) * dertaY;
			uchar g2 = g_ru + (g_rd - g_ru) * dertaY;
			uchar r2 = r_ru + (r_rd - r_ru) * dertaY;

			//std::cout << "b1=" << b1 << "  b2=" << b2 << "\n";// "  x2=" << x2 << " y1=" << y1 << " y2=" << y2 << "\n";

			double dertaX = x0 - x1;
			(*ptr)[0] = cv::saturate_cast<uchar>(b1 + dertaX * (b2 - b1));
			(*ptr)[1] = cv::saturate_cast<uchar>(g1 + dertaX * (g2 - g1));
			(*ptr)[2] = cv::saturate_cast<uchar>(r1 + dertaX * (r2 - r1));

		}
	}
	/*
	调试用
	cv::resize(out, out, cv::Size(0, 0), 0.5, 0.5);
	cv::imshow("cylinder", out);
	cvWaitKey();*/
	Output = out;
}

// 加权间接平差
cv::Mat adjust(const cv::Mat& A, const cv::Mat& l, const cv::Mat& P)
{
	// 条件平差
	cv::Mat AT, lT;
	cv::transpose(A, AT);
	cv::transpose(l, lT);

	cv::Mat ATPA = AT * P * A;

	cv::Mat inv_ATPA;
	cv::invert(ATPA, inv_ATPA);

	// 参数
	return inv_ATPA * AT * P * l;
	
}

// 间接平差
cv::Mat adjust(const cv::Mat& A, const cv::Mat& l)
{
	// 条件平差
	cv::Mat AT, lT;
	cv::transpose(A, AT);
	cv::transpose(l, lT);

	cv::Mat ATA = AT * A;

	cv::Mat inv_ATA;
	cv::invert(ATA, inv_ATA);

	// 参数
	return inv_ATA * AT * l;

}

// 计算中误差
double computeSSE(const std::vector<cv::Point2f>& srcPoints, 
	const std::vector<cv::Point2f>& dstPoints, const cv::Mat& M)
{
	double sse = 0;
	for (int i = 0;i < srcPoints.size();i++)
	{
		cv::Point2f srcPt = srcPoints[i];
		cv::Point2f dstPt = dstPoints[i];

		cv::Mat srcPt_vec = (cv::Mat_<double>(3, 1) << srcPt.x, srcPt.y, 1);
		cv::Mat srcPt_trans = M * srcPt_vec;

		double x = srcPt_trans.at <double>(0, 0);
		double y = srcPt_trans.at <double>(1, 0);
		double z = srcPt_trans.at <double>(2, 0);
		double errX = dstPt.x - x / z;
		double errY = dstPt.y - y / z;

		sse += errX * errX + errY * errY;

	}

	return sse;
}


float findHomography(const std::vector<cv::Point2f> srcPoints, 
	const std::vector<cv::Point2f> dstPoints, 
	cv::Mat& homo, const cv::Mat& image,
	double thrd = 10, double eps = 0.001, 
	double ratio = 0.5, int max_iteration = 2000)
{
	cv::Mat X;


	int totalNum = srcPoints.size();
	std::cout << "筛选前总点数：" << totalNum << "\n";

	int adjust_num = 5;

	int max_vote = 0;
	int max_iter = std::min(int(log(eps) / log(1 - pow(ratio, adjust_num))), max_iteration);


	cv::Mat best_X;
	
	srand((unsigned)time(NULL));

	cv::Mat A_total = cv::Mat(0, 8, CV_64FC1);
	cv::Mat l_total = cv::Mat(0, 1, CV_64FC1);
	// 总的系数阵
	for (int i = 0;i < totalNum;i++)
	{
		cv::Point2f src = srcPoints[i];
		cv::Point2f dst = dstPoints[i];

		float Xsrc = src.x, Ysrc = src.y;
		float Xdst = dst.x, Ydst = dst.y;

		cv::Mat A_ = (cv::Mat_<double>(2, 8) << -Xsrc, -Ysrc, -1, 0, 0, 0, Xdst * Xsrc, Xdst * Ysrc,
			0, 0, 0, -Xsrc, -Ysrc, -1, Ydst * Xsrc, Ydst * Ysrc);
		cv::Mat l_ = (cv::Mat_<double>(2, 1) << -Xdst, -Ydst);

		cv::vconcat(A_total, A_, A_total);
		cv::vconcat(l_total, l_, l_total);

	}

	// 找出投票数最多的模型
	int iter = 0;
	for (;iter < max_iter && max_vote < totalNum * ratio;iter++)
	{
		cv::Ptr<int> ptIdx = new int[adjust_num];
		int count = 0;
		
		while (count < adjust_num)//随机生成10个不一样的数
		{
			int rd = rand() % totalNum;
			int is_repeat = false;
			for (int i = 0;i < count;i++)
			{
				if (rd == ptIdx[i])
				{
					is_repeat = true;
					break;
				}
			}
			if (!is_repeat)
			{
				ptIdx[count] = rd;
				count++;
			}
		}


		cv::Mat A = cv::Mat(0, 8, CV_64FC1);
		cv::Mat l = cv::Mat(0, 1, CV_64FC1);
		
		// 系数阵
		for (int i = 0;i < adjust_num;i++)
		{
			cv::Point2f src = srcPoints[ptIdx[i]];
			cv::Point2f dst = dstPoints[ptIdx[i]];

			float Xsrc = src.x, Ysrc = src.y;
			float Xdst = dst.x, Ydst = dst.y;

			cv::Mat A_ = (cv::Mat_<double>(2, 8) << -Xsrc, -Ysrc, -1, 0, 0, 0, Xdst * Xsrc, Xdst * Ysrc,
													0, 0, 0, -Xsrc, -Ysrc, -1, Ydst * Xsrc, Ydst * Ysrc);
			cv::Mat l_ = (cv::Mat_<double>(2, 1) << -Xdst, -Ydst);

			

			cv::vconcat(A, A_, A);
			cv::vconcat(l, l_, l);
			

		}

		
		// 间接平差
		X = adjust(A, l);

		cv::Mat V = A_total * X - l_total;
		cv::Mat VT;
		cv::transpose(V, VT);


		double* ptr = VT.ptr<double>(0);
		int vote = 0;
		for (int i = 0;i < totalNum;i++, ptr += 2)
		{
			double v1 = *ptr;
			double v2 = *(ptr+1);
			double error = v1 * v1 + v2 * v2;

			if (error < thrd)
				vote++;
		}
		if (vote > max_vote)
		{
			best_X = X.clone();
			max_vote = vote;
		}

	}


	// 用最好的模型筛选掉离群点
	cv::Mat V = A_total * best_X - l_total;
	cv::Mat VT;
	cv::transpose(V, VT);


	// 用内点重新计算
	std::vector<cv::Point2f> srcIn, dstIn;
	double* ptr = VT.ptr<double>(0);
	for (int i = 0;i < totalNum;i++, ptr += 2)
	{
		double v1 = *ptr;
		double v2 = *(ptr + 1);
		double error = v1 * v1 + v2 * v2;

		if (error < thrd)
		{
			srcIn.push_back(srcPoints[i]);
			dstIn.push_back(dstPoints[i]);
		}
	}
	

	// 构造系数阵
	cv::Mat A_in = cv::Mat(0, 8, CV_64FC1);
	cv::Mat l_in = cv::Mat(0, 1, CV_64FC1);
	cv::Mat P_in = cv::Mat(srcIn.size() * 2, srcIn.size() * 2, CV_64FC1);

	cv::Mat imgGray, imgWeight;
	cv::cvtColor(image, imgGray, CV_BGR2GRAY);
	cv::Laplacian(imgGray, imgWeight, CV_8UC1, 5);
	for (int i = 0;i < srcIn.size();i++)
	{
		cv::Point2f src = srcIn[i];
		cv::Point2f dst = dstIn[i];

		float Xsrc = src.x, Ysrc = src.y;
		float Xdst = dst.x, Ydst = dst.y;

		cv::Mat A_ = (cv::Mat_<double>(2, 8) << -Xsrc, -Ysrc, -1, 0, 0, 0, Xdst * Xsrc, Xdst * Ysrc,
			0, 0, 0, -Xsrc, -Ysrc, -1, Ydst * Xsrc, Ydst * Ysrc);
		cv::Mat l_ = (cv::Mat_<double>(2, 1) << -Xdst, -Ydst);

		int weight = pow(int(imgWeight.at<uchar>(round(src.y), round(src.x))),1);
		P_in.at<double>(2*i, 2*i) = weight;
		P_in.at<double>(2*i+1, 2*i+1) = weight;

		cv::vconcat(A_in, A_, A_in);
		cv::vconcat(l_in, l_, l_in);

	}

	//X = adjust(A_in, l_in);
	X = adjust(A_in, l_in, P_in);

	std::cout << "RANSAC内点个数为: " << srcIn.size() << "\n";
	std::cout << "RANSAC共迭代次数：" << iter << "\n";
	float f1 = X.at<double>(0, 0);
	float g1 = X.at<double>(1, 0);
	float h1 = X.at<double>(2, 0);
	float f2 = X.at<double>(3, 0);
	float g2 = X.at<double>(4, 0);
	float h2 = X.at<double>(5, 0);
	float f0 = X.at<double>(6, 0);
	float g0 = X.at<double>(7, 0);

	cv::Mat H = (cv::Mat_<double>(3, 3) << f1, g1, h1, f2, g2, h2, f0, g0, 1);
	homo = H;

	// 计算最终标准差
	double SSE = computeSSE(srcIn, dstIn, H);
	double sigma = SSE / (2 * srcIn.size() - 8);
	return sqrt(sigma);
}


void ComputeMBR(const cv::Mat& M, const cv::Mat& img,
	double& upper, double& bottom, double& left, double& right)
{
	cv::Mat upl = (cv::Mat_<double>(3, 1) << 0, 0, 1);
	cv::Mat upr = (cv::Mat_<double>(3, 1) << img.cols, 0, 1);
	cv::Mat lowl = (cv::Mat_<double>(3, 1) << 0, img.rows, 1);
	cv::Mat lowr = (cv::Mat_<double>(3, 1) << img.cols, img.rows, 1);


	upl = M * upl;
	upr = M * upr;
	lowl = M * lowl;
	lowr = M * lowr;

	std::vector<cv::Mat> corners;
	corners.push_back(upl);
	corners.push_back(upr);
	corners.push_back(lowl);
	corners.push_back(lowr);

	// 找出边界
	upper = upl.at<double>(1, 0);
	bottom = upl.at<double>(1, 0);
	left = upl.at<double>(0, 0);
	right = upl.at<double>(0, 0);
	for (int i = 0;i < 4;i++)
	{
		cv::Mat tmp = corners[i];
		float x = tmp.at<double>(0, 0) / tmp.at<double>(2, 0);//齐次坐标转化
		float y = tmp.at<double>(1, 0) / tmp.at<double>(2, 0);
		if (y < upper)
		{
			upper = y;
		}
		if (y > bottom)
		{
			bottom = y;
		}
		if (x < left)
		{
			left = x;
		}
		if (x > right)
		{
			right = x;
		}
	}
}


// 将img1拼接到img2上
cv::Mat MStitcher::Stitch2Image(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& M)
{
	double up, bottom, left, right;
	ComputeMBR(M, img1, up, bottom, left, right);

	// 计算各个方向画布扩大的像素
	int left_span = std::max(0, int(-left + 0.5));
	int right_span = std::max(0, int(right + 0.5) - img2.cols);
	int up_span = std::max(0, int(-up + 0.5));
	int bottom_span = std::max(0, int(bottom + 0.5) - img2.rows);

	// 更新已拼接图像的四角坐标
	for (int i = 0;i < corners.size();i++)
	{
		Corner cn = corners[i];
		cn.move(left_span, up_span);
		corners[i] = cn;
	}


	// 用于img1平移
	cv::Mat matrix = (cv::Mat_<double>(3, 3) <<
		1, 0, left_span,
		0, 1, up_span,
		0, 0, 1);
	matrix = matrix * M;


	Corner cn(img1.cols, img1.rows);
	cn.transPerspective(matrix);
	corners.push_back(cn);


	int final_width = img2.cols + right_span + left_span;
	int final_height = img2.rows + up_span + bottom_span;

	cv::Mat canvas;
	cv::warpPerspective(img1, canvas, matrix, cv::Size(final_width, final_height), CV_INTER_NN);


	double alpha;//trans的权重
	double overlay;
	bool flag;
	if (left_span < right_span)
	{
		flag = true; // true 则拼接在右边
		overlay = right - left - right_span;
	}
	else
	{
		flag = false;// false 则拼在左边
		overlay = right - left - left_span;
	}


	// 拼接
	for (int i = 0;i < img2.rows;i++)
	{
		const cv::Vec3b* p = img2.ptr<cv::Vec3b>(i);

		// 加偏移量
		cv::Vec3b* p_canvas = &canvas.ptr<cv::Vec3b>(i + up_span)[left_span];

		for (int j = 0;j < img2.cols;j++)
		{
			cv::Vec3b bgr = *p;
			cv::Vec3b& bgr_canvas = *p_canvas;

			// 如果两张图像都不为黑色无像素区域
			if (!(bgr[0] == 0 && bgr[1] == 0 && bgr[2] == 0) && !(bgr_canvas[0] == 0 && bgr_canvas[1] == 0 && bgr_canvas[2] == 0)) 
			{
				switch (mode_seam)
				{
				case Seam::SEAM_COS:
				{
					if (flag == true)// 拼接在右边
					{
						int d1 = j - (img2.cols - overlay);
						int d2 = overlay - d1;
						if (d2 < 0)
							alpha = 0;
						else if (d1 < 0)
							alpha = 1;
						else
						{
							alpha = 0.5 * (1 - cos(3.1415926535 * d1 / overlay));
						}

					}
					else
					{
						int d2 = j;
						int d1 = overlay - d2;
						if (d2 < 0)
							alpha = 0;
						else if (d1 < 0)
							alpha = 1;
						else
						{
							alpha = 0.5 * (1 - cos(3.1415926535 * d1 / overlay));

						}
					}
					break;
				}
				case Seam::SEAM_LINEAR:
				{
					if (flag == true)// 拼接在右边
					{
						
						alpha = std::max(0.0, (j - (img2.cols - overlay)) / overlay);
					}
					else
					{
						alpha = std::max(0.0, 1 - j / overlay);
					}

					break;
				}
				default:
					break;
				}
				

				for (int k = 0;k < 3;k++)
				{
					bgr_canvas[k] = uchar(alpha * bgr_canvas[k] + (1 - alpha) * bgr[k]);
				}

			}

			// 如果画布为黑色无像素区域
			else if (bgr_canvas[0] == 0 && bgr_canvas[1] == 0 && bgr_canvas[2] == 0)
			{
				for (int k = 0;k < 3;k++)
				{
					bgr_canvas[k] = bgr[k];
				}
			}
			p++;
			p_canvas++;
		}
	}


	cv::Mat imgGray1, imgGray2;
	cv::cvtColor(canvas, imgGray1, CV_BGR2GRAY);
	cv::cvtColor(img2, imgGray2, CV_BGR2GRAY);
	
	double mean1 = 0, mean2 = 0;
	int count = 0;//计算重叠部分像素个数;
	for (int i = 0;i < img2.rows;i++)
	{
		uchar* p2 = imgGray2.ptr<uchar>(i);
		// 加偏移量
		uchar* p1 = &imgGray1.ptr<uchar>(i + up_span)[left_span];

		for (int j = 0;j < img2.cols;j++)
		{

			// 如果两张图像都不为黑色无像素区域
			if (*p1 != 0 && *p2 != 0)
			{
				mean1 += double(*p1);
				mean2 += double(*p2);
				count++;
			}
			p1++;
			p2++;
		}
	}

	mean1 /= double(count);
	mean2 /= double(count);
	
	double var1 = 0, var2 = 0;
	double covariance = 0;
	for (int i = 0;i < img2.rows;i++)
	{
		uchar* p2 = imgGray2.ptr<uchar>(i);
		// 加偏移量
		uchar* p1 = &imgGray1.ptr<uchar>(i + up_span)[left_span];

		for (int j = 0;j < img2.cols;j++)
		{

			// 如果两张图像都不为黑色无像素区域
			if (*p1 != 0 && *p2 != 0)
			{
				var1 += pow((*p1 - mean1), 2);
				var2 += pow((*p2 - mean2), 2);
				covariance += (*p1 - mean1) * (*p2 - mean2);
			}
			p1++;
			p2++;
		}
	}

	var1 = sqrt(var1/double(count));
	var2 = sqrt(var2/double(count));
	covariance /= double(count);


	//评价精度的指标的常数
	double K1 = 0.01, K2 = 0.03;
	double L = 255;
	double C1 = pow((K1 * L), 2);
	double C2 = pow((K2 * L), 2);
	double ssim = (2 * mean1 * mean2 + C1) * (2 * covariance + C2) /
		((mean1 * mean1 + mean2 * mean2 + C1) * (var1 * var1 + var2 * var2 + C2));
	SSIMscores.push_back(ssim);
	return canvas;
}


float MStitcher::Stitch(const std::vector<cv::Mat> images, cv::Mat& out,
	double threshold, double eps,
	double ratio, int max_iteration)
{
	std::vector<cv::Mat> images_cyld;

	// 交换顺序，以中间图像为基准
	int center_idx = images.size() / 2;
	cv::Mat img = images[center_idx];
	cv::Mat cyld;
	CylinderProject(img, cyld);

	images_cyld.push_back(cyld);

	for (int i = center_idx+1;i < images.size();i++)
	{
		cv::Mat img = images[i];
		cv::Mat cyld;
		CylinderProject(img, cyld);

		images_cyld.push_back(cyld);
	}
	for (int i = center_idx - 1;i >=0;i--)
	{
		cv::Mat img = images[i];
		cv::Mat cyld;
		CylinderProject(img, cyld);

		images_cyld.push_back(cyld);
	}


	cv::Mat imgTrans, imgBase;
	cv::Mat imgTransGray, imgBaseGray;


	cv::Ptr<cv::Feature2D> f2d = cv::xfeatures2d::SIFT::create();
	cv::BFMatcher matcher = cv::BFMatcher();

	std::vector<cv::KeyPoint> kpBase, kpTrans;// 特征点
	cv::Mat descBase, descTrans;// 描述子

	float sum = 0;//求平均标准差

	imgBase = images_cyld[0];

	corners.push_back(Corner(imgBase.cols, imgBase.rows));

	for (int i = 1;i < images_cyld.size();i++)
	{
		imgTrans = images_cyld[i];

		cv::cvtColor(imgBase, imgBaseGray, CV_BGR2GRAY);
		cv::cvtColor(imgTrans, imgTransGray, CV_BGR2GRAY);

		f2d->detectAndCompute(imgBaseGray, cv::noArray(), kpBase, descBase);
		f2d->detectAndCompute(imgTransGray, cv::noArray(), kpTrans, descTrans);

		std::vector<cv::DMatch> matches;
		matcher.match(descTrans, descBase, matches);

		if (matches.size() < 5) // 匹配失败
		{
			break;
		}



		std::vector<cv::KeyPoint> kpsrc, kpdst;
		std::vector<cv::DMatch> goodMatches;
		// 筛选掉肯定错误的点
		for (int k = 0;k < matches.size();k++)
		{
			cv::DMatch m = matches[k];
			cv::KeyPoint kpt = kpTrans[m.queryIdx];
			cv::KeyPoint kpd = kpBase[m.trainIdx];
			if (abs(kpt.pt.y - kpd.pt.y) < imgTrans.rows / 6)
			{
				kpsrc.push_back(kpt);
				kpdst.push_back(kpd);
				cv::DMatch mm(m);
				mm.trainIdx = kpsrc.size() - 1;
				mm.queryIdx = kpsrc.size() - 1;
				goodMatches.push_back(mm);
			}
		}


		std::vector<cv::Point2f> goodPointsSrc, goodPointsDst;
		FindGoodMatches(kpsrc, kpdst, goodMatches, goodPointsSrc, goodPointsDst, 4);

		/*cv::Mat show;
		cv::drawMatches(imgTrans, kpTrans, imgBase, kpBase, matches, show);
		cv::resize(show, show, cv::Size(0, 0), 0.4, 0.4);
		cv::imshow("s", show);
		cvWaitKey();*/

		cv::Mat M;
		float error = findHomography(goodPointsSrc, goodPointsDst, M,imgTrans,
			threshold,eps,ratio,max_iteration);

		// 中误差大于3像素判断拼接失败
		if (error > 3)
		{
			printf("拼接失败！\n");
			clear();
			return -1;
		}
			
		sum += error;
		printf("第%i张拼接中误差为：%.5f\n\n", i + 1, error);

		imgBase = Stitch2Image(imgTrans, imgBase, M);
	}

	out = imgBase;

	return sum / (images.size()-1);
}
