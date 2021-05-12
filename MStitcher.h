#pragma once
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"


class Corner
{
public:
	Corner(int width, int height);
	~Corner() {};

	// 透视变换后的坐标
	void move(int x, int y);
	void transPerspective(cv::Mat M);


	cv::Point2f leftUp;
	cv::Point2f leftLow;
	cv::Point2f rigthUp;
	cv::Point2f rightLow;

};


class MStitcher
{
public:
	MStitcher() {};
	~MStitcher() {};
	
	/*
	影像融合加权方式
	*/
	static enum Seam
	{
		SEAM_LINEAR = 0,
		SEAM_COS = 1,
	};

	/*
	* RANSAC+间接平差 计算单应矩阵
	thrd:判断是否为内点的阈值，5为经验值
	eps:找不到最佳模型的概率
	ratio:内点比例
	max_iterration:RANSAC最大迭代次数
	*/
	float Stitch(const std::vector<cv::Mat> images, cv::Mat& out,
		double threshold = 5, double eps = 0.001,
		double ratio = 0.5, int max_iteration = 2000);

	//绘制图框
	void drawBoundry(cv::Mat& image, int thick);

	//设置f
	void setFocal(float f);

	//设置融合模式
	void setSeamMode(Seam mode);

	void clear();// 清空内存

	//返回拼接后所有图像的角点
	std::vector<Corner> getCorners();

	//返回拼接结果的SSIM指数
	std::vector<double> getSSIM();

	double getMeanSSIM();

	// 计算变形系数
	double calcDeformation();

	//计算标准差
	std::vector<double> getDev();

private:
	float ratio = 0.84;
	Seam mode_seam = SEAM_COS;
	std::vector<Corner> corners;
	std::vector<double> SSIMscores;
	std::vector<double> stds;

	// 拼接两张
	cv::Mat Stitch2Image(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& M);

	void CylinderProject(const cv::Mat& Input, cv::Mat& Output);
};

