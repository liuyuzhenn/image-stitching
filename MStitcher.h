#pragma once
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"


class Corner
{
public:
	Corner(int width, int height);
	~Corner() {};

	// ͸�ӱ任�������
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
	Ӱ���ںϼ�Ȩ��ʽ
	*/
	static enum Seam
	{
		SEAM_LINEAR = 0,
		SEAM_COS = 1,
	};

	/*
	* RANSAC+���ƽ�� ���㵥Ӧ����
	thrd:�ж��Ƿ�Ϊ�ڵ����ֵ��5Ϊ����ֵ
	eps:�Ҳ������ģ�͵ĸ���
	ratio:�ڵ����
	max_iterration:RANSAC����������
	*/
	float Stitch(const std::vector<cv::Mat> images, cv::Mat& out,
		double threshold = 5, double eps = 0.001,
		double ratio = 0.5, int max_iteration = 2000);

	//����ͼ��
	void drawBoundry(cv::Mat& image, int thick);

	//����f
	void setFocal(float f);

	//�����ں�ģʽ
	void setSeamMode(Seam mode);

	void clear();// ����ڴ�

	//����ƴ�Ӻ�����ͼ��Ľǵ�
	std::vector<Corner> getCorners();

	//����ƴ�ӽ����SSIMָ��
	std::vector<double> getSSIM();

	double getMeanSSIM();

	// �������ϵ��
	double calcDeformation();

	//�����׼��
	std::vector<double> getDev();

private:
	float ratio = 0.84;
	Seam mode_seam = SEAM_COS;
	std::vector<Corner> corners;
	std::vector<double> SSIMscores;
	std::vector<double> stds;

	// ƴ������
	cv::Mat Stitch2Image(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& M);

	void CylinderProject(const cv::Mat& Input, cv::Mat& Output);
};

