#include "iostream"
#include "MStitcher.h"

int main(int argc, char** argv)
{
	float f;
	if (argc != 2)
	{
		std::cout << "Usage: [*.exe] [f]" << "\n";
		std::cout << "set f to 0.84\n";
		f = 0.84;
	}
	else
	{
		f = atof(argv[1]);
	}
	
	std::string dirPath = "images\\";

	double scale = 1;
	std::vector<cv::Mat> images;
	for (int i = 0;i < 8;i++)
	{
		cv::Mat img = cv::imread(dirPath + "\\" + std::to_string(i) + ".jpg");
		if (img.empty())
		{
			printf("Í¼Æ¬¶ÁÈ¡Ê§°Ü!\n");
			return -1;
		}
		cv::resize(img, img, cv::Size(0, 0), scale, scale);
		images.push_back(img);
	}

	MStitcher stitcher;
	stitcher.setFocal(f);
	cv::Mat imgShow;
	stitcher.Stitch(images, imgShow);

	cv::resize(imgShow, imgShow, cv::Size(0, 0), 0.4, 0.4);
	cv::imshow("result", imgShow);
	cv::waitKey();

	system("pause");
	return 0;
}