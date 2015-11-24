#include <iostream>
#include <opencv2\opencv.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <stdint.h>
#include "SelfOrganisingMap.h"

uint8_t getOTSUThreshhold(cv::InputArray _src,cv::InputArray mask);
cv::Mat get2DGaussianKernel(cv::Size size,double sigma);
void GradientVectorFlow(cv::InputArray _src,cv::OutputArray _dst,cv::InputArray mask);
void edgeExtract(cv::Mat& src,cv::Mat& dst);
void equalize(cv::Mat &src,cv::Mat &dst,cv::Mat &mask);
void findOpticDisk(char* image_path,char* image_mask,char* save_path,char* mask_path,int mark);
void train(char* inp,char* inp_mask);

SelfOrganisingMap som(3,1,25);

using namespace cv;
using namespace std;

void trainSOM(cv::Mat &inp,cv::Mat &mask)
{
	if ((inp.channels()!=3))
	{
		exit(-1);
	}
	if(mask.channels()>1)	//	If mask is multichannel desaturate it.
		cv::cvtColor(mask,mask,COLOR_BGR2GRAY);
	if (mask.size() != inp.size())
	{
		exit(-1);
	}

	double min[3];
	double max[3];
	for (int i = 0; i < 3; i++)
	{
		cv::Mat chl;
		cv::extractChannel(inp,chl,i);
		cv::minMaxLoc(chl,&min[i],&max[i]);
	}

	uint8_t* mask_data=(uint8_t*)mask.datastart;
	uint8_t* src_data=(uint8_t*)inp.datastart;
	while (mask_data!=mask.dataend)
	{
		if (*mask_data)
		{
			double train_data[3];
			uint8_t* tmp=src_data;
			for (int i = 0; i < 3; i++)
			{
				train_data[i]=255*((double)tmp[i])/max[i%3];
			}

			som.train(train_data);
		}
		src_data+=3;
		mask_data++;
	}
}

cv::Mat somGradient(cv::Mat &inp,cv::Mat &mask)
{
	uint8_t color_grad[25][3]=
	{
		{0,0,25},
		{0,0,50},
		{0,0,75},
		{0,0,100},
		{0,0,125},
		{0,0,150},
		{0,0,175},
		{0,0,200},
		{0,0,225},
		{0,0,250},
		{0,25,0},
		{0,50,0},
		{0,75,0},
		{0,100,0},
		{0,125,0},
		{0,150,0},
		{0,175,0},
		{0,200,0},
		{0,225,0},
		{0,250,0},
		{50,0,0},
		{100,0,0},
		{150,0,0},
		{200,0,0},
		{250,0,0},
	};

	double min[3];
	double max[3];
	for (int i = 0; i < 3; i++)
	{
		cv::Mat chl;
		cv::extractChannel(inp,chl,i);
		cv::minMaxLoc(chl,&min[i],&max[i]);
	}
	cv::Mat ret(inp.size(),CV_8UC3,0.0);
	uint8_t* mask_data=(uint8_t*)mask.datastart;
	uint8_t* src_data=(uint8_t*)inp.datastart;
	uint8_t* ret_data=(uint8_t*)ret.datastart;
	while (mask_data!=mask.dataend)
	{
		if (*mask_data)
		{
			double calculation_data[3];
			for (int i = 0; i < 3; i++)
			{
				calculation_data[i] = 255*((double)src_data[i])/max[(i%3)];
			}
			som.calculate(calculation_data);
			int active=som.getLowest();
			ret_data[0]=color_grad[active][0];
			ret_data[1]=color_grad[active][1];
			ret_data[2]=color_grad[active][2];
		}
		src_data+=3;
		ret_data+=3;
		mask_data++;
	}
	return ret;
}

cv::Mat somThresh(cv::Mat &inp,cv::Mat &mask)
{
	double min[3];
	double max[3];
	for (int i = 0; i < 3; i++)
	{
		cv::Mat chl;
		cv::extractChannel(inp,chl,i);
		cv::minMaxLoc(chl,&min[i],&max[i]);
	}
	cv::Mat ret(inp.size(),CV_8UC1,0.0);
	uint8_t* mask_data=(uint8_t*)mask.datastart;
	uint8_t* src_data=(uint8_t*)inp.datastart;
	uint8_t* ret_data=(uint8_t*)ret.datastart;
	while (mask_data!=mask.dataend)
	{
		if (*mask_data)
		{
			double calculation_data[3];
			for (int i = 0; i < 3; i++)
			{
				calculation_data[i] = 255*((double)src_data[i])/max[i];
			}
			som.calculate(calculation_data);
			int active=som.getLowest();
			if ((active>=22)&&(active<=25))
			{
				ret_data[0]=255;
			}
			else
			{
				ret_data[0]=0;
			}

		}
		src_data+=3;
		ret_data+=1;
		mask_data++;
	}
	return ret;
}

int main(int argc ,char* argv[])
{
	FILE *fp=fopen("som_data.txt","r");
	som.deserialize(fp);
	if(fp)
		fclose(fp);

	if (argc==6)
	{
		if (!strcmp(argv[1],"O"))
		{
			findOpticDisk(argv[2],argv[3],argv[4],argv[5],1);
		}
		else if (!strcmp(argv[1],"T"))
		{
			findOpticDisk(argv[2],argv[3],argv[4],argv[5],0);
			train(argv[4],argv[5]);
		}

	}


	/*for (int i = 7; i < 8; i++)
	{
	char path[100]={0};
	sprintf(path,"C:\\Users\\Srijan\\Desktop\\glaucoma\\%.2d_g.jpg",i);
	char mask[]="C:\\Users\\Srijan\\Desktop\\glaucoma\\mask.jpg";
	char save_path[100]={0};
	char mask_path[100]={0};
	char grad_path[100]={0};
	sprintf(save_path,"C:\\Users\\Srijan\\Desktop\\training\\%.2d_g.jpg",i);
	sprintf(mask_path,"C:\\Users\\Srijan\\Desktop\\training\\%.2d_gm.jpg",i);
	//sprintf(grad_path,"C:\\Users\\Srijan\\Desktop\\training\\%.2d_gg.jpg",i);

	findOpticDisk(path,mask,save_path,mask_path,1);
	}*/
}

void train(char* inp,char* inp_mask)
{
	//	load all images.
	//std::vector<cv::Mat> img;
	//std::vector<cv::Mat> OpticDiskMask;
	cv::Mat img=cv::imread(inp);
	cv::Mat OpticDiskMask=cv::imread(inp_mask);
	/*for (int j = 1; j < 16; j++)
	{
	std::cout<<"Loading "<<j<<"\n";
	char save_path[100]={0};
	char mask_path[100]={0};

	sprintf(save_path,"C:\\Users\\Srijan\\Desktop\\training\\%.2d_g.jpg",j);
	sprintf(mask_path,"C:\\Users\\Srijan\\Desktop\\training\\%.2d_gm.jpg",j);

	img.push_back(cv::imread(save_path));
	OpticDiskMask.push_back(cv::imread(mask_path));
	//cv::cvtColor(img[j-1],img[j-1],CV_BGR2HLS);
	cv::cvtColor(OpticDiskMask[j-1],OpticDiskMask[j-1],CV_BGR2GRAY);

	}*/
	//for (int i = 0; i < 50; i++)
	//{
	//std::cout<<som.learn<<"\n";
	//for (int j = 1; j < 16; j++)
	//{
	//std::cout<<j<<"\n";

	char grad_path[100]={0};

	sprintf(grad_path,"somgrad.jpg");

	trainSOM(img,OpticDiskMask);
	cv::Mat gradientMap = somGradient(img,OpticDiskMask);

	cv::imwrite(grad_path,gradientMap);
	//}
	FILE* fp=fopen("som_data.txt","w");
	som.serialize(fp);
	if(fp)
		fclose(fp);
	//}
}

void findOpticDisk(char* image_path,char* image_mask,char* save_path,char* mask_path,int mark)
{
	//std::cout<<"Image "<<i<<"\n";

	cv::Mat img			= cv::imread(image_path);
	cv::Mat maskImage	= cv::imread(image_mask);
	cv::Mat channel[3];
	cv::Mat output;
	cv::Size sz=img.size();
	sz.height=(sz.height*700)/sz.width;
	sz.width=700;
	cv::resize(img,img,sz);
	cv::resize(maskImage,maskImage,sz);

	cv::Mat original;
	img.copyTo(original);

	for (int i = 0; i < 3; i++)
		cv::extractChannel(img,channel[i],i);
	cv::extractChannel(maskImage,maskImage,0);
	//cv::imwrite("01_green_channel.jpg",channel[1]);
	cv::Mat gaussianKernel=get2DGaussianKernel(cv::Size(9,9),1.8);
	cv::filter2D(channel[1],output,channel[1].depth(),gaussianKernel);
	//cv::imwrite("02_gaussiam_matched_fiter.jpg",output);
	int t=getOTSUThreshhold(output,maskImage);
	cv::threshold(output,output,t,255,THRESH_BINARY);
	//cv::imwrite("03_otsu_threshold.jpg",output);
	cv::Mat total=cv::Mat(maskImage.rows,maskImage.cols,maskImage.type(),0.0);

	cv::medianBlur(output,output,5);

	for (int i = 0; i < 3; i++)
	{
		channel[i]=channel[i]-output;

		cv::Mat kernel = cv::getStructuringElement(MORPH_ELLIPSE,cv::Size(5,5),cv::Point(0,0));
		cv::morphologyEx(channel[i],channel[i],cv::MORPH_DILATE,kernel);
		cv::morphologyEx(channel[i],channel[i],cv::MORPH_DILATE,kernel);
		cv::morphologyEx(channel[i],channel[i],cv::MORPH_DILATE,kernel);
		cv::morphologyEx(channel[i],channel[i],cv::MORPH_DILATE,kernel);
		cv::morphologyEx(channel[i],channel[i],cv::MORPH_DILATE,kernel);
		cv::morphologyEx(channel[i],channel[i],cv::MORPH_ERODE,kernel);
		cv::morphologyEx(channel[i],channel[i],cv::MORPH_ERODE,kernel);
		cv::morphologyEx(channel[i],channel[i],cv::MORPH_ERODE,kernel);
		cv::morphologyEx(channel[i],channel[i],cv::MORPH_ERODE,kernel);
		cv::morphologyEx(channel[i],channel[i],cv::MORPH_ERODE,kernel);


		cv::Mat tmp;

		edgeExtract(channel[i],tmp);
		total+=tmp;

		cv::insertChannel(channel[i],img,i);
	}
	//cv::imwrite("04_morphological_closed.jpg",img);
	//cv::imshow("abc",total);
	//cv::GaussianBlur(total,total,cv::Size(3,3),5.0,5.0);

	//t=getOTSUThreshhold(total,maskImage);
	//cv::threshold(total,total,t,255,CV_THRESH_BINARY);
	cv::adaptiveThreshold(total,total,255,ADAPTIVE_THRESH_GAUSSIAN_C,THRESH_BINARY,49,-35);
	cv::imwrite("05_sobel_threshold.jpg",total);
	cv::medianBlur(total,output,3);
	//output=total;
	//cv::imshow("abc",output);
	cv::Mat kernel = cv::getStructuringElement(0,cv::Size(3,3),cv::Point(0,0));
	cv::morphologyEx(output,output,cv::MORPH_CLOSE,kernel);

	cv::Mat OpticDiskMask=cv::Mat(maskImage.rows,maskImage.cols,maskImage.type(),0.0);
	std::vector<cv::Vec3f> circles;
	int param1=25;
	int param2=25;
	while (circles.size()==0)
	{
		cv::HoughCircles(output,circles,HOUGH_GRADIENT,1,5,param1,param2,30,55);
		param1--;
		param2--;
	}

	cv::Point c(0,0);
	int mid=img.size().height/2;

	cv::Mat gray;
	cv::cvtColor(original,gray,COLOR_BGR2GRAY);
	double min,max;
	cv::Point m;
	cv::minMaxLoc(gray,&min,&max,0,&m);
	double disk_radius=0;


	for( size_t i = 0; i < circles.size(); i++ )
	{
		cv::Point centre(circles[i][0],circles[i][1]);
		centre-=m;
		cv::Point tmp=m-c;
		if(centre.dot(centre)<=tmp.dot(tmp))
		{
			c.x=circles[i][0];
			c.y=circles[i][1];
			disk_radius=circles[i][2];
		}
	}

	cv::circle( OpticDiskMask, cv::Point(c.x, c.y), disk_radius, cv::Scalar(255,255,255), 4,LINE_AA);
	//cv::imwrite("06_optic_disk.jpg",OpticDiskMask);

	kernel = cv::getStructuringElement(0,cv::Size(80,80),cv::Point(0,0));
	cv::morphologyEx(OpticDiskMask,OpticDiskMask,cv::MORPH_CLOSE,kernel);

	kernel = cv::getStructuringElement(2,cv::Size(10,10),cv::Point(0,0));
	cv::morphologyEx(OpticDiskMask,OpticDiskMask,cv::MORPH_DILATE,kernel);

	//cv::cvtColor(img,img,CV_BGR2HLS);
	//trainSOM(img,OpticDiskMask);

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;

	//contours.clear();
	//hierarchy.clear();
	if(mark==1)
	{
		cv::Mat gradientMap = somThresh(img,OpticDiskMask);
		//cv::imwrite("07_som.jpg",gradientMap);

		//cv::imwrite(grad_path,gradientMap);
		//cv::imwrite("C:\\Users\\Srijan\\Desktop\\gradient_output.jpg",gradientMap);

		//cv::imshow("OUTPUT",output);
		//cv::imshow("OpticDiskMask",OpticDiskMask);
		//cv::imshow("img",gradientMap);
		cv::findContours(gradientMap,contours,hierarchy,RETR_TREE,CHAIN_APPROX_SIMPLE,cv::Point(0,0));

		std::vector<cv::Point> all_points;
		for (int i = 0; i < contours.size(); i++)
		{
			all_points.insert(all_points.end(),contours[i].begin(),contours[i].end());
		}
		if (all_points.size()>=0)
		{
			cv::Point2f center;
			float radius;
			cv::minEnclosingCircle(all_points,center,radius);
			cv::circle(original,center,radius,cv::Scalar(0,0,0),1,LINE_AA);
			std::cout<<radius/disk_radius<<"\n";
		}
		cv::circle( original, cv::Point(c.x, c.y), disk_radius, cv::Scalar(0,0,0), 1, LINE_AA);
	}
	cv::imwrite(save_path,original);
	cv::imwrite(mask_path,OpticDiskMask);

	//cv::imwrite("08_result.jpg",original);

	cv::imshow("cup",original);
	cv::waitKey(16);
}

uint8_t getOTSUThreshhold(cv::InputArray _src,cv::InputArray mask)
{
	cv::Mat	msk	=	mask.getMat();
	cv::Mat	src	=	_src.getMat();
	if (src.type()!=CV_8U)
	{
		return 0;
	}

	if (_src.size()!=mask.size())
		return 0;

	uint8_t* mask_data=(uint8_t*)msk.datastart;
	uint8_t* src_data=(uint8_t*)src.datastart;

	uint8_t histogram[256]={0};
	size_t total_pixels=0;

	//	calculate threshhold.

	while (mask_data!=msk.dataend)
	{
		if (*mask_data++)
		{
			histogram[*src_data]++;
			total_pixels++;
		}
		src_data++;
	}

	double p1,p2,m1,m2;
	double sigma_max=0;

	uint32_t threshhold=0;

	for (int i = 0; i< 256; i++)
	{
		p1=p2=m1=m2=0;
		for (int j = 0; j < i; j++)
		{
			double p=(double)histogram[j]/(double)total_pixels;
			p1+=p;
			m1+=j*p;
		}
		for (int j = i; j < 256; j++)
		{
			double p=(double)histogram[j]/(double)total_pixels;
			p2+=p;
			m2+=j*p;
		}
		double sigma=p1*p2*pow((m1-m2),2);
		if (sigma>sigma_max)
		{
			sigma_max=sigma;
			threshhold=i;
		}
	}
	return threshhold;
}

cv::Mat get2DGaussianKernel(cv::Size size,double sigma)
{
	cv::Mat kernelX=cv::getGaussianKernel(size.width,sigma);
	cv::Mat kernelY=cv::getGaussianKernel(size.height,sigma);

	cv::Mat gaussianKernel = (kernelX*kernelY.t() - ((double)1/size.area()));

	double* data=(double*)gaussianKernel.datastart;
	gaussianKernel=-gaussianKernel/(data[(size.height/2) + ((size.width/2)*(size.height))]);
	return gaussianKernel;
}

void GradientVectorFlow(cv::InputArray _src,cv::OutputArray _dst,cv::InputArray mask)
{
	cv::Size sz=_src.size();
	cv::Mat ext1(sz,CV_32F);
	cv::Mat ext2(sz,CV_32F);

}

void edgeExtract(cv::Mat& src,cv::Mat& dst)
{
	cv::Mat sbl[6];
	int index=0;
	double scale[]={1.0f,-1.0f};
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j <= 1; j++)
		{
			for (int k = 0; k <= 1; k++)
			{
				if ((j==0)&&(k==0))
				{
					continue;
				}
				cv::Sobel(src,sbl[index++],src.depth(),j,k,3,scale[i]);
			}
		}
	}
	dst=sbl[0]+sbl[1]+sbl[2]+sbl[3]+sbl[4]+sbl[5];
}

void equalize(cv::Mat &src,cv::Mat &dst,cv::Mat &mask)
{
	cv::Mat output(src.size(),src.type());
	cv::Mat m;
	cv::extractChannel(mask,m,0);
	for (int channel_no = 0; channel_no < src.channels(); channel_no++)
	{
		cv::Mat channel;
		cv::extractChannel(src,channel,channel_no);

		uint32_t hist[256]={0};
		uint32_t total_pixels=0;

		uint8_t* mask_data =(uint8_t*) m.datastart;
		for (uint8_t *i = (uint8_t*)channel.datastart; i != channel.dataend; i++)
		{
			if(*mask_data>0)
			{
				hist[*i]++;
				total_pixels++;
			}
			mask_data++;
		}

		uint32_t cumulative=0;
		uint8_t new_map[256];
		for (int i = 0; i < 256; i++)
		{
			cumulative+=hist[i];
			new_map[i]=256*((double)cumulative/(double)total_pixels);
		}

		uint8_t	*output_data=(uint8_t*)output.datastart+channel_no;
		for (uint8_t *i = (uint8_t*)channel.datastart; i != channel.dataend; i++)
		{
			*output_data=new_map[*i];
			output_data+=output.channels();
		}
	}
	dst=output;
}