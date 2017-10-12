#include<opencv2\opencv.hpp>
#include<opencv2\face.hpp>
#include<iostream> 
#include "Kinect.h"
#include "ImageHelper.h"
using namespace std;
using namespace cv;
using namespace cv::face;
int main() {
	vector<Mat> images;
	vector<int> labels;
	Size size = Size(200,200);
	string filename = "G:\\BodyRecognitionAlgorithem\\att_faces.csv";
	//create algorithm eigenface recognizer
	Ptr<FaceRecognizer> model = LBPHFaceRecognizer::create(1, 8, 8, 8, 1000);
	ImageHelper::read_images(filename, images, labels, size);
	cout << "Training begins...." << endl;
	model->train(images, labels);
	cout << "Training ends...." << endl;

	IKinectSensor   * mySensor = nullptr;
	GetDefaultKinectSensor(&mySensor);  //��ȡ��Ӧ��
	mySensor->Open();           //�򿪸�Ӧ��

	IColorFrameSource   * mySource = nullptr;   //ȡ�ò�ɫ����
	mySensor->get_ColorFrameSource(&mySource);

	int height = 0, width = 0;
	IFrameDescription   * myDescription = nullptr;  //ȡ�ò�ɫ���ݵķֱ���
	mySource->get_FrameDescription(&myDescription);
	myDescription->get_Height(&height);
	myDescription->get_Width(&width);
	myDescription->Release();

	IColorFrameReader   * myReader = nullptr;
	mySource->OpenReader(&myReader);    //�򿪲�ɫ���ݵ�Reader

	IColorFrame * myFrame = nullptr;
	Mat frame(height, width, CV_8UC4);   //����ͼ�����
	Mat gray;

	CascadeClassifier cascade;
	cascade.load("./cascades/haarcascade_frontalface_default.xml");
	while (1)
	{
		if (myReader->AcquireLatestFrame(&myFrame) == S_OK) //ͨ��Reader���Ի�ȡ���µ�һ֡������ݣ��������֡��,���ж��Ƿ�ɹ���ȡ
		{
			myFrame->CopyConvertedFrameDataToArray(height * width * 4, (BYTE *)frame.data, ColorImageFormat_Bgra); //�Ȱ����ݴ���8λ��ͼ�������
			cvtColor(frame, gray, COLOR_BGR2GRAY);
			Mat gray;
			//�������ڴ����������������  
			vector<Rect> faces(0);
			cvtColor(frame, gray, COLOR_BGR2GRAY);
			cascade.detectMultiScale(gray, faces, 1.3, 5);
			//vector<Mat> images_new;
			//vector<int> labels_new;

			for (int i = 0; i < faces.size(); i++)
			{
				//region of interest
				Rect face_i = faces[i];

				//crop the roi from gray image
				Mat face = gray(face_i);

				//resizing the cropped image to suit to database image sizes
				Mat face_resized;
				cv::resize(face, face_resized, size, 1.0, 1.0, INTER_LINEAR);

				//recognizing what faces detected
				int label = -1; double confidence = 0;
				model->predict(face_resized, label, confidence);

				cout <<"label:" <<label<< ", confidencde: " << confidence << endl;

				//drawing green rectagle in recognize face
				rectangle(frame, face_i, CV_RGB(0, 255, 0), 1);
				string text = "Detected";
				int pos_x = max(face_i.tl().x - 10, 0);
				int pos_y = max(face_i.tl().y - 10, 0);

				stringstream ss;
				ss << label;
				string   s = ss.str();
				//name the person who is in the image
				putText(frame, s, Point(pos_x, pos_y), FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);

				//images_new.push_back(face_resized);
				//labels_new.push_back(36);
			}
			//model->update(images_new, labels_new);
			imshow("face recognition", frame);
			myFrame->Release();
		}
		if (waitKey(30) == VK_ESCAPE)
			break;
	}
	myReader->Release();        //�ͷŲ��õı������ҹرո�Ӧ��
	myDescription->Release();
	mySource->Release();
	mySensor->Close();
	mySensor->Release();
}