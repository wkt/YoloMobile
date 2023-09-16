//
// Created by wkt on 2023/3/11.
//

#ifndef YOLOMOBILE_YOLO_H
#define YOLOMOBILE_YOLO_H

#include <vector>
#include <string>
#include <ncnn/net.h>

#ifdef ANDROID
class AAssetManager;
#endif

#define MAX(a,b) ((a)>=(b)?(a):(b))

class Yolo {
public:
    struct Point{
        float x=-1;
        float y=-1;
        float p=0; // point confidence

        Point();
        Point(float _x,float _y,float _p);
    };

    struct BBox{
        int                 c; //class id
        float               p; //bbox confidence
        float               x; //top left x
        float               y; //top left y
        float               w; // bbox width
        float               h; // bbox height
        std::vector<Point>  keyPoints;
        BBox();
        float area() const {return w * h;}
    };

private:
    class IMPL;
    IMPL *impl;

public:
    Yolo();
    ~Yolo();
    Yolo* inputSize(int size);
    Yolo* inputName(const std::string& name);
    Yolo* boxThreshold(float box);
    Yolo* iouThreshold(float iou);
    Yolo* outputName(const std::string& name);
    Yolo* outputStride(int stride);
    Yolo* outputAnchors(const ncnn::Mat& anchors);
    Yolo* numKeypoint(int num);
    Yolo* ver(int v);
    Yolo* kpThreshold(float kp);
    int load(const std::string& param,const std::string& bin);

#ifdef ANDROID
    int load(AAssetManager*asset,const std::string& param,const std::string& bin);
#endif

    int forward(const ncnn::Mat& inp,std::vector<BBox>&boxes);
};


#endif //YOLOMOBILE_YOLO_H
