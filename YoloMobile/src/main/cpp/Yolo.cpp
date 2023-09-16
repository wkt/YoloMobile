//
// Created by wkt on 2023/3/11.
//

#include <ncnn/cpu.h>
#include "Yolo.h"
#define TAG "YOLO"
#include "jlog.h"

static inline std::string to_string(const ncnn::Mat&m)
{
    std::string s = typeid(m).name();
    char bf[64] = {0};
    s = "<" + s;
    s += " w="+std::to_string(m.w);
    s += " h="+std::to_string(m.h);
    s += " d="+std::to_string(m.d);
    s += " c="+std::to_string(m.c);
    s += " dims="+std::to_string(m.dims);
    s += " cstep="+std::to_string(m.cstep);
    s += " elemsize="+std::to_string(m.elemsize);
    s += " elempack="+std::to_string(m.elempack);
    s += " refcount="+std::to_string(*m.refcount);
    snprintf(bf, sizeof(bf)-1,"%p",m.data);
    s += " data=" + std::string(bf);
    s += ">";
    return s;
}

static inline float intersection_area(const Yolo::BBox& a, const Yolo::BBox& b)
{
    float x1=a.x;
    float y1=a.y;
    float x2=a.x+a.w;
    float y2=b.y+a.h;

    float x3 = b.x;
    float y3 = b.y;
    float x4 = b.x+b.w;
    float y4 = b.y+b.h;
    // gives bottom-left point
    // of intersection rectangle
    float x5 = std::max(x1, x3);
    float y5 = std::max(y1, y3);

    // gives top-right point
    // of intersection rectangle
    float x6 = std::min(x2, x4);
    float y6 = std::min(y2, y4);

    // no intersection
    if (x5 > x6 || y5 > y6) {
        return {};
    }


    // gives top-left point
    // of intersection rectangle
    float x7 = x5;
    float y7 = y5;


    // gives bottom-right point
    // of intersection rectangle
    float x8 = x6;
    float y8 = y6;

    return (x8-x7)*(y8-y7);
}

static void qsort_descent_inplace(std::vector<Yolo::BBox>& objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].p;

    while (i <= j)
    {
        while (objects[i].p > p)
            i++;

        while (objects[j].p < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(objects[i], objects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(objects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(objects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Yolo::BBox>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Yolo::BBox>& faceobjects, std::vector<int>& picked, float nms_threshold, bool agnostic = false)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].area();
    }

    for (int i = 0; i < n; i++)
    {
        const Yolo::BBox& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Yolo::BBox& b = faceobjects[picked[j]];

            if (!agnostic && a.c != b.c)
                continue;

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

static inline void generate_proposals_v8(const ncnn::Mat& blob, const int& in_w,const int& in_h,
                                      float prob_threshold, std::vector<Yolo::BBox>& objects,
                                      int nkpt=0)
{
    //LOGD("channel: %s",to_string(blob.depth(0)).data());
    const int cj = 4;
    float x,y,w,h;
    int ki;
    int max_cj = -1;
    float max_p = 0;
    const int num_class = blob.h - cj - nkpt*3;
    for(int i=0;i< blob.w;i++){
        x = *(blob.row(0)+i);
        y = *(blob.row(1)+i);
        w = *(blob.row(2)+i);
        h = *(blob.row(3)+i);
        x -= w/2;
        y -= h/2;
        max_cj = -1;
        max_p = 0;
        if(x<0 || y < 0 || w <=0 || h<=0 )continue;
        Yolo::BBox b;
        for(int j=0;j< num_class;j++){
            const float p = *(blob.row(cj+j)+i);
            if(p>=prob_threshold && p>max_p){
                max_p = p;
                max_cj=j;
                for(int k=0;k<nkpt;k++){
                    Yolo::Point kp(-1,-1,0);
                    ki=cj+j+k*3;
                    kp.p = *(blob.row(ki)+i);
                    if(kp.p>1e-5){
                        kp.x = *(blob.row(ki+1)+i);
                        kp.y = *(blob.row(ki+2)+i);
                    }
                    /// LOGD("k: %d, ki: %d,x=%.2f,y=%.2f,p=%.6f",k,ki, kp.x,kp.y,kp.p);
                    b.keyPoints.push_back(kp);
                }
            }
        }
        if(max_cj<0)continue;
        b.x = x;
        b.y = y;
        b.w = w;
        b.h = h;
        b.p = max_p;
        b.c = max_cj;
        objects.push_back(b);
    }
}

static inline void generate_proposals(const ncnn::Mat& anchors, int stride,
                                      const int& in_w,const int& in_h, const ncnn::Mat& feat_blob,
                                      float prob_threshold, std::vector<Yolo::BBox>& objects,
                                      int nkpt=0)
{
    const int num_grid = feat_blob.h;

    int num_grid_x;
    int num_grid_y;
    if (in_w > in_h)
    {
        num_grid_x = in_w / stride;
        num_grid_y = num_grid / num_grid_x;
    }
    else
    {
        num_grid_y = in_h / stride;
        num_grid_x = num_grid / num_grid_y;
    }

    const int num_class = feat_blob.w - 5-nkpt*3;

    const int num_anchors = anchors.w / 2;
    //LOGD("num_class: %d",num_class);
    for (int q = 0; q < num_anchors; q++)
    {
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];

        const ncnn::Mat feat = feat_blob.channel(q);

        for (int i = 0; i < num_grid_y; i++)
        {
            for (int j = 0; j < num_grid_x; j++)
            {
                const float* featptr = feat.row(i * num_grid_x + j);
                float box_confidence = sigmoid(featptr[4]);
                if (box_confidence >= prob_threshold)
                {
                    // find class index with max class score
                    int class_index = 0;
                    float class_score = -FLT_MAX;
                    for (int k = 0; k < num_class; k++)
                    {
                        float score = featptr[5 + k];
                        if (score > class_score)
                        {
                            class_index = k;
                            class_score = score;
                        }
                    }
                    float confidence = box_confidence * sigmoid(class_score);
                    if (confidence >= prob_threshold)
                    {
                        float dx = sigmoid(featptr[0]);
                        float dy = sigmoid(featptr[1]);
                        float dw = sigmoid(featptr[2]);
                        float dh = sigmoid(featptr[3]);

                        float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                        float pb_cy = (dy * 2.f - 0.5f + i) * stride;

                        float pb_w = pow(dw * 2.f, 2) * anchor_w;
                        float pb_h = pow(dh * 2.f, 2) * anchor_h;

                        float x0 = pb_cx - pb_w * 0.5f;
                        float y0 = pb_cy - pb_h * 0.5f;
                        float x1 = pb_cx + pb_w * 0.5f;
                        float y1 = pb_cy + pb_h * 0.5f;

                        Yolo::BBox obj;
                        obj.x = x0;
                        obj.y = y0;
                        obj.w = x1 - x0;
                        obj.h = y1 - y0;
                        obj.c = class_index;
                        obj.p = confidence;

                        for(int ik=0;ik<nkpt;ik++){
                            int i0 = 6+ik*3;
                            float dx = (featptr[i0]);
                            float dy = (featptr[i0+1]);
                            float prob = sigmoid(featptr[i0+2]);
                            //LOGD("i0=%d,dx=%.3f,dy=%.3f",i0, dx,dy);
                            float kx = (dx*2. - 0.5 + j) * stride;
                            float ky = (dy*2. - 0.5 + i) * stride;

                            obj.keyPoints.emplace_back(kx,ky,prob);
                        }

                        objects.push_back(obj);
                    }
                }
            }
        }
    }
}

struct FixInfo{
    float scale;
    int srcW;
    int srcH;
    int dW;
    int dH;
};


static inline FixInfo fixImageSize(const ncnn::Mat& src,ncnn::Mat& dst,
                                   int input_size=640)
{
    FixInfo info={0};
    int tw = 0;
    int th = 0;
    double scale = 1.0;
    info.srcW = src.w;
    info.srcH = src.h;
    if(src.w != input_size || src.h != input_size){
        ncnn::Mat resized;
        scale = input_size*1.0/MAX(src.w,src.h);
        tw = int(scale*src.w+0.499);
        th = int(scale*src.h+0.499);
        ncnn::resize_bilinear(src,resized,tw,th);
        int dw0 = int((input_size-tw)/2.);
        int dw1 = input_size-tw-dw0;
        int dh0 = int((input_size-th)/2.);
        int dh1 = input_size-th-dh0;
        info.dW = dw0;
        info.dH = dh0;
        ncnn::copy_make_border(resized,dst, dh0,dh1,dw0,dw1,ncnn::BORDER_CONSTANT,114.f);
    }else{
        dst = src;
        info.dW = 0;
        info.dH = 0;
    }
    info.scale = scale;
    return info;
}

class Yolo::IMPL{
public:
    float                       box_thr;
    float                       iou_thr;
    int                         inputSize;
    int                         nkpt;
    std::string                 inputName;
    std::vector<ncnn::Mat>      anchors;
    std::vector<std::string>    outputNames;
    std::vector<int>            outputStrides;
    ncnn::Net                   net;
    int                         ver;
    float                       kp_thr;
    IMPL():box_thr(0.5f),iou_thr(0.45f)
          ,inputSize(416),inputName("images")
          ,anchors(),outputNames()
          ,outputStrides(),nkpt(0),net()
          ,ver(7),kp_thr(0.1){}
};

Yolo::Yolo():impl(new IMPL) {
    //vulkan is bad on ncnn
    impl->net.opt.use_vulkan_compute = false;
}

Yolo::~Yolo() {
    delete impl;
    impl = nullptr;
}

int Yolo::load(const std::string &param, const std::string &bin) {
    impl->net.load_param(param.data());
    int r = impl->net.load_model(bin.data());
    return r;
}

#ifdef ANDROID
int Yolo::load(AAssetManager *asset, const std::string &param, const std::string &bin) {
    impl->net.load_param(asset,param.data());
    int r = impl->net.load_model(asset,bin.data());
    return r;
}
#endif

int Yolo::forward(const ncnn::Mat &inp, std::vector<BBox> &boxes) {
    //LOGD("input_size: %d,input_name: %s, box_thr: %.3f, iou_thr: %.3f",impl->inputSize,impl->inputName.data(),impl->box_thr,impl->iou_thr);
    ncnn::Mat in;
    FixInfo info = fixImageSize(inp,in,impl->inputSize);
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in.substract_mean_normalize(nullptr, norm_vals);

    std::vector<ncnn::Mat>outs;
    std::vector<Yolo::BBox> proposals;

    ncnn::Extractor ex = impl->net.create_extractor();
    ex.set_num_threads(ncnn::get_big_cpu_count());
    ex.input(impl->inputName.data(),in);
    const auto& outputNames = impl->outputNames;
    for(int i=0;i<outputNames.size();i++){
        const std::string &n=outputNames[i];
        ncnn::Mat out;
        ex.extract(n.data(),out);
        outs.push_back(out);
    }
    const auto& anchors = impl->anchors;
    const auto& strides = impl->outputStrides;
    for(int i=0;i<anchors.size();i++){
        std::vector<Yolo::BBox> objects;
        if(impl->ver == 8){
            generate_proposals_v8(outs[i],impl->inputSize,impl->inputSize,impl->box_thr, objects,impl->nkpt);
        }else{
            generate_proposals(anchors[i], strides[i], impl->inputSize,impl->inputSize, outs[i], impl->box_thr, objects,impl->nkpt);
        }
        proposals.insert(proposals.end(), objects.begin(), objects.end());
    }

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, impl->iou_thr);

    int count = picked.size();

    boxes.resize(count);
    for (int i = 0; i < count; i++)
    {
        boxes[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (boxes[i].x - info.dW) / info.scale;
        float y0 = (boxes[i].y - info.dH) / info.scale;
        float x1 = (boxes[i].x+boxes[i].w - info.dW) / info.scale;
        float y1 = (boxes[i].y+boxes[i].h - info.dH) / info.scale;

        // clip
        x0 = std::max(std::min(x0, (float)(info.srcW - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(info.srcH - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(info.srcW - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(info.srcH - 1)), 0.f);

        boxes[i].x = x0;
        boxes[i].y = y0;
        boxes[i].w = x1-x0;
        boxes[i].h = y1-y0;
        for(auto & k : boxes[i].keyPoints){
            if(k.p>=impl->kp_thr){
                k.x = (k.x-info.dW)/info.scale;
                k.y = (k.y-info.dH)/info.scale;
            }else{
                k.x = -1;
                k.y = -1;
                k.p = 0;
            }
        }
    }

    return 0;

}

Yolo *Yolo::outputName(const std::string &name) {
    impl->outputNames.push_back(name);
    return this;
}

Yolo *Yolo::inputName(const std::string &name) {
    impl->inputName = name;
    return this;
}

Yolo *Yolo::inputSize(int size) {
    impl->inputSize = size;
    return this;
}

Yolo *Yolo::boxThreshold(float box) {
    impl->box_thr = box;
    return this;
}

Yolo *Yolo::iouThreshold(float iou) {
    impl->iou_thr = iou;
    return this;
}

Yolo *Yolo::outputStride(int stride) {
    impl->outputStrides.push_back(stride);
    return this;
}

Yolo *Yolo::outputAnchors(const ncnn::Mat &anchors) {
    impl->anchors.push_back(anchors);
    return this;
}

Yolo *Yolo::numKeypoint(int num) {
    impl->nkpt = num;
    return this;
}

Yolo* Yolo::ver(int v){
    impl->ver = v;
    return this;
}

Yolo* Yolo::kpThreshold(float kp)
{
    impl->kp_thr = kp;
    return this;
}

Yolo::BBox::BBox():c(0),p(0),x(-1),y(-1),w(0),h(0),keyPoints(){

}

Yolo::Point::Point(float _x, float _y, float _p):x(_x),y(_y),p(_p){

}

Yolo::Point::Point() :x(-1),y(-1),p(0){

}
