#include <jni.h>
#include <string>

#include <jni.h>
#include "Yolo.h"
#include <android/asset_manager_jni.h>
#include <ncnn/cpu.h>

#define TAG "YoloInfer"
#define CLASS_NAME ("com/weiketing/yolomobile/YoloInfer")
#include "jlog.h"


static inline std::vector<std::string> split(const std::string& str, const std::string& delimiter)
{
    std::vector<std::string> tokens;
    size_t start = 0, end = 0;
    while ((end = str.find(delimiter, start)) != std::string::npos)
    {
        tokens.push_back(str.substr(start, end - start));
        start = end + delimiter.length();
    }
    tokens.push_back(str.substr(start));
    return tokens;
}

static jfieldID objFieldId = nullptr;

static void JNICALL
YoloInfer_release(JNIEnv *env, jobject thiz) {
    Yolo* yolo = reinterpret_cast<Yolo *>(env->GetLongField(thiz, objFieldId));
    if(yolo == nullptr)return;
    LOGD(__FUNCTION__);
    delete yolo;
    env->SetLongField(thiz,objFieldId, (jlong)0);
}

static void JNICALL
YoloInfer_init(JNIEnv *env, jobject thiz)
{
    LOGD(__FUNCTION__);
    if(objFieldId == nullptr)
        objFieldId = env->GetFieldID(env->GetObjectClass(thiz),"_obj", "J");
    YoloInfer_release(env,thiz);
    Yolo *yolo = new Yolo;
    env->SetLongField(thiz, objFieldId, reinterpret_cast<jlong>(yolo));
}

static void JNICALL
YoloInfer_loadModel(JNIEnv *env, jobject thiz, jobject asset,
                    jstring param, jstring bin){
    LOGD(__FUNCTION__);
    Yolo* yolo = reinterpret_cast<Yolo *>(env->GetLongField(thiz, objFieldId));
    jboolean isCopy = JNI_FALSE;
    const char *p = env->GetStringUTFChars(param,&isCopy);
    const char *b = env->GetStringUTFChars(bin,&isCopy);
    LOGD("loadModel, p: %s, b: %s",p,b);
    AAssetManager *mgr = AAssetManager_fromJava(env, asset);
    int r = yolo->load(mgr, p, b);
    LOGD("loadModel, end with r = %d",r);
    env->ReleaseStringUTFChars(param,p);
    env->ReleaseStringUTFChars(bin,b);
}

static void JNICALL
YoloInfer_loadModel_fullPath(JNIEnv *env, jobject thiz, jstring param_path, jstring bin_path)
{
    LOGD(__FUNCTION__);

    Yolo* yolo = reinterpret_cast<Yolo *>(env->GetLongField(thiz, objFieldId));
    jboolean isCopy = JNI_FALSE;
    const char *p = env->GetStringUTFChars(param_path,&isCopy);
    const char *b = env->GetStringUTFChars(bin_path,&isCopy);
    LOGD("loadModel, p: %s, b: %s",p,b);
    int r = yolo->load(p, b);
    LOGD("loadModel, end with r = %d",r);
    env->ReleaseStringUTFChars(param_path,p);
    env->ReleaseStringUTFChars(bin_path,b);
}

static jfloatArray JNICALL
YoloInfer_forward(JNIEnv *env, jobject thiz, jobject img) {
    if(objFieldId == nullptr)return nullptr;
    Yolo* yolo = reinterpret_cast<Yolo *>(env->GetLongField(thiz, objFieldId));
    ncnn::Mat bm = ncnn::Mat::from_android_bitmap(env,img,ncnn::Mat::PIXEL_RGBA2RGB);
    std::vector<Yolo::BBox>boxes;
    yolo->forward(bm,boxes);
    int n = boxes.size();
    int s = 0;
    std::vector<float> numbers;
    jfloatArray array = nullptr;
    /// LOGD("forward, boxes: %d",n);
    if(n>0){
        for(auto &b:boxes){
            int nk = b.keyPoints.size();
            if(s == 0){
                s = 6;
                s += nk*3;
            }
            numbers.push_back(b.x);
            numbers.push_back(b.y);
            numbers.push_back(b.w);
            numbers.push_back(b.h);
            numbers.push_back(b.c);
            numbers.push_back(b.p);
            //LOGD("forward,x: %.1f, y: %.1f, w: %.1f, h: %.1f, c: %d, p: %.3f",b.x,b.y,b.w,b.h,b.c,b.p);
            for(auto& k:b.keyPoints){
                numbers.push_back(k.x);
                numbers.push_back(k.y);
                numbers.push_back(k.p);
            }
        }
    }
    array = env->NewFloatArray(2+numbers.size());
    float buf[]={static_cast<float>(n),static_cast<float>(s)};
    env->SetFloatArrayRegion(array,0,2,buf);
    env->SetFloatArrayRegion(array,2,numbers.size(),numbers.data());
    return array;
}

static void JNICALL
YoloInfer_update(JNIEnv *env, jobject thiz, jstring key,
                 jstring value) {
    jboolean isCopy = JNI_FALSE;
    const char *k = env->GetStringUTFChars(key,&isCopy);
    const char *v = env->GetStringUTFChars(value,&isCopy);
    if(objFieldId == nullptr){
        //LOGW("You must call init first; k=%s,v=%s",k,v);
        return;
    }else{
        Yolo* yolo = reinterpret_cast<Yolo *>(env->GetLongField(thiz, objFieldId));
        //LOGE("k: %s, v: %s, objFieldId: %p, yolo: %p,",k,v,objFieldId,yolo);
        if(strcmp("input_size",k) == 0){
            yolo->inputSize(atoi(v));
        } else if(strcmp("input_name",k) == 0){
            yolo->inputName(v);
        }else if(strcmp("box_thr",k) == 0){
            yolo->boxThreshold(atof(v));
        }else if(strcmp("iou_thr",k) == 0){
            yolo->iouThreshold(atof(v));
        }else if(strcmp("output_name",k) == 0){
            yolo->outputName(v);
        }else if(strcmp("output_stride",k) == 0){
            yolo->outputStride(atoi(v));
        }else if(strcmp("output_anchors",k) == 0){
            const std::vector<std::string> &as = split(v, ",");
            ncnn::Mat anchors(as.size());
            for(int i=0;i<as.size();i++)anchors[i]=atoi(as[i].data());
            yolo->outputAnchors(anchors);
        }else if(strcmp("nkpt",k) == 0){
            yolo->numKeypoint(atoi(v));
        }else if(strcmp(k,"ver") == 0) {
            yolo->ver(atoi(v));
        }else if(strcmp(k,"kp_thr") == 0) {
            yolo->kpThreshold(atof(v));
        }else{
            LOGE("Unknown k=%s, v=%s",k,v);
        }
    }
    env->ReleaseStringUTFChars(key,k);
    env->ReleaseStringUTFChars(value,v);
}

static void JNICALL
YoloInfer_bind_cpu(JNIEnv *env, jobject thiz, jint flag) {
    ncnn::set_cpu_powersave(flag);
}

static JNINativeMethod methods[] = {
        {"init", "()V", (void*)YoloInfer_init},
        {"load_model", "(Landroid/content/res/AssetManager;Ljava/lang/String;Ljava/lang/String;)V",(void*)YoloInfer_loadModel},
        {"forward", "(Landroid/graphics/Bitmap;)[F",(void*)YoloInfer_forward},
        {"update", "(Ljava/lang/String;Ljava/lang/String;)V",(void*)YoloInfer_update},
        {"release", "()V",(void*)YoloInfer_release},
        {"load_model_fullpath", "(Ljava/lang/String;Ljava/lang/String;)V",(void*)YoloInfer_loadModel_fullPath},
        {"bind_cpu", "(I)V",(void*)YoloInfer_bind_cpu},
};

JNIEXPORT jint JNICALL
JNI_OnLoad(JavaVM* vm, void* reserved) {
    JNIEnv* env = nullptr;
    if (vm->GetEnv((void**) &env, JNI_VERSION_1_6) != JNI_OK) {
        return -1;
    }
    jclass clazz = env->FindClass(CLASS_NAME);
    if (clazz == nullptr) {
        return -1;
    }

    if (env->RegisterNatives(clazz, methods, sizeof(methods) / sizeof(methods[0])) < 0) {
        return -1;
    }
    return JNI_VERSION_1_6;
}
