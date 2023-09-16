//
// Created by wkt on 2022/10/15.
//

#ifndef NCNNINFER_JLOG_H
#define NCNNINFER_JLOG_H
#include <android/log.h>

#ifndef TAG
#define TAG "JNINativeTag"
#endif

#if defined(DEBUG) && DEBUG && defined(ANDROID)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG,TAG,__VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,TAG,__VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,TAG,__VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,TAG,__VA_ARGS__)
#else
#define LOGD(...)
#define LOGE(...)
#define LOGW(...)
#define LOGI(...)
#endif

#endif //NCNNINFER_JLOG_H
