#include "Longimila-jni.hpp"
#include "LonginusDetector.hpp"
#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif
#include <string>
#include <vector>
#include <iostream>

#include <jni.h>
#include <string.h>
#include <android/log.h>
  
#define TAG    "Longimila-jni"
#define LOGW(...)  __android_log_print(ANDROID_LOG_WARN,TAG,__VA_ARGS__)

static const char *FaceRectClassPath = "com/glasssix/Longimila/FaceRect";
static const char *FaceRectwithFaceInfoClassPath = "com/glasssix/Longimila/FaceRectwithFaceInfo";
static const char *FaceRectwithFaceInfoPairPath = "com/glasssix/Longimila/FaceRectwithFaceInfoPair";
static const char *PointClassPath = "com/glasssix/Longimila/Point";
static const char *MatchRetvalClassPath = "com/glasssix/Longimila/Match_Retval";


JNIEXPORT void JNICALL Java_com_glasssix_Longimila_Longimila_init(JNIEnv *env, jobject thiz, jint device)
{
	glasssix::longinus::LonginusDetector *pDetector = new glasssix::longinus::LonginusDetector(device);
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	env->SetLongField(thiz, fid_mObject, (jlong)pDetector);
	
	env->DeleteLocalRef(clazz);
}

JNIEXPORT void JNICALL Java_com_glasssix_Longimila_Longimila_set(JNIEnv *env, jobject thiz, jint detectionType, jint device)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::longinus::LonginusDetector *pDetector = (glasssix::longinus::LonginusDetector *)p;
	
	switch(detectionType)
	{
		case 0:
			pDetector->set(glasssix::longinus::DetectionType::FRONTALVIEW, device);
			break;
		case 1:
			pDetector->set(glasssix::longinus::DetectionType::FRONTALVIEW_REINFORCE, device);
			break;
		case 2:
			pDetector->set(glasssix::longinus::DetectionType::MULTIVIEW, device);
			break;
		case 3:
			pDetector->set(glasssix::longinus::DetectionType::MULTIVIEW_REINFORCE, device);
			break;
	}
	
	env->DeleteLocalRef(clazz);
}

JNIEXPORT void JNICALL Java_com_glasssix_Longimila_Longimila_finalize(JNIEnv *env, jobject thiz)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::longinus::LonginusDetector *pDetector = (glasssix::longinus::LonginusDetector *)p;
	if(pDetector != nullptr)
	{
		delete pDetector;
		pDetector = nullptr;
		env->SetLongField(thiz, fid_mObject, (jlong)pDetector);
	}
	
	env->DeleteLocalRef(clazz);
}

#ifdef USE_OPENCV
JNIEXPORT jobjectArray JNICALL Java_com_glasssix_Longimila_Longimila_detectbyMat(JNIEnv *env, jobject thiz, jlong grayNativeObj, jint minSize, jfloat scale, jint minNeighbors)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::longinus::LonginusDetector *pDetector = (glasssix::longinus::LonginusDetector *)p;
	
	cv::Mat &gray = *(cv::Mat *)grayNativeObj;
	std::vector<glasssix::longinus::FaceRect> rects = pDetector->detect(gray.data, gray.cols, gray.rows, gray.step[0], minSize, scale, minNeighbors, false, false);
	glasssix::longinus::sort_descend(rects);
	jsize size = rects.size();
	jclass FaceRectClazz = env->FindClass(FaceRectClassPath);
	jobjectArray array = env->NewObjectArray(size, FaceRectClazz, nullptr);
	
	jmethodID mid_FaceRect_constructor = env->GetMethodID(FaceRectClazz, "<init>", "(IIIIID)V");
	
    for (size_t i = 0; i < size; ++i)
	{
		jobject FaceRectObj = env->NewObject(FaceRectClazz, mid_FaceRect_constructor, 
												rects[i].x, rects[i].y, rects[i].width, rects[i].height, 
												rects[i].neighbors, rects[i].confidence);
		
		env->SetObjectArrayElement(array, i, FaceRectObj);
		env->DeleteLocalRef(FaceRectObj);
    }
	
	env->DeleteLocalRef(FaceRectClazz);
	env->DeleteLocalRef(clazz);
	
	return array;
}

JNIEXPORT jobjectArray JNICALL Java_com_glasssix_Longimila_Longimila_detectwithInfobyMat(JNIEnv *env, jobject thiz, jlong grayNativeObj, jint minSize, jfloat scale, jint minNeighbors, jint order)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::longinus::LonginusDetector *pDetector = (glasssix::longinus::LonginusDetector *)p;
	
	cv::Mat &gray = *(cv::Mat *)grayNativeObj;
	std::vector<glasssix::longinus::FaceRectwithFaceInfo> rects = pDetector->detect(gray.data, gray.cols, gray.rows, gray.step[0], minSize, scale, minNeighbors, order, false, false);
	glasssix::longinus::sort_descend(rects);
	jsize size = rects.size();
	jclass FaceRectwithFaceInfoClazz = env->FindClass(FaceRectwithFaceInfoClassPath);
	jobjectArray array = env->NewObjectArray(size, FaceRectwithFaceInfoClazz, nullptr);
	jmethodID mid_FaceRectwithFaceInfo_constructor = env->GetMethodID(FaceRectwithFaceInfoClazz, "<init>", "()V");
	jfieldID fid_pts = env->GetFieldID(FaceRectwithFaceInfoClazz, "pts", "[Lcom/glasssix/Longimila/Point;");
	jfieldID fid_yaw = env->GetFieldID(FaceRectwithFaceInfoClazz, "yaw", "F");
	jfieldID fid_pitch = env->GetFieldID(FaceRectwithFaceInfoClazz, "pitch", "F");
	jfieldID fid_roll = env->GetFieldID(FaceRectwithFaceInfoClazz, "roll", "F");
	
	jclass FaceRectClazz = env->FindClass(FaceRectClassPath);
	
	jfieldID fid_x = env->GetFieldID(FaceRectClazz, "x", "I");
	jfieldID fid_y = env->GetFieldID(FaceRectClazz, "y", "I");
	jfieldID fid_width = env->GetFieldID(FaceRectClazz, "width", "I");
	jfieldID fid_height = env->GetFieldID(FaceRectClazz, "height", "I");
	jfieldID fid_neighbors = env->GetFieldID(FaceRectClazz, "neighbors", "I");
	jfieldID fid_confidence = env->GetFieldID(FaceRectClazz, "confidence", "D");
	
	jclass PointClazz = env->FindClass(PointClassPath);
	jmethodID mid_Point_constructor = env->GetMethodID(PointClazz, "<init>", "()V");
	
	jfieldID fid_Point_x = env->GetFieldID(PointClazz, "x", "I");
	jfieldID fid_Point_y = env->GetFieldID(PointClazz, "y", "I");
	
    for (size_t i = 0; i < size; ++i)
	{
		jobject FaceRectwithFaceInfoObj = env->NewObject(FaceRectwithFaceInfoClazz, mid_FaceRectwithFaceInfo_constructor);

		env->SetIntField(FaceRectwithFaceInfoObj, fid_x, rects[i].x);
		env->SetIntField(FaceRectwithFaceInfoObj, fid_y, rects[i].y);
		env->SetIntField(FaceRectwithFaceInfoObj, fid_width, rects[i].width);
		env->SetIntField(FaceRectwithFaceInfoObj, fid_height, rects[i].height);
		env->SetIntField(FaceRectwithFaceInfoObj, fid_neighbors, rects[i].neighbors);
		env->SetDoubleField(FaceRectwithFaceInfoObj, fid_confidence, rects[i].confidence);
		
		jobjectArray ptsArray = (jobjectArray)env->GetObjectField(FaceRectwithFaceInfoObj, fid_pts);
		jsize ptsArraySize = env->GetArrayLength(ptsArray);
		for (size_t j = 0; j < ptsArraySize; j++)
		{
			jobject PointObj = env->NewObject(PointClazz, mid_Point_constructor);
			env->SetIntField(PointObj, fid_Point_x, rects[i].pts[j].x);
			env->SetIntField(PointObj, fid_Point_y, rects[i].pts[j].y);
			env->SetObjectArrayElement(ptsArray, j, PointObj);
			env->DeleteLocalRef(PointObj);
		}

		env->SetFloatField(FaceRectwithFaceInfoObj, fid_yaw, rects[i].yaw);
		env->SetFloatField(FaceRectwithFaceInfoObj, fid_pitch, rects[i].pitch);
		env->SetFloatField(FaceRectwithFaceInfoObj, fid_roll, rects[i].roll);
		
		env->SetObjectArrayElement(array, i, FaceRectwithFaceInfoObj);

		env->DeleteLocalRef(ptsArray);
		env->DeleteLocalRef(FaceRectwithFaceInfoObj);
    }

	env->DeleteLocalRef(PointClazz);
	env->DeleteLocalRef(FaceRectClazz);
	env->DeleteLocalRef(clazz);
	return array;
}
#endif

JNIEXPORT jobjectArray JNICALL Java_com_glasssix_Longimila_Longimila_detectbyMetaData(JNIEnv *env, jobject thiz, jbyteArray dataArray,
	jint width, jint height, jint minSize, jfloat scale, jint minNeighbors)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::longinus::LonginusDetector *pDetector = (glasssix::longinus::LonginusDetector *)p;

	jsize dataSize = env->GetArrayLength(dataArray);
	std::vector<unsigned char> data_vec(dataSize, 0);
	env->GetByteArrayRegion(dataArray, 0, dataSize, (jbyte *)data_vec.data());
	jsize size;
	std::vector<glasssix::longinus::FaceRect> rects;
	if (data_vec.size() != width * height)
	{
		std::cout << "Data size dismatch! " << "In line " << __LINE__ << " of " << __FILE__ << std::endl;
		size = 0;
	}
	else
	{
		rects = pDetector->detect((unsigned char *)data_vec.data(), width, height, width * sizeof(unsigned char), minSize, scale, minNeighbors, false, false);
		glasssix::longinus::sort_descend(rects);
	}
	size = rects.size();
	jclass FaceRectClazz = env->FindClass(FaceRectClassPath);
	jobjectArray array = env->NewObjectArray(size, FaceRectClazz, nullptr);

	jmethodID mid_FaceRect_constructor = env->GetMethodID(FaceRectClazz, "<init>", "(IIIIID)V");

	for (size_t i = 0; i < size; ++i)
	{
		jobject FaceRectObj = env->NewObject(FaceRectClazz, mid_FaceRect_constructor,
			rects[i].x, rects[i].y, rects[i].width, rects[i].height,
			rects[i].neighbors, rects[i].confidence);

		env->SetObjectArrayElement(array, i, FaceRectObj);
		env->DeleteLocalRef(FaceRectObj);
	}

	env->DeleteLocalRef(FaceRectClazz);
	env->DeleteLocalRef(clazz);

	return array;
}

JNIEXPORT jobjectArray JNICALL Java_com_glasssix_Longimila_Longimila_detectwithInfobyMetaData(JNIEnv *env, jobject thiz, jbyteArray dataArray, 
	jint width, jint height, jint minSize, jfloat scale, jint minNeighbors, jint order)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::longinus::LonginusDetector *pDetector = (glasssix::longinus::LonginusDetector *)p;

	jsize dataSize = env->GetArrayLength(dataArray);
	std::vector<unsigned char> data_vec(dataSize, 0);
	env->GetByteArrayRegion(dataArray, 0, dataSize, (jbyte *)data_vec.data());
	jsize size;
	std::vector<glasssix::longinus::FaceRectwithFaceInfo> rects;
	if (data_vec.size() != width * height)
	{
		std::cout << "Data size dismatch! " << "In line " << __LINE__ << " of " << __FILE__ << std::endl;
		size = 0;
	}
	else
	{
		rects = pDetector->detect((unsigned char *)data_vec.data(), width, height, width * sizeof(unsigned char), minSize, scale, minNeighbors, order, false, false);
		glasssix::longinus::sort_descend(rects);
	}
	size = rects.size();
	jclass FaceRectwithFaceInfoClazz = env->FindClass(FaceRectwithFaceInfoClassPath);
	jobjectArray array = env->NewObjectArray(size, FaceRectwithFaceInfoClazz, nullptr);
	jmethodID mid_FaceRectwithFaceInfo_constructor = env->GetMethodID(FaceRectwithFaceInfoClazz, "<init>", "()V");
	jfieldID fid_pts = env->GetFieldID(FaceRectwithFaceInfoClazz, "pts", "[Lcom/glasssix/Longimila/Point;");
	jfieldID fid_yaw = env->GetFieldID(FaceRectwithFaceInfoClazz, "yaw", "F");
	jfieldID fid_pitch = env->GetFieldID(FaceRectwithFaceInfoClazz, "pitch", "F");
	jfieldID fid_roll = env->GetFieldID(FaceRectwithFaceInfoClazz, "roll", "F");

	jclass FaceRectClazz = env->FindClass(FaceRectClassPath);

	jfieldID fid_x = env->GetFieldID(FaceRectClazz, "x", "I");
	jfieldID fid_y = env->GetFieldID(FaceRectClazz, "y", "I");
	jfieldID fid_width = env->GetFieldID(FaceRectClazz, "width", "I");
	jfieldID fid_height = env->GetFieldID(FaceRectClazz, "height", "I");
	jfieldID fid_neighbors = env->GetFieldID(FaceRectClazz, "neighbors", "I");
	jfieldID fid_confidence = env->GetFieldID(FaceRectClazz, "confidence", "D");

	jclass PointClazz = env->FindClass(PointClassPath);
	jmethodID mid_Point_constructor = env->GetMethodID(PointClazz, "<init>", "()V");

	jfieldID fid_Point_x = env->GetFieldID(PointClazz, "x", "I");
	jfieldID fid_Point_y = env->GetFieldID(PointClazz, "y", "I");

	for (size_t i = 0; i < size; ++i)
	{
		jobject FaceRectwithFaceInfoObj = env->NewObject(FaceRectwithFaceInfoClazz, mid_FaceRectwithFaceInfo_constructor);

		env->SetIntField(FaceRectwithFaceInfoObj, fid_x, rects[i].x);
		env->SetIntField(FaceRectwithFaceInfoObj, fid_y, rects[i].y);
		env->SetIntField(FaceRectwithFaceInfoObj, fid_width, rects[i].width);
		env->SetIntField(FaceRectwithFaceInfoObj, fid_height, rects[i].height);
		env->SetIntField(FaceRectwithFaceInfoObj, fid_neighbors, rects[i].neighbors);
		env->SetDoubleField(FaceRectwithFaceInfoObj, fid_confidence, rects[i].confidence);

		jobjectArray ptsArray = (jobjectArray)env->GetObjectField(FaceRectwithFaceInfoObj, fid_pts);
		jsize ptsArraySize = env->GetArrayLength(ptsArray);
		for (size_t j = 0; j < ptsArraySize; j++)
		{
			jobject PointObj = env->NewObject(PointClazz, mid_Point_constructor);
			env->SetIntField(PointObj, fid_Point_x, rects[i].pts[j].x);
			env->SetIntField(PointObj, fid_Point_y, rects[i].pts[j].y);
			env->SetObjectArrayElement(ptsArray, j, PointObj);
			env->DeleteLocalRef(PointObj);
		}

		env->SetFloatField(FaceRectwithFaceInfoObj, fid_yaw, rects[i].yaw);
		env->SetFloatField(FaceRectwithFaceInfoObj, fid_pitch, rects[i].pitch);
		env->SetFloatField(FaceRectwithFaceInfoObj, fid_roll, rects[i].roll);

		env->SetObjectArrayElement(array, i, FaceRectwithFaceInfoObj);

		env->DeleteLocalRef(ptsArray);
		env->DeleteLocalRef(FaceRectwithFaceInfoObj);
	}

	env->DeleteLocalRef(PointClazz);
	env->DeleteLocalRef(FaceRectClazz);
	env->DeleteLocalRef(clazz);
	return array;
}

jstring char2Jstring(JNIEnv *env, const char *pat, size_t len)
{
	jclass strClazz = env->FindClass("java/lang/String");
	jmethodID mid_String_constructor = env->GetMethodID(strClazz, "<init>", "([BLjava/lang/String;)V");
	jbyteArray bytes = env->NewByteArray(len);
	env->SetByteArrayRegion(bytes, 0, len, (jbyte *)pat);
	jstring encoding = env->NewStringUTF("utf-8");
	
	jstring jstr = (jstring)env->NewObject(strClazz, mid_String_constructor, bytes, encoding);
	
	env->DeleteLocalRef(encoding);
	env->DeleteLocalRef(bytes);
	env->DeleteLocalRef(strClazz);
	
	return jstr;
}

JNIEXPORT jstring JNICALL Java_com_glasssix_Longimila_Longimila_getVersion(JNIEnv *env, jclass clazz)
{
	std::string version = glasssix::longinus::LonginusDetector::getVersion();
	return char2Jstring(env, version.c_str(), version.length());
}

JNIEXPORT jobjectArray JNICALL Java_com_glasssix_Longimila_Longimila_match(JNIEnv *env, jobject thiz, jobjectArray faceRectArray, jint frame_extract_frequency, jfloat distance_fractor)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::longinus::LonginusDetector *pDetector = (glasssix::longinus::LonginusDetector *)p;
	
	jclass FaceRectClazz = env->FindClass(FaceRectClassPath);
	jfieldID fid_x = env->GetFieldID(FaceRectClazz, "x", "I");
	jfieldID fid_y = env->GetFieldID(FaceRectClazz, "y", "I");
	jfieldID fid_width = env->GetFieldID(FaceRectClazz, "width", "I");
	jfieldID fid_height = env->GetFieldID(FaceRectClazz, "height", "I");
	jfieldID fid_neighbors = env->GetFieldID(FaceRectClazz, "neighbors", "I");
	jfieldID fid_confidence = env->GetFieldID(FaceRectClazz, "confidence", "D");
	
	std::vector<glasssix::longinus::FaceRect> faceRects;
	jsize arrSize = env->GetArrayLength(faceRectArray);
	for(size_t i = 0; i < arrSize; i++)
	{
		jobject FaceRectObj = env->GetObjectArrayElement(faceRectArray, i);
		int x = env->GetIntField(FaceRectObj, fid_x);
		int y = env->GetIntField(FaceRectObj, fid_y);
		int width = env->GetIntField(FaceRectObj, fid_width);
		int height = env->GetIntField(FaceRectObj, fid_height);
		int neighbors = env->GetIntField(FaceRectObj, fid_neighbors);
		double confidence = env->GetDoubleField(FaceRectObj, fid_confidence);
		faceRects.emplace_back(x, y, width, height, neighbors, confidence);
		
		env->DeleteLocalRef(FaceRectObj);
	}
	
	std::vector<glasssix::longinus::Match_Retval> match_vec = pDetector->match(faceRects, frame_extract_frequency, distance_fractor);
	
	jsize vec_size = match_vec.size();
	jclass MatchRetvalClazz = env->FindClass(MatchRetvalClassPath);
	jmethodID mid_MatchRetval_constructor = env->GetMethodID(MatchRetvalClazz, "<init>", "()V");
	jmethodID mid_FaceRect_constructor = env->GetMethodID(FaceRectClazz, "<init>", "()V");
	
	jobjectArray match_array = env->NewObjectArray(vec_size, MatchRetvalClazz, nullptr);
	
	jfieldID fid_rect = env->GetFieldID(MatchRetvalClazz, "rect", "Ljava/lang/Object;");
	jfieldID fid_id = env->GetFieldID(MatchRetvalClazz, "id", "Ljava/lang/String;");
	jfieldID fid_is_new = env->GetFieldID(MatchRetvalClazz, "is_new", "Z");
	
	for(size_t i = 0; i < vec_size; i++)
	{
		jobject FaceRectObj = env->NewObject(FaceRectClazz, mid_FaceRect_constructor);
		jobject MatchRetvalObj = env->NewObject(MatchRetvalClazz, mid_MatchRetval_constructor);
		env->SetIntField(FaceRectObj, fid_x, match_vec[i].rect.x);
		env->SetIntField(FaceRectObj, fid_y, match_vec[i].rect.y);
		env->SetIntField(FaceRectObj, fid_width, match_vec[i].rect.width);
		env->SetIntField(FaceRectObj, fid_height, match_vec[i].rect.height);
		env->SetIntField(FaceRectObj, fid_neighbors, match_vec[i].rect.neighbors);
		env->SetDoubleField(FaceRectObj, fid_confidence, match_vec[i].rect.confidence);
		
		env->SetObjectField(MatchRetvalObj, fid_rect, FaceRectObj);
		jstring jstr_id = char2Jstring(env, match_vec[i].id.c_str(), match_vec[i].id.length());
		env->SetObjectField(MatchRetvalObj, fid_id, jstr_id);
		env->SetBooleanField(MatchRetvalObj, fid_is_new, match_vec[i].is_new);
		
		env->SetObjectArrayElement(match_array, i, MatchRetvalObj);
		
		env->DeleteLocalRef(jstr_id);
		env->DeleteLocalRef(MatchRetvalObj);
		env->DeleteLocalRef(FaceRectObj);
	}
	
	
	env->DeleteLocalRef(MatchRetvalClazz);
	env->DeleteLocalRef(FaceRectClazz);
	env->DeleteLocalRef(clazz);
	
	return match_array;
}

#ifdef USE_OPENCV
JNIEXPORT jbyteArray JNICALL Java_com_glasssix_Longimila_Longimila_alignFacebyMat(JNIEnv *env, jobject thiz, jlong grayNativeObj, jobjectArray bboxArray, jobjectArray landmarkArray)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::longinus::LonginusDetector *pDetector = (glasssix::longinus::LonginusDetector *)p;

	cv::Mat &mat = *(cv::Mat *)grayNativeObj;
	jsize bboxArraySize = env->GetArrayLength(bboxArray);
	jsize landmarkArraySize = env->GetArrayLength(landmarkArray);

	jintArray bbox0 = (jintArray)env->GetObjectArrayElement(bboxArray, 0);
	jintArray landmark0 = (jintArray)env->GetObjectArrayElement(landmarkArray, 0);

	jsize bboxDimension = env->GetArrayLength(bbox0);
	jsize landmarkDimension = env->GetArrayLength(landmark0);

	env->DeleteLocalRef(landmark0);
	env->DeleteLocalRef(bbox0);

	std::vector<std::vector<int> > bbox_vec(bboxArraySize, std::vector<int>(bboxDimension));
	std::vector<std::vector<int> > landmark_vec(landmarkArraySize, std::vector<int>(landmarkDimension));

	for (size_t i = 0; i < bboxArraySize; i++)
	{
		jintArray bbox = (jintArray)env->GetObjectArrayElement(bboxArray, i);
		env->GetIntArrayRegion(bbox, 0, bboxDimension, (jint *)bbox_vec[i].data());
		env->DeleteLocalRef(bbox);
	}

	for (size_t i = 0; i < landmarkArraySize; i++)
	{
		jintArray landmark = (jintArray)env->GetObjectArrayElement(landmarkArray, i);
		env->GetIntArrayRegion(landmark, 0, landmarkDimension, (jint *)landmark_vec[i].data());
		env->DeleteLocalRef(landmark);
	}


	std::vector<unsigned char> aligned_vec = pDetector->alignFace(mat.data, bbox_vec.size(), mat.channels(), mat.rows, mat.cols, bbox_vec, landmark_vec);

	jsize alignedSize = aligned_vec.size();

	jbyteArray aligned_array = env->NewByteArray(alignedSize);
	env->SetByteArrayRegion(aligned_array, 0, alignedSize, (const jbyte *)aligned_vec.data());


	env->DeleteLocalRef(clazz);

	return aligned_array;
}
#endif

JNIEXPORT jbyteArray JNICALL Java_com_glasssix_Longimila_Longimila_alignFacebyMetaData(JNIEnv *env, jobject thiz,
	jbyteArray dataArray, jint width, jint height, jobjectArray bboxArray, jobjectArray landmarkArray)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::longinus::LonginusDetector *pDetector = (glasssix::longinus::LonginusDetector *)p;

	jsize bboxArraySize = env->GetArrayLength(bboxArray);
	jsize landmarkArraySize = env->GetArrayLength(landmarkArray);

	jintArray bbox0 = (jintArray)env->GetObjectArrayElement(bboxArray, 0);
	jintArray landmark0 = (jintArray)env->GetObjectArrayElement(landmarkArray, 0);

	jsize bboxDimension = env->GetArrayLength(bbox0);
	jsize landmarkDimension = env->GetArrayLength(landmark0);

	env->DeleteLocalRef(landmark0);
	env->DeleteLocalRef(bbox0);

	std::vector<std::vector<int> > bbox_vec(bboxArraySize, std::vector<int>(bboxDimension));
	std::vector<std::vector<int> > landmark_vec(landmarkArraySize, std::vector<int>(landmarkDimension));

	for (size_t i = 0; i < bboxArraySize; i++)
	{
		jintArray bbox = (jintArray)env->GetObjectArrayElement(bboxArray, i);
		env->GetIntArrayRegion(bbox, 0, bboxDimension, (jint *)bbox_vec[i].data());
		env->DeleteLocalRef(bbox);
	}

	for (size_t i = 0; i < landmarkArraySize; i++)
	{
		jintArray landmark = (jintArray)env->GetObjectArrayElement(landmarkArray, i);
		env->GetIntArrayRegion(landmark, 0, landmarkDimension, (jint *)landmark_vec[i].data());
		env->DeleteLocalRef(landmark);
	}

	jsize dataSize = env->GetArrayLength(dataArray);
	std::vector<unsigned char> data_vec(dataSize, 0);
	env->GetByteArrayRegion(dataArray, 0, dataSize, (jbyte *)data_vec.data());
	jsize alignedSize;
	std::vector<unsigned char> aligned_vec;
	if (data_vec.size() != width * height)
	{
		std::cout << "Data size dismatch! " << "In line " << __LINE__ << " of " << __FILE__ << std::endl;
		alignedSize = 0;
	}
	else
	{
		aligned_vec = pDetector->alignFace((unsigned char *)data_vec.data(), bbox_vec.size(), 1, height, width, bbox_vec, landmark_vec);
		alignedSize = aligned_vec.size();
	}
	jbyteArray aligned_array = env->NewByteArray(alignedSize);
	env->SetByteArrayRegion(aligned_array, 0, alignedSize, (const jbyte *)aligned_vec.data());


	env->DeleteLocalRef(clazz);

	return aligned_array;
}

#ifndef TRIAL
#ifdef USE_OPENCV
JNIEXPORT jobjectArray JNICALL Java_com_glasssix_Longimila_Longimila_detectExbyMat(JNIEnv *env, jobject thiz, jlong matNativeObj, jint minSize, jfloatArray threshold, jfloat factor, jint stage, jint order)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::longinus::LonginusDetector *pDetector = (glasssix::longinus::LonginusDetector *)p;
	
	jsize thresholdSize = env->GetArrayLength(threshold);
	std::vector<float> threshold_vec(thresholdSize);
	env->GetFloatArrayRegion(threshold, 0, thresholdSize, (jfloat *)threshold_vec.data());
	
	cv::Mat &mat = *(cv::Mat *)matNativeObj;
	std::vector<glasssix::longinus::FaceRectwithFaceInfo> rects = pDetector->detectEx(mat.data, mat.channels(), mat.rows, mat.cols, minSize, threshold_vec.data(), 1.0f / factor, stage, order);
	glasssix::longinus::sort_descend(rects);
	jsize size = rects.size();
	jclass FaceRectwithFaceInfoClazz = env->FindClass(FaceRectwithFaceInfoClassPath);
	jobjectArray array = env->NewObjectArray(size, FaceRectwithFaceInfoClazz, nullptr);
	jmethodID mid_FaceRectwithFaceInfo_constructor = env->GetMethodID(FaceRectwithFaceInfoClazz, "<init>", "()V");
	jfieldID fid_pts = env->GetFieldID(FaceRectwithFaceInfoClazz, "pts", "[Lcom/glasssix/Longimila/Point;");
	jfieldID fid_yaw = env->GetFieldID(FaceRectwithFaceInfoClazz, "yaw", "F");
	jfieldID fid_pitch = env->GetFieldID(FaceRectwithFaceInfoClazz, "pitch", "F");
	jfieldID fid_roll = env->GetFieldID(FaceRectwithFaceInfoClazz, "roll", "F");
	
	jclass FaceRectClazz = env->FindClass(FaceRectClassPath);
	
	jfieldID fid_x = env->GetFieldID(FaceRectClazz, "x", "I");
	jfieldID fid_y = env->GetFieldID(FaceRectClazz, "y", "I");
	jfieldID fid_width = env->GetFieldID(FaceRectClazz, "width", "I");
	jfieldID fid_height = env->GetFieldID(FaceRectClazz, "height", "I");
	jfieldID fid_neighbors = env->GetFieldID(FaceRectClazz, "neighbors", "I");
	jfieldID fid_confidence = env->GetFieldID(FaceRectClazz, "confidence", "D");
	
	jclass PointClazz = env->FindClass(PointClassPath);
	jmethodID mid_Point_constructor = env->GetMethodID(PointClazz, "<init>", "()V");
	
	jfieldID fid_Point_x = env->GetFieldID(PointClazz, "x", "I");
	jfieldID fid_Point_y = env->GetFieldID(PointClazz, "y", "I");
	
    for (size_t i = 0; i < size; ++i)
	{
		jobject FaceRectwithFaceInfoObj = env->NewObject(FaceRectwithFaceInfoClazz, mid_FaceRectwithFaceInfo_constructor);

		env->SetIntField(FaceRectwithFaceInfoObj, fid_x, rects[i].x);
		env->SetIntField(FaceRectwithFaceInfoObj, fid_y, rects[i].y);
		env->SetIntField(FaceRectwithFaceInfoObj, fid_width, rects[i].width);
		env->SetIntField(FaceRectwithFaceInfoObj, fid_height, rects[i].height);
		env->SetIntField(FaceRectwithFaceInfoObj, fid_neighbors, rects[i].neighbors);
		env->SetDoubleField(FaceRectwithFaceInfoObj, fid_confidence, rects[i].confidence);
		
		jobjectArray ptsArray = (jobjectArray)env->GetObjectField(FaceRectwithFaceInfoObj, fid_pts);
		jsize ptsArraySize = env->GetArrayLength(ptsArray);
		for (size_t j = 0; j < ptsArraySize; j++)
		{
			jobject PointObj = env->NewObject(PointClazz, mid_Point_constructor);
			env->SetIntField(PointObj, fid_Point_x, rects[i].pts[j].x);
			env->SetIntField(PointObj, fid_Point_y, rects[i].pts[j].y);
			env->SetObjectArrayElement(ptsArray, j, PointObj);
			env->DeleteLocalRef(PointObj);
		}

		env->SetFloatField(FaceRectwithFaceInfoObj, fid_yaw, rects[i].yaw);
		env->SetFloatField(FaceRectwithFaceInfoObj, fid_pitch, rects[i].pitch);
		env->SetFloatField(FaceRectwithFaceInfoObj, fid_roll, rects[i].roll);
		
		env->SetObjectArrayElement(array, i, FaceRectwithFaceInfoObj);

		env->DeleteLocalRef(ptsArray);
		env->DeleteLocalRef(FaceRectwithFaceInfoObj);
    }

	env->DeleteLocalRef(PointClazz);
	env->DeleteLocalRef(FaceRectClazz);
	env->DeleteLocalRef(clazz);
	return array;
}

JNIEXPORT jobjectArray JNICALL Java_com_glasssix_Longimila_Longimila_detectExMobilebyMat(JNIEnv *env, jobject thiz, jlong matNativeObj, jint minSize, jfloatArray threshold, jfloat factor, jint stage, jint order)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::longinus::LonginusDetector *pDetector = (glasssix::longinus::LonginusDetector *)p;

	jsize thresholdSize = env->GetArrayLength(threshold);
	std::vector<float> threshold_vec(thresholdSize);
	env->GetFloatArrayRegion(threshold, 0, thresholdSize, (jfloat *)threshold_vec.data());

	cv::Mat &mat = *(cv::Mat *)matNativeObj;
	std::vector<glasssix::longinus::FaceRectwithFaceInfo> rects = pDetector->detectEx_mobile(mat.data, mat.channels(), mat.rows, mat.cols, minSize, threshold_vec.data(), 1.0f / factor, stage, order);
	glasssix::longinus::sort_descend(rects);
	jsize size = rects.size();
	jclass FaceRectwithFaceInfoClazz = env->FindClass(FaceRectwithFaceInfoClassPath);
	jobjectArray array = env->NewObjectArray(size, FaceRectwithFaceInfoClazz, nullptr);
	jmethodID mid_FaceRectwithFaceInfo_constructor = env->GetMethodID(FaceRectwithFaceInfoClazz, "<init>", "()V");
	jfieldID fid_pts = env->GetFieldID(FaceRectwithFaceInfoClazz, "pts", "[Lcom/glasssix/Longimila/Point;");
	jfieldID fid_yaw = env->GetFieldID(FaceRectwithFaceInfoClazz, "yaw", "F");
	jfieldID fid_pitch = env->GetFieldID(FaceRectwithFaceInfoClazz, "pitch", "F");
	jfieldID fid_roll = env->GetFieldID(FaceRectwithFaceInfoClazz, "roll", "F");

	jclass FaceRectClazz = env->FindClass(FaceRectClassPath);

	jfieldID fid_x = env->GetFieldID(FaceRectClazz, "x", "I");
	jfieldID fid_y = env->GetFieldID(FaceRectClazz, "y", "I");
	jfieldID fid_width = env->GetFieldID(FaceRectClazz, "width", "I");
	jfieldID fid_height = env->GetFieldID(FaceRectClazz, "height", "I");
	jfieldID fid_neighbors = env->GetFieldID(FaceRectClazz, "neighbors", "I");
	jfieldID fid_confidence = env->GetFieldID(FaceRectClazz, "confidence", "D");

	jclass PointClazz = env->FindClass(PointClassPath);
	jmethodID mid_Point_constructor = env->GetMethodID(PointClazz, "<init>", "()V");

	jfieldID fid_Point_x = env->GetFieldID(PointClazz, "x", "I");
	jfieldID fid_Point_y = env->GetFieldID(PointClazz, "y", "I");

	for (size_t i = 0; i < size; ++i)
	{
		jobject FaceRectwithFaceInfoObj = env->NewObject(FaceRectwithFaceInfoClazz, mid_FaceRectwithFaceInfo_constructor);

		env->SetIntField(FaceRectwithFaceInfoObj, fid_x, rects[i].x);
		env->SetIntField(FaceRectwithFaceInfoObj, fid_y, rects[i].y);
		env->SetIntField(FaceRectwithFaceInfoObj, fid_width, rects[i].width);
		env->SetIntField(FaceRectwithFaceInfoObj, fid_height, rects[i].height);
		env->SetIntField(FaceRectwithFaceInfoObj, fid_neighbors, rects[i].neighbors);
		env->SetDoubleField(FaceRectwithFaceInfoObj, fid_confidence, rects[i].confidence);

		jobjectArray ptsArray = (jobjectArray)env->GetObjectField(FaceRectwithFaceInfoObj, fid_pts);
		jsize ptsArraySize = env->GetArrayLength(ptsArray);
		for (size_t j = 0; j < ptsArraySize; j++)
		{
			jobject PointObj = env->NewObject(PointClazz, mid_Point_constructor);
			env->SetIntField(PointObj, fid_Point_x, rects[i].pts[j].x);
			env->SetIntField(PointObj, fid_Point_y, rects[i].pts[j].y);
			env->SetObjectArrayElement(ptsArray, j, PointObj);
			env->DeleteLocalRef(PointObj);
		}

		env->SetFloatField(FaceRectwithFaceInfoObj, fid_yaw, rects[i].yaw);
		env->SetFloatField(FaceRectwithFaceInfoObj, fid_pitch, rects[i].pitch);
		env->SetFloatField(FaceRectwithFaceInfoObj, fid_roll, rects[i].roll);

		env->SetObjectArrayElement(array, i, FaceRectwithFaceInfoObj);

		env->DeleteLocalRef(ptsArray);
		env->DeleteLocalRef(FaceRectwithFaceInfoObj);
	}

	env->DeleteLocalRef(PointClazz);
	env->DeleteLocalRef(FaceRectClazz);
	env->DeleteLocalRef(clazz);
	return array;
}

JNIEXPORT jbyteArray Java_com_glasssix_Longimila_Longimila_alignSingleFacebyMat(JNIEnv *env, jobject thiz, jlong grayNativeObj)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::longinus::LonginusDetector *pDetector = (glasssix::longinus::LonginusDetector *)p;

	cv::Mat &gray = *(cv::Mat *)grayNativeObj;

	std::vector<unsigned char> aligned_vec = pDetector->alignFace(gray.data, 1, gray.channels(), gray.rows, gray.cols);

	jsize alignedSize = aligned_vec.size();

	jbyteArray aligned_array = env->NewByteArray(alignedSize);
	env->SetByteArrayRegion(aligned_array, 0, alignedSize, (const jbyte *)aligned_vec.data());


	env->DeleteLocalRef(clazz);

	return aligned_array;
}

JNIEXPORT jobjectArray JNICALL Java_com_glasssix_Longimila_Longimila_detectExMobileNIRbyMat(JNIEnv *env, jobject thiz, jlong matNativeObj, jint minSize, jfloatArray threshold, jfloat factor, jint stage, jint order)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::longinus::LonginusDetector *pDetector = (glasssix::longinus::LonginusDetector *)p;

	jsize thresholdSize = env->GetArrayLength(threshold);
	std::vector<float> threshold_vec(thresholdSize);
	env->GetFloatArrayRegion(threshold, 0, thresholdSize, (jfloat *)threshold_vec.data());

	cv::Mat &mat = *(cv::Mat *)matNativeObj;
	std::vector<glasssix::longinus::FaceRectwithFaceInfo> rects = pDetector->detectEx_mobile_nir(mat.data, mat.channels(), mat.rows, mat.cols, minSize, threshold_vec.data(), 1.0f / factor, stage, order);
	glasssix::longinus::sort_descend(rects);
	jsize size = rects.size();
	jclass FaceRectwithFaceInfoClazz = env->FindClass(FaceRectwithFaceInfoClassPath);
	jobjectArray array = env->NewObjectArray(size, FaceRectwithFaceInfoClazz, nullptr);
	jmethodID mid_FaceRectwithFaceInfo_constructor = env->GetMethodID(FaceRectwithFaceInfoClazz, "<init>", "()V");
	jfieldID fid_pts = env->GetFieldID(FaceRectwithFaceInfoClazz, "pts", "[Lcom/glasssix/Longimila/Point;");
	jfieldID fid_yaw = env->GetFieldID(FaceRectwithFaceInfoClazz, "yaw", "F");
	jfieldID fid_pitch = env->GetFieldID(FaceRectwithFaceInfoClazz, "pitch", "F");
	jfieldID fid_roll = env->GetFieldID(FaceRectwithFaceInfoClazz, "roll", "F");

	jclass FaceRectClazz = env->FindClass(FaceRectClassPath);

	jfieldID fid_x = env->GetFieldID(FaceRectClazz, "x", "I");
	jfieldID fid_y = env->GetFieldID(FaceRectClazz, "y", "I");
	jfieldID fid_width = env->GetFieldID(FaceRectClazz, "width", "I");
	jfieldID fid_height = env->GetFieldID(FaceRectClazz, "height", "I");
	jfieldID fid_neighbors = env->GetFieldID(FaceRectClazz, "neighbors", "I");
	jfieldID fid_confidence = env->GetFieldID(FaceRectClazz, "confidence", "D");

	jclass PointClazz = env->FindClass(PointClassPath);
	jmethodID mid_Point_constructor = env->GetMethodID(PointClazz, "<init>", "()V");

	jfieldID fid_Point_x = env->GetFieldID(PointClazz, "x", "I");
	jfieldID fid_Point_y = env->GetFieldID(PointClazz, "y", "I");

	for (size_t i = 0; i < size; ++i)
	{
		jobject FaceRectwithFaceInfoObj = env->NewObject(FaceRectwithFaceInfoClazz, mid_FaceRectwithFaceInfo_constructor);

		env->SetIntField(FaceRectwithFaceInfoObj, fid_x, rects[i].x);
		env->SetIntField(FaceRectwithFaceInfoObj, fid_y, rects[i].y);
		env->SetIntField(FaceRectwithFaceInfoObj, fid_width, rects[i].width);
		env->SetIntField(FaceRectwithFaceInfoObj, fid_height, rects[i].height);
		env->SetIntField(FaceRectwithFaceInfoObj, fid_neighbors, rects[i].neighbors);
		env->SetDoubleField(FaceRectwithFaceInfoObj, fid_confidence, rects[i].confidence);

		jobjectArray ptsArray = (jobjectArray)env->GetObjectField(FaceRectwithFaceInfoObj, fid_pts);
		jsize ptsArraySize = env->GetArrayLength(ptsArray);
		for (size_t j = 0; j < ptsArraySize; j++)
		{
			jobject PointObj = env->NewObject(PointClazz, mid_Point_constructor);
			env->SetIntField(PointObj, fid_Point_x, rects[i].pts[j].x);
			env->SetIntField(PointObj, fid_Point_y, rects[i].pts[j].y);
			env->SetObjectArrayElement(ptsArray, j, PointObj);
			env->DeleteLocalRef(PointObj);
		}

		env->SetFloatField(FaceRectwithFaceInfoObj, fid_yaw, rects[i].yaw);
		env->SetFloatField(FaceRectwithFaceInfoObj, fid_pitch, rects[i].pitch);
		env->SetFloatField(FaceRectwithFaceInfoObj, fid_roll, rects[i].roll);

		env->SetObjectArrayElement(array, i, FaceRectwithFaceInfoObj);

		env->DeleteLocalRef(ptsArray);
		env->DeleteLocalRef(FaceRectwithFaceInfoObj);
	}

	env->DeleteLocalRef(PointClazz);
	env->DeleteLocalRef(FaceRectClazz);
	env->DeleteLocalRef(clazz);
	return array;
}

JNIEXPORT jobject JNICALL Java_com_glasssix_Longimila_Longimila_detectExMobilePairbyMat(JNIEnv *env, jobject thiz, jlong vsl_matNativeObj, jint vsl_minSize, jfloatArray vsl_threshold, jfloat vsl_factor, jint vsl_stage, jint vsl_order, jlong nir_matNativeObj, jint nir_minSize, jfloatArray nir_threshold, jfloat nir_factor, jint nir_stage, jint nir_order)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::longinus::LonginusDetector *pDetector = (glasssix::longinus::LonginusDetector *)p;

	jsize vsl_thresholdSize = env->GetArrayLength(vsl_threshold);
	std::vector<float> vsl_threshold_vec(vsl_thresholdSize);
	env->GetFloatArrayRegion(vsl_threshold, 0, vsl_thresholdSize, (jfloat *)vsl_threshold_vec.data());
	
	jsize nir_thresholdSize = env->GetArrayLength(nir_threshold);
	std::vector<float> nir_threshold_vec(nir_thresholdSize);
	env->GetFloatArrayRegion(nir_threshold, 0, nir_thresholdSize, (jfloat *)nir_threshold_vec.data());

	cv::Mat &vsl_mat = *(cv::Mat *)vsl_matNativeObj;
	cv::Mat &nir_mat = *(cv::Mat *)nir_matNativeObj;
	std::vector<std::vector<glasssix::longinus::FaceRectwithFaceInfo> > pair_rects = pDetector->detectEx_mobile_pair(vsl_mat.data, vsl_mat.channels(), vsl_mat.rows, vsl_mat.cols, vsl_minSize, vsl_threshold_vec.data(), 1.0f / vsl_factor, vsl_stage, vsl_order, nir_mat.data, nir_mat.channels(), nir_mat.rows, nir_mat.cols, nir_minSizeSize, nir_threshold_vec.data(), 1.0f / nir_factor, nir_stage, nir_order);
	glasssix::longinus::sort_descend(pair_rects[0]);
	glasssix::longinus::sort_descend(pair_rects[1]);
	
	jclass FaceRectwithFaceInfoPairClazz = env->FindClass(FaceRectwithFaceInfoPairPath);
	jmethodID mid_FaceRectwithFaceInfoPair_constructor = env->GetMethodID(FaceRectwithFaceInfoPairClazz, "<init>", "()V");
	jfieldID fid_vsl = env->GetFieldID(FaceRectwithFaceInfoPairClazz, "vsl", "[Lcom/glasssix/Longimila/FaceRectwithFaceInfo;");
	jfieldID fid_nir = env->GetFieldID(FaceRectwithFaceInfoPairClazz, "nir", "[Lcom/glasssix/Longimila/FaceRectwithFaceInfo;");
	jobject pairObj = env->NewObject(FaceRectwithFaceInfoPairClazz, mid_FaceRectwithFaceInfoPair_constructor);
	
	jclass FaceRectwithFaceInfoClazz = env->FindClass(FaceRectwithFaceInfoClassPath);
	jmethodID mid_FaceRectwithFaceInfo_constructor = env->GetMethodID(FaceRectwithFaceInfoClazz, "<init>", "()V");
	
	jsize vsl_size = pair_rects[0].size();
	jsize nir_size = pair_rects[1].size();
	
	jobjectArray vsl_array = env->NewObjectArray(vsl_size, FaceRectwithFaceInfoClazz, nullptr);
	jobjectArray nir_array = env->NewObjectArray(nir_size, FaceRectwithFaceInfoClazz, nullptr);
	
	jfieldID fid_pts = env->GetFieldID(FaceRectwithFaceInfoClazz, "pts", "[Lcom/glasssix/Longimila/Point;");
	jfieldID fid_yaw = env->GetFieldID(FaceRectwithFaceInfoClazz, "yaw", "F");
	jfieldID fid_pitch = env->GetFieldID(FaceRectwithFaceInfoClazz, "pitch", "F");
	jfieldID fid_roll = env->GetFieldID(FaceRectwithFaceInfoClazz, "roll", "F");

	jclass FaceRectClazz = env->FindClass(FaceRectClassPath);

	jfieldID fid_x = env->GetFieldID(FaceRectClazz, "x", "I");
	jfieldID fid_y = env->GetFieldID(FaceRectClazz, "y", "I");
	jfieldID fid_width = env->GetFieldID(FaceRectClazz, "width", "I");
	jfieldID fid_height = env->GetFieldID(FaceRectClazz, "height", "I");
	jfieldID fid_neighbors = env->GetFieldID(FaceRectClazz, "neighbors", "I");
	jfieldID fid_confidence = env->GetFieldID(FaceRectClazz, "confidence", "D");

	jclass PointClazz = env->FindClass(PointClassPath);
	jmethodID mid_Point_constructor = env->GetMethodID(PointClazz, "<init>", "()V");

	jfieldID fid_Point_x = env->GetFieldID(PointClazz, "x", "I");
	jfieldID fid_Point_y = env->GetFieldID(PointClazz, "y", "I");

	for (size_t i = 0; i < vsl_size; ++i)
	{
		jobject FaceRectwithFaceInfoObj = env->NewObject(FaceRectwithFaceInfoClazz, mid_FaceRectwithFaceInfo_constructor);

		env->SetIntField(FaceRectwithFaceInfoObj, fid_x, pair_rects[0][i].x);
		env->SetIntField(FaceRectwithFaceInfoObj, fid_y, pair_rects[0][i].y);
		env->SetIntField(FaceRectwithFaceInfoObj, fid_width, pair_rects[0][i].width);
		env->SetIntField(FaceRectwithFaceInfoObj, fid_height, pair_rects[0][i].height);
		env->SetIntField(FaceRectwithFaceInfoObj, fid_neighbors, pair_rects[0][i].neighbors);
		env->SetDoubleField(FaceRectwithFaceInfoObj, fid_confidence, pair_rects[0][i].confidence);

		jobjectArray ptsArray = (jobjectArray)env->GetObjectField(FaceRectwithFaceInfoObj, fid_pts);
		jsize ptsArraySize = env->GetArrayLength(ptsArray);
		for (size_t j = 0; j < ptsArraySize; j++)
		{
			jobject PointObj = env->NewObject(PointClazz, mid_Point_constructor);
			env->SetIntField(PointObj, fid_Point_x, pair_rects[0][i].pts[j].x);
			env->SetIntField(PointObj, fid_Point_y, pair_rects[0][i].pts[j].y);
			env->SetObjectArrayElement(ptsArray, j, PointObj);
			env->DeleteLocalRef(PointObj);
		}

		env->SetFloatField(FaceRectwithFaceInfoObj, fid_yaw, rects[i].yaw);
		env->SetFloatField(FaceRectwithFaceInfoObj, fid_pitch, rects[i].pitch);
		env->SetFloatField(FaceRectwithFaceInfoObj, fid_roll, rects[i].roll);

		env->SetObjectArrayElement(vsl_array, i, FaceRectwithFaceInfoObj);

		env->DeleteLocalRef(ptsArray);
		env->DeleteLocalRef(FaceRectwithFaceInfoObj);
	}
	
	for (size_t i = 0; i < nir_size; ++i)
	{
		jobject FaceRectwithFaceInfoObj = env->NewObject(FaceRectwithFaceInfoClazz, mid_FaceRectwithFaceInfo_constructor);

		env->SetIntField(FaceRectwithFaceInfoObj, fid_x, pair_rects[1][i].x);
		env->SetIntField(FaceRectwithFaceInfoObj, fid_y, pair_rects[1][i].y);
		env->SetIntField(FaceRectwithFaceInfoObj, fid_width, pair_rects[1][i].width);
		env->SetIntField(FaceRectwithFaceInfoObj, fid_height, pair_rects[1][i].height);
		env->SetIntField(FaceRectwithFaceInfoObj, fid_neighbors, pair_rects[1][i].neighbors);
		env->SetDoubleField(FaceRectwithFaceInfoObj, fid_confidence, pair_rects[1][i].confidence);

		jobjectArray ptsArray = (jobjectArray)env->GetObjectField(FaceRectwithFaceInfoObj, fid_pts);
		jsize ptsArraySize = env->GetArrayLength(ptsArray);
		for (size_t j = 0; j < ptsArraySize; j++)
		{
			jobject PointObj = env->NewObject(PointClazz, mid_Point_constructor);
			env->SetIntField(PointObj, fid_Point_x, pair_rects[1][i].pts[j].x);
			env->SetIntField(PointObj, fid_Point_y, pair_rects[1][i].pts[j].y);
			env->SetObjectArrayElement(ptsArray, j, PointObj);
			env->DeleteLocalRef(PointObj);
		}

		env->SetFloatField(FaceRectwithFaceInfoObj, fid_yaw, rects[i].yaw);
		env->SetFloatField(FaceRectwithFaceInfoObj, fid_pitch, rects[i].pitch);
		env->SetFloatField(FaceRectwithFaceInfoObj, fid_roll, rects[i].roll);

		env->SetObjectArrayElement(nir_array, i, FaceRectwithFaceInfoObj);

		env->DeleteLocalRef(ptsArray);
		env->DeleteLocalRef(FaceRectwithFaceInfoObj);
	}
	
	env->setObjectField(pairObj, fid_vsl, vsl_array);
	env->setObjectField(pairObj, fid_nir, nir_array);

	env->DeleteLocalRef(PointClazz);
	env->DeleteLocalRef(FaceRectClazz);
	env->DeleteLocalRef(FaceRectwithFaceInfoClazz);
	env->DeleteLocalRef(nir_array);
	env->DeleteLocalRef(vsl_array);
	env->DeleteLocalRef(FaceRectwithFaceInfoPairClazz);
	env->DeleteLocalRef(clazz);
	return pairObj;
}

JNIEXPORT jboolean JNICALL blurJudgeVSLbyMat(JNIEnv *env, jobject thiz, jlong vsl_color_image_mat_NativeObj, jobjectArray bbox, jobjectArray landmarks, jfloatArray thresh, jfloatArray value, jint order)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::longinus::LonginusDetector *pDetector = (glasssix::longinus::LonginusDetector *)p;

	cv::Mat &vsl_color_image_mat = *(cv::Mat *)vsl_color_image_mat_NativeObj;
	
	jsize bboxSize = env->GetArrayLength(bbox);
	jsize landmarksSize = env->GetArrayLength(landmarks);
	if(bboxSize != landmarksSize)
	{
		LOGW("The size of bbox not match landmarks!");
		env->DeleteLocalRef(clazz);
		return JNI_FALSE;
	}
	
	if(bboxSize == 0)
	{
		LOGW("bbox or landmarks are empty!");
		env->DeleteLocalRef(clazz);
		return JNI_FALSE;
	}
	
	jsize threshSize = env->GetArrayLength(thresh);
	if(threshSize != 2)
	{
		LOGW("threshSize != 2");
		env->DeleteLocalRef(clazz);
		return JNI_FALSE;
	}
	
	jintArray bbox0 = (jintArray)env->GetObjectArrayElement(bbox, 0);
	jintArray landmarks0 = (jintArray)env->GetObjectArrayElement(landmarks, 0);

	jsize bboxDimension = env->GetArrayLength(bbox0);
	jsize landmarksDimension = env->GetArrayLength(landmarks0);
	
	if(bboxDimension != 4 || landmarksDimension != 10)
	{
		LOGW("bboxDimension != 4 or landmarksDimension != 10")
		env->DeleteLocalRef(landmarks0);
		env->DeleteLocalRef(bbox0);
		env->DeleteLocalRef(clazz);
		return JNI_FALSE;
	}

	std::vector<std::vector<int> > bbox_vec, landmarks_vec;
	
	std::vector<int> bbox_vec0(4);
	env->GetIntArrayRegion(bbox0, 0, bboxDimension, (jint *)bbox_vec0.data());
	bbox_vec.push_back(bbox_vec0);
	
	std::vector<int> landmarks_vec0(10);
	env->GetIntArrayRegion(landmarks0, 0, landmarksDimension, (jint *)landmarks_vec0.data());
	landmarks_vec.push_back(landmarks_vec0);
	
	env->DeleteLocalRef(landmarks0);
	env->DeleteLocalRef(bbox0);
		
	for(size_t i = 1; i < bboxSize; i++)
	{
		jintArray bboxi = (jintArray)env->GetObjectArrayElement(bbox, i);
		jintArray landmarksi = (jintArray)env->GetObjectArrayElement(landmarks, i);
		std::vector<int> bbox_veci(4);
		env->GetIntArrayRegion(bboxi, 0, bboxDimension, (jint *)bbox_vec0.data());
		bbox_vec.push_back(bbox_veci);
		
		std::vector<int> landmarks_veci(10);
		env->GetIntArrayRegion(landmarksi, landmarksDimension, (jint *)landmarks_veci.data());
		landmarks_vec.push_back(landmarks_veci);
		
		env->DeleteLocalRef(landmarksi);
		env->DeleteLocalRef(bboxi);
	}
	
	float thresh_[2];
	float value_[2];
	env->GetFloatArrayRegion(thresh, 0, 2, (jfloat *)thresh_);
	bool ret = pDetector->blur_judge_vsl(vsl_color_image_mat.data, vsl_color_image_mat.rows, vsl_color_image_mat.cols, bbox_vec, landmarks_vec, thresh_, value_, order);
	
	env->DeleteLocalRef(value);
	value = env->NewFloatArray(2);
	env->SetFloatArrayRegion(valueArray,0,2,(const jfloat *)value_);

	env->DeleteLocalRef(clazz);

	return ret ? JNI_TRUE : JNI_FALSE;
}
JNIEXPORT jboolean JNICALL blackWhiteJudgeVSLbyMat(JNIEnv *env, jobject thiz, jlong vsl_color_image_mat_NativeObj, jobjectArray bbox, jobjectArray landmarks, jfloatArray thresh, jfloatArray value, jint order)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::longinus::LonginusDetector *pDetector = (glasssix::longinus::LonginusDetector *)p;

	cv::Mat &vsl_color_image_mat = *(cv::Mat *)vsl_color_image_mat_NativeObj;
	
	jsize bboxSize = env->GetArrayLength(bbox);
	jsize landmarksSize = env->GetArrayLength(landmarks);
	if(bboxSize != landmarksSize)
	{
		LOGW("The size of bbox not match landmarks!");
		env->DeleteLocalRef(clazz);
		return JNI_FALSE;
	}
	
	if(bboxSize == 0)
	{
		LOGW("bbox or landmarks are empty!");
		env->DeleteLocalRef(clazz);
		return JNI_FALSE;
	}
	
	jsize threshSize = env->GetArrayLength(thresh);
	if(threshSize != 2)
	{
		LOGW("threshSize != 2");
		env->DeleteLocalRef(clazz);
		return JNI_FALSE;
	}
	
	jintArray bbox0 = (jintArray)env->GetObjectArrayElement(bbox, 0);
	jintArray landmarks0 = (jintArray)env->GetObjectArrayElement(landmarks, 0);

	jsize bboxDimension = env->GetArrayLength(bbox0);
	jsize landmarksDimension = env->GetArrayLength(landmarks0);
	
	if(bboxDimension != 4 || landmarksDimension != 10)
	{
		LOGW("bboxDimension != 4 or landmarksDimension != 10")
		env->DeleteLocalRef(landmarks0);
		env->DeleteLocalRef(bbox0);
		env->DeleteLocalRef(clazz);
		return JNI_FALSE;
	}

	std::vector<std::vector<int> > bbox_vec, landmarks_vec;
	
	std::vector<int> bbox_vec0(4);
	env->GetIntArrayRegion(bbox0, 0, bboxDimension, (jint *)bbox_vec0.data());
	bbox_vec.push_back(bbox_vec0);
	
	std::vector<int> landmarks_vec0(10);
	env->GetIntArrayRegion(landmarks0, 0, landmarksDimension, (jint *)landmarks_vec0.data());
	landmarks_vec.push_back(landmarks_vec0);
	
	env->DeleteLocalRef(landmarks0);
	env->DeleteLocalRef(bbox0);
		
	for(size_t i = 1; i < bboxSize; i++)
	{
		jintArray bboxi = (jintArray)env->GetObjectArrayElement(bbox, i);
		jintArray landmarksi = (jintArray)env->GetObjectArrayElement(landmarks, i);
		std::vector<int> bbox_veci(4);
		env->GetIntArrayRegion(bboxi, 0, bboxDimension, (jint *)bbox_vec0.data());
		bbox_vec.push_back(bbox_veci);
		
		std::vector<int> landmarks_veci(10);
		env->GetIntArrayRegion(landmarksi, landmarksDimension, (jint *)landmarks_veci.data());
		landmarks_vec.push_back(landmarks_veci);
		
		env->DeleteLocalRef(landmarksi);
		env->DeleteLocalRef(bboxi);
	}
	
	float thresh_[2];
	float value_[2];
	env->GetFloatArrayRegion(thresh, 0, 2, (jfloat *)thresh_);
	bool ret = pDetector->black_white_judge_vsl(vsl_color_image_mat.data, vsl_color_image_mat.rows, vsl_color_image_mat.cols, bbox_vec, landmarks_vec, thresh_, value_, order);
	
	env->DeleteLocalRef(value);
	value = env->NewFloatArray(2);
	env->SetFloatArrayRegion(valueArray,0,2,(const jfloat *)value_);

	env->DeleteLocalRef(clazz);

	return ret ? JNI_TRUE : JNI_FALSE;
}
JNIEXPORT jboolean JNICALL facenoseJudgeNIRbyMat(JNIEnv *env, jobject thiz, jlong nir_color_image_mat_NativeObj, jobjectArray bbox, jobjectArray landmarks, jfloatArray thresh, jfloatArray value, jint order)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::longinus::LonginusDetector *pDetector = (glasssix::longinus::LonginusDetector *)p;

	cv::Mat &nir_color_image_mat = *(cv::Mat *)nir_color_image_mat_NativeObj;
	
	jsize bboxSize = env->GetArrayLength(bbox);
	jsize landmarksSize = env->GetArrayLength(landmarks);
	if(bboxSize != landmarksSize)
	{
		LOGW("The size of bbox not match landmarks!");
		env->DeleteLocalRef(clazz);
		return JNI_FALSE;
	}
	
	if(bboxSize == 0)
	{
		LOGW("bbox or landmarks are empty!");
		env->DeleteLocalRef(clazz);
		return JNI_FALSE;
	}
	
	jsize threshSize = env->GetArrayLength(thresh);
	if(threshSize != 2)
	{
		LOGW("threshSize != 2");
		env->DeleteLocalRef(clazz);
		return JNI_FALSE;
	}
	
	jintArray bbox0 = (jintArray)env->GetObjectArrayElement(bbox, 0);
	jintArray landmarks0 = (jintArray)env->GetObjectArrayElement(landmarks, 0);

	jsize bboxDimension = env->GetArrayLength(bbox0);
	jsize landmarksDimension = env->GetArrayLength(landmarks0);
	
	if(bboxDimension != 4 || landmarksDimension != 10)
	{
		LOGW("bboxDimension != 4 or landmarksDimension != 10")
		env->DeleteLocalRef(landmarks0);
		env->DeleteLocalRef(bbox0);
		env->DeleteLocalRef(clazz);
		return JNI_FALSE;
	}

	std::vector<std::vector<int> > bbox_vec, landmarks_vec;
	
	std::vector<int> bbox_vec0(4);
	env->GetIntArrayRegion(bbox0, 0, bboxDimension, (jint *)bbox_vec0.data());
	bbox_vec.push_back(bbox_vec0);
	
	std::vector<int> landmarks_vec0(10);
	env->GetIntArrayRegion(landmarks0, 0, landmarksDimension, (jint *)landmarks_vec0.data());
	landmarks_vec.push_back(landmarks_vec0);
	
	env->DeleteLocalRef(landmarks0);
	env->DeleteLocalRef(bbox0);
		
	for(size_t i = 1; i < bboxSize; i++)
	{
		jintArray bboxi = (jintArray)env->GetObjectArrayElement(bbox, i);
		jintArray landmarksi = (jintArray)env->GetObjectArrayElement(landmarks, i);
		std::vector<int> bbox_veci(4);
		env->GetIntArrayRegion(bboxi, 0, bboxDimension, (jint *)bbox_vec0.data());
		bbox_vec.push_back(bbox_veci);
		
		std::vector<int> landmarks_veci(10);
		env->GetIntArrayRegion(landmarksi, landmarksDimension, (jint *)landmarks_veci.data());
		landmarks_vec.push_back(landmarks_veci);
		
		env->DeleteLocalRef(landmarksi);
		env->DeleteLocalRef(bboxi);
	}
	
	float thresh_[2];
	float value_[2];
	env->GetFloatArrayRegion(thresh, 0, 2, (jfloat *)thresh_);
	bool ret = pDetector->face_nose_judge_nir(nir_color_image_mat.data, nir_color_image_mat.rows, nir_color_image_mat.cols, bbox_vec, landmarks_vec, thresh_, value_, order);
	
	env->DeleteLocalRef(value);
	value = env->NewFloatArray(2);
	env->SetFloatArrayRegion(valueArray,0,2,(const jfloat *)value_);

	env->DeleteLocalRef(clazz);

	return ret ? JNI_TRUE : JNI_FALSE;
}
#endif

JNIEXPORT jobjectArray JNICALL Java_com_glasssix_Longimila_Longimila_detectExbyMetaData(JNIEnv *env, jobject thiz, jbyteArray dataArray,
	jint width, jint height, jint minSize, jfloatArray threshold, jfloat factor, jint stage, jint order)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::longinus::LonginusDetector *pDetector = (glasssix::longinus::LonginusDetector *)p;

	jsize thresholdSize = env->GetArrayLength(threshold);
	std::vector<float> threshold_vec(thresholdSize);
	env->GetFloatArrayRegion(threshold, 0, thresholdSize, (jfloat *)threshold_vec.data());

	jsize dataSize = env->GetArrayLength(dataArray);
	std::vector<unsigned char> data_vec(dataSize, 0);
	env->GetByteArrayRegion(dataArray, 0, dataSize, (jbyte *)data_vec.data());
	jsize size;
	std::vector<glasssix::longinus::FaceRectwithFaceInfo> rects;
	if (data_vec.size() != width * height * 3)
	{
		std::cout << "Data size dismatch! " << "In line " << __LINE__ << " of " << __FILE__ << std::endl;
		size = 0;
	}
	else
	{
		rects = pDetector->detectEx((unsigned char *)data_vec.data(), 3, height, width, minSize, threshold_vec.data(), 1.0f / factor, stage, order);
		glasssix::longinus::sort_descend(rects);
		size = rects.size();
	}
	jclass FaceRectwithFaceInfoClazz = env->FindClass(FaceRectwithFaceInfoClassPath);
	jobjectArray array = env->NewObjectArray(size, FaceRectwithFaceInfoClazz, nullptr);
	jmethodID mid_FaceRectwithFaceInfo_constructor = env->GetMethodID(FaceRectwithFaceInfoClazz, "<init>", "()V");
	jfieldID fid_pts = env->GetFieldID(FaceRectwithFaceInfoClazz, "pts", "[Lcom/glasssix/Longimila/Point;");
	jfieldID fid_yaw = env->GetFieldID(FaceRectwithFaceInfoClazz, "yaw", "F");
	jfieldID fid_pitch = env->GetFieldID(FaceRectwithFaceInfoClazz, "pitch", "F");
	jfieldID fid_roll = env->GetFieldID(FaceRectwithFaceInfoClazz, "roll", "F");

	jclass FaceRectClazz = env->FindClass(FaceRectClassPath);

	jfieldID fid_x = env->GetFieldID(FaceRectClazz, "x", "I");
	jfieldID fid_y = env->GetFieldID(FaceRectClazz, "y", "I");
	jfieldID fid_width = env->GetFieldID(FaceRectClazz, "width", "I");
	jfieldID fid_height = env->GetFieldID(FaceRectClazz, "height", "I");
	jfieldID fid_neighbors = env->GetFieldID(FaceRectClazz, "neighbors", "I");
	jfieldID fid_confidence = env->GetFieldID(FaceRectClazz, "confidence", "D");

	jclass PointClazz = env->FindClass(PointClassPath);
	jmethodID mid_Point_constructor = env->GetMethodID(PointClazz, "<init>", "()V");

	jfieldID fid_Point_x = env->GetFieldID(PointClazz, "x", "I");
	jfieldID fid_Point_y = env->GetFieldID(PointClazz, "y", "I");

	for (size_t i = 0; i < size; ++i)
	{
		jobject FaceRectwithFaceInfoObj = env->NewObject(FaceRectwithFaceInfoClazz, mid_FaceRectwithFaceInfo_constructor);

		env->SetIntField(FaceRectwithFaceInfoObj, fid_x, rects[i].x);
		env->SetIntField(FaceRectwithFaceInfoObj, fid_y, rects[i].y);
		env->SetIntField(FaceRectwithFaceInfoObj, fid_width, rects[i].width);
		env->SetIntField(FaceRectwithFaceInfoObj, fid_height, rects[i].height);
		env->SetIntField(FaceRectwithFaceInfoObj, fid_neighbors, rects[i].neighbors);
		env->SetDoubleField(FaceRectwithFaceInfoObj, fid_confidence, rects[i].confidence);

		jobjectArray ptsArray = (jobjectArray)env->GetObjectField(FaceRectwithFaceInfoObj, fid_pts);
		jsize ptsArraySize = env->GetArrayLength(ptsArray);
		for (size_t j = 0; j < ptsArraySize; j++)
		{
			jobject PointObj = env->NewObject(PointClazz, mid_Point_constructor);
			env->SetIntField(PointObj, fid_Point_x, rects[i].pts[j].x);
			env->SetIntField(PointObj, fid_Point_y, rects[i].pts[j].y);
			env->SetObjectArrayElement(ptsArray, j, PointObj);
			env->DeleteLocalRef(PointObj);
		}

		env->SetFloatField(FaceRectwithFaceInfoObj, fid_yaw, rects[i].yaw);
		env->SetFloatField(FaceRectwithFaceInfoObj, fid_pitch, rects[i].pitch);
		env->SetFloatField(FaceRectwithFaceInfoObj, fid_roll, rects[i].roll);

		env->SetObjectArrayElement(array, i, FaceRectwithFaceInfoObj);

		env->DeleteLocalRef(ptsArray);
		env->DeleteLocalRef(FaceRectwithFaceInfoObj);
	}

	env->DeleteLocalRef(PointClazz);
	env->DeleteLocalRef(FaceRectClazz);
	env->DeleteLocalRef(clazz);
	return array;
}

JNIEXPORT jobjectArray JNICALL Java_com_glasssix_Longimila_Longimila_detectExMobilebyMetaData(JNIEnv *env, jobject thiz, jbyteArray dataArray,
	jint width, jint height, jint minSize, jfloatArray threshold, jfloat factor, jint stage, jint order)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::longinus::LonginusDetector *pDetector = (glasssix::longinus::LonginusDetector *)p;

	jsize thresholdSize = env->GetArrayLength(threshold);
	std::vector<float> threshold_vec(thresholdSize);
	env->GetFloatArrayRegion(threshold, 0, thresholdSize, (jfloat *)threshold_vec.data());

	jsize dataSize = env->GetArrayLength(dataArray);
	std::vector<unsigned char> data_vec(dataSize, 0);
	env->GetByteArrayRegion(dataArray, 0, dataSize, (jbyte *)data_vec.data());
	jsize size;
	std::vector<glasssix::longinus::FaceRectwithFaceInfo> rects;
	if (data_vec.size() != width * height * 3)
	{
		std::cout << "Data size dismatch! " << "In line " << __LINE__ << " of " << __FILE__ << std::endl;
		size = 0;
	}
	else
	{
		rects = pDetector->detectEx_mobile((unsigned char *)data_vec.data(), 3, height, width, minSize, threshold_vec.data(), 1.0f / factor, stage, order);
		glasssix::longinus::sort_descend(rects);
		size = rects.size();
	}
	jclass FaceRectwithFaceInfoClazz = env->FindClass(FaceRectwithFaceInfoClassPath);
	jobjectArray array = env->NewObjectArray(size, FaceRectwithFaceInfoClazz, nullptr);
	jmethodID mid_FaceRectwithFaceInfo_constructor = env->GetMethodID(FaceRectwithFaceInfoClazz, "<init>", "()V");
	jfieldID fid_pts = env->GetFieldID(FaceRectwithFaceInfoClazz, "pts", "[Lcom/glasssix/Longimila/Point;");
	jfieldID fid_yaw = env->GetFieldID(FaceRectwithFaceInfoClazz, "yaw", "F");
	jfieldID fid_pitch = env->GetFieldID(FaceRectwithFaceInfoClazz, "pitch", "F");
	jfieldID fid_roll = env->GetFieldID(FaceRectwithFaceInfoClazz, "roll", "F");

	jclass FaceRectClazz = env->FindClass(FaceRectClassPath);

	jfieldID fid_x = env->GetFieldID(FaceRectClazz, "x", "I");
	jfieldID fid_y = env->GetFieldID(FaceRectClazz, "y", "I");
	jfieldID fid_width = env->GetFieldID(FaceRectClazz, "width", "I");
	jfieldID fid_height = env->GetFieldID(FaceRectClazz, "height", "I");
	jfieldID fid_neighbors = env->GetFieldID(FaceRectClazz, "neighbors", "I");
	jfieldID fid_confidence = env->GetFieldID(FaceRectClazz, "confidence", "D");

	jclass PointClazz = env->FindClass(PointClassPath);
	jmethodID mid_Point_constructor = env->GetMethodID(PointClazz, "<init>", "()V");

	jfieldID fid_Point_x = env->GetFieldID(PointClazz, "x", "I");
	jfieldID fid_Point_y = env->GetFieldID(PointClazz, "y", "I");

	for (size_t i = 0; i < size; ++i)
	{
		jobject FaceRectwithFaceInfoObj = env->NewObject(FaceRectwithFaceInfoClazz, mid_FaceRectwithFaceInfo_constructor);

		env->SetIntField(FaceRectwithFaceInfoObj, fid_x, rects[i].x);
		env->SetIntField(FaceRectwithFaceInfoObj, fid_y, rects[i].y);
		env->SetIntField(FaceRectwithFaceInfoObj, fid_width, rects[i].width);
		env->SetIntField(FaceRectwithFaceInfoObj, fid_height, rects[i].height);
		env->SetIntField(FaceRectwithFaceInfoObj, fid_neighbors, rects[i].neighbors);
		env->SetDoubleField(FaceRectwithFaceInfoObj, fid_confidence, rects[i].confidence);

		jobjectArray ptsArray = (jobjectArray)env->GetObjectField(FaceRectwithFaceInfoObj, fid_pts);
		jsize ptsArraySize = env->GetArrayLength(ptsArray);
		for (size_t j = 0; j < ptsArraySize; j++)
		{
			jobject PointObj = env->NewObject(PointClazz, mid_Point_constructor);
			env->SetIntField(PointObj, fid_Point_x, rects[i].pts[j].x);
			env->SetIntField(PointObj, fid_Point_y, rects[i].pts[j].y);
			env->SetObjectArrayElement(ptsArray, j, PointObj);
			env->DeleteLocalRef(PointObj);
		}

		env->SetFloatField(FaceRectwithFaceInfoObj, fid_yaw, rects[i].yaw);
		env->SetFloatField(FaceRectwithFaceInfoObj, fid_pitch, rects[i].pitch);
		env->SetFloatField(FaceRectwithFaceInfoObj, fid_roll, rects[i].roll);

		env->SetObjectArrayElement(array, i, FaceRectwithFaceInfoObj);

		env->DeleteLocalRef(ptsArray);
		env->DeleteLocalRef(FaceRectwithFaceInfoObj);
	}

	env->DeleteLocalRef(PointClazz);
	env->DeleteLocalRef(FaceRectClazz);
	env->DeleteLocalRef(clazz);
	return array;
}

JNIEXPORT jbyteArray Java_com_glasssix_Longimila_Longimila_alignSingleFacebyMetaData(JNIEnv *env, jobject thiz, jbyteArray dataArray, jint width, jint height)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::longinus::LonginusDetector *pDetector = (glasssix::longinus::LonginusDetector *)p;

	jsize dataSize = env->GetArrayLength(dataArray);
	std::vector<unsigned char> data_vec(dataSize, 0);
	env->GetByteArrayRegion(dataArray, 0, dataSize, (jbyte *)data_vec.data());
	std::vector<unsigned char> aligned_vec;
	if (data_vec.size() != width * height * 1)
	{
		std::cout << "Data size dismatch! " << "In line " << __LINE__ << " of " << __FILE__ << std::endl;
	}
	else
	{
		aligned_vec = pDetector->alignFace((unsigned char *)data_vec.data(), 1, 1, height, width);
	}
	jsize alignedSize = aligned_vec.size();

	jbyteArray aligned_array = env->NewByteArray(alignedSize);
	env->SetByteArrayRegion(aligned_array, 0, alignedSize, (const jbyte *)aligned_vec.data());


	env->DeleteLocalRef(clazz);

	return aligned_array;
}

JNIEXPORT jobjectArray JNICALL Java_com_glasssix_Longimila_Longimila_detectExMobileNIRbyMetaData(JNIEnv *env, jobject thiz, jbyteArray dataArray, jint width, jint height, jint minSize, jfloatArray threshold, jfloat factor, jint stage, jint order)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::longinus::LonginusDetector *pDetector = (glasssix::longinus::LonginusDetector *)p;

	jsize thresholdSize = env->GetArrayLength(threshold);
	std::vector<float> threshold_vec(thresholdSize);
	env->GetFloatArrayRegion(threshold, 0, thresholdSize, (jfloat *)threshold_vec.data());

	jsize dataSize = env->GetArrayLength(dataArray);
	std::vector<unsigned char> data_vec(dataSize, 0);
	env->GetByteArrayRegion(dataArray, 0, dataSize, (jbyte *)data_vec.data());
	
	jsize size;
	std::vector<glasssix::longinus::FaceRectwithFaceInfo> rects;
	if (data_vec.size() != width * height * 3)
	{
		std::cout << "Data size dismatch! " << "In line " << __LINE__ << " of " << __FILE__ << std::endl;
		size = 0;
	}
	else
	{
		rects = pDetector->detectEx((unsigned char *)data_vec.data(), 3, height, width, minSize, threshold_vec.data(), 1.0f / factor, stage, order);
		glasssix::longinus::sort_descend(rects);
		size = rects.size();
	}
	
	jclass FaceRectwithFaceInfoClazz = env->FindClass(FaceRectwithFaceInfoClassPath);
	jobjectArray array = env->NewObjectArray(size, FaceRectwithFaceInfoClazz, nullptr);
	jmethodID mid_FaceRectwithFaceInfo_constructor = env->GetMethodID(FaceRectwithFaceInfoClazz, "<init>", "()V");
	jfieldID fid_pts = env->GetFieldID(FaceRectwithFaceInfoClazz, "pts", "[Lcom/glasssix/Longimila/Point;");
	jfieldID fid_yaw = env->GetFieldID(FaceRectwithFaceInfoClazz, "yaw", "F");
	jfieldID fid_pitch = env->GetFieldID(FaceRectwithFaceInfoClazz, "pitch", "F");
	jfieldID fid_roll = env->GetFieldID(FaceRectwithFaceInfoClazz, "roll", "F");

	jclass FaceRectClazz = env->FindClass(FaceRectClassPath);

	jfieldID fid_x = env->GetFieldID(FaceRectClazz, "x", "I");
	jfieldID fid_y = env->GetFieldID(FaceRectClazz, "y", "I");
	jfieldID fid_width = env->GetFieldID(FaceRectClazz, "width", "I");
	jfieldID fid_height = env->GetFieldID(FaceRectClazz, "height", "I");
	jfieldID fid_neighbors = env->GetFieldID(FaceRectClazz, "neighbors", "I");
	jfieldID fid_confidence = env->GetFieldID(FaceRectClazz, "confidence", "D");

	jclass PointClazz = env->FindClass(PointClassPath);
	jmethodID mid_Point_constructor = env->GetMethodID(PointClazz, "<init>", "()V");

	jfieldID fid_Point_x = env->GetFieldID(PointClazz, "x", "I");
	jfieldID fid_Point_y = env->GetFieldID(PointClazz, "y", "I");

	for (size_t i = 0; i < size; ++i)
	{
		jobject FaceRectwithFaceInfoObj = env->NewObject(FaceRectwithFaceInfoClazz, mid_FaceRectwithFaceInfo_constructor);

		env->SetIntField(FaceRectwithFaceInfoObj, fid_x, rects[i].x);
		env->SetIntField(FaceRectwithFaceInfoObj, fid_y, rects[i].y);
		env->SetIntField(FaceRectwithFaceInfoObj, fid_width, rects[i].width);
		env->SetIntField(FaceRectwithFaceInfoObj, fid_height, rects[i].height);
		env->SetIntField(FaceRectwithFaceInfoObj, fid_neighbors, rects[i].neighbors);
		env->SetDoubleField(FaceRectwithFaceInfoObj, fid_confidence, rects[i].confidence);

		jobjectArray ptsArray = (jobjectArray)env->GetObjectField(FaceRectwithFaceInfoObj, fid_pts);
		jsize ptsArraySize = env->GetArrayLength(ptsArray);
		for (size_t j = 0; j < ptsArraySize; j++)
		{
			jobject PointObj = env->NewObject(PointClazz, mid_Point_constructor);
			env->SetIntField(PointObj, fid_Point_x, rects[i].pts[j].x);
			env->SetIntField(PointObj, fid_Point_y, rects[i].pts[j].y);
			env->SetObjectArrayElement(ptsArray, j, PointObj);
			env->DeleteLocalRef(PointObj);
		}

		env->SetFloatField(FaceRectwithFaceInfoObj, fid_yaw, rects[i].yaw);
		env->SetFloatField(FaceRectwithFaceInfoObj, fid_pitch, rects[i].pitch);
		env->SetFloatField(FaceRectwithFaceInfoObj, fid_roll, rects[i].roll);

		env->SetObjectArrayElement(array, i, FaceRectwithFaceInfoObj);

		env->DeleteLocalRef(ptsArray);
		env->DeleteLocalRef(FaceRectwithFaceInfoObj);
	}

	env->DeleteLocalRef(PointClazz);
	env->DeleteLocalRef(FaceRectClazz);
	env->DeleteLocalRef(FaceRectwithFaceInfoClazz);
	env->DeleteLocalRef(clazz);
	return array;
}

JNIEXPORT jobject JNICALL Java_com_glasssix_Longimila_Longimila_detectExMobilePairbybyMetaData(JNIEnv *env, jobject thiz, jbyteArray vsl_dataArray, jint vsl_width, jint vsl_height, jint vsl_minSize, jfloatArray vsl_threshold, jfloat vsl_factor, jint vsl_stage, jint vsl_order, jbyteArray nir_dataArray, jint nir_width, jint nir_height, jint nir_minSize, jfloatArray nir_threshold, jfloat nir_factor, jint nir_stage, jint nir_order)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::longinus::LonginusDetector *pDetector = (glasssix::longinus::LonginusDetector *)p;
	
	jclass FaceRectwithFaceInfoPairClazz = env->FindClass(FaceRectwithFaceInfoPairPath);
	jmethodID mid_FaceRectwithFaceInfoPair_constructor = env->GetMethodID(FaceRectwithFaceInfoPairClazz, "<init>", "()V");
	jfieldID fid_vsl = env->GetFieldID(FaceRectwithFaceInfoPairClazz, "vsl", "[Lcom/glasssix/Longimila/FaceRectwithFaceInfo;");
	jfieldID fid_nir = env->GetFieldID(FaceRectwithFaceInfoPairClazz, "nir", "[Lcom/glasssix/Longimila/FaceRectwithFaceInfo;");
	jobject pairObj = env->NewObject(FaceRectwithFaceInfoPairClazz, mid_FaceRectwithFaceInfoPair_constructor);

	jsize vsl_thresholdSize = env->GetArrayLength(vsl_threshold);
	std::vector<float> vsl_threshold_vec(vsl_thresholdSize);
	env->GetFloatArrayRegion(vsl_threshold, 0, vsl_thresholdSize, (jfloat *)vsl_threshold_vec.data());
	
	jsize nir_thresholdSize = env->GetArrayLength(nir_threshold);
	std::vector<float> nir_threshold_vec(nir_thresholdSize);
	env->GetFloatArrayRegion(nir_threshold, 0, nir_thresholdSize, (jfloat *)nir_threshold_vec.data());

	jsize vsl_dataSize = env->GetArrayLength(vsl_dataArray);
	std::vector<unsigned char> vsl_data_vec(vsl_dataSize, 0);
	env->GetByteArrayRegion(vsl_dataArray, 0, vsl_dataSize, (jbyte *)vsl_data_vec.data());
	
	jsize nir_dataSize = env->GetArrayLength(nir_dataArray);
	std::vector<unsigned char> nir_data_vec(nir_dataSize, 0);
	env->GetByteArrayRegion(nir_dataArray, 0, nir_dataSize, (jbyte *)nir_data_vec.data());
	
	if (vsl_data_vec.size() != vsl_width * vsl_height * 3 || nir_data_vec.size() != nir_width * nir_height * 3)
	{
		std::cout << "Data size dismatch! " << "In line " << __LINE__ << " of " << __FILE__ << std::endl;
		env->DeleteLocalRef(FaceRectwithFaceInfoPairClazz);
		env->DeleteLocalRef(clazz);
		return pairObj;
	}
	
	std::vector<std::vector<glasssix::longinus::FaceRectwithFaceInfo> > pair_rects = pDetector->detectEx_mobile_pair((unsigned char *)vsl_data_vec.data(), 3, vsl_height, vsl_width, vsl_minSize, vsl_threshold_vec.data(), 1.0f / vsl_factor, vsl_stage, vsl_order, nir_data_vec.data(), 3, nir_height, nir_width, nir_minSizeSize, nir_threshold_vec.data(), 1.0f / nir_factor, nir_stage, nir_order);

	glasssix::longinus::sort_descend(pair_rects[0]);
	glasssix::longinus::sort_descend(pair_rects[1]);
	
	jclass FaceRectwithFaceInfoClazz = env->FindClass(FaceRectwithFaceInfoClassPath);
	jmethodID mid_FaceRectwithFaceInfo_constructor = env->GetMethodID(FaceRectwithFaceInfoClazz, "<init>", "()V");
	
	jsize vsl_size = pair_rects[0].size();
	jsize nir_size = pair_rects[1].size();
	
	jobjectArray vsl_array = env->NewObjectArray(vsl_size, FaceRectwithFaceInfoClazz, nullptr);
	jobjectArray nir_array = env->NewObjectArray(nir_size, FaceRectwithFaceInfoClazz, nullptr);
	
	jfieldID fid_pts = env->GetFieldID(FaceRectwithFaceInfoClazz, "pts", "[Lcom/glasssix/Longimila/Point;");
	jfieldID fid_yaw = env->GetFieldID(FaceRectwithFaceInfoClazz, "yaw", "F");
	jfieldID fid_pitch = env->GetFieldID(FaceRectwithFaceInfoClazz, "pitch", "F");
	jfieldID fid_roll = env->GetFieldID(FaceRectwithFaceInfoClazz, "roll", "F");

	jclass FaceRectClazz = env->FindClass(FaceRectClassPath);

	jfieldID fid_x = env->GetFieldID(FaceRectClazz, "x", "I");
	jfieldID fid_y = env->GetFieldID(FaceRectClazz, "y", "I");
	jfieldID fid_width = env->GetFieldID(FaceRectClazz, "width", "I");
	jfieldID fid_height = env->GetFieldID(FaceRectClazz, "height", "I");
	jfieldID fid_neighbors = env->GetFieldID(FaceRectClazz, "neighbors", "I");
	jfieldID fid_confidence = env->GetFieldID(FaceRectClazz, "confidence", "D");

	jclass PointClazz = env->FindClass(PointClassPath);
	jmethodID mid_Point_constructor = env->GetMethodID(PointClazz, "<init>", "()V");

	jfieldID fid_Point_x = env->GetFieldID(PointClazz, "x", "I");
	jfieldID fid_Point_y = env->GetFieldID(PointClazz, "y", "I");

	for (size_t i = 0; i < vsl_size; ++i)
	{
		jobject FaceRectwithFaceInfoObj = env->NewObject(FaceRectwithFaceInfoClazz, mid_FaceRectwithFaceInfo_constructor);

		env->SetIntField(FaceRectwithFaceInfoObj, fid_x, pair_rects[0][i].x);
		env->SetIntField(FaceRectwithFaceInfoObj, fid_y, pair_rects[0][i].y);
		env->SetIntField(FaceRectwithFaceInfoObj, fid_width, pair_rects[0][i].width);
		env->SetIntField(FaceRectwithFaceInfoObj, fid_height, pair_rects[0][i].height);
		env->SetIntField(FaceRectwithFaceInfoObj, fid_neighbors, pair_rects[0][i].neighbors);
		env->SetDoubleField(FaceRectwithFaceInfoObj, fid_confidence, pair_rects[0][i].confidence);

		jobjectArray ptsArray = (jobjectArray)env->GetObjectField(FaceRectwithFaceInfoObj, fid_pts);
		jsize ptsArraySize = env->GetArrayLength(ptsArray);
		for (size_t j = 0; j < ptsArraySize; j++)
		{
			jobject PointObj = env->NewObject(PointClazz, mid_Point_constructor);
			env->SetIntField(PointObj, fid_Point_x, pair_rects[0][i].pts[j].x);
			env->SetIntField(PointObj, fid_Point_y, pair_rects[0][i].pts[j].y);
			env->SetObjectArrayElement(ptsArray, j, PointObj);
			env->DeleteLocalRef(PointObj);
		}

		env->SetFloatField(FaceRectwithFaceInfoObj, fid_yaw, rects[i].yaw);
		env->SetFloatField(FaceRectwithFaceInfoObj, fid_pitch, rects[i].pitch);
		env->SetFloatField(FaceRectwithFaceInfoObj, fid_roll, rects[i].roll);

		env->SetObjectArrayElement(vsl_array, i, FaceRectwithFaceInfoObj);

		env->DeleteLocalRef(ptsArray);
		env->DeleteLocalRef(FaceRectwithFaceInfoObj);
	}
	
	for (size_t i = 0; i < nir_size; ++i)
	{
		jobject FaceRectwithFaceInfoObj = env->NewObject(FaceRectwithFaceInfoClazz, mid_FaceRectwithFaceInfo_constructor);

		env->SetIntField(FaceRectwithFaceInfoObj, fid_x, pair_rects[1][i].x);
		env->SetIntField(FaceRectwithFaceInfoObj, fid_y, pair_rects[1][i].y);
		env->SetIntField(FaceRectwithFaceInfoObj, fid_width, pair_rects[1][i].width);
		env->SetIntField(FaceRectwithFaceInfoObj, fid_height, pair_rects[1][i].height);
		env->SetIntField(FaceRectwithFaceInfoObj, fid_neighbors, pair_rects[1][i].neighbors);
		env->SetDoubleField(FaceRectwithFaceInfoObj, fid_confidence, pair_rects[1][i].confidence);

		jobjectArray ptsArray = (jobjectArray)env->GetObjectField(FaceRectwithFaceInfoObj, fid_pts);
		jsize ptsArraySize = env->GetArrayLength(ptsArray);
		for (size_t j = 0; j < ptsArraySize; j++)
		{
			jobject PointObj = env->NewObject(PointClazz, mid_Point_constructor);
			env->SetIntField(PointObj, fid_Point_x, pair_rects[1][i].pts[j].x);
			env->SetIntField(PointObj, fid_Point_y, pair_rects[1][i].pts[j].y);
			env->SetObjectArrayElement(ptsArray, j, PointObj);
			env->DeleteLocalRef(PointObj);
		}

		env->SetFloatField(FaceRectwithFaceInfoObj, fid_yaw, rects[i].yaw);
		env->SetFloatField(FaceRectwithFaceInfoObj, fid_pitch, rects[i].pitch);
		env->SetFloatField(FaceRectwithFaceInfoObj, fid_roll, rects[i].roll);

		env->SetObjectArrayElement(nir_array, i, FaceRectwithFaceInfoObj);

		env->DeleteLocalRef(ptsArray);
		env->DeleteLocalRef(FaceRectwithFaceInfoObj);
	}
	
	env->setObjectField(pairObj, fid_vsl, vsl_array);
	env->setObjectField(pairObj, fid_nir, nir_array);

	env->DeleteLocalRef(PointClazz);
	env->DeleteLocalRef(FaceRectClazz);
	env->DeleteLocalRef(FaceRectwithFaceInfoClazz);
	env->DeleteLocalRef(nir_array);
	env->DeleteLocalRef(vsl_array);
	env->DeleteLocalRef(FaceRectwithFaceInfoPairClazz);
	env->DeleteLocalRef(clazz);
	return pairObj;
}

JNIEXPORT jboolean JNICALL blurJudgeVSLbyMetaData(JNIEnv *env, jobject thiz, jbyteArray vsl_color_image_dataArray, jint width, jint height, jobjectArray bbox, jobjectArray landmarks, jfloatArray thresh, jfloatArray value, jint order)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::longinus::LonginusDetector *pDetector = (glasssix::longinus::LonginusDetector *)p;
	
	jsize bboxSize = env->GetArrayLength(bbox);
	jsize landmarksSize = env->GetArrayLength(landmarks);
	if(bboxSize != landmarksSize)
	{
		LOGW("The size of bbox not match landmarks!");
		env->DeleteLocalRef(clazz);
		return JNI_FALSE;
	}
	
	if(bboxSize == 0)
	{
		LOGW("bbox or landmarks are empty!");
		env->DeleteLocalRef(clazz);
		return JNI_FALSE;
	}
	
	jsize threshSize = env->GetArrayLength(thresh);
	if(threshSize != 2)
	{
		LOGW("threshSize != 2");
		env->DeleteLocalRef(clazz);
		return JNI_FALSE;
	}
	
	jsize dataSize = env->GetArrayLength(vsl_color_image_dataArray);
	
	std::vector<glasssix::longinus::FaceRectwithFaceInfo> rects;
	if (dataSize != width * height * 3)
	{
		LODW("Data size dismatch! In line %d of %s", __LINE__, __FILE__);
		env->DeleteLocalRef(clazz);
		return JNI_FALSE;
	}
	
	std::vector<unsigned char> data_vec(dataSize, 0);
	env->GetByteArrayRegion(vsl_color_image_dataArray, 0, dataSize, (jbyte *)data_vec.data());
	
	jintArray bbox0 = (jintArray)env->GetObjectArrayElement(bbox, 0);
	jintArray landmarks0 = (jintArray)env->GetObjectArrayElement(landmarks, 0);

	jsize bboxDimension = env->GetArrayLength(bbox0);
	jsize landmarksDimension = env->GetArrayLength(landmarks0);
	
	if(bboxDimension != 4 || landmarksDimension != 10)
	{
		LOGW("bboxDimension != 4 or landmarksDimension != 10")
		env->DeleteLocalRef(landmarks0);
		env->DeleteLocalRef(bbox0);
		env->DeleteLocalRef(clazz);
		return JNI_FALSE;
	}

	std::vector<std::vector<int> > bbox_vec, landmarks_vec;
	
	std::vector<int> bbox_vec0(4);
	env->GetIntArrayRegion(bbox0, 0, bboxDimension, (jint *)bbox_vec0.data());
	bbox_vec.push_back(bbox_vec0);
	
	std::vector<int> landmarks_vec0(10);
	env->GetIntArrayRegion(landmarks0, 0, landmarksDimension, (jint *)landmarks_vec0.data());
	landmarks_vec.push_back(landmarks_vec0);
	
	env->DeleteLocalRef(landmarks0);
	env->DeleteLocalRef(bbox0);
		
	for(size_t i = 1; i < bboxSize; i++)
	{
		jintArray bboxi = (jintArray)env->GetObjectArrayElement(bbox, i);
		jintArray landmarksi = (jintArray)env->GetObjectArrayElement(landmarks, i);
		std::vector<int> bbox_veci(4);
		env->GetIntArrayRegion(bboxi, 0, bboxDimension, (jint *)bbox_vec0.data());
		bbox_vec.push_back(bbox_veci);
		
		std::vector<int> landmarks_veci(10);
		env->GetIntArrayRegion(landmarksi, landmarksDimension, (jint *)landmarks_veci.data());
		landmarks_vec.push_back(landmarks_veci);
		
		env->DeleteLocalRef(landmarksi);
		env->DeleteLocalRef(bboxi);
	}
	
	float thresh_[2];
	float value_[2];
	env->GetFloatArrayRegion(thresh, 0, 2, (jfloat *)thresh_);
	bool ret = pDetector->blur_judge_vsl(data_vec.data(), height, width, bbox_vec, landmarks_vec, thresh_, value_, order);
	
	env->DeleteLocalRef(value);
	value = env->NewFloatArray(2);
	env->SetFloatArrayRegion(valueArray,0,2,(const jfloat *)value_);

	env->DeleteLocalRef(clazz);

	return ret ? JNI_TRUE : JNI_FALSE;
}
JNIEXPORT jboolean JNICALL blackWhiteJudgeVSLbyMetaDatat(JNIEnv *env, jobject thiz, jbyteArray vsl_color_image_dataArray, jint width, jint height, jobjectArray bbox, jobjectArray landmarks, jfloatArray thresh, jfloatArray value, jint order)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::longinus::LonginusDetector *pDetector = (glasssix::longinus::LonginusDetector *)p;
	
	jsize bboxSize = env->GetArrayLength(bbox);
	jsize landmarksSize = env->GetArrayLength(landmarks);
	if(bboxSize != landmarksSize)
	{
		LOGW("The size of bbox not match landmarks!");
		env->DeleteLocalRef(clazz);
		return JNI_FALSE;
	}
	
	if(bboxSize == 0)
	{
		LOGW("bbox or landmarks are empty!");
		env->DeleteLocalRef(clazz);
		return JNI_FALSE;
	}
	
	jsize threshSize = env->GetArrayLength(thresh);
	if(threshSize != 2)
	{
		LOGW("threshSize != 2");
		env->DeleteLocalRef(clazz);
		return JNI_FALSE;
	}
	
	jsize dataSize = env->GetArrayLength(vsl_color_image_dataArray);
	
	std::vector<glasssix::longinus::FaceRectwithFaceInfo> rects;
	if (dataSize != width * height * 3)
	{
		LODW("Data size dismatch! In line %d of %s", __LINE__, __FILE__);
		env->DeleteLocalRef(clazz);
		return JNI_FALSE;
	}
	
	std::vector<unsigned char> data_vec(dataSize, 0);
	env->GetByteArrayRegion(vsl_color_image_dataArray, 0, dataSize, (jbyte *)data_vec.data());
	
	jintArray bbox0 = (jintArray)env->GetObjectArrayElement(bbox, 0);
	jintArray landmarks0 = (jintArray)env->GetObjectArrayElement(landmarks, 0);

	jsize bboxDimension = env->GetArrayLength(bbox0);
	jsize landmarksDimension = env->GetArrayLength(landmarks0);
	
	if(bboxDimension != 4 || landmarksDimension != 10)
	{
		LOGW("bboxDimension != 4 or landmarksDimension != 10")
		env->DeleteLocalRef(landmarks0);
		env->DeleteLocalRef(bbox0);
		env->DeleteLocalRef(clazz);
		return JNI_FALSE;
	}

	std::vector<std::vector<int> > bbox_vec, landmarks_vec;
	
	std::vector<int> bbox_vec0(4);
	env->GetIntArrayRegion(bbox0, 0, bboxDimension, (jint *)bbox_vec0.data());
	bbox_vec.push_back(bbox_vec0);
	
	std::vector<int> landmarks_vec0(10);
	env->GetIntArrayRegion(landmarks0, 0, landmarksDimension, (jint *)landmarks_vec0.data());
	landmarks_vec.push_back(landmarks_vec0);
	
	env->DeleteLocalRef(landmarks0);
	env->DeleteLocalRef(bbox0);
		
	for(size_t i = 1; i < bboxSize; i++)
	{
		jintArray bboxi = (jintArray)env->GetObjectArrayElement(bbox, i);
		jintArray landmarksi = (jintArray)env->GetObjectArrayElement(landmarks, i);
		std::vector<int> bbox_veci(4);
		env->GetIntArrayRegion(bboxi, 0, bboxDimension, (jint *)bbox_vec0.data());
		bbox_vec.push_back(bbox_veci);
		
		std::vector<int> landmarks_veci(10);
		env->GetIntArrayRegion(landmarksi, landmarksDimension, (jint *)landmarks_veci.data());
		landmarks_vec.push_back(landmarks_veci);
		
		env->DeleteLocalRef(landmarksi);
		env->DeleteLocalRef(bboxi);
	}
	
	float thresh_[2];
	float value_[2];
	env->GetFloatArrayRegion(thresh, 0, 2, (jfloat *)thresh_);
	bool ret = pDetector->black_white_judge_vsl(data_vec.data(), height, width, bbox_vec, landmarks_vec, thresh_, value_, order);
	
	env->DeleteLocalRef(value);
	value = env->NewFloatArray(2);
	env->SetFloatArrayRegion(valueArray,0,2,(const jfloat *)value_);

	env->DeleteLocalRef(clazz);

	return ret ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL facenoseJudgeNIRbyMetaData(JNIEnv *env, jobject thiz, jbyteArray nir_color_image_dataArray, jint width, jint height, jobjectArray bbox, jobjectArray landmarks, jfloatArray thresh, jfloatArray value, jint order)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::longinus::LonginusDetector *pDetector = (glasssix::longinus::LonginusDetector *)p;
	
	jsize bboxSize = env->GetArrayLength(bbox);
	jsize landmarksSize = env->GetArrayLength(landmarks);
	if(bboxSize != landmarksSize)
	{
		LOGW("The size of bbox not match landmarks!");
		env->DeleteLocalRef(clazz);
		return JNI_FALSE;
	}
	
	if(bboxSize == 0)
	{
		LOGW("bbox or landmarks are empty!");
		env->DeleteLocalRef(clazz);
		return JNI_FALSE;
	}
	
	jsize threshSize = env->GetArrayLength(thresh);
	if(threshSize != 2)
	{
		LOGW("threshSize != 2");
		env->DeleteLocalRef(clazz);
		return JNI_FALSE;
	}
	
	jsize dataSize = env->GetArrayLength(nir_color_image_dataArray);
	
	std::vector<glasssix::longinus::FaceRectwithFaceInfo> rects;
	if (dataSize != width * height * 3)
	{
		LODW("Data size dismatch! In line %d of %s", __LINE__, __FILE__);
		env->DeleteLocalRef(clazz);
		return JNI_FALSE;
	}
	
	std::vector<unsigned char> data_vec(dataSize, 0);
	env->GetByteArrayRegion(nir_color_image_dataArray, 0, dataSize, (jbyte *)data_vec.data());
	
	jintArray bbox0 = (jintArray)env->GetObjectArrayElement(bbox, 0);
	jintArray landmarks0 = (jintArray)env->GetObjectArrayElement(landmarks, 0);

	jsize bboxDimension = env->GetArrayLength(bbox0);
	jsize landmarksDimension = env->GetArrayLength(landmarks0);
	
	if(bboxDimension != 4 || landmarksDimension != 10)
	{
		LOGW("bboxDimension != 4 or landmarksDimension != 10")
		env->DeleteLocalRef(landmarks0);
		env->DeleteLocalRef(bbox0);
		env->DeleteLocalRef(clazz);
		return JNI_FALSE;
	}

	std::vector<std::vector<int> > bbox_vec, landmarks_vec;
	
	std::vector<int> bbox_vec0(4);
	env->GetIntArrayRegion(bbox0, 0, bboxDimension, (jint *)bbox_vec0.data());
	bbox_vec.push_back(bbox_vec0);
	
	std::vector<int> landmarks_vec0(10);
	env->GetIntArrayRegion(landmarks0, 0, landmarksDimension, (jint *)landmarks_vec0.data());
	landmarks_vec.push_back(landmarks_vec0);
	
	env->DeleteLocalRef(landmarks0);
	env->DeleteLocalRef(bbox0);
		
	for(size_t i = 1; i < bboxSize; i++)
	{
		jintArray bboxi = (jintArray)env->GetObjectArrayElement(bbox, i);
		jintArray landmarksi = (jintArray)env->GetObjectArrayElement(landmarks, i);
		std::vector<int> bbox_veci(4);
		env->GetIntArrayRegion(bboxi, 0, bboxDimension, (jint *)bbox_vec0.data());
		bbox_vec.push_back(bbox_veci);
		
		std::vector<int> landmarks_veci(10);
		env->GetIntArrayRegion(landmarksi, landmarksDimension, (jint *)landmarks_veci.data());
		landmarks_vec.push_back(landmarks_veci);
		
		env->DeleteLocalRef(landmarksi);
		env->DeleteLocalRef(bboxi);
	}
	
	float thresh_[2];
	float value_[2];
	env->GetFloatArrayRegion(thresh, 0, 2, (jfloat *)thresh_);
	bool ret = pDetector->face_nose_judge_nir(data_vec.data(), height, width, bbox_vec, landmarks_vec, thresh_, value_, order);
	
	env->DeleteLocalRef(value);
	value = env->NewFloatArray(2);
	env->SetFloatArrayRegion(valueArray,0,2,(const jfloat *)value_);

	env->DeleteLocalRef(clazz);

	return ret ? JNI_TRUE : JNI_FALSE;
}

//JNIEXPORT void JNICALL Java_com_glasssix_Longimila_Longimila_extract_biggest_faceinfo(JNIEnv *env, jclass clazz, jobjectArray face_info_Array, jobjectArray bboxArray, jobjectArray landmarkArray)
//{
//	jclass FaceRectwithFaceInfoClazz = env->FindClass(FaceRectwithFaceInfoClassPath);
//	jfieldID fid_pts = env->GetFieldID(FaceRectwithFaceInfoClazz, "pts", "[Lcom/glasssix/Longimila/Point;");
//	jfieldID fid_yaw = env->GetFieldID(FaceRectwithFaceInfoClazz, "yaw", "F");
//	jfieldID fid_pitch = env->GetFieldID(FaceRectwithFaceInfoClazz, "pitch", "F");
//	jfieldID fid_roll = env->GetFieldID(FaceRectwithFaceInfoClazz, "roll", "F");
//
//	jclass FaceRectClazz = env->FindClass(FaceRectClassPath);
//	jfieldID fid_x = env->GetFieldID(FaceRectClazz, "x", "I");
//	jfieldID fid_y = env->GetFieldID(FaceRectClazz, "y", "I");
//	jfieldID fid_width = env->GetFieldID(FaceRectClazz, "width", "I");
//	jfieldID fid_height = env->GetFieldID(FaceRectClazz, "height", "I");
//	jfieldID fid_neighbors = env->GetFieldID(FaceRectClazz, "neighbors", "I");
//	jfieldID fid_confidence = env->GetFieldID(FaceRectClazz, "confidence", "D");
//
//	jclass PointClazz = env->FindClass(PointClassPath);
//	jfieldID fid_Point_x = env->GetFieldID(PointClazz, "x", "I");
//	jfieldID fid_Point_y = env->GetFieldID(PointClazz, "y", "I");
//
//	std::vector<glasssix::longinus::FaceRectwithFaceInfo> face_info_vec;
//	jsize face_num = env->GetArrayLength(face_info_Array);
//	for (size_t i = 0; i < face_num; i++)
//	{
//		jobject FaceInfoObj = env->GetObjectArrayElement(face_info_Array, i);
//		jobjectArray ptsArray = (jobjectArray)env->GetObjectField(FaceInfoObj, fid_pts);
//		jsize ptsArraySize = env->GetArrayLength(ptsArray);
//		glasssix::longinus::FaceRectwithFaceInfo face_info_;
//		for (size_t j = 0; j < ptsArraySize; j++)
//		{
//			jobject PointObj = env->GetObjectArrayElement(ptsArray, j);
//			face_info_.pts[j].x = env->GetIntField(PointObj, fid_Point_x);
//			face_info_.pts[j].y = env->GetIntField(PointObj, fid_Point_y);
//			env->DeleteLocalRef(PointObj);
//		}
//		env->DeleteLocalRef(ptsArray);
//
//		face_info_.yaw = env->GetFloatField(FaceInfoObj, fid_yaw);
//		face_info_.pitch = env->GetFloatField(FaceInfoObj, fid_pitch);
//		face_info_.roll = env->GetFloatField(FaceInfoObj, fid_roll);
//
//		face_info_.x = env->GetIntField(FaceInfoObj, fid_x);
//		face_info_.y = env->GetIntField(FaceInfoObj, fid_y);
//		face_info_.width = env->GetIntField(FaceInfoObj, fid_width);
//		face_info_.height = env->GetIntField(FaceInfoObj, fid_height);
//		face_info_.neighbors = env->GetIntField(FaceInfoObj, fid_neighbors);
//		face_info_.confidence = env->GetDoubleField(FaceInfoObj, fid_confidence);
//
//		face_info_vec.push_back(face_info_);
//
//		env->DeleteLocalRef(FaceInfoObj);
//	}
//	
//	std::vector<std::vector<int> > bboxes;
//	std::vector<std::vector<int> > landmarks;
//
//	glasssix::longinus::extract_biggest_faceinfo(face_info_vec, bboxes, landmarks);
//
//	jintArray bbox0 = (jintArray)env->GetObjectArrayElement(bboxArray, 0);
//	jsize bboxSize = env->GetArrayLength(bbox0);
//	env->SetIntArrayRegion(bbox0, 0, bboxSize, (const jint *)bboxes[0].data());
//
//	///////It will be continue
//
//	env->DeleteLocalRef(PointClazz);
//	env->DeleteLocalRef(FaceRectClazz);
//	env->DeleteLocalRef(clazz);
//}

#endif