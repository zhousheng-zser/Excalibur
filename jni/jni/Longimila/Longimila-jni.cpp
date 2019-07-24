#include "Longimila-jni.hpp"
#include "LonginusDetector.hpp"
#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif
#include <string>
#include <vector>

static const char *FaceRectClassPath = "com/glasssix/Longimila/FaceRect";
static const char *FaceRectwithFaceInfoClassPath = "com/glasssix/Longimila/FaceRectwithFaceInfo";
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
JNIEXPORT jobjectArray JNICALL Java_com_glasssix_Longimila_Longimila_detect(JNIEnv *env, jobject thiz, jlong grayNativeObj, jint minSize, jfloat scale, jint minNeighbors)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::longinus::LonginusDetector *pDetector = (glasssix::longinus::LonginusDetector *)p;
	
	cv::Mat &gray = *(cv::Mat *)grayNativeObj;
	std::vector<glasssix::longinus::FaceRect> rects = pDetector->detect(gray.data, gray.cols, gray.rows, gray.step[0], minSize, scale, minNeighbors, false, false);
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

JNIEXPORT jobjectArray JNICALL Java_com_glasssix_Longimila_Longimila_detectwithInfo(JNIEnv *env, jobject thiz, jlong grayNativeObj, jint minSize, jfloat scale, jint minNeighbors, jint order)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::longinus::LonginusDetector *pDetector = (glasssix::longinus::LonginusDetector *)p;
	
	cv::Mat &gray = *(cv::Mat *)grayNativeObj;
	std::vector<glasssix::longinus::FaceRectwithFaceInfo> rects = pDetector->detect(gray.data, gray.cols, gray.rows, gray.step[0], minSize, scale, minNeighbors, order, false, false);
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

JNIEXPORT jobjectArray JNICALL Java_com_glasssix_Longimila_Longimila_match(JNIEnv *env, jobject thiz, jobjectArray faceRectArray, jint frame_extract_frequency)
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
	
	std::vector<glasssix::longinus::Match_Retval> match_vec = pDetector->match(faceRects, frame_extract_frequency);
	
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
JNIEXPORT jbyteArray JNICALL Java_com_glasssix_Longimila_Longimila_alignFace(JNIEnv *env, jobject thiz, jlong grayNativeObj, jobjectArray bboxArray, jobjectArray landmarkArray)
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

JNIEXPORT jbyteArray Java_com_glasssix_Longimila_Longimila_alignSingleFace(JNIEnv *env, jobject thiz, jlong grayNativeObj)
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
#endif

#ifndef TRIAL
#ifdef USE_OPENCV
JNIEXPORT jobjectArray JNICALL Java_com_glasssix_Longimila_Longimila_detectEx(JNIEnv *env, jobject thiz, jlong matNativeObj, jint minSize, jfloatArray threshold, jfloat factor, jint stage, jint order)
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
#endif