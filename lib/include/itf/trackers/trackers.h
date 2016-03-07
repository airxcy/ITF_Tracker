//
//  klt_gpu.h
//  ITF_Inegrated
//
//  Created by Chenyang Xia on 8/18/2015.
//  Copyright (c) 2015 CUHK. All rights reserved.
//

#ifndef KLTTRACKER_H
#define KLTTRACKER_H
#include "itf/trackers/buffgpu.h"
#include "itf/trackers/buffers.h"
#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <ctime>
#include <fstream>

#define PI 3.14159265

#define minDist 2
#define minGSize 1
#define TIMESPAN 20
#define COSTHRESH 0.4
#define VeloThresh 0.1
#define KnnK 40
#define MoveFactor 0.0001
#define coNBThresh 0
#define minTrkLen 2
#define Pers2Range(pers) pers/6;

class TargetFinder
{
public:
    TargetFinder(int maxCorners = 1000, double qualityLevel = 0.01,
        int blockSize = 3);
    int maxCorners;
    double qualityLevel;
    int blockSize;
    cv::gpu::Stream _stream;
    cudaStream_t _stream_t;
    void releaseMemory();


    void setPersMat(cv::gpu::GpuMat& m,int w,int h);
    //! return 1 rows matrix with CV_32FC2 type
    void operator ()(const cv::gpu::GpuMat& image, cv::gpu::GpuMat& corners, const cv::gpu::GpuMat& mask= cv::gpu::GpuMat());
private:

    int fw,fh;
    cv::Mat cpuPersMap;
    cv::Mat rangeMat;
    cv::Mat tmpMat;
    Buff<float> pointBuff;
    cv::gpu::GpuMat persMap;
    cv::gpu::GpuMat Dx_;
    cv::gpu::GpuMat Dy_;
    cv::gpu::GpuMat buf_;
    cv::gpu::GpuMat eig_;
    cv::gpu::GpuMat minMaxbuf_;
    cv::gpu::GpuMat tmpCorners_;
};
enum TrackerStatus {FINE=0,TRACKINGERROR};
class CrowdTracker
{
public:
    CrowdTracker();

    ~CrowdTracker();
    /***** CPU *****/
    int init(int w,int h,unsigned char* framedata,int nPoints);
    int selfinit(unsigned char* framedata);
    int updateAframe(unsigned char* framedata,int fidx);
    void releaseMemory();
    void setUpPersMap(float *srcMap);
    void updateROICPU(float* aryPtr,int length);
    void updateSegCPU(unsigned char* ptr);
    void updateSegNeg(float* aryPtr,int length);
    void Render(unsigned char * framedata);

    bool isPersDone(){return persDone;}
    void setGroupTrack(bool flag){groupOnFlag=flag;}
    void useRender(bool flag){render=flag;}
    void useSeg(bool flag){applyseg=flag;}
    bool isSegOn(){return applyseg;}
    float* getDistCo(){return distCo->cpu_ptr();}
    float* getVeloCo(){return veloCo->cpu_ptr();}
    float* getCosCo(){return cosCo->cpu_ptr();}
    int* getNbCount(){return nbCount->cpu_ptr();}
    Tracks* getTracks(){return tracksGPU;}
    int getNFeatures(){return nFeatures;}
    int getNSearch(){return nSearch;}
    float2* getCorners(){return corners->cpu_ptr();}
    int* getLabel(){return label->cpu_ptr();}
    int* getPreLabel(){return prelabel->cpu_ptr();}
    unsigned char* getClrvec(){return clrvec->cpu_ptr();}
    Groups& getGroups(){return *groups;}
    GroupTracks& getGroupsTrk(){return *groupsTrk;}
    TrackerStatus curStatus;
	std::ofstream debuggingFile;
//private:
    /** Basic **/
    int frame_width=0, frame_height=0;
    int frameidx=0;
    bool persDone=false,render=true;


    /** Point Tracking and Detecting **/
    int nFeatures=0,nSearch=0;
    TracksInfo trkInfo;
    Tracks* tracksGPU;
    bool applyseg=false;

    bool checkTrackMoving(FeatBuff &strk);
    void PointTracking();
    void findPoints();
    void filterTrackGPU();


    /** Grouping **/
    bool groupOnFlag=true;
    int curK=0,pregroupN=0,groupN=0,maxgroupN=0;
    void PersExcludeMask();
    void pointCorelate();
    std::vector<int> bfsearchitems;
    void bfsearch();
    void newBFS();
    void makeGroups();
    void matchGroups();
    void updateGroupsTracks();
	void KLT_updates_Group(int );
    MemBuff<int>* prelabel,* label,* ptNumInGroup;

    /** Render Frame **/
    int zoomW=340,zoomH=255;
    unsigned char * h_zoomFrame;

    /***** GPU *****/
    cudaStream_t cornerStream;
    cv::gpu::Stream cornerCVStream;
    cudaStream_t CorrStream;
    std::vector<cudaStream_t> streams;
    //cv::gpu::GoodFeaturesToTrackDetector_GPU* detector;
    TargetFinder* detector;
    cv::gpu::PyrLKOpticalFlow* tracker;
    cv::gpu::GpuMat gpuGray, gpuPreGray,rgbMat,maskMat,roiMaskMat,gpuPersMap,gpuSegMat,
            gpuCorners, gpuPrePts, gpuNextPts,gpuStatus;
    //Basic
    unsigned char * d_rgbframedata;
    MemBuff<float>* persMap;
    //Rendering
    MemBuff<unsigned char>* renderMask;
    MemBuff<unsigned char>* clrvec;
    //Tracking
    MemBuff<unsigned char> *mask,*roimask,*segmask,*segNeg;
    unsigned char * d_status;
    MemBuff<float2>* corners;
	//Background subtraction
	MemBuff<unsigned char>* foreground;
	cv::gpu::GpuMat fgmat;
	cv::gpu::MOG2_GPU* mog2;
    //Grouping
    MemBuff<int>* nbCount;
    MemBuff<float>* distCo,*cosCo,*veloCo,*correlation;
    int *tmpn,*idxmap;
    Groups* groups;
    GroupTracks* groupsTrk;
    MemBuff<int>* overLap;
    MemBuff<int>* matchOld2New,* matchNew2Old;
    MemBuff<int>* rankingOld,* rankingNew,* rankCountOld,* rankCountNew;
    MemBuff<float>* scoreOld,* scoreNew;
	MemBuff<float2>* kltUpdateVec;
	MemBuff<BBox>* kltUpdateBoxVec;
    //cublasHandle_t handle;
};

#endif
