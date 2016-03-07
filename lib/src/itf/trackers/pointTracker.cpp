#include "itf/trackers/trackers.h"

#include <cmath>
#include <fstream>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/gpu/stream_accessor.hpp>

#include <cuda_runtime.h>
#include "itf/trackers/gpucommon.hpp"
#include "itf/trackers/utils.h"
using namespace cv;
using namespace cv::gpu;


CrowdTracker::CrowdTracker()
{
    frame_width=0, frame_height=0;
    frameidx=0;
    nFeatures=0,nSearch=0; 
    /**cuda **/
    persDone=false;
    groupOnFlag=true;
}
CrowdTracker::~CrowdTracker()
{
    releaseMemory();
}
void setHW(int w,int h);
int CrowdTracker::init(int w, int h,unsigned char* framedata,int nPoints)
{
    /** Checking Device Properties **/
    int nDevices;
    int maxthread=0;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        /*
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6);
        debuggingFile << "maxgridDim" << prop.maxGridSize[0] << "," << prop.maxGridSize[1] << "," << prop.maxGridSize[2] << std::endl;
        debuggingFile<<"maxThreadsPerBlock:"<<prop.maxThreadsPerBlock<<std::endl;
        */

        //cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize,MyKernel, 0, arrayCount);
        if(maxthread==0)maxthread=prop.maxThreadsPerBlock;
        //debuggingFile << prop.major << "," << prop.minor << std::endl;
    }


    /** Basic **/
    frame_width = w,frame_height = h;
    setHW(w,h);
    frameidx=0;
    gpuGray=gpu::GpuMat(frame_height, frame_width, CV_8UC1 );
    gpuPreGray=gpu::GpuMat(frame_height, frame_width, CV_8UC1 );

    gpu_zalloc(d_rgbframedata,frame_height*frame_width*3,sizeof(unsigned char));
    rgbMat = gpu::GpuMat(frame_height, frame_width, CV_8UC3 ,d_rgbframedata);
    persMap =  new MemBuff<float>(frame_height*frame_width);
    gpuPersMap= gpu::GpuMat(frame_height, frame_width, CV_32F ,persMap->gpu_ptr());
    roimask =  new MemBuff<unsigned char>(frame_height*frame_width);
    roiMaskMat = gpu::GpuMat(frame_height, frame_width, CV_8UC1 ,roimask->gpu_ptr());
	debuggingFile.open("trackerDump.txt", std::ofstream::out);

    /** Point Tracking and Detecting **/
    nFeatures = maxthread;//(maxthread>1024)?1024:maxthread;
    nFeatures = (maxthread>nPoints)?nPoints:maxthread;
    nSearch=nFeatures;
    tracksGPU  = new Tracks();
    tracksGPU->init(nFeatures,nFeatures);
    //detector=new  gpu::GoodFeaturesToTrackDetector_GPU(nSearch,1e-30,0,3);
    detector =new TargetFinder(nSearch,1e-30,3);
    tracker =new  gpu::PyrLKOpticalFlow();
    tracker->winSize=Size(9,9);
    tracker->maxLevel=3;
    tracker->iters=10;

    corners= new MemBuff<float2>(nSearch);

    cornerCVStream=gpu::Stream();
    cornerStream=gpu::StreamAccessor::getStream(cornerCVStream);
    detector->_stream=cornerCVStream;
    detector->_stream_t=cornerStream;
    cudaStreamCreate(&CorrStream);

    gpuCorners=gpu::GpuMat(1, nSearch, CV_32FC2,corners->gpu_ptr());
    gpu_zalloc(d_status,nFeatures,sizeof(unsigned char*));
    gpuStatus=gpu::GpuMat(1,nFeatures,CV_8UC1,d_status);
    mask = new MemBuff<unsigned char>(frame_height*frame_width);
    maskMat = gpu::GpuMat(frame_height, frame_width, CV_8UC1 ,mask->gpu_ptr());
    segmask = new MemBuff<unsigned char>(frame_height*frame_width);
    gpuSegMat =gpu::GpuMat(frame_height, frame_width, CV_8UC1 ,segmask->gpu_ptr());
    segNeg = new MemBuff<unsigned char>(frame_height*frame_width);
	/** Background Subtraction **/
	foreground = new MemBuff<unsigned char>(frame_height*frame_width);
	fgmat = gpu::GpuMat(frame_height, frame_width, CV_8UC1, foreground->gpu_ptr());
	mog2 = new gpu::MOG2_GPU(10);
	//mog2->
    /** Grouping **/
    //Neighbor Search
    nbCount = new MemBuff<int>(nFeatures*nFeatures);
    distCo= new MemBuff<float>(nFeatures*nFeatures);
    cosCo = new MemBuff<float>(nFeatures*nFeatures);
    veloCo= new MemBuff<float>(nFeatures*nFeatures);
    correlation = new MemBuff<float>(nFeatures);

    //k-NN

    //BFsearch
    curK=0,groupN=0;
    bfsearchitems.resize(nFeatures);
    tmpn = (int*)zalloc(nFeatures,sizeof(int));
    idxmap= (int*)zalloc(nFeatures,sizeof(int));
    //label Re-Map
    maxgroupN=0;
    prelabel = new MemBuff<int>(nFeatures);
    label = new MemBuff<int>(nFeatures);
    ptNumInGroup = new MemBuff<int>(nFeatures);
    groups = new Groups();
    groups->init(nFeatures,tracksGPU);
    groupsTrk = new GroupTracks();
    groupsTrk->init(nFeatures);
    streams=std::vector<cudaStream_t>(MAXSTREAM);
    for(int i=0;i<MAXSTREAM;i++)
    {
        cudaStreamCreate(&streams[i]);
    }
    overLap = new MemBuff<int>(nFeatures*nFeatures);
    matchOld2New = new MemBuff<int>(nFeatures);
    matchNew2Old = new MemBuff<int>(nFeatures);

    rankCountNew = new MemBuff<int>(nFeatures);
    rankCountOld = new MemBuff<int>(nFeatures);
    rankingNew = new MemBuff<int>(nFeatures*nFeatures);
    rankingOld = new MemBuff<int>(nFeatures*nFeatures);
    scoreNew = new MemBuff<float>(nFeatures*nFeatures);
    scoreOld = new MemBuff<float>(nFeatures*nFeatures);
	kltUpdateVec = new MemBuff<float2>(nFeatures);
	kltUpdateBoxVec = new MemBuff<BBox>(nFeatures);
    /**  render **/
    h_zoomFrame=(unsigned char *)zalloc(zoomW*zoomH*3,sizeof(unsigned char));
    renderMask=new MemBuff<unsigned char>(frame_width*frame_height,3);
    clrvec = new MemBuff<unsigned char>(nFeatures,3);



    /** Self Init **/
    selfinit(framedata);
    debuggingFile<< "inited" << std::endl;
    return 1;
}
void CrowdTracker::releaseMemory()
{
    tracker->releaseMemory();
    detector->releaseMemory();
    gpuGray.release();
    gpuPreGray.release();
    rgbMat.release();
    gpuCorners.release();
    gpuPrePts.release();
    gpuNextPts.release();
    gpuStatus.release();
}
int CrowdTracker::selfinit(unsigned char* framedata)
{
    Mat curframe(frame_height,frame_width,CV_8UC3,framedata);
    rgbMat.upload(curframe);
    gpu::cvtColor(rgbMat,gpuGray,CV_RGB2GRAY);
    gpuGray.copyTo(gpuPreGray);
    (*detector)(gpuGray, gpuCorners);
    gpuCorners.copyTo(gpuPrePts);
    cudaMemset(mask->gpu_ptr(),255,frame_width*frame_height*sizeof(unsigned char));
    cudaMemset(roimask->gpu_ptr(),255,frame_width*frame_height*sizeof(unsigned char));
    return true;
}

void CrowdTracker::setUpPersMap(float* srcMap)
{
	/*
	// camera calibration
    for(int y=0;y<frame_height;y++)
        for(int x=0;x<frame_width;x++)
        {
            float cdist=(frame_width/2.0-abs(x-frame_width/2.0))/frame_width*10;
            srcMap[y*frame_width+x]=srcMap[y*frame_width+x]+cdist*cdist;
        }
		*/
    persMap->updateCPU(srcMap);
    persMap->SyncH2D();
    detector->setPersMat(gpuPersMap,frame_width,frame_height);
}
void CrowdTracker::updateROICPU(float* aryPtr,int length)
{
    roimask->toZeroD();
    roimask->toZeroH();
    unsigned char* h_roimask=roimask->cpu_ptr();
    std::vector<Point2f> roivec;
    int counter=0;
    for(int i=0;i<length;i++)
    {
        Point2f p(aryPtr[i*2],aryPtr[i*2+1]);
        roivec.push_back(p);
    }
    for(int i=0;i<frame_height;i++)
    {
        for(int j=0;j<frame_width;j++)
        {
            if(pointPolygonTest(roivec,Point2f(j,i),true)>0)
            {
                h_roimask[i*frame_width+j]=255;
                counter++;

            }
        }
    }
    debuggingFile<<counter<<std::endl;
    roimask->SyncH2D();
    cudaMemcpy(mask->gpu_ptr(),roimask->gpu_ptr(),frame_height*frame_width*sizeof(unsigned char),cudaMemcpyDeviceToDevice);
}
void CrowdTracker::updateSegCPU(unsigned char* ptr)
{
    //Mat kernel=Mat::ones(5,5,CV_8UC1);
    cudaMemcpy(segmask->gpu_ptr(),ptr,frame_height*frame_width,cudaMemcpyHostToDevice);
    //dilate(gpuSegMat, gpuDiaMat, kernel, Point(-3, -3));
    //cudaMemcpy(d_segmask,gpuDiaMat.data,frame_height*frame_width,cudaMemcpyHostToDevice);

}
void CrowdTracker::updateSegNeg(float* aryPtr,int length)
{
    unsigned char * h_segNeg = segNeg->cpu_ptr();
    segNeg->toZeroD();
    segNeg->toZeroH();
    std::vector<Point2f> roivec;
    int counter=0;
    for(int i=0;i<length;i++)
    {
        Point2f p(aryPtr[i*2],aryPtr[i*2+1]);
        roivec.push_back(p);
    }
    for(int i=0;i<frame_height;i++)
    {
        for(int j=0;j<frame_width;j++)
        {
            if(pointPolygonTest(roivec,Point2f(j,i),true)>0)
            {
                h_segNeg[i*frame_width+j]=255;
                counter++;

            }
        }
    }
    debuggingFile<<counter<<std::endl;
    segNeg->SyncH2D();
}


bool CrowdTracker::checkTrackMoving(FeatBuff &strk)
{
    bool isTrkValid = true;
    if(strk.len>1)
    {
        float* h_persMap=persMap->cpu_ptr();
        PntT xb=strk.cur_frame_ptr->x,yb=strk.cur_frame_ptr->y;
        float persval = h_persMap[yb*frame_width+xb];
        PntT prex=strk.getPtr(strk.len-2)->x, prey=strk.getPtr(strk.len-2)->y;
        double trkdist=abs(prex-xb)+abs(prey-yb);
        if(trkdist>persval)return false;
        int Movelen=150/sqrt(persval),startidx=max(strk.len-Movelen,0);
        if(strk.len>Movelen)
        {
            FeatPts* aptr = strk.getPtr(startidx);
            PntT xa=aptr->x,ya=aptr->y;
            double displc=sqrt((xb-xa)*(xb-xa) + (yb-ya)*(yb-ya));
            if((strk.len -startidx)*MoveFactor>displc)
            {
                isTrkValid = false;
            }
        }
    }
    return isTrkValid;
}

void CrowdTracker::PointTracking()
{
    debuggingFile<<"tracker"<<std::endl;
    tracker->sparse(gpuPreGray, gpuGray, gpuPrePts, gpuNextPts, gpuStatus);
}
int CrowdTracker::updateAframe(unsigned char* framedata, int fidx)
{
    std::clock_t start=std::clock();
    curStatus=FINE;


    frameidx=fidx;
    debuggingFile<<"frameidx:"<<frameidx<<std::endl;
    gpuGray.copyTo(gpuPreGray);

    Mat curframe(frame_height,frame_width,CV_8UC3,framedata);
    rgbMat.upload(curframe);

    //if(fidx!=16)
        gpu::cvtColor(rgbMat,gpuGray,CV_RGB2GRAY);

    PointTracking();
    debuggingFile<<"here"<<std::endl;
    findPoints();
    filterTrackGPU();
    /** Grouping  **/
    if(groupOnFlag)
    {
        pointCorelate();

        nbCount->SyncD2H();
        cudaMemcpy(bfsearchitems.data(),tracksGPU->lenVec,nFeatures*sizeof(int),cudaMemcpyDeviceToHost);
        for(int i=0;i<nFeatures;i++)
        {
            bfsearchitems[i]=i*(bfsearchitems[i]>0);
        }
        prelabel->copyFrom(label);
        pregroupN = groupN;
        //bfsearch();
        newBFS();
        if(groupN>0)
        {
            updateGroupsTracks();
            if(groupN>maxgroupN)maxgroupN=groupN;

            unsigned char* h_clrvec=clrvec->cpu_ptr();
            for(int i=0;i<maxgroupN;i++)
            {
                HSVtoRGB(h_clrvec+i*3,h_clrvec+i*3+1,h_clrvec+i*3+2,i/(maxgroupN+0.01)*360,1,1);
            }
            makeGroups();
            clrvec->SyncH2D();
            matchGroups();
        }
    }

    tracksGPU->Sync();
    PersExcludeMask();
    Render(framedata);
    float duration = ( std::clock() - start ) / (float) CLOCKS_PER_SEC;
    debuggingFile<<"Total Time"<<duration<<std::endl;
    return 1;
}
void CrowdTracker::updateGroupsTracks()
{
    // Update Tracking Group
	
    std::clock_t start=std::clock();
    for(int i=0;i<groupsTrk->numGroup;i++)
    {
		int count = (*rankCountOld)[i];
		if (count == 0)
		{
			//lost 
			groupsTrk->lost(i);
		}
		if (count == 1)
		{
			int newIdx = (*rankingOld)[i*nFeatures];
			float scrOld = (*scoreOld)[i*nFeatures];
			int countnew = (*rankCountNew)[newIdx];
			if (countnew == 1)
			{
				//update
				//std::cout << "updated" << std::endl;
				groupsTrk->getPtr(i)->updateFrom(groups, newIdx);
			}
			else if (countnew > 1)
			{
				// wait see lost
				// can update with klt displacement
				KLT_updates_Group(i);
			}
		}
		if (count >1)
		{
			/*
			int* ranking = rankingOld->cpu_ptr()+i*nFeatures;
			float* scr = scoreOld->cpu_ptr() + i*nFeatures;
			int maxi = 0;
			float maxscr = scr[maxi];
			for (int j = 0; j < count; j++)
			{
				if (scr[j]>maxscr)
				{
					maxi = j;
					maxscr = scr[j];
				}
			}
			int updatei = ranking[maxi];
			groupsTrk->getPtr(i)->updateFrom(groups, updatei);
			*/
			//lost 
			// wait to match New
			debuggingFile << "lost:" << i << std::endl;
			groupsTrk->clear(i);
			
			for (int newi = 0; newi < count; newi++)
			{
				int newIdx = (*rankingOld)[i*nFeatures + newi];
				//std::cout << newIdx << std::endl;
				groupsTrk->addGroup(groups, newIdx);
			}
        }
    }
    // Adding New Group

    for(int i=1;i<groups->numGroups;i++)
	{
		int count = (*rankCountNew)[i];
		if (count==0)
        {
            int addidx = groupsTrk->addGroup(groups,i);
            //debuggingFile<<"adding:"<<i<<"added:"<<addidx<<std::endl;
            BBox* bBox=groupsTrk->getCurBBox(addidx);
            //debuggingFile<<"bBox:"<<bBox->left<<","<<bBox->right<<","<<bBox->top<<","<<bBox->bottom<<std::endl;
        }
		/*
		else if (count == 1)
		{
			int oldIdx = (*rankingNew)[i*nFeatures];
			float scrNew = (*scoreNew)[i*nFeatures];
			int countold = (*rankCountOld)[oldIdx*nFeatures];
			if(countold>1)
			{
				//add new 
				//or match old
				int addidx = groupsTrk->addGroup(groups, i);
			}
		}
		*/
    }
    float duration = ( std::clock() - start ) / (float) CLOCKS_PER_SEC;
    debuggingFile<<"Total ReGroup Time"<<duration<<std::endl;
}
void CrowdTracker::newBFS()
{
    label->toZeroH();
    int* h_label=label->cpu_ptr();
    ptNumInGroup->toZeroH();
    int* h_gcount=ptNumInGroup->cpu_ptr();
    int* h_neighbor=nbCount->cpu_ptr();
    memset(idxmap,0,nFeatures*sizeof(int));
    memset(tmpn,0,nFeatures*sizeof(int));

    int idx=0;
    int total=0;
    bool unset=true;
    int gcount=0;
    for(int i=0;i<nFeatures;i++)
    {
        if(unset&&bfsearchitems[i]>0){idx=i;unset=false;}
        total+=(bfsearchitems[i]>0);
        if(!unset)
        {
            tmpn[i]+=(h_neighbor[idx*nFeatures+i]>0);
        }
    }
    bfsearchitems[idx]=0;
    total--;
    debuggingFile<<"total BFS:"<<total<<std::endl;
    curK=1;
    groupN=0;
    gcount++;
    while(total>0)
    {
        int ii=0;
        for(idx=0;idx<nFeatures;idx++)
        {
            if(!ii)ii=idx*(bfsearchitems[idx]>0);
            if(bfsearchitems[idx]&&tmpn[idx])
            {
                int nc=0,nnc=0;
                float nscore=0;
                for(int i=0;i<nFeatures;i++)
                {
                    if(h_neighbor[idx*nFeatures+i])
                    {
                        nc+=(h_neighbor[idx*nFeatures+i]>0);
                        nnc+=(tmpn[i]>0);
                    }
                }
                if(nnc>nc*0.1+1)
                {
                    gcount++;
                    h_label[idx]=curK;
                    for(int i=0;i<nFeatures;i++)
                    {
                        tmpn[i]+=(h_neighbor[idx*nFeatures+i]>0);
                    }
                    bfsearchitems[idx]=0;
                    total--;
                    if(ii==idx)ii=0;
                }
            }
        }
        if(gcount>0)
        {
            h_gcount[curK]+=gcount;
            gcount=0;
        }
        else if(total>0)
        {
            if(h_gcount[curK]>minGSize)
            {
                groupN++;
                idxmap[curK]=groupN;
            }
            curK++;
            gcount=0;
            memset(tmpn,0,nFeatures*sizeof(int));
            idx=ii;
            gcount++;
            h_label[idx]=curK;
            for(int i=0;i<nFeatures;i++)
            {
                tmpn[i]+=(h_neighbor[idx*nFeatures+i]>0);
            }
            bfsearchitems[idx]=0;
            total--;
        }
    }
    for(int i=0;i<nFeatures;i++)
    {
        h_label[i]=idxmap[h_label[i]];
    }
}
void CrowdTracker::bfsearch()
{
    int pos=0;
    bool isempty=false;
    int gcount=0;
    curK=1;
    groupN=0;
    int* h_neighbor=nbCount->cpu_ptr();
    label->toZeroH();
    ptNumInGroup->toZeroH();
    int* h_label=label->cpu_ptr();
    int* h_gcount=ptNumInGroup->cpu_ptr();
    memset(idxmap,0,nFeatures*sizeof(int));
    memset(tmpn,0,nFeatures*sizeof(int));
    memset(h_label,0,nFeatures*sizeof(int));
    memset(h_gcount,0,nFeatures*sizeof(int));
    int idx = bfsearchitems[pos];
    h_label[idx]=curK;
    for(int i=0;i<nFeatures;i++)
    {
        tmpn[i]=(h_neighbor[idx*nFeatures+i]>TIMESPAN);
    }
    bfsearchitems[pos]=0;
    gcount++;
    while (!isempty) {
        isempty=true;
        int ii=0;
        for(pos=0;pos<bfsearchitems.size();pos++)
        {
            idx=bfsearchitems[pos];
            if(idx)
            {
                if(ii==0)ii=pos;
                isempty=false;
                if(tmpn[idx])
                {
                    int nc=0,nnc=0;
                    float nscore=0;
                    for(int i=0;i<nFeatures;i++)
                    {
                        if(h_neighbor[idx*nFeatures+i])
                        {
//                            if(trackBuff[idx].len>100&&trackBuff[i].len>100)
//                            {
//                                nscore=(h_neighbor[idx*nFeatures+i])/float(trackBuff[idx].len+trackBuff[i].len)*2;
//                                if(nscore>0.9)
//                                {
//                                    nc+=(h_neighbor[idx*nFeatures+i]>0);
//                                    nnc+=(tmpn[i]>0);
//                                }
//                            }
//                            else
//                            {
                            nc+=(h_neighbor[idx*nFeatures+i]>TIMESPAN);
                            nnc+=(tmpn[i]>0);
//                            }
                        }
                    }
                    if(nnc>nc*0.1+1)
                    {
                        gcount++;
                        h_label[idx]=curK;
                        for(int i=0;i<nFeatures;i++)
                        {
                            tmpn[i]+=(h_neighbor[idx*nFeatures+i]>TIMESPAN);
                        }
                        bfsearchitems[pos]=0;
                        if(ii==pos)ii=0;
                    }
                }
            }
        }
        if(gcount>0)
        {
            h_gcount[curK]+=gcount;
            gcount=0;
        }
        else if(!isempty)
        {
            if(h_gcount[curK]>minGSize)
            {
                groupN++;
                idxmap[curK]=groupN;
            }
            curK++;
            gcount=0;
            memset(tmpn,0,nFeatures*sizeof(int));
            pos=ii;
            idx=bfsearchitems[pos];
            gcount++;
            h_label[idx]=curK;
            for(int i=0;i<nFeatures;i++)
            {
                tmpn[i]+=(h_neighbor[idx*nFeatures+i]);
            }
            bfsearchitems[pos]=0;
        }
    }
    for(int i=0;i<nFeatures;i++)
    {
        h_label[i]=idxmap[h_label[i]];
    }

}
