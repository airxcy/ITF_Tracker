#include "itf/trackers/trackers.h"
#include "itf/trackers/gpucommon.hpp"
#include "itf/trackers/utils.h"
#include "thrust/sort.h"
#include <iostream>
#include <stdio.h>
#include <numeric>

#include <cudnn.h>

using namespace cv;
using namespace cv::gpu;
__device__ int d_framewidth[1],d_frameheight[1];
__device__ int lockOld[NUMTHREAD],lockNew[NUMTHREAD];

void __global__ clearLockKernel()
{
    lockOld[threadIdx.x]=0;
    lockNew[threadIdx.x]=0;
}
void clearLock()
{
    clearLockKernel<<<1,NUMTHREAD>>>();
}
void setHW(int w,int h)
{
    cudaMemcpyToSymbol(d_framewidth,&w,sizeof(int));
    cudaMemcpyToSymbol(d_frameheight,&h,sizeof(int));
}

__global__ void applyPersToMask(unsigned char* d_mask,float* d_curvec,float* d_persMap)
{
    int pidx=blockIdx.x;
    float px=d_curvec[pidx*2],py=d_curvec[pidx*2+1];
    int blocksize = blockDim.x;
    int w=d_framewidth[0],h=d_frameheight[0];
    int localx = threadIdx.x,localy=threadIdx.y;
    int pxint = px+0.5,pyint = py+0.5;
    float persval =d_persMap[pyint*w+pxint];
    float range=Pers2Range(persval);
    int offset=range+0.5;
    int yoffset = localy-blocksize/2;
    int xoffset = localx-blocksize/2;
    if(abs(yoffset)<range&&abs(xoffset)<range)
    {
        int globalx=xoffset+pxint,globaly=yoffset+pyint;
        d_mask[globaly*d_framewidth[0]+globalx]=0;
    }
}
__global__ void  addNewPts(FeatPts* cur_ptr,int* lenVec,float2* new_ptr,float2* nextPtrs)
{
    int idx=threadIdx.x;
    int dim=blockDim.x;
    __shared__ int counter[1];
    counter[0]=0;
    __syncthreads();

    if(lenVec[idx]<=0)
    {
        int posidx = atomicAdd(counter,1);
        //printf("(%d,%.2f,%.2f)",posidx,new_ptr[posidx].x,new_ptr[posidx].y);
        if(posidx<dim)
        {
            cur_ptr[idx].x=new_ptr[posidx].x;
            cur_ptr[idx].y=new_ptr[posidx].y;
            lenVec[idx]+=1;
        }
    }
    nextPtrs[idx].x=cur_ptr[idx].x;
    nextPtrs[idx].y=cur_ptr[idx].y;
}
__global__ void applyPointPersMask(unsigned char* d_mask,FeatPts* cur_ptr,int* lenVec,float* d_persMap)
{
    int pidx=blockIdx.x;
    int len=lenVec[pidx];
    if(len>0)
    {
        float px=cur_ptr[pidx].x,py=cur_ptr[pidx].y;
        int blocksize = blockDim.x;
        int w=d_framewidth[0],h=d_frameheight[0];
        int localx = threadIdx.x,localy=threadIdx.y;
        int pxint = px+0.5,pyint = py+0.5;
        float persval =d_persMap[pyint*w+pxint];
        float range=Pers2Range(persval);
        int offset=range+0.5;
        int yoffset = localy-blocksize/2;
        int xoffset = localx-blocksize/2;
        if(abs(yoffset)<range&&abs(xoffset)<range)
        {
            int globalx=xoffset+pxint,globaly=yoffset+pyint;
            d_mask[globaly*w+globalx]=0;
        }
    }
}
void CrowdTracker::PersExcludeMask()
{
    addNewPts<<<1,nFeatures,0,cornerStream>>>(tracksGPU->curTrkptr,tracksGPU->lenVec,corners->gpu_ptr(),(float2* )gpuPrePts.data);
    debuggingFile<<"applyPersToMask:"<<std::endl;
    cudaMemcpyAsync(mask->gpu_ptr(),roimask->gpu_ptr(),frame_height*frame_width*sizeof(unsigned char),cudaMemcpyDeviceToDevice,cornerStream);
    dim3 block(32, 32,1);
    applyPointPersMask<<<nFeatures,block,0,cornerStream>>>(mask->gpu_ptr(),tracksGPU->curTrkptr,tracksGPU->lenVec,persMap->gpu_ptr());
    corners->SyncD2HStream(cornerStream);
}

__global__ void applySegMask(unsigned char* d_mask,unsigned char* d_segmask,unsigned char* d_segNeg)
{
    int offset=blockIdx.x*blockDim.x+threadIdx.x;
    int w=d_framewidth[0],h=d_frameheight[0];
    int totallen =w*h;
    int y=offset/w;
    int x=offset%w;
    if(offset<totallen&&!d_segNeg[offset]&&!d_segmask[offset])
    {
        d_mask[offset]=0;
    }
}

__global__ void renderFrame(unsigned char* d_renderMask,unsigned char* d_frameptr,int totallen)
{
    int offset=(blockIdx.x*blockDim.x+threadIdx.x)*3;
    int offsetp=blockIdx.x*blockDim.x+threadIdx.x;
    int maskval = d_renderMask[offset];
    if(offsetp<totallen)
    {
        d_frameptr[offset]=d_frameptr[offset]*0.5+d_renderMask[offset]*0.5;
        d_frameptr[offset+1]=d_frameptr[offset+1]*0.5+d_renderMask[offset+1]*0.5;
        d_frameptr[offset+2]=d_frameptr[offset+2]*0.5+d_renderMask[offset+2]*0.5;
//        d_frameptr[offset]=d_renderMask[offset]*0.5;
//        d_frameptr[offset+1]=d_renderMask[offset+1]*0.5;
//        d_frameptr[offset+2]=d_renderMask[offset+2]*0.5;
    }
}

__global__ void renderGroup(unsigned char* d_renderMask,FeatPts* cur_ptr,int* lenVec,unsigned char* d_clrvec,float* d_persMap,int* d_neighbor)
{
    int pidx=blockIdx.x;
    int len=lenVec[pidx];
    if(len>0)
    {
        float px=cur_ptr[pidx].x,py=cur_ptr[pidx].y;
        int blocksize = blockDim.x;
        int w=d_framewidth[0],h=d_frameheight[0];
        int localx = threadIdx.x,localy=threadIdx.y;
        int pxint = px+0.5,pyint = py+0.5;
        float persval =d_persMap[pyint*w+pxint];
        float range=Pers2Range(persval);
        int offset=range+0.5;
        int yoffset = localy-blocksize/2;
        int xoffset = localx-blocksize/2;
        if(abs(yoffset)<range&&abs(xoffset)<range)
        {
            int globalx=xoffset+pxint,globaly=yoffset+pyint;
            int globalOffset=(globaly*w+globalx)*3;
            d_renderMask[globalOffset]=255;
            d_renderMask[globalOffset+1]=0;
            d_renderMask[globalOffset+2]=0;
        }
    }
}

void CrowdTracker::Render(unsigned char* framedata)
{
    /*
    dim3 blockSize(32,32,1);
    tracksGPU->lenData->SyncD2H();
    int val=0;
    for(int i=0;i<nFeatures;i++)
    {
        val+=(tracksGPU->lenData->cpu_ptr()[i]>0);
    }
    debuggingFile<<"Render:"<<val<<std::endl;
    renderGroup<<<nFeatures,blockSize>>>(renderMask->gpu_ptr(),tracksGPU->curTrkptr,tracksGPU->lenVec,clrvec->gpu_ptr(),persMap->gpu_ptr(),nbCount->gpu_ptr());
    */
    int nblocks = (frame_height*frame_width)/nFeatures;
    //renderMask->toZeroD();
    renderFrame<<<nblocks,nFeatures>>>(renderMask->gpu_ptr(),rgbMat.data,frame_width*frame_height);
    renderMask->toZeroD();
    cudaMemcpy(framedata,rgbMat.data,frame_height*frame_width*3*sizeof(unsigned char),cudaMemcpyDeviceToHost);
}

__global__ void searchNeighbor(TracksInfo trkinfo,
                               int* d_neighbor,float* d_cosine,float* d_velo,float* d_distmat,
                               float * d_persMap, int nFeatures)
{
    int c = threadIdx.x, r = blockIdx.x;
    int clen = trkinfo.lenVec[c],rlen = trkinfo.lenVec[r];
    FeatPts* cur_ptr=trkinfo.curTrkptr;
    if(clen>minTrkLen&&rlen>minTrkLen&&r<c)
    {
//        int offset = (tailidx+bufflen-minTrkLen)%bufflen;
//        FeatPts* pre_ptr=data_ptr+NQue*offset;
//        FeatPts* pre_ptr=trkinfo.preTrkptr;//trkinfo.getVec_(trkinfo.trkDataPtr,minTrkLen-1);
//        float cx0=pre_ptr[c].x,cy0=pre_ptr[c].y;
//        float rx0=pre_ptr[r].x,ry0=pre_ptr[r].y;
        float cx1=cur_ptr[c].x,cy1=cur_ptr[c].y;
        float rx1=cur_ptr[r].x,ry1=cur_ptr[r].y;
        float dx = abs(rx1 - cx1), dy = abs(ry1 - cy1);
        float dist = sqrt(dx*dx+dy*dy);
        int  ymid = (ry1 + cy1) / 2.0+0.5,xmid = (rx1 + cx1) / 2.0+0.5;
        float persval=0;
        int ymin=min(ry1,cy1),xmin=min(rx1,cx1);
        persval =d_persMap[ymin*d_framewidth[0]+xmin];
        float hrange=persval,wrange=persval;
        if(hrange<2)hrange=2;
        if(wrange<2)wrange=2;
        float distdecay=0.05,cosdecay=0.1,velodecay=0.05;
        /*
        float vx0 = rx1 - rx0, vx1 = cx1 - cx0, vy0 = ry1 - ry0, vy1 = cy1 - cy0;
        float norm0 = sqrt(vx0*vx0 + vy0*vy0), norm1 = sqrt(vx1*vx1 + vy1*vy1);
        float veloCo = abs(norm0-norm1)/(norm0+norm1);
        float cosine = (vx0*vx1 + vy0*vy1) / norm0 / norm1;
        */
        float vrx = trkinfo.curVeloPtr[r].x, vry = trkinfo.curVeloPtr[r].y
                , vcx = trkinfo.curVeloPtr[c].x, vcy = trkinfo.curVeloPtr[c].y;
        float normr=trkinfo.curSpdPtr[r],normc=trkinfo.curSpdPtr[c];
        float veloCo = abs(normr-normc)/(normr+normc);
        float cosine = (vrx*vcx + vry*vcy) / normr / normc;
        dist = wrange*1.5/(dist+0.01);
        dist=2*dist/(1+abs(dist))-1;
        //dist=-((dist > wrange) - (dist < wrange));
        d_distmat[r*nFeatures+c]=dist+d_distmat[r*nFeatures+c]*(1-distdecay);
        d_distmat[c*nFeatures+r]=dist+d_distmat[c*nFeatures+r]*(1-distdecay);
        d_cosine[r*nFeatures+c]=cosine+d_cosine[r*nFeatures+c]*(1-cosdecay);
        d_cosine[c*nFeatures+r]=cosine+d_cosine[c*nFeatures+r]*(1-cosdecay);
        d_velo[r*nFeatures+c]=veloCo+d_velo[r*nFeatures+c]*(1-velodecay);
        d_velo[c*nFeatures+r]=veloCo+d_velo[c*nFeatures+r]*(1-velodecay);
        if(d_distmat[r*nFeatures+c]>5&&d_cosine[r*nFeatures+c]>1)//&&d_velo[r*nFeatures+c]<(14*velodecay)*0.9)
        {
            d_neighbor[r*nFeatures+c]+=1;
            d_neighbor[c*nFeatures+r]+=1;
        }
        else
        {
            d_neighbor[r*nFeatures+c]/=2.0;
            d_neighbor[c*nFeatures+r]/=2.0;
        }

    }
}

__global__ void clearLostStats(int* lenVec,int* d_neighbor,float* d_cosine,float* d_velo,float* d_distmat,int nFeatures)
{
    int c=threadIdx.x,r=blockIdx.x;
    if(r<nFeatures,c<nFeatures)
    {
        bool flag1=(lenVec[c]>0),flag2=(lenVec[r]>0);
        bool flag=flag1&&flag2;
        if(!flag)
        {

            d_neighbor[r*nFeatures+c]=0;
            d_neighbor[c*nFeatures+r]=0;
            d_cosine[r*nFeatures+c]=0;
            d_cosine[c*nFeatures+r]=0;
            d_velo[r*nFeatures+c]=0;
            d_velo[c*nFeatures+r]=0;
            d_distmat[r*nFeatures+c]=0;
            d_distmat[c*nFeatures+r]=0;
        }
    }
}
__global__ void filterTracks(TracksInfo trkinfo,uchar* status,float2* update_ptr,float* d_persMap)
{
    int idx=threadIdx.x;
    int len = trkinfo.lenVec[idx];
    bool flag = status[idx];
    float x=update_ptr[idx].x,y=update_ptr[idx].y;
    int frame_width=d_framewidth[0],frame_heigh=d_frameheight[0];
    trkinfo.nextTrkptr[idx].x=x;
    trkinfo.nextTrkptr[idx].y=y;
    float curx=trkinfo.curTrkptr[idx].x,cury=trkinfo.curTrkptr[idx].y;
    float dx = x-curx,dy = y-cury;
    float dist = sqrt(dx*dx+dy*dy);
    float cumDist=dist+trkinfo.curDistPtr[idx];
    trkinfo.nextDistPtr[idx]=cumDist;
    if(flag&&len>0)
    {

        int xb=x+0.5,yb=y+0.5;
        float persval=d_persMap[yb*frame_width+xb];
//        int prex=trkinfo.curTrkptr[idx].x+0.5, prey=trkinfo.curTrkptr[idx].y+0.5;
//        int trkdist=abs(prex-xb)+abs(prey-yb);
        float trkdist=abs(dx)+abs(dy);
        if(trkdist>persval)
        {
            flag=false;
        }
        //printf("%d,%.2f,%d|",trkdist,persval,flag);
        int Movelen=150/sqrt(persval);
        //Movelen is the main factor wrt perspective
//        printf("%d\n",Movelen);
        if(flag&&Movelen<len)
        {
//            int offset = (tailidx+bufflen-Movelen)%bufflen;
//            FeatPts* dataptr = next_ptr-tailidx*NQue;
//            FeatPts* aptr = dataptr+offset*NQue;
//            float xa=aptr[idx].x,ya=aptr[idx].y;
            FeatPts* ptr = trkinfo.getPtr_(trkinfo.trkDataPtr,idx,Movelen);
            float xa=ptr->x,ya=ptr->y;
            float displc=sqrt((x-xa)*(x-xa) + (y-ya)*(y-ya));
            float curveDist=cumDist-*(trkinfo.getPtr_(trkinfo.distDataPtr,idx,Movelen));
            //if(persval*0.1>displc)
            if(curveDist<3&&displc<3)
            {
                flag=false;
            }
        }
    }
    int newlen =flag*(len+(len<trkinfo.buffLen));
    trkinfo.lenVec[idx]=newlen;
    if(newlen>minTrkLen)
    {
        FeatPts* pre_ptr=trkinfo.preTrkptr;
        float prex=pre_ptr[idx].x,prey=pre_ptr[idx].y;
        float vx = (x-prex)/minTrkLen,vy = (y-prey)/minTrkLen;
        float spd = sqrt(vx*vx+vy*vy);
        trkinfo.nextSpdPtr[idx]=spd;
        trkinfo.nextVeloPtr[idx].x=vx,trkinfo.nextVeloPtr[idx].y=vy;
    }
}
void CrowdTracker::filterTrackGPU()
{
    trkInfo=tracksGPU->getInfoGPU();
    trkInfo.preTrkptr=trkInfo.getVec_(trkInfo.trkDataPtr,minTrkLen-1);
    /*
    filterTracks<<<1,nFeatures>>>(tracksGPU->cur_gpu_ptr,tracksGPU->next_gpu_ptr,(float2 *)gpuNextPts.data,
                                  tracksGPU->lendata->gpu_ptr(),gpuStatus.data,persMap->gpu_ptr(),
                                  tracksGPU->NQue,tracksGPU->buff_len,tracksGPU->tailidx);
    */
    filterTracks<<<1,nFeatures>>>(trkInfo,gpuStatus.data,(float2 *)gpuNextPts.data,persMap->gpu_ptr());
    tracksGPU->increPtr();
    trkInfo=tracksGPU->getInfoGPU();
    trkInfo.preTrkptr=trkInfo.getVec_(trkInfo.trkDataPtr,minTrkLen);
}



__global__ void  makeGroupKernel(int* labelidx,Groups groups,TracksInfo trkinfo)
{
    int pidx=threadIdx.x;
    int gidx=blockIdx.x;
    int* idx_ptr=groups.trkPtsIdxPtr;
    int* count_ptr=groups.ptsNumPtr;
    int nFeatures=groups.trkPtsNum;
    int* cur_gptr = idx_ptr+gidx*nFeatures;
    FeatPts* cur_Trkptr=trkinfo.curTrkptr+pidx;
    float2* cur_veloPtr=trkinfo.curVeloPtr+pidx;
    float2* trkPtsPtr=groups.trkPtsPtr+gidx*nFeatures;
    __shared__ int counter;
    __shared__ float com[2],velo[2];
    __shared__ int left,right,top,bot;
    left=9999,right=0,top=9999,bot=0;
    com[0]=0,com[1]=0;
    velo[0]=0,velo[1]=0;
    counter=0;
    __syncthreads();
    if(labelidx[pidx]==gidx)
    {
        float x=cur_Trkptr->x,y=cur_Trkptr->y;
        int px=x+0.5,py=y+0.5;
        int pos=atomicAdd(&counter,1);
        cur_gptr[pos]=pidx;
        trkPtsPtr[pos].x=x;
        trkPtsPtr[pos].y=y;
        atomicAdd(com,x);
        atomicAdd((com+1),y);
        atomicAdd(velo,cur_veloPtr->x);
        atomicAdd((velo+1),cur_veloPtr->y);
        atomicMin(&left,px);
        atomicMin(&top,py);
        atomicMax(&right,px);
        atomicMax(&bot,py);
    }
    __syncthreads();
    if(threadIdx.x==0)
    {
        count_ptr[gidx]=counter;
        groups.comPtr[gidx].x=com[0]/counter;
        groups.comPtr[gidx].y=com[1]/counter;
        groups.veloPtr[gidx].x=velo[0]/counter;
        groups.veloPtr[gidx].y=velo[1]/counter;
        groups.bBoxPtr[gidx].left=left;
        groups.bBoxPtr[gidx].top=top;
        groups.bBoxPtr[gidx].right=right;
        groups.bBoxPtr[gidx].bottom=bot;
        float area=(bot-top)*(right-left);
        groups.areaPtr[gidx]=area;
        //printf("%d,%f/",gidx,area);
    }
}
__global__ void  groupProp(int* labelidx,Groups groups,TracksInfo trkinfo)
{

}
__host__ __device__ __forceinline__ float cross_(const cvxPnt& O, const cvxPnt& A, const cvxPnt &B)
{
    return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
}

__global__ void genPolygonKernel(Groups groups)
{
    int gidx=blockIdx.x;
    int nFeatures=groups.trkPtsNum;
    int count=groups.ptsNumPtr[gidx];
    const int* countPtr=groups.ptsNumPtr+gidx;
    cvxPnt* H=(cvxPnt*)groups.polygonPtr+gidx*nFeatures;
    cvxPnt* P=(cvxPnt*)groups.trkPtsPtr+gidx*nFeatures;

    int n = count, k = 0;
    //thrust::sort(thrust::seq,P,P+count);
    // Build lower hull
    for (int i = 0; i < n; ++i) {
        while (k >= 2 && cross_(H[k-2], H[k-1], P[i]) <= 0) k--;
        H[k++]=P[i];
    }

    // Build upper hull
    for (int i = n-2, t = k+1; i >= 0; i--) {
        while (k >= t && cross_(H[k-2], H[k-1], P[i]) <= 0) k--;
        H[k++]=P[i];
    }
    groups.polyCountPtr[gidx]=k;
}

__global__ void  matchGroupKernel(GroupTrack groupTrk,Groups groups,int preIdx,int* overlap,int maxGroup,unsigned char* renderMask,unsigned char* clrvec)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int newIdx = blockIdx.z;
    if(newIdx)
    {

        BBox& preBBox =*groupTrk.getCur_(groupTrk.bBoxPtr),newBBox=*groups.getPtr_(groups.bBoxPtr,newIdx);
        int globalx=x+preBBox.left,globaly=y+preBBox.top;
        float2 velo=*groupTrk.getCur_(groupTrk.veloPtr);
        BBoxFloat curBBox={preBBox.left+velo.x,preBBox.top+velo.y,preBBox.right+velo.x,preBBox.bottom+velo.y};
        if(!(curBBox.left>newBBox.right||curBBox.right<newBBox.left||curBBox.top>newBBox.bottom||curBBox.bottom<curBBox.top))
        {
            if(ptInBox(globalx,globaly,preBBox)&&ptInBox(globalx,globaly,newBBox))
            {
                int * overlapCounter=overlap+preIdx*maxGroup+newIdx;
                if(globaly<d_frameheight[0]&&globalx<d_framewidth[0])
                {
                    int offset=(globaly*d_framewidth[0]+globalx)*3;
                    unsigned char curR = clrvec[preIdx*3],curG = clrvec[preIdx*3+1],curB = clrvec[preIdx*3+2];
                    unsigned char newR = clrvec[newIdx*3],newG = clrvec[newIdx*3+1],newB = clrvec[newIdx*3+2];
                    unsigned char r=(curR+newR)/2.0,g=(curG+newG)/2.0,b=(curB+newB)/2.0;
                    renderMask[offset]=renderMask[offset]*0.3+r*0.7;
                    renderMask[offset+1]=renderMask[offset+1]*0.3+g*0.7;
                    renderMask[offset+2]=renderMask[offset+2]*0.3+b*0.7;
                }
                atomicAdd(overlapCounter,1);
            }
        }
    }
}
__global__ void rankingKernel(int* overLap,int nFeatures
                              ,int* rankCountNew,int* rankCountOld
                              ,int* rankingNew,int* rankingOld
                              ,float* scoreNew,float* scoreOld
                              ,GroupTrack* groupsTrk,Groups groups
                              ,int* vacancy)
{
    int oldN = gridDim.x;
    int newN = blockDim.x;
    int oldIdx = blockIdx.x,newIdx = threadIdx.x;
    int* mutexOld=lockOld+oldIdx,* mutexNew=lockOld+oldIdx;
    //printf("%d,%d|",oldIdx,newIdx);
    if(vacancy[oldIdx]&&newIdx)
    {

        float overLapVal = overLap[oldIdx*nFeatures+newIdx];
        GroupTrack& oldTrk = groupsTrk[oldIdx];
        float areaOld = *(oldTrk.getCur_(oldTrk.areaPtr));
        float areaNew = groups.areaPtr[newIdx];
        float scrOld = overLapVal/areaOld;
        float scrNew = overLapVal/areaNew;
        float score = overLapVal/(areaOld+areaNew-overLapVal);
        //unsigned char& old_lock =lockOld[oldIdx],new_lock=lockNew[newIdx];
        int* counterNew =  rankCountNew+newIdx,*counterOld = rankCountOld+oldIdx;
        float* oldScr = scoreOld+oldIdx*nFeatures,*newScr = scoreNew+newIdx*nFeatures;
        int* rankOld = rankingOld+oldIdx*nFeatures,*rankNew = rankingNew+newIdx*nFeatures;
		float areaScore = abs(areaNew - areaOld) / (areaNew + areaOld + 1);
		if ((scrOld>0.5 || scrNew>0.5))
        {
			/* TODO Lock Free Ranking
            bool isSet = false;
            int counter=0;
            do
            {

                if (isSet = (atomicCAS(mutexOld, 0, 1) == 0))
                {
					int curCount = *counterOld;
					//if (curCount<3)
					//{
						// critical section goes here
					int insertPos=0;
					for(insertPos=0;insertPos<curCount;insertPos++)
					{
						if(scrOld<oldScr[insertPos])
						{
							break;
						}
					}
					float tempScr=oldScr[insertPos];
					int tempIdx=rankOld[insertPos];
					oldScr[insertPos]=scrOld;
					rankOld[insertPos]=newIdx;

					int i=0;
					for(i=insertPos+1;i<curCount+1;i++)
					{
						oldScr[i]=tempScr;
						rankOld[i]=tempIdx;
						tempScr=oldScr[i+1];
						tempIdx=rankOld[i+1];
					}
					printf("[%d,%d,%d]",oldIdx, curCount, insertPos);
					(*counterOld)++;
					//}
				}
				if (isSet)
				{
					mutexOld = 0;
				}
				counter++;
            }
            while (!isSet);
			if (counter >= 1000)printf("{%d,%d}", oldIdx, *counterOld);
			*/
            //printf("%f,%d,%d,%f,%f,%d|",overLapVal,oldIdx,newIdx,areaNew,scrNew);

            int pos = atomicAdd(counterOld,1);
            oldScr[pos]=scrOld;
            rankOld[pos]=newIdx;

			 pos = atomicAdd(counterNew, 1);
			newScr[pos] = scrNew;
			rankNew[pos] = oldIdx;
        }
    }
}
__global__ void KLT_update_Group_Kernel(GroupTrack* groupsTrk, int* vacancy, TracksInfo trkinfo, float2* updateVec, BBox* kltUpdateBoxVec)
{

	int pidx = threadIdx.x;
	int gidx = blockIdx.x;
	updateVec[gidx].x = 0;
	updateVec[gidx].y = 0;
	kltUpdateBoxVec[gidx].left = 0, kltUpdateBoxVec[gidx].top = 0, kltUpdateBoxVec[gidx].right = 0, kltUpdateBoxVec[gidx].bottom=0;
	GroupTrack group = groupsTrk[gidx];
	BBox bbox = *group.getCur_(group.bBoxPtr);
	__shared__ float updatex, updatey;
	__shared__ int counter;
	__shared__ int  left, top, right, bot;
	updatex = 0; updatey = 0;
	counter = 0;
	left = 9999, right = 0, top = 9999, bot = 0;
	__syncthreads();
	if (vacancy[gidx] && trkinfo.lenVec[pidx]>1)
	{
		FeatPts curPt = *trkinfo.getPtr_(trkinfo.trkDataPtr,pidx, 0);
		FeatPts prePt = *trkinfo.getPtr_(trkinfo.trkDataPtr, pidx, 1);
		if (prePt.x >= bbox.left&&prePt.x <= bbox.right&&prePt.y >= bbox.top&&prePt.y <= bbox.bottom)
		{
			float vx = curPt.x - prePt.x, vy = curPt.y - prePt.y;
			atomicAdd(&updatex,vx);
			atomicAdd(&updatey,vy);
			atomicAdd(&counter,1);
			atomicMin(&left, curPt.x);
			atomicMin(&top, curPt.y);
			atomicMax(&right, curPt.x);
			atomicMax(&bot, curPt.y);
		}
		
	}
	__syncthreads();
	if (pidx == 0)
	{
		if (counter > 0)
		{
			float ux = updatex / counter, uy = updatey / counter;
			updateVec[gidx].x = ux;
			updateVec[gidx].y = uy;
			kltUpdateBoxVec[gidx].left = (left + bbox.left + ux) / 2;
			kltUpdateBoxVec[gidx].top = (top + bbox.top + uy) / 2;
			kltUpdateBoxVec[gidx].right = (right + bbox.right + ux) / 2;
			kltUpdateBoxVec[gidx].bottom = (bot + bbox.bottom + uy) / 2;
		}
		else
		{
			updateVec[gidx].x = 0;
			updateVec[gidx].y = 0;
			kltUpdateBoxVec[gidx] = bbox;
		}
	}

}

void CrowdTracker::KLT_updates_Group(int idx)
{
	GroupTrack& gTrk = (*groupsTrk)[idx];
	float2 updates = (*kltUpdateVec)[idx];
	BBox oldbbox = *gTrk.getCurBBox();
	BBox newbbox;
	newbbox.left = oldbbox.left + updates.x,
	newbbox.right = oldbbox.right + updates.x,
	newbbox.top = oldbbox.top + updates.y,
	newbbox.bottom = oldbbox.bottom + updates.y;
	memcpy(gTrk.getNext_(gTrk.bBox->cpu_ptr()), &newbbox, sizeof(BBox));
	cudaMemcpy(gTrk.getNext_(gTrk.bBoxPtr), &newbbox, sizeof(BBox), cudaMemcpyHostToDevice);

	float2 newcom;
	newcom.x = gTrk.getCurCom()->x + updates.x;
	newcom.y = gTrk.getCurCom()->y + updates.y;
	memcpy(gTrk.getNext_(gTrk.com->cpu_ptr()), &newcom, sizeof(float2));
	cudaMemcpy(gTrk.getNext_(gTrk.comPtr), &newcom, sizeof(float2), cudaMemcpyHostToDevice);

	memcpy(gTrk.getNext_(gTrk.velo->cpu_ptr()), &updates, sizeof(float2));
	cudaMemcpy(gTrk.getNext_(gTrk.veloPtr), &updates, sizeof(float2), cudaMemcpyHostToDevice);

	memcpy(gTrk.getNext_(gTrk.area->cpu_ptr()), gTrk.getCur_(gTrk.area->cpu_ptr()), sizeof(float));
	cudaMemcpy(gTrk.getNext_(gTrk.areaPtr), gTrk.getCur_(gTrk.areaPtr), sizeof(float), cudaMemcpyDeviceToDevice);

	memcpy(gTrk.getNext_(gTrk.trkPtsIdx->cpu_ptr()), gTrk.getCur_(gTrk.trkPtsIdx->cpu_ptr()), sizeof(int)*nFeatures);
	cudaMemcpy(gTrk.getNext_(gTrk.trkPtsIdxPtr), gTrk.getCur_(gTrk.trkPtsIdxPtr), sizeof(int)*nFeatures, cudaMemcpyDeviceToDevice);

	memcpy(gTrk.getNext_(gTrk.ptsNum->cpu_ptr()), gTrk.getCur_(gTrk.ptsNum->cpu_ptr()), sizeof(int));
	cudaMemcpy(gTrk.getNext_(gTrk.ptsNumPtr), gTrk.getCur_(gTrk.ptsNumPtr), sizeof(int), cudaMemcpyDeviceToDevice);

	memcpy(gTrk.getNext_(gTrk.trkPts->cpu_ptr()), gTrk.getCur_(gTrk.trkPts->cpu_ptr()), sizeof(float2)*nFeatures);
	cudaMemcpy(gTrk.getNext_(gTrk.trkPtsPtr), gTrk.getCur_(gTrk.trkPtsPtr), sizeof(float2)*nFeatures, cudaMemcpyDeviceToDevice);

	gTrk.increPtr();
}

void CrowdTracker::matchGroups()
{
    /** compute Score**/

    int streamIdx=0;
//    for(int i=0;i<groups->numGroups;i++)
//    {
    debuggingFile<<"groupsTrk->numGroup:"<<groupsTrk->numGroup<<std::endl;
    overLap->toZeroD();
    for(int j=0;j<groupsTrk->numGroup;j++)
    {
        int vacancyVal = (*groupsTrk->vacancy)[j];
        if(vacancyVal )
        {
            //debuggingFile<<"old:"<<j<<std::endl;
            BBox* bBox=groupsTrk->getCurBBox(j);
            int maxW=bBox->right-bBox->left,maxH=bBox->bottom-bBox->top;
            if(maxW>0&&maxH>0)
            {
            //debuggingFile<<"bBox:"<<bBox->right<<","<<bBox->left<<","<<bBox->top<<","<<bBox->bottom<<std::endl;
            dim3 block(32, 32,1);
            dim3 grid(divUp(maxW,32),divUp(maxH,32),groups->numGroups);
            //debuggingFile<<"grid:"<<grid.x<<","<<grid.y<<std::endl;
            //debuggingFile<<"block:"<<block.x<<","<<block.y<<std::endl;
            //if(grid.x>0&&grid.y>0)
            matchGroupKernel<<<grid,block,0, streams[streamIdx]>>>((*groupsTrk)[j],*groups,j
                                                                    ,overLap->gpu_ptr(),nFeatures,renderMask->gpu_ptr(),clrvec->gpu_ptr());
//            else
//            {
//                curStatus=ERROR;
//            }
            }
            streamIdx=(streamIdx+1)%MAXSTREAM;
        }
    }
//    }
    // ranking
    std::clock_t start=std::clock();
    clearLock();
    rankCountNew->toZeroD();
    rankCountOld->toZeroD();
    overLap->SyncD2H();
	/*
    for(int i=0;i<groupsTrk->numGroup;i++)
    {
        for(int j=0;j<groups->numGroups;j++)
        {
            //debuggingFile<<"("<<(*overLap)[i*nFeatures+j]<<","<<groupsTrk->getCurArea(i)<<","<<(*(groups->area))[j]<<")";
            debuggingFile<<(*overLap)[i*nFeatures+j]<<",";
        }
        debuggingFile<<std::endl;
    }
	*/
    if(groupsTrk->numGroup>0&&groups->numGroups)
    {
        rankingKernel<<<groupsTrk->numGroup,groups->numGroups>>>(overLap->gpu_ptr(),nFeatures
                                                             ,rankCountNew->gpu_ptr(),rankCountOld->gpu_ptr()
                                                                ,rankingNew->gpu_ptr(),rankingOld->gpu_ptr()
                                                                ,scoreNew->gpu_ptr(),scoreOld->gpu_ptr()
                                                             ,groupsTrk->groupTracks->gpu_ptr(),*groups
                                                             ,groupsTrk->vacancy->gpu_ptr());
    }
	if (groupsTrk->numGroup > 0)
	{
		KLT_update_Group_Kernel << < groupsTrk->numGroup, nFeatures >> >(groupsTrk->groupTracks->gpu_ptr(), groupsTrk->vacancy->gpu_ptr(),
			trkInfo, kltUpdateVec->gpu_ptr(),kltUpdateBoxVec->gpu_ptr());
		kltUpdateVec->SyncD2H();
		kltUpdateBoxVec->SyncD2H();
	}
    rankCountOld->SyncD2H();
    rankCountNew->SyncD2H();
    rankingNew->SyncD2H();
    rankingOld->SyncD2H();
    scoreNew->SyncD2H();
    scoreOld->SyncD2H();
	
	debuggingFile << "New ranking" << std::endl;
    for(int i=0;i<groups->numGroups;i++)
    {
        int numChild = (*rankCountNew)[i];
        int* rankNew=rankingNew->cpu_ptr()+i*nFeatures;
        float* scrNew = scoreNew->cpu_ptr()+i*nFeatures;
        debuggingFile<<i<<">>>>";
        for(int j=0;j<numChild;j++)
        {
            debuggingFile<<"("<<rankNew[j]<<","<<scrNew[j]<<")";
        }
		debuggingFile << std::endl;
    }

	debuggingFile << "Old ranking" << std::endl;
	for (int i = 0; i<groupsTrk->numGroup; i++)
	{
		int numChild = (*rankCountOld)[i];
		int* rankOld = rankingOld->cpu_ptr() + i*nFeatures;
		float* scrOld = scoreOld->cpu_ptr() + i*nFeatures;
		debuggingFile << i << ">>>>";
		for (int j = 0; j<numChild; j++)
		{
			debuggingFile << "(" << rankOld[j] << "," << scrOld[j] << ")";
		}
		debuggingFile << std::endl;
	}
    float duration = ( std::clock() - start ) / (float) CLOCKS_PER_SEC;
    debuggingFile<<"match Time:"<<duration<<std::endl;
}
__global__ void applyFgMask(unsigned char* d_mask, unsigned char* d_fmask)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	int w = d_framewidth[0], h = d_frameheight[0];
	int offset = x*w + y;
	__shared__ unsigned char rect[32][32];
	rect[threadIdx.y][threadIdx.x] = 0;
	if (x < w&&y < h&&!d_fmask[offset])
	{
		d_mask[offset] = 0;
	}
}
void CrowdTracker::findPoints()
{
    debuggingFile<<"applySegMask"<<std::endl;
	(*mog2)(gpuGray, fgmat);
    if(applyseg)
    {
        int nblocks = (frame_height*frame_width)/nFeatures;
        applySegMask<<<nblocks,nFeatures>>>(mask->gpu_ptr(),segmask->gpu_ptr(),segNeg->gpu_ptr());
    }

	dim3 block(32, 32, 1);
	dim3 grid(divUp(frame_width, 32), divUp(frame_height, 32));
	applyFgMask <<< grid, block >>> (mask->gpu_ptr(),foreground->gpu_ptr());

    debuggingFile<<"detector"<<std::endl;
    (*detector)(gpuGray, gpuCorners, maskMat);
}

void CrowdTracker::pointCorelate()
{
    clearLostStats<<<nFeatures,nFeatures>>>(tracksGPU->lenData->gpu_ptr(),
                                                         nbCount->gpu_ptr(),cosCo->gpu_ptr(),veloCo->gpu_ptr(),distCo->gpu_ptr(),nFeatures);

    searchNeighbor <<<nFeatures, nFeatures>>>(trkInfo,nbCount->gpu_ptr(),cosCo->gpu_ptr(),veloCo->gpu_ptr(),distCo->gpu_ptr(),persMap->gpu_ptr(), nFeatures);

}
inline void buildPolygon(float2* pts,int& ptsCount,float2* polygon,int& polyCount)
{
    cvxPnt* P=(cvxPnt*)pts;
    cvxPnt* H=(cvxPnt*)polygon;
    int n = ptsCount, k = 0;
    // Sort points lexicographically
    std::sort(P,P+ptsCount);
    // Build lower hull
    for (int i = 0; i < n; ++i) {
        while (k >= 2 && cross_(H[k-2], H[k-1], P[i]) <= 0) k--;
        H[k++]=P[i];
    }

    // Build upper hull
    for (int i = n-2, t = k+1; i >= 0; i--) {
        while (k >= t && cross_(H[k-2], H[k-1], P[i]) <= 0) k--;
        H[k++]=P[i];
    }
    polyCount=k;
}

void CrowdTracker::makeGroups()
{

    label->SyncH2D();
    prelabel->SyncH2D();
    groups->numGroups=groupN;

    makeGroupKernel<<<groupN,nFeatures>>>(label->gpu_ptr(),*groups,trkInfo);
    groups->SyncD2H();
    for(int i=0;i<groups->numGroups;i++)
    {
        BBox& bbox= (*groups->bBox)[i];
        //debuggingFile<<i<<":"<<bbox.top<<","<<bbox.left<<","<<bbox.bottom<<","<<bbox.right<<std::endl;
    }
    /*
    for(int i=1;i<=groupN;i++)
    {
        buildPolygon(groups->trkPts->cpu_ptr()+i*nFeatures,groups->ptsNum->cpu_ptr()[i]
                    ,groups->polygon->cpu_ptr()+i*nFeatures,groups->polyCount->cpu_ptr()[i]);
    }
    groups->polySyncH2D();
    */
    //genPolygonKernel<<<nFeatures,1>>>(*groups);

}





