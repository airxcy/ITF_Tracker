#include "itf/trackers/buffgpu.h"
#include "itf/trackers/gpucommon.hpp"

#include <iostream>
template <typename ELEM_T>
MemBuff<ELEM_T>::MemBuff(int n, int c)
{
    count_size=n;
    channel=c;
    elem_size=sizeof(ELEM_T);
    byte_size=count_size*channel*elem_size;
    gpu_zalloc(d_data,byte_size,1);
    h_data =(ELEM_T *)zalloc(byte_size,1);
}
template <typename ELEM_T>
void MemBuff<ELEM_T>::SyncD2H()
{
    cudaMemcpy(h_data,d_data,byte_size,cudaMemcpyDeviceToHost);
}
template <typename ELEM_T>
void MemBuff<ELEM_T>::SyncD2HStream(cudaStream_t& stream)
{
    cudaMemcpyAsync(h_data,d_data,byte_size,cudaMemcpyDeviceToHost,stream);
}
template <typename ELEM_T>
void MemBuff<ELEM_T>::SyncH2D()
{
    cudaMemcpy(d_data,h_data,byte_size,cudaMemcpyHostToDevice);
}
template <typename ELEM_T>
void MemBuff<ELEM_T>::SyncH2DStream(cudaStream_t& stream)
{
    cudaMemcpyAsync(d_data,h_data,byte_size,cudaMemcpyHostToDevice,stream);
}
template <typename ELEM_T>
void MemBuff<ELEM_T>::updateGPU(ELEM_T* ptr)
{
    cudaMemcpy(d_data,ptr,byte_size,cudaMemcpyDeviceToDevice);
}
template <typename ELEM_T>
void MemBuff<ELEM_T>::updateCPU(ELEM_T* ptr)
{
    memcpy(h_data,ptr,byte_size);
}
template <typename ELEM_T>
void MemBuff<ELEM_T>::toZeroD()
{
    cudaMemset(d_data,0,byte_size);
}
template <typename ELEM_T>
void MemBuff<ELEM_T>::toZeroH()
{
    memset(h_data,0,byte_size);
}
template <typename ELEM_T>
void MemBuff<ELEM_T>::copyFrom(MemBuff<ELEM_T>* src)
{
    cudaMemcpy(d_data,src->gpu_ptr(),byte_size,cudaMemcpyDeviceToDevice);
    memcpy(h_data,src->cpu_ptr(),byte_size);
}
template class MemBuff<float>;
template class MemBuff<int>;
template class MemBuff<FeatPts>;
template class MemBuff<TrkPts>;
template class MemBuff<float2>;
template class MemBuff<unsigned char>;
template class MemBuff<GroupTrack>;

__global__ void updateVecKernel(FeatPts* next_ptr,FeatPts* gpuBUff_Ptr,int* lenVec,int* status,int bufflen)
{
    int idx=threadIdx.x;
    int len = lenVec[idx];
    bool flag= status[idx];
    next_ptr[idx]=gpuBUff_Ptr[idx];
    lenVec[idx]=flag*(len+(len<bufflen));
}

void Tracks::init(int n,int l)
{
    nQue=n,buffLen=l,tailidx=0;
    trkData = new MemBuff<FeatPts>(nQue*buffLen);
    trkDataPtr=trkData->gpu_ptr();
    lenData = new MemBuff<int>(nQue);
    lenVec=lenData->gpu_ptr();
    veloData = new MemBuff<float2>(nQue*buffLen);
    veloDataPtr=veloData->gpu_ptr();
    distData = new MemBuff<float>(nQue*buffLen);
    distDataPtr=distData->gpu_ptr();
    curCpuPtr=trkData->cpu_ptr()+tailidx*nQue;
    spdData = new MemBuff<float>(nQue*buffLen);
    spdDataPtr=spdData->gpu_ptr();
    TracksInfo::init(n,l);
}

void Tracks::Sync()
{
    trkData->SyncD2H();
    lenData->SyncD2H();
}
void Group::init(int maxn,Tracks* trks)
{
    tracks=trks;
    trkPtsNum=tracks->nQue;
    trkPtsIdx = new MemBuff<int>(maxn,trkPtsNum);
    trkPtsIdxPtr=trkPtsIdx->gpu_ptr();
    ptsNum = new MemBuff<int>(maxn);
    ptsNumPtr=ptsNum->gpu_ptr();
    trkPts = new MemBuff<float2>(maxn,trkPtsNum);
    trkPtsPtr=trkPts->gpu_ptr();
    com = new MemBuff<float2>(maxn);
    comPtr=com->gpu_ptr();
    velo = new MemBuff<float2>(maxn);
    veloPtr=velo->gpu_ptr();
    bBox = new MemBuff<BBox>(maxn);
    bBoxPtr = bBox->gpu_ptr();
    polygon= new MemBuff<float2>(maxn,trkPtsNum);
    polygonPtr=polygon->gpu_ptr();
    polyCount = new MemBuff<int>(maxn);
    polyCountPtr=polyCount->gpu_ptr();
    area= new MemBuff<float>(maxn);
    areaPtr = area->gpu_ptr();
}
void Group::SyncD2H()
{
    trkPtsIdx->SyncD2H();
    ptsNum->SyncD2H();
    trkPts->SyncD2H();
    com->SyncD2H();
    velo->SyncD2H();
    bBox->SyncD2H();
    polygon->SyncD2H();
    polyCount->SyncD2H();
    area->SyncD2H();
}
void Group::trkPtsSyncD2H()
{
    ptsNum->SyncD2H();
    trkPts->SyncD2H();
}
void Group::polySyncH2D()
{
    polygon->SyncH2D();
    polyCount->SyncH2D();
}
void Groups::init(int maxn,Tracks* trks)
{
    maxNumGroup=maxn;
    numGroups=0;
    Group::init(maxNumGroup,trks);
    /*
    tracks=trks;
    trkPtsNum=tracks->nQue;
    trkPtsIdx = new MemBuff<int>(trkPtsNum*maxNumGroup);
    trkPtsIdxPtr=trkPtsIdx->gpu_ptr();
    ptsNum = new MemBuff<int>(maxNumGroup);
    ptsNumPtr=ptsNum->gpu_ptr();
    trkPts = new MemBuff<float2>(trkPtsNum*maxNumGroup);
    trkPtsPtr=trkPts->gpu_ptr();
    com = new MemBuff<float2>(maxNumGroup);
    comPtr=com->gpu_ptr();
    velo = new MemBuff<float2>(maxNumGroup);
    veloPtr=velo->gpu_ptr();
    bBox = new MemBuff<int>(maxNumGroup,4);
    bBoxPtr = bBox->gpu_ptr();
    */
}
void GroupTrack::init(int maxn,Tracks* trks)
{
    buffLen=maxn;
    tailidx=0,len=0;
    Group::init(buffLen,trks);
}
void GroupTrack::clear()
{
	tailidx = 0, len = 0;
}
BBox* GroupTrack::getCurBBox()
{
    return getCur_(bBox->cpu_ptr());
}
float GroupTrack::getCurArea()
{
    return *(getCur_(area->cpu_ptr()));
}
float2* GroupTrack::getCurCom()
{
	return (getCur_(com->cpu_ptr()));
}

#define copyFeat(feat) \
    memcpy(getNext_(feat->cpu_ptr()),groups->feat->cpuAt(idx),feat->channel*feat->elem_size); \
    cudaMemcpy(getNext_(feat->gpu_ptr()),groups->feat->gpuAt(idx),feat->channel*feat->elem_size,cudaMemcpyDeviceToDevice);
void GroupTrack::updateFrom(Groups* groups,int idx)
{
//    memcpy(getNext_(trkPtsIdx->cpu_ptr()),groups->trkPtsIdx->cpuAt(idx),trkPtsIdx->channel*trkPtsIdx->elem_size);
//    cudaMemcpy(getNext_(trkPtsIdx->gpu_ptr()),groups->trkPtsIdx->gpuAt(idx),trkPtsIdx->channel*trkPtsIdx->elem_size);
    copyFeat(trkPtsIdx)
    copyFeat(ptsNum)
    copyFeat(trkPts)
    copyFeat(com)
    copyFeat(velo)
    copyFeat(bBox)
    copyFeat(polygon)
    copyFeat(polyCount)
    copyFeat(area)
    increPtr();
}




void GroupTracks::clear(int idx)
{
    if(idx<numGroup)
    {
        GroupTrack* cpuPtr = getPtr(idx);
        GroupTrack* gpuPtr = groupTracks->gpu_ptr()+idx;
        cpuPtr->clear();
        cudaMemcpy(gpuPtr,cpuPtr,sizeof(GroupTrack),cudaMemcpyHostToDevice);
        (*vacancy)[idx]=0;
    }
    vacancy->SyncH2D();
}
int GroupTracks::addGroup(Groups* groups,int newIdx)
{
    int addidx = numGroup;
    for(int i=0; i<numGroup; i++)
    {
        if( !(*vacancy)[i] )
        {
            addidx = i;
            break;
        }
    }
    if(addidx>=numGroup&&numGroup<maxNumGroup)
    {
        GroupTrack* nextGroup = getPtr(addidx);
        nextGroup->init(buffLen,groups->tracks);
        nextGroup->updateFrom(groups,newIdx);
        GroupTrack* gpuPtr = groupTracks->gpu_ptr()+numGroup;
        cudaMemcpy(gpuPtr,nextGroup,sizeof(GroupTrack),cudaMemcpyHostToDevice);
        (*vacancy)[addidx]=1;
        numGroup++;
    }
    else if(addidx<numGroup)
    {
        GroupTrack* cpuPtr = getPtr(addidx);
        GroupTrack* gpuPtr = groupTracks->gpu_ptr()+addidx;
        cpuPtr->clear();
        cpuPtr->updateFrom(groups,newIdx);
        cudaMemcpy(gpuPtr,cpuPtr,sizeof(GroupTrack),cudaMemcpyHostToDevice);
        (*vacancy)[addidx]=1;
    }
    vacancy->SyncH2D();
    return addidx;
}

void GroupTracks::lost(int idx)
{
	clear(idx);
}

BBox* GroupTracks::getCurBBox(int i)
{
    return getPtr(i)->getCurBBox();
}
float GroupTracks::getCurArea(int i)
{
    return getPtr(i)->getCurArea();
}
void GroupTracks::init(int maxn)
{
    numGroup=0,buffLen=10,maxNumGroup=maxn;
    groupTracks = new MemBuff<GroupTrack>(maxn);
    vacancy = new MemBuff<int>(maxn);
	lostvec = new MemBuff<int>(maxn);
}
