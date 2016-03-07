#include "Qts/viewqt.h"
#include "Qts/modelsqt.h"
#include "Qts/streamthread.h"

#include <iostream>
#include <stdio.h>

#include <QPainter>
#include <QBrush>
#include <QPixmap>
#include <cmath>
#include <QGraphicsSceneEvent>
#include <QMimeData>
#include <QByteArray>
#include <QFont>
char viewstrbuff[200];
QPointF points[100];

void DefaultScene::mousePressEvent ( QGraphicsSceneMouseEvent * event )
{
    emit clicked(event);
}
void DefaultScene::drawBackground(QPainter * painter, const QRectF & rect)
{
    QPen pen;
    QFont txtfont("Roman",40);
    txtfont.setBold(true);
    pen.setColor(QColor(255,255,255));
    pen.setCapStyle(Qt::RoundCap);
    pen.setJoinStyle(Qt::RoundJoin);
    pen.setWidth(10);
    painter->setPen(QColor(243,134,48,150));
    painter->setFont(txtfont);
    painter->drawText(rect, Qt::AlignCenter,"打开文件\nOpen File");
}
TrkScene::TrkScene(const QRectF & sceneRect, QObject * parent):QGraphicsScene(sceneRect, parent)
{
    streamThd=NULL;
}
TrkScene::TrkScene(qreal x, qreal y, qreal width, qreal height, QObject * parent):QGraphicsScene( x, y, width, height, parent)
{
    streamThd=NULL;
}
void TrkScene::drawBackground(QPainter * painter, const QRectF & rect)
{
    //debuggingFile<<streamThd->inited<<std::endl;
    if(streamThd!=NULL&&streamThd->inited)
    {
        updateFptr(streamThd->frameptr, streamThd->frameidx);
    }
    painter->setBrush(bgBrush);
    painter->drawRect(rect);
//    painter->setBrush(QColor(0,0,0,100));
//    painter->drawRect(rect);
    painter->setBrush(Qt::NoBrush);
    if(streamThd!=NULL&&streamThd->inited)
    {
        int* neighbor = streamThd->tracker->getNbCount();
        int nFeatures= streamThd->tracker->getNFeatures();
        int nSearch=streamThd->tracker->getNFeatures();
        float* distmat=streamThd->tracker->getDistCo();
        float* cosine=streamThd->tracker->getCosCo();
        float* velo=streamThd->tracker->getVeloCo();
        int* labelVec=streamThd->tracker->getLabel();
        unsigned char* clrvec=streamThd->tracker->getClrvec();
        float2* corners = streamThd->tracker->getCorners();

        linepen.setColor(QColor(255,200,200));
        linepen.setWidth(3);
        painter->setPen(linepen);
        painter->setFont(QFont("System",20,2));
        QString infoString="fps:"+QString::number(streamThd->fps)+"\n"
                +"use Seg:"+QString::number(streamThd->tracker->isSegOn())+"\n"
                +"Prop Idx:"+QString::number(showModeIdx)+"\n"
                +"thresh:"+QString::number(thresh)+"\n";
        painter->drawText(rect, Qt::AlignLeft|Qt::AlignTop,infoString);
        painter->setFont(QFont("System",20,2));
		float* persMap = streamThd->tracker->persMap->cpu_ptr();
        Tracks* tracks = streamThd->tracker->getTracks();
        Groups& groups = streamThd->tracker->getGroups();
        GroupTracks& groupsTrk = streamThd->tracker->getGroupsTrk();
        float x0,y0,x1,y1;
        linepen.setWidth(2);
        for(int i=0;i<tracks->nQue;i++)
        {
            int trklen = tracks->getLen(i);
            int label=labelVec[i];
            unsigned char r=255,g=255,b=255;
			/*
            if(label)
            {
                r=clrvec[label*3],g=clrvec[label*3+1],b=clrvec[label*3+2];
            }
			*/
            if(trklen>1)
            {
                FeatPts curPt = *tracks->getPtr(i);
				FeatPts prePt = *tracks->getPtr(i, 1);
                linepen.setColor(QColor(r, g, b));
                linepen.setWidth(1);
                painter->setPen(linepen);
				painter->drawLine(prePt.x, prePt.y,curPt.x,curPt.y);
            }
        }

		CrowdTracker& tracker = *(streamThd->tracker);
        for(int i =0;i<groupsTrk.numGroup;i++)
        {
            if((*groupsTrk.vacancy)[i])
            {
				GroupTrack& gtrk=groupsTrk[i];
				BBox* bbox = gtrk.getCurBBox();
				linepen.setWidth(0);
				linepen.setColor(QColor(255, 255, 255));
				painter->setPen(linepen);
                painter->drawRect(bbox->left,bbox->top,bbox->right-bbox->left,bbox->bottom-bbox->top);
				linepen.setStyle(Qt::SolidLine);
				linepen.setColor(QColor(0,0,255));
				linepen.setWidth(2);
				painter->setPen(linepen);
				float2 trkcom = *gtrk.getCurCom();
				QString printstr = QString::number(i) + "," + QString::number(gtrk.len)+"(";
				float areaOld = gtrk.getCurArea();
				for (int j = 0; j < (*tracker.rankCountOld)[i]; j++)
				{
					int newIdx = (*tracker.rankingOld)[i*nFeatures + j];
					float areaNew = (*groups.area)[newIdx];
					float areaScore = abs(areaNew - areaOld) / (areaNew + areaOld + 1);
					printstr += QString::number(newIdx) + ",";
					float2 com = (*groups.com)[newIdx];
					float2 center = { (trkcom.x + com.x) / 2, (trkcom.y + com.y) / 2 };
					painter->drawLine(trkcom.x, trkcom.y, com.x, com.y);
					painter->drawText(center.x, center.y, QString::number(areaScore)+"|");
				}
				printstr += ")";
				linepen.setColor(QColor(255, 255, 0,50));
				painter->setPen(linepen);
				painter->drawText(trkcom.x, trkcom.y, printstr);
            }
        }

        int* groupSize = groups.ptsNum->cpu_ptr();
        int* groupVec= groups.trkPtsIdx->cpu_ptr();
        float2* groupVelo=groups.velo->cpu_ptr();
        BBox* groupbBox=groups.bBox->cpu_ptr();

        for(int i=1;i<=groups.numGroups;i++)
        {
			linepen.setStyle(Qt::SolidLine);
            int * idx_ptr=groupVec+nFeatures*i;
            unsigned char r=clrvec[i*3],g=clrvec[i*3+1],b=clrvec[i*3+2];
            linepen.setWidth(0);
            linepen.setColor(QColor(r,g,b,150));
            painter->setPen(linepen);
            for(int j=0;j<groupSize[i];j++)
            {
                int idx=idx_ptr[j];
                float x =tracks->getPtr(idx)->x,y=tracks->getPtr(idx)->y;
                painter->drawRect(x-3,y-3,5,5);
            }
			linepen.setColor(QColor(255, 0, 0));
			linepen.setStyle(Qt::DashLine);
            painter->setPen(linepen);
            BBox& bb = groupbBox[i];
			float bbh = bb.bottom - bb.top, bbw = bb.right - bb.left;
			painter->drawRect(bb.left, bb.top, bbw, bbh);
			float persVal= persMap[bb.top * 1920 + bb.left]*2;
			float persScore = abs(bbh - persVal)/(persVal)*abs(bbw-persVal/3)/persVal*3;
			linepen.setStyle(Qt::SolidLine);
			painter->setPen(linepen);
			painter->drawRect(bb.left, bb.top,  persVal / 3, persVal);
			painter->drawText(bb.left, bb.top, QString::number(persScore));
			/*
			linepen.setColor(QColor(0,0,0));
			linepen.setWidth(2);
			painter->setPen(linepen);
			int count = (*streamThd->tracker->rankCountNew)[i];
			painter->drawText(com[i].x, com[i].y, QString::number(count));

			float dstx=com[i].x+groupVelo[i].x,dsty=com[i].y+groupVelo[i].y;
			linepen.setWidth(0);
			painter->setPen(linepen);
			painter->drawLine(com[i].x,com[i].y,dstx,dsty);

            linepen.setStyle(Qt::SolidLine);
            float2* polygon = groups->polygon->cpu_ptr()+i*nFeatures;
            int polyCount=groups->polyCount->cpu_ptr()[i];
            for(int j=1;j<polyCount;j++)
            {
                painter->drawLine(polygon[j-1].x,polygon[j-1].y,polygon[j].x,polygon[j].y);
            }
			*/
        }
    }

    //update();
    //views().at(0)->update();
}
void TrkScene::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
	/*
    if(event->button()==Qt::RightButton)
    {
        int x = event->scenePos().x(),y=event->scenePos().y();
        DragBBox* newbb = new DragBBox(x-10,y-10,x+10,y+10);
        int pid = dragbbvec.size();
        newbb->bbid=pid;
        newbb->setClr(255,255,255);
        sprintf(newbb->txt,"%c\0",pid+'A');
        dragbbvec.push_back(newbb);
        addItem(newbb);
    }
    QGraphicsScene::mousePressEvent(event);
	*/
}
void TrkScene::updateFptr(unsigned char * fptr,int fidx)
{
    bgBrush.setTextureImage(QImage(fptr,streamThd->framewidth,streamThd->frameheight,QImage::Format_RGB888));
    frameidx=fidx;
    //debuggingFile<<frameidx<<std::endl;
}
void TrkScene::clear()
{
    bgBrush.setStyle(Qt::NoBrush);
}
