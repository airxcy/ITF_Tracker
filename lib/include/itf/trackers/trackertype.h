#ifndef TRACKERTYPE_H
#define TRACKERTYPE_H

typedef struct //corelation pair i0 i1 are idx
{
    int i0, i1;
    float correlation;
}ppair, p_ppair;

typedef unsigned char BYTE;
typedef float REAL;
typedef int PntT;

typedef struct//Int Track Points
{
    PntT x;
    PntT y;
    int t;
} TrkPts, *TrkPts_p;

typedef struct//Float Track Points
{
    REAL x;
    REAL y;
    int t;
}FeatPts,*FeatPts_p;

typedef struct//optical flow vector (x0,y0)->(x1,y1) len is frame Time Span, idx is the id
{
    REAL x0, y0, x1, y1;
    int  len, idx;
} ofv, *ofv_p;

typedef struct//Bounding Box
{
    int left;
    int top;
    int right;
    int bottom;
}BBox, *BBox_p;
/*
BBox operator +(const BBox &a, const float2 &b) {
	BBox tmpBox;
	tmpBox.left = a.left + b.x;
	tmp
}
*/
typedef struct//Bounding Box
{
    int left;
    int top;
    int right;
    int bottom;
}BBoxFloat, *BBoxFloat_p;

struct cvxPnt {
    float x, y;

    bool operator <(const cvxPnt &p) const {
        return x < p.x || (x == p.x && y < p.y);
    }
};

#define UperLowerBound(val,minv,maxv) {int ind=val>(minv);val=ind*val+(!ind)*(minv);ind=val<(maxv);val=ind*val+(!ind)*(maxv);}

#endif // TRACKERTYPE_H

