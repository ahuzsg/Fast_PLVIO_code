#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <vector>
#include <algorithm>
#define SS 0
#define SE 1
#define ES 2
#define EE 3

#define EDGE_VERTICAL   1
#define EDGE_HORIZONTAL 2

#define ANCHOR_PIXEL  254
#define EDGE_PIXEL    255

#define LEFT  1
#define RIGHT 2
#define UP    3
#define DOWN  4
#define choosed_num 300
#define choosed_num2 300
using namespace std;
using namespace cv;
enum GradientOperator { PREWITT_OPERATOR = 101, SOBEL_OPERATOR = 102, SCHARR_OPERATOR = 103 };

struct StackNode {
	int r, c;   
	int parent; 
	int dir;   
};

struct Chain {

	int dir;                   
	int len;                  
	int parent;                
	int children[2];           
	cv::Point *pixels;         
};
struct chainscore {
    bool IFsimilar;
    int x, y;
    double score;
    chainscore(){};
    chainscore(bool _similar, int _x,int _y, double _score) {
        IFsimilar = _similar;
        x = _x;
        y = _y;
        score = _score;
    }
};



struct LS {
    cv::Point2d start;
    cv::Point2d end;
    
    LS(cv::Point2d _start, cv::Point2d _end)
    {
        start = _start;
        end = _end;
    }
};


struct LineSegment {
    double a, b;    
    int invert;
    
    double sx, sy;        
    double ex, ey;        
    
    int segmentNo;        
    int firstPixelIndex; 
    int len;  
    
    LineSegment(){};
    double meanpix;
    double meanpiy;
    LineSegment(double _a, double _b, int _invert, double _sx, double _sy, double _ex, double _ey, int _segmentNo, int _firstPixelIndex, int _len
        , double _meanpix = 0,double _meanpiy=0
    ) {
        a = _a;
        b = _b;
        invert = _invert;
        sx = _sx;
        sy = _sy;
        ex = _ex;
        ey = _ey;
        segmentNo = _segmentNo;
        firstPixelIndex = _firstPixelIndex;
        len = _len;
        meanpix = _meanpix;
        meanpiy = _meanpiy;
    }
};

class EDLines {
							
public:
	EDLines(cv::Mat _srcImage, int _useadaptive=0, GradientOperator _op = SOBEL_OPERATOR, int _gradThresh =9, int _anchorThresh = 5, int _scanInterval = 1, int _minPathLen = 10, double _sigma = 1.0, bool _sumFlag = true, double _line_error = 1.5, int _min_line_len = -1, double _max_distance_between_two_lines = 3, double _max_error =1.1);
    void LienMatch(const std::vector<LineSegment> _linesk1);
    void LienMatch_tree(std::vector<LineSegment> _linesk1);
    void compute_chainscore();
    void rowdetect(int i, int j, int find);
    void coldetect(int i, int j, int find);
	cv::Mat getEdgeImage();
	cv::Mat getAnchorImage();
	cv::Mat getSmoothImage();
	cv::Mat getGradImage();
    cv::Mat getLineImage();
    cv::Mat drawOnImage();
	
	int getSegmentNo();
	int getAnchorNo();
	
	std::vector<cv::Point> getAnchorPoints();
	std::vector<std::vector<cv::Point>> getSegments();
	std::vector<std::vector<cv::Point>> getSortedSegments();
    std::vector<LineSegment> EdlineChoose(int linenum);
    cv::Mat drawChoosedEdline(cv::Mat image, std::vector<LS> lines);
    Mat drawMatchLines(Mat _imagek, Mat _imagek1, EDLines Handles);
	cv::Mat drawParticularSegments(std::vector<int> list);
    std::vector<cv::Point2d> imgPointGra;
    std::vector<LS> getLines();
    int getLinesNo();
    std::vector<LineSegment> choosedlines;
    std::vector<LS> choosedlinePoints;
    std::vector<struct chainscore> chain_score;
    int mapL[choosed_num][choosed_num2];
    int mapLrow[choosed_num] = { 0 };
    int mapLcol[choosed_num2] = { 0 };
protected:
	int width;
	int height;

	uchar *srcImg; 
	std::vector<std::vector< cv::Point> > segmentPoints;
	double sigma; 
	cv::Mat smoothImage;
	uchar *edgeImg;
	uchar *smoothImg; 
	int segmentNos;
	int minPathLen;
	cv::Mat srcImage;
private:
    int similar = 1;
    int unsimilar = 0;
    int useadaptiveGra;
	void ComputeGradient();
    void ComputeGradient_adaptive();
	void ComputeAnchorPoints();
    void ComputeAnchorPoints_adaptive();
	void JoinAnchorPointsUsingSortedAnchors(); 
    void JoinAnchorPointsUsingSortedAnchors_adaptive();
	int* sortAnchorsByGradValue1();

	static int LongestChain(Chain *chains, int root);
	static int RetrieveChainNos(Chain *chains, int root, int chainNos[]);

	int anchorNos;
	std::vector<cv::Point> anchorPoints;
	std::vector<cv::Point> edgePoints;

	cv::Mat edgeImage;
	cv::Mat gradImage;
    cv::Mat threshImage;
    cv::Mat adaptive_gradImage;
	uchar *dirImg; 
	short *gradImg; 
    short* adaptive_gradimg;
	GradientOperator gradOperator; 
	int gradThresh;
	int anchorThresh;
	int scanInterval;
	bool sumFlag;
    
    std::vector<LineSegment> lines;
    
    std::vector<LineSegment> invalidLines;
    std::vector<LS> linePoints;

    int linesNo;
    int min_line_len;
    double line_error;
    double max_distance_between_two_lines;
    double max_error;
    double prec;
    
    
    int ComputeMinLineLength();
    void SplitSegment2Lines(double *x, double *y, int noPixels, int segmentNo);
    void JoinCollinearLines();
    
    bool TryToJoinTwoLineSegments(LineSegment *ls1, LineSegment *ls2, int changeIndex);
    
    static double ComputeMinDistance(double x1, double y1, double a, double b, int invert);
    static void ComputeClosestPoint(double x1, double y1, double a, double b, int invert, double &xOut, double &yOut);
    static void LineFit(double *x, double *y, int count, double &a, double &b, int invert);
    static void LineFit(double *x, double *y, int count, double &a, double &b, double &e, int &invert);
    static double ComputeMinDistanceBetweenTwoLines(LineSegment *ls1, LineSegment *ls2, int *pwhich);
    static void UpdateLineParameters(LineSegment *ls);
    static void EnumerateRectPoints(double sx, double sy, double ex, double ey,int ptsx[], int ptsy[], int *pNoPoints);

};


