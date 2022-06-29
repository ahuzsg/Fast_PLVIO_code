#include "EDLines.h"
#define pi 3.1415
#define dis_score 30.0//yuzhi for distance of two line
#define rate_score 0.5//yuzhi for len of two line
#define choosed_len 5//choosed lines' lowset len
#define set_score 120//the lowset score to match
#define Gra_score 100//means the gradant should low to 30;
using namespace cv;
using namespace std;

EDLines::EDLines(Mat _srcImage,int _useadaptive, GradientOperator _op,
    int _gradThresh, int _anchorThresh, int _scanInterval,
    int _minPathLen, double _sigma, bool _sumFlag, double _line_error,
    int _min_line_len, double _max_distance_between_two_lines, double _max_error)
{

    if (_gradThresh < 1) _gradThresh = 1;
    if (_anchorThresh < 0) _anchorThresh = 0;
    if (_sigma < 1.0) _sigma = 1.0;

    srcImage = _srcImage;
    height = srcImage.rows;
    width = srcImage.cols;
    useadaptiveGra = _useadaptive;
    gradOperator = _op;
    gradThresh = _gradThresh;
    anchorThresh = _anchorThresh;
    scanInterval = _scanInterval;
    minPathLen = _minPathLen;
    sigma = _sigma;
    sumFlag = _sumFlag;

    segmentNos = 0;
   
    segmentPoints.push_back(vector<Point>());//next<12ms 
    
    edgeImage = Mat(height, width, CV_8UC1, Scalar(0)); 
    smoothImage = Mat(height, width, CV_8UC1);
    gradImage = Mat(height, width, CV_16SC1);
    adaptive_gradImage = Mat(height, width, CV_16SC1);
    //xGra = smoothImage;
    //yGra = smoothImage;

    smoothImg = smoothImage.data;
    gradImg = (short*)gradImage.data;
    adaptive_gradimg = (short*)adaptive_gradImage.data;
    edgeImg = edgeImage.data;

    srcImg = srcImage.data;
    dirImg = new unsigned char[width * height];

  
    if (sigma == 1.0)
        GaussianBlur(srcImage, smoothImage, Size(5, 5), sigma);
    else
        GaussianBlur(srcImage, smoothImage, Size(), sigma); 
    if (useadaptiveGra) {
        anchorThresh =6;
        ComputeGradient_adaptive();
        ComputeAnchorPoints_adaptive();
        JoinAnchorPointsUsingSortedAnchors_adaptive();
    }
       
    else {
        ComputeGradient();//20ms

        ComputeAnchorPoints();//36ms
        JoinAnchorPointsUsingSortedAnchors();//50
    }

    delete[] dirImg;

    min_line_len = _min_line_len;
    line_error = _line_error;
    max_distance_between_two_lines = _max_distance_between_two_lines;
    max_error = _max_error;

    if (min_line_len == -1) 
        min_line_len = ComputeMinLineLength();

    if (min_line_len < 9) 
        min_line_len = 9;



    double* x = new double[(width + height) * 8];
    double* y = new double[(width + height) * 8];

    linesNo = 0;

   
    for (int segmentNumber = 0; segmentNumber < segmentPoints.size(); segmentNumber++) {//13ms
        std::vector<Point> segment = segmentPoints[segmentNumber];
        for (int k = 0; k < segment.size(); k++) {
            x[k] = segment[k].x;
            y[k] = segment[k].y;
        }
        SplitSegment2Lines(x, y, segment.size(), segmentNumber);
    }

 
    JoinCollinearLines();//2ms

    for (int i = 0; i < linesNo; i++) {
        Point2d start(lines[i].sx, lines[i].sy);
        Point2d end(lines[i].ex, lines[i].ey);

        linePoints.push_back(LS(start, end));
    }

    delete[] x;
    delete[] y;
};


    Mat EDLines::getEdgeImage()
    {
        return edgeImage;
    };

/// <summary>
/// EdlineChoose time cost about 1-2ms
/// </summary>
/// <returns></returns>
std::vector<LineSegment> EDLines::EdlineChoose(int linenum) {
    choosedlines = lines;
    choosedlinePoints = linePoints;
  
    std::vector<LineSegment> usedLine = lines;
    if (usedLine.empty()) {
        cout << "there is no lines" << endl;
        return vector<LineSegment>();
    }
    choosedlines.clear();
    choosedlinePoints.clear();
    int num = 0;
    for (auto & line : usedLine) {
        float dx2 = (line.sx - line.ex) * (line.sx - line.ex);
        float dy2=(line.sy - line.ey)* (line.sy - line.ey);
        float diss =  dx2+dy2 ;
        float dis = sqrt(diss);
        /// <summary>
        /// set the choosed_len=50 to pre_choose
        /// </summary>
        /// <returns></returns>
        if (dis >choosed_len) {
            num++;
        
            if ((line.ex < line.sx)|| (line.ex == line.sx&& line.sy < line.ey)) {
                double x = line.ex, y = line.ey;
                line.ex = line.sx;
                line.ey = line.sy;
                line.sx = x;
                line.sy = y;
            }
         
            line.len = dis;
            int pix, piy;
            int sumcnt = 0;
            pix = line.sx;
            piy = line.sy;
            double sumpix=0, sumpiy=0;      
                if (line.invert == 0) {
                    while (pix <= line.ex && sumcnt <20) {
                        if ((piy * width + pix) > imgPointGra.size())
                        {
                            cout << "chujie" <<"pix="<<pix<<" piy="<<piy<< endl;
                            break;
                        }
                        sumcnt++;
                        sumpix = sumpix + imgPointGra[(piy * width + pix)].x;
                        sumpiy = sumpiy + imgPointGra[(piy * width + pix)].y;
                        if (line.b == 0) {
                            pix++;
                        }
                        else {
                            pix++;
                            piy = int(line.b * pix + line.a);
                        }
                        if(pix>=width||piy>=height){
                            break;
                        }
                    }
                    if(sumcnt==0)
                        cout <<"invert=0,sumcnt="<< sumcnt << endl;
                }
                else if (line.invert == 1) {
                    while (pix <= line.ex && sumcnt < 20) {
                        if ((piy * width + pix) > imgPointGra.size())
                        {
                            cout << piy * width + pix << endl;
                            cout << "chujie" << "pix=" << pix << " piy=" << piy << " point size=" << imgPointGra.size() << endl;
                            break;
                        }
                            
                        sumcnt++;
                        sumpix = sumpix + imgPointGra[(piy * width + pix)].x;
                        sumpiy = sumpiy + imgPointGra[(piy * width + pix)].y;
                        if (line.sy>=line.ey) {
                            piy--;
                            pix = int(line.b * piy + line.a);
                            if (piy < 0||pix<0)
                                break;
                        }
                        else {
                            piy++;
                            if(piy>=height||pix>=width){
                                break;
                            }
                            pix = int(line.b * piy + line.a);
                        }
                    }
                    if(sumcnt==0)
                        cout << "invert=1,sumcnt=" << sumcnt<<" piy="<<piy<<" line.ey="<<line.ey<<" line.sx="<<line.sx<<" line.ex=" <<line.ex<< endl;
                }
            line.meanpix = sumpix / sumcnt;
            line.meanpiy = sumpiy / sumcnt;
            UpdateLineParameters(&line);
            /// <summary>
            /// sort all the line length out choosed_len pix 
            /// </summary>
            /// <returns></returns>
            if (num == 1) {
                choosedlines.push_back(line);
                choosedlinePoints.push_back(LS{ Point2d(line.sx,line.sy),Point2d(line.ex,line.ey) });
                continue;
            }
            int pos = -1;
            bool insert_signal = false;
            for (auto& rangLine : choosedlines) {
                pos++;
                if (line.len > rangLine.len) {
                    insert_signal = true;
                    choosedlines.insert(choosedlines.begin() + pos, line);
                    choosedlinePoints.insert(choosedlinePoints.begin() + pos, LS{ Point2d(line.sx,line.sy),Point2d(line.ex,line.ey) });
                    break;
                }
            }
            if(insert_signal==false){ 
                choosedlines.push_back(line); 
                choosedlinePoints.push_back(LS{ Point2d(line.sx,line.sy),Point2d(line.ex,line.ey) }); 
            }
           
        }
    }
    /// <summary>
    ///choose linenum longest line
    /// </summary>
    /// <returns></returns>
    if(choosedlines.size()>linenum){
        choosedlines.erase(choosedlines.begin() + linenum, choosedlines.end());
        choosedlines.pop_back();
        choosedlinePoints.erase(choosedlinePoints.begin() + linenum, choosedlinePoints.end());
        choosedlinePoints.pop_back();
        num = linenum;
    }

    cout << "there is " << choosedlines.size() << " line after del" << endl;

    return choosedlines;
};
/// <summary>
/// there is still a problem :in the k+1 Image,if there are some line (pos and length and roat) similar to the line of kth Image,we cant delete it's disorb;
/// LineMarch time cost <1ms
/// </summary>
/// <param name="_linesk1"></param>
void EDLines::LienMatch(const std::vector<LineSegment> _linesk1) {
    vector<LineSegment> LinesK = choosedlines;
    vector<LineSegment> LinesK1 = _linesk1;
    int idk = -1;

    for (auto &linesk : LinesK) {
        idk++;
        int idk1 = -1;
        int min_score_i = idk, min_score_j = 0;//min_score's row and col
        double min_score = set_score;
        for (auto &linesk1 : LinesK1) {
            idk1++;
            mapL[idk][idk1] = 0;
            if(linesk.meanpix*linesk1.meanpix<0||linesk.meanpiy*linesk1.meanpiy<0){
                continue;
            }
            
            float Graratex = Gra_score-Gra_score *fabs(1 -fabs(linesk.meanpix/linesk1.meanpix));
            float Graratey =Gra_score- Gra_score * fabs(1 - fabs(linesk.meanpiy / linesk1.meanpiy));
            if(Graratex<0||Graratey<0) continue;
                float bk = linesk.invert * (atan(linesk.b)) + (1.0 - linesk.invert) * atan(linesk.b);
                float bk1 = linesk1.invert * atan(linesk1.b) + (1.0 - linesk1.invert) * atan(linesk1.b);
                float roat = fabs( bk-bk1);
                bool kmatch = false;
                float a1 = fabs(linesk.b - 1);
                float a2 = fabs(linesk1.b - 1);
                float a3 = fabs(linesk.b + 1);
                float a4 = fabs(linesk1.b + 1);
                if (a1 < 1e-1 && a2 < 1e-1) kmatch = true;
                if (a3 < 1e-1 && a4 < 1e-1) kmatch = true;
                if ((roat <pi/9&& linesk.invert == linesk1.invert)||(kmatch))  {//first select by roat
                    float rate = float(linesk.len) / float(linesk1.len);
                    if (fabs(rate-1) < rate_score ) {
                        /// <summary>
                        /// compute the rate of length'score and distance's score,and save this min_score to mapL
                        /// </summary>
                        /// <param name="_linesk1"></param>
                        double score_rate = 50.0+50 * (rate_score - fabs(rate - 1))/rate_score;
                        //double dis_s = ComputeMinDistance(linesk1.sx, linesk1.sy, linesk.a, linesk.b, linesk.invert);
                        //double dis_e = ComputeMinDistance(linesk1.ex, linesk1.ey, linesk.a, linesk.b, linesk.invert);
                        double dis_s = sqrt((linesk.sx - linesk1.sx) * (linesk.sx - linesk1.sx) + (linesk.sy - linesk1.sy)*(linesk.sy - linesk1.sy));
                        double dis_e = sqrt((linesk.ex - linesk1.ex) * (linesk.ex - linesk1.ex) + (linesk.ey - linesk1.ey) * (linesk.ey - linesk1.ey));
                        double dis = (dis_s + dis_e) / 2;
                        if (dis <dis_score) {
                            double score_dis = 50.0+50*(1-dis/dis_score);
                            if((score_dis+score_rate+Graratex+Graratey)>min_score){
                                min_score = score_dis + score_rate+Graratex+Graratey;
                                mapL[min_score_i][min_score_j] = 0;
                                min_score_i = idk;
                                min_score_j = idk1;
                                mapL[idk][idk1] = min_score;
                               
                            }

                        }
                    }
            }

        }


    }
    /// <summary>
    /// delete the one match many lines,by the score of mapL
    /// </summary>
    /// <param name="_linesk1"></param>
    for (int i = 0; i < choosed_num2; i++) {
        double max_score = 0;
        int max_score_i = i, max_score_j = 0;
        for (int j = 0; j < choosed_num; j++) {
            if (mapL[j][i] == 0)
                continue;
            if (mapL[j][i] > max_score) {
                max_score = mapL[j][i];
                mapL[max_score_j][max_score_i] = 0;
                max_score_i = i;
                max_score_j = j;
                mapL[j][i]= max_score;
            }
            else 
                mapL[j][i] = 0;

        }
    }
    cout << "match finished" << endl;


};
void EDLines::LienMatch_tree(std::vector<LineSegment> _linesk1) {
    vector<LineSegment> LinesK = choosedlines;
    vector<LineSegment> LinesK1 = _linesk1;
    int idk = -1;

    for (auto& linesk : LinesK) {
        idk++;
        int idk1 = -1;
        int min_score_i = idk, min_score_j = 0;//min_score's row and col
        double min_score = set_score;
        int min_score_i2 = idk, min_score_j2 = 0;//min_score2's row and col
        double min_score2 = set_score;
        bool first = true;
        for (auto& linesk1 : LinesK1) {
            idk1++;
            mapL[idk][idk1] = 0;
            if (linesk.meanpix * linesk1.meanpix < 0 || linesk.meanpiy * linesk1.meanpiy < 0) {
                continue;
            }
            float Graratex = Gra_score * fabs(1 - fabs(linesk.meanpix - linesk1.meanpix) / Gra_score);
            float Graratey = Gra_score * fabs(1 - fabs(linesk.meanpiy - linesk1.meanpiy) / Gra_score);
            float bk = linesk.invert * (atan(linesk.b)) + (1.0 - linesk.invert) * atan(linesk.b);
            float bk1 = linesk1.invert * atan(linesk1.b) + (1.0 - linesk1.invert) * atan(linesk1.b);
            
            float roat = fabs(bk - bk1);
            bool kmatch = false;
            float a1 = fabs(linesk.b- 1);
            float a2 = fabs(linesk1.b - 1);
            float a3 = fabs(linesk.b + 1);
            float a4 = fabs(linesk1.b+1);
            if (a1 < 1e-1 && a2 < 1e-1) {
                kmatch = true;
                //cout << "haha1" << endl;
            }
            if (a3 < 1e-1 && a4 < 1e-1) {
                kmatch = true;
                //cout << "haha2" << endl;
            }
            if (roat < pi / 9 && linesk.invert == linesk1.invert)
            {
                kmatch = true;
                //cout << "haha3" << endl;
            }
            if (kmatch) {//first select by roat
                float rate = float(linesk.len) / float(linesk1.len);
                if (fabs(rate - 1) < rate_score) {
                    /// <summary>
                    /// compute the rate of length'score and distance's score,and save this min_score to mapL
                    /// </summary>
                    /// <param name="_linesk1"></param>
                    double score_rate = 50.0 + 50 * (rate_score - fabs(rate - 1)) / rate_score;
                    double dis_s = sqrt((linesk.sx - linesk1.sx) * (linesk.sx - linesk1.sx) + (linesk.sy - linesk1.sy) * (linesk.sy - linesk1.sy));
                    double dis_e = sqrt((linesk.ex - linesk1.ex) * (linesk.ex - linesk1.ex) + (linesk.ey - linesk1.ey) * (linesk.ey - linesk1.ey));
                    double dis = (dis_s + dis_e) / 2;
                    if (dis < dis_score) {
                        double score_dis = 50.0 + 50 * (1 - dis / dis_score);
                        if (first) {
                            if ((score_dis + score_rate) > min_score) {
                                min_score = score_dis + score_rate;
                                min_score2 = min_score;
                                mapL[min_score_i][min_score_j] = 0;
                                mapL[min_score_i2][min_score_j2] = 0;
                                min_score_i = idk;
                                min_score_j = idk1;
                                min_score_i2 = idk;
                                min_score_j2 = idk1;
                                mapL[idk][idk1] = min_score;
                                first = false;
                                //mapLrow[idk] = 1;
                            }
                        }
                        else {
                            if ((score_dis + score_rate) >= min_score && (score_dis + score_rate) >= min_score2) {
                                min_score2 = min_score;
                                mapL[min_score_i2][min_score_j2] = 0;
                                min_score_i2 = min_score_i;
                                min_score_j2 = min_score_j;
                                mapL[min_score_i2][min_score_j2] = min_score2;
                                min_score= score_dis + score_rate;                                                                                                                              
                                min_score_i = idk;
                                min_score_j = idk1;
                                mapL[min_score_i][min_score_j] = min_score;
                                //mapLrow[idk] = 0;
                                
                            }
                            else if ((score_dis + score_rate) <min_score && (score_dis + score_rate) >= min_score2) {
                                min_score2 = score_dis + score_rate;
                                mapL[min_score_i2][min_score_j2] = 0;
                                min_score_i2 = idk;
                                min_score_j2 = idk1;
                                mapL[idk][idk1] = min_score2;
                                //mapLrow[idk] = 0;
                            }
                            else if ((score_dis + score_rate)>set_score&&(score_dis + score_rate) < min_score2 && min_score_i2 == min_score_i && min_score_j == min_score_j2) {
                                min_score2 = score_dis + score_rate;
                                min_score_i2 = idk;
                                min_score_j2 = idk1;
                                mapL[idk][idk1] = min_score2;
                            }
                        }
                    }
                }
            }

        }


    }
    /// <summary>
    /// delete the one match many lines,by the score of mapL
    /// </summary>
    /// <param name="_linesk1"></param>
    for (int i = 0; i < choosed_num2; i++) {
        double max_score = 0;
        int max_score_i = i, max_score_j = 0;
        double max_score2 = 0;
        int max_score_i2 = i, max_score_j2 = 0;
        bool first = true;
        for (int j = 0; j < choosed_num; j++) {
            if (mapL[j][i] == 0)
                continue;
            if (first) {
                if (mapL[j][i] >max_score&&mapL[j][i]>max_score2) {
                    max_score = mapL[j][i];
                    max_score2 = mapL[j][i];
                    mapL[max_score_j][max_score_i] = 0;
                    mapL[max_score_j2][max_score_i2] = 0;
                    max_score_i = i;
                    max_score_j = j;
                    max_score_i2 = i;
                    max_score_j2 = j;
                    mapL[max_score_j][max_score_i] = max_score;
                    mapL[max_score_j2][max_score_i2] = max_score2;
                    first = false;
                }
            }
            
            else
            {
                if (mapL[j][i] >= max_score && mapL[j][i] >= max_score2) {
                    max_score2 = max_score;
                    mapL[max_score_j2][max_score_i2] = 0;
                    max_score_i2 = max_score_i;
                    max_score_j2 = max_score_j;
                    mapL[max_score_j2][max_score_i2] = max_score;
                    max_score = mapL[j][i];                    
                    max_score_i = i;
                    max_score_j = j;
                }
                else if (mapL[j][i] < max_score && mapL[j][i] >= max_score2) {
                    max_score2 = mapL[j][i];
                    mapL[max_score_j2][max_score_i2] = 0;
                    max_score_i2 = i;
                    max_score_j2 = j;
                    mapL[max_score_j2][max_score_i2] = max_score2;
                }
                else if (mapL[j][i]>0&&mapL[j][i] < max_score2&& max_score_i2== max_score_i&& max_score_j== max_score_j2) {
                    max_score2 = mapL[j][i];
                    
                    max_score_i2 = i;
                    max_score_j2 = j;
                }
                else {
                    mapL[j][i] = 0;
                }
                
            }

        }
    }

    for (int i = 0; i < LinesK.size(); i++) {
        if (mapLrow[i] == 1) {
            continue;
        }
        for (int j = 0; j < LinesK1.size(); j++) {
            if (mapL[i][j]) {
                bool ship = true;
                chain_score.push_back(chainscore(ship, i,j, mapL[i][j]));
                rowdetect(i, j, unsimilar);
                coldetect(i, j, unsimilar);
                if (chain_score.size() == 1) {
                    chain_score.clear();
                }
                else if(chain_score.size()>1)
                    compute_chainscore();
                j = LinesK1.size();
            }
        }
        mapLrow[i] = 1;
    }


};
void EDLines::compute_chainscore() {
    if (chain_score.empty())
        return ;
    if (chain_score.size() == 1) {
        chain_score.clear();
        return;
    }
    
    double Tscore = 0, Fscore = 0;
    for (auto scorePoint : chain_score) {
        
        if (scorePoint.IFsimilar == true)
            Tscore += scorePoint.score;
        else if (scorePoint.IFsimilar == false)
            Fscore += scorePoint.score;
        //cout<<"fine chain:"<< "(x,y)=" << scorePoint.x << "," << scorePoint.y << "score=" << mapL[scorePoint.x][scorePoint.y]<<"similar="<< scorePoint.IFsimilar << endl;
    }
    bool deletsimilar;
    if (Tscore >= Fscore) {
        deletsimilar = false;
    }
    else if (Tscore < Fscore)
        deletsimilar = true;
    for (auto scorePoint : chain_score) {
        if (scorePoint.IFsimilar == deletsimilar) {
            //cout << "delete" << "(x,y)=" << scorePoint.x << "," << scorePoint.y << "score=" << mapL[scorePoint.x][scorePoint.y] << endl;
            if (mapL[scorePoint.x][scorePoint.y] != scorePoint.score)
                cout << "error" << endl;
            mapL[scorePoint.x][scorePoint.y] = 0;
        }

    }
    chain_score.clear();
};
void EDLines::rowdetect(int _i, int _j, int find) {
    if (mapLrow[_i] == 1)
        return;
    mapLrow[_i] = 1;
    for (int j =0; j < choosed_num2; j++) {
        if (j == _j)
            continue;
        if (mapL[_i][j] > 0) {
            bool ship;
            //cv::Point2d coordinate(_i, j);
            if (mapLcol[j] == 1)
                return;
            if (find == 1) {
                ship = true;
                chain_score.push_back(chainscore(ship, _i,j, mapL[_i][j]));
                if(mapLcol[j]==0)
                    coldetect(_i, j, unsimilar);
            }
            else {
                ship = false;
                chain_score.push_back(chainscore(ship, _i,j, mapL[_i][j]));
                if (mapLcol[j] == 0)
                    coldetect(_i, j, similar);
            }

            return ;
        }


    }
};
void EDLines::coldetect(int _i, int _j, int find) {
    if (mapLcol[_j] == 1)
        return;
    mapLcol[_j] = 1;
    for (int i =0; i < choosed_num; i++) {
        if (i == _i)
            continue;
        if (mapL[i][_j] > 0) {
            bool ship;
            ///cv::Point2d coordinate(i, _j);
            if (mapLrow[i] == 1)
                return;
            if (find == 1) {
                ship = true;
                chain_score.push_back(chainscore(ship, i,_j, mapL[i][_j]));
                if (mapLrow[i] == 0)
                    rowdetect(i, _j, unsimilar);
            }
            else {
                ship = false;
                chain_score.push_back(chainscore(ship, i,_j, mapL[i][_j]));
                if (mapLrow[i] == 0)
                    rowdetect(i, _j, similar);
            }

            return;
        }


    }
};
/// <summary>
/// drawMatchlines time cost about 60-70ms
/// </summary>
/// <param name="_imagek"></param>
/// <param name="_imagek1"></param>
/// <param name="Handles"></param>
/// <returns></returns>
Mat EDLines::drawMatchLines(Mat _imagek, Mat _imagek1, EDLines Handles) {

     //LienMatch(Handles.choosedlines);

    /// <summary>
    /// draw the line after choosed,and hebing to showimage
    /// </summary>
    /// <param name="_imagek"></param>
    /// <param name="_imagek1"></param>
    /// <param name="Handles"></param>
    //Mat imagek = drawChoosedEdline(_imagek,choosedlinePoints);
   // Mat imagek1 = drawChoosedEdline(_imagek1, Handles.choosedlinePoints);
    int wid = _imagek.cols;
    int hight = _imagek.rows;
    Mat showimage(hight+20, wid * 2, _imagek.type());
    _imagek.copyTo(showimage(Rect(0, 0, wid, hight)));
    _imagek1.copyTo(showimage(Rect(wid, 0, wid, hight)));
    int numLines = 0;
    for (int i = 0; i < choosed_num;i++) {
        for (int j = 0; j < choosed_num2; j++) {
            if (mapL[i][j]> 1) {
                line(showimage, Point2d(choosedlines[i].sx, choosedlines[i].sy), Point2d(choosedlines[i].ex, choosedlines[i].ey), Scalar(0, 255, 0), 1, LINE_AA, 0);//green
                circle(showimage, Point2d(choosedlines[i].sx, choosedlines[i].sy), 3, Scalar(0, 0, 255), 1, LINE_AA, 0);//hong
                circle(showimage, Point2d(choosedlines[i].ex, choosedlines[i].ey), 3, Scalar(255, 0, 0), 1, LINE_AA, 0);//blue
                line(showimage, Point2d(Handles.choosedlines[j].sx + wid, Handles.choosedlines[j].sy), Point2d(Handles.choosedlines[j].ex + wid, Handles.choosedlines[j].ey), Scalar(0, 255, 0), 1, LINE_AA, 0);//green
                circle(showimage, Point2d(Handles.choosedlines[j].sx + wid, Handles.choosedlines[j].sy), 3, Scalar(0, 0, 255), 1, LINE_AA, 0);//hong
                circle(showimage, Point2d(Handles.choosedlines[j].ex + wid, Handles.choosedlines[j].ey), 3, Scalar(255, 0, 0), 1, LINE_AA, 0);//blue

                putText(showimage,to_string(numLines), Point2d(choosedlines[i].sx, choosedlines[i].sy),1,1,Scalar(0,255,255));
                line(showimage, Point2d(choosedlines[i].sx,choosedlines[i].sy),Point2d( Handles.choosedlines[j].sx+wid, Handles.choosedlines[j].sy), Scalar(255, 255, 0), 1, LINE_AA, 0);//blue
                line(showimage, Point2d(choosedlines[i].ex, choosedlines[i].ey), Point2d(Handles.choosedlines[j].ex+wid, Handles.choosedlines[j].ey), Scalar(0, 255, 255), 1, LINE_AA, 0);//yellow
                numLines++;
                cout << numLines << "score" << mapL[i][j] << endl;
               cout << Point2d(Handles.choosedlinePoints[j].start.x , Handles.choosedlinePoints[j].start.y) << choosedlinePoints[i].start << endl;
                cout << Point2d(Handles.choosedlinePoints[j].end.x, Handles.choosedlinePoints[j].end.y) << choosedlinePoints[i].end << endl;
            }
        
        }
    }
    ///cout << "there is " << numLines << " matched lines" << endl;
    return showimage;
};
/// <summary>
/// draw choosed lines cost 4-6ms
/// </summary>
/// <param name="image"></param>
/// <param name="choosedlines"></param>
/// <returns></returns>
Mat EDLines::drawChoosedEdline(cv::Mat image, std::vector<LS> choosedlines) {
    for (auto &line1 : choosedlines) {
        line(image, line1.start, line1.end, Scalar(0, 255, 0), 1, LINE_AA, 0);//green
        circle(image, line1.start, 3, Scalar(0, 0, 255), 1, LINE_AA, 0);//hong
        circle(image, line1.end, 3, Scalar(255,0, 0), 1, LINE_AA, 0);//blue
    }
    return image;
};
Mat EDLines::getAnchorImage()
{
    Mat anchorImage = Mat(edgeImage.size(), edgeImage.type(), Scalar(0));

    std::vector<Point>::iterator it;

    for (it = anchorPoints.begin(); it != anchorPoints.end(); it++)
        anchorImage.at<uchar>(*it) = 255;

    return anchorImage;
};

Mat EDLines::getSmoothImage()
{
    return smoothImage;
};

Mat EDLines::getGradImage()
{
    Mat result8UC1;
    convertScaleAbs(gradImage, result8UC1);

    return result8UC1;
};


int EDLines::getSegmentNo()
{
    return segmentNos;
};

int EDLines::getAnchorNo()
{
    return anchorNos;
};

std::vector<Point> EDLines::getAnchorPoints()
{
    return anchorPoints;
};

std::vector<std::vector<Point>> EDLines::getSegments()
{
    return segmentPoints;
};

std::vector<std::vector<Point>> EDLines::getSortedSegments()
{
    
    std::sort(segmentPoints.begin(), segmentPoints.end(), [](const std::vector<Point>& a, const std::vector<Point>& b) { return a.size() > b.size(); });

    return segmentPoints;
};

Mat EDLines::drawParticularSegments(std::vector<int> list)
{
    Mat segmentsImage = Mat(edgeImage.size(), edgeImage.type(), Scalar(0));

    std::vector<Point>::iterator it;
    std::vector<int>::iterator itInt;

    for (itInt = list.begin(); itInt != list.end(); itInt++)
        for (it = segmentPoints[*itInt].begin(); it != segmentPoints[*itInt].end(); it++)
            segmentsImage.at<uchar>(*it) = 255;

    return segmentsImage;
};
void EDLines::ComputeGradient_adaptive() {
    std::vector<int> gxV, gyV;
    for (int j = 0; j < width; j++) {
        imgPointGra.push_back(cv::Point2d(0, 0));
    }
    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {

            int com1 = smoothImg[(i + 1) * width + j + 1] - smoothImg[(i - 1) * width + j - 1];
            int com2 = smoothImg[(i - 1) * width + j + 1] - smoothImg[(i + 1) * width + j - 1];

            int gx;
            int gy;
            int gx9;
            int gy9;
            if (j == 1)
                imgPointGra.push_back(cv::Point2d(0, 0));
            switch (gradOperator)
            {
            case PREWITT_OPERATOR:
                gx = com1 + com2 + (smoothImg[i * width + j + 1] - smoothImg[i * width + j - 1]);
                gy = com1 - com2 + (smoothImg[(i + 1) * width + j] - smoothImg[(i - 1) * width + j]);
                //xGra.at<P> = gx;
                //yGra[i - 1][j - 1] = gy;
                imgPointGra.push_back(cv::Point2d(gx, gy));
                gx = abs(gx);
                gy = abs(gy);
                break;
            case SOBEL_OPERATOR:
                gx = com1 + com2 + 2 * (smoothImg[i * width + j + 1] - smoothImg[i * width + j - 1]);
                gy = com1 - com2 + 2 * (smoothImg[(i + 1) * width + j] - smoothImg[(i - 1) * width + j]);
                gx9 = gx;
                gy9 = gy;
                if ((j + 3) < width && (j - 2) >= 0 && (i + 3) < height && (i - 2) >= 0)
                {
                    gx9 = gx + 2 * (smoothImg[i * width + j + 2] - smoothImg[i * width + j - 2]);
                    gy9 = gy + 2 * (smoothImg[(i + 2) * width + j] - smoothImg[(i - 2) * width + j]);
                }
                else {
                    gx9 = gx + 2 * (smoothImg[i * width + j + 1] - smoothImg[i * width + j - 1]);
                    gy9 = gy + 2 * (smoothImg[(i + 1) * width + j] - smoothImg[(i - 1) * width + j]);
                }
                imgPointGra.push_back(cv::Point2d(gx9, gy9));

                gx = abs(gx);
                gy = abs(gy);
                break;
            case SCHARR_OPERATOR:
                gx = abs(3 * (com1 + com2) + 10 * (smoothImg[i * width + j + 1] - smoothImg[i * width + j - 1]));
                gy = abs(3 * (com1 - com2) + 10 * (smoothImg[(i + 1) * width + j] - smoothImg[(i - 1) * width + j]));
            }

            int sum;

            if (sumFlag)
                sum = gx + gy;
            else
                sum = (int)sqrt((double)gx * gx + gy * gy);

            int index = i * width + j;
            gradImg[index] = sum;
            
            gxV.push_back(gx);
            gyV.push_back(gy);
        }
        imgPointGra.push_back(cv::Point2d(0, 0));
    }
    for (int j = 0; j < width; j++) {
        imgPointGra.push_back(cv::Point2d(0, 0));
    }

    for (int j = 0; j < width; j++) {
        gradImg[j] = 0;
        adaptive_gradimg[j] = 256;
        gradImg[(height - 1) * width + j] = 0;
        adaptive_gradimg[(height - 1) * width + j] = 256;
    }
    for (int i = 0; i < height ; i++) {
        gradImg[i * width] = 0;
        adaptive_gradimg[i * width] = 256;
        gradImg[(i + 1) * width - 1] = 0;
        adaptive_gradimg[(i + 1) * width - 1] = 256;

    }
    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {
            if (gradImg[i * width + j] < 5) {
                adaptive_gradimg[i * width + j] = 5;
                continue;
            }
               
            float avrg1 =float(gradImg[i * width + j] + gradImg[i * width + j+1] + gradImg[i * width + j-1] + gradImg[(i+1) * width + j] + gradImg[(i -1) * width + j]+0.5*anchorThresh)/5;
            float avrg2 = float(gradImg[i * width + j] + gradImg[(i+1) * width + j + 1] + gradImg[(i + 1) * width + j - 1] + gradImg[(i + 1) * width + j-1] + gradImg[(i - 1) * width + j+1] + 0.5*anchorThresh)/5;
            float tichu = gradImg[i * width + j + 1] + gradImg[i * width + j - 1] + gradImg[(i + 1) * width + j] + gradImg[(i - 1) * width + j] +
                gradImg[(i + 1) * width + j + 1] + gradImg[(i + 1) * width + j - 1] + gradImg[(i + 1) * width + j - 1] + gradImg[(i - 1) * width + j + 1];

            adaptive_gradimg[i * width + j] = min(avrg1, avrg2);
            if (gradImg[i * width + j]>= adaptive_gradimg[i * width + j]) {
                //if (i > 2 && i <(height - 4) && j>2 && j < width - 4 && tichu < 8)
                    //continue;
                int gx = gxV[(i - 1) * (width - 2) + (j - 1)];
                int gy = gyV[(i - 1) * (width - 2) + (j - 1)];
                if (gx >= gy) dirImg[i * width + j] = EDGE_VERTICAL;
                else          dirImg[i * width + j] = EDGE_HORIZONTAL;
            }
        }
    }
}
void EDLines::ComputeAnchorPoints_adaptive()
{
    for (int i = 2; i < height -2; i++) {
        int start =2;
        int inc = 1;
        if (i % scanInterval != 0) { start = scanInterval; inc = scanInterval; }

        for (int j = start; j < width - 2; j += inc) {
            if (gradImg[i * width + j] < adaptive_gradimg[i * width + j]) continue;

            if (dirImg[i * width + j] == EDGE_VERTICAL) {
                // vertical edge
                int diff1 = gradImg[i * width + j] - gradImg[i * width + j - 1];
                int diff2 = gradImg[i * width + j] - gradImg[i * width + j + 1];
                if (diff1 >= anchorThresh && diff2 >= anchorThresh) {
                    edgeImg[i * width + j] = ANCHOR_PIXEL;
                    anchorPoints.push_back(Point(j, i));
                }

            }
            else if (dirImg[i * width + j] == EDGE_HORIZONTAL) {
                // horizontal edge
                int diff1 = gradImg[i * width + j] - gradImg[(i - 1) * width + j];
                int diff2 = gradImg[i * width + j] - gradImg[(i + 1) * width + j];
                if (diff1 >= anchorThresh && diff2 >= anchorThresh) {
                    edgeImg[i * width + j] = ANCHOR_PIXEL;
                    anchorPoints.push_back(Point(j, i));
                }
            }
        }
    }

    anchorNos = anchorPoints.size();
};

void EDLines::ComputeGradient()
{
 
    for (int j = 0; j < width; j++) { gradImg[j] = gradImg[(height - 1) * width + j] = gradThresh - 1; }
    for (int i = 1; i < height - 1; i++) { gradImg[i * width] = gradImg[(i + 1) * width - 1] = gradThresh - 1; }
    for (int j = 0; j < width; j++) {
        imgPointGra.push_back(cv::Point2d(0, 0));
    }
    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {

            int com1 = smoothImg[(i + 1) * width + j + 1] - smoothImg[(i - 1) * width + j - 1];
            int com2 = smoothImg[(i - 1) * width + j + 1] - smoothImg[(i + 1) * width + j - 1];

            int gx;
            int gy;
            int gx9;
            int gy9;
            if(j==1)
                imgPointGra.push_back(cv::Point2d(0, 0));
            switch (gradOperator)
            {
            case PREWITT_OPERATOR:
                gx = com1 + com2 + (smoothImg[i * width + j + 1] - smoothImg[i * width + j - 1]);
                gy = com1 - com2 + (smoothImg[(i + 1) * width + j] - smoothImg[(i - 1) * width + j]);
                //xGra.at<P> = gx;
                //yGra[i - 1][j - 1] = gy;
                imgPointGra.push_back(cv::Point2d(gx, gy));
                gx = abs(gx);
                gy = abs(gy);
                break;
            case SOBEL_OPERATOR:
                gx = com1 + com2 + 2 * (smoothImg[i * width + j + 1] - smoothImg[i * width + j - 1]);
                gy = com1 - com2 + 2 * (smoothImg[(i + 1) * width + j] - smoothImg[(i - 1) * width + j]);
                gx9 = gx;
                gy9 = gy;
                if ((j + 3) < width && (j - 2) >= 0 && (i + 3) < height && (i - 2) >= 0)
                {
                    gx9 = gx + 2 * (smoothImg[i * width + j + 2] - smoothImg[i * width + j - 2]);
                    gy9 = gy + 2 * (smoothImg[(i + 2) * width + j] - smoothImg[(i - 2) * width + j]);
                }
                else {
                    gx9 = gx + 2 * (smoothImg[i * width + j + 1] - smoothImg[i * width + j - 1]);
                    gy9 = gy + 2 * (smoothImg[(i + 1) * width + j] - smoothImg[(i - 1) * width + j]);
                }
                imgPointGra.push_back(cv::Point2d(gx9, gy9));
                
                gx = abs(gx);
                gy = abs(gy);
                break;
            case SCHARR_OPERATOR:
                gx = abs(3 * (com1 + com2) + 10 * (smoothImg[i * width + j + 1] - smoothImg[i * width + j - 1]));
                gy = abs(3 * (com1 - com2) + 10 * (smoothImg[(i + 1) * width + j] - smoothImg[(i - 1) * width + j]));
            }

            int sum;

            if (sumFlag)
                sum = gx + gy;
            else
                sum = (int)sqrt((double)gx * gx + gy * gy);

            int index = i * width + j;
            gradImg[index] = sum;

            if (sum >= gradThresh) {
                if (gx >= gy) dirImg[index] = EDGE_VERTICAL;
                else          dirImg[index] = EDGE_HORIZONTAL;
            }
        }
     imgPointGra.push_back(cv::Point2d(0, 0));
    }
    for (int j = 0; j < width; j++) {
        imgPointGra.push_back(cv::Point2d(0, 0));
    }
};

void EDLines::ComputeAnchorPoints()
{
    for (int i = 2; i < height - 2; i++) {
        int start = 2;
        int inc = 1;
        if (i % scanInterval != 0) { start = scanInterval; inc = scanInterval; }

        for (int j = start; j < width - 2; j += inc) {
            if (gradImg[i * width + j] < gradThresh) continue;

            if (dirImg[i * width + j] == EDGE_VERTICAL) {
                // vertical edge
                int diff1 = gradImg[i * width + j] - gradImg[i * width + j - 1];
                int diff2 = gradImg[i * width + j] - gradImg[i * width + j + 1];
                if (diff1 >= anchorThresh && diff2 >= anchorThresh) {
                    edgeImg[i * width + j] = ANCHOR_PIXEL;
                    anchorPoints.push_back(Point(j, i));
                }

            }
            else {
                // horizontal edge
                int diff1 = gradImg[i * width + j] - gradImg[(i - 1) * width + j];
                int diff2 = gradImg[i * width + j] - gradImg[(i + 1) * width + j];
                if (diff1 >= anchorThresh && diff2 >= anchorThresh) {
                    edgeImg[i * width + j] = ANCHOR_PIXEL;
                    anchorPoints.push_back(Point(j, i));
                }
            }
        }
    }

    anchorNos = anchorPoints.size(); 
};
void EDLines::JoinAnchorPointsUsingSortedAnchors_adaptive()
{
    int* chainNos = new int[(width + height) * 8];

    Point* pixels = new Point[width * height];
    StackNode* stack = new StackNode[width * height];
    Chain* chains = new Chain[width * height];


    int* A = sortAnchorsByGradValue1();


    int totalPixels = 0;

    for (int k = anchorNos - 1; k >= 0; k--) {
        int pixelOffset = A[k];

        int i = pixelOffset / width;
        int j = pixelOffset % width;


        if (edgeImg[i * width + j] != ANCHOR_PIXEL) continue;

        chains[0].len = 0;
        chains[0].parent = -1;
        chains[0].dir = 0;
        chains[0].children[0] = chains[0].children[1] = -1;
        chains[0].pixels = NULL;


        int noChains = 1;
        int len = 0;
        int duplicatePixelCount = 0;
        int top = -1;

        if (dirImg[i * width + j] == EDGE_VERTICAL) {
            stack[++top].r = i;
            stack[top].c = j;
            stack[top].dir = DOWN;
            stack[top].parent = 0;

            stack[++top].r = i;
            stack[top].c = j;
            stack[top].dir = UP;
            stack[top].parent = 0;

        }
        else {
            stack[++top].r = i;
            stack[top].c = j;
            stack[top].dir = RIGHT;
            stack[top].parent = 0;

            stack[++top].r = i;
            stack[top].c = j;
            stack[top].dir = LEFT;
            stack[top].parent = 0;
        }


    StartOfWhile:
        while (top >= 0 && duplicatePixelCount < width * height) {
            int r = stack[top].r;
            int c = stack[top].c;
            int dir = stack[top].dir;
            int parent = stack[top].parent;
            top--;

            if (edgeImg[r * width + c] != EDGE_PIXEL)
                duplicatePixelCount++;
            if (duplicatePixelCount >= width * height)
                continue;
            chains[noChains].dir = dir;
            chains[noChains].parent = parent;
            chains[noChains].children[0] = chains[noChains].children[1] = -1;


            int chainLen = 0;

            chains[noChains].pixels = &pixels[len];

            pixels[len].y = r;
            pixels[len].x = c;
            len++;
            chainLen++;

            if (dir == LEFT) {
                while (dirImg[r * width + c] == EDGE_HORIZONTAL) {
                    edgeImg[r * width + c] = EDGE_PIXEL;
                    if (edgeImg[(r - 1) * width + c] == ANCHOR_PIXEL) edgeImg[(r - 1) * width + c] = 0;
                    if (edgeImg[(r + 1) * width + c] == ANCHOR_PIXEL) edgeImg[(r + 1) * width + c] = 0;


                    if (edgeImg[r * width + c - 1] >= ANCHOR_PIXEL) { c--; }
                    else if (edgeImg[(r - 1) * width + c - 1] >= ANCHOR_PIXEL) { r--; c--; }
                    else if (edgeImg[(r + 1) * width + c - 1] >= ANCHOR_PIXEL) { r++; c--; }
                    else {

                        int A = gradImg[(r - 1) * width + c - 1];
                        int B = gradImg[r * width + c - 1];
                        int C = gradImg[(r + 1) * width + c - 1];

                        if (A > B) {
                            if (A > C) r--;
                            else       r++;
                        }
                        else  if (C > B) r++;
                        c--;
                    }

                    if (edgeImg[r * width + c] == EDGE_PIXEL || gradImg[r * width + c] < adaptive_gradimg[r * width + c]) {
                        if (chainLen > 0) {
                            chains[noChains].len = chainLen;
                            chains[parent].children[0] = noChains;
                            noChains++;
                        }
                        goto StartOfWhile;
                    }


                    pixels[len].y = r;
                    pixels[len].x = c;
                    len++;
                    chainLen++;
                }

                stack[++top].r = r;
                stack[top].c = c;
                stack[top].dir = DOWN;
                stack[top].parent = noChains;

                stack[++top].r = r;
                stack[top].c = c;
                stack[top].dir = UP;
                stack[top].parent = noChains;

                len--;
                chainLen--;

                chains[noChains].len = chainLen;
                chains[parent].children[0] = noChains;
                noChains++;

            }
            else if (dir == RIGHT) {
                while (dirImg[r * width + c] == EDGE_HORIZONTAL) {
                    edgeImg[r * width + c] = EDGE_PIXEL;

                    if (edgeImg[(r + 1) * width + c] == ANCHOR_PIXEL) edgeImg[(r + 1) * width + c] = 0;
                    if (edgeImg[(r - 1) * width + c] == ANCHOR_PIXEL) edgeImg[(r - 1) * width + c] = 0;


                    if (edgeImg[r * width + c + 1] >= ANCHOR_PIXEL) { c++; }
                    else if (edgeImg[(r + 1) * width + c + 1] >= ANCHOR_PIXEL) { r++; c++; }
                    else if (edgeImg[(r - 1) * width + c + 1] >= ANCHOR_PIXEL) { r--; c++; }
                    else {

                        int A = gradImg[(r - 1) * width + c + 1];
                        int B = gradImg[r * width + c + 1];
                        int C = gradImg[(r + 1) * width + c + 1];

                        if (A > B) {
                            if (A > C) r--;
                            else       r++;
                        }
                        else if (C > B) r++;
                        c++;
                    }

                    if (edgeImg[r * width + c] == EDGE_PIXEL || gradImg[r * width + c] < adaptive_gradimg[r * width + c]) {
                        if (chainLen > 0) {
                            chains[noChains].len = chainLen;
                            chains[parent].children[1] = noChains;
                            noChains++;
                        }
                        goto StartOfWhile;
                    }


                    pixels[len].y = r;
                    pixels[len].x = c;
                    len++;
                    chainLen++;
                }

                stack[++top].r = r;
                stack[top].c = c;
                stack[top].dir = DOWN;
                stack[top].parent = noChains;

                stack[++top].r = r;
                stack[top].c = c;
                stack[top].dir = UP;
                stack[top].parent = noChains;

                len--;
                chainLen--;

                chains[noChains].len = chainLen;
                chains[parent].children[1] = noChains;
                noChains++;

            }
            else if (dir == UP) {
                while (dirImg[r * width + c] == EDGE_VERTICAL) {
                    edgeImg[r * width + c] = EDGE_PIXEL;


                    if (edgeImg[r * width + c - 1] == ANCHOR_PIXEL) edgeImg[r * width + c - 1] = 0;
                    if (edgeImg[r * width + c + 1] == ANCHOR_PIXEL) edgeImg[r * width + c + 1] = 0;

                    if (edgeImg[(r - 1) * width + c] >= ANCHOR_PIXEL) { r--; }
                    else if (edgeImg[(r - 1) * width + c - 1] >= ANCHOR_PIXEL) { r--; c--; }
                    else if (edgeImg[(r - 1) * width + c + 1] >= ANCHOR_PIXEL) { r--; c++; }
                    else {
                        int A = gradImg[(r - 1) * width + c - 1];
                        int B = gradImg[(r - 1) * width + c];
                        int C = gradImg[(r - 1) * width + c + 1];

                        if (A > B) {
                            if (A > C) c--;
                            else       c++;
                        }
                        else if (C > B) c++;
                        r--;
                    }

                    if (edgeImg[r * width + c] == EDGE_PIXEL || gradImg[r * width + c] < adaptive_gradimg[r * width + c]) {
                        if (chainLen > 0) {
                            chains[noChains].len = chainLen;
                            chains[parent].children[0] = noChains;
                            noChains++;
                        }
                        goto StartOfWhile;
                    }


                    pixels[len].y = r;
                    pixels[len].x = c;

                    len++;
                    chainLen++;
                }

                stack[++top].r = r;
                stack[top].c = c;
                stack[top].dir = RIGHT;
                stack[top].parent = noChains;

                stack[++top].r = r;
                stack[top].c = c;
                stack[top].dir = LEFT;
                stack[top].parent = noChains;

                len--;
                chainLen--;

                chains[noChains].len = chainLen;
                chains[parent].children[0] = noChains;
                noChains++;

            }
            else {
                while (dirImg[r * width + c] == EDGE_VERTICAL) {
                    edgeImg[r * width + c] = EDGE_PIXEL;

                    if (edgeImg[r * width + c + 1] == ANCHOR_PIXEL) edgeImg[r * width + c + 1] = 0;
                    if (edgeImg[r * width + c - 1] == ANCHOR_PIXEL) edgeImg[r * width + c - 1] = 0;

                    if (edgeImg[(r + 1) * width + c] >= ANCHOR_PIXEL) { r++; }
                    else if (edgeImg[(r + 1) * width + c + 1] >= ANCHOR_PIXEL) { r++; c++; }
                    else if (edgeImg[(r + 1) * width + c - 1] >= ANCHOR_PIXEL) { r++; c--; }
                    else {

                        int A = gradImg[(r + 1) * width + c - 1];
                        int B = gradImg[(r + 1) * width + c];
                        int C = gradImg[(r + 1) * width + c + 1];

                        if (A > B) {
                            if (A > C) c--;
                            else       c++;
                        }
                        else if (C > B) c++;
                        r++;
                    }

                    if (edgeImg[r * width + c] == EDGE_PIXEL || gradImg[r * width + c] < adaptive_gradimg[r * width + c]) {
                        if (chainLen > 0) {
                            chains[noChains].len = chainLen;
                            chains[parent].children[1] = noChains;
                            noChains++;
                        }
                        goto StartOfWhile;
                    }

                    pixels[len].y = r;
                    pixels[len].x = c;

                    len++;
                    chainLen++;
                }

                stack[++top].r = r;
                stack[top].c = c;
                stack[top].dir = RIGHT;
                stack[top].parent = noChains;

                stack[++top].r = r;
                stack[top].c = c;
                stack[top].dir = LEFT;
                stack[top].parent = noChains;

                len--;
                chainLen--;

                chains[noChains].len = chainLen;
                chains[parent].children[1] = noChains;
                noChains++;
            }

        }

        if (duplicatePixelCount >= width * height)
            continue;
        if (len - duplicatePixelCount < minPathLen) {
            for (int k = 0; k < len; k++) {

                edgeImg[pixels[k].y * width + pixels[k].x] = 0;
                edgeImg[pixels[k].y * width + pixels[k].x] = 0;

            }

        }
        else {

            int noSegmentPixels = 0;

            int totalLen = LongestChain(chains, chains[0].children[1]);

            if (totalLen > 0) {
                int count = RetrieveChainNos(chains, chains[0].children[1], chainNos);


                for (int k = count - 1; k >= 0; k--) {
                    int chainNo = chainNos[k];


                    int fr = chains[chainNo].pixels[chains[chainNo].len - 1].y;
                    int fc = chains[chainNo].pixels[chains[chainNo].len - 1].x;

                    int index = noSegmentPixels - 2;
                    while (index >= 0) {
                        int dr = abs(fr - segmentPoints[segmentNos][index].y);
                        int dc = abs(fc - segmentPoints[segmentNos][index].x);

                        if (dr <= 1 && dc <= 1) {

                            segmentPoints[segmentNos].pop_back();
                            noSegmentPixels--;
                            index--;
                        }
                        else break;
                    }

                    if (chains[chainNo].len > 1 && noSegmentPixels > 0) {
                        fr = chains[chainNo].pixels[chains[chainNo].len - 2].y;
                        fc = chains[chainNo].pixels[chains[chainNo].len - 2].x;

                        int dr = abs(fr - segmentPoints[segmentNos][noSegmentPixels - 1].y);
                        int dc = abs(fc - segmentPoints[segmentNos][noSegmentPixels - 1].x);

                        if (dr <= 1 && dc <= 1) chains[chainNo].len--;
                    }

                    for (int l = chains[chainNo].len - 1; l >= 0; l--) {
                        segmentPoints[segmentNos].push_back(chains[chainNo].pixels[l]);
                        noSegmentPixels++;
                    }

                    chains[chainNo].len = 0;
                }
            }

            totalLen = LongestChain(chains, chains[0].children[0]);
            if (totalLen > 1) {

                int count = RetrieveChainNos(chains, chains[0].children[0], chainNos);


                int lastChainNo = chainNos[0];
                chains[lastChainNo].pixels++;
                chains[lastChainNo].len--;

                for (int k = 0; k < count; k++) {
                    int chainNo = chainNos[k];

                    int fr = chains[chainNo].pixels[0].y;
                    int fc = chains[chainNo].pixels[0].x;

                    int index = noSegmentPixels - 2;
                    while (index >= 0) {
                        int dr = abs(fr - segmentPoints[segmentNos][index].y);
                        int dc = abs(fc - segmentPoints[segmentNos][index].x);

                        if (dr <= 1 && dc <= 1) {
                            segmentPoints[segmentNos].pop_back();
                            noSegmentPixels--;
                            index--;
                        }
                        else break;
                    }

                    int startIndex = 0;
                    int chainLen = chains[chainNo].len;
                    if (chainLen > 1 && noSegmentPixels > 0) {
                        int fr = chains[chainNo].pixels[1].y;
                        int fc = chains[chainNo].pixels[1].x;

                        int dr = abs(fr - segmentPoints[segmentNos][noSegmentPixels - 1].y);
                        int dc = abs(fc - segmentPoints[segmentNos][noSegmentPixels - 1].x);

                        if (dr <= 1 && dc <= 1) { startIndex = 1; }
                    }

                    for (int l = startIndex; l < chains[chainNo].len; l++) {
                        segmentPoints[segmentNos].push_back(chains[chainNo].pixels[l]);
                        noSegmentPixels++;
                    }

                    chains[chainNo].len = 0;
                }
            }



            int fr = segmentPoints[segmentNos][1].y;
            int fc = segmentPoints[segmentNos][1].x;


            int dr = abs(fr - segmentPoints[segmentNos][noSegmentPixels - 1].y);
            int dc = abs(fc - segmentPoints[segmentNos][noSegmentPixels - 1].x);


            if (dr <= 1 && dc <= 1) {
                segmentPoints[segmentNos].erase(segmentPoints[segmentNos].begin());
                noSegmentPixels--;
            } //end-if

            segmentNos++;
            segmentPoints.push_back(vector<Point>());


            for (int k = 2; k < noChains; k++) {
                if (chains[k].len < 2) continue;

                totalLen = LongestChain(chains, k);

                if (totalLen >= 10) {

                    int count = RetrieveChainNos(chains, k, chainNos);


                    noSegmentPixels = 0;
                    for (int k = 0; k < count; k++) {
                        int chainNo = chainNos[k];

                        int fr = chains[chainNo].pixels[0].y;
                        int fc = chains[chainNo].pixels[0].x;

                        int index = noSegmentPixels - 2;
                        while (index >= 0) {
                            int dr = abs(fr - segmentPoints[segmentNos][index].y);
                            int dc = abs(fc - segmentPoints[segmentNos][index].x);

                            if (dr <= 1 && dc <= 1) {

                                segmentPoints[segmentNos].pop_back();
                                noSegmentPixels--;
                                index--;
                            }
                            else break;
                        }

                        int startIndex = 0;
                        int chainLen = chains[chainNo].len;
                        if (chainLen > 1 && noSegmentPixels > 0) {
                            int fr = chains[chainNo].pixels[1].y;
                            int fc = chains[chainNo].pixels[1].x;

                            int dr = abs(fr - segmentPoints[segmentNos][noSegmentPixels - 1].y);
                            int dc = abs(fc - segmentPoints[segmentNos][noSegmentPixels - 1].x);

                            if (dr <= 1 && dc <= 1) { startIndex = 1; }
                        }
                        for (int l = startIndex; l < chains[chainNo].len; l++) {
                            segmentPoints[segmentNos].push_back(chains[chainNo].pixels[l]);
                            noSegmentPixels++;
                        }

                        chains[chainNo].len = 0;
                    }
                    segmentPoints.push_back(vector<Point>());
                    segmentNos++;
                }
            }

        }

    }

    segmentPoints.pop_back();

    delete[] A;
    delete[] chains;
    delete[] stack;
    delete[] chainNos;
    delete[] pixels;
};
void EDLines::JoinAnchorPointsUsingSortedAnchors()
{
	int *chainNos = new int[(width + height) * 8];

	Point *pixels = new Point[width*height];
	StackNode *stack = new StackNode[width*height];
	Chain *chains = new Chain[width*height];

	
	int *A = sortAnchorsByGradValue1();

	
	int totalPixels = 0;

	for (int k = anchorNos - 1; k >= 0; k--) {
		int pixelOffset = A[k];

		int i = pixelOffset / width;
		int j = pixelOffset % width;


		if (edgeImg[i*width + j] != ANCHOR_PIXEL) continue;

		chains[0].len = 0;
		chains[0].parent = -1;
		chains[0].dir = 0;
		chains[0].children[0] = chains[0].children[1] = -1;
		chains[0].pixels = NULL;


		int noChains = 1;
		int len = 0;
		int duplicatePixelCount = 0;
		int top = -1; 

		if (dirImg[i*width + j] == EDGE_VERTICAL) {
			stack[++top].r = i;
			stack[top].c = j;
			stack[top].dir = DOWN;
			stack[top].parent = 0;

			stack[++top].r = i;
			stack[top].c = j;
			stack[top].dir = UP;
			stack[top].parent = 0;

		}
		else {
			stack[++top].r = i;
			stack[top].c = j;
			stack[top].dir = RIGHT;
			stack[top].parent = 0;

			stack[++top].r = i;
			stack[top].c = j;
			stack[top].dir = LEFT;
			stack[top].parent = 0;
		} 

		
	StartOfWhile:
		while (top >= 0) {
			int r = stack[top].r;
			int c = stack[top].c;
			int dir = stack[top].dir;
			int parent = stack[top].parent;
			top--;

			if (edgeImg[r*width + c] != EDGE_PIXEL) duplicatePixelCount++;

			chains[noChains].dir = dir;   
			chains[noChains].parent = parent;
			chains[noChains].children[0] = chains[noChains].children[1] = -1;


			int chainLen = 0;

			chains[noChains].pixels = &pixels[len];

			pixels[len].y = r;
			pixels[len].x = c;
			len++;
			chainLen++;

			if (dir == LEFT) {
				while (dirImg[r*width + c] == EDGE_HORIZONTAL) {
					edgeImg[r*width + c] = EDGE_PIXEL;
					if (edgeImg[(r - 1)*width + c] == ANCHOR_PIXEL) edgeImg[(r - 1)*width + c] = 0;
					if (edgeImg[(r + 1)*width + c] == ANCHOR_PIXEL) edgeImg[(r + 1)*width + c] = 0;


					if (edgeImg[r*width + c - 1] >= ANCHOR_PIXEL) { c--; }
					else if (edgeImg[(r - 1)*width + c - 1] >= ANCHOR_PIXEL) { r--; c--; }
					else if (edgeImg[(r + 1)*width + c - 1] >= ANCHOR_PIXEL) { r++; c--; }
					else {
				
						int A = gradImg[(r - 1)*width + c - 1];
						int B = gradImg[r*width + c - 1];
						int C = gradImg[(r + 1)*width + c - 1];

						if (A > B) {
							if (A > C) r--;
							else       r++;
						}
						else  if (C > B) r++;
						c--;
					}

					if (edgeImg[r*width + c] == EDGE_PIXEL || gradImg[r*width + c] < gradThresh) {
						if (chainLen > 0) {
							chains[noChains].len = chainLen;
							chains[parent].children[0] = noChains;
							noChains++;
						}
						goto StartOfWhile;
					}


					pixels[len].y = r;
					pixels[len].x = c;
					len++;
					chainLen++;
				}

				stack[++top].r = r;
				stack[top].c = c;
				stack[top].dir = DOWN;
				stack[top].parent = noChains;

				stack[++top].r = r;
				stack[top].c = c;
				stack[top].dir = UP;
				stack[top].parent = noChains;

				len--;
				chainLen--;

				chains[noChains].len = chainLen;
				chains[parent].children[0] = noChains;
				noChains++;

			}
			else if (dir == RIGHT) {
				while (dirImg[r*width + c] == EDGE_HORIZONTAL) {
					edgeImg[r*width + c] = EDGE_PIXEL;

					if (edgeImg[(r + 1)*width + c] == ANCHOR_PIXEL) edgeImg[(r + 1)*width + c] = 0;
					if (edgeImg[(r - 1)*width + c] == ANCHOR_PIXEL) edgeImg[(r - 1)*width + c] = 0;


					if (edgeImg[r*width + c + 1] >= ANCHOR_PIXEL) { c++; }
					else if (edgeImg[(r + 1)*width + c + 1] >= ANCHOR_PIXEL) { r++; c++; }
					else if (edgeImg[(r - 1)*width + c + 1] >= ANCHOR_PIXEL) { r--; c++; }
					else {
						
						int A = gradImg[(r - 1)*width + c + 1];
						int B = gradImg[r*width + c + 1];
						int C = gradImg[(r + 1)*width + c + 1];

						if (A > B) {
							if (A > C) r--;       
							else       r++;      
						}
						else if (C > B) r++;  
						c++;
					}

					if (edgeImg[r*width + c] == EDGE_PIXEL || gradImg[r*width + c] < gradThresh) {
						if (chainLen > 0) {
							chains[noChains].len = chainLen;
							chains[parent].children[1] = noChains;
							noChains++;
						}
						goto StartOfWhile;
					}


					pixels[len].y = r;
					pixels[len].x = c;
					len++;
					chainLen++;
				}

				stack[++top].r = r;
				stack[top].c = c;
				stack[top].dir = DOWN;
				stack[top].parent = noChains;

				stack[++top].r = r;
				stack[top].c = c;
				stack[top].dir = UP;
				stack[top].parent = noChains;

				len--;
				chainLen--;

				chains[noChains].len = chainLen;
				chains[parent].children[1] = noChains;
				noChains++;

			}
			else if (dir == UP) {
				while (dirImg[r*width + c] == EDGE_VERTICAL) {
					edgeImg[r*width + c] = EDGE_PIXEL;


					if (edgeImg[r*width + c - 1] == ANCHOR_PIXEL) edgeImg[r*width + c - 1] = 0;
					if (edgeImg[r*width + c + 1] == ANCHOR_PIXEL) edgeImg[r*width + c + 1] = 0;

					if (edgeImg[(r - 1)*width + c] >= ANCHOR_PIXEL) { r--; }
					else if (edgeImg[(r - 1)*width + c - 1] >= ANCHOR_PIXEL) { r--; c--; }
					else if (edgeImg[(r - 1)*width + c + 1] >= ANCHOR_PIXEL) { r--; c++; }
					else {
						int A = gradImg[(r - 1)*width + c - 1];
						int B = gradImg[(r - 1)*width + c];
						int C = gradImg[(r - 1)*width + c + 1];

						if (A > B) {
							if (A > C) c--;
							else       c++;
						}
						else if (C > B) c++;
						r--;
					}

					if (edgeImg[r*width + c] == EDGE_PIXEL || gradImg[r*width + c] < gradThresh) {
						if (chainLen > 0) {
							chains[noChains].len = chainLen;
							chains[parent].children[0] = noChains;
							noChains++;
						}
						goto StartOfWhile;
					}


					pixels[len].y = r;
					pixels[len].x = c;

					len++;
					chainLen++;
				}

				stack[++top].r = r;
				stack[top].c = c;
				stack[top].dir = RIGHT;
				stack[top].parent = noChains;

				stack[++top].r = r;
				stack[top].c = c;
				stack[top].dir = LEFT;
				stack[top].parent = noChains;

				len--;
				chainLen--;

				chains[noChains].len = chainLen;
				chains[parent].children[0] = noChains;
				noChains++;

			}
			else {
				while (dirImg[r*width + c] == EDGE_VERTICAL) {
					edgeImg[r*width + c] = EDGE_PIXEL;

					if (edgeImg[r*width + c + 1] == ANCHOR_PIXEL) edgeImg[r*width + c + 1] = 0;
					if (edgeImg[r*width + c - 1] == ANCHOR_PIXEL) edgeImg[r*width + c - 1] = 0;

					if (edgeImg[(r + 1)*width + c] >= ANCHOR_PIXEL) { r++; }
					else if (edgeImg[(r + 1)*width + c + 1] >= ANCHOR_PIXEL) { r++; c++; }
					else if (edgeImg[(r + 1)*width + c - 1] >= ANCHOR_PIXEL) { r++; c--; }
					else {
					
						int A = gradImg[(r + 1)*width + c - 1];
						int B = gradImg[(r + 1)*width + c];
						int C = gradImg[(r + 1)*width + c + 1];

						if (A > B) {
							if (A > C) c--;      
							else       c++;     
						}
						else if (C > B) c++;  
						r++;
					}

					if (edgeImg[r*width + c] == EDGE_PIXEL || gradImg[r*width + c] < gradThresh) {
						if (chainLen > 0) {
							chains[noChains].len = chainLen;
							chains[parent].children[1] = noChains;
							noChains++;
						}
						goto StartOfWhile;
					}

					pixels[len].y = r;
					pixels[len].x = c;

					len++;
					chainLen++;
				}

				stack[++top].r = r;
				stack[top].c = c;
				stack[top].dir = RIGHT;
				stack[top].parent = noChains;

				stack[++top].r = r;
				stack[top].c = c;
				stack[top].dir = LEFT;
				stack[top].parent = noChains;

				len--;
				chainLen--;

				chains[noChains].len = chainLen;
				chains[parent].children[1] = noChains;
				noChains++;
			}

		}


		if (len - duplicatePixelCount < minPathLen) {
			for (int k = 0; k<len; k++) {

				edgeImg[pixels[k].y*width + pixels[k].x] = 0;
				edgeImg[pixels[k].y*width + pixels[k].x] = 0;

			}

		}
		else {

			int noSegmentPixels = 0;

			int totalLen = LongestChain(chains, chains[0].children[1]);

			if (totalLen > 0) {
				int count = RetrieveChainNos(chains, chains[0].children[1], chainNos);

		
				for (int k = count - 1; k >= 0; k--) {
					int chainNo = chainNos[k];

 
                    int fr = chains[chainNo].pixels[chains[chainNo].len - 1].y;
                    int fc = chains[chainNo].pixels[chains[chainNo].len - 1].x;

                    int index = noSegmentPixels - 2;
                    while (index >= 0) {
                        int dr = abs(fr - segmentPoints[segmentNos][index].y);
                        int dc = abs(fc - segmentPoints[segmentNos][index].x);

                        if (dr <= 1 && dc <= 1) {
                      
                            segmentPoints[segmentNos].pop_back();
                            noSegmentPixels--;
                            index--;
                        }
                        else break;
                    }

                    if (chains[chainNo].len > 1 && noSegmentPixels > 0) {
                        fr = chains[chainNo].pixels[chains[chainNo].len - 2].y;
                        fc = chains[chainNo].pixels[chains[chainNo].len - 2].x;

                        int dr = abs(fr - segmentPoints[segmentNos][noSegmentPixels - 1].y);
                        int dc = abs(fc - segmentPoints[segmentNos][noSegmentPixels - 1].x);

                        if (dr <= 1 && dc <= 1) chains[chainNo].len--;
                    }

					for (int l = chains[chainNo].len - 1; l >= 0; l--) {
						segmentPoints[segmentNos].push_back(chains[chainNo].pixels[l]);
						noSegmentPixels++;
					}

					chains[chainNo].len = 0;  
				}
			}

			totalLen = LongestChain(chains, chains[0].children[0]);
			if (totalLen > 1) {
				
				int count = RetrieveChainNos(chains, chains[0].children[0], chainNos);
                
             
				int lastChainNo = chainNos[0];
				chains[lastChainNo].pixels++;
				chains[lastChainNo].len--;

				for (int k = 0; k<count; k++) {
					int chainNo = chainNos[k];

					int fr = chains[chainNo].pixels[0].y;
					int fc = chains[chainNo].pixels[0].x;

					int index = noSegmentPixels - 2;
					while (index >= 0) {
						int dr = abs(fr - segmentPoints[segmentNos][index].y);
						int dc = abs(fc - segmentPoints[segmentNos][index].x);

						if (dr <= 1 && dc <= 1) {
							segmentPoints[segmentNos].pop_back();
							noSegmentPixels--;
							index--;
						}
						else break;
					}

					int startIndex = 0;
					int chainLen = chains[chainNo].len;
					if (chainLen > 1 && noSegmentPixels > 0) {
						int fr = chains[chainNo].pixels[1].y;
						int fc = chains[chainNo].pixels[1].x;

						int dr = abs(fr - segmentPoints[segmentNos][noSegmentPixels - 1].y);
						int dc = abs(fc - segmentPoints[segmentNos][noSegmentPixels - 1].x);

						if (dr <= 1 && dc <= 1) { startIndex = 1; }
					}

					for (int l = startIndex; l<chains[chainNo].len; l++) {
						segmentPoints[segmentNos].push_back(chains[chainNo].pixels[l]);
						noSegmentPixels++;
					}

					chains[chainNo].len = 0; 
				}
			}


			
			int fr = segmentPoints[segmentNos][1].y;
			int fc = segmentPoints[segmentNos][1].x;


			int dr = abs(fr - segmentPoints[segmentNos][noSegmentPixels - 1].y);
			int dc = abs(fc - segmentPoints[segmentNos][noSegmentPixels - 1].x);


			if (dr <= 1 && dc <= 1) {
				segmentPoints[segmentNos].erase(segmentPoints[segmentNos].begin());
				noSegmentPixels--;
			} //end-if

			segmentNos++;
			segmentPoints.push_back(vector<Point>());

													
			for (int k = 2; k<noChains; k++) {
				if (chains[k].len < 2) continue;

				totalLen = LongestChain(chains, k);

				if (totalLen >= 10) {

					int count = RetrieveChainNos(chains, k, chainNos);

		
					noSegmentPixels = 0;
					for (int k = 0; k<count; k++) {
						int chainNo = chainNos[k];
					
						int fr = chains[chainNo].pixels[0].y;
						int fc = chains[chainNo].pixels[0].x;

						int index = noSegmentPixels - 2;
						while (index >= 0) {
							int dr = abs(fr - segmentPoints[segmentNos][index].y);
							int dc = abs(fc - segmentPoints[segmentNos][index].x);

							if (dr <= 1 && dc <= 1) {
						
								segmentPoints[segmentNos].pop_back();
								noSegmentPixels--;
								index--;
							}
							else break;
						}

						int startIndex = 0;
						int chainLen = chains[chainNo].len;
						if (chainLen > 1 && noSegmentPixels > 0) {
							int fr = chains[chainNo].pixels[1].y;
							int fc = chains[chainNo].pixels[1].x;

							int dr = abs(fr - segmentPoints[segmentNos][noSegmentPixels - 1].y);
							int dc = abs(fc - segmentPoints[segmentNos][noSegmentPixels - 1].x);

							if (dr <= 1 && dc <= 1) { startIndex = 1; }
						}
						for (int l = startIndex; l<chains[chainNo].len; l++) {
							segmentPoints[segmentNos].push_back(chains[chainNo].pixels[l]);
							noSegmentPixels++;
						}

						chains[chainNo].len = 0;  
					}
					segmentPoints.push_back(vector<Point>());
					segmentNos++;
				}
			}

		}

	}

	segmentPoints.pop_back();

	delete[] A;
	delete[] chains;
	delete[] stack;
	delete[] chainNos;
	delete[] pixels;
};


int* EDLines::sortAnchorsByGradValue1()
{
    int SIZE = 128 * 256;
    int* C = new int[SIZE];
    memset(C, 0, sizeof(int) * SIZE);

    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {
            if (edgeImg[i * width + j] != ANCHOR_PIXEL) continue;

            int grad = gradImg[i * width + j];
            C[grad]++;
        }
    } //end-for 


    for (int i = 1; i < SIZE; i++) C[i] += C[i - 1];

    int noAnchors = C[SIZE - 1];
    int* A = new int[noAnchors];
    memset(A, 0, sizeof(int) * noAnchors);


    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {
            if (edgeImg[i * width + j] != ANCHOR_PIXEL) continue;

            int grad = gradImg[i * width + j];
            int index = --C[grad];
            A[index] = i * width + j;   
        }
    }

    delete[] C;

    return A;

};


int EDLines::LongestChain(Chain* chains, int root) {
    if (root == -1 || chains[root].len == 0) return 0;

    int len0 = 0;
    if (chains[root].children[0] != -1) len0 = LongestChain(chains, chains[root].children[0]);

    int len1 = 0;
    if (chains[root].children[1] != -1) len1 = LongestChain(chains, chains[root].children[1]);

    int max = 0;

    if (len0 >= len1) {
        max = len0;
        chains[root].children[1] = -1;

    }
    else {
        max = len1;
        chains[root].children[0] = -1;
    }

    return chains[root].len + max;
};

int EDLines::RetrieveChainNos(Chain* chains, int root, int chainNos[])
{
    int count = 0;

    while (root != -1) {
        chainNos[count] = root;
        count++;

        if (chains[root].children[0] != -1) root = chains[root].children[0];
        else                                root = chains[root].children[1];
    }

    return count;
};


vector<LS> EDLines::getLines()
{
    return linePoints;
};

int EDLines::getLinesNo()
{
    return linesNo;
};

Mat EDLines::getLineImage()
{
    Mat lineImage = Mat(height, width, CV_8UC1, Scalar(0));
    for (int i = 0; i < linesNo; i++) {
        line(lineImage, linePoints[i].start, linePoints[i].end, Scalar(255), 1, LINE_AA, 0);
    }

    return lineImage;
};

Mat EDLines::drawOnImage()
{
    Mat colorImage = Mat(height, width, CV_8UC1, srcImg);
    cvtColor(colorImage, colorImage, COLOR_GRAY2BGR);
    for (int i = 0; i < linesNo; i++) {
        line(colorImage, linePoints[i].start, linePoints[i].end, Scalar(0, 255, 0), 1, LINE_AA, 0); 
    }

    return colorImage;
};


int EDLines::ComputeMinLineLength() {

    double logNT = 2.0 * (log10((double)width) + log10((double)height));
    return (int)round((-logNT / log10(0.125)) * 0.5);
};

void EDLines::SplitSegment2Lines(double* x, double* y, int noPixels, int segmentNo)
{

    int firstPixelIndex = 0;

    while (noPixels >= min_line_len) {
    
        bool valid = false;
        double lastA, lastB, error;
        int lastInvert;

        while (noPixels >= min_line_len) {
            LineFit(x, y, min_line_len, lastA, lastB, error, lastInvert);
            if (error <= 0.5) { valid = true; break; }

            noPixels -= 1;
            x += 1; y += 1;
            firstPixelIndex += 1;
        }

        if (valid == false) return;

        int index = min_line_len;
        int len = min_line_len;

        while (index < noPixels) {
            int startIndex = index;
            int lastGoodIndex = index - 1;
            int goodPixelCount = 0;
            int badPixelCount = 0;
            while (index < noPixels) {
                double d = ComputeMinDistance(x[index], y[index], lastA, lastB, lastInvert);

                if (d <= line_error) {
                    lastGoodIndex = index;
                    goodPixelCount++;
                    badPixelCount = 0;

                }
                else {
                    badPixelCount++;
                    if (badPixelCount >= 5) break;
                }

                index++;
            }

            if (goodPixelCount >= 2) {
                len += lastGoodIndex - startIndex + 1;
                LineFit(x, y, len, lastA, lastB, lastInvert);
                index = lastGoodIndex + 1;
            }

            if (goodPixelCount < 2 || index >= noPixels) {
             
                double sx, sy, ex, ey;

                int index = 0;
                while (ComputeMinDistance(x[index], y[index], lastA, lastB, lastInvert) > line_error) index++;
                ComputeClosestPoint(x[index], y[index], lastA, lastB, lastInvert, sx, sy);
                int noSkippedPixels = index;

                index = lastGoodIndex;
                while (ComputeMinDistance(x[index], y[index], lastA, lastB, lastInvert) > line_error) index--;
                ComputeClosestPoint(x[index], y[index], lastA, lastB, lastInvert, ex, ey);

        
                lines.push_back(LineSegment(lastA, lastB, lastInvert, sx, sy, ex, ey, segmentNo, firstPixelIndex + noSkippedPixels, index - noSkippedPixels + 1));
                linesNo++;
                len = index + 1;
                break;
            }
        }

        noPixels -= len;
        x += len;
        y += len;
        firstPixelIndex += len;
    }
};


void EDLines::JoinCollinearLines()
{
    int lastLineIndex = -1;
    int i = 0;
    while (i < linesNo) {
        int segmentNo = lines[i].segmentNo;

        lastLineIndex++;
        if (lastLineIndex != i)
            lines[lastLineIndex] = lines[i];

        int firstLineIndex = lastLineIndex;

        int count = 1;
        for (int j = i + 1; j < linesNo; j++) {
            if (lines[j].segmentNo != segmentNo) break;
          
            if (TryToJoinTwoLineSegments(&lines[lastLineIndex], &lines[j],
                lastLineIndex) == false) {
                lastLineIndex++;
                if (lastLineIndex != j)
                    lines[lastLineIndex] = lines[j];

            }

            count++;
        }


        if (firstLineIndex != lastLineIndex)
            if (TryToJoinTwoLineSegments(&lines[firstLineIndex], &lines[lastLineIndex],
                firstLineIndex)) {
                lastLineIndex--;
            }


        i += count;
    }

    linesNo = lastLineIndex + 1;
};

double EDLines::ComputeMinDistance(double x1, double y1, double a, double b, int invert)
{
    double x2, y2;

    if (invert == 0) {
        if (b == 0) {
            x2 = x1;
            y2 = a;///distance=(y1-a)^2

        }
        else {
            double d = -1.0 / (b);
            double c = y1 - d * x1;

            x2 = (a - c) / (d - b);//
            y2 = a + b * x2;
        }

    }
    else {
        if (b == 0) {
            x2 = a;
            y2 = y1;

        }
        else {
            double d = -1.0 / (b);
            double c = x1 - d * y1;

            y2 = (a - c) / (d - b);
            x2 = a + b * y2;
        }
    }

    return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
};

void EDLines::ComputeClosestPoint(double x1, double y1, double a, double b, int invert, double& xOut, double& yOut)
{
    double x2, y2;

    if (invert == 0) {
        if (b == 0) {
            x2 = x1;
            y2 = a;

        }
        else {
            double d = -1.0 / (b);
            double c = y1 - d * x1;

            x2 = (a - c) / (d - b);
            y2 = a + b * x2;
        }

    }
    else {
        if (b == 0) {
            x2 = a;
            y2 = y1;

        }
        else {
            double d = -1.0 / (b);
            double c = x1 - d * y1;

            y2 = (a - c) / (d - b);
            x2 = a + b * y2;
        }
    }

    xOut = x2;
    yOut = y2;
};


void EDLines::LineFit(double* x, double* y, int count, double& a, double& b, int invert)
{
    if (count < 2) return;

    double S = count, Sx = 0.0, Sy = 0.0, Sxx = 0.0, Sxy = 0.0;
    for (int i = 0; i < count; i++) {
        Sx += x[i];
        Sy += y[i];
    } 

    if (invert) {
     
        double* t = x;
        x = y;
        y = t;

        double d = Sx;
        Sx = Sy;
        Sy = d;
    }
    for (int i = 0; i < count; i++) {
        Sxx += x[i] * x[i];
        Sxy += x[i] * y[i];
    } //end-for

    double D = S * Sxx - Sx * Sx;
    a = (Sxx * Sy - Sx * Sxy) / D;
    b = (S * Sxy - Sx * Sy) / D;
};
void EDLines::LineFit(double* x, double* y, int count, double& a, double& b, double& e, int& invert)
{
    if (count < 2) return;

    double S = count, Sx = 0.0, Sy = 0.0, Sxx = 0.0, Sxy = 0.0;
    for (int i = 0; i < count; i++) {
        Sx += x[i];
        Sy += y[i];
    }

    double mx = Sx / count;
    double my = Sy / count;

    double dx = 0.0;
    double dy = 0.0;
    for (int i = 0; i < count; i++) {
        dx += (x[i] - mx) * (x[i] - mx);
        dy += (y[i] - my) * (y[i] - my);
    }

    if (dx < dy) {
        invert = 1;
        double* t = x;
        x = y;
        y = t;

        double d = Sx;
        Sx = Sy;
        Sy = d;

    }
    else {
        invert = 0;
    }

    for (int i = 0; i < count; i++) {
        Sxx += x[i] * x[i];
        Sxy += x[i] * y[i];
    }

    double D = S * Sxx - Sx * Sx;
    a = (Sxx * Sy - Sx * Sxy) / D;
    b = (S * Sxy - Sx * Sy) / D;

    if (b == 0.0) {
        double error = 0.0;
        for (int i = 0; i < count; i++) {
            error += fabs((a)-y[i]);
        }
        e = error / count;

    }
    else {
        double error = 0.0;
        for (int i = 0; i < count; i++) {
            double d = -1.0 / (b);
            double c = y[i] - d * x[i];
            double x2 = ((a)-c) / (d - (b));
            double y2 = (a)+(b)*x2;

            double dist = (x[i] - x2) * (x[i] - x2) + (y[i] - y2) * (y[i] - y2);
            error += dist;
        }

        e = sqrt(error / count);
    }
};

bool EDLines::TryToJoinTwoLineSegments(LineSegment * ls1, LineSegment * ls2, int changeIndex)
{
    int which;
    double dist = ComputeMinDistanceBetweenTwoLines(ls1, ls2, &which);
    if (dist > max_distance_between_two_lines) return false;

    double dx = ls1->sx - ls1->ex;
    double dy = ls1->sy - ls1->ey;
    double prevLen = sqrt(dx*dx + dy*dy);
    
    dx = ls2->sx - ls2->ex;
    dy = ls2->sy - ls2->ey;
    double nextLen = sqrt(dx*dx + dy*dy);
    

    LineSegment *shorter = ls1;
    LineSegment *longer = ls2;
    
    if (prevLen > nextLen) { shorter = ls2; longer = ls1; }
    

    dist = ComputeMinDistance(shorter->sx, shorter->sy, longer->a, longer->b, longer->invert);
    dist += ComputeMinDistance((shorter->sx + shorter->ex) / 2.0, (shorter->sy + shorter->ey) / 2.0, longer->a, longer->b, longer->invert);
    dist += ComputeMinDistance(shorter->ex, shorter->ey, longer->a, longer->b, longer->invert);
    
    dist /= 3.0;
    
    if (dist > max_error) return false;
    

    dx = fabs(ls1->sx - ls2->sx);
    dy = fabs(ls1->sy - ls2->sy);
    double d = dx + dy;
    double max = d;
    which = 1;
    

    dx = fabs(ls1->sx - ls2->ex);
    dy = fabs(ls1->sy - ls2->ey);
    d = dx + dy;
    if (d > max) {
        max = d;
        which = 2;
    }

    dx = fabs(ls1->ex - ls2->sx);
    dy = fabs(ls1->ey - ls2->sy);
    d = dx + dy;
    if (d > max) {
        max = d;
        which = 3;
    }

    dx = fabs(ls1->ex - ls2->ex);
    dy = fabs(ls1->ey - ls2->ey);
    d = dx + dy;
    if (d > max) {
        max = d;
        which = 4;
    }
    
    if (which == 1) {

        ls1->ex = ls2->sx;
        ls1->ey = ls2->sy;
        
    }
    else if (which == 2) {
        
        ls1->ex = ls2->ex;
        ls1->ey = ls2->ey;
        
    }
    else if (which == 3) {

        ls1->sx = ls2->sx;
        ls1->sy = ls2->sy;
        
    }
    else {

        ls1->sx = ls1->ex;
        ls1->sy = ls1->ey;
        
        ls1->ex = ls2->ex;
        ls1->ey = ls2->ey;
    }
    

    

    if (ls1->firstPixelIndex + ls1->len + 5 >= ls2->firstPixelIndex) ls1->len += ls2->len;
    else if (ls2->len > ls1->len) {
        ls1->firstPixelIndex = ls2->firstPixelIndex;
        ls1->len = ls2->len;
    }
    
    UpdateLineParameters(ls1);
    lines[changeIndex] = *ls1;
    
    return true;
};

double EDLines::ComputeMinDistanceBetweenTwoLines(LineSegment* ls1, LineSegment* ls2, int* pwhich)
{
    double dx = ls1->sx - ls2->sx;
    double dy = ls1->sy - ls2->sy;
    double d = sqrt(dx * dx + dy * dy);
    double min = d;
    int which = SS;

    dx = ls1->sx - ls2->ex;
    dy = ls1->sy - ls2->ey;
    d = sqrt(dx * dx + dy * dy);
    if (d < min) { min = d; which = SE; }

    dx = ls1->ex - ls2->sx;
    dy = ls1->ey - ls2->sy;
    d = sqrt(dx * dx + dy * dy);
    if (d < min) { min = d; which = ES; }

    dx = ls1->ex - ls2->ex;
    dy = ls1->ey - ls2->ey;
    d = sqrt(dx * dx + dy * dy);
    if (d < min) { min = d; which = EE; }

    if (pwhich) *pwhich = which;
    return min;
};


void EDLines::UpdateLineParameters(LineSegment* ls)
{
    double dx = ls->ex - ls->sx;
    double dy = ls->ey - ls->sy;

    if (fabs(dx) >= fabs(dy)) {
        ls->invert = 0;
        if (fabs(dy) < 1e-3) { ls->b = 0; ls->a = (ls->sy + ls->ey) / 2; }
        else {
            ls->b = dy / dx;
            ls->a = ls->sy - (ls->b) * ls->sx;
        }

    }
    else {
        ls->invert = 1;
        if (fabs(dx) < 1e-3) { ls->b = 0; ls->a = (ls->sx + ls->ex) / 2; }
        else {
            ls->b = dx / dy;
            ls->a = ls->sx - (ls->b) * ls->sy;
        }
    }
};

void EDLines::EnumerateRectPoints(double sx, double sy, double ex, double ey, int ptsx[], int ptsy[], int * pNoPoints)
{
    double vxTmp[4], vyTmp[4];
    double vx[4], vy[4];
    int n, offset;
    
    double x1 = sx;
    double y1 = sy;
    double x2 = ex;
    double y2 = ey;
    double width = 2;
    
    double dx = x2 - x1;
    double dy = y2 - y1;
    double vLen = sqrt(dx*dx + dy*dy);
    
    dx = dx / vLen;
    dy = dy / vLen;
    
    vxTmp[0] = x1 - dy * width / 2.0;
    vyTmp[0] = y1 + dx * width / 2.0;
    vxTmp[1] = x2 - dy * width / 2.0;
    vyTmp[1] = y2 + dx * width / 2.0;
    vxTmp[2] = x2 + dy * width / 2.0;
    vyTmp[2] = y2 - dx * width / 2.0;
    vxTmp[3] = x1 + dy * width / 2.0;
    vyTmp[3] = y1 - dx * width / 2.0;
    
    if (x1 < x2 && y1 <= y2) offset = 0;
    else if (x1 >= x2 && y1 < y2) offset = 1;
    else if (x1 > x2 && y1 >= y2) offset = 2;
    else                          offset = 3;
    
    for (n = 0; n<4; n++) {
        vx[n] = vxTmp[(offset + n) % 4];
        vy[n] = vyTmp[(offset + n) % 4];
    }
    

    int x = (int)ceil(vx[0]) - 1;
    int y = (int)ceil(vy[0]);
    double ys = -DBL_MAX, ye = -DBL_MAX;
    
    int noPoints = 0;
    while (1) {
       
        y++;
        while (y > ye && x <= vx[2]) {
            x++;
            
            if (x > vx[2]) break;
            if ((double)x < vx[3]) {
                if (fabs(vx[0] - vx[3]) <= 0.01) {
                    if (vy[0]<vy[3]) ys = vy[0];
                    else if (vy[0]>vy[3]) ys = vy[3];
                    else     ys = vy[0] + (x - vx[0]) * (vy[3] - vy[0]) / (vx[3] - vx[0]);
                }
                else
                    ys = vy[0] + (x - vx[0]) * (vy[3] - vy[0]) / (vx[3] - vx[0]);
                
            }
            else {
                if (fabs(vx[3] - vx[2]) <= 0.01) {
                    if (vy[3]<vy[2]) ys = vy[3];
                    else if (vy[3]>vy[2]) ys = vy[2];
                    else     ys = vy[3] + (x - vx[3]) * (y2 - vy[3]) / (vx[2] - vx[3]);
                }
                else
                    ys = vy[3] + (x - vx[3]) * (vy[2] - vy[3]) / (vx[2] - vx[3]);
            }

            if ((double)x < vx[1]) {
                /* интерполяция */
                if (fabs(vx[0] - vx[1]) <= 0.01) {
                    if (vy[0]<vy[1]) ye = vy[1];
                    else if (vy[0]>vy[1]) ye = vy[0];
                    else     ye = vy[0] + (x - vx[0]) * (vy[1] - vy[0]) / (vx[1] - vx[0]);
                }
                else
                    ye = vy[0] + (x - vx[0]) * (vy[1] - vy[0]) / (vx[1] - vx[0]);
                
            }
            else {
             
                if (fabs(vx[1] - vx[2]) <= 0.01) {
                    if (vy[1]<vy[2]) ye = vy[2];
                    else if (vy[1]>vy[2]) ye = vy[1];
                    else     ye = vy[1] + (x - vx[1]) * (vy[2] - vy[1]) / (vx[2] - vx[1]);
                }
                else
                    ye = vy[1] + (x - vx[1]) * (vy[2] - vy[1]) / (vx[2] - vx[1]);
            }
            
            y = (int)ceil(ys);
        }
        
        if (x > vx[2]) break;
        
        ptsx[noPoints] = x;
        ptsy[noPoints] = y;
        noPoints++;
    }
    
    *pNoPoints = noPoints;
};

