#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
/* Author: Benjamin T. James */
/* 6.866 Final Project */
int get_gradients(const Mat& cur, const Mat& prev, Mat& Ex, Mat& Ey, Mat& Et);
int get_uv(const Mat& XX,const Mat& XY, const Mat& YY,
	   const Mat& XT, const Mat& YT,
	   int rbegin, int rend,
	   int cbegin, int cend,
	   double& u, double& v, double& eig);
int sum_matrix(const Mat& raw, Mat& out);

#define NUM 6
#define RNUM 48
#define CNUM 64

int main(int argc, char** argv )
{
	std::string out_file = "/dev/null";
	if (argc == 1) {
		std::cout << "usage: " << *argv << " in.mp4 [out.mp4]" << std::endl;
		return 1;
	}
	if (argc == 3) {
		out_file = std::string(argv[2]);
	}
	VideoCapture cap(argv[1]);
	VideoWriter writer;
	writer.open(out_file, VideoWriter::fourcc('M', 'P', '4', 'V'), 30, Size(640,480), 0);
	Mat frame, cur, prev, tmp;
	Mat Ex, Ey, Et;
	namedWindow("6.866 Optical Flow", WINDOW_AUTOSIZE );
	cap >> frame;
	resize(frame, tmp, cv::Size(640,480));
	cvtColor(tmp, prev, COLOR_BGR2GRAY);
	double u[RNUM][CNUM][NUM];
	double v[RNUM][CNUM][NUM];
	const std::vector<std::pair<int,int> > every = {std::make_pair(320,240),
							std::make_pair(160,120),
							std::make_pair(80, 60),
							std::make_pair(40,30),
							std::make_pair(20,20),
							std::make_pair(10,10) };
	while (1) {
		if (!cap.read(frame)) {
			break;
		}
		resize(frame, tmp, cv::Size(640,480));
		cvtColor(tmp, cur, COLOR_BGR2GRAY);
		get_gradients(cur, prev, Ex, Ey, Et);
		Mat XX = Ex.mul(Ex); /* .mul() is elementwise multiplication */
		Mat XY = Ex.mul(Ey);
		Mat YY = Ey.mul(Ey);
		Mat XT = Ex.mul(Et);
		Mat YT = Ey.mul(Et);
		Mat SXX(XX.rows, XX.cols, CV_32SC1);
		Mat SXY(XY.rows, XY.cols, CV_32SC1);
		Mat SYY(YY.rows, YY.cols, CV_32SC1);
		Mat SXT(XT.rows, XT.cols, CV_32SC1);
		Mat SYT(XT.rows, XT.cols, CV_32SC1);
		{
			std::vector<Mat*> fvec, tvec;
			fvec.push_back(&XX); tvec.push_back(&SXX);
			fvec.push_back(&XY); tvec.push_back(&SXY);
			fvec.push_back(&YY); tvec.push_back(&SYY);
			fvec.push_back(&XT); tvec.push_back(&SXT);
			fvec.push_back(&YT); tvec.push_back(&SYT);
#pragma omp parallel for
			for (int i = 0; i < 5; i++) {
				sum_matrix(*fvec[i], *tvec[i]);
			}
		}
		/* Create sum matrices.
		 * Will be helpful in extracting submatrix sums by just accessing four elements
		 */
		std::vector<double> magnitude[NUM];
		std::vector<double> eiglist[NUM];
		double magtop[NUM];
#pragma omp parallel for collapse(2)
		for (int i = 0; i < XX.rows; i += 10) {
			for (int j = 0; j < XX.cols; j += 10) {
				for (size_t k = 0; k < every.size(); k++) {
					int xslice = every[k].first;
					int yslice = every[k].second;
					int re = min(XX.rows-1, i + yslice / 2);
					int rb = max(0, i - yslice / 2);
					int cb = max(0, j - xslice / 2);
					int ce = min(XX.cols-1, j + xslice / 2);
					double tu, tv, eig;
					get_uv(SXX, SXY, SYY, SXT, SYT,
					       rb, re, cb, ce,
					       tu, tv, eig);
					u[i/10][j/10][k] = tu;
					v[i/10][j/10][k] = tv;
					#pragma omp critical
					{
						magnitude[k].push_back(hypot(tu, tv));
						eiglist[k].push_back(abs(eig));
					}
				}
			}
		}
		std::vector<double> weight;
		double weight_sum = 0;
		for (int k = 0; k < NUM; k++) {
			int pos = magnitude[k].size() * 9 / 10;
			std::nth_element(magnitude[k].begin(), magnitude[k].begin() + pos, magnitude[k].end());
			magtop[k] = magnitude[k][pos];

			double ek = log10(sum(eiglist[k])[0] + 1);
			weight.push_back(ek);
			weight_sum += ek;
		}
		/* Drawing */
		cur.copyTo(prev);

		/* Combination of multi-scale u,v and printing to frame */
		for (int i = 0; i < RNUM; i++) {
			for (int j = 0; j < CNUM; j++) {
				int py = cur.rows * (2*i+1) / (2*RNUM);
				int px = cur.cols * (2*j+1) / (2*CNUM);
				double cu = 0, cv = 0;
				std::vector<double> au, av, mag;
				int e_size = (int)every.size();
				for (int k = 0; k < e_size; k++) {
					au.push_back(u[i][j][k]);
					av.push_back(v[i][j][k]);

					double cmag = hypot(au[k],av[k]);
					mag.push_back(cmag);
					if (cmag > magtop[k]) {
						au[k] = magtop[k] * au[k] / cmag;
						av[k] = magtop[k] * av[k] / cmag;
					}
				}
				cu = 3 * au[every.size()-1] * weight[weight.size()-1] / weight_sum;
				cv = 3 * av[every.size()-1] * weight[weight.size()-1] / weight_sum;
				for (int k = e_size - 2; k >= 0; k--) {
					if (mag[k] > mag[k+1]) {
						au[k] = au[k] * mag[k+1] / mag[k];
							av[k] = av[k] * mag[k+1] / mag[k];
					}
					cu += 3 * au[k]* weight[k] / weight_sum;
					cv += 3 * av[k]* weight[k] / weight_sum;
				}
				double cmag = hypot(cu, cv);

				if (cmag > 2) {
					arrowedLine(cur, Point(px, py), Point(px + cu, py + cv), Scalar(255,255,255));
				}
			}
		}
		if (out_file != "/dev/null") {
			writer.write(cur);
		}
		imshow("6.866 Optical Flow", cur);
		if (waitKey(1) >= 0) {
			break;
		}
	}
	cap.release();
	writer.release();
	return 0;
}

int get_gradients(const Mat& C, const Mat& P, Mat& Ex, Mat& Ey, Mat& Et)
{

	int r = C.rows;
	int c = C.cols;
	Mat P00 = P(Range(0, r-1), Range(0, c-1));
	Mat P01 = P(Range(0, r-1), Range(1, c));
	Mat P10 = P(Range(1, r),   Range(0, c-1));
	Mat P11 = P(Range(1, r),   Range(1, c));

	Mat C00 = C(Range(0, r-1), Range(0, c-1));
	Mat C01 = C(Range(0, r-1), Range(1, c));
	Mat C10 = C(Range(1, r),   Range(0, c-1));
	Mat C11 = C(Range(1, r),   Range(1, c));

	Ex = (P01 - P00 + P11 - P10 +
	      C01 - C00 + C11 - C10) / 4;
	Ey = (P10 - P00 + P11 - P01 +
	      C10 - C00 + C11 - C01) / 4;
	Et = (C00 - P00 + C10 - P10 +
	      C01 - P01 + C11 - P11) / 4;
	return 0;
}

int get_uv(const Mat& XX,const Mat& XY, const Mat& YY,
	   const Mat& XT, const Mat& YT,
	   int rb, int re, int cb, int ce,
	   double& u, double& v, double &eig)
{
	double xx = XX.at<int>(re,ce) - XX.at<int>(rb,ce) - XX.at<int>(re, cb) + XX.at<int>(rb,cb);
	double xy = XY.at<int>(re,ce) - XY.at<int>(rb,ce) - XY.at<int>(re, cb) + XY.at<int>(rb,cb);
	double yy = YY.at<int>(re,ce) - YY.at<int>(rb,ce) - YY.at<int>(re, cb) + YY.at<int>(rb,cb);
	double xt = XT.at<int>(re,ce) - XT.at<int>(rb,ce) - XT.at<int>(re, cb) + XT.at<int>(rb,cb);
	double yt = YT.at<int>(re,ce) - YT.at<int>(rb,ce) - YT.at<int>(re, cb) + YT.at<int>(rb,cb);
	double det = xx * yy - xy * xy;
	if (det < 10 * std::numeric_limits<double>::epsilon()) {
		u = 0;
		v = 0;
		eig = 0;
		return 1;
	}
	u = (xy * yt - yy * xt) / det;
	v = (xy * xt - xx * yt) / det;
	double d = sqrt((xx-yy)*(xx-yy) + 4*xy*xy);
        eig = ((xx + yy) + d) / 2;
	return 0;
}

inline int sum_matrix(const Mat& in, Mat& out)
{
	for (int i = 0; i < in.rows; i++) {
		for (int j = 0; j < in.cols; j++) {
			out.at<int>(i,j) = in.at<uint8_t>(i,j) +
				(j == 0 ? 0 : out.at<int>(i,j-1)) +
				(i == 0 ? 0 : out.at<int>(i-1,j)) -
				(i > 0 && j > 0 ? out.at<int>(i-1,j-1) : 0);
		}
	}
	return 0;
}
