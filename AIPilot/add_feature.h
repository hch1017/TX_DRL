#include<vector>
#include<iostream>
#define _USE_MATH_DEFINES
#include<cmath>
#include<math.h>
#include<ctime>


using namespace std;




bool matri_product(const double matrix_1[][3], const double matrix_2[][3], double mat[][3])
{
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			for (int a = 0; a < 3; a++) {
				mat[i][j] += matrix_1[i][a] * matrix_2[j][a];
			}
		}
	}
	return mat;
}

double* reward_angle(double a, double b, double c, double component[3])
{
	double matrix_1[3][3] = {{cos((a + M_PI / 2)) ,  -sin((a + M_PI / 2)),   0},
							{sin((a + M_PI / 2)), cos((a + M_PI / 2)) ,   0},
							{0           ,  0           ,   1} };
	double matrix_2[3][3] = { {cos(b)      ,  0           ,   -sin(b)},
							{0           ,  1           ,   0 },
							{sin(b)     ,  0           ,   cos(b)} };
	double matrix_3[3][3] = { {1           ,  0           ,   0 },
							{0           ,  -cos(c)      ,   sin(c) },
							{0           ,  -sin(c)      ,   -cos(c)} };
	double matrix_4[3][3] = {0.0};
	double mat[3][3] = { 0.0 };
	matri_product(matrix_1, matrix_2, matrix_4);
	matri_product(matrix_4, matrix_3, mat);
	double res[3] = {0.0};
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			res[i] += mat[i][j] * component[j];
		}
	}
	return res;
}



void generate_state(std::vector<double> inputs0, std::vector<double> &model_inputs0) {
	model_inputs0.insert(model_inputs0.end(), inputs0.begin() + 2, inputs0.begin() + 32);
	model_inputs0.insert(model_inputs0.end(), inputs0.begin() + 34, inputs0.begin() + 44);
	model_inputs0.insert(model_inputs0.end(), inputs0.begin() + 123, inputs0.begin() + 126);
	model_inputs0.insert(model_inputs0.end(), inputs0.begin() + 234, inputs0.begin() + 241);
	int i = 243;
	for (int j = 0; j < 8; j++) {
		model_inputs0.insert(model_inputs0.end(), inputs0.begin() + i, inputs0.begin() + i + 4);
		i += 5;
	}
	i = 284;
	for (int j = 0; j < 8; j++) {
		model_inputs0.insert(model_inputs0.end(), inputs0.begin() + i, inputs0.begin() + i + 2);
		i += 3;
	}
	i = 309;
	for (int j = 0; j < 8; j++) {
		model_inputs0.insert(model_inputs0.end(), inputs0.begin() + i, inputs0.begin() + i + 7);
		i += 8;
	}
	model_inputs0.insert(model_inputs0.end(), inputs0.begin() + 428, inputs0.begin() + 430);
	return;
}
