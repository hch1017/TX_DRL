#include<vector>
#include<iostream>
#include<cmath>
#include<ctime>
using namespace std;

bool SoftMax(std::vector<float> &input)
{
    int n = input.size(), i;
    double sum = 0;
    for (i = 0; i < n; i++)
    {
        input[i] = exp(input[i]);
        sum += input[i];
    }
    for (i = 0; i < n; i++)
    {
        input[i] /= sum;
    }
    return true;
}
// int Sample(std::vector<double> input)
// {
//     int n = input.size(), i, j;
//     for (i = 0; i< n-1; i++)
//     {
//         for (j = i+1; j < n; j++)
//         {
//             input[i] += input[j];
//         }
//     }
//     double r = (double)(rand()/(double)RAND_MAX);
//     for (i = 0; i < n-1; i++)
//     {
//         if (r > input[i+1]) break;
//     }
//     return i;
// }
int Sample(std::vector<float> input)
{

    srand((unsigned int)time(NULL));
    int n = input.size(), i = 0;
    double sum = 0;
    double r = (double)(rand()/(double)RAND_MAX);
    for (i = 0; i < n-1; i++)
    {   
        sum += input[i];
        if (r < sum) break;
    }
    return i;
}