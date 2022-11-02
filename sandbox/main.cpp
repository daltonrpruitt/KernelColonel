/*
    Simple test file for matrix computation...
*/

#include <crs_mat.h>


#include <iostream>
#include <string>

using std::cout; 
using std::cerr; 
using std::endl; 
using std::string; 



int main(int argc, char *argv[])
{

    if (argc < 2)
	{
		cerr << "Usage: " << argv[0] << " [martix-market-filename]" << endl;
		exit(1);
	}

    string filename(argv[1]);

    CRSMat<int, double> matrix(filename);
    matrix.dump(40);

    CRSMat<int, double, 4> matrix4(filename);
    matrix4.dump(40);

    CRSMat<int, double, 5> matrix5(filename);
    matrix5.dump(40);

    return 0;

}