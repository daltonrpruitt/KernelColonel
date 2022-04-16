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

    CRSMat<> matrix(filename);

    matrix.dump();


    return 0;

}