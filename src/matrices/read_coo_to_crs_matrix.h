/* 
*   Based on :  Matrix Market I/O example program
*
*/
#pragma once

#include <string>
// #include
#include <stdio.h>
#include <stdlib.h>
#include <mmio.h>
#include <crs_mat.h>

using std::string;
using std::cout;
using std::cerr;
using std::endl;

// template <typename it, typename vt>
int read_coo_to_crs_matrix(string filename, CRSMat &mat)
{
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    // int M, N, nz;   
    // int i, *I, *J;
    double *val;

    if (filename.length() == 0 )
	{
		cerr <<  "Must provide filename to matrix reader!\n");
		exit(1);
	}
    else    
    { 
        if ((f = fopen(filename.c_str(), "r")) == NULL) 
            exit(1);
    }

    if (mm_read_banner(f, &matcode) != 0)
    {
        cerr <<  "Could not process Matrix Market banner." << endl ;
        exit(1);
    }


    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) ) // && mm_is_sparse(matcode) )
    {
        cerr << "Sorry, this application does not support " << endl;
        cerr << "Market Market type: [" << mm_typecode_to_str(matcode) << "]" << endl;
        exit(1);
    }

    /* find out size of sparse matrix .... */

    if ((ret_code = mm_read_mtx_crd_size(f, &mat.m, &mat.n, &mat.nnz)) !=0)
        exit(1);


    /* reseve memory for matrices */

    I = (int *) malloc(nz * sizeof(int));
    J = (int *) malloc(nz * sizeof(int));
    val = (double *) malloc(nz * sizeof(double));


    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (i=0; i<nz; i++)
    {
        fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);
        I[i]--;  /* adjust from 1-based to 0-based */
        J[i]--;
    }

    if (f !=stdin) fclose(f);

    // I am lazy...
    // https://stackoverflow.com/questions/23583975/convert-coo-to-csr-format-in-c
    double *coo_val = val;  
    int *coo_row = I;
    int *coo_col = J;

    
    // double csr_val[nz]      = { 0 };
    // int csr_col[nz]      = { 0 };
    // int csr_row[M + 1]   = { 0 };

    mat.values  = (double *) malloc(nz * sizeof(double));
    mat.indices = (int *) malloc(nz * sizeof(int));
    mat.offsets = (int *) malloc((mat.m + 1)  * sizeof(int));
    mat.offsets = (int *) malloc((mat.m + 1)  * sizeof(int));

    for(i=0; i < mat.m +1; i++){
        mat.offsets[i] = 0;
    }

    for (i = 0; i < nz; i++)
    {
        mat.values[i] = coo_val[i];
        mat.indices[i] = coo_col[i];
        mat.offsets[coo_row[i] + 1]++;
    }
    for (int i = 0; i < rows; i++)
    {
        mat.offsets[i + 1] += csr_row[i];
    }


    /************************/
    /* now write out matrix */
    /************************/
/*
    mm_write_banner(stdout, matcode);
    mm_write_mtx_crd_size(stdout, M, N, nz);
    for (i=0; i<nz; i++)
        fprintf(stdout, "%d %d %20.19g\n", I[i]+1, J[i]+1, val[i]);
*/

	return 0;
}
