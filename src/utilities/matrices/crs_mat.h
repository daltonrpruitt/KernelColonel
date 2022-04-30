/**
 * @file crs_mat.h
 * @author Dalton Winans-Pruitt (daltonrpruitt@gmail.com)
 * @brief Provides basic and complete structs for Compressed Row Storage matrices.
 * @version 0.1
 * @date 2022-04-04
 * 
 * @copyright Copyright (c) 2022
 * 
 * Poorly-designed (not designed) scrapped together matrix parser.
 * Meant to read Matrix Market files and output
 * 
 * 
 * 
 */
#pragma once

#pragma GCC diagnostic push 
#pragma GCC diagnostic ignored "-Wwrite-strings"
#include <mmio.c>
#pragma GCC diagnostic pop

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

using std::cout;
using std::cerr;
using std::endl;
using std::string;
using std::min;
using std::vector;

template <typename mat_t>
bool read_coo_to_crs_matrix(string, mat_t&);

template <typename it=int, typename vt=double>
struct CRSMat_gpu {
    int m, n, nnz;
    vt* values;
    it* indices;
    it* offsets;
};

template <typename it=int, typename vt=double, int const_valence=-1>
class CRSMat {
    public:
        string filename; 
        int m, n, nnz;
        vt* values;
        it* indices;
        it* offsets;

    CRSMat() { 
        static_assert(const_valence==-1 || const_valence==4 || const_valence==5);
        values = nullptr;
        indices = nullptr; 
        offsets = nullptr;
    }

    CRSMat(string filename_input){
        from_file(filename_input);
    }
    
    ~CRSMat() {
        if(values) { delete values; values = nullptr; }
        if(indices) { delete indices; indices = nullptr; }
        if(offsets) { delete offsets; offsets = nullptr; }
    }

    void from_file(string filename_input) {
        filename = filename_input;
        if(filename.find("crs") != string::npos) {
            if(!read_crs_matrix(filename, *this)){
                nnz = -1;
            }
        } else {
            if(const_valence > 0) {
                cerr << "Cannot use valence with non-crs files!" << endl;
                exit(EXIT_FAILURE);
            }
            if(!read_coo_to_crs_matrix(filename, *this)){
                cerr << "Did not read " << filename << "successfully!" << endl;
                nnz = -1;
            }
        }
    }

    // void from_file(string filename_input) {
    //     filename = filename_input;
    //     if(!read_coo_to_crs_matrix(filename, *this)){
    //         int nnz = -1;
    //     }
    // }

    void dump(int num_to_output = 32){
        if(values)  { cout << " Values[]: " ;  for(int i=0; i < min(num_to_output, nnz); ++i) { cout << " " << values[i];  } cout << endl; }
        if(indices) { cout << " indices[]: " ; for(int i=0; i < min(num_to_output, nnz); ++i) { cout << " " << indices[i]; } cout << endl; }
        if(offsets) { cout << " offsets[]: " ; for(int i=0; i < min(num_to_output, m); ++i)   { cout << " " << offsets[i]; } cout << endl; }
    }


};

struct pt {
    int r,c;
    double val;
    pt(int a, int b, double v) : r(a), c(b), val(v) {};
};
bool compare_pts(pt &a, pt &b) { 
    if (a.r < b.r ) return true;
    else if(a.r > b.r) return false;
    else // (a.r == b.r ) 
        return a.c < b.c;  
};

/* generate a random floating point number from min to max */
double randfrom(double min, double max) 
{
    double range = (max - min); 
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

template <typename it=int, typename vt=double, int const_valence=-1>
bool read_crs_matrix(string filename, CRSMat<it, vt, const_valence> &mat) {

    FILE *f;
    int i, *I, *J;
    double *val;

    if (filename.length() == 0 )
	{
		cerr <<  "Must provide filename to matrix reader!" << endl;
		return false;
	}
    else    
    { 
        if ((f = fopen(filename.c_str(), "r")) == NULL) 
		return false;
    }

    // int M=0, N=0, nz=0;   
    if (fscanf(f, "%d %d %d\n", &mat.m, &mat.n, &mat.nnz) != 3){
		cerr <<  "First line of "<<filename << " must be '<rows> <cols> <nnz>' !" << endl;
        cerr << "Exiting!" << endl;
        exit(EXIT_FAILURE);
    };
    if(const_valence >0 ) {
        mat.nnz = mat.m*const_valence;
    }

    mat.values = (vt *) malloc(mat.nnz * sizeof(vt));
    mat.indices = (it *) malloc(mat.nnz * sizeof(it));
    mat.offsets = (it *) malloc((mat.m+1) * sizeof(it));

    it cur_entry = 0;
    for (int i=0; i<mat.m; i++) {
        int valence;
        if(fscanf(f, "%d ", &valence) == EOF) {
            cerr << "Premature EOF for " << filename << " at line "<< i+1 << "!" << endl;
            cerr << "Exiting!" << endl;
            exit(EXIT_FAILURE);
        }
        if(const_valence > 0 && valence > const_valence) {
            cerr << "Expected matrix to have valence of "<<const_valence<<" or less, but is actually "<< valence <<" for row " << i <<"!" << endl;
            cerr << "Exiting!" << endl;
            exit(EXIT_FAILURE);
        }
        if (const_valence > 0) {
            mat.offsets[i+1] = const_valence;
        } else {
            mat.offsets[i+1] = valence;
        }
        
#ifdef DEBUG
        if(i < 25) { cout << "On row " << i << " with " << valence  << " values." <<endl; }
#endif        
        it idx=0; 
        char next_char;
        for(int j=0; j < valence; ++j){
            fscanf(f, "%d%c", &idx, &next_char);
            mat.indices[cur_entry] = idx;
            mat.values[cur_entry] = randfrom(-1.0, 1.0);
            cur_entry++;
        }
        if(const_valence > 0) {
            for(int j=0; j < const_valence-valence; ++j) {
                mat.indices[cur_entry] = mat.indices[cur_entry-valence];
                mat.values[cur_entry] = 0;
                cur_entry++;
            }
        }

    }

    mat.offsets[0]=0;
    for(int j=1; j<mat.m+1;  ++j){
        mat.offsets[j] += mat.offsets[j-1];
    }

    return true;
}



template <typename it=int, typename vt=double>
bool read_coo_to_crs_matrix(string filename, CRSMat<it, vt> &mat) {

    MM_typecode matcode;
    FILE *f;
    // int M, N, nz;   
    int i, *I, *J;
    double *val;

    if (filename.length() == 0 )
	{
		cerr <<  "Must provide filename to matrix reader!" << endl;
		return false;
	}
    else    
    { 
        if ((f = fopen(filename.c_str(), "r")) == NULL) 
		return false;
    }

    if (mm_read_banner(f, &matcode) != 0)
    {
        cerr <<  "Could not process Matrix Market banner." << endl ;
		return false;
    }


    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) ) // && mm_is_sparse(matcode) )
    {
        cerr << "Sorry, this application does not support " << endl;
        cerr << "Market Market type: [" << mm_typecode_to_str(matcode) << "]" << endl;
		return false;
    }

    /* find out size of sparse matrix .... */

    if (mm_read_mtx_crd_size(f, &mat.m, &mat.n, &mat.nnz) !=0) {
        cerr << "Could not read matrix size!" << endl;
		return false;
    }


    /* reseve memory for matrices */

    I = (int *) malloc(mat.nnz * sizeof(int));
    J = (int *) malloc(mat.nnz * sizeof(int));
    val = (double *) malloc(mat.nnz * sizeof(double));


    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (i=0; i<mat.nnz; i++)
    {
        fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);
        I[i]--;  /* adjust from 1-based to 0-based */
        J[i]--;
    }

    if (f !=stdin) fclose(f);

    // Co-sorting
    // https://stackoverflow.com/questions/34878329/how-to-sort-two-vectors-simultaneously-in-c-without-using-boost-or-creating-te
    vector<pt> pts;
    for (i=0; i<mat.nnz; i++){
        pt new_pt(I[i], J[i], val[i]);
        pts.push_back(new_pt);
    }
    delete I; 
    delete J;
    delete val;

    // for (i=0; i<64; i++){
    //     cout << " " << i << ":  " << pts[i].r << " " << pts[i].c << " " << pts[i].val << endl;
    // }

    std::sort(pts.begin(), pts.end(), compare_pts);
    // for (i=0; i<64; i++){
    //     cout << " " << i << ":  " << pts[i].r << " " << pts[i].c << " " << pts[i].val << endl;
    // }


    // I am lazy...
    // https://stackoverflow.com/questions/23583975/convert-coo-to-csr-format-in-c
    // double *coo_val = val;  
    // int *coo_row = I;
    // int *coo_col = J;

    
    // double csr_val[nz]      = { 0 };
    // int csr_col[nz]      = { 0 };
    // int csr_row[M + 1]   = { 0 };

    mat.values  = (vt *) malloc(mat.nnz * sizeof(vt));
    mat.indices = (it *) malloc(mat.nnz * sizeof(it));
    mat.offsets = (it *) malloc((mat.m + 1)  * sizeof(it));

    for(i=0; i < mat.m +1; i++){
        mat.offsets[i] = 0;
    }

    for (i = 0; i < mat.nnz; i++)
    {
        mat.values[i] = pts[i].val;
        mat.indices[i] = pts[i].c;
        mat.offsets[pts[i].r + 1]++;
    }
    for (int i = 0; i < mat.m; i++)
    {
        mat.offsets[i + 1] += mat.offsets[i];
    }

    return true;
}
