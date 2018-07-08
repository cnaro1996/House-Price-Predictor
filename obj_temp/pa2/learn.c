#include <stdlib.h>
#include <stdio.h>


/*frees the space of a matrix*/
void release(double** A){

  return;
}


/*Generate Matrix X from the predicting data file (K cols M rows)*/
void genpMat(int K, int M, double** X, char* predictfile){
  FILE* fp = NULL;
  int i, j, a;

  fp = fopen(predictfile, "r");
  fscanf(fp, "%d/n", &a);
  
  for(i = 0; i<M; i++){
    for(j = 0; j<K; j++){
      fscanf(fp, "%lf,", &X[i][j]);
    }
    fscanf(fp, "\n");
  }

  return;
}

/*Populates Matrices X and Y from training data file*/
void gentMat(double** X, double** Y, char* trainfile){ 
  int i, j, K, N;
  double temp;
  FILE* fp = fopen(trainfile, "r");
  fscanf(fp, "%d\n", &K);
  fscanf(fp, "%d\n", &N);
  
  for(i = 0; i<N; i++){
    for(j = 0; j<K+1; j++){
      /*First column of X is 1's*/
      if(j == 0){
        fscanf(fp, "%lf,", &temp);
        X[i][j] = 1;
        j++;
        X[i][j] = temp;
        continue;
      }
      fscanf(fp, "%lf,", &temp);
      X[i][j] = temp;
    }
    /*Last column of X is Y*/
    fscanf(fp, "%lf", &temp);
    Y[i][0] = temp;
    fscanf(fp, "\n");
  }

  fclose(fp);
  return;
}


/*Multiplies two matrices together*/
double** mMat(double** A, double** B, int Arow, int Acol, int Brow,
int Bcol){
  int i, j, k;

  double** product = (double**) malloc(Arow*sizeof(double*));
  for(i = 0; i<Arow; i++){
    product[i] = (double*) calloc(Bcol, sizeof(double));
  }

  for(i=0; i<Arow; i++){
    for(j=0; j<Bcol; j++){
      for(k=0; k<Acol; k++){
        product[i][j] = product[i][j] + (A[i][k]*B[k][j]);
      }
    }
  }
  
  return product;
}

/*Multiplies a row of a matrix by a scalar*/
void rowMul(double** mat, int mlength, int rowindex, double scalar){
  int i;
  for(i=0; i<mlength; i++){
    mat[rowindex][i] = mat[rowindex][i]*scalar;
  }
}

/*Divides a row of a matrix by a constant*/
void rowDiv(double** mat, int mlength, int rowindex, double divisor){
  int i;
  for(i=0; i<mlength; i++){
    mat[rowindex][i] = mat[rowindex][i]/divisor;
  }
}

/*Subtracts a row of a matrix from another row (row1-row2)*/
void rowSub(double** mat, int mlength, int row1, int row2){
  int i;
  for(i=0; i<mlength; i++){
    mat[row1][i] = mat[row1][i]-mat[row2][i];
  }
}

/*Multiplies row1 by a scalar, stores the result in temprow, subtracts temprow from row2 (row2 - row1*sc) */
void rowMulSub(double** mat, int mlength, int row1, int row2, double scalar){
  double* temp;
  int i;
  
  temp = (double*) calloc(mlength, sizeof(double));

  for(i=0; i<mlength; i++){
    temp[i] = mat[row1][i]*scalar;
  }
  
  for(i=0; i<mlength; i++){
    mat[row2][i] = mat[row2][i]-temp[i];
  }
}

/*Calculates a Matrix's inverse through Gauss-Jordan Elimination without row swaps*/
double** iMat(double** A, int length){
  double** imat = NULL;
  double** amat; /*augmented matrix*/
  int i, j;

  /*Construct augmented Matrix*/
  amat = (double**) malloc(length*sizeof(double*));
  for(i=0; i<length; i++){
    amat[i] = (double*) calloc(length*2, sizeof(double));
  }

  for(i=0; i<length; i++){
    for(j=0; j<length; j++){
      amat[i][j] = A[i][j];
    }
  }

  for(i=0; i<length; i++){
    for(j=length; j<length*2; j++){
      if(j-length == i){
        amat[i][j] = 1;
      }
    }
  }

  /*Perform row operations to get A inverse on amat*/
  for(i=0; i<length; i++){
    for(j=0; j<length; j++){
      if(amat[i][i] != 1){
        if(amat[i][i] < 1){
          rowMul(amat, length*2, i, 10.0);
        }
        if(amat[i][i] > 1){
          rowDiv(amat, length*2, i, amat[i][i]);
        }
        j--;
        continue;
      }
      if(j == i){continue;}/*ignore diagonal 1 values*/
      /*headaches inevitable, just create the seperate method.*/
      rowMulSub(amat, length*2, i, j, amat[j][i]);
    }
  }

  /*Construct inverted matrix imat*/
  imat = (double**) malloc(length*sizeof(double*));
  for(i=0; i<length; i++){
    imat[i] = (double*) calloc(length, sizeof(double));
  }

  /*Plug values from amat into imat (starting from index length)*/
  for(i=0; i<length; i++){
    for(j=length; j<length*2; j++){
      imat[i][j-length] = amat[i][j];
    }
  }

  return imat;
}

/*Transposes a matrix*/
double** tMat(double** A, int Arow, int Acol){
  double** tmat;
  int i, j;
  
  tmat = (double**) malloc(Acol*sizeof(double*));
  for(i=0; i<Acol; i++){
    tmat[i] = (double*) calloc(Arow, sizeof(double));
  }

  for(i=0; i<Arow; i++){
    for(j=0; j<Acol; j++){
      tmat[j][i] = A[i][j];
    }
  }
  
  return tmat;
}

int main(int argc, char** argv){
  char* trainfile = argv[1];
  char* predictfile = argv[2];
  FILE* fp = NULL;
  int K, N, M, i, j;/*K = cols/attributes, N = rows/examples*/
  double** W; /*W is K+1x1*/
  double** X; /*First col of X is all 1's for some reason, X is NxK+1*/
  double** Y; /*Y is Nx1*/
  double** Xt; /*for X^t*/
  double** temp; /*for X^t*X */
  double** itemp; /*temp's inverse pointer*/

  fp = fopen(trainfile, "r");
  fscanf(fp, "%d\n", &K);
  fscanf(fp, "%d\n", &N);
  fclose(fp);
  
  /*Allocate space for W, X, Y*/
  W = (double**) malloc((K+1)*sizeof(double*));
  for(i = 0; i<K+1; i++){
    W[i] = (double*) calloc(1, sizeof(double));
  }
  X = (double**) malloc(N*sizeof(double*));
  for(i = 0; i<N; i++){
    X[i] = (double*) calloc(K+1, sizeof(double));
  }
  Y = (double**) malloc(N*sizeof(double*));
  for(i = 0; i<N; i++){
    Y[i] = (double*) calloc(1, sizeof(double));
  }

  /*Populate X, Y*/
  gentMat(X, Y, trainfile);

  /*Compute W matrix*/
  Xt = tMat(X, N, K+1);
  temp = mMat(Xt, X, K+1, N, N, K+1);
  itemp = iMat(temp, K+1);
  release(temp);
  temp = mMat(itemp, Xt, K+1, K+1, K+1, N);
  W = mMat(temp, Y, K+1, N, N, 1);

  /*Generate new X & Y from predict datafile*/
  release(Y);
  release(X);
  fp = fopen(predictfile, "r");
  fscanf(fp, "%d\n", &M);
  fclose(fp);
  
  X = (double**) malloc(M*sizeof(double*));
  for(i = 0; i<M; i++){
    X[i] = (double*) calloc(K, sizeof(double));
  }
  Y = (double**) malloc(M*sizeof(double*));
  for(i = 0; i<M; i++){
    Y[i] = (double*) calloc(1, sizeof(double));
  }

  genpMat(K, M, X, predictfile);

  /*Compute Y matrix from dot product of X's rows and W (skip W0, add it at the end)*/
  for(i=0; i<M; i++){
    for(j=0; j<K; j++){
      Y[i][0]= Y[i][0] + (X[i][j] * W[j+1][0]);
    }
    Y[i][0] = Y[i][0] + W[0][0];
  }
  
  /*Print results*/
  for(i = 0; i<M; i++){
    printf("%0.0lf\n", Y[i][0]);
  }

  return 0;
}
