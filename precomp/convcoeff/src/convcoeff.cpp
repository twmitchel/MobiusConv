#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <fstream>
#include <string> 

#include "CmdLineParser.h"

#ifndef M_PI
#define M_PI		3.14159265358979323846
#endif 

cmdLineString savePath( "savePath" );
cmdLineInt BandLimit( "B") , FilterBandLimit("M");

cmdLineReadable* params[] = { &savePath , &BandLimit, &FilterBandLimit, NULL };

         
std::pair<double, double> triHalfRecur(int l, int m, int n)
{
   double denom = (-1.0 + l)*std::sqrt( (l-m)*(l+m)*(l-n)*(l+n));
   
   double c1 = (1.0 - 2.0*l)*m*n / denom;
   
   double c2 = -1.0*l*std::sqrt( double( ( (l-1)*(l-1) - m*m) * ( (l-1)*(l-1) - n*n) ) )/denom;

   
   return std::pair<double, double>(c1, c2);
}

std::pair<double, double> epspq(int p, int q)
{
   
   if ( (p + q) == 1)
   {
      return std::pair<double, double>(0.0, -1.0 * M_PI / 2.0);
   }
   else if ( (p + q) == -1 )
   {
      return std::pair<double, double>(0.0,  M_PI / 2.0);
   }
   else
   {
      return std::pair<double, double> ( -1.0 * (1.0 + std::pow(-1.0, std::abs(p+q))) / ( (1.0 + p + q) * (p + q - 1.0) ), 0.0 );
   }
   
}

template<class Real>
std::vector<std::vector<std::vector<Real>>> generateLittleHalf (int B)
{
   
   std::vector<std::vector<std::vector<Real>>> d;

   d.resize(B);
   
   for (int l = 0; l < B; l++)
   {
      d[l].resize(2*B + 1);
      
      for (int m = 0; m < 2*B + 1; m++)
      {
         d[l][m].resize(2*B + 1, 0.0);
      }
   }
   
   Real sqrt2 = (Real) std::sqrt(2.0);
   
   // Fill out first two levels
   d[0][B-1][B-1] = (Real) 1.0;
   
   d[1][-1 +(B-1)][-1 + (B-1)] = 0.5;
   d[1][ -1 +(B-1)][ B-1] = 1.0 / sqrt2;
   d[1][ -1 +(B-1)][ B] = 0.5;
        
   d[1][ (B-1)][ -1 + (B-1)] = -1.0 / sqrt2;
   d[1][ (B-1)][ B] = 1.0 / sqrt2;
        
   d[1][ B][ -1 + (B-1)] = 0.5;
   d[1][ B][ (B-1)] = -1.0 / sqrt2;
   d[1][ B][ B] = 0.5;
   
   // Fill out the rest through recursion
   
   for (int l = 2; l < B; l++)
   {
      for (int m = 0; m < l; m++)
      {
         for (int n = 0; n < l; n++)
         {
         
            if ( (m == 0) && (n == 0) )
            {
               d[l][B-1][B-1] = (Real) -1.0 *  (double(l-1.0)/l) * d[l-2][B-1][B-1];
               
            }
            else
            {
               std::pair<double, double> c12 = triHalfRecur(l, m, n);
               
               d[l][m + (B-1)][n + (B-1)] = (Real) c12.first * d[l-1][ m + (B-1)][ n + (B-1)] + c12.second * d[l-2][m + (B-1)][n + (B-1)];
               
            }
            
         }
      }
      
      for (int m = 0; m <= l; m++)
      {
         Real lnV = (Real) 0.5 * ( std::lgamma( 2*l + 1) - std::lgamma(l+m + 1)  - std::lgamma(l - m + 1) )  - l * std::log(2.0);
         
         d[l][m+(B-1)][l+(B-1)] = (Real) std::exp(lnV);
         
         d[l][l+(B-1)][m + (B-1)] = (Real) std::pow(-1.0, std::abs(l - m)) * std::exp(lnV);
         
      }
      
      for (int m = 0; m <= l; m++)
      {
         for (int n = 0; n <= l; n++)
         {
            Real val = d[l][ m+(B-1)][ n+(B-1)];
            
        
            d[l][-m + (B-1)][-n + (B-1)] = (Real) std::pow(-1.0, std::abs(m-n)) * val;
            d[l][-m + (B-1)][n + (B-1)] = (Real) std::pow(-1.0, std::abs(l-n)) * val;
            d[l][m + (B-1)][-n + (B-1)] = (Real) std::pow(-1.0, std::abs(l+m)) * val;
         }
      }
      
   }
   
   return d;
            
}



template< class Real >
int run( void )
{

   int B = BandLimit.value;
   int M = FilterBandLimit.value;
   int threads = omp_get_num_threads();
   std::string saveP = savePath.value;
   
   std::vector<std::vector<std::vector<Real>>> d = generateLittleHalf<Real>(B);
   
   
   std::vector<Real> deltaRe;
   std::vector<Real> deltaIm;
   
   std::vector<std::vector<int>> ind;
   
   
   for (int m = -M; m <= M; m++)
   {
      for (int k = -(B-1); k < B; k++)
      {     
         int s0 = std::max(std::abs(m), std::abs(k));
         int l0 = std::abs(k);
         
         for (int s = s0; s < B; s++)
         {
            for (int l = l0; l < B; l++)
            {
               std::vector<int> index;
               index.push_back(l);
               index.push_back(s);
               index.push_back(k);
               index.push_back(m);
               
               ind.push_back(index);
            }
         }
      }
   }
   
   int numInd = (int) ind.size();
   
   int checkPoint = (int) std::round(numInd / 10.0);
   
   deltaRe.resize(ind.size(), 0.0);
   deltaIm.resize(ind.size(), 0.0);
   
   int prog = 0;

   #pragma omp parallel for
   for (int i = 0; i < ind.size(); i++)
   {
      int m = ind[i][3];
      int k = ind[i][2];
      int s = ind[i][1];
      int l = ind[i][0];
      
      
      Real delRe = 0.0;
      Real delIm = 0.0;
      
      for (int p = -l; p <= l; p++)
      {
         for (int r = -s; r <= s; r++)
         {
         
            std::pair<double, double> ep = epspq(p, r);
            
            double coeff = d[l][p+(B-1)][-k + (B-1)] * d[l][p+(B-1)][B-1] * d[s][r+(B-1)][k + (B-1)] * d[s][r+(B-1)][m + (B-1)];
            
            delRe += coeff * ep.first;
            delIm += coeff * ep.second;
            
         }
      }
      
      double c = std::pow(-1.0, std::abs(k)) * std::sqrt( (2.0 * l + 1) * M_PI);
      
      delRe *= c;
      delIm *= c;
      
      if ( (m % 4) == 0 )
      {
         deltaRe[i] = delRe ;
         deltaIm[i] =  delIm;
      }
      else if ( (m % 4) == 1 )
      {
         deltaRe[i] = delIm;
         deltaIm[i] = -1.0*delRe;
      }
      else if ( (m % 4) == 2 )
      {
         deltaRe[i] = -1.0 * delRe;
         deltaIm[i] = -1.0 * delIm;
      }
      else
      {
         deltaRe[i] = -1.0*delIm;
         deltaIm[i] = delRe;
      }
      
      #pragma omp critical
      {
         prog += 1;

         if (prog % checkPoint == 0)               
         {
            printf("%f complete\n", ((float) prog) / numInd);
            fflush(stdout);
         }
      }
      
     
   }     
               
 
   // Save file
   
   std::string saveFile = saveP + "/convCoeff_" + std::to_string(B) + "_" + std::to_string(M) + ".txt";


   std::ofstream vF;
   vF.open ( saveFile.c_str () );

   for ( int i = 0; i < numInd; i++)
   {
      vF << ind[i][0] << " " << ind[i][1] << " " << ind[i][2] << " " << ind[i][3] << " " << deltaRe[i] << " " << deltaIm[i] << std::endl;

   }
   
   vF.close ();
}
                  


int main( int argc , char* argv[] )
{
   std::vector< std::string > tempStrings;
	cmdLineParse( argc , argv , params , tempStrings );

   run<double>();
   
   return 1;
   
}


