kernel = """
#include <pyopencl-complex.h>   
__kernel void ml_decoder(__global cfloat_t* decoded, __global cfloat_t* r, __global cfloat_t* MatP, __global cfloat_t* MatQ, __global cfloat_t* MatAdjP, __global cfloat_t* QAMconstell, const int N,const int sizeQAM, const float frobNorm) {
  /* Pick any two symbols from the constellation. this is x3 and x4 */

  cfloat_t s_bar[2][1];
  cfloat_t c_bar[2][1];
  cfloat_t Pc[2][1];
  cfloat_t Qs[2][1];
  cfloat_t rQs[2][1];
  cfloat_t cbar_temp[2][1];
  cfloat_t tempMs[2][1];
  cfloat_t matq[2][2];
  cfloat_t decoded_sbar[2][1], decoded_cbar[2][1];
  float ceil1, ceil2, floor1, floor2;
  float temp = 10000.0 ;
  float Ms;
  for (int i = 0; i< 2; i++)
  {
    for (int j = 0; j < 2; j++)
    matq[i][j] = MatQ[i+j];
  }
  unsigned int indexr = get_global_id(0);
  for (int i = 0 ; i <  sizeQAM ; i++)
  {
    for(int j = 0 ; j < sizeQAM ; j++)
    {
           s_bar[0][0] = QAMconstell[i];
           s_bar[1][0] = QAMconstell[j]; 

           /* Calculating c_bar from s_bar */

           /* Multiplying Q and s_bar matrices */
           Qs[0][0] = cfloat_add(cfloat_mul(MatQ[0 + 0] , s_bar[0][0]), cfloat_mul( MatQ[0 + 1] , s_bar[1][0]));
           Qs[1][0] = cfloat_add(cfloat_mul(MatQ[0 + 2] , s_bar[0][0]), cfloat_mul( MatQ[0 + 3] , s_bar[1][0]));

           /* r - Qs */
           rQs[0][0] = cfloat_add(r[indexr + 0] ,-Qs[0][0]);
           rQs[1][0] = cfloat_add(r[indexr + N] , -Qs[1][0]);

           /* Calculate c_bar */
           cbar_temp[0][0] =  cfloat_add(cfloat_mul(MatAdjP[0 + 0] , rQs[0][0]) , cfloat_mul(MatAdjP[0 + 1] , rQs[1][0])) ;
           cbar_temp[1][0] =  cfloat_add(cfloat_mul(MatAdjP[0 + 2] , rQs[0][0]) , cfloat_mul(MatAdjP[1 + 3] , rQs[1][0])) ;
       c_bar[0][0] =  cfloat_mul((2/frobNorm) , cbar_temp[0][0]);
       c_bar[1][0] = cfloat_mul((2/frobNorm) , cbar_temp[1][0]);

       /* Multiplying P and c matrices */
       Pc[0][0] = cfloat_add(cfloat_mul(MatP[0 + 0], c_bar[0][0]), cfloat_mul(MatP[0 + 1], c_bar[1][0]));
       Pc[1][0] = cfloat_add(cfloat_mul(MatP[0 + 2], c_bar[0][0]), cfloat_mul(MatP[0 + 3], c_bar[1][0]));

       /* Calculate ceiling of c_bar[0][0] and floor of c_bar[1][0] */
       ceil1 = ceil(fabs(cfloat_real(c_bar[0][0])));
       floor1 = floor(fabs(cfloat_real(c_bar[1][0])));
       ceil2 = ceil(fabs(cfloat_imag(c_bar[0][0])));
       floor2 = floor(fabs(cfloat_real(c_bar[1][0])));

       c_bar[0][0] = ceil1 + 1j*ceil2;
       c_bar[1][0] = floor1 + 1i*floor2;

       /* Calculate Ms */
       /* First, we calculate the complex numbers' abs and then proceed with over all ||Ms|| calculation */
       tempMs[0][0] = cfloat_add(r[indexr + 0], cfloat_add(-Pc[0][0], -Qs[0][0]));
       tempMs[1][0] = cfloat_add(r[indexr + N], cfloat_add(-Pc[1][0], -Qs[1][0]));
       Ms = pow(sqrt(pow((cfloat_real(tempMs[0][0])),2) + pow((cfloat_imag(tempMs[0][0])),2)) + sqrt(pow((cfloat_real(tempMs[1][0])),2) + pow((cfloat_imag(tempMs[1][0])),2)),2);
       
       /* Check if Ms < temp, if TRUE, then store Ms in temp and store decoded_sbar and decoded_cbar with their respective values */
       if (Ms < temp)
       {
        decoded_cbar[0][0] = c_bar[0][0];
        decoded_cbar[1][0] = c_bar[1][0];
        decoded_sbar[0][0] = s_bar[0][0];
        decoded_sbar[1][0] = s_bar[1][0];
        temp = Ms;
       }
    }
  }
  decoded[2*indexr] = decoded_cbar[0][0];
  decoded[2*indexr+1] = decoded_cbar[1][0];
  decoded[2*indexr + 2*N] = decoded_sbar[0][0];
  decoded[2*indexr + 2*N +1] = decoded_sbar[1][0];
}
"""