```c
#define _USE_MATH_DEFINES

#include <math.h>
#define MAX(a,b) ((a) > (b) ? (a) : (b))




// T can be float, double, ...
template <typename T>
class skewEngine {
    int dimx;
    int dimy;
    int N;
    T fAngle;
    int *target;
    T *weight;
    double torads=M_PI/180.0;

    // Regular or Transposed
    bool isT;
    // Below or above 90, is mirrored
    bool isM;
public:
    skewEngine(int dimx, int dimy, T *input0, T *input1, T *input2, T *input3);
    virtual ~skewEngine();
    void skew(int angle);
    void deskew();
    
    int sectorType;
    T *input0;
    T *input1;
    T *input2;
    T *input3;
    T *output0=nullptr;
    T *output1=nullptr;
    T *output2=nullptr;
    T *output3=nullptr;
    T *skewInput;
    T *skewOutput;
    int dim_i,dim_o,dim_l,dim_skewx,dim_skewy;
    bool has0=false,has1=false,has2=false,has3=false;
    int skewHeight;
    int *first;
    int *last;
    int a;
    int newAngle;
    T scale;
    int offset;
    double skewness,iskewness;

};

// dimx, dimy, input has been initiated with arguments values
template<typename T>
skewEngine<T>::skewEngine(int dimx, int dimy, T *input0, T *input1, T *input2, T *input3) : dimx(dimx), dimy(dimy), input0(input0), input1(input1), input2(input2), input3(input3)
{
    N=dimx*dimy;
    fAngle=(T)atan2(dimy,dimx)*180/M_PI;

    //There will be an object of this class per thread, so,
    //this object will reuse intermediate storage for several angles

    //fAngle is a frontier:
    //Any angle in range [-bAngle,+bAngle] will use Normal (suffix N) storage, while...
    //any angle in range ]+bAngle,-bAngle[ will use Transposed (suffix T) storage

    //Largest dimension
    dim_l=MAX(dimx,dimy);
    
    weight=new T[dim_l]();
    target=new int[dim_l]();
    first=new int[2*dim_l]();
    last=new int[2*dim_l]();
    skewInput=new T[N*2+dim_l];
    skewOutput=new T[N*2+dim_l];
}

template<typename T>
skewEngine<T>::~skewEngine() {
    delete[] skewInput;
    delete[] skewOutput;
    delete[] target;
    delete[] weight;
    if(has0)delete[] output0;
    if(has1)delete[] output1;
    if(has2)delete[] output2;
    if(has3)delete[] output3;
    delete[] first;
    delete[] last;
}

/**
 * Remap {image/dem/x-y data/etc.}  to its skewed position, calculates weights, and limits
 * Remenber: any pixel will be mapped to a couple of pixels, using the weights w and 1-w;
 * @tparam T
 * @param angle
 */
template<typename T>
void skewEngine<T>::skew(int angle){
    a=angle;
    isT= angle > fAngle && a < 180-fAngle;
    isM= angle > 90;
    if(!isT&&!isM){has0=true;sectorType=0;newAngle=a;}
    if( isT&&!isM){has1=true;sectorType=1;newAngle=90-a;}
    if( isT&& isM){has2=true;sectorType=2;newAngle=a-90;}
    if(!isT&& isM){has3=true;sectorType=3;newAngle=180-a;}

    scale=1/pow(cos(torads*newAngle),2);
    dim_skewx=isT?dimy:dimx;
    dim_skewy=isT?2*dimx:2*dimy;

    // skew coefficient
    //skewness=isT?1/tan(angle*torads):tan(angle*torads);
    //if(isM) skewness = -skewness;
    skewness=tan(newAngle*torads);

    iskewness=1/skewness;

    dim_i=isT?dimy:dimx; //inner dimension
    dim_o=isT?dimx:dimy; //outer dimension

    offset=dim_i  * skewness;
    // Reset data

    for (int j = 0; j < dim_i; j++) {
        double drift = skewness * j;
        target[j] = drift;
        weight[j] = drift - target[j]; //weight is lower when drift is close to target
    }
    skewHeight=dim_o+target[dim_i-1]+1;//Excluded
    //for(int i=0;i<*N+dim_2i;i++)skewInput[i]=skewOutput[i]=0;
    for(int i=0;i<=skewHeight*dim_i;i++)skewInput[i]=skewOutput[i]=0;

    T *source=isT?(isM?input2:input1):(isM?input3:input0);
    for (int i = 0; i < dim_o; i++) {
        for (int j = 0; j < dim_i; j++) {
            int row = i + target[j];
            skewInput[row * dim_i + j] += (1.0 - weight[j]) * source[dim_i * i + j];
            skewInput[(row + 1) * dim_i + j] += weight[j] * source[dim_i * i + j];
        }
        //test

    }
    if(newAngle==0){skewHeight=dim_o;}
    for (int i = 0;i< skewHeight; i++) {
        first[i]=0;
        last[i]=dim_i;
        if(i<offset)last[i]=(i+1)*iskewness+1; //excluded
        if(i>dim_o)first[i]=(i-dim_o)*iskewness+1;//
        //skewInput[i * dim_i + first[i]] = 160;
        //skewInput[i * dim_i + last[i]-1] = 191;
    }
}

/**
 * An intermediate (external) processing function will perform a line by line operation over skewed data
 * to generate skewed result:
 *
 * for(int i=lowerl; i< upperl; i++) //embarrassing parallel loop -> gpu candidate
 *    intensive CPU kernel for aligned data:
 *    for(int j=leftl(i); j< rightl(i); j++) ... skewOutput[i] = function(skewOutput[j])
 */




/**
 * After processing aligned data in the external algorithm (for example, total viewshed kernel,
 * this function maps-back skewed results (outputSkew)to its original position (output).
 * Now, the final pixel in output is a weighted average of skewed output data
 * @tparam T
 */
template<typename T>
void skewEngine<T>::deskew(){
    // https://stackoverflow.com/questions/7546620/operator-new-initializes-memory-to-zero
    // Un thread crea la estructura sólo si la va a necesitar
    if(has0&&output0== nullptr)output0=new T[N]();
    if(has1&&output1== nullptr)output1=new T[N]();
    if(has2&&output2== nullptr)output2=new T[N]();
    if(has3&&output3== nullptr)output3=new T[N]();
    T *output;
    if(sectorType==0)output=output0;
    if(sectorType==1)output=output1;
    if(sectorType==2)output=output2;
    if(sectorType==3)output=output3;
    for(int i=0;i<dim_o;i++)
        for(int j=0;j<dim_i;j++)
            output[i*dim_i+j]+=
                    (1-weight[j])*skewOutput[(i+target[j])*dim_i+j]+
                    weight[j]*skewOutput[(i+target[j]+1)*dim_i+j];

}

```