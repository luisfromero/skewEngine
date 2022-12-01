```c

#define _USE_MATH_DEFINES
template <typename T>
struct inputData{
    T *input0;
    T *input1;
    T *input2;
    T *input3;
};


#include <math.h>
#define MAX(a,b) ((a) > (b) ? (a) : (b))




// T can be float, double, ...
template <typename T>
class skewEngine {


    inputData<T> input;
    int dimx;
    int dimy;
    int N;
    double fAngle;
    int *target;
    double *weight;
    double torads=M_PI/180.0;

    // Regular or Transposed
    bool isT;
    // Below or above 90, is mirrored
    bool isM;

    int sectorType;
    T *output0=nullptr;
    T *output1=nullptr;
    T *output2=nullptr;
    T *output3=nullptr;
    int dim_i,dim_o,dim_l;

public:
    skewEngine(int dimx, int dimy, inputData<T> inData );
    virtual ~skewEngine();
    void skew(int angle);
    void deskew();
    void reduce(T **output);
    static inputData<T> prepare(T *inputD, int dimx, int dimy);
    void identity();
    void kernel( void (*function)(skewEngine<T> *));
    
    
    
    T *skewInput;
    T *skewOutput;
    int dim_skewx,dim_skewy;
    bool has0=false,has1=false,has2=false,has3=false;
    int skewHeight, skewWidth;
    int *first;
    int *last;
    int a;
    int newAngle;
    double scale;
    int offset;
    double skewness,iskewness;

};

// dimx, dimy, input has been initiated with arguments values
template<typename T>
skewEngine<T>::skewEngine(int dimx, int dimy, inputData<T> input) : dimx(dimx), dimy(dimy), input(input)
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

    weight=new double[dim_l]();
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
        weight[j] = drift - floor(drift); //weight is lower when drift is close to target
    }
    skewHeight=dim_o+target[dim_i-1]+1;//Excluded
    if(newAngle==0){skewHeight=dim_o;} //Needed?
    skewWidth=dim_i;

    //Set all to zero for clean draw
    //for(int i=0;i<2*N+dim_i;i++)skewInput[i]=skewOutput[i]=0;
    for(int i=0;i<=skewHeight*dim_i;i++)skewInput[i]=skewOutput[i]=0;
    for (int i = 0;i< skewHeight; i++) {
        first[i] = dim_i;
        last[i] = 0;
    }
    T *source=(T *)isT?(isM?input.input2:input.input1):(isM?input.input3:input.input0);
    for (int i = 0; i < dim_o; i++) {
        for (int j = 0; j < dim_i; j++) {
            int row = i + target[j];
            if(j<first[row])first[row]=j;
            if(j+1>last[row])last[row]=j+1;
            skewInput[row * dim_i + j] += (1.0 - weight[j]) * source[dim_i * i + j];
            row++;
            if(j<first[row])first[row]=j;
            if(j+1>last[row])last[row]=j+1;
            skewInput[row * dim_i + j] += weight[j] * source[dim_i * i + j];
        }
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

template<typename T>
void skewEngine<T>::kernel(void (*function)(skewEngine<T> *)) {return function(this);}



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

/**
 * Critical reduction of outputs from threads
 * Any thread can have data of any of four kind of sectors
 * @tparam T 
 * @param output 
 */
template<typename T>
void skewEngine<T>::reduce(T **output) {
    for (int i = 0; i < dimy; i++){
        int ci=dimy-1-i;
        for (int j = 0; j < dimx; j++) {
            int cj=dimx-1-j;
            if (has0)output[i][j] += output0[i * dimx + j];
            if (has1)output[i][j] += output1[cj * dimy +ci];
            if (has2)output[i][j] += output2[j * dimy +ci];
            if (has3)output[i][j] += output3[i * dimx +cj];
        }
    }
}

/**
 * Static method to create the 4 versions of input data: NN, TN, TM, NM
 * @tparam T 
 * @param inputD 
 * @param dimx 
 * @param dimy 
 * @return 
 */
template<typename T>
inputData<T> skewEngine<T>::prepare(T *inputD, int dimx, int dimy) {
    T *input0=inputD;
    T *input1=new float[dimx*dimy];
    T *input2=new float[dimx*dimy];
    T *input3=new float[dimx*dimy];

    // A transpose copy of input, which is computed once to improve locality
    for(int i=0;i<dimy;i++) {
        int ci=dimy-1-i;
        for (int j = 0; j < dimx; j++) {
            int cj=dimx - 1 - j;
            T val =
            input0[dimx * i +  j];
            input1[dimy *cj + ci] = val; //Blue arrow
            input2[dimy * j + ci] = val; //Black arrow
            input3[dimx * i + cj] = val;//Yellow arrow
        }
    }
    inputData<T> result;
    result.input0=input0;
    result.input1=input1;
    result.input2=input2;
    result.input3=input3;

    return result;

}

/**
 * use ->kernel(identity) instead of this
 * @tparam T 
 */
// Unused. Replaced by kernel(identity)
template<typename T>
void skewEngine<T>::identity() {
    for(int i=0;i<skewHeight;i++){
        int k=dim_i*i;
        for(int j=first[i];j<last[i];j++)
            skewOutput[k+j]=skewInput[k+j];
    }
}


```