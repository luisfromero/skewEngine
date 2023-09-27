/**
 * @file
 * @author Felipe Romero
 * @brief Functions for the creation and manipulation of color maps, HSV or RGB data, etc.
 */
//
// Created by Felipe on 17-11-22.
//
#include <cmath>
#include "color.h"
extern color *my_palette;


float interp(float val, float y0, float x0, float y1, float x1) {

    return (val - x0) * (y1 - y0) / (x1 - x0) + y0;
}

float base(float val) {

    if (val <= -0.75) return 0;
    else if (val <= -0.25) return interp(val, 0.0, -0.75, 1.0, -0.25);
    else if (val <= 0.25) return 1.0;
    else if (val <= 0.75) return interp(val, 1.0, 0.25, 0.0, 0.75);
    else return 0.0;
}


color get_color_jet2(int index) {

    float v = index;
    float dv = 1024;
    color c = {255, 255, 255};
    if (v < (0.25 * dv)) {
        //Del azul va hacia el verde, primero manteniendo azul 100%
        c.R = 0;
        c.G = 256 * (4 * v / dv);
    }
    else if (v < (0.5 * dv)) { 					//El azul desaparece gradualmente y queda solo verde
        c.R = 0;
        c.B = 255 * (1 + 4 * (0.25 * dv - v) / dv) ;
    }
    else if (v < (0.75 * dv)) {
        c.R = 256 * ( 4 * (v - 0.5 * dv) / dv);
        c.B = 0;
    }
    else {
        c.G = 255 * (1 + 4 * (0.75 * dv - v) / dv);
        c.B = 0;
    }

    return c;
}

color * build_palette() {

    color *cm = new color[1024];
    for (int i = 0; i < 1024; i++)
        cm[i] = get_color_jet2(i);
    return cm;
}


float red(float gray) {

    return 256 * base(gray - 0.5);
}

float green(float gray) {

    return 256 * base(gray);
}

float blue(float gray) {

    return 256 * base(gray + 0.5);
}


rgbColor HSVtoRGB(hsvColor hsv){
    float H= hsv.H;
    float S= hsv.S;
    float V= hsv.V;
    if(H>360 || H<0 || S>100 || S<0 || V>100 || V<0)
        return {0,0,0};
    float s = S/100;
    float v = V/100;
    float C = s*v;
    float X = C*(1-abs(fmod(H/60.0, 2)-1));
    float m = v-C;
    float r,g,b;
    if(H >= 0 && H < 60){
        r = C,g = X,b = 0;
    }
    else if(H >= 60 && H < 120){
        r = X,g = C,b = 0;
    }
    else if(H >= 120 && H < 180){
        r = 0,g = C,b = X;
    }
    else if(H >= 180 && H < 240){
        r = 0,g = X,b = C;
    }
    else if(H >= 240 && H < 300){
        r = X,g = 0,b = C;
    }
    else{
        r = C,g = 0,b = X;
    }

    int R = (r+m)*255;
    int G = (g+m)*255;
    int B = (b+m)*255;
    return {static_cast<unsigned char>(R),static_cast<unsigned char>(G),static_cast<unsigned char>(B)};
}
