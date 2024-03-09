#ifdef GL_ES
precision mediump float;
#endif


varying vec3 vColor;


vec3 toSRGB(vec3 color)
{
    vec3 result;
    for (int i=0; i<3; ++i) {
        float value = color[i];
        if (value <= 0.0031308)
            result[i] = 12.92 * value;
        else
            result[i] = (1.0 + 0.055) * pow(value, 1.0/2.4) -  0.055;
    }

    return result;
}


void main(void)
{
    //gl_FragColor = vec4(vColor, 1.0);
    gl_FragColor = vec4(toSRGB(vColor), 1.0);
}