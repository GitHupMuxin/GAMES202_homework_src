attribute vec3 aVertexPosition;
attribute vec3 aNormalPosition;
attribute mat3 aPrecomputeLT;


uniform mat4 uProjectionMatrix;
uniform mat4 uModelMatrix;
uniform mat4 uViewMatrix;

uniform mat3 uPrecomputeL[3];

varying highp vec3 vNormal;
varying highp vec3 vColor;

float dotProduction(mat3 precomputeL, mat3 precomputeLT)
{
    float result = 0.0;
    for (int i = 0; i < 3; i++)
    {
        vec3 vecPreL = precomputeL[i];
        vec3 vecPreLT = precomputeLT[i];
        result += dot(vecPreL, vecPreLT);
    }
    return result;
}


void main(void)
{
    vNormal = (uModelMatrix * vec4(vNormal, 1.0)).xyz;


    for (int i = 0; i < 3; i++)
    {
        vColor[i] = dotProduction(uPrecomputeL[i], aPrecomputeLT);  
    }

    gl_Position = uProjectionMatrix * uViewMatrix * uModelMatrix * vec4(aVertexPosition, 1.0);
}