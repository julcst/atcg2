#version 330 core

layout (points) in;
layout (line_strip, max_vertices=2) out;

in VS_OUT 
{
	int instance;
    vec3 points[4];
}gs_in[];

vec3 hermite(float u, vec3 p0, vec3 p1, vec3 t0, vec3 t1)
{
    float u2 = u*u;
    float h0 = (1-u2) * (1+2*u);
    float h1 = u * (1-u) * (1-u);
    float h2 = -u2 * (1-u);
    float h3 = (3 - 2*u) * u2;
    vec3 p = h0 * p0 + h1 * t0 + h2 * t1 + h3 * p1;
    return p;
}

uniform int discretization;
uniform mat4 P;

void main()
{
    vec3 p0 = gs_in[0].points[0];
    vec3 t0 = gs_in[0].points[1] - p0;
    vec3 p1 = gs_in[0].points[3];
    vec3 t1 = p1 - gs_in[0].points[2];

    gl_Position = P * vec4(hermite(gs_in[0].instance/float(discretization), p0, p1, t0, t1),1);
    EmitVertex();
    gl_Position = P * vec4(hermite((gs_in[0].instance+1)/float(discretization), p0, p1, t0, t1),1);
    EmitVertex();
    EndPrimitive();
    
}