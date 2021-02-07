#version 120

// transform matrices to send the cliped vertex back
uniform mat4 p3d_ModelViewMatrix;
uniform mat4 p3d_ProjectionMatrix;

varying out vec3 bc; // send the barycenter coordinates to the fragment shader

// actual vertex position
attribute vec4 p3d_Vertex;
// barycenter coordinates provided inside the vertex
attribute vec3 barycenter;

void main() {
    gl_Position = p3d_ProjectionMatrix * p3d_ModelViewMatrix * p3d_Vertex;
    bc = barycenter;
}
