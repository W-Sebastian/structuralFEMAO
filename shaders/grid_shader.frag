#version 120

uniform vec3 color; // the desired color of the element
varying in vec3 bc; // barycentric coordinates for the current pixel

const float lineWidth = 1; // thickness for the wireframe; if it is too small some will disappear

float edgeFactor() {
    vec3 d = fwidth(bc);
    vec3 f = step(d * lineWidth, bc);
    return min(f.x, f.z);
}

void main() {
    gl_FragColor = vec4(min(vec3(edgeFactor()), color), 1.0);
}
