#version 450

layout (binding = 0) uniform UniformBufferObject {
    vec3 cameraPosition;
    vec4 cameraRotation;
    vec3 objectPosition;
    vec4 objectRotation;
    vec2 fieldOfView;
} ubo;

layout (location = 0) in vec3 inPosition;

layout (location = 0) out vec4 fragColor;

void main(){
    gl_PointSize = 1.0;
    gl_Position = vec4(inPosition.x, inPosition.y,0.0, 1.0);
    fragColor = vec4(1,0,0,1);
}