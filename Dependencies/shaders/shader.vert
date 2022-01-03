#version 450
#define M_PI 3.141592653

layout (binding = 0) uniform UniformBufferObject {
    vec3 cameraPosition;
    vec4 cameraRotation;
    vec3 objectPosition;
    vec4 objectRotation;
    vec2 fieldOfView;
} ubo;

layout (location = 0) in vec3 inPosition;

layout (location = 0) out vec4 fragColor;

vec3 amount;
vec3 displacement;

vec3 getPositionfromOrigin(vec3 pos, vec4 rotation){
    vec3 zUnit = rotation.xyz;

    vec3 xUnit, yUnit;
    float xNormFactor, yNormFactor;

    xNormFactor = rotation.z / sqrt(pow(rotation.x,2) + pow(rotation.z,2));
    yNormFactor = rotation.z / sqrt(pow(rotation.y,2) + pow(rotation.z,2));

    xUnit = vec3(xNormFactor ,0,-rotation.x/rotation.z * xNormFactor);
    yUnit = vec3(0,yNormFactor, -rotation.y/rotation.z * yNormFactor);

    if(rotation.z <0 && rotation.y !=0){
        xUnit *=-1;
    }
    

    vec3 xx = xUnit * cos(rotation.w);
    vec3 xy = yUnit * sin(rotation.w);
    vec3 yx = -xUnit * sin(rotation.w);
    vec3 yy = yUnit * cos(rotation.w);

    xUnit = xx + xy;
    yUnit = yx + yy;

    
    return xUnit * pos.x + yUnit * pos.y + zUnit * pos.z;
}

vec3 solve(vec4 rotation, vec3 displacement) {
    vec3 relativeSpace;
    // For explanation, picture a sphere
    // For default: D = 1, Dx = 0, Dy = 0, Dz = 5
    float xz = sqrt(pow(rotation.x, 2) + pow(rotation.z, 2));
    float yz = sqrt(pow(rotation.y, 2) + pow(rotation.z, 2));
    
    float cosy = rotation.z / yz;
    float cosx = rotation.z / xz;

    float sinx = rotation.x / xz;
    float siny = rotation.y / yz;

    relativeSpace.x = displacement.x * rotation.z 
    relativeSpace.y = displacement.y * cosy + displacement.x * siny * sinx + displacement.z * siny * cosx;
    relativeSpace.z = displacement.z * cosx * cosy + displacement.x * cosx * siny + displacement.y * siny;
    return relativeSpace;
    //
}

void main(){
    
    vec3 relativePos = getPositionfromOrigin(inPosition, ubo.objectRotation);

    displacement = ubo.objectPosition - ubo.cameraPosition + relativePos;

    vec3 relativeSpace = solve(ubo.cameraRotation, displacement);

    if (relativeSpace.z <0){
        relativeSpace.z = 0.001;
        fragColor = vec4(0,0.3,0,0);
    }else{
        fragColor = vec4(0,0.3,0,1);
    }

    vec2 screenSpace = vec2(relativeSpace.x/relativeSpace.z / ubo.fieldOfView.x ,-relativeSpace.y/relativeSpace.z / ubo.fieldOfView.y);

    
    gl_Position = vec4(screenSpace,0.0, 1.0);

}
