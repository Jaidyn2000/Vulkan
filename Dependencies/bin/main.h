// This file includes: defines, includes, usings, functions and structs

#pragma once
#define _USE_MATH_DEFINES
#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#include "glfw3.h"
#define GLFW_EXPOSE_NATIVE_WIN32
#include <glfw3native.h>

#include "math.h"
#include <chrono>
#include <thread>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <cerrno>
#include <map>
#include <array>
#include <vector>
#include <optional>
#include <set>
#include <cstdint>
#include <algorithm>
#include <Windows.h>

#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif

const int MAX_FRAMES_IN_FLIGHT = 2;

static std::vector<char> readFile(const std::string& fileName){
    std::ifstream file(fileName, std::ios::ate | std::ios::binary);

    if(!file.is_open()){
        throw std::runtime_error("failed to open file");
    }

    size_t fileSize = (size_t) file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();

    return buffer;
}

VkResult CreateDebugMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger){
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if(func != nullptr){
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    }else{
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType,
const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData){
    if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT){
        std::cerr << "Validation layer: " << pCallbackData->pMessage << std::endl;
    }

    return VK_FALSE;
}

std::vector<const char*> getRequiredExtensions(){
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    if (enableValidationLayers){
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions;
}

bool checkValidationLayerSupport(const std::vector<const char*> validationLayers){

    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char* layerName : validationLayers){
        bool layerFound = false;

        for (const auto& layerProperties : availableLayers){
            if (strcmp(layerName, layerProperties.layerName) == 0){
                layerFound = true;
                break;
            }
        }
        if (!layerFound){
            return false;
        }
    }
    return true;
}

struct vec2 {
    float x;
    float y;

    vec2(){
        x = 0;
        y = 0;
    }
    vec2(float x, float y){
        this->x = x;
        this->y = y;
    }

    vec2 operator +(vec2 v){
        x += v.x;
        y += v.y;
        return vec2(x,y);
    }
};

struct vec3 {
    float x;
    float y;
    float z;

    vec3(){
        x = 0;
        y = 0;
        z = 0;
    }
    vec3(float x, float y, float z){
        this->x = x;
        this->y = y;
        this->z = z;
    }
    
    vec3 operator +(vec3 v){
        x += v.x;
        y += v.y;
        z += v.z;
        return vec3(x,y,z);
    }
    vec3 operator +=(vec3 v){
        x += v.x;
        y += v.y;
        z += v.z;
        return vec3(x,y,z);
    }

    vec3 operator -(vec3 v){
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return vec3(x,y,z);
    }
    vec3 operator -=(vec3 v){
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return vec3(x,y,z);
    }

    vec3 operator *(float k){
        x *= k;
        y *= k;
        z *= k;
        return vec3(x,y,z);
    }
    vec3 operator *=(float k){
        x *= k;
        y *= k;
        z *= k;
        return vec3(x,y,z);
    }

    float norm(){
        return sqrt(x*x + y*y + z*z);
    }
};

struct vec4 {
    float x, y, z, w;

    vec4() {
        this->w = 0;
        this->x = 0;
        this->y = 0;
        this->z = 1;
    }
    vec4(float w, float x, float y, float z) {  
        this->w = w;
        this->x = x;
        this->y = y;
        this->z = z;
    }
    vec3* GetUnitVectors() {
        vec3* unitVectors = new vec3[3];

        vec3 zUnit = vec3(this->x, this->y, this->z);


        float cosProject = (float)cos(M_PI * this->w);
        float sinProject = (float)sin(M_PI * this->w);
        vec3 xUnit, yUnit;

        if (this->z != 0) {
            xUnit = vec3(1, 0, (this->x) / this->z);
            yUnit = vec3(0, 1, (this->y) / this->z);
        }
        else {
            xUnit = vec3(1, 0, 0);
            yUnit = vec3(0, 1, 0);
        }

        vec3 xx = vec3(xUnit);
        vec3 xy = vec3(yUnit);
        vec3 yx = vec3(xUnit);
        vec3 yy = vec3(yUnit);

        xx *= cosProject;
        xy *= sinProject;
        yx *= sinProject;
        yy *= cosProject;

        xUnit = xx + xy;
        yUnit = yx + yy;

        xUnit *= 1 / xUnit.norm();
        yUnit *= 1 / yUnit.norm();

        unitVectors[0] = xUnit;
        unitVectors[1] = yUnit;
        unitVectors[2] = zUnit;
        return unitVectors;
    }

    // Give angle in radians for a desired rotation
    // This function converts an axis and an angle into a vec4
    vec4 getFromVector(vec3 v, float angle) {
        vec4 out;
        out.w = cos(angle / 2);
        out.x = v.x * sin(angle / 2);
        out.y = v.y * sin(angle / 2);
        out.z = v.z * sin(angle / 2);

        return out;
    }

    vec4 operator *(vec4 q1) {
        vec4 out;
        out.w = q1.w * w - q1.x * x - q1.y * y - q1.z * z;
        out.x = q1.w * x + w * q1.x + q1.y * z - q1.z * y;
        out.y = q1.w * y + w * q1.y - q1.x * z + q1.z * x;
        out.z = q1.w * z + w * q1.z + q1.x * y - q1.y * x;
        return out;
    }

    vec4 operator *=(vec4 q1) {
        vec4 out;
        out.w = q1.w * w - q1.x * x - q1.y * y - q1.z * z;
        out.x = q1.w * x + w * q1.x + q1.y * z - q1.z * y;
        out.y = q1.w * y + w * q1.y - q1.x * z + q1.z * x;
        out.z = q1.w * z + w * q1.z + q1.x * y - q1.y * x;
        return out;
    }

    vec4 conjugate() {
        vec4 out;
        out.w = this->w;
        out.x = -this->x;
        out.y = -this->y;
        out.z = -this->z;
        return out;
    }
    vec4 getVectorQuaternion() {
       vec4 out;
        out.w = 0;
        out.x = this->x;
        out.y = this->y;
        out.z = this->z;
        return out;
    }
};
struct Transform {
    vec3 position, scale;
    vec4 rotation;
    vec3 xUnit, yUnit, zUnit;


    Transform() {
        this->position = vec3();
        this->rotation =vec4();
        this->scale = vec3(1, 1, 1);

        vec3* unitVectors = new vec3[3];
        unitVectors = this->rotation.GetUnitVectors();
        this->xUnit = unitVectors[0];
        this->yUnit = unitVectors[1];
        this->zUnit = unitVectors[2];

    }

    Transform(vec3 position) {
        this->position = position;
        this->rotation =vec4();
        this->scale = vec3(1, 1, 1);

        vec3* unitVectors = new vec3[3];
        unitVectors = this->rotation.GetUnitVectors();
        this->xUnit = unitVectors[0];
        this->yUnit = unitVectors[1];
        this->zUnit = unitVectors[2];
    }

    Transform(vec3 position,vec4 rotation) {
        this->position = position;
        this->rotation = rotation;
        this->scale = vec3(1, 1, 1);

        vec3* unitVectors = new vec3[3];
        unitVectors = this->rotation.GetUnitVectors();
        this->xUnit = unitVectors[0];
        this->yUnit = unitVectors[1];
        this->zUnit = unitVectors[2];
    }

    Transform(vec3 position,vec4 rotation, vec3 scale) {
        this->position = position;
        this->rotation = rotation;
        this->scale = scale;

        vec3* unitVectors = new vec3[3];
        unitVectors = this->rotation.GetUnitVectors();
        this->xUnit = unitVectors[0];
        this->yUnit = unitVectors[1];
        this->zUnit = unitVectors[2];
    }

    void Move(vec3 delta) {
        position += delta;
    }
    void Scale(vec3 delta) {
        this->scale += delta;
    }
    // Give angles in radians
    // zUnit is the unit vector of the sphere, xUnit and yUnit are the perpendicular vectors based on zUnit and w.
    // rotation is always in the form of a vector quaternion (we are applying p' = qp(q^-1)) (which for unit quaternions always equals the conjugate)
    // We store w but do not include it in the rotation calculation (it is only used to determine xUnit and yUnit)
    void Rotate(vec3 angles) {
       vec4 q;
       vec4 qOut;
        qOut = this->rotation.getVectorQuaternion();

        vec3 angleX = this->xUnit;
        vec3 angleY = this->yUnit;

        angleX *= angles.x;
        angleY *= angles.y;
        vec3 rotUnitVector = angleX + angleY;

        float rotVectorNorm = rotUnitVector.norm();
        if (rotVectorNorm != 0) {
            rotUnitVector *= 1 / rotVectorNorm;
        }

        float angleNorm = (float)sqrt(pow(angles.x, 2) + pow(angles.y, 2));

        q = q.getFromVector(rotUnitVector, angleNorm);

        // ORDER OF MULTIPLICATION OF QUATERNIONS MATTERS!!!!
        qOut = q * qOut; // For example, you cannot simply state that qOut *= q
        qOut = qOut * q.conjugate();

        qOut.w = this->rotation.w + angles.z;

        this->rotation = qOut;

        vec3* unitVectors = new vec3[3];
        unitVectors = this->rotation.GetUnitVectors();
        this->xUnit = unitVectors[0];
        this->yUnit = unitVectors[1];
        this->zUnit = unitVectors[2];
    }

    vec3 solve(vec4 rotation, vec3 displacement) {
        float xAmount, yAmount, zAmount;
        // For explanation, picture a sphere

        float xz = (float)sqrt(pow(rotation.x, 2) + pow(rotation.z, 2));
        float yz = (float)sqrt(pow(rotation.y, 2) + pow(rotation.z, 2));

        float x = displacement.x;
        float y = displacement.y;
        float z = displacement.z;

        float cosx = rotation.z / xz;
        float cosy = rotation.z / yz;
        float sinx = -rotation.x / xz;
        float siny = rotation.y / yz;

        xAmount = z * sinx + x * cosx + y * siny * sinx;
        yAmount = y * cosy + x * siny * sinx + z * siny * cosx;
        zAmount = z * cosx + x * sinx + y * siny * sinx;
        return vec3(xAmount, yAmount, zAmount);
        //
    }
};
struct Camera {
    Transform transform;
    float fieldOfView, vertFOV;
    int xRes, yRes;
    Camera() {
        this->transform = Transform(vec3(0,0,0));
        this->fieldOfView = 11 * (float)M_PI / 18;
        this->xRes = 1920;
        this->yRes = 1080;
        this->vertFOV = atan(tan(fieldOfView / 2) * yRes / xRes) * 2;
    }
    Camera(Transform transform) {
        this->transform = transform;
        this->fieldOfView = 11 * (float)M_PI/ 18;
        this->xRes = 1920;
        this->yRes = 1080;
        this->vertFOV = atan(tan(fieldOfView / 2) * yRes / xRes) * 2;
    }


    GLFWwindow* createScreen() {
        //glfwGetPrimaryMonitor();
        GLFWwindow* window = glfwCreateWindow(xRes, yRes, "VulkanEngine", glfwGetPrimaryMonitor(), NULL);
        return window;
    }

    void updateUnitVectors() {
        vec3* unitVectors = new vec3[3];
        unitVectors = this->transform.rotation.GetUnitVectors();
        this->transform.xUnit = unitVectors[0];
        this->transform.yUnit = unitVectors[1];
        this->transform.zUnit = unitVectors[2];
    }
};

struct Vertex2D{
    vec2 pos;

    static VkVertexInputBindingDescription getBindingDescription(){
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex2D);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription,1> getAttributeDescriptions(){
        std::array<VkVertexInputAttributeDescription,1> attributeDescriptions{};
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex2D,pos);

        return attributeDescriptions;
    }
};

struct Vertex{
    vec3 pos;

    static VkVertexInputBindingDescription getBindingDescription(){
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription,1> getAttributeDescriptions(){
        std::array<VkVertexInputAttributeDescription,1> attributeDescriptions{};
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex,pos);

        return attributeDescriptions;
    }
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;

    SwapChainSupportDetails querySwapChainSupport (VkPhysicalDevice device, VkSurfaceKHR surface){
        SwapChainSupportDetails details;

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

        if (formatCount !=0){
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
        }

        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

        if (presentModeCount !=0){
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());          
        }

        return details;
    }

    VkSurfaceFormatKHR chooseSwapSurfaceFormat (const std::vector<VkSurfaceFormatKHR>& availableFormats){
        for (const auto& availableFormat : availableFormats){
            if(availableFormat.format == VK_FORMAT_B8G8R8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR){
                return availableFormat;
            }
        }
        return availableFormats[0];
    }

    VkPresentModeKHR chooseSwapPresentMode (const std::vector<VkPresentModeKHR>& availablePresentModes){
        for(const auto& availablePresentMode : availablePresentModes){
            if(availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR){
                return availablePresentMode;
            }
        }

        return VK_PRESENT_MODE_IMMEDIATE_KHR;
    }

    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities, GLFWwindow* window){
        if(capabilities.currentExtent.width != UINT32_MAX){
            return capabilities.currentExtent;
        }
        else{
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            VkExtent2D actualExtent = {static_cast<uint32_t>(width),static_cast<uint32_t>(height)};

            actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
            return actualExtent;
        }
    }

};

struct QueueFamilyIndices{
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete(){
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct UniformBufferObject {
    alignas(16) vec3 cameraPosition;
    alignas(16) vec4 cameraRotation;
    alignas(16) vec3 objectPosition;
    alignas(16) vec4 objectRotation;
    alignas(8) vec2 fieldOfView;

    UniformBufferObject(vec3 cameraPosition, vec4 cameraRotation, vec3 objectPosition, vec4 objectRotation, vec2 fieldOfView){
        this->cameraPosition = cameraPosition;
        this->cameraRotation = cameraRotation;
        this->objectPosition = objectPosition;
        this->objectRotation = objectRotation;
        this->fieldOfView = fieldOfView;
    }
};

struct App{
    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkDevice device;
    VkPhysicalDevice physicalDevice;
    VkQueue graphicsQueue;
    VkQueue presentQueue;
    VkSurfaceKHR surface;
    VkSwapchainKHR swapChain;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    VkShaderModule vertShaderModule;
    VkShaderModule fragShaderModule;
    VkDescriptorSetLayout descriptorSetLayout; 
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;
    VkRenderPass renderPass;
    VkCommandPool commandPool;
    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;
    VkRenderPassCreateInfo renderPassInfo{};
    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;

    QueueFamilyIndices indices;
    const std::vector<const char*> validationLayers = {"VK_LAYER_KHRONOS_validation"};
    const std::vector<const char*> deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    std::vector<VkImageView> swapChainImageViews;
    std::vector<VkImage> swapChainImages;
    std::vector<VkFramebuffer> swapChainFrameBuffers;
    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<VkSemaphore> imageAvailableSemaphore;
    std::vector<VkSemaphore> renderFinishedSemaphore;
    std::vector<VkFence> inFlightFences;
    std::vector<VkFence> imagesInFlight;
    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    Camera camera = Camera(Transform(vec3(0,0,-5)));
    UniformBufferObject ubo = UniformBufferObject(vec3(0,0,-5),vec4(0,0,0,1), vec3(2,0,0),vec4(0,0,0,1), vec2(tan(camera.fieldOfView/2.0f),tan(camera.vertFOV/2.0f)));
    GLFWwindow* window;
    size_t currentFrame = 0;
    bool frameBufferResized = false;
    std::vector<Vertex> vertices = {
        {{-0.5f,-0.5f,-0.5f}},
        {{-0.5f,0.5f,-0.5f}},
        {{0.5f,-0.5f,-0.5f}},
        {{0.5f,0.5f,-0.5f}},
        {{-0.5f,-0.5f,0.5f}},
        {{-0.5f,0.5f,0.5f}},
        {{0.5f,-0.5f,0.5f}},
        {{0.5f,0.5f,0.5f}},
        {{1.5f,-0.5f,-0.5f}},
        {{1.5f,0.5f,-0.5f}},
        {{2.5f,-0.5f,-0.5f}},
        {{2.5f,0.5f,-0.5f}},
        {{1.5f,-0.5f,0.5f}},
        {{1.5f,0.5f,0.5f}},
        {{2.5f,-0.5f,0.5f}},
        {{2.5f,0.5f,0.5f}},        
    };
    std::vector<Vertex2D> mandlebrot{};
    std::vector<uint32_t> vertexIndices = {0,1,2,1,3,2, 4,5,6,5,7,6, 0,4,1,1,4,5, 2,3,6,3,7,6, 0,2,4,2,6,4, 1,5,3,3,5,7, 
                        8,9,10,9,11,10, 12,13,14,13,15,14, 8,12,9,9,12,13, 10,11,14,11,15,14, 8,10,12,10,14,12, 9,13,11,11,13,15};
    std::vector<uint32_t> mandlebrotIndices{};
    unsigned char mKeyState[256] = {};
    unsigned char mOldKeyState[256] = {};
    float speed = 3;

    void start() {
        //init2DPixels();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, VK_TRUE);

        window = camera.createScreen();
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, frameBufferResizeCallback);
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain(false);
        createImageViews();
        createRenderPass();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createFrameBuffers();
        createCommandPool();
        createVertexBuffer();
        createIndexBuffer();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
        createSyncObjects();
        std::cout << "Initialised Vulkan!" << std::endl;
    }

    static void frameBufferResizeCallback(GLFWwindow* window, int width, int height){
        auto app = reinterpret_cast<App*>(glfwGetWindowUserPointer(window));
        app->frameBufferResized = true;
    }

    void update(float deltaTime) {
        drawFrame(deltaTime);
        input(deltaTime);
    }

    void cleanupSwapChain(){
        for(auto frameBuffer : swapChainFrameBuffers){
            vkDestroyFramebuffer(device, frameBuffer, nullptr);
        }

        vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());
        
        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyRenderPass(device, renderPass, nullptr);

        for (auto imageView : swapChainImageViews){
            vkDestroyImageView(device, imageView, nullptr);
        }

        vkDestroySwapchainKHR(device, swapChain, nullptr);

        for(size_t i=0; i<swapChainImages.size(); i++){
            vkDestroyBuffer(device, uniformBuffers[i], nullptr);
            vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
        }

        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    }

    void recreateSwapChain(){
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0){
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }
        vkDeviceWaitIdle(device);

        cleanupSwapChain();

        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

        createSwapChain(true);
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createFrameBuffers();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();

        imagesInFlight.resize(swapChainImages.size(), VK_NULL_HANDLE);
    }
    void cleanup(){
        cleanupSwapChain();

        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

        vkDestroyBuffer(device, indexBuffer, nullptr);
        vkFreeMemory(device, indexBufferMemory, nullptr);

        vkDestroyBuffer(device, vertexBuffer, nullptr);
        vkFreeMemory(device, vertexBufferMemory, nullptr);

        for (size_t i=0; i<MAX_FRAMES_IN_FLIGHT; i++){
            vkDestroySemaphore(device, renderFinishedSemaphore[i], nullptr);
            vkDestroySemaphore(device, imageAvailableSemaphore[i], nullptr);
            vkDestroyFence(device, inFlightFences[i], nullptr);
        }
        vkDestroyCommandPool(device, commandPool, nullptr);

      
        vkDestroyDevice(device, nullptr);
        if (enableValidationLayers){
            DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }
        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);
        glfwDestroyWindow(window);
        glfwTerminate();

        vkDestroyShaderModule(device,fragShaderModule,nullptr);
        vkDestroyShaderModule(device,vertShaderModule,nullptr);
    }

    void createInstance(){

        if(enableValidationLayers && !checkValidationLayerSupport(validationLayers)){
            throw std::runtime_error("Validation layers requested but not available.");
        }

        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Triangle Drawing";
        appInfo.applicationVersion = VK_MAKE_VERSION(1,2,198);
        appInfo.pEngineName = "VulkanEngine";
        appInfo.engineVersion = VK_MAKE_VERSION(1,2,198);
        appInfo.apiVersion = VK_API_VERSION_1_2;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
        if(enableValidationLayers){
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();

            populateDebugMessengerCreateInfo(debugCreateInfo);
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }else{
            createInfo.enabledLayerCount = 0;
            createInfo.pNext = nullptr;
        }

        auto glfwExtensions = getRequiredExtensions();
        createInfo.enabledExtensionCount = static_cast<uint32_t>(glfwExtensions.size());
        createInfo.ppEnabledExtensionNames = glfwExtensions.data();

        createInfo.enabledLayerCount = 0;

        if(vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS){
            std::cout << "Failed to create instance" << std::endl;
        }

        uint32_t extensionCount;
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> extensions(extensionCount);
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

    }

    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
        createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
    }

    void setupDebugMessenger(){
        if (!enableValidationLayers) return;

        VkDebugUtilsMessengerCreateInfoEXT createInfo{};
        populateDebugMessengerCreateInfo(createInfo);

        if(CreateDebugMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) !=VK_SUCCESS){
            throw std::runtime_error("Failed to setup debug messenger!");
        }
    }

    void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator){
        auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
        if(func != nullptr){
            func(instance, debugMessenger, pAllocator);
        }
    }

    void pickPhysicalDevice(){
        this->physicalDevice = VK_NULL_HANDLE;

        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

        if (deviceCount == 0){
            std::cout << "Failed to find devices with Vulkan support." << std::endl;
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        for (const auto& device: devices){
            if(isDeviceSuitable(device)){
                this->physicalDevice = device;
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE){
            std::cout << "Failed to find a suitable GPU." << std::endl;
        }
    }

    void createLogicalDevice(){
        findQueueFamilies(physicalDevice);

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(),indices.presentFamily.value()};

        float queuePriority = 1.0f;
        
        for (uint32_t queueFamily: uniqueQueueFamilies){
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        VkPhysicalDeviceFeatures deviceFeatures{};
        deviceFeatures.largePoints = VK_TRUE;

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pEnabledFeatures = &deviceFeatures;

        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        if(enableValidationLayers){
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }else{
            createInfo.enabledLayerCount = 0;
        }

        if(vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS){
            std::cout << "Failed to create logical device" << std::endl;
        }

        vkGetDeviceQueue(device, indices.graphicsFamily.value(),0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.presentFamily.value(),0, &presentQueue);
    }

    void createSurface(){
        if(glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS){
            std::cout << "failed to create window surface." << std::endl;
        }

    }

    bool isDeviceSuitable(VkPhysicalDevice physicalDevice){
        findQueueFamilies(physicalDevice);

        bool extensionsSupported = checkDeviceExtensionSupport(physicalDevice);

        bool swapChainAdequate = false;

        if (extensionsSupported){
            SwapChainSupportDetails swapChainSupport = swapChainSupport.querySwapChainSupport(physicalDevice,surface);
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }

        return this->indices.isComplete() && extensionsSupported && swapChainAdequate;
    }

    void findQueueFamilies(VkPhysicalDevice physicalDevice){

        uint32_t queueFamilyCount;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

        int i=0; 
        for (const auto& queueFamily : queueFamilies){
            if(queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT){
                indices.graphicsFamily = i;
            }
            if(indices.isComplete()){
                break;
            }
            i++;
        }

        VkBool32 presentSupport;
        vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface, &presentSupport);

        if(presentSupport){
            indices.presentFamily = i;
        }

        this->indices = indices;
    }

    bool checkDeviceExtensionSupport(VkPhysicalDevice device){
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        for(const auto& extension : availableExtensions){
            requiredExtensions.erase(extension.extensionName);
        }
        return requiredExtensions.empty();
    }

    void createSwapChain(bool recreate){
        SwapChainSupportDetails swapChainSupport = swapChainSupport.querySwapChainSupport(physicalDevice, surface);

        VkSurfaceFormatKHR surfaceFormat = swapChainSupport.chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode = swapChainSupport.chooseSwapPresentMode(swapChainSupport.presentModes);
        VkExtent2D extent = swapChainSupport.chooseSwapExtent(swapChainSupport.capabilities, window);

        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        if (imageCount > swapChainSupport.capabilities.maxImageCount){
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;

        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        uint32_t QueueFamilyIndices[] = {indices.graphicsFamily.value(),indices.presentFamily.value()};

        if (indices.graphicsFamily != indices.presentFamily){
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = QueueFamilyIndices;
        }
        else{
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
            createInfo.queueFamilyIndexCount = 0;
            createInfo.pQueueFamilyIndices = nullptr;
        }

        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;
        if (recreate){
            createInfo.oldSwapchain = swapChain;
        }else{
            createInfo.oldSwapchain = VK_NULL_HANDLE;
        }

        if(vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS){
            throw std::runtime_error("failed to create swap chain!");
        }   

        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
    }

    void createImageViews(){
        swapChainImageViews.resize(swapChainImages.size());

        for(size_t i = 0; i < swapChainImages.size(); i++){
            VkImageViewCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.image = swapChainImages[i];
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            createInfo.format = swapChainImageFormat;

            createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

            createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            createInfo.subresourceRange.baseMipLevel = 0;
            createInfo.subresourceRange.levelCount = 1;
            createInfo.subresourceRange.baseArrayLayer = 0;
            createInfo.subresourceRange.layerCount = 1;

            if(vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS){
                throw std::runtime_error("failed to create image views!");
            }
        }
    }
    VkShaderModule createShaderModule(const std::vector<char>& code){
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule shaderModule;
        if(vkCreateShaderModule(device,&createInfo,nullptr, &shaderModule) !=VK_SUCCESS){
            throw std::runtime_error("Failed to create shader module");
        }   

        return shaderModule;
    }
    void createGraphicsPipeline(){
        auto vertShaderCode = readFile("Dependencies/shaders/vert.spv");
        auto fragShaderCode = readFile("Dependencies/shaders/frag.spv");  

        vertShaderModule = createShaderModule(vertShaderCode);
        fragShaderModule = createShaderModule(fragShaderCode);

        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float) swapChainExtent.width;
        viewport.height = (float) swapChainExtent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D scissor{};
        scissor.offset ={0,0};
        scissor.extent = swapChainExtent;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.pViewports = &viewport;
        viewportState.scissorCount = 1;
        viewportState.pScissors = &scissor;

        VkPipelineRasterizationStateCreateInfo rasteriser{};
        rasteriser.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasteriser.depthClampEnable = VK_FALSE;
        rasteriser.rasterizerDiscardEnable = VK_FALSE;
        rasteriser.polygonMode = VK_POLYGON_MODE_FILL;
        rasteriser.lineWidth = 1.0f;
        rasteriser.cullMode = VK_CULL_MODE_NONE;
        rasteriser.frontFace = VK_FRONT_FACE_CLOCKWISE;
        rasteriser.depthBiasEnable = VK_FALSE;
        rasteriser.depthBiasConstantFactor = 0.0f;
        rasteriser.depthBiasClamp = 0.0f;
        rasteriser.depthBiasSlopeFactor = 0.0f;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multisampling.minSampleShading = 1.0f;
        multisampling.pSampleMask = nullptr;
        multisampling.alphaToCoverageEnable = VK_FALSE;
        multisampling.alphaToOneEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_TRUE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;

        VkDynamicState dynamicStates[] = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_LINE_WIDTH
        };
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = 2;
        dynamicState.pDynamicStates = dynamicStates;

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 0;
        pipelineLayoutInfo.pPushConstantRanges = nullptr;

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS){
            throw std::runtime_error("failed to create pipeline layout!");
        } 

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;

        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasteriser;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = nullptr;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = nullptr;

        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo.basePipelineIndex = -1;

        if(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS){
            throw std::runtime_error("failed to create graphics pipeline");
        }
    }

    void createRenderPass(){
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapChainImageFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;

        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &colorAttachment;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;

        if(vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS){
            throw std::runtime_error("failed to create render pass");
        }

    }

    void createFrameBuffers(){
        swapChainFrameBuffers.resize(swapChainImageViews.size());

        for (size_t i=0; i<swapChainImageViews.size(); i++){
            VkImageView attachments[] = {swapChainImageViews[i]};

            VkFramebufferCreateInfo frameBufferInfo{};
            frameBufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            frameBufferInfo.renderPass = renderPass;
            frameBufferInfo.attachmentCount = 1;
            frameBufferInfo.pAttachments = attachments;
            frameBufferInfo.width = swapChainExtent.width;
            frameBufferInfo.height = swapChainExtent.height;

            if(vkCreateFramebuffer(device, &frameBufferInfo, nullptr, &swapChainFrameBuffers[i]) != VK_SUCCESS){
                throw std::runtime_error("failed to create frame buffer");
            }

        }
    }

    void createCommandPool(){
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = indices.graphicsFamily.value();
        poolInfo.flags = 0;

        if(vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS){
            throw std::runtime_error("failed to create command pool.");
        }
    }

    void createCommandBuffers(){
        commandBuffers.resize(swapChainFrameBuffers.size());

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = (uint32_t) commandBuffers.size();

        if(vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS){
            throw std::runtime_error("failed to allocate command buffers.");
        }

        for (size_t i=0; i<commandBuffers.size(); i++){
            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = 0;
            beginInfo.pInheritanceInfo = nullptr;
      
            if(vkBeginCommandBuffer(commandBuffers[i], &beginInfo) != VK_SUCCESS){
                throw std::runtime_error("failed to begin recording command buffer");
            }

            VkRenderPassBeginInfo renderPassInfo{};
            renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            renderPassInfo.renderPass = renderPass;
            renderPassInfo.framebuffer = swapChainFrameBuffers[i];

            renderPassInfo.renderArea.offset = {0,0};
            renderPassInfo.renderArea.extent = swapChainExtent;

            VkClearValue clearColor = {{{0.0f,0.0f,0.0f,1.0f}}};
            renderPassInfo.clearValueCount = 1;
            renderPassInfo.pClearValues = &clearColor;
            vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
                vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

                VkBuffer vertexBuffers[] = {vertexBuffer};
                VkDeviceSize offsets[] = {0};
                vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, vertexBuffers, offsets);
                vkCmdBindIndexBuffer(commandBuffers[i], indexBuffer, 0, VK_INDEX_TYPE_UINT32);

                vkCmdBindDescriptorSets(commandBuffers[i],VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[i], 0, nullptr);
                vkCmdDrawIndexed(commandBuffers[i], static_cast<uint32_t>(vertexIndices.size()), 1, 0, 0, 0);
            vkCmdEndRenderPass(commandBuffers[i]);
            if(vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS){
                throw std::runtime_error("failed to record command buffer!");
            }
        }
    }


    void drawFrame(float deltaTime){
        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphore[currentFrame], inFlightFences[currentFrame], &imageIndex);

        if(result == VK_ERROR_OUT_OF_DATE_KHR){
            recreateSwapChain();
        }else if(result !=VK_SUCCESS && result != VK_SUBOPTIMAL_KHR){
            throw std::runtime_error("failed to acquire swap chain image.");
        }

        if(imagesInFlight[imageIndex] != VK_NULL_HANDLE){
            vkWaitForFences(device, 1, &imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);
        }
        imagesInFlight[imageIndex] = inFlightFences[currentFrame];

        updateUniformBuffer(imageIndex, deltaTime);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &imageAvailableSemaphore[currentFrame];
        submitInfo.pWaitDstStageMask = waitStages;

        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[imageIndex];

        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &renderFinishedSemaphore[currentFrame];

        vkResetFences(device, 1, &inFlightFences[currentFrame]);

        if(vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS){
            throw std::runtime_error("failed to submit command buffer to graphics queue.");
        }
        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;
        

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &renderFinishedSemaphore[currentFrame];

        VkSwapchainKHR swapChains[] = {swapChain};
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;
        presentInfo.pResults = nullptr;

        result = vkQueuePresentKHR(presentQueue, &presentInfo);

        if(result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || frameBufferResized){
            frameBufferResized = false;
            recreateSwapChain();
        }else if (result != VK_SUCCESS){
            throw std::runtime_error("failed to present swap chain image.");
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    void createSyncObjects(){
        imageAvailableSemaphore.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphore.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
        imagesInFlight.resize(swapChainImages.size());

        VkSemaphoreCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (size_t i=0; i< MAX_FRAMES_IN_FLIGHT; i++){
            if(vkCreateSemaphore(device, &createInfo, nullptr, &imageAvailableSemaphore[i]) != VK_SUCCESS || 
              vkCreateSemaphore(device, &createInfo, nullptr, &renderFinishedSemaphore[i]) != VK_SUCCESS || 
              vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create synchronisation objects for a frame");
           }        
        }
    }
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory){
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS){
            throw std::runtime_error("failed to create vertex buffer!");
        }

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS){
            throw std::runtime_error("failed to allocate memory for vertex buffer.");
        }

        vkBindBufferMemory(device, buffer, bufferMemory,0);
    }

    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size){
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;
        
        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);
            VkBufferCopy copyRegion{};
            copyRegion.srcOffset = 0;
            copyRegion.dstOffset = 0;
            copyRegion.size = size;
            vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);

        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    }

    void createVertexBuffer(){     
        VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;

        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
            memcpy(data, vertices.data(), (size_t) bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);
        copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    void createIndexBuffer(){
        VkDeviceSize bufferSize = sizeof(vertexIndices[0]) * vertexIndices.size();
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;

        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
            memcpy(data, vertexIndices.data(), (size_t) bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, vertexBufferMemory);
        copyBuffer(stagingBuffer, indexBuffer, bufferSize);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties){
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

        for (uint32_t i=0; i<memProperties.memoryTypeCount; i++){
            if (typeFilter & (1 << i) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties){
                return i;
            }
        }
        throw std::runtime_error("failed to find suitable memory type!");
    }

    virtual void onKeyDown(int key,float deltaTime){
        if (key == 'W') {
            camera.transform.position += vec3(0, 0, deltaTime * speed);
        }
        if (key == 'A') {
            camera.transform.position += vec3(-deltaTime * speed, 0, 0);
        }
        if (key == 'S') {
            camera.transform.position += vec3(0, 0, -deltaTime * speed);
        }
        if (key == 'D') {
            camera.transform.position += vec3(deltaTime * speed, 0, 0);
        }

    }
    
    virtual void onKeyUp(int key, float deltaTime){

    }

    void input(float deltaTime){
        GetKeyboardState(mKeyState);

        for (unsigned int i = 0; i < 256; i++) {
            if (mKeyState[i] & 0x80) {
                onKeyDown(i, deltaTime);
            }

            else if (mKeyState[i] != mOldKeyState[i]) {
                onKeyUp(i, deltaTime);
            }
        }
        ::memcpy(mOldKeyState, mKeyState, sizeof(unsigned char) * 256);
    }

    void createDescriptorSetLayout(){
        VkDescriptorSetLayoutBinding uboLayoutBinding{};
        uboLayoutBinding.binding = 0;
        uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboLayoutBinding.descriptorCount = 1;
        uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        uboLayoutBinding.pImmutableSamplers = nullptr;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 1;
        layoutInfo.pBindings = &uboLayoutBinding;

        if(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS){
            throw std::runtime_error("failed to create descriptor set layout");
        }
    }

    void createUniformBuffers(){
        VkDeviceSize bufferSize = sizeof(UniformBufferObject);

        uniformBuffers.resize(swapChainImages.size());
        uniformBuffersMemory.resize(swapChainImages.size());

        for(size_t i=0; i< swapChainImages.size(); i++){
            createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i],
            uniformBuffersMemory[i]);
        }
    }

    void updateUniformBuffer(uint32_t currentImage, float deltaTime){
        Transform trans = Transform(ubo.objectPosition, ubo.objectRotation);
        trans.Rotate(vec3(0,0,deltaTime));
        trans.position += vec3(0,0,deltaTime);
        //camera.transform.Rotate(vec3(0,deltaTime,0));

        ubo.objectPosition = trans.position;
        ubo.objectRotation = trans.rotation;
        ubo.cameraPosition = camera.transform.position;
        ubo.cameraRotation = camera.transform.rotation;

        void* data;
        vkMapMemory(device, uniformBuffersMemory[currentImage], 0, sizeof(ubo), 0, &data);
            memcpy(data, &ubo, sizeof(ubo));
        vkUnmapMemory(device, uniformBuffersMemory[currentImage]);
    }

    void createDescriptorPool(){
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSize.descriptorCount = static_cast<uint32_t>(swapChainImages.size());

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        poolInfo.maxSets = static_cast<uint32_t>(swapChainImages.size());

        if(vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS){
            throw std::runtime_error("failed to create descriptor pool");
        }
    }

    void createDescriptorSets(){
        std::vector<VkDescriptorSetLayout> layouts(swapChainImages.size(),descriptorSetLayout);
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(swapChainImages.size());
        allocInfo.pSetLayouts = layouts.data();

        descriptorSets.resize(swapChainImages.size());
        if(vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS){
            throw std::runtime_error("failed to create descriptor sets");
        }

        for (size_t i=0; i<swapChainImages.size(); i++){
            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = uniformBuffers[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);

            VkWriteDescriptorSet descriptorWrite{};
            descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrite.dstSet = descriptorSets[i];
            descriptorWrite.dstBinding = 0;
            descriptorWrite.dstArrayElement = 0;
            descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrite.descriptorCount = 1;
            descriptorWrite.pBufferInfo = &bufferInfo;
            descriptorWrite.pImageInfo = nullptr;
            descriptorWrite.pTexelBufferView = nullptr;

            vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
        }      
    }
    void init2DPixels(){
        for (int i=0; i<camera.xRes; i++){
            for (int j=0; j<camera.yRes; j++){
                Vertex2D temp = {{-1.0f + 2.0f*(float)(i)/(float)camera.xRes, -1.0f + 2.0f*(float)(j)/(float)camera.yRes}};
                mandlebrot.push_back(temp);
                mandlebrotIndices.push_back(camera.yRes * i + j);
            }
        }
    }
};

