#include <main.h>

// Called at engine launch
void start(){
}

// Called every frame
void update(){

}

int main() {

    glfwInit();
    float deltaTime;
    App app;
    app.start();
    start();

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(app.window))
    {   
        auto t_start = std::chrono::high_resolution_clock::now();

        app.update(deltaTime);
        update();
        glfwSwapBuffers(app.window);
        glfwPollEvents();
        auto t_end = std::chrono::high_resolution_clock::now();
        deltaTime = std::chrono::duration<float>(t_end - t_start).count();
    }

    vkDeviceWaitIdle(app.device);
    app.cleanup();
    return 0;
}