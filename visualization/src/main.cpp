// Put this define in whatever header you need to use raygui.h in
//#ifndef RAYGUI_IMPLEMENTAION
//    #define RAYGUI_IMPLEMENTATION
//#endif
//#include "raygui.h"

#include <iostream>
#include "raylib.h"
#include "rlgl.h"
#include "GridWorld.h"

int main(int argc, char* argv[])
{
    if (argc != 3) {
        std::cerr << "Usage: ./GridWorldEnv <env_size> <output_path>" << std::endl;
        exit(1);
    }

    int size = std::stoi(argv[1]);
    std::string outputPath{argv[2]};

    const int screenWidth = 1000;
    const int screenHeight = 600;

    GWEnv::GridWorld env(size, size, size, outputPath);
    InitWindow(screenWidth, screenHeight, "Grid World");

    SetTargetFPS(60);                   // Set our game to run at 60 frames-per-second

    // Main game loop
    while (!WindowShouldClose())        // Detect window close button or ESC key
    {
        env.Render();
    }

    CloseWindow();        // Close window and OpenGL context

    return 0;
}
