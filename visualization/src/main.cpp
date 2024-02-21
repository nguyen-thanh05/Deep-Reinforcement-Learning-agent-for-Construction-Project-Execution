// Put this define in whatever header you need to use raygui.h in
//#ifndef RAYGUI_IMPLEMENTAION
//    #define RAYGUI_IMPLEMENTATION
//#endif
//#include "raygui.h"

#include <iostream>
#include "raylib.h"
#include "rlgl.h"
#include "GridWorld.h"

void PrintError(){
    std::cerr << "Usage: ./GridWorldEnv <env_size> <s|l> <path>\n\n"
              << "<env_size>: size of the environment\n"
              << "<s|l>     : whether to save to file or load from file\n"
              << "<path>    : path to file to load from / save to\n";
    exit(1);
}

int main(int argc, char* argv[])
{
    if (argc != 4) PrintError();

    int size = std::stoi(argv[1]);

    std::string operation{argv[2]};
    if (operation != "s" && operation != "l") PrintError();
    bool save = operation == "s";

    std::string path{argv[3]};

    const int screenWidth = 1000;
    const int screenHeight = 600;

    GWEnv::GridWorld env(size, size, size, path, save);
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
