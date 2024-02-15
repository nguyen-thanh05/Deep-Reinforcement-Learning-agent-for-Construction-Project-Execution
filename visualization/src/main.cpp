#include "raylib.h"
#include "rlgl.h"
#include "GridWorld.h"

int main()
{
    // Initialization
    //--------------------------------------------------------------------------------------
    const int screenWidth = 1000;
    const int screenHeight = 600;

    GWEnv::GridWorld env(5, 5, 4, std::make_unique<GWEnv::InteractiveInput>());
    InitWindow(screenWidth, screenHeight, "Grid World - Interactive input");

    SetTargetFPS(60);                   // Set our game to run at 60 frames-per-second

    // Main game loop
    while (!WindowShouldClose())        // Detect window close button or ESC key
    {
        env.Render();
    }

    CloseWindow();        // Close window and OpenGL context

    return 0;
}
