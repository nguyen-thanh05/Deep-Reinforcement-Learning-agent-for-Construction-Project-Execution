/*******************************************************************************************
*
*   raylib [core] example - Picking in 3d mode
*
*   Example originally created with raylib 1.3, last time updated with raylib 4.0
*
*   Example licensed under an unmodified zlib/libpng license, which is an OSI-certified,
*   BSD-like license that allows static linking with closed source software
*
*   Copyright (c) 2015-2024 Ramon Santamaria (@raysan5)
*
********************************************************************************************/

#include <vector>
#include "raylib.h"
#include "rlgl.h"


void DrawPlanes(int x, int y, int z, float spacing)
{
    float xLength = (float) x * spacing;
    float yLength = (float) y * spacing;
    float zLength = (float) z * spacing;


    rlBegin(RL_LINES);
    for (int i = 0; i <= x; i++) {
        float lineOffset = (float) i * spacing;
        rlColor3f(0.25f, 0.25f, 1.0f);
        rlVertex3f(lineOffset, 0.0f, 0.0f);
        rlVertex3f(lineOffset, yLength, 0.0f);

        rlColor3f(1.0f, 0.25f, 0.25f);
        rlVertex3f(lineOffset, 0.0f, 0.0f);
        rlVertex3f(lineOffset, 0.0f, zLength);
    }

    for (int i = 0; i <= y; i++) {
        float lineOffset = (float) i * spacing;
        rlColor3f(0.25f, 0.25f, 1.0f);
        rlVertex3f(0.0f, lineOffset, 0.0f);
        rlVertex3f(xLength, lineOffset, 0.0f);

        rlColor3f(0.25f, 1.0f, 0.25f);
        rlVertex3f(0.0f, lineOffset, 0.0f);
        rlVertex3f(0.0f, lineOffset, zLength);
    }

    for (int i = 0; i <= z; i++) {
        float lineOffset = (float) i * spacing;
        rlColor3f(1.0f, 0.25f, 0.25f);
        rlVertex3f(0.0f, 0.0f, lineOffset);
        rlVertex3f(xLength, 0.0f, lineOffset);

        rlColor3f(0.25f, 1.0f, 0.25f);
        rlVertex3f(0.0f, 0.0f, lineOffset);
        rlVertex3f(0.0f, yLength, lineOffset);
    }

    rlEnd();
}

//------------------------------------------------------------------------------------
// Program main entry point
//------------------------------------------------------------------------------------
int main()
{
    // Initialization
    //--------------------------------------------------------------------------------------
    const int screenWidth = 800;
    const int screenHeight = 450;

    std::vector<Vector3> boxes {{0.5f, 0.5f, 0.5f}};

    InitWindow(screenWidth, screenHeight, "raylib [core] example - 3d picking");

    // Define the camera to look into our 3d world
    Camera camera = { 0 };
    camera.position = (Vector3){ 10.0f, 10.0f, 10.0f }; // Camera position
    camera.target = (Vector3){ 0.0f, 0.0f, 0.0f };      // Camera looking at point
    camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };          // Camera up vector (rotation towards target)
    camera.fovy = 45.0f;                                // Camera field-of-view Y
    camera.projection = CAMERA_PERSPECTIVE;             // Camera projection type

    Vector3 cubeSize = { 1.0f, 1.0f, 1.0f };

    Ray ray = { 0 };                    // Picking line ray
    RayCollision collision = { 0 };     // Ray collision hit info

    SetTargetFPS(60);                   // Set our game to run at 60 frames-per-second
    //--------------------------------------------------------------------------------------

    Vector3* adjacentCube = nullptr;
    // Main game loop
    while (!WindowShouldClose())        // Detect window close button or ESC key
    {
        // Update
        //----------------------------------------------------------------------------------
        if (IsCursorHidden()) UpdateCamera(&camera, CAMERA_FREE);
        bool foundAdjCube = false;


        // Toggle camera controls
        if (IsMouseButtonPressed(MOUSE_BUTTON_RIGHT))
        {
            if (IsCursorHidden()) EnableCursor();
            else DisableCursor();
        }

        //----------------------------------------------------------------------------------

        // Draw
        //----------------------------------------------------------------------------------
        BeginDrawing();

        ClearBackground(RAYWHITE);

        BeginMode3D(camera);

        DrawPlanes(5, 5, 4, 1.0f);
        for (Vector3& cube : boxes) {
            DrawCube(cube, cubeSize.x, cubeSize.y, cubeSize.z, RED);
            DrawCubeWires(cube, cubeSize.x, cubeSize.y, cubeSize.z, MAGENTA);

            if (!foundAdjCube) {
                ray = GetMouseRay(GetMousePosition(), camera);
                collision = GetRayCollisionBox(ray,
                                               {{ cube.x - cubeSize.x/2, cube.y - cubeSize.y/2, cube.z - cubeSize.z/2 },
                                                { cube.x + cubeSize.x/2, cube.y + cubeSize.y/2, cube.z + cubeSize.z/2 }});
                if (collision.hit) {
                    foundAdjCube = true;
                    adjacentCube = &cube;
                }
            }
        }
        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT))
        {
            if (foundAdjCube && adjacentCube != nullptr) {
                Vector3 placement = *adjacentCube;
                Vector3& collPoint = collision.point;
                if (collPoint.x == adjacentCube->x + 0.5) {
                    placement.x = adjacentCube->x + 1;
                } else if (collPoint.x == adjacentCube->x - 0.5) {
                    placement.x = adjacentCube->x - 1;
                } else if (collPoint.y == adjacentCube->y + 0.5) {
                    placement.y = adjacentCube->y + 1;
                } else if (collPoint.y == adjacentCube->y - 0.5) {
                    placement.y = adjacentCube->y - 1;
                } else if (collPoint.z == adjacentCube->z + 0.5) {
                    placement.z = adjacentCube->z + 1;
                } else if (collPoint.z == adjacentCube->z - 0.5) {
                    placement.z = adjacentCube->z - 1;
                }

                boxes.push_back(placement);
            }
        }

        EndMode3D();

        DrawFPS(10, 10);

        EndDrawing();
        //----------------------------------------------------------------------------------
    }

    // De-Initialization
    //--------------------------------------------------------------------------------------
    CloseWindow();        // Close window and OpenGL context
    //--------------------------------------------------------------------------------------

    return 0;
}
