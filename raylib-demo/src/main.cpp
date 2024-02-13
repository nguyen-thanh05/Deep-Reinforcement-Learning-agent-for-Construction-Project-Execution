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

#include "raylib.h"
#include "rlgl.h"


void DrawPlanes(int x, int y, int z, float spacing)
{
    int max_dim = x > y ? x : (y > z ? y : z);
    float lineLength = (float) max_dim * spacing;

    rlBegin(RL_LINES);
    for (int i = 0; i <= max_dim; i++) {
        float lineOffset = (float) i * spacing;
        if (i == 0) {
            rlColor3f(0.0f, 0.5f, 0.5f);
            rlColor3f(0.0f, 0.5f, 0.5f);
            rlColor3f(0.0f, 0.5f, 0.5f);
            rlColor3f(0.0f, 0.5f, 0.5f);
        } else {
            rlColor3f(0.75f, 0.75f, 0.75f);
            rlColor3f(0.75f, 0.75f, 0.75f);
            rlColor3f(0.75f, 0.75f, 0.75f);
            rlColor3f(0.75f, 0.75f, 0.75f);
        }
        
        rlVertex3f(0.0f, 0.0f, lineOffset);
        rlVertex3f(0.0f, lineLength, lineOffset);
        rlVertex3f(lineOffset, 0.0f, 0.0f);
        rlVertex3f(lineOffset, lineLength, 0.0f);

        rlVertex3f(0.0f, 0.0f, lineOffset);
        rlVertex3f(lineLength, 0.0f, lineOffset);
        rlVertex3f(0.0f, lineOffset, 0.0f);
        rlVertex3f(lineLength, lineOffset, 0.0f);

        rlVertex3f(0.0f, lineOffset, 0.0f);
        rlVertex3f(0.0f, lineOffset, lineLength);
        rlVertex3f(lineOffset, 0.0f, 0.0f);
        rlVertex3f(lineOffset, 0.0f, lineLength);

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

    InitWindow(screenWidth, screenHeight, "raylib [core] example - 3d picking");

    // Define the camera to look into our 3d world
    Camera camera = { 0 };
    camera.position = (Vector3){ 10.0f, 10.0f, 10.0f }; // Camera position
    camera.target = (Vector3){ 0.0f, 0.0f, 0.0f };      // Camera looking at point
    camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };          // Camera up vector (rotation towards target)
    camera.fovy = 45.0f;                                // Camera field-of-view Y
    camera.projection = CAMERA_PERSPECTIVE;             // Camera projection type

    Vector3 cubePosition = { 0.5f, 0.5f, 0.5f };
    Vector3 cubeSize = { 1.0f, 1.0f, 1.0f };

    Ray ray = { 0 };                    // Picking line ray
    RayCollision collision = { 0 };     // Ray collision hit info

    SetTargetFPS(60);                   // Set our game to run at 60 frames-per-second
    //--------------------------------------------------------------------------------------

    // Main game loop
    while (!WindowShouldClose())        // Detect window close button or ESC key
    {
        // Update
        //----------------------------------------------------------------------------------
        if (IsCursorHidden()) UpdateCamera(&camera, CAMERA_FREE);

        // Toggle camera controls
        if (IsMouseButtonPressed(MOUSE_BUTTON_RIGHT))
        {
            if (IsCursorHidden()) EnableCursor();
            else DisableCursor();
        }

        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT))
        {
            if (!collision.hit)
            {
                ray = GetMouseRay(GetMousePosition(), camera);

                // Check collision between ray and box
                collision = GetRayCollisionBox(ray,
                                               (BoundingBox){(Vector3){ cubePosition.x - cubeSize.x/2, cubePosition.y - cubeSize.y/2, cubePosition.z - cubeSize.z/2 },
                                                             (Vector3){ cubePosition.x + cubeSize.x/2, cubePosition.y + cubeSize.y/2, cubePosition.z + cubeSize.z/2 }});
            }
            else collision.hit = false;
        }
        //----------------------------------------------------------------------------------

        // Draw
        //----------------------------------------------------------------------------------
        BeginDrawing();

        ClearBackground(RAYWHITE);

        BeginMode3D(camera);

        if (collision.hit)
        {
            DrawCube(cubePosition, cubeSize.x, cubeSize.y, cubeSize.z, RED);
            DrawCubeWires(cubePosition, cubeSize.x, cubeSize.y, cubeSize.z, MAROON);

            DrawCubeWires(cubePosition, cubeSize.x + 0.2f, cubeSize.y + 0.2f, cubeSize.z + 0.2f, GREEN);
        }
        else
        {
            DrawCube(cubePosition, cubeSize.x, cubeSize.y, cubeSize.z, GRAY);
            DrawCubeWires(cubePosition, cubeSize.x, cubeSize.y, cubeSize.z, DARKGRAY);
        }

//        DrawRay(ray, MAROON);
//        DrawXZGrid(3, 1.0f);
//        DrawXYGrid(4, 1.0f);
//        DrawYZGrid(5, 1.0f);
        DrawPlanes(3,3,4, 1.0f);

        EndMode3D();
        if (collision.hit) DrawText("BOX SELECTED", (screenWidth - MeasureText("BOX SELECTED", 30)) / 2, (int)(screenHeight * 0.1f), 30, GREEN);
        
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
