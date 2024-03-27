#include <fstream>
#include <iostream>
#include <thread>
#include "GridWorld.h"

namespace GWEnv{

void GridWorld::Render() {
    if (IsCursorHidden()) UpdateCamera(&camera, CAMERA_FREE);

    // Toggle camera controls
    if (IsMouseButtonPressed(MOUSE_BUTTON_RIGHT)) {
        if (IsCursorHidden()) EnableCursor();
        else DisableCursor();
    }
    Action action = Step();
    switch (action) {
        case Action::NONE: break;
        case Action::PLACE:
            AddBlock(agentPos.x, agentPos.y, agentPos.z, currBlockType);
            break;
        case Action::REMOVE:
            RemoveBlock(agentPos.x, agentPos.y, agentPos.z);
            break;
        default:
            Move(action);
            break;
    }

    if (recording && action != Action::NONE) sequence.push_back(action);

    BeginDrawing();
    ClearBackground(RAYWHITE);
    BeginMode3D(camera);

        DrawPlanes();
        DrawBlocks();

    EndMode3D();

    DrawRectangle( 10, 10, 240, 150, Fade(SKYBLUE, 0.5f));
    DrawRectangleLines( 10, 10, 240, 150, BLUE);

    DrawText(IsCursorHidden() ? "Mode: Free camera - right click to switch"
        : "Mode: Fixed Camera - right click to switch", 20, 20, 10, BLACK);
    DrawText("WASD: Horizontal move", 40, 40, 10, DARKGRAY);
    DrawText("QE: Vertical move", 40, 60, 10, DARKGRAY);
    DrawText("X: Delete block", 40, 80, 10, DARKGRAY);
    DrawText("SPACE: Place block", 40, 100, 10, DARKGRAY);
    DrawText("ENTER: Save to file", 40, 120, 10, DARKGRAY);
    if (recording) {
        DrawText("R: Stop recording", 40, 140, 10, DARKGRAY);
        DrawRectangle(140, 140, 10, 10, RED);
    } else {
        DrawText("R: Start recording", 40, 140, 10, DARKGRAY);
        DrawTriangle({150, 145}, {140, 140}, {140, 150}, RED);
    }

    DrawRectangle( 900, 10, 70, 70, Fade(blockColors[currBlockType], 0.5f));
    DrawRectangleLines( 900, 10, 70, 70, blockColors[currBlockType]);
    DrawText("Block type", 910, 20, 10, DARKGRAY);
    DrawText(std::to_string(currBlockType).c_str(), 930, 35, 20, DARKGRAY);
    EndDrawing();
}

void GridWorld::DrawPlanes() const {
    float hLength = (float) h * spacing;
    float wLength = (float) w * spacing;
    float dLength = (float) d * spacing;

    rlBegin(RL_LINES);
    for (int i = 0; i <= h; i++) {
        float lineOffset = (float) i * spacing;
        rlColor3f(0.25f, 0.25f, 1.0f);
        rlVertex3f(lineOffset, 0.0f, 0.0f);
        rlVertex3f(lineOffset, wLength, 0.0f);

        rlColor3f(1.0f, 0.25f, 0.25f);
        rlVertex3f(lineOffset, 0.0f, 0.0f);
        rlVertex3f(lineOffset, 0.0f, dLength);
    }

    for (int i = 0; i <= w; i++) {
        float lineOffset = (float) i * spacing;
        rlColor3f(0.25f, 0.25f, 1.0f);
        rlVertex3f(0.0f, lineOffset, 0.0f);
        rlVertex3f(hLength, lineOffset, 0.0f);

        rlColor3f(0.25f, 1.0f, 0.25f);
        rlVertex3f(0.0f, lineOffset, 0.0f);
        rlVertex3f(0.0f, lineOffset, dLength);
    }

    for (int i = 0; i <= d; i++) {
        float lineOffset = (float) i * spacing;
        rlColor3f(1.0f, 0.25f, 0.25f);
        rlVertex3f(0.0f, 0.0f, lineOffset);
        rlVertex3f(hLength, 0.0f, lineOffset);

        rlColor3f(0.25f, 1.0f, 0.25f);
        rlVertex3f(0.0f, 0.0f, lineOffset);
        rlVertex3f(0.0f, wLength, lineOffset);
    }

    rlEnd();
}

bool GridWorld::AddBlock(int x, int y, int z, int blockType) {
    if (x >= h || y >= w || z >= d || grid[x][y][z] == blockType || !QueryPlacement(x, y, z))
        return false;

    grid[x][y][z] = blockType;
    return true;
}

void GridWorld::Move(Action direction) {
    switch (direction) {
        case Action::RIGHT:
            agentPos.x = std::min(w - 1, agentPos.x + 1);
            break;
        case Action::LEFT:
            agentPos.x = std::max(0, agentPos.x - 1);
            break;
        case Action::DOWN:
            agentPos.y = std::max(0, agentPos.y - 1);
            break;
        case Action::UP:
            agentPos.y = std::min(h - 1, agentPos.y + 1);
            break;
        case Action::BACKWARD:
            agentPos.z = std::min(d - 1, agentPos.z + 1);
            break;
        case Action::FORWARD:
            agentPos.z = std::max(0, agentPos.z - 1);
            break;
        default:
            break;
    }
}

bool GridWorld::QueryPlacement(int x, int y, int z) const {
    if (y == 0) return true;
    if ( (x > 0 && grid[x-1][y][z]) ||
        (x < h-1 && grid[x+1][y][z]) ) return true;

    else if ( (y > 0 && grid[x][y-1][z]) ||
        (y < w-1 && grid[x][y+1][z]) ) return true;

    else if ( (z > 0 && grid[x][y][z-1]) ||
        (z < d-1 && grid[x][y][z+1]) ) return true;

    return false;
}

void GridWorld::DrawBlocks() const {
    // Draw every block in grid
    for (int i = 0; i < w; i++) {
        for (int j = 0; j < h; j++) {
            for (int k = 0; k < d; k++) {
                if (grid[i][j][k]) {
                    float posX = static_cast<float>(i) * spacing + 0.5f;
                    float posY = static_cast<float>(j) * spacing + 0.5f;
                    float posZ = static_cast<float>(k) * spacing + 0.5f;

                    DrawCube({posX, posY, posZ},
                             spacing, spacing, spacing,
                             Fade(blockColors[grid[i][j][k]], 0.5f));
//                    DrawCubeWires({posX, posY, posZ},
//                                  spacing, spacing, spacing, RED);
                }
            }
        }
    }

    // Draw current agent position
    DrawCubeWires({static_cast<float>(agentPos.x) * spacing + 0.5f,
                   static_cast<float>(agentPos.y) * spacing + 0.5f,
                   static_cast<float>(agentPos.z) * spacing + 0.5f},
                  spacing, spacing, spacing, PURPLE);
}

void GridWorld::RemoveBlock(int x, int y, int z) {
    grid[x][y][z] = 0;
}

void GridWorld::ResizeGrid(uint32_t _w, uint32_t _h, uint32_t _d) {
    grid = {_w, vec<vec<int>>(_h, vec<int>(_d, 0))};
    agentPos = {0, 0, 0};
}

void GridWorld::SaveTarget() const {
    std::ofstream targetOutStream;
    targetOutStream.open(filePath, std::fstream::out);

    targetOutStream << w << ' ' << h << ' ' << d << ' ';

    for (int i=0; i < w; i++) {
        for (int j=0; j < h; j++) {
            for (int k=0; k < d; k++) {
                targetOutStream << grid[i][j][k] << " ";
            }
        }
        targetOutStream << std::endl;
    }
    targetOutStream << std::endl;
    targetOutStream.close();
}

void GridWorld::LoadFromFile() {
    std::ifstream file;
    int x, y, z;

    file.open(filePath, std::fstream::in);
    file >> x >> y >> z;
    if (x != w || y != h || z != d) ResizeGrid(x, y, z);

    for (int i=0; i < w; i++) {
        for (int j=0; j < h; j++) {
            for (int k=0; k < d; k++) {
                file >> grid[i][j][k];
            }
        }
    }
    file.close();
}

Action GridWorld::Step() {
    // Do not allow save file if nothing changes
    static bool enterDelayed = false;

    // C-style enums lmao :) hope raylib doesn't change this.
    for (int i = 1; i <= nBlockTypes; i++) {
        if (IsKeyPressed(i + KEY_ZERO)) {
            currBlockType = i;
            return Action::NONE;
        }
    }
    if (IsKeyPressed(KEY_R)) {
        recording = !recording;
        Record();
        return Action::NONE;
    }

    if (IsKeyPressed(KEY_ENTER)) {
        if (enterDelayed) return Action::NONE;
        enterDelayed = true;
        SaveSequence();
        SaveTarget();
        return Action::NONE;
    }

    if (IsCursorHidden()) return Action::NONE;

    if (IsKeyPressed(KEY_W)) return Action::FORWARD;
    if (IsKeyPressed(KEY_S)) return Action::BACKWARD;
    if (IsKeyPressed(KEY_A)) return Action::LEFT;
    if (IsKeyPressed(KEY_D)) return Action::RIGHT;
    if (IsKeyPressed(KEY_E)) return Action::UP;
    if (IsKeyPressed(KEY_Q)) return Action::DOWN;

    if (IsKeyPressed(KEY_SPACE)) {
        enterDelayed = false;
        return Action::PLACE;
    }
    if (IsKeyPressed(KEY_X)) {
        enterDelayed = false;
        return Action::REMOVE;
    }

    return Action::NONE;
}

void GridWorld::SaveSequence() const {
    std::ofstream outStream(filePath + ".seq", std::fstream::out);
    outStream << "This sequence file should be accompanied by a corresponding target file of name " << filePath << std::endl;
    outStream << "Dimensions: " << w << ' ' << h << ' ' << d << std::endl;
    outStream << "Starting position: x " << startingPos.x << ' ' << startingPos.y << ' ' << startingPos.z << std::endl;
    outStream << "Number of actions: " << sequence.size() << std::endl << std::endl;

    for (auto action : sequence) {
        outStream << ActionName[action] << std::endl;
    }
    outStream.close();
}
}

