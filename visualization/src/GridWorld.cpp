#include <fstream>
#include <iostream>
#include <thread>
#include <sstream>
#include "GridWorld.h"

namespace GWEnv{

void GridWorld::Render() {
    if (IsCursorHidden()) UpdateCamera(&camera, CAMERA_FREE);

    // Toggle camera controls
    if (IsMouseButtonPressed(MOUSE_BUTTON_RIGHT))
    {
        if (IsCursorHidden()) EnableCursor();
        else DisableCursor();
    }
    Action action = Step();
    switch (action) {
        case Action::NONE: break;
        case Action::PLACE:
            AddBlock(agentPos.x, agentPos.y, agentPos.z, 1);
            break;
        case Action::REMOVE:
            RemoveBlock(agentPos.x, agentPos.y, agentPos.z);
            break;
        default:
            Move(action);
            break;
    }

    BeginDrawing();
    ClearBackground(RAYWHITE);
    BeginMode3D(camera);

        DrawPlanes();
        DrawBlocks();

    EndMode3D();
    DrawFPS(10, 10);
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
    if (x >= h || y >= w || z >= d || grid[x][y][z] != 0 || !QueryPlacement(x, y, z))
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
                             spacing, spacing, spacing, BLUE);
                    DrawCubeWires({posX, posY, posZ}, spacing, spacing, spacing, RED);
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

/* Trying out binary format. Sorry Thanh
void GridWorld::SaveToFile() const {
    std::ofstream myfile;
    myfile.open(filePath, std::fstream::out);

    for (int i=0; i < w; i++) {
        for (int j=0; j < d; j++) {
            for (int k=0; k < h; k++) {
                myfile << grid[i][j][k] << " ";
            }
            myfile << std::endl;
        }
        myfile << std::endl;
    }
    myfile << std::endl;
    myfile.close();
}
*/

void GridWorld::SaveToFile() const {
    std::cout << "Saving to file" << std::endl;
    size_t size = w * d * h + 3;
    std::unique_ptr<int []> flatArray = std::make_unique<int[]>(size);
    flatArray[0] = w;
    flatArray[1] = h;
    flatArray[2] = d;

    size_t index = 3;

    for (int i=0; i < w; i++) {
        for (int j=0; j < h; j++) {
            for (int k=0; k < d; k++) {
                flatArray[index++] = grid[i][j][k];
            }
        }
    }

    std::ofstream file(filePath, std::ios_base::binary);
    file.write((char *)flatArray.get(), static_cast<std::streamsize>(size * sizeof(int)));
    file.close();
}

Action GridWorld::Step() {
    // Do not allow save file if nothing changes
    static bool enterDelayed = false;

    if (IsKeyPressed(KEY_ENTER)) {
        if (enterDelayed) return Action::NONE;
        enterDelayed = true;

        SaveToFile();
        return Action::NONE;
    }

    // Save to pddl
    if (IsKeyPressed(KEY_P)) {
        if(enterDelayed) return Action::NONE;
        enterDelayed = true;

        CreatePDDLFile("test");
        return Action::NONE;
    }

    // A little hack to avoid setting enterDelayed = true in every if statement
    bool prevDelayedStatus = enterDelayed;
    enterDelayed = false;
    if (IsCursorHidden()) return Action::NONE;

    if (IsKeyPressed(KEY_W)) return Action::FORWARD;
    if (IsKeyPressed(KEY_S)) return Action::BACKWARD;
    if (IsKeyPressed(KEY_A)) return Action::LEFT;
    if (IsKeyPressed(KEY_D)) return Action::RIGHT;
    if (IsKeyPressed(KEY_UP)) return Action::UP;
    if (IsKeyPressed(KEY_DOWN)) return Action::DOWN;

    if (IsKeyPressed(KEY_SPACE)) return Action::PLACE;
    if (IsKeyPressed(KEY_X)) return Action::REMOVE;

    enterDelayed = prevDelayedStatus;
    return Action::NONE;
}

void GridWorld::LoadFromFile() {
    if (!fs::exists(filePath)) {
        std::cerr << "File " << filePath << " does not exist" << std::endl;
        exit(1);
    }

    std::ifstream file(filePath, std::ios_base::binary);
    int x, y, z;
    file.read((char *) &x, sizeof(int));
    file.read((char *) &y, sizeof(int));
    file.read((char *) &z, sizeof(int));
    if (x != w || y != h || z != d) ResizeGrid(x, y, z);

    // Could read directly into grid instead of array. Not sure which way is more efficient
    std::unique_ptr<int []> flatArray = std::make_unique<int[]>(x * y * z);
    file.read((char *) flatArray.get(), static_cast<std::streamsize>(x * y * z * sizeof(int)));
    size_t index = 0;

    for (int i=0; i < w; i++) {
        for (int j=0; j < h; j++) {
            for (int k=0; k < d; k++) {
                grid[i][j][k] = flatArray[index++];
            }
        }
    }
    file.close();
}

void GridWorld::CreatePDDLFile(std::string problemName)  const {
    std::ofstream file(filePath);

    // Add the start of the file template
    file << "(define (problem " << problemName << ")\n";
    file << "\t(:domain cubeworld)\n";

    // Define the objects
    file << "\t(:objects\n";

    file << "\t\t";
    for(int i = 0; i < w;i++) {
        for(int j = 0; j < h; j++) {
            for(int k = 0; k < d; k++)
                file << "p" << i << "_" << j << "_" << k << " ";
        }
    }
    file << "- position\n";
    file << "\t)\n\n";

    // Define the initial state
    file << "\t(:init\n";

    // Create the starting location for the agent
    file << "\t\t(at-agent p1_1_1)\n";
    for(int i = 0; i < w; i++) {
        for(int j = 0; j < h; j++) {
            for(int k = 0; k < d; k++) {
                // Adjacent to left block
                if(i > 0)
                    file << adjacent(i, j, k, i - 1, j, k);

                // Adjacent to right block
                if(i < w - 1)
                    file << adjacent(i, j, k, i + 1, j, k);

                // Adjacent to backward block
                if(j > 0)
                    file << adjacent(i, j, k, i, j - 1, k);

                // Adjacent to forward block
                if(j < h - 1)
                    file << adjacent(i, j, k, i, j + 1, k);

                // Adjacent to downward block
                if(k > 0)
                    file << adjacent(i, j, k, i, j, k - 1);

                // Adjacent to upward block
                if(j < d - 1)
                    file << adjacent(i, j, k, i, j, k + 1);
            }
        }
    }
    file << "\t)\n\n";

    // Define the goal state
    file << "\t(:goal (and\n";

    for(int i = 0; i < w; i++) {
        for(int j = 0; j < h; j++) {
            for(int k = 0; k < d; k++) {
                if(grid[i][j][k]) {
                    file << "\t\t(full p" << i << "_" << j << "_" << k << ")\n";
                }
            }
        }
    }

    file << "\t))\n";
    file << ")\n";

    file.close();
}

std::string GridWorld::adjacent(int i1, int j1, int k1, int i2, int j2, int k2) {
   std::stringstream adj;
   adj << "\t\t";
   adj << "(adjacent ";
   adj << "p" << i1 << "_" << j1 << "_" << k1 << " ";
   adj << "p" << i2 << "_" << j2 << "_" << k2 << ")\n";
   return adj.str();
}

}

