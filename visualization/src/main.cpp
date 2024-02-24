#include <iostream>
#include "raylib.h"
#include "rlgl.h"
#include "GridWorld.h"

void PrintError(){
    std::cerr << "Usage: ./GridWorldEnv <env_size> <num_block_types> <s|l> <file_name>\n\n"
              << "<env_size>       : size of the environment\n"
              << "<num_block_types>: number of block types\n"
              << "<s|l>            : whether to save to file or load from file\n"
              << "<file_name>      : path to file to load from / save to\n";
    exit(1);
}

int main(int argc, char* argv[])
{
    for (int i =0; i < argc; i++) {
        std::cout << argv[i] << " ";
    }
    std::cout << argc << std::endl;

    if (argc != 5) PrintError();

    int size = std::stoi(argv[1]);
    int blockTypes = std::stoi(argv[2]);

    if (blockTypes > 13) {
        std::cerr << "Support for more than 13 block types not supported yet";
        exit(1);
    }

    std::string operation{argv[3]};
    if (operation != "s" && operation != "l") PrintError();
    bool save = operation == "s";

    fs::path modelsPath = fs::current_path().parent_path().parent_path() / "targets";

    // Check if the directory exists
    if (!fs::exists(modelsPath)) {
        // Directory does not exist, create it
        if (!fs::create_directories(modelsPath)) {
            std::cerr << "Failed to create directory: " << modelsPath << std::endl;
            return 1;
        }
    }

    fs::path filePath = modelsPath / argv[4];

    const int screenWidth = 1000;
    const int screenHeight = 600;

    GWEnv::GridWorld env(size, size, size, blockTypes, filePath.generic_string(), save);
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
