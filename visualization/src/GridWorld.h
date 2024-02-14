#ifndef GRIDWORLDENV_GRIDWORLD_H
#define GRIDWORLDENV_GRIDWORLD_H

#include "IEnvStep.h"
#include "raylib.h"
#include "rlgl.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace GWEnv {
    struct Coordinates {
        uint32_t x, y, z;
    };

    class GridWorld {
        template<class T> using vec = std::vector<T>;

    public:
        GridWorld(int _w, int _h, int _d, std::unique_ptr<IEnvStep> _agent)
                : w(_w), h(_h), d(_d),
                  grid(w, vec<vec<int>>(h, vec<int>(d, 0))) {
            agent = std::move(_agent);

            camera.position = (Vector3) {6.5f, 8.5f, 8.5f};       // Camera position
            camera.target = (Vector3) {0.0f, 0.0f, 0.0f};         // Camera looking at point
            camera.up = (Vector3) {0.0f, 1.0f, 0.0f};             // Camera up vector (rotation towards target)
            camera.fovy = 45.0f;                                            // Camera field-of-view Y
            camera.projection = CAMERA_PERSPECTIVE;                         // Camera projection type
        }

        void Render();


        GridWorld(const GridWorld &) = delete;

        GridWorld operator=(const GridWorld &) = delete;

    private:
        void DrawBlocks() const;

        void DrawPlanes() const;

        bool AddBlock(uint32_t x, uint32_t y, uint32_t z, int blockType);

        void RemoveBlock(uint32_t x, uint32_t y, uint32_t z);

        void Move(Action direction);

        bool QueryPlacement(uint32_t x, uint32_t y, uint32_t z) const;

        std::unique_ptr<IEnvStep> agent;
        Coordinates agentPos = {0, 0, 0};
        Camera camera = {0};

        // Dimensions of grid
        uint32_t w, h, d;
        vec<vec<vec<int>>> grid;
        float spacing = 1.0f;
    };
}
#endif //GRIDWORLDENV_GRIDWORLD_H
