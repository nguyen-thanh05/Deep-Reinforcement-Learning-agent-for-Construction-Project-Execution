#ifndef GRIDWORLDENV_IENVSTEP_H
#define GRIDWORLDENV_IENVSTEP_H

#include <iostream>
#include "raylib.h"

namespace GWEnv {
    enum class Action {
        UP, DOWN, LEFT, RIGHT, FORWARD, BACKWARD, PLACE, REMOVE, NONE
    };

    /*
     * Interface that describes an agent's behavior
     */
    class IEnvStep {
    public:
        virtual Action Step() = 0;

        virtual ~IEnvStep() = default;
    };

    class InteractiveInput : public IEnvStep {
    public:
        InteractiveInput() = default;

        Action Step() override {
            if (IsCursorHidden()) return Action::NONE;

            if (IsKeyPressed(KEY_W)) return Action::FORWARD;
            if (IsKeyPressed(KEY_S)) return Action::BACKWARD;
            if (IsKeyPressed(KEY_A)) return Action::LEFT;
            if (IsKeyPressed(KEY_D)) return Action::RIGHT;
            if (IsKeyPressed(KEY_E)) return Action::UP;
            if (IsKeyPressed(KEY_Q)) return Action::DOWN;

            if (IsKeyPressed(KEY_ENTER)) return Action::PLACE;
            if (IsKeyPressed(KEY_X)) return Action::REMOVE;

            return Action::NONE;
        }

    private:
    };
}
#endif //GRIDWORLDENV_IENVSTEP_H
