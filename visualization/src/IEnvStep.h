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

//            if (IsKeyDown(KEY_A)) return Action::NONE
            if (IsKeyPressed(KEY_ENTER)) return Action::PLACE;
            if (IsKeyDown(KEY_X)) {
                return Action::REMOVE;
            }
            return Action::NONE;
        }

    private:
    };
}
#endif //GRIDWORLDENV_IENVSTEP_H
