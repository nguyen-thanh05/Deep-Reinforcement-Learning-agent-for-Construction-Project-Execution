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
 *
 * NOTE: The orginal idea was to have two forms of input
 * (humans placing a block using the gui / taking an action from the replays
 * of an agent). This interface was supposed to abstract away the Step() method.
 * But agent replays could be rendered directly in Python with matplotlib
 * so this could be simplified into GridWorld::Step(). To be decided later
 */
class IEnvStep {
public:
    virtual Action Step() = 0;

    virtual ~IEnvStep() = default;
};

/*
 * Interactive mode where human specifies the input
 */
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
};

}
#endif //GRIDWORLDENV_IENVSTEP_H
