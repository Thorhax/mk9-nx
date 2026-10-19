// Still snapshots of the game image for testing (cvar mk9_snapshot_s), read from the presenter's guest output, so
// they show what the game presents whatever the window or display does (minimized, hidden X server, ...).

#pragma once

#include <functional>

namespace rex::ui {
class Presenter;
}

namespace mk9::snapshot {

// Starts the snapshot thread when mk9_snapshot_s is set.
void Start(std::function<rex::ui::Presenter*()> get_presenter);

// Stops it.
void Stop();

}  // namespace mk9::snapshot
